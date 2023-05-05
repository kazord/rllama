
/*
 *
 * MovableTensors for RLLaMA
 *
 * This is not a general Tensor library; but it has just enough to run the transformers in LLaMA
 * model.
 *
 *
 * The main structure you work here is storage independant Tensor, which is a 2D matrix on CPU or GPU.
 *
 * Operations have this naming convention:
 *
 *   If it's "to_XXX", then it returns a new object in the specified format.
 *   If it's "XXX_inplace", then it has a &mut self and it modifies in place.
 */

use crate::tensor::{Tensor, TensorDType};
#[cfg(feature = "opencl")]
use crate::tensor_opencl_support::{OpenCL, OpenCLError, OpenCLEvent, OpenCLTensor};

#[cfg(feature = "opencl")]
#[derive(Debug)]
#[derive(Clone)]
pub struct Moving {
    tensor: Tensor,
    opencltensor: OpenCLTensor,
    moving: OpenCLEvent,
    togpu: bool,
}

#[derive(Debug)]
pub enum MovableTensor {
    CPU(Tensor),
    #[cfg(feature = "opencl")]
    GPU(OpenCLTensor),
    #[cfg(feature = "opencl")]
    Moving(Moving)
}

impl MovableTensor {
    pub fn rows(&self) -> i64 {
        match &self {
            MovableTensor::CPU(t) => t.rows(),
            //TODO
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(g) => g.rows(),
            #[cfg(feature = "opencl")]
            //both have row info
            MovableTensor::Moving(m) => m.tensor.rows(),
        }
    }
    pub fn to_f16(&self) -> MovableTensor {
        match &self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.to_f16()),
            //TODO
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(_) => unimplemented!(),
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    pub fn to_f32(&mut self) -> MovableTensor {
        match &self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.to_f32()),
            //TODO
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(_) => self.sync_move_to_cpu().to_f32(),
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    pub fn row(&self, row: i64) -> MovableTensor {
        match &self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.row(row)),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(s) => {
                let mut out = s.cl().uninitialized(1, s.cols());
                match &mut out {
                    Ok(res) => {
                        match res.copy_a_row(s, row) {
                            Ok(e) => {
                                e.wait_for();
                                MovableTensor::GPU(res.clone())
                            },
                            Err(a) => panic!("unable to copy data on GPU {:?}", a),
                        }
                    }
                    Err(_) => panic!("unable to init tensor on GPU"),
                }
            }
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    pub fn view(&self, rows: i64, cols: i64) -> MovableTensor {
        match &self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.view(rows, cols)),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(s) => {
                if rows == self.rows() {
                    return MovableTensor::GPU(s.clone());
                }
                let mut out = s.cl().zeros(rows, cols);
                match &mut out {
                    Ok(res) => {
                        match res.view(s) {
                            Ok(e) => {
                                e.wait_for();
                                MovableTensor::GPU(res.clone())
                            },
                            Err(a) => panic!("unable to copy data on GPU {:?}", a),
                        }
                    }
                    Err(_) => panic!("unable to init tensor on GPU"),
                }
            }
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    //should not be use outside
    pub fn is_on_gpu(&self) -> bool {
        match &self {
            MovableTensor::CPU(_) => false,
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(_) => true,
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    #[inline]
    pub fn dtype(&self) -> TensorDType {
        match &self {
            MovableTensor::CPU(t) => t.dtype(),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(_) => TensorDType::Float16,
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => TensorDType::Float16,
        }
    }
    
    pub fn matrix_mul_transposed(&self, other: &MovableTensor) -> MovableTensor {
        match (&self, other) {
            (MovableTensor::CPU(s),MovableTensor::CPU(o)) => MovableTensor::CPU(s.matrix_mul_transposed(o)),
            
            //TODO
            #[cfg(feature = "opencl")]
            (MovableTensor::CPU(s),_) => unimplemented!(),
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(s), MovableTensor::GPU(o)) => {
                let mut out = s.cl().uninitialized(s.rows(), o.rows());
                match &mut out {
                    Ok(res) => match res.matrix_mul_inplace_transposed(s,o) {
                        Ok(e) => {e.wait_for();MovableTensor::GPU(res.clone())},
                        Err(_) => panic!("unable to do matrix_mul_inplace_transposed on GPUxGPU")
                    }
                    Err(_) => panic!("unable to initi on GPU")
                }
            },
            #[cfg(feature = "opencl")] //move to cpu or to gpu ?
            (MovableTensor::GPU(g), MovableTensor::CPU(o)) => {
                if o.cols() == 0 || o.rows() == 0 {
                    //panic!("gpu unable to buffer 0 len");
                }
                self.matrix_mul_transposed(&other.sync_move_to_gpu(&g.cl()))
            }
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(_), MovableTensor::Moving(m)) => {//we need m ready
                m.moving.wait_for();
                match m.togpu {
                    true => self.matrix_mul_transposed(&MovableTensor::GPU(m.opencltensor.clone())),
                    false => self.matrix_mul_transposed(&MovableTensor::CPU(m.tensor.clone())),
                }
            },
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(_), _) => unimplemented!(),

            #[cfg(feature = "opencl")]
            (MovableTensor::Moving(_), _) => {
                //we need m ready
                self.wait_for().matrix_mul_transposed(other)
            }
        }
    }
    pub fn matrix_mul_inplace_transposed(&mut self, source: &MovableTensor, other: &MovableTensor) {
         match (self, source, other) {
            (MovableTensor::CPU(ref mut s), MovableTensor::CPU(t), MovableTensor::CPU(o)) => {s.matrix_mul_inplace_transposed(&t, &o)},
            
            //TODO
            #[cfg(feature = "opencl")]
            (MovableTensor::CPU(s),_,_) => unimplemented!(),
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(ref mut s), MovableTensor::GPU(t), MovableTensor::GPU(o)) => {
                match s.matrix_mul_inplace_transposed(&t,&o) {
                   Ok(e) => {e.wait_for();},
                   Err(_) => panic!("unable to do matrix_mul_inplace_transposed on GPUxGPU")
                }
            },
            #[cfg(feature = "opencl")] //move to cpu or to gpu ?
            (MovableTensor::GPU(g), MovableTensor::CPU(o), _) => {unimplemented!()
                //if o.cols == 0 || o.rows == 0 {
                    //panic!("gpu unable to buffer 0 len");
                //}
                //self.matrix_mul_transposed(&other.sync_move_to_gpu(&g.cl()))
            }
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(_), _, _) => unimplemented!(),

            #[cfg(feature = "opencl")]
            (MovableTensor::Moving(_), _, _) => unimplemented!(),
        }
    }
    pub fn into_dtype(&mut self, dtype: TensorDType) -> MovableTensor {
        match &self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.clone().into_dtype(dtype)),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(g) => match dtype {
                TensorDType::K4BitQuantization => unimplemented!(),
                TensorDType::Float16 => MovableTensor::GPU(g.clone()),
                TensorDType::Float32 => {
                    //println!("call into_dtype f32 for GPU");
                    self.sync_move_to_cpu().into_dtype(dtype)
                }
            },
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    pub fn silu(&self) -> MovableTensor {
        match &self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.silu()),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(s) => {
                let mut out = s.cl().uninitialized(s.rows(), s.cols());
                match &mut out {
                    Ok(res) => {
                        match res.copy_inplace(s) {
                            Ok(e) => {
                                match res.silu_inplace() {
                                    Ok(e) => {
                                        e.wait_for();
                                        MovableTensor::GPU(res.clone())
                                    },
                                    Err(_) => panic!("unable to silu on GPU"),
                                }
                            }
                            Err(e) => panic!("unable to copy data on GPU : {:?}", e),
                        }
                    }
                    Err(_) => panic!("unable to init tensor on GPU"),
                }
            },
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    //TODO remove it
    pub fn to_tensor(&mut self) -> Tensor {
        match &self {
             MovableTensor::CPU(t) => t.clone(),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(_) => self.sync_move_to_cpu().to_tensor(),
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    
     pub fn hadamard_product(&self, other: &MovableTensor) -> MovableTensor {
         match (&self, other) {
            (MovableTensor::CPU(s),MovableTensor::CPU(o)) => MovableTensor::CPU(s.hadamard_product(o)),
            //TODO
            (MovableTensor::CPU(s),_) => unimplemented!(),
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(s), MovableTensor::GPU(o)) => {
                let mut out = s.cl().uninitialized(s.rows(), s.cols());
                match &mut out {
                    Ok(res) => {
                        match res.copy_inplace(s) {
                            Ok(e) => {
                                match res.hadamard_product_inplace(o) {
                                    Ok(e) => {
                                        e.wait_for();
                                        MovableTensor::GPU(res.clone())
                                    },
                                    Err(_) => panic!("unable to hadamard on GPU"),
                                }
                            }
                            Err(_) => panic!("unable to copy data on GPU"),
                        }
                    }
                    Err(_) => panic!("unable to init tensor on GPU"),
                }
            },
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(_), _) => unimplemented!(),
            #[cfg(feature = "opencl")]
            (MovableTensor::Moving(_), _) => unimplemented!(),
        }
     }
     pub fn transpose(&self) -> MovableTensor {
        match &self {
             MovableTensor::CPU(t) => MovableTensor::CPU(t.transpose()),
            #[cfg(feature = "opencl")]
             MovableTensor::GPU(s) => {
                let mut out = s.cl().uninitialized(s.cols(), s.rows());
                match &mut out {
                    Ok(res) => {
                        match res.transpose_from(s) {
                            Ok(e) => {
                                e.wait_for();
                                MovableTensor::GPU(res.clone())
                            },
                            Err(_) => panic!("unable to transpose on GPU"),
                        }
                    }
                    Err(_) => panic!("unable to init tensor on GPU"),
                }
            },
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
    }
    pub fn to_same_type(&self, other: &MovableTensor) -> MovableTensor {
         match (&self, other) {
            (MovableTensor::CPU(s),MovableTensor::CPU(o)) => MovableTensor::CPU(s.to_same_type(o)),
            //TODO
            #[cfg(feature = "opencl")]
            (MovableTensor::CPU(s),MovableTensor::GPU(o)) => self.to_f16().sync_move_to_gpu(&o.cl()),
            #[cfg(feature = "opencl")] //other is moving to gpu f16 then
            (MovableTensor::CPU(s),MovableTensor::Moving(m)) => self.to_f16().sync_move_to_gpu(&m.opencltensor.cl()),
            (MovableTensor::CPU(s),_) => unimplemented!(),
            #[cfg(feature = "opencl")]
            (MovableTensor::GPU(_), _) => unimplemented!(),
            #[cfg(feature = "opencl")]
            (MovableTensor::Moving(_), _) => unimplemented!(),
        }
     }
     #[cfg(feature = "opencl")]
     pub fn sync_move_to_gpu(&self, cl: &OpenCL) -> MovableTensor {
        match self {
             MovableTensor::CPU(t) => {
                 if t.dtype() != TensorDType::Float16 {
                    panic!("to_gpu_inplace: Only float16 tensors are supported on the GPU");
                 }
                 
                 if let Ok(cl_tensor) = cl.data_u16_to_gpu(
                    t.data() as *const u16,
                    t.layout(),
                    (t.rows() * t.capacity_cols()) as usize,
                    t.rows(),
                    t.cols(),
                    t.capacity_cols(),
                    )
                {
                    if let Some(ev) = cl_tensor.current_event() {
                        ev.wait_for();
                        return  MovableTensor::GPU(cl_tensor);
                    }
                }
                //fail
                    MovableTensor::CPU(t.clone())
             },
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(g) => MovableTensor::GPU(g.clone()),
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(_) => unimplemented!(),
        }
     }
     #[cfg(feature = "opencl")]
     pub fn move_to_gpu(mut self, cl: &OpenCL) -> MovableTensor {
         match self {
             MovableTensor::CPU(t) => {
                 //println!("called async move, care form data pointer");
                 if t.dtype() != TensorDType::Float16 {
                    panic!("to_gpu_inplace: Only float16 tensors are supported on the GPU");
                 }
                 if let Ok(cl_tensor) = cl.data_u16_to_gpu(
                    t.data() as *const u16,
                    t.layout(),
                    (t.rows() * t.capacity_cols()) as usize,
                    t.rows(),
                    t.cols(),
                    t.capacity_cols(),
                    )
                { 
                    if let Some(ev) = cl_tensor.current_event() {
                        return  MovableTensor::Moving(Moving { tensor: t, opencltensor: cl_tensor, moving: ev, togpu: true})
                    }
                }
                //fail
                    MovableTensor::CPU(t.clone())
             },
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(g) => MovableTensor::GPU(g.clone()),
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(ref mut m) => {
                if m.moving.is_complete().unwrap_or(false)  {
                    return MovableTensor::GPU(m.opencltensor.clone())
                }
                else {
                    return MovableTensor::Moving(m.clone())
                }
            },
        }
    }
    pub fn wait_for(&self) -> MovableTensor {
        match &self {
             MovableTensor::CPU(a) => MovableTensor::CPU(a.clone()),
            #[cfg(feature = "opencl")]
             MovableTensor::GPU(_) => unimplemented!(),
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(m) => match m.moving.wait_for() {
                Err(_) => unimplemented!(),
                Ok(_) => {
                    match m.togpu {
                        true => MovableTensor::GPU(m.opencltensor.clone()),
                        false => MovableTensor::CPU(m.tensor.clone()),
                    }
                },
            },
        }
    }
    pub fn finish(&self) {
        match &self {
             MovableTensor::CPU(_) => (),
            #[cfg(feature = "opencl")]
             MovableTensor::GPU(_) => (),
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(m) => match m.moving.wait_for() {
                Err(_) => (),
                Ok(_) => (),
            },
        }
    }
    pub fn sync_move_to_cpu(&mut self) -> MovableTensor {
        match self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.clone()),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(g) => {
                //println!("Moving data from gpu to cpu");
                unsafe {
                    let mut t : Tensor = Tensor::uninitialized(g.rows(), g.cols(), TensorDType::Float16);
                    let data = unsafe { std::alloc::alloc(g.layout()) };
                    if data.is_null() {
                        panic!("to_cpu_inplace: Failed to allocate tensor");
                    }
                    if let Ok(ev) = g.data_u16_from_gpu(data as *mut u16) {
                        t.set_data(data as *mut u16 as *mut u8);
                        ev.wait_for();
                        MovableTensor::CPU(t.clone())
                    }
                    else {
                        MovableTensor::GPU(g.clone())
                    }
                }

            },
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(m) => {
                if m.moving.is_complete().unwrap_or(false) {
                    return MovableTensor::CPU(m.tensor.clone())
                }
                else {
                    return MovableTensor::Moving(m.clone())
                }
            },
        }
     }
    pub fn move_to_cpu(&mut self) -> MovableTensor {
        //TODO fix cloning/borrow
        match self {
            MovableTensor::CPU(t) => MovableTensor::CPU(t.clone()),
            #[cfg(feature = "opencl")]
            MovableTensor::GPU(g) => {
                println!("Moving async data from gpu to cpu");
                unsafe {
                    let mut t : Tensor = Tensor::uninitialized(g.rows(), g.cols(), TensorDType::Float16);
                    let data = unsafe { std::alloc::alloc(g.layout()) };
                    if data.is_null() {
                        panic!("to_cpu_inplace: Failed to allocate tensor");
                    }
                    if let Ok(ev) = g.data_u16_from_gpu(data as *mut u16) {
                        t.set_data(data as *mut u16 as *mut u8);
                        MovableTensor::Moving(Moving { tensor: t.clone(), opencltensor: g.clone() ,moving: ev, togpu: false})
                    }
                    else {
                        MovableTensor::GPU(g.clone())
                    }
                }

            },
            #[cfg(feature = "opencl")]
            MovableTensor::Moving(m) => {
                if m.moving.is_complete().unwrap_or(false) {
                    return MovableTensor::CPU(m.tensor.clone())
                }
                else {
                    return MovableTensor::Moving(m.clone())
                }
            },
        }
     }
     //pub fn concat(pieces: &[&Movable]) -> Movable {
        //todo
        /*match self {
            Movable::CPU(t) => Movable::CPU(t.clone()),
            #[cfg(feature = "opencl")]
            Movable::GPU(g) => {
                println!("Moving async data from gpu to cpu");
                unsafe {
                    let mut t : Tensor = Tensor::uninitialized(g.rows(), g.cols(), TensorDType::Float16);
                    let data = unsafe { std::alloc::alloc(g.layout()) };
                    if data.is_null() {
                        panic!("to_cpu_inplace: Failed to allocate tensor");
                    }
                    TENSORS_BYTES_ALLOCATED.fetch_add(g.layout().size(), std::sync::atomic::Ordering::Relaxed);
                    if let Ok(ev) = g.data_u16_from_gpu(data as *mut u16) {
                        t.data = data as *mut u16 as *mut u8;
                        Movable::Moving(Moving { tensor: t.clone(), opencltensor: g.clone() ,moving: ev})
                    }
                    else {
                        Movable::GPU(g.clone())
                    }
                }

            },
            #[cfg(feature = "opencl")]
            Movable::Moving(m) => {
                if m.moving.is_complete().unwrap_or(false) {
                    return Movable::CPU(m.tensor.clone())
                }
                else {
                    return Movable::Moving(m.clone())
                }
            },
        }*/
        /*CPU CODE if pieces.is_empty() {
            return Tensor::empty();
        }
        let mut total_rows: i64 = 0;
        let expected_cols: i64 = pieces[0].cols;
        let expected_dtype: TensorDType = pieces[0].dtype;
        for piece in pieces {
            if piece.cols != expected_cols {
                panic!("Invalid tensor concatenation, wrong number of columns");
            }
            if piece.dtype != expected_dtype {
                panic!("Invalid tensor concatenation, wrong dtype");
            }
            total_rows += piece.rows;
        }
        let mut result =
            unsafe { Tensor::uninitialized(total_rows, expected_cols, pieces[0].dtype) };
        let mut row_offset = 0;
        for piece in pieces {
            for row in 0..piece.rows {
                for col in 0..piece.cols {
                    let val = piece.get_f32(row, col);
                    result.set_f32(row_offset + row, col, val);
                }
            }
            row_offset += piece.rows;
        }
        result*/
     //}
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;

#[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_transposed_is_close_to_cpu_matrix_mul_transposed_512x1024() {
        let cl = OpenCL::new(false, 0).unwrap();
        let a = Tensor::random(512, 1024, TensorDType::Float32);
        let b = Tensor::random(768, 1024, TensorDType::Float32);
        let mut a2 = MovableTensor::CPU(a.to_f16());
        let mut b2 = MovableTensor::CPU(b.to_f16());
        let mut c = Tensor::random(512, 768, TensorDType::Float32);
        //let mut c2 = Movable::CPU(Tensor::zeros(512, 768, TensorDType::Float32).to_f16());
        
        a2 = a2.sync_move_to_gpu(&cl);
        b2 = b2.sync_move_to_gpu(&cl);
        //c2.to_gpu_inplace(&cl).unwrap();
        let c = a.matrix_mul_transposed(&b);
        let mut c2 = a2.matrix_mul_transposed(&b2);
        let c2t = c2.to_tensor();
        
        assert_eq!(c2.is_on_gpu(), true);
        assert_eq!(c.rows(), c2t.rows());
        assert_eq!(c.cols(), c2t.cols());

        for row in 0..c.rows() {
            for col in 0..c.cols() {
                assert_relative_eq!(c.get_f32(row, col), c2t.get_f32(row, col), epsilon = 1e-1);
            }
        }
    }
    
    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_transposed_is_close_to_cpu_matrix_mul_transposed_1024x1024() {
        let cl = OpenCL::new(false, 0).unwrap();
        let a = Tensor::random(1024, 1024, TensorDType::Float32);
        let b = Tensor::random(1024, 1024, TensorDType::Float32);
        let mut a2 = MovableTensor::CPU(a.to_f16());
        let mut b2 = MovableTensor::CPU(b.to_f16());
        let mut c = Tensor::random(1024, 1024, TensorDType::Float32);
        //let mut c2 = Tensor::zeros(1024, 1024, TensorDType::Float32).to_f16();
        a2 = a2.move_to_gpu(&cl);
        b2 = b2.move_to_gpu(&cl);
        a2 = a2.wait_for();
        b2 = b2.wait_for();
        //c2.to_gpu_inplace(&cl).unwrap();
        c.matrix_mul_inplace_transposed(&a, &b);
        let mut c2 = a2.matrix_mul_transposed(&b2);
        let c2t = c2.to_tensor();

         assert_eq!(c2.is_on_gpu(), true);
        assert_eq!(c.rows(), c2t.rows());
        assert_eq!(c.cols(), c2t.cols());

        for row in 0..c.rows() {
            for col in 0..c.cols() {
                assert_relative_eq!(c.get_f32(row, col), c2t.get_f32(row, col), epsilon = 1e-1);
            }
        }
    }
    
    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_silu_and_cpu_silu_agree() {
        let cl = OpenCL::new(false, 0).unwrap();

        for _trial in 0..300 {
            let mut rng = rand::thread_rng();
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = MovableTensor::CPU(mat1.clone());
            let mut mat2 = mat2.to_f16();
            mat2 = mat2.sync_move_to_gpu(&cl);

            let mat1_result = mat1.silu();
            let mut mat2_result = mat2.silu();
            let mut mat2_result_tensor = mat2_result.sync_move_to_cpu().to_tensor();

            assert_eq!(mat1_result.rows(), mat2_result_tensor.rows());
            assert_eq!(mat1_result.cols(), mat2_result_tensor.cols());

            for row in 0..mat1_result.rows() {
                for col in 0..mat1_result.cols() {
                    assert_relative_eq!(
                        mat1_result.get_f32(row, col),
                        mat2_result_tensor.get_f32(row, col),
                        epsilon = 1e-2
                    );
                }
            }
        }
    }
    
    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_hadamard_product_and_cpu_hadamard_product_agree() {
        let cl = OpenCL::new(false, 0).unwrap();

        for _trial in 0..300 {
            let mut rng = rand::thread_rng();
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = Tensor::random(a, b, TensorDType::Float16);

            let mut mat1_gpu = MovableTensor::CPU(mat1.to_f16());
            let mut mat2_gpu = MovableTensor::CPU(mat2.to_f16());
            mat1_gpu = mat1_gpu.sync_move_to_gpu(&cl);
            mat2_gpu = mat2_gpu.sync_move_to_gpu(&cl);

            let result1 = mat1.hadamard_product(&mat2);
            let mut result2 = mat1_gpu.hadamard_product(&mat2_gpu).sync_move_to_cpu().to_tensor();

            assert_eq!(result1.rows(), result2.rows());
            assert_eq!(result1.cols(), result2.cols());

            for row in 0..result1.rows() {
                for col in 0..result2.cols() {
                    assert_relative_eq!(
                        result1.get_f32(row, col),
                        result2.get_f32(row, col),
                        epsilon = 1e-2
                    );
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_transpose_and_cpu_transpose_agree() {
        let cl = OpenCL::new(false, 0).unwrap();
        let mut rng = rand::thread_rng();
        for _trial in 0..300 {
            let a = rng.gen_range(1..=100);
            let b = rng.gen_range(1..=100);
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mut mat1_gpu = MovableTensor::CPU(mat1.to_f16());
            mat1_gpu = mat1_gpu.sync_move_to_gpu(&cl);

            let mat1_transposed = mat1.transpose();
            let mut mat1_gpu_transposed = mat1_gpu.transpose().sync_move_to_cpu().to_tensor();

            assert_eq!(mat1_transposed.rows(), mat1_gpu_transposed.rows());
            assert_eq!(mat1_transposed.cols(), mat1_gpu_transposed.cols());

            for row in 0..mat1_transposed.rows() {
                for col in 0..mat1_transposed.cols() {
                    assert_relative_eq!(
                        mat1_transposed.get_f32(row, col),
                        mat1_gpu_transposed.get_f32(row, col),
                        epsilon = 1e-2,
                    );
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_transposed_is_close_to_cpu_matrix_mul_transposed() {
        let cl = OpenCL::new(false, 0).unwrap();
        let mut rng = rand::thread_rng();

        for _trial in 0..300 {
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let c = rng.gen_range(1..=300);

            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = Tensor::random(c, b, TensorDType::Float16);
            let mat3 = Tensor::random(a, c, TensorDType::Float16);
            let mut mat1_gpu = MovableTensor::CPU(mat1.clone()).sync_move_to_gpu(&cl);
            let mut mat2_gpu = MovableTensor::CPU(mat2.clone()).sync_move_to_gpu(&cl);
            let mut mat3_gpu = MovableTensor::CPU(mat3.clone()).sync_move_to_gpu(&cl);
           
            let mat1 = mat1.to_f32();
            let mat2 = mat2.to_f32();
            let mut mat3 = mat3.to_f32();

            mat3.matrix_mul_inplace_transposed(&mat1, &mat2);
            mat3_gpu.matrix_mul_inplace_transposed(&mat1_gpu, &mat2_gpu);
            let mat3_gpu = mat3_gpu.sync_move_to_cpu().to_tensor();

            assert_eq!(mat3.rows(), mat3_gpu.rows());
            assert_eq!(mat3.cols(), mat3_gpu.cols());

            for row in 0..mat3.rows() {
                for col in 0..mat3.cols() {
                    assert_relative_eq!(
                        mat3.get_f32(row, col),
                        mat3_gpu.get_f32(row, col),
                        epsilon = 1e-2,
                    );
                }
            }
        }
    }
    
    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_vector_transposed_is_close_to_cpu_matrix_mul_vector_transposed_1() {
        let cl = OpenCL::new(false, 0).unwrap();
        let mut rng = rand::thread_rng();

        // src.rows == 1

        for _trial in 0..300 {
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);

            let mat1 = Tensor::random(1, a, TensorDType::Float16);
            let mat2 = Tensor::random(b, a, TensorDType::Float16);
            let mat3 = Tensor::random(1, b, TensorDType::Float16);
            let mut mat1_gpu = MovableTensor::CPU(mat1.clone()).sync_move_to_gpu(&cl);
            let mut mat2_gpu = MovableTensor::CPU(mat2.clone()).sync_move_to_gpu(&cl);
            let mut mat3_gpu = MovableTensor::CPU(mat3.clone()).sync_move_to_gpu(&cl);

            let mat1 = mat1.to_f32();
            let mat2 = mat2.to_f32();
            let mut mat3 = mat3.to_f32();

            mat3.matrix_mul_inplace_transposed(&mat1, &mat2);
            mat3_gpu.matrix_mul_inplace_transposed(&mat1_gpu, &mat2_gpu);
            let mat3_gpu = mat3_gpu.sync_move_to_cpu().to_tensor();

            assert_eq!(mat3.rows(), mat3_gpu.rows());
            assert_eq!(mat3.cols(), mat3_gpu.cols());

            for row in 0..mat3.rows() {
                for col in 0..mat3.cols() {
                    assert_relative_eq!(
                        mat3.get_f32(row, col),
                        mat3_gpu.get_f32(row, col),
                        epsilon = 1e-2,
                    );
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_vector_transposed_is_close_to_cpu_matrix_mul_vector_transposed_2() {
        let cl = OpenCL::new(false, 0).unwrap();
        let mut rng = rand::thread_rng();

        // other.rows == 1

        for _trial in 0..300 {
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);

            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = Tensor::random(1, b, TensorDType::Float16);
            let mat3 = Tensor::random(a, 1, TensorDType::Float16);
            let mut mat1_gpu = MovableTensor::CPU(mat1.clone()).sync_move_to_gpu(&cl);
            let mut mat2_gpu = MovableTensor::CPU(mat2.clone()).sync_move_to_gpu(&cl);
            let mut mat3_gpu = MovableTensor::CPU(mat3.clone()).sync_move_to_gpu(&cl);

            let mat1 = mat1.to_f32();
            let mat2 = mat2.to_f32();
            let mut mat3 = mat3.to_f32();

            mat3.matrix_mul_inplace_transposed(&mat1, &mat2);
            mat3_gpu.matrix_mul_inplace_transposed(&mat1_gpu, &mat2_gpu);
            let mat3_gpu = mat3_gpu.sync_move_to_cpu().to_tensor();

            assert_eq!(mat3.rows(), mat3_gpu.rows());
            assert_eq!(mat3.cols(), mat3_gpu.cols());

            for row in 0..mat3.rows() {
                for col in 0..mat3.cols() {
                    assert_relative_eq!(
                        mat3.get_f32(row, col),
                        mat3_gpu.get_f32(row, col),
                        epsilon = 1e-2,
                    );
                }
            }
        }
    }
    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_row_is_close_to_cpu_row() {
        let cl = OpenCL::new(false, 0).unwrap();

        for _trial in 0..300 {
            let mut rng = rand::thread_rng();
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let c = rng.gen_range(0..=a-1);
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = MovableTensor::CPU(mat1.clone());
            let mut mat2 = mat2.to_f16();
            mat2 = mat2.sync_move_to_gpu(&cl);

            let mat1_result = mat1.row(c);
            let mut mat2_result = mat2.row(c);
            let mut mat2_result_tensor = mat2_result.sync_move_to_cpu().to_tensor();

            assert_eq!(mat1_result.rows(), mat2_result_tensor.rows());
            assert_eq!(mat1_result.cols(), mat2_result_tensor.cols());

            for row in 0..mat1_result.rows() {
                for col in 0..mat1_result.cols() {
                    assert_relative_eq!(
                        mat1_result.get_f32(row, col),
                        mat2_result_tensor.get_f32(row, col),
                        epsilon = 1e-2
                    );
                }
            }
        }
    }

      #[cfg(feature = "opencl")]
    #[test]
    fn gpu_view_is_close_to_cpu_view() {
        let cl = OpenCL::new(false, 0).unwrap();

        for _trial in 0..300 {
            let mut rng = rand::thread_rng();
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let mut c = rng.gen_range(1..=a-1);
            while (a * b) % c != 0 {
                c -= 1
            }
            let d = (a * b) / c;
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = MovableTensor::CPU(mat1.clone());
            let mut mat2 = mat2.to_f16();
            mat2 = mat2.sync_move_to_gpu(&cl);

            let mat1_result = mat1.view(c,d);
            let mut mat2_result = mat2.view(c,d);
            let mut mat2_result_tensor = mat2_result.sync_move_to_cpu().to_tensor();

            assert_eq!(mat1_result.rows(), mat2_result_tensor.rows());
            assert_eq!(mat1_result.cols(), mat2_result_tensor.cols());

            for row in 0..mat1_result.rows() {
                for col in 0..mat1_result.cols() {
                    assert_relative_eq!(
                        mat1_result.get_f32(row, col),
                        mat2_result_tensor.get_f32(row, col),
                        epsilon = 1e-2
                    );
                }
            }
        }
    }
}