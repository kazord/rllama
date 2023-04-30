/*
 * OpenCL stuff to run (some) of the tensor operations.
 */

use ocl::{
    enums::DeviceInfo, enums::DeviceInfoResult, Buffer, Context, Device, DeviceType, Event, Kernel,
    Platform, Program, Queue,
};
use std::alloc::Layout;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug)]
#[allow(dead_code)]
struct Programs {
    matrix_mul_transposed_f16_program: Program,
    matrix_mul_transposed_f16: Kernel,
    matrix_mul_transposed_one_row_f16_program: Program,
    matrix_mul_transposed_one_row_f16: Kernel,
    matrix_mul_transposed_f16_cpu_optimized_program: Program,
    matrix_mul_transposed_f16_cpu_optimized: Kernel,
    silu_f16_program: Program,
    silu_f16: Kernel,
    hadamard_product_f16_program: Program,
    hadamard_product_f16: Kernel,
    transpose_f16_program: Program,
    transpose_f16: Kernel,
    //broken program ?
    pow_f16_program: Program,
    pow_f16: Kernel,
    mean_cols_f16_program: Program,
    mean_cols_f16: Kernel,
    add_scalar_f16_program: Program,
    add_scalar_f16: Kernel,
    scalar_multiply_broadcast_f16_program: Program,
    scalar_multiply_broadcast_f16: Kernel,
    hadamard_product_broadcast_f16_program: Program,
    hadamard_product_broadcast_f16: Kernel,
    rsqrt_f16_program: Program,
    rsqrt_f16: Kernel,
    add_f16_program: Program,
    add_f16: Kernel,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OpenCL {
    ctx: Context,
    queue: Queue,
    programs: Arc<Mutex<Programs>>,
    is_cpu_device: bool,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct OpenCLTensor {
    buf: Buffer<u16>, // really is f16
    initial_write_event: Option<ocl::Event>,
    last_event: Option<ocl::Event>,
    data_layout: Layout,
    nitems: usize,
    rows: i64,
    cols: i64,
    cols_capacity: i64,
    queue: Queue,
    cl: OpenCL,
}

//#[derive(Debug)]
pub type OpenCLEvent = ocl::Event;
/*pub struct OpenCLEvent {
    event: ocl::Event,
}*/

impl Drop for OpenCLTensor {
    fn drop(&mut self) {
        if self.initial_write_event.is_some() {
            self.initial_write_event
                .as_ref()
                .unwrap()
                .wait_for()
                .unwrap();
        }
        self.initial_write_event = None;
    }
}
unsafe impl Send for OpenCLTensor {}
unsafe impl Sync for OpenCLTensor {}

#[derive(Error, Debug)]
pub enum OpenCLError {
    #[error("OpenCL error: {0}")]
    OpenCL(#[from] ocl::Error),
    #[error("Cannot select device")]
    OpenCLDeviceSelection,
}

impl OpenCL {
    pub fn new(verbose: bool, nth_device: usize) -> Result<OpenCL, OpenCLError> {
        let platforms = Platform::list();
        let mut devices: Vec<(Platform, Device)> = Vec::new();
        for platform in platforms {
            for device in Device::list_all(platform)? {
                devices.push((platform, device));
            }
        }
        if verbose {
            println!("Enumerating OpenCL devices:");
        }
        for (idx, (_plat, device)) in devices.iter().enumerate() {
            if verbose {
                println!("OpenCL {} device: {}", idx, device.name()?,);
            }
        }
        if nth_device > devices.len() {
            return Err(OpenCLError::OpenCLDeviceSelection);
        }
        if verbose {
            println!("---");
            println!("Selected OpenCL device: {}", devices[nth_device].1.name()?);
        }

        let ctx = Context::builder()
            .platform(devices[nth_device].0)
            .devices(devices[nth_device].1)
            .build()?;

        let is_cpu_device = match devices[nth_device].1.info(DeviceInfo::Type)? {
            DeviceInfoResult::Type(DeviceType::CPU) => true,
            _ => false,
        };

        let queue = Queue::new(&ctx, devices[nth_device].1, None)?;
        let programs = make_programs(&ctx, &queue)?;
        Ok(OpenCL {
            ctx: ctx,
            queue: queue,
            programs: Arc::new(Mutex::new(programs)),
            is_cpu_device,
        })
    }

    pub fn flush(&self) {
        let _ = self.queue.flush();
    }
    pub fn finish(&self) {
        self.queue.finish();
    }

    pub fn data_u16_to_gpu(
        &self,
        data: *const u16,
        data_layout: Layout,
        nitems: usize,
        rows: i64,
        cols: i64,
        cols_capacity: i64,
    ) -> Result<OpenCLTensor, OpenCLError> {
        unsafe {
            let buf = Buffer::builder()
                .queue(self.queue.clone())
                .len(nitems)
                .build()?;
            let mut event = Event::empty();
            let data_slice: &[u16] = std::slice::from_raw_parts(data, nitems);
            buf.cmd()
                .write(data_slice)
                .block(false)
                .enew(&mut event)
                .enq()?;
            Ok(OpenCLTensor {
                buf,
                initial_write_event: Some(event),
                last_event: None,
                data_layout,
                nitems,
                rows,
                cols,
                cols_capacity,
                queue: self.queue.clone(),
                cl: self.clone(),
            })
        }
    }
    pub fn uninitialized(&self, 
        rows: i64,
        cols: i64,
    ) -> Result<OpenCLTensor, OpenCLError> {
        unsafe {
            let cols_capacity = if cols % 16 == 0 { cols } else { cols + 16 - cols % 16 };
            let nitems : usize = (rows*cols_capacity) as usize;
            let data_layout = Layout::from_size_align(nitems * 2, 32).unwrap();
            let buf = Buffer::builder()
                .queue(self.queue.clone())
                .len(nitems)
                .build()?;
            let mut event = Event::empty();
            buf.cmd()
                .fill(0u16, None)
                .block(false)
                .enew(&mut event)
                .enq()?;
            Ok(OpenCLTensor {
                buf,
                initial_write_event: None,
                last_event: None,
                data_layout: data_layout,
                nitems,
                rows,
                cols,
                cols_capacity,
                queue: self.queue.clone(),
                cl: self.clone(),
            })
        }
    }
}

impl OpenCLTensor {
    pub fn cl(&self) -> OpenCL {
        self.cl.clone()
    }
    pub fn layout(&self) -> Layout {
        self.data_layout.clone()
    }
    pub fn rows(&self) -> i64 {
        self.rows
    }
    pub fn cols(&self) -> i64 {
        self.cols
    }
    pub fn nitems(&self) -> usize {
        self.nitems
    }
    pub fn current_event(&self) -> Option<OpenCLEvent> {
        if let Some(e) = &self.initial_write_event {
            return Some(e.clone());
        }
        else if let Some(e) = &self.last_event {
            return Some(e.clone());
        }
        return None;
    }
    pub fn is_ready(&self) -> bool {
        if self.initial_write_event.is_some() {
            if let Ok(b) = self.initial_write_event.as_ref().unwrap().is_complete() {
                b
            }
            else {
                false
            }
        }
        else {
            true
        }
    }
    pub fn wait_until_ready(&mut self) {
        if self.last_event.is_some() {
            self.last_event.as_ref().unwrap().wait_for().unwrap();
            self.last_event = None;
        }
        if self.initial_write_event.is_some() {
            self.initial_write_event
                .as_ref()
                .unwrap()
                .wait_for()
                .unwrap();
            self.initial_write_event = None;
        }
       /* if !self.data.is_null() {
            unsafe {
                std::alloc::dealloc(self.data as *mut u8, self.data_layout);
            }
            self.data = std::ptr::null();
        }*/
    }

    pub fn data_u16_from_gpu(&mut self, data: *mut u16) -> Result<OpenCLEvent, OpenCLError> {
        unsafe {
            let mut event = Event::empty();
            let data_slice: &mut [u16] = std::slice::from_raw_parts_mut(data, self.nitems);
            let b = self
                .buf
                .cmd()
                .read(data_slice)
                .block(false)
                .enew(&mut event);
            b.enq()?;
            self.last_event = Some(event.clone());
            return Ok(event.clone());
        }
    }

    /// Copies all values from another tensor
    pub fn copy_inplace(&mut self, other: &OpenCLTensor) -> Result<OpenCLEvent, OpenCLError> {
        if other.rows != self.rows || other.cols != self.cols {
            panic!(
                "Cannot in-place copy tensors of different sizes: {}x{} <-- {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut event = Event::empty();
        other
            .buf
            .cmd()
            //.queue(&other.queue)
            .copy(&self.buf, None, None)
            .enew(&mut event)
            .enq()?;
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }
    pub fn add_scalar_inplace(&mut self, scalar: f32) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.add_scalar_f16.set_arg(0, self.buf.clone()).unwrap();
        prg.add_scalar_f16
            .set_arg(1, self.cols_capacity as i32)
            .unwrap();
        prg.add_scalar_f16.set_arg(2, scalar).unwrap();
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .add_scalar_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }

    pub fn scalar_multiply_broadcast_inplace(
        &mut self,
        other: &OpenCLTensor,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.scalar_multiply_broadcast_f16
            .set_arg(0, self.buf.clone())
            .unwrap();
        prg.scalar_multiply_broadcast_f16
            .set_arg(1, other.buf.clone())
            .unwrap();
        prg.scalar_multiply_broadcast_f16
            .set_arg(2, self.cols_capacity as i32)
            .unwrap();
        prg.scalar_multiply_broadcast_f16
            .set_arg(3, other.cols_capacity as i32)
            .unwrap();
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .scalar_multiply_broadcast_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, (self.cols_capacity / 16) as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }

    pub fn transpose_from(&mut self, other: &OpenCLTensor) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.transpose_f16.set_arg(0, self.buf.clone()).unwrap();
        prg.transpose_f16.set_arg(1, other.buf.clone()).unwrap();
        prg.transpose_f16
            .set_arg(2, self.cols_capacity as i32)
            .unwrap();
        prg.transpose_f16
            .set_arg(3, other.cols_capacity as i32)
            .unwrap();
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .transpose_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq().unwrap();
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }

    pub fn hadamard_product_inplace(
        &mut self,
        other: &OpenCLTensor,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.hadamard_product_f16.set_arg(0, self.buf.clone())?;
        prg.hadamard_product_f16.set_arg(1, other.buf.clone())?;
        prg.hadamard_product_f16
            .set_arg(2, self.cols_capacity as i32)?;
        prg.hadamard_product_f16
            .set_arg(3, other.cols_capacity as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .hadamard_product_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }
    pub fn hadamard_product_broadcast_inplace(
        &mut self,
        other: &OpenCLTensor,
    ) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.hadamard_product_broadcast_f16
            .set_arg(0, self.buf.clone())?;
        prg.hadamard_product_broadcast_f16
            .set_arg(1, other.buf.clone())?;
        prg.hadamard_product_broadcast_f16
            .set_arg(2, self.cols_capacity as i32)?;
        prg.hadamard_product_broadcast_f16
            .set_arg(3, other.cols_capacity as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .hadamard_product_broadcast_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, (self.cols_capacity as usize) / 16])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }
     pub fn mean_cols_from(&mut self, other: &OpenCLTensor) -> Result<OpenCLEvent, OpenCLError> {
        if self.cols != 1 {
            panic!(
                "mean_cols_from: number of columns in target is not 1: {}",
                self.cols
            );
        }
        if self.rows != other.rows {
            panic!(
                "mean_cols_from: number of rows in target is not equal to number of rows in source: {} != {}",
                self.rows, other.rows
            );
        }
        let prg = self.cl.programs.lock().unwrap();
        prg.mean_cols_f16.set_arg(0, self.buf.clone())?;
        prg.mean_cols_f16.set_arg(1, other.buf.clone())?;
        prg.mean_cols_f16.set_arg(2, self.cols_capacity as i32)?;
        prg.mean_cols_f16.set_arg(3, other.cols_capacity as i32)?;
        prg.mean_cols_f16.set_arg(4, other.cols as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .mean_cols_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, 1])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }

    pub fn pow_inplace(&mut self, scalar: f32) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.pow_f16.set_arg(0, self.buf.clone())?;
        prg.pow_f16.set_arg(1, self.cols_capacity as i32)?;
        prg.pow_f16.set_arg(2, scalar)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .pow_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }
    pub fn silu_inplace(&mut self) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.silu_f16.set_arg(0, self.buf.clone())?;
        prg.silu_f16.set_arg(1, self.cols_capacity as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .silu_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }
    pub fn add_inplace(&mut self, left: &OpenCLTensor) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.add_f16.set_arg(0, self.buf.clone())?;
        prg.add_f16.set_arg(1, left.buf.clone())?;
        prg.add_f16.set_arg(2, self.cols_capacity as i32)?;
        prg.add_f16.set_arg(3, left.cols_capacity as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .add_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }

    pub fn rsqrt_inplace(&mut self) -> Result<OpenCLEvent, OpenCLError> {
        let prg = self.cl.programs.lock().unwrap();
        prg.rsqrt_f16.set_arg(0, self.buf.clone())?;
        prg.rsqrt_f16.set_arg(1, self.cols_capacity as i32)?;
        let mut event = Event::empty();
        unsafe {
            let b = prg
                .rsqrt_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }
    pub fn matrix_mul_inplace_transposed(
        &mut self,
        src: &OpenCLTensor,
        other: &OpenCLTensor,
    ) -> Result<OpenCLEvent, OpenCLError> {
        if src.cols != other.cols {
            panic!(
                "OpenCL matrix_mul_inplace_transposed: src.cols must equal other.cols: {}x{} vs {}x{}",
                src.rows, src.cols, other.rows, other.cols
            );
        }
        if self.rows != src.rows || self.cols != other.rows {
            panic!(
                "OpenCL matrix_mul_inplace_transposed: self.rows must equal src.rows and self.cols must equal other.cols: {}x{} vs {}x{} vs {}x{}",
                self.rows, self.cols, src.rows, src.cols, other.rows, other.cols
            );
        }

        // Clear out the target memory.
        unsafe { self.buf.cmd().fill(0u16, None).block(false).enq()? };

        let prg = self.cl.programs.lock().unwrap();

        // 0 = CPU optimized
        // 1 = GPU optimized
        // 2 = GPU optimized vector multiply (other.rows == 1)
        const CPU: u8 = 0;
        const GPU: u8 = 1;
        const GPU2: u8 = 2;
        let strategy: u8 = if self.cl.is_cpu_device {
            CPU
        } else {
            if src.rows == 1 {
                GPU2
            } else {
                GPU
            }
        };

        let prg = if strategy == CPU {
            &prg.matrix_mul_transposed_f16_cpu_optimized
        } else if strategy == GPU {
            &prg.matrix_mul_transposed_f16
        } else {
            &prg.matrix_mul_transposed_one_row_f16
        };
        prg.set_arg(0, self.buf.clone())?;
        prg.set_arg(1, src.buf.clone())?;
        prg.set_arg(2, other.buf.clone())?;
        prg.set_arg(3, src.cols_capacity as i32)?;
        prg.set_arg(4, other.cols_capacity as i32)?;
        prg.set_arg(5, self.cols_capacity as i32)?;
        prg.set_arg(6, self.rows as i32)?;
        prg.set_arg(7, self.cols as i32)?;
        prg.set_arg(8, src.cols as i32)?;
        let mut event = Event::empty();

        let rows16 = if self.rows % 16 == 0 {
            self.rows
        } else {
            self.rows + 16 - (self.rows % 16)
        };
        let cols16 = if self.cols % 16 == 0 {
            self.cols
        } else {
            self.cols + 16 - (self.cols % 16)
        };

        unsafe {
            if strategy == CPU {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([self.cols as usize, self.rows as usize])
                    .enew(&mut event);
                b.enq()?;
            } else if strategy == GPU {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([cols16 as usize, rows16 as usize])
                    .local_work_size([16, 16])
                    .enew(&mut event);
                b.enq()?;
            } else if strategy == GPU2 {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([cols16 as usize, 1])
                    .local_work_size([16, 1])
                    .enew(&mut event);
                b.enq()?;
            } else {
                let b = prg
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size([self.cols as usize, self.rows as usize])
                    .enew(&mut event);
                b.enq()?;
            }
        }
        self.last_event = Some(event.clone());
        Ok(event.clone())
    }
}

/*impl OpenCLEvent {
    #[inline]
    pub fn wait(&self) {
        self.event.wait_for().unwrap();
    }
}*/

fn make_programs(ctx: &Context, queue: &Queue) -> Result<Programs, OpenCLError> {
    fn make_program_with_src(ctx: &Context, src: &str) -> Result<Program, OpenCLError> {
        let program = Program::builder().src(src).build(&ctx)?;
        Ok(program)
    }

    let matrix_mul_transposed_f16_program =
        make_program_with_src(ctx, MATRIX_MUL_TRANSPOSED_F16_SRC)?;
    let matrix_mul_transposed_f16 = Kernel::builder()
        .program(&matrix_mul_transposed_f16_program)
        .name("matrix_mul_transposed_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let matrix_mul_transposed_f16_cpu_optimized_program =
        make_program_with_src(ctx, MATRIX_MUL_TRANSPOSED_F16_CPU_OPTIMIZED_SRC)?;
    let matrix_mul_transposed_f16_cpu_optimized = Kernel::builder()
        .program(&matrix_mul_transposed_f16_cpu_optimized_program)
        .name("matrix_mul_transposed_f16_cpu_optimized")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let matrix_mul_transposed_one_row_f16_program =
        make_program_with_src(ctx, MATRIX_MUL_TRANSPOSED_F16_ONE_ROW_SRC)?;
    let matrix_mul_transposed_one_row_f16 = Kernel::builder()
        .program(&matrix_mul_transposed_one_row_f16_program)
        .name("matrix_mul_transposed_one_row_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let silu_f16_program = make_program_with_src(ctx, SILU_F16_SRC)?;
    let silu_f16 = Kernel::builder()
        .program(&silu_f16_program)
        .name("silu_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let hadamard_product_f16_program = make_program_with_src(ctx, HADAMARD_PRODUCT_F16_SRC)?;
    let hadamard_product_f16 = Kernel::builder()
        .program(&hadamard_product_f16_program)
        .name("hadamard_product_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let transpose_f16_program = make_program_with_src(ctx, TRANSPOSE_F16_SRC)?;
    let transpose_f16 = Kernel::builder()
        .program(&transpose_f16_program)
        .name("transpose_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    //broken programs ?
    let pow_f16_program = make_program_with_src(ctx, POW_F16_SRC)?;
    let pow_f16 = Kernel::builder()
        .program(&pow_f16_program)
        .name("pow_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0.0f32)
        .queue(queue.clone())
        .build()?;
    let mean_cols_f16_program = make_program_with_src(ctx, MEAN_COLS_F16_SRC)?;
    let mean_cols_f16 = Kernel::builder()
        .program(&mean_cols_f16_program)
        .name("mean_cols_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let add_scalar_f16_program = make_program_with_src(ctx, ADD_SCALAR_F16_SRC)?;
    let add_scalar_f16 = Kernel::builder()
        .program(&add_scalar_f16_program)
        .name("add_scalar_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0.0f32)
        .queue(queue.clone())
        .build()?;
    let scalar_multiply_broadcast_f16_program =
        make_program_with_src(ctx, SCALAR_MULTIPLY_BROADCAST_F16_SRC)?;
    let scalar_multiply_broadcast_f16 = Kernel::builder()
        .program(&scalar_multiply_broadcast_f16_program)
        .name("scalar_multiply_broadcast_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let hadamard_product_broadcast_f16_program =
        make_program_with_src(ctx, HADAMARD_PRODUCT_BROADCAST_F16_SRC)?;
    let hadamard_product_broadcast_f16 = Kernel::builder()
        .program(&hadamard_product_broadcast_f16_program)
        .name("hadamard_product_broadcast_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let rsqrt_f16_program = make_program_with_src(ctx, RSQRT_F16_SRC)?;
    let rsqrt_f16 = Kernel::builder()
        .program(&rsqrt_f16_program)
        .name("rsqrt_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    let add_f16_program = make_program_with_src(ctx, ADD_F16_SRC)?;
    let add_f16 = Kernel::builder()
        .program(&add_f16_program)
        .name("add_f16")
        .arg(None::<&Buffer<u16>>)
        .arg(None::<&Buffer<u16>>)
        .arg(&0)
        .arg(&0)
        .queue(queue.clone())
        .build()?;
    Ok(Programs {
        matrix_mul_transposed_f16_program,
        matrix_mul_transposed_f16,
        matrix_mul_transposed_one_row_f16_program,
        matrix_mul_transposed_one_row_f16,
        matrix_mul_transposed_f16_cpu_optimized_program,
        matrix_mul_transposed_f16_cpu_optimized,
        silu_f16_program,
        silu_f16,
        hadamard_product_f16_program,
        hadamard_product_f16,
        transpose_f16_program,
        transpose_f16,
        //broken ?
        pow_f16_program,
        pow_f16,
        mean_cols_f16_program,
        mean_cols_f16,
        add_scalar_f16_program,
        add_scalar_f16,
        scalar_multiply_broadcast_f16_program,
        scalar_multiply_broadcast_f16,
        hadamard_product_broadcast_f16_program,
        hadamard_product_broadcast_f16,
        rsqrt_f16_program,
        rsqrt_f16,
        add_f16_program,
        add_f16,
    })
}

const MATRIX_MUL_TRANSPOSED_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matrix_mul_transposed_f16(
    __global half *tgt,
    __global const half *left,
    __global const half *right,
    const int left_cols_capacity,
    const int right_cols_capacity,
    const int ncols_capacity,
    const int nrows,
    const int ncols,  // size of target
    const int shared_sz
) {
    __local float lefttile[16][16];
    __local float righttile[16][16];

    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int num_tiles = (shared_sz + 15) / 16;

    float sum = 0.0f;
    for (int t = 0; t < num_tiles; ++t) {
        if (global_y < nrows) {
            lefttile[local_y][local_x] = vload_half(global_y * left_cols_capacity + t * 16 + local_x, left);
        } else {
            lefttile[local_y][local_x] = 0.0f;
        }
        if (global_x < ncols) {
            righttile[local_y][local_x] = vload_half(global_x * right_cols_capacity + t * 16 + local_y, right);
        } else {
            righttile[local_y][local_x] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) {
            sum += lefttile[local_y][k] * righttile[k][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_x < ncols && global_y < nrows) {
        vstore_half(sum, global_y * ncols_capacity + global_x, (__global half*) tgt);
    }
}
"#;

const MATRIX_MUL_TRANSPOSED_F16_ONE_ROW_SRC: &str = r#"
__kernel void matrix_mul_transposed_one_row_f16(
    __global half *tgt,
    __global const half *left,
    __global const half *right,
    const int left_cols_capacity,
    const int right_cols_capacity,
    const int ncols_capacity,
    const int nrows,
    const int ncols,  // size of target
    const int shared_sz
) {
    // assertions:
    // nrows == 1
    // left_rows == 1
    __local float lefttile[16];
    __local float righttile[16][16];

    const int global_x = get_global_id(0);
    const int local_x = get_local_id(0);
    const int num_tiles = (shared_sz + 15) / 16;
    const int x_tile = (global_x / 16) * 16;

    float sum = 0.0f;
    if (x_tile + 15 < ncols) {
        for (int t = 0; t < num_tiles; ++t) {
            lefttile[local_x] = vload_half(t * 16 + local_x, left);
            for (int k = 0; k < 16; ++k) {
                righttile[k][local_x] = vload_half(t * 16 + local_x + (x_tile + k) * right_cols_capacity, right);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < 16; ++k) {
                sum += lefttile[k] * righttile[local_x][k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    } else {
        for (int t = 0; t < num_tiles; ++t) {
            lefttile[local_x] = vload_half(t * 16 + local_x, left);
            for (int k = 0; k < 16; ++k) {
                if (x_tile + k >= ncols) {
                    righttile[k][local_x] = 0.0f;
                } else {
                    righttile[k][local_x] = vload_half(t * 16 + local_x + (x_tile + k) * right_cols_capacity, right);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < 16; ++k) {
                sum += lefttile[k] * righttile[local_x][k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (global_x < ncols) {
        vstore_half(sum, global_x, (__global half*) tgt);
    }
}"#;

const MATRIX_MUL_TRANSPOSED_F16_CPU_OPTIMIZED_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matrix_mul_transposed_f16_cpu_optimized(
    __global half *tgt,
    __global const half *left,
    __global const half *right,
    const int left_cols_capacity,
    const int right_cols_capacity,
    const int ncols_capacity,
    const int nrows,
    const int ncols,  // size of target
    const int shared_sz
) {
    const int tgt_col = get_global_id(0);
    const int tgt_row = get_global_id(1);
    int col_iterations = shared_sz / 16;
    if (shared_sz % 16 != 0) {
        col_iterations = col_iterations + 1;
    }
    float16 sum = 0;
    for (int col16 = 0; col16 < col_iterations; col16++) {
        const float16 left8 = vload_half16((tgt_row * left_cols_capacity)/16 + col16, (__global const half*) left);
        const float16 right8 = vload_half16((tgt_col * right_cols_capacity)/16 + col16, (__global const half*) right);
        // hadamard product FMA add it to sum
        // const float16 result8 = left8 * right8;
        // sum += result8;
        sum = fma(left8, right8, sum);
    }
    // Reduce as accurately as possible
    float sum1 = sum.s0 + sum.s1;
    float sum2 = sum.s2 + sum.s3;
    float sum3 = sum.s4 + sum.s5;
    float sum4 = sum.s6 + sum.s7;
    float sum5 = sum.s8 + sum.s9;
    float sum6 = sum.sa + sum.sb;
    float sum7 = sum.sc + sum.sd;
    float sum8 = sum.se + sum.sf;
    float sum11 = sum1 + sum2;
    float sum12 = sum3 + sum4;
    float sum13 = sum5 + sum6;
    float sum14 = sum7 + sum8;
    float sum21 = sum11 + sum12;
    float sum22 = sum13 + sum14;
    float total = sum21 + sum22;
    vstore_half(total, 0, (__global half*) &tgt[tgt_row * ncols_capacity + tgt_col]);
}
"#;

/// Computes SILU for every f16 value in the tensor
const SILU_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void silu_f16(__global half *tgt,
                       const int ncols_capacity)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float val = vload_half(tgt_row * ncols_capacity + tgt_col, (__global const half*) tgt);
    const float result = val * (1.0 / (1.0 + exp(-val)));
    vstore_half(result, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Computes hadamard product of two identially sized tensors
const HADAMARD_PRODUCT_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void hadamard_product_f16(__global half *tgt,
                                   __global const half *left,
                                   const int ncols_capacity,
                                   const int left_cols_capacity) {
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float tgt_value = vload_half(tgt_row * ncols_capacity + tgt_col, (__global const half*) tgt);
    const float left_value = vload_half(tgt_row * left_cols_capacity + tgt_col, (__global const half*) left);
    const float result = tgt_value * left_value;
    vstore_half(result, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Computes the transpose of a matrix
const TRANSPOSE_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void transpose_f16(__global half *tgt,
                            __global const half *left,
                            const int ncols_capacity,
                            const int left_cols_capacity)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const int src_row = tgt_col;
    const int src_col = tgt_row;
    const float val = vload_half(src_row * left_cols_capacity + src_col, (__global const half*) left);
    vstore_half(val, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;
/// Computes x^scalar for every f16 value in the tensor
const POW_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void pow_f16(__global half *tgt,
                      const int ncols_capacity,
                      const float scalar)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float val = vload_half(tgt_row * ncols_capacity + tgt_col, (__global const half*) tgt);
    const float result = pow(val, scalar);
    vstore_half(result, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Computes the mean of each column in a tensor
const MEAN_COLS_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void mean_cols_f16(__global half *tgt,
                            __global const half *left,
                            const int ncols_capacity,
                            const int left_cols_capacity,
                            const int ncolumns)
{
    // global work group size is nrows x 1
    const int row = get_global_id(0);
    float16 src_value = 0.0;
    for (int col16 = 0; col16 < left_cols_capacity; col16 += 16) {
        const int actual_col = col16;
        if (actual_col >= ncolumns) {
            break;
        }
        src_value += vload_half16((row * left_cols_capacity)/16 + col16/16, (__global const half*) left);
    }
    float src_value_sum = src_value.s0 + src_value.s1 + src_value.s2 + src_value.s3 + src_value.s4 + src_value.s5 + src_value.s6 + src_value.s7 + src_value.s8 + src_value.s9 + src_value.sa + src_value.sb + src_value.sc + src_value.sd + src_value.se + src_value.sf;
    src_value_sum = src_value_sum / (float) ncolumns;
    vstore_half(src_value_sum, row * ncols_capacity, (__global half*) tgt);
}
"#;

/// Adds a scalar to a tensor
const ADD_SCALAR_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void add_scalar_f16(__global half *tgt, const int ncols_capacity, const float scalar)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float val = vload_half(tgt_row * ncols_capacity + tgt_col, (__global const half*) tgt);
    const float result = val + scalar;
    vstore_half(result, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Adds scalars from a row vector to each row of a tensor
const SCALAR_MULTIPLY_BROADCAST_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void scalar_multiply_broadcast_f16(__global half *tgt,
                                            __global const half *left,
                                            const int ncols_capacity,
                                            const int left_cols_capacity)
{
    // global work group size is nrows x (ncols/16)
    const int row = get_global_id(0);
    const int col = get_global_id(1) * 16;
    const float scalar = vload_half(row * left_cols_capacity, (__global const half*) left);
    float16 src_value = vload_half16((row * ncols_capacity)/16 + col/16, (__global const half*) tgt) * scalar;
    vstore_half16(src_value, (row * ncols_capacity)/16 + col/16, (__global half*) tgt);
}
"#;

/// Does a hadamard product from a column vector to each column of a tensor
const HADAMARD_PRODUCT_BROADCAST_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void hadamard_product_broadcast_f16(__global half *tgt,
                                             __global const half *left,
                                             const int ncols_capacity,
                                             const int left_cols_capacity)
{
    // global work group size is nrows x (ncols/16)
    const int row = get_global_id(0);
    const int col16 = get_global_id(1) * 16;
    const float16 product_value = vload_half16(col16/16, (__global const half*) left);
    const float16 src_value = vload_half16((row * ncols_capacity)/16 + col16/16, (__global const half*) tgt);
    const float16 result = src_value * product_value;
    vstore_half16(result, (row * ncols_capacity)/16 + col16/16, (__global half*) tgt);
}
"#;

/// Computes 1/sqrt(x) for each f16 value in the tensor
const RSQRT_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rsqrt_f16(__global half *tgt, const int ncols_capacity)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float val = vload_half(tgt_row * ncols_capacity + tgt_col, (__global const half*) tgt);
    const float result = rsqrt(val);
    vstore_half(result, tgt_row * ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;

/// Computes sum of two tensors
const ADD_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void add_f16(__global half *tgt,
                     __global const half *left,
                     const int tgt_ncols_capacity,
                     const int left_ncols_capacity)
{
    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);
    const float tgt_v = vload_half(tgt_row * tgt_ncols_capacity + tgt_col, (__global const half*) tgt);
    const float left_v = vload_half(tgt_row * left_ncols_capacity + tgt_col, (__global const half*) left);
    const float result = tgt_v + left_v;
    vstore_half(result, tgt_row * tgt_ncols_capacity + tgt_col, (__global half*) tgt);
}
"#;