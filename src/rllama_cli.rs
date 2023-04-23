use crate::data_source::DataSource;
use crate::embedding::Embedding;
use crate::model_params::ModelParams;

#[cfg(feature = "opencl")]
use crate::tensor_opencl_support::OpenCL;

use crate::token_sampler::TokenSampler;
use crate::tokenizer::{TokenId, Tokenizer};
use crate::transformer::{DataSettings, Transformer};

use clap::Parser;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::Arc;
#[cfg(feature = "speech")]
use tts::*;

// Refer to README.md to see what all these options mean.
#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    tokenizer_path: String,
    #[arg(long)]
    param_path: String,

    #[arg(short, long, action)]
    quiet: bool,
 
    #[arg(long, action)]
    f16: bool,
    #[arg(long)]
    max_seq_len: Option<usize>,

    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<i32>,
    #[arg(long)]
    repetition_penalty: Option<f32>,

    #[arg(long)]
    max_threads: Option<usize>,

    #[cfg(feature = "opencl")]
    #[arg(long)]
    opencl_device: Option<usize>,

    #[cfg(feature = "opencl")]
    #[arg(long)]
    percentage_to_gpu: Option<f32>,
    
    #[cfg(feature = "speech")]
    #[arg(long, action)]
    speak: bool,

    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<String>,
    #[arg(long)]
    query: Option<String>,

    #[arg(long, action)]
    start_interactive: bool,
    #[arg(long, action)]
    is_interactive: bool,
    #[arg(long, action)]
    show_interactions: bool,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    let model_path = cli.model_path.clone();
    let tokenizer_path = cli.tokenizer_path.clone();
    let param_path = cli.param_path.clone();


    
    
    #[cfg(feature = "speech")]
    let speak = cli.speak;

    let max_threads: usize = match cli.max_threads {
        None => rayon::current_num_threads(),
        Some(max_threads) => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .unwrap();
            max_threads
        }
    };

    #[cfg(feature = "opencl")]
    let percentage_to_gpu: f32 = cli.percentage_to_gpu.unwrap_or(1.0);

    let mut be_quiet: bool = false;
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        be_quiet = true;
    }
    if cli.quiet {
        be_quiet = true;
    }
    if be_quiet {
        colored::control::SHOULD_COLORIZE.set_override(false);
    }

    // Custom println-like macro that respects be_quiet
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    #[cfg(feature = "opencl")]
    let opencl: Option<OpenCL> = {
        let opencl_device = cli.opencl_device.unwrap_or(0);
        match OpenCL::new(!be_quiet, opencl_device) {
            Err(openclerr) => {
                eprintln!("OpenCL error: {}", openclerr);
                eprintln!("OpenCL is disabled because it failed to initialize.");
                None
            }
            Ok(opencl) => {
                println!("OpenCL initialized.");
                Some(opencl)
            }
        }
    };

    #[cfg(feature = "opencl")]
    let has_opencl = opencl.is_some();

    // Read ModelParams from param_path, we expect it to be JSON
    let mut fs = std::fs::File::open(&param_path)?;
    let mut bs = Vec::new();
    fs.read_to_end(&mut bs)?;
    std::mem::drop(fs);

    let prompt: String = match (&cli.prompt, &cli.prompt_file) {
        (Some(ref prompt), None) => {
            pln!("Using prompt: {}", prompt);
            prompt.clone()
        }
        (None, Some(ref prompt_file)) => {
            pln!("Using prompt file: {}", prompt_file);
            let mut fs = std::fs::File::open(prompt_file)?;
            let mut bs = Vec::new();
            fs.read_to_end(&mut bs)?;
            std::mem::drop(fs);
            String::from_utf8(bs)?
        }
        _ => {
            eprintln!("Please provide either a prompt or a prompt file.");
            return Err("Please provide either a prompt or a prompt file.".into());
        }
    };
    
    let (interactive_system_prompt, interactive_prompt_prefix, interactive_prompt_postfix) = match (prompt.find("{SYSTEM_PROMPT}"),prompt.find("{USER_QUERY}")) {
        (Some(ref sys_position), Some(ref user_position)) => {
            (&prompt[0..*sys_position],&prompt[*sys_position+15..*user_position],&prompt[*user_position+12..])
        }
        _ => {
            ("","","")
        }
    };
    let (start_interactive, user_query): (bool, String) = match cli.query {
        Some(ref query) => {
            (query.is_empty(), query.to_string())
        }
        _ => {
            (true, "".to_string())
        }
    };
    pln!("Prompt loaded {} {} {}", interactive_system_prompt, interactive_prompt_prefix, interactive_prompt_postfix);
    let interactive_stop = interactive_prompt_prefix;
    
    let is_interactive = cli.is_interactive || start_interactive;
    let show_interactions = cli.show_interactions;


    pln!("Starting up. Loading tokenizer from {}...", tokenizer_path);
    let tok = Tokenizer::load(tokenizer_path.as_str())?;
    pln!("Tokenizer loaded. Loading model from {}...", model_path);

    let model_data_source = DataSource::from_inferred_source(model_path.clone())?;

    let params: ModelParams = serde_json::from_slice(&bs)?;
    pln!("Loaded model parameters from {}.", param_path);

    pln!("Loading embeddings from {}...", model_path);
    let emb = Embedding::from_unpickled(model_data_source.clone())?;

    let max_seq_len = cli.max_seq_len.unwrap_or(1024);

    let mut data_settings = {
        #[cfg(feature = "opencl")]
        {
            if let Some(opencl) = opencl {
                let ds = DataSettings::new(Some(opencl));
                ds.percentage_to_gpu(percentage_to_gpu).use_opencl()
            } else {
                DataSettings::new(None)
            }
        }
        #[cfg(not(feature = "opencl"))]
        DataSettings::new()
    };

    #[cfg(feature = "opencl")]
    if cli.f16 || has_opencl {
        data_settings = data_settings.force_f16();
    }
    #[cfg(not(feature = "opencl"))]
    if cli.f16 {
        data_settings = data_settings.force_f16();
    }

    pln!("Loading transformer weights from {}...", model_path);
    let tr = Transformer::from_unpickled(
        emb,
        params.dim,
        params.n_layers,
        params.n_heads,
        max_seq_len,
        params.norm_eps,
        data_settings,
        model_data_source,
    )?;
    pln!("All is loaded. Starting inference.");

    let tr: Arc<Transformer> = Arc::new(tr);
    let tok: Arc<Tokenizer> = Arc::new(tok);


        command_line_inference(
            cli.clone(),
            tr.clone(),
            tok.clone(),
            prompt.clone(),
            user_query.clone(),
            interactive_stop,
            interactive_system_prompt,
            interactive_prompt_prefix,
            interactive_prompt_postfix,
            start_interactive,
            is_interactive,
            show_interactions,
            #[cfg(feature = "speak")]
            speak,
            be_quiet,
            max_seq_len,
            params.clone(),
            max_threads,
        )

}



fn command_line_inference(
    cli: Cli,
    tr: Arc<Transformer>,
    tok: Arc<Tokenizer>,
    prompt: String,
    query: String,
    interactive_stop: &str,
    interactive_system_prompt: &str,
    interactive_prompt_prefix: &str,
    interactive_prompt_postfix: &str,
    start_interactive: bool,
    is_interactive: bool,
    show_interactions: bool,
    #[cfg(feature = "speak")]
    speak: bool,
    be_quiet: bool,
    max_seq_len: usize,
    params: ModelParams,
    max_threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Custom println-like macro that respects be_quiet
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    #[cfg(feature = "speech")]
    let mut tts = Tts::default()?;

    let mut prompt = prompt;

    //let mut toks_id: Vec<TokenId> = tok.tokenize_to_ids(prompt.clone());
    let mut toks_id: Vec<TokenId> = tok.tokenize_to_ids(interactive_system_prompt);
        toks_id.append(&mut tok.more_tokenize_to_ids(interactive_prompt_prefix));
    if !start_interactive {
        toks_id.append(&mut tok.more_tokenize_to_ids(query.clone()));
        toks_id.append(&mut tok.more_tokenize_to_ids(interactive_prompt_postfix));
    }
    let mut prev_pos = 0;
    let mut token_sampler = TokenSampler::new()
        .temperature(1.0)
        .top_p(1.0)
        .top_k(20)
        .repetition_penalty(1.0);

    if let Some(temperature) = cli.temperature {
        token_sampler = token_sampler.temperature(temperature);
    }
    if let Some(top_p) = cli.top_p {
        token_sampler = token_sampler.top_p(top_p);
    }
    if let Some(top_k) = cli.top_k {
        token_sampler = token_sampler.top_k(top_k as usize);
    }
    if let Some(repetition_penalty) = cli.repetition_penalty {
        token_sampler = token_sampler.repetition_penalty(repetition_penalty);
    }
    let mut stop_tokens = tok.more_tokenize_to_ids(interactive_stop.clone());
    pln!("---");
    pln!(" dim: {}", params.dim);
    pln!(" n_heads: {}", params.n_heads);
    pln!(" n_layers: {}", params.n_layers);
    pln!(" norm_eps: {}", params.norm_eps);
    pln!(" vocab_size: {}", params.vocab_size);
    pln!("---");
    pln!(" maximum number of threads: {}", max_threads);
    pln!("---");
    pln!("Max sequence length: {}", max_seq_len);
    pln!("Temperature: {}", token_sampler.get_temperature());
    pln!("Top P: {}", token_sampler.get_top_p());
    pln!("Top K: {}", token_sampler.get_top_k());
    pln!(
        "Repetition penalty: {}",
        token_sampler.get_repetition_penalty()
    );
    if is_interactive {
        pln!(
            "  Interactive mode stop token sequences: {:?}",
            interactive_stop
        );
        pln!("---");
        pln!("System prompt:");
        pln!("  {}", interactive_system_prompt);
        pln!("Interactive prompt prefix: {}", interactive_prompt_prefix);
        pln!("Interactive prompt postfix: {}", interactive_prompt_postfix);
    }
    pln!("---");
    pln!(
        "{}",
        "  This is the color of the initial prompt".truecolor(128, 128, 255)
    );
    pln!(
        "{}",
        "  This is the color of the generated text".truecolor(128, 255, 128)
    );
    pln!("---");
    print!("{}{}{}{}", interactive_system_prompt.truecolor(128, 128, 255),
        interactive_prompt_prefix.truecolor(128, 128, 255),
        query.as_str(),
        interactive_prompt_postfix.truecolor(128, 128, 255));

    let _ = std::io::stdout().flush();

    let mut first_token_time: std::time::Duration = std::time::Duration::new(0, 0);
    let mut times_per_token: Vec<std::time::Duration> = vec![];
    let mut caches = tr.make_caches();
    let mut first: bool = true;
    let mut stop_seen: bool = false;
    let mut interactive = start_interactive;
    let mut user_token: Vec<TokenId> = vec![];
    #[cfg(feature = "speech")]
    let mut tok_speak: String = "".to_string();
    while toks_id.len() < max_seq_len {
        let now = std::time::Instant::now();
        let preds = tr.forward(&toks_id[prev_pos..], prev_pos, &mut caches);
        if interactive {
            let mut newinput = String::new();
            loop {
                let mut newline = String::new();
                std::io::stdin().read_line(&mut newline)?;
                //removing new line from input
                if newline.ends_with('\n') {
                    newline.pop();
                    if newline.ends_with('\r') {
                        newline.pop();
                    }
                }
                
                if !newline.ends_with('\\') {
                    newinput = newinput + &newline.clone();
                    break;
                }
                else {
                    newline.pop();
                    newinput = newinput + &newline.clone() + "\n";
                }
            }
            //exit clause
            if newinput.starts_with("</s>") { stop_seen = true; break;}

            newinput = interactive_prompt_prefix.to_string() + &newinput;

            newinput += &interactive_prompt_postfix;
            user_token.append(&mut tok.more_tokenize_to_ids(newinput.clone()));
            
            interactive = false;
        if !show_interactions {
                if interactive_prompt_postfix.starts_with('\n') {
                    //is that safe ?
                    print!("{}", &interactive_prompt_postfix[1..].truecolor(128, 128, 255));
                }
                if interactive_prompt_postfix.starts_with('\r') {
                    //is that safe ? windows prompt file...
                    print!("{}", &interactive_prompt_postfix[2..].truecolor(128, 128, 255));
                }
                else {
                    print!("{}", interactive_prompt_postfix.truecolor(128, 128, 255));
                }
            }
        }
        let (mut highest_pred_idx, mut token_prob);

        if user_token.len() > 0 {
            highest_pred_idx = user_token.remove(0);
            token_prob = 0.0;
        } else {
            (highest_pred_idx, token_prob) = token_sampler.sample(&preds, &tok, &toks_id);
        }
        toks_id.push(highest_pred_idx as TokenId);

        for (tok_idx, tok_id) in toks_id[prev_pos + 1..].iter().enumerate() {
            if *tok_id == 1 {
                continue;
            }
            let mut tok_print: String = "".to_string();
            let tok_str = tok.id_to_str(*tok_id);
            if tok_str == "</s>" {
                stop_seen = true;
            }
            else if tok_str == "<0x0A>" {
                tok_print += "\n";
            } else {
                tok_print += tok_str.replace('▁', " ").as_str();
            }
            if first && tok_idx < toks_id.len() - 2 {
                // intentionally left empty, already print
            }
            else if !show_interactions && token_prob == 0.0 {
            // intentionally left empty, User print
            }
            else {
                let redness: f32 = token_prob * 255.0;
                let redness = if redness > 255.0 {
                    255
                } else if redness < 0.0 {
                    0
                } else {
                    redness as u8
                };
                print!(
                    "{}",
                    tok_print.truecolor(128 + redness / 2, 255 - redness / 2, 128)
                );
                #[cfg(feature = "speech")]
                if speak
                {
                    tok_speak += tok_print.as_str();
                    if tok_speak.ends_with(".")
                        || (tok_speak.ends_with(",") && !tts.is_speaking()?)
                        || tok_speak.ends_with("?")
                        || tok_speak.ends_with("!")
                        || tok_speak.ends_with("</s>")
                        || tok_speak.ends_with("\n")
                    {
                        tts.speak(tok_speak.clone(), false)?;
                        tok_speak.clear();
                    }
                }
                
                

            };
            if !first
                && tok_id == stop_tokens.last().unwrap()
                && tok_idx + prev_pos > stop_tokens.len()
                && toks_id
                    [prev_pos + 1 + tok_idx - (stop_tokens.len() - 1)..prev_pos + 1 + tok_idx + 1]
                    == stop_tokens
            {
                if is_interactive {
                    interactive = true;
                }
            }
        }
        if first {
            first_token_time = now.elapsed();
        } else {
            times_per_token.push(now.elapsed());
        }
        let _ = std::io::stdout().flush();
        prev_pos = toks_id.len() - 1;
        first = false;
        if stop_seen {
            if is_interactive {
                //don't stop, just ask for more
                stop_seen = false;
                if !show_interactions {
                    print!("{}",interactive_stop.truecolor(128, 128, 255));
                    let _ = std::io::stdout().flush();
                }
                user_token = tok.more_tokenize_to_ids(interactive_stop.clone());
            }
            else {
                break;
            }
        }
    }
    println!();
    if stop_seen && !be_quiet {
        println!("Stop token seen. Stopping.");
    }
    if !be_quiet {
        println!("---");
        println!(
            "Time taken to generate first token: {:?}ms",
            first_token_time.as_millis()
        );
        if times_per_token.len() > 0 {
            println!(
                "Time taken per token (excluding first token): {:?}ms",
                times_per_token.iter().map(|t| t.as_millis()).sum::<u128>()
                    / times_per_token.len() as u128
            );
        } else {
            println!("No token generated");
        }
    }
    Ok(())
}
