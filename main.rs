use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaModel, AddBos, Special};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::sampling::LlamaSampler;
use clap::Parser;
use rand::Rng;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use chrono::Local;
use num_cpus;
use std::collections::HashSet;

// ============================================================================
// CLI ARGUMENTS
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "llama_selfmod")]
#[command(about = "Self-modifying AI inference engine")]
struct Args {
    /// Model path (.gguf file)
    #[arg(short, long, default_value = "models/model.gguf")]
    model: PathBuf,

    /// Prompt (if not provided, runs in interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Context size
    #[arg(long, default_value = "2048")]
    ctx_size: u32,

    /// Temperature
    #[arg(short, long, default_value = "0.7")]
    temperature: f32,

    /// Top-K sampling
    #[arg(long, default_value = "40")]
    top_k: i32,

    /// Top-P sampling
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Min-P sampling
    #[arg(long, default_value = "0.05")]
    min_p: f32,

    /// Repeat penalty
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,

    /// Max tokens to generate
    #[arg(short, long, default_value = "256")]
    n_predict: usize,

    /// Confidence threshold (lower = more retractions)
    #[arg(long, default_value = "0.5")]
    confidence_threshold: f32,

    /// Enable aggressive self-modification
    #[arg(long)]
    aggressive: bool,

    /// Max number of retractions allowed
    #[arg(long, default_value = "3")]
    max_retractions: usize,

    /// GPU layers to offload (0 = CPU only, -1 = all)
    #[arg(long, default_value = "0")]
    n_gpu_layers: i32,

    /// Number of threads (auto-detected if not set)
    #[arg(long)]
    threads: Option<u32>,

    /// Memory log file
    #[arg(long, default_value = "selfmod_memory.jsonl")]
    memory_file: PathBuf,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,

    /// Minimal output
    #[arg(long)]
    quiet: bool,
}

// ============================================================================
// MODIFICATION TRACKING
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModificationEvent {
    token_idx: usize,
    mod_type: String,
    confidence: f32,
    description: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct MemoryEntry {
    timestamp: String,
    prompt: String,
    response: String,
    modifications: Vec<ModificationEvent>,
    avg_confidence: f32,
    final_temperature: f32,
    retractions: usize,
}

// ============================================================================
// SELF-MODIFICATION STATE
// ============================================================================

struct SelfModState {
    temperature: f32,
    repeat_penalty: f32,
    top_p: f32,
    min_p: f32,
    top_k: i32,
    confidence_history: Vec<f32>,
    modifications: Vec<ModificationEvent>,
    token_history: Vec<LlamaToken>,
    retract_count: usize,
    confidence_threshold: f32,
    aggressive: bool,
}

impl SelfModState {
    fn new(args: &Args) -> Self {
        Self {
            temperature: args.temperature,
            repeat_penalty: args.repeat_penalty,
            top_p: args.top_p,
            min_p: args.min_p,
            top_k: args.top_k,
            confidence_history: Vec::new(),
            modifications: Vec::new(),
            token_history: Vec::new(),
            retract_count: 0,
            confidence_threshold: args.confidence_threshold,
            aggressive: args.aggressive,
        }
    }

    fn assess_and_modify(&mut self, confidence: f32, token_idx: usize) -> bool {
        self.confidence_history.push(confidence);
        
        // Calculate recent average confidence
        let recent_avg = if self.confidence_history.len() >= 5 {
            let len = self.confidence_history.len();
            self.confidence_history[len-5..].iter().sum::<f32>() / 5.0
        } else {
            confidence
        };

        let mut should_retract = false;

        // CRITICAL: Very low confidence - retract
        if recent_avg < 0.35 && self.aggressive {
            self.temperature = (self.temperature * 1.4).min(1.5);
            self.modifications.push(ModificationEvent {
                token_idx,
                mod_type: "CRITICAL_RETRACT".to_string(),
                confidence: recent_avg,
                description: format!("Critical low conf {:.3}, retracting, temp -> {:.2}", 
                    recent_avg, self.temperature),
            });
            should_retract = true;
            self.retract_count += 1;
        } 
        // MODERATE: Low confidence - adjust parameters
        else if recent_avg < self.confidence_threshold {
            self.temperature = (self.temperature * 1.2).min(1.3);
            self.repeat_penalty = (self.repeat_penalty * 1.1).min(1.5);
            self.modifications.push(ModificationEvent {
                token_idx,
                mod_type: "PARAM_ADJUST".to_string(),
                confidence: recent_avg,
                description: format!("Low conf {:.3}, boosting exploration, temp -> {:.2}", 
                    recent_avg, self.temperature),
            });
        } 
        // HIGH: Sharpen output
        else if recent_avg > 0.85 && self.temperature > 0.4 {
            self.temperature = (self.temperature * 0.88).max(0.3);
            self.modifications.push(ModificationEvent {
                token_idx,
                mod_type: "SHARPEN".to_string(),
                confidence: recent_avg,
                description: format!("High conf {:.3}, sharpening, temp -> {:.2}", 
                    recent_avg, self.temperature),
            });
        }

        // Detect repetition
        if self.token_history.len() >= 10 {
            let last_10 = &self.token_history[self.token_history.len()-10..];
            let unique: HashSet<_> = last_10.iter().collect();
            if unique.len() < 5 {
                self.repeat_penalty = (self.repeat_penalty * 1.15).min(2.0);
                self.modifications.push(ModificationEvent {
                    token_idx,
                    mod_type: "ANTI_REPETITION".to_string(),
                    confidence: recent_avg,
                    description: format!("Repetition detected, repeat_penalty -> {:.2}", 
                        self.repeat_penalty),
                });
            }
        }

        // Aggressive mode: random exploration
        if self.aggressive && rand::thread_rng().gen::<f32>() < 0.05 {
            let noise = rand::thread_rng().gen_range(-0.15..0.15);
            self.temperature = (self.temperature + noise).clamp(0.3, 1.5);
            self.modifications.push(ModificationEvent {
                token_idx,
                mod_type: "EXPLORATION_BURST".to_string(),
                confidence: recent_avg,
                description: format!("Random exploration, temp -> {:.2}", self.temperature),
            });
        }

        should_retract
    }

    fn get_avg_confidence(&self) -> f32 {
        if self.confidence_history.is_empty() {
            0.0
        } else {
            self.confidence_history.iter().sum::<f32>() / self.confidence_history.len() as f32
        }
    }
}

// ============================================================================
// SAMPLING WITH DYNAMIC PARAMETERS
// ============================================================================

fn sample_token(
    ctx: &mut LlamaContext,
    state: &mut SelfModState,
    last_tokens: &[LlamaToken],
) -> Option<(LlamaToken, f32)> {
    // get token data array for last token
    let mut data_array = ctx.token_data_array();

    // Build sampler chain using current dynamic parameters
    // Use penalties to emulate repetition penalties, then top_k/top_p/min_p/temp and greedy/dist sampling
    let penalties_sampler = LlamaSampler::penalties(
        64,
        state.repeat_penalty,
        0.0,
        0.0,
    );

    // Build chain: penalties -> top_k -> top_p -> min_p -> temp -> greedy
    let mut sampler_chain = LlamaSampler::chain_simple([
        penalties_sampler,
        LlamaSampler::top_k(state.top_k),
        LlamaSampler::top_p(state.top_p, 1),
        LlamaSampler::min_p(state.min_p, 1),
        LlamaSampler::temp(state.temperature),
        LlamaSampler::greedy(),
    ]);

    // Feed previous tokens into sampler state (for repetition-aware samplers)
    sampler_chain = sampler_chain.with_tokens(last_tokens.iter().copied());

    // Apply sampler to data_array
    data_array.apply_sampler(&mut sampler_chain);

    // Get confidence from top candidate (if any)
    let confidence = data_array.data.first().map(|d| d.p()).unwrap_or(0.0);

    // Sample token (greedy because chain ends with greedy)
    let token = data_array.sample_token_greedy();

    Some((token, confidence))
}

// ============================================================================
// GENERATION WITH SELF-MODIFICATION
// ============================================================================

fn generate_with_selfmod(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    initial_tokens: Vec<LlamaToken>,
    args: &Args,
) -> (String, SelfModState, Vec<LlamaToken>) {
    let mut state = SelfModState::new(args);
    let mut tokens: Vec<LlamaToken> = initial_tokens.clone();
    let mut response = String::new();
    let mut n_past: i32;

    if !args.quiet {
        println!("🔥 Self-Modifying Generation Active...\n");
    }

    // Initial evaluation - process the prompt
    let mut batch = LlamaBatch::new(args.ctx_size as usize, 1);
    for (i, &token) in tokens.iter().enumerate() {
        batch.add(token, i as i32, &[0], i == tokens.len() - 1).unwrap();
    }
    ctx.decode(&mut batch).expect("Failed to decode initial batch");
    n_past = tokens.len() as i32;

    // Generation loop
    for gen_idx in 0..args.n_predict {
        // Sample next token with current dynamic parameters
        let last_n_tokens: Vec<LlamaToken> = if tokens.len() > 64 {
            tokens[tokens.len()-64..].to_vec()
        } else {
            tokens.clone()
        };
        
        let (next_token, confidence) = match sample_token(ctx, &mut state, &last_n_tokens) {
            Some(t) => t,
            None => break,
        };

        // Check for end of sequence
        if next_token == model.token_eos() {
            break;
        }

        tokens.push(next_token);
        state.token_history.push(next_token);

        // Convert token to text
        let piece = match model.token_to_str(next_token, Special::Tokenize) {
            Ok(s) => s,
            Err(_) => "[?]".to_string(),
        };
        response.push_str(&piece);
        print!("{}", piece);
        io::stdout().flush().unwrap();

        // SELF-MODIFICATION DECISION POINT
        let should_retract = state.assess_and_modify(confidence, gen_idx);

        if should_retract && tokens.len() > initial_tokens.len() + 3 && 
           state.retract_count <= args.max_retractions {
            
            // TRUE RETRACTION: Roll back tokens and KV cache
            let retract_n = 3.min(tokens.len() - initial_tokens.len());
            
            if !args.quiet {
                println!("\n⚠️  [RETRACTING {} TOKENS - Confidence: {:.3}]", retract_n, confidence);
            }
            
            // Remove tokens from history
            for _ in 0..retract_n {
                tokens.pop();
                state.token_history.pop();
                // Remove characters from response (approximate)
                for _ in 0..10 {
                    if response.is_empty() { break; }
                    response.pop();
                }
            }

            // Reset context to retracted state
            n_past -= retract_n as i32;
            ctx.clear_kv_cache_seq(Some(0), Some(n_past as u32), None).expect("Failed to clear KV cache seq");

            if !args.quiet {
                print!("🔄 [CONTINUING - TEMP={:.2}] ", state.temperature);
                io::stdout().flush().unwrap();
            }
            continue;
        }

        // Continue normal generation
        let mut batch = LlamaBatch::new(args.ctx_size as usize, 1);
        batch.add(next_token, n_past, &[0], true).unwrap();
        
        if let Err(e) = ctx.decode(&mut batch) {
            eprintln!("\nDecode error: {:?}", e);
            break;
        }
        n_past += 1;
    }

    println!("\n");

    let generated_tokens = tokens[initial_tokens.len()..].to_vec();
    (response, state, generated_tokens)
}

// ============================================================================
// SAVE TO MEMORY LOG
// ============================================================================

fn save_memory(file: &PathBuf, prompt: &str, response: &str, state: &SelfModState) {
    let entry = MemoryEntry {
        timestamp: Local::now().to_rfc3339(),
        prompt: prompt.to_string(),
        response: response.to_string(),
        modifications: state.modifications.clone(),
        avg_confidence: state.get_avg_confidence(),
        final_temperature: state.temperature,
        retractions: state.retract_count,
    };

    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(file) {
        if let Ok(json) = serde_json::to_string(&entry) {
            writeln!(f, "{}", json).ok();
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let args = Args::parse();

    println!("🚀 llama_selfmod - Self-Modifying AI Engine");
    println!("{}", "━".repeat(60));
    println!();

    // Initialize llama.cpp backend
    let backend = LlamaBackend::init().expect("Failed to initialize backend");

    // Load model
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(args.n_gpu_layers as u32);
    
    if !args.quiet {
        println!("📦 Loading model: {:?}", args.model);
    }

    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .expect("Failed to load model - check path and model format");

    // Create context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(args.ctx_size).unwrap()))
        .with_n_threads(args.threads.unwrap_or(num_cpus::get() as u32).try_into().unwrap())
        .with_n_threads_batch(args.threads.unwrap_or(num_cpus::get() as u32).try_into().unwrap());

    let mut ctx = model.new_context(&backend, ctx_params)
        .expect("Failed to create context");

    if !args.quiet {
        println!("✅ Model loaded | ctx={} | gpu_layers={} | aggressive={}\n", 
            args.ctx_size, args.n_gpu_layers, args.aggressive);
    }

    // Single-shot or interactive mode
    if let Some(prompt) = &args.prompt {
        // Single prompt mode
        let full_prompt = format!("User: {}\nAssistant:", prompt);
        let tokens = model.str_to_token(&full_prompt, AddBos::Always)
            .expect("Failed to tokenize");
        
        let (response, state, _generated_tokens) = generate_with_selfmod(&mut ctx, &model, tokens, &args);
        
        if !args.quiet {
            println!("📊 Generation Stats:");
            println!("   • Modifications: {}", state.modifications.len());
            println!("   • Avg Confidence: {:.3}", state.get_avg_confidence());
            println!("   • Final Temp: {:.2}", state.temperature);
            println!("   • Retractions: {}", state.retract_count);
        }

        save_memory(&args.memory_file, prompt, &response, &state);
    } else {
        // Interactive mode
        println!("💬 Interactive Mode (type 'exit' to quit)\n");
        let mut conversation_tokens: Vec<LlamaToken> = Vec::new();
        loop {
            print!("User: ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "exit" || input == "quit" {
                break;
            }

            if input.is_empty() {
                continue;
            }

            let prompt_prefix = if conversation_tokens.is_empty() { "".to_string() } else { "\n".to_string() };
            let full_prompt = format!("{}User: {}\nAssistant:", prompt_prefix, input);
            let add_bos = if conversation_tokens.is_empty() { AddBos::Always } else { AddBos::Never };
            let new_tokens = model.str_to_token(&full_prompt, add_bos)
                .expect("Failed to tokenize");
            
            conversation_tokens.extend(new_tokens);

            // Truncate if necessary
            let max_ctx = args.ctx_size as usize;
            let buffer = args.n_predict;
            if conversation_tokens.len() > max_ctx - buffer {
                let keep_len = max_ctx - buffer;
                conversation_tokens = conversation_tokens[conversation_tokens.len() - keep_len..].to_vec();
            }

            ctx.clear_kv_cache();
            let (response, state, generated_tokens) = generate_with_selfmod(&mut ctx, &model, conversation_tokens.clone(), &args);
            
            conversation_tokens.extend(generated_tokens);

            if !args.quiet {
                println!("📊 [{} mods | conf: {:.2} | temp: {:.2}]\n", 
                    state.modifications.len(), 
                    state.get_avg_confidence(),
                    state.temperature);
            }

            save_memory(&args.memory_file, input, &response, &state);
        }
    }

    println!("👋 Session ended. Memory saved to {:?}", args.memory_file);
}
