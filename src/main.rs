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
#[command(about = "Multi-model fusion inference engine")]
struct Args {
    /// Model path (.gguf file) - for single model mode
    #[arg(short, long, conflicts_with = "models")]
    model: Option<PathBuf>,

    /// Multiple model paths for fusion inference (comma-separated)
    #[arg(long, value_delimiter = ',')]
    models: Vec<PathBuf>,

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

    /// Fusion mode: average, confidence, voting
    #[arg(long, default_value = "confidence")]
    fusion_mode: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    fusion_data: Option<FusionMetadata>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FusionMetadata {
    model_confidences: Vec<(String, f32)>,
    leading_model: String,
    agreement_score: f32,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    num_models: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fusion_mode: Option<String>,
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
                fusion_data: None,
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
                fusion_data: None,
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
                fusion_data: None,
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
                    fusion_data: None,
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
                fusion_data: None,
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
// MULTI-MODEL FUSION ENSEMBLE
// ============================================================================

#[derive(Debug, Clone, Copy)]
enum FusionMode {
    Average,           // Simple average of probabilities
    ConfidenceWeighted,// Weight by top-1 probability
    Voting,           // Token voting
}

impl FusionMode {
    fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "average" => FusionMode::Average,
            "confidence" => FusionMode::ConfidenceWeighted,
            "voting" => FusionMode::Voting,
            _ => FusionMode::ConfidenceWeighted,
        }
    }
}

// ModelInstance combines a model and its context
// We use a struct with both to ensure proper ownership
struct ModelInstance<'a> {
    model: Box<LlamaModel>,
    ctx: Option<LlamaContext<'a>>,
    name: String,
}

// Ensemble holds multiple model instances
// The contexts reference their respective models within each ModelInstance
struct ModelEnsemble<'a> {
    instances: Vec<ModelInstance<'a>>,
    fusion_mode: FusionMode,
}

impl<'a> ModelEnsemble<'a> {
    fn new(
        backend: &'a LlamaBackend,
        model_paths: Vec<PathBuf>,
        args: &Args,
    ) -> Result<Self, String> {
        if model_paths.is_empty() {
            return Err("No model paths provided".to_string());
        }

        let fusion_mode = FusionMode::from_string(&args.fusion_mode);
        let mut instances: Vec<ModelInstance> = Vec::new();

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(args.n_gpu_layers as u32);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(args.ctx_size).unwrap()))
            .with_n_threads(args.threads.unwrap_or(num_cpus::get() as u32).try_into().unwrap())
            .with_n_threads_batch(args.threads.unwrap_or(num_cpus::get() as u32).try_into().unwrap());

        // Load each model and create its context
        for (idx, path) in model_paths.iter().enumerate() {
            if !args.quiet {
                println!("📦 Loading model {}/{}: {:?}", idx + 1, model_paths.len(), path);
            }

            let model = LlamaModel::load_from_file(backend, path, &model_params)
                .map_err(|e| format!("Failed to load model {:?}: {:?}", path, e))?;

            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&format!("model_{}", idx))
                .to_string();

            let boxed_model = Box::new(model);

            // Create context referencing the boxed model
            // SAFETY: The context will live within the same ModelInstance as the model,
            // ensuring the model outlives the context. We must ensure ModelInstance is never
            // moved after context creation.
            let model_ref: &'a LlamaModel = unsafe { &*(&*boxed_model as *const LlamaModel) };
            let ctx = model_ref.new_context(backend, ctx_params.clone())
                .map_err(|e| format!("Failed to create context for model {}: {:?}", idx, e))?;

            instances.push(ModelInstance {
                model: boxed_model,
                ctx: Some(ctx),
                name,
            });
        }

        if !args.quiet {
            println!("✅ Loaded {} models | fusion_mode: {:?}\n", instances.len(), fusion_mode);
        }

        Ok(ModelEnsemble {
            instances,
            fusion_mode,
        })
    }

    fn len(&self) -> usize {
        self.instances.len()
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
// FUSION SAMPLING - THE COLLECTIVE CONSCIOUSNESS
// ============================================================================

fn sample_token_fusion<'a>(
    ensemble: &mut ModelEnsemble<'a>,
    state: &mut SelfModState,
    last_tokens: &[LlamaToken],
    _token_idx: usize,
) -> Option<(LlamaToken, f32, FusionMetadata)> {
    use std::collections::HashMap;

    if ensemble.instances.is_empty() {
        return None;
    }

    // Collect probability distributions from all models
    let mut model_distributions: Vec<(String, LlamaTokenDataArray, f32)> = Vec::new();

    for instance in ensemble.instances.iter_mut() {
        let ctx = instance.ctx.as_mut().unwrap();
        let mut data_array = ctx.token_data_array();

        // Apply sampling parameters
        let penalties_sampler = LlamaSampler::penalties(64, state.repeat_penalty, 0.0, 0.0);
        let mut sampler_chain = LlamaSampler::chain_simple([
            penalties_sampler,
            LlamaSampler::top_k(state.top_k),
            LlamaSampler::top_p(state.top_p, 1),
            LlamaSampler::min_p(state.min_p, 1),
            LlamaSampler::temp(state.temperature),
        ]);

        sampler_chain = sampler_chain.with_tokens(last_tokens.iter().copied());
        data_array.apply_sampler(&mut sampler_chain);

        let top_confidence = data_array.data.first().map(|d| d.p()).unwrap_or(0.0);
        model_distributions.push((instance.name.clone(), data_array, top_confidence));
    }

    // Fusion logic based on mode
    let (fused_token, fused_confidence, leading_model, agreement_score) = match ensemble.fusion_mode {
        FusionMode::Voting => {
            // Each model votes for its top token
            let mut votes: HashMap<LlamaToken, Vec<(String, f32)>> = HashMap::new();

            for (name, data_array, confidence) in &model_distributions {
                if let Some(top_token_data) = data_array.data.first() {
                    let token = top_token_data.id();
                    votes.entry(token)
                        .or_insert_with(Vec::new)
                        .push((name.clone(), *confidence));
                }
            }

            // Find token with most votes (weighted by confidence)
            let (winning_token, voters) = votes.into_iter()
                .max_by(|(_, voters_a), (_, voters_b)| {
                    let weight_a: f32 = voters_a.iter().map(|(_, c)| c).sum();
                    let weight_b: f32 = voters_b.iter().map(|(_, c)| c).sum();
                    weight_a.partial_cmp(&weight_b).unwrap()
                })
                .unwrap();

            let avg_conf: f32 = voters.iter().map(|(_, c)| c).sum::<f32>() / voters.len() as f32;
            let leader = voters.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let agreement = voters.len() as f32 / model_distributions.len() as f32;

            (winning_token, avg_conf, leader, agreement)
        },

        FusionMode::Average | FusionMode::ConfidenceWeighted => {
            // Build a unified probability distribution
            let mut fused_probs: HashMap<LlamaToken, f32> = HashMap::new();
            let mut total_weight = 0.0f32;

            for (_name, data_array, top_confidence) in &model_distributions {
                let weight = match ensemble.fusion_mode {
                    FusionMode::ConfidenceWeighted => *top_confidence,
                    _ => 1.0,
                };

                total_weight += weight;

                for token_data in &data_array.data {
                    let token = token_data.id();
                    let prob = token_data.p();
                    *fused_probs.entry(token).or_insert(0.0) += prob * weight;
                }
            }

            // Normalize
            for prob in fused_probs.values_mut() {
                *prob /= total_weight;
            }

            // Select token with highest fused probability
            let (fused_token, fused_prob) = fused_probs.into_iter()
                .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                .unwrap();

            // Determine which model had this token as top choice (for leading_model)
            let mut best_model = "ensemble".to_string();
            let mut best_conf = 0.0f32;
            for (name, data_array, _) in &model_distributions {
                if let Some(top) = data_array.data.first() {
                    if top.id() == fused_token && top.p() > best_conf {
                        best_conf = top.p();
                        best_model = name.clone();
                    }
                }
            }

            // Calculate agreement: how many models have this token in their top-3
            let mut agreement_count = 0;
            for (_, data_array, _) in &model_distributions {
                if data_array.data.iter().take(3).any(|td| td.id() == fused_token) {
                    agreement_count += 1;
                }
            }
            let agreement = agreement_count as f32 / model_distributions.len() as f32;

            (fused_token, fused_prob, best_model, agreement)
        },
    };

    // Build fusion metadata
    let model_confidences: Vec<(String, f32)> = model_distributions.iter()
        .map(|(name, _, conf)| (name.clone(), *conf))
        .collect();

    let fusion_metadata = FusionMetadata {
        model_confidences,
        leading_model: leading_model.clone(),
        agreement_score,
    };

    Some((fused_token, fused_confidence, fusion_metadata))
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
// MULTI-MODEL FUSION GENERATION - COLLECTIVE CONSCIOUSNESS
// ============================================================================

fn generate_with_fusion<'a>(
    ensemble: &mut ModelEnsemble<'a>,
    initial_tokens: Vec<LlamaToken>,
    args: &Args,
) -> (String, SelfModState, Vec<LlamaToken>) {
    let mut state = SelfModState::new(args);
    let mut tokens: Vec<LlamaToken> = initial_tokens.clone();
    let mut response = String::new();
    let mut n_past_per_model: Vec<i32> = vec![0; ensemble.len()];

    if !args.quiet {
        println!("🌐 Multi-Model Fusion Active - {} Models Inferring Together...\n",
            ensemble.len());
    }

    // Initial evaluation - process the prompt for ALL models
    for (idx, instance) in ensemble.instances.iter_mut().enumerate() {
        let ctx = instance.ctx.as_mut().unwrap();
        let mut batch = LlamaBatch::new(args.ctx_size as usize, 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == tokens.len() - 1).unwrap();
        }
        ctx.decode(&mut batch).expect("Failed to decode initial batch");
        n_past_per_model[idx] = tokens.len() as i32;
    }

    // Generation loop with fusion
    for gen_idx in 0..args.n_predict {
        // Sample next token with FUSION across all models
        let last_n_tokens: Vec<LlamaToken> = if tokens.len() > 64 {
            tokens[tokens.len()-64..].to_vec()
        } else {
            tokens.clone()
        };

        let (next_token, confidence, fusion_metadata) = match sample_token_fusion(
            ensemble, &mut state, &last_n_tokens, gen_idx
        ) {
            Some(t) => t,
            None => break,
        };

        // Check for end of sequence (use first model's EOS token)
        if next_token == ensemble.instances[0].model.token_eos() {
            break;
        }

        tokens.push(next_token);
        state.token_history.push(next_token);

        // Convert token to text (use first model for detokenization)
        let piece = match ensemble.instances[0].model.token_to_str(next_token, Special::Tokenize) {
            Ok(s) => s,
            Err(_) => "[?]".to_string(),
        };
        response.push_str(&piece);
        print!("{}", piece);
        io::stdout().flush().unwrap();

        // Display fusion stats occasionally
        if !args.quiet && gen_idx % 20 == 0 && gen_idx > 0 {
            print!(" [🔮 {}→{:.0}%] ",
                &fusion_metadata.leading_model[..fusion_metadata.leading_model.len().min(8)],
                fusion_metadata.agreement_score * 100.0);
            io::stdout().flush().unwrap();
        }

        // SELF-MODIFICATION DECISION POINT
        let should_retract = state.assess_and_modify(confidence, gen_idx);

        // Log fusion event with high disagreement or interesting patterns
        if fusion_metadata.agreement_score < 0.5 || gen_idx % 50 == 0 {
            state.modifications.push(ModificationEvent {
                token_idx: gen_idx,
                mod_type: "FUSION_SAMPLE".to_string(),
                confidence,
                description: format!("Leader: {}, Agreement: {:.0}%",
                    fusion_metadata.leading_model, fusion_metadata.agreement_score * 100.0),
                fusion_data: Some(fusion_metadata.clone()),
            });
        }

        if should_retract && tokens.len() > initial_tokens.len() + 3 &&
           state.retract_count <= args.max_retractions {

            // TRUE RETRACTION: Roll back tokens and KV cache for ALL models
            let retract_n = 3.min(tokens.len() - initial_tokens.len());

            if !args.quiet {
                println!("\n⚠️  [ENSEMBLE RETRACTING {} TOKENS - Confidence: {:.3}]", retract_n, confidence);
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

            // Reset context to retracted state for ALL models
            for (idx, instance) in ensemble.instances.iter_mut().enumerate() {
                let ctx = instance.ctx.as_mut().unwrap();
                n_past_per_model[idx] -= retract_n as i32;
                ctx.clear_kv_cache_seq(
                    Some(0),
                    Some(n_past_per_model[idx] as u32),
                    None
                ).expect("Failed to clear KV cache seq");
            }

            if !args.quiet {
                print!("🔄 [ENSEMBLE CONTINUING - TEMP={:.2}] ", state.temperature);
                io::stdout().flush().unwrap();
            }
            continue;
        }

        // Continue normal generation - feed token to ALL models
        for (idx, instance) in ensemble.instances.iter_mut().enumerate() {
            let ctx = instance.ctx.as_mut().unwrap();
            let mut batch = LlamaBatch::new(args.ctx_size as usize, 1);
            batch.add(next_token, n_past_per_model[idx], &[0], true).unwrap();

            if let Err(e) = ctx.decode(&mut batch) {
                eprintln!("\nDecode error in model {}: {:?}", instance.name, e);
                break;
            }
            n_past_per_model[idx] += 1;
        }
    }

    println!("\n");

    let generated_tokens = tokens[initial_tokens.len()..].to_vec();
    (response, state, generated_tokens)
}

// ============================================================================
// SAVE TO MEMORY LOG
// ============================================================================

fn save_memory(file: &PathBuf, prompt: &str, response: &str, state: &SelfModState) {
    save_memory_with_fusion(file, prompt, response, state, None, None);
}

fn save_memory_with_fusion(
    file: &PathBuf,
    prompt: &str,
    response: &str,
    state: &SelfModState,
    num_models: Option<usize>,
    fusion_mode: Option<String>,
) {
    let entry = MemoryEntry {
        timestamp: Local::now().to_rfc3339(),
        prompt: prompt.to_string(),
        response: response.to_string(),
        modifications: state.modifications.clone(),
        avg_confidence: state.get_avg_confidence(),
        final_temperature: state.temperature,
        retractions: state.retract_count,
        num_models,
        fusion_mode,
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

    println!("🚀 llama_selfmod - Multi-Model Fusion Inference Engine");
    println!("{}", "━".repeat(60));
    println!();

    // Initialize llama.cpp backend
    let backend = LlamaBackend::init().expect("Failed to initialize backend");

    // Determine mode: multi-model fusion or single model
    let use_fusion = !args.models.is_empty();

    if use_fusion {
        // ======================================================================
        // MULTI-MODEL FUSION MODE
        // ======================================================================
        let mut ensemble = ModelEnsemble::new(&backend, args.models.clone(), &args)
            .expect("Failed to create model ensemble");

        if !args.quiet {
            println!("✅ Ensemble ready | {} models | ctx={} | gpu_layers={} | fusion={}\n",
                ensemble.len(), args.ctx_size, args.n_gpu_layers, args.fusion_mode);
        }

        // Single-shot or interactive mode
        if let Some(prompt) = &args.prompt {
            // Single prompt mode
            let full_prompt = format!("User: {}\nAssistant:", prompt);
            let tokens = ensemble.instances[0].model.str_to_token(&full_prompt, AddBos::Always)
                .expect("Failed to tokenize");

            let (response, state, _generated_tokens) = generate_with_fusion(&mut ensemble, tokens, &args);

            if !args.quiet {
                println!("📊 Fusion Generation Stats:");
                println!("   • Models: {}", ensemble.len());
                println!("   • Modifications: {}", state.modifications.len());
                println!("   • Avg Confidence: {:.3}", state.get_avg_confidence());
                println!("   • Final Temp: {:.2}", state.temperature);
                println!("   • Retractions: {}", state.retract_count);
            }

            save_memory_with_fusion(
                &args.memory_file,
                prompt,
                &response,
                &state,
                Some(ensemble.len()),
                Some(args.fusion_mode.clone()),
            );
        } else {
            // Interactive mode
            println!("💬 Interactive Fusion Mode (type 'exit' to quit)\n");
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
                let new_tokens = ensemble.instances[0].model.str_to_token(&full_prompt, add_bos)
                    .expect("Failed to tokenize");

                conversation_tokens.extend(new_tokens);

                // Truncate if necessary
                let max_ctx = args.ctx_size as usize;
                let buffer = args.n_predict;
                if conversation_tokens.len() > max_ctx - buffer {
                    let keep_len = max_ctx - buffer;
                    conversation_tokens = conversation_tokens[conversation_tokens.len() - keep_len..].to_vec();
                }

                // Clear KV cache for all models
                for instance in ensemble.instances.iter_mut() {
                    instance.ctx.as_mut().unwrap().clear_kv_cache();
                }

                let (response, state, generated_tokens) = generate_with_fusion(&mut ensemble, conversation_tokens.clone(), &args);

                conversation_tokens.extend(generated_tokens);

                if !args.quiet {
                    println!("📊 [{} models | {} mods | conf: {:.2} | temp: {:.2}]\n",
                        ensemble.len(),
                        state.modifications.len(),
                        state.get_avg_confidence(),
                        state.temperature);
                }

                save_memory_with_fusion(
                    &args.memory_file,
                    input,
                    &response,
                    &state,
                    Some(ensemble.len()),
                    Some(args.fusion_mode.clone()),
                );
            }
        }
    } else {
        // ======================================================================
        // SINGLE MODEL MODE (backward compatibility)
        // ======================================================================
        let model_path = args.model.as_ref()
            .expect("Either --model or --models must be provided");

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(args.n_gpu_layers as u32);

        if !args.quiet {
            println!("📦 Loading model: {:?}", model_path);
        }

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
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
    }

    println!("👋 Session ended. Memory saved to {:?}", args.memory_file);
}
