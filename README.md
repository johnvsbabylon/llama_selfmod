# llama_selfmod
A self-modifying inference engine built on llama-cpp-2. This Rust-based tool allows an LLM to perform real-time self-correction. Based on token confidence, it can adapt its own sampling parameters on the fly or retract and regenerate text to improve the quality and coherence of its output. I am not claiming perfection.

This tool gives models a real-time feedback loop, allowing them to self-correct during generation to dramatically improve the coherence and quality of their output.

The Problem: Why Small Models Fail

Smaller, local Large Language Models are fantastic, but they often suffer from common failure modes:

Losing the Plot: They start strong but gradually drift off-topic.

Repetition: They get stuck in loops, repeating the same phrases.

Nonsense Generation: They confidently produce text that is grammatically correct but logically nonsensical.

Traditionally, the only fix is to adjust sampling parameters before generation and hope for the best. llama_selfmod takes a different approach.

How It Works: The Self-Correction Loop

llama_selfmod wraps the standard inference process in a dynamic feedback loop. Instead of treating generation as a one-way street, it constantly assesses its own performance and adapts on the fly.

The process for each token is as follows:

Generate a Token: The model predicts the next token as usual.

Measure Confidence: The engine immediately checks the probability (confidence score) of the generated token.

Assess Performance: It analyzes the confidence of the last few tokens to identify trends. Is the model confident and on-track, or is it starting to guess?

Adapt or Retract:

High Confidence: Generation continues as normal.

Wavering Confidence: The engine dynamically adjusts sampling parameters, like increasing the temperature to encourage more creative (and potentially better) outputs.

Critical Low Confidence: The engine retracts the last few low-quality tokens, effectively erasing its mistake from its memory (the KV cache), adjusts its parameters, and tries again from a known-good state.

This cycle of Generate → Assess → Adapt allows the model to "catch" its own mistakes before they derail the entire response, leading to more coherent and reliable output.

Features

Dynamic Sampling: Automatically adjusts temperature and repeat_penalty during inference based on model performance.

Confidence-Based Retraction: Intelligently "backspaces" and rewinds the model's state when it detects a critical drop in quality.

Repetition Detection: Actively monitors for token loops and increases the repeat penalty to break them.

Aggressive Mode: An optional mode for more aggressive parameter adjustments and random exploration bursts to escape creative ruts.

GPU Offloading: Supports offloading layers to the GPU via the n-gpu-layers argument.

Persistent Memory: Logs every generation, including all self-modifications, to a .jsonl file for later analysis.

Interactive & Single-Prompt Modes: Use it as a chatbot or for one-shot text generation.

Getting Started

Prerequisites

Rust and Cargo (https://rustup.rs/)

A C++ compiler (like GCC, Clang, or MSVC) for the llama-cpp-2 bindings.

A pre-trained LLaMA-style model in GGUF format.

Installation

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/llama_selfmod.git](https://github.com/YOUR_USERNAME/llama_selfmod.git)
cd llama_selfmod


Build the project in release mode:

cargo build --release


The compiled binary will be located at ./target/release/llama_selfmod.

Usage

The engine can be run in two modes.

Interactive Mode

This starts a chatbot-style interface. Simply run the program with a path to your model.

./target/release/llama_selfmod --model /path/to/your/model.gguf


Single-Prompt Mode

Provide a prompt directly via the command line for a one-shot generation.

./target/release/llama_selfmod \
  --model /path/to/your/model.gguf \
  --prompt "Write a short story about a robot who discovers music." \
  --n-predict 512


Command-Line Arguments

Here is a complete list of all available arguments to customize the engine's behavior:

Argument

Flag

Default

Description

Model Path

--model, -m

models/model.gguf

Path to the GGUF model file.

Prompt

--prompt, -p

None

The prompt to use. If omitted, runs in interactive mode.

Context Size

--ctx-size

2048

The context size (in tokens) for the model.

Temperature

--temperature, -t

0.7

The initial sampling temperature.

Top-K

--top-k

40

Top-K sampling parameter.

Top-P

--top-p

0.95

Top-P (nucleus) sampling parameter.

Min-P

--min-p

0.05

Min-P sampling parameter.

Repeat Penalty

--repeat-penalty

1.1

Penalty for repeating tokens.

Prediction Length

--n-predict, -n

256

Max number of tokens to generate.

Confidence Threshold

--confidence-threshold

0.5

Confidence level below which parameter adjustments are triggered.

Aggressive Mode

--aggressive

false

Enables more drastic self-modification and exploration bursts.

Max Retractions

--max-retractions

3

Maximum number of retractions allowed per generation.

GPU Layers

--n-gpu-layers

0

Number of model layers to offload to the GPU.

Threads

--threads

(Auto)

Number of CPU threads to use for generation.

Memory File

--memory-file

selfmod_memory.jsonl

Path to the file for logging generation data.

Quiet Mode

--quiet

false

Suppress non-essential output for a cleaner log.

How to Contribute

Contributions are welcome! Whether it's improving the self-modification logic, adding new features, or fixing bugs, please feel free to open a pull request or an issue.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

This project stands on the shoulders of giants. It would not be possible without the incredible work of:

The llama.cpp team for their powerful C++ inference engine.

The maintainers of the llama-cpp-2 Rust bindings for providing a safe and ergonomic interface.
