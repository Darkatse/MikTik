//! # MikTik
//!
//! Tokenizers for LLMs behind one API. [kitoken](https://docs.rs/kitoken) reads every format: the
//! OpenAI encodings that ship with the crate, Hugging Face `tokenizer.json` files, SentencePiece
//! models and tiktoken rank files.
//!
//! - [`Model`] is the table of tokenizers MikTik knows. [`Model::resolve`] maps a model name such as
//!   `chatgpt-4o-latest` or `qwen3-max` to a row, or to nothing; the caller decides what to fall
//!   back to.
//! - [`TokenizerRegistry`] holds one [`Tokenizer`] per model and loads it on first use. Built-in
//!   models need nothing; the others need a registered source.
//! - [`Tokenizer`] encodes, decodes and counts tokens, and counts chat messages with the model's
//!   [`Framing`].
//!
//! ```no_run
//! use miktik::{Message, TokenizerRegistry};
//!
//! let registry = TokenizerRegistry::new();
//! let gpt = registry.get("gpt-4o")?;
//! let tokens = gpt.count_tokens("Hello, world!")?;
//!
//! // Needs the `huggingface` feature.
//! registry.register_model_file("claude", "/path/to/claude/tokenizer.json")?;
//! let claude = registry.get("claude-sonnet-4-5")?;
//! let tokens = claude.count_messages(&[Message::new("user", "What is Rust?")])?;
//! # Ok::<(), miktik::TokenizerError>(())
//! ```
//!
//! ## Features
//!
//! - `openai` (default): the built-in OpenAI encodings.
//! - `huggingface`: `tokenizer.json` sources.
//! - `sentencepiece`: SentencePiece model sources.
//! - `tiktoken-bpe`: tiktoken rank file sources.
//! - `full`: all of the above.

#![forbid(unsafe_code)]

mod error;
mod message;
mod model;
mod registry;
mod source;
mod tokenizer;

pub use error::TokenizerError;
pub use message::Message;
pub use model::{Encoding, Format, Framing, Model, Source};
pub use registry::TokenizerRegistry;
pub use tokenizer::Tokenizer;

// Tokenizers are shared across threads.
const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Model>();
    send_sync::<Tokenizer>();
    send_sync::<TokenizerRegistry>();
};
