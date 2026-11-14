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
//!   [`Framing`]. These results are exact.
//!
//! ```no_run
//! use miktik::{Message, TokenizerRegistry};
//!
//! let registry = TokenizerRegistry::new();
//! let gpt = registry.get("gpt-4o")?;
//! let tokens = gpt.count_tokens("Hello, world!")?;
//! let estimate = gpt.estimate_tokens("Hello, world!");
//!
//! // Needs the `huggingface` feature.
//! registry.register_model_file("claude", "/path/to/claude/tokenizer.json")?;
//! let claude = registry.get("claude-sonnet-4-5")?;
//! let tokens = claude.count_messages(&[Message::new("user", "What is Rust?")])?;
//! # Ok::<(), miktik::TokenizerError>(())
//! ```
//!
//! ## Estimates
//!
//! The `estimate_*` methods of [`Tokenizer`] are approximations several times faster than exact
//! counting, for callers that need a count rather than token ids. An estimate covers the text with
//! the longest pieces of the tokenizer's vocabulary, one token per piece:
//!
//! - Full-width forms are folded to ASCII if the tokenizer normalizes them, and the text is cut
//!   into segments after each newline that is followed by something other than whitespace.
//! - Each segment is covered from its start; every step takes the longest piece the rest starts
//!   with, or one byte where no piece does.
//! - The pieces are the vocabulary and the special tokens that [`Tokenizer::count_tokens`]
//!   recognizes. All of this is derived from the tokenizer's definition, not from the model's name.
//!
//! An estimate depends only on the text, and the estimate of a text is the sum of the estimates of
//! its segments. On the texts MikTik is tested with, every model's estimate of a whole book is
//! within 2% of the exact count, and 95% of 1 KiB pieces are within 5%.
//!
//! Because segments add up, a registry made with [`TokenizerRegistry::with_segment_cache`] keeps
//! the estimates of recent segments, and text estimated again, such as a chat that grows at its
//! end, costs only its new segments. The cache changes no result.
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
mod estimate;
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
