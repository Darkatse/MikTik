//! # MikTik
//!
//! A unified, multi-backend tokenizer library for LLMs
//!
//! MikTik provides a single interface to count tokens across different
//! model families (OpenAI, Claude, LLaMA, Mistral, etc.) by dispatching
//! to `tiktoken-rs` for OpenAI models and Kitoken for Hugging Face JSON or
//! SentencePiece `.model` tokenizers.
//!
//! ## Quick Start
//!
//! ```no_run
//! use miktik::{TokenizerRegistry, Message};
//!
//! let registry = TokenizerRegistry::new();
//!
//! // Count tokens in plain text
//! let count = registry.count_tokens("gpt-4o", "Hello, world!").unwrap();
//! let cumulative = registry
//!     .estimate_cumulative_token_counts(
//!         "gpt-4o",
//!         "Hello",
//!         &[", world", "!"],
//!         Default::default(),
//!     )
//!     .unwrap();
//! assert_eq!(cumulative.len(), 2);
//!
//! // Count tokens in a chat conversation
//! let messages = vec![
//!     Message::new("user", "What is Rust?"),
//!     Message::new("assistant", "Rust is a systems programming language."),
//! ];
//! // Requires `huggingface` feature.
//! registry
//!     .register_model_file("claude", "/path/to/claude-tokenizer.json")
//!     .unwrap();
//! let count = registry.count_messages("claude", &messages).unwrap();
//! ```

pub mod backend;
pub mod error;
pub mod model;
pub mod registry;
pub mod tokenizer;

pub use error::TokenizerError;
pub use model::Message;
pub use registry::TokenizerRegistry;
pub use tokenizer::{AutoTokenizer, CumulativeEstimateOptions};
