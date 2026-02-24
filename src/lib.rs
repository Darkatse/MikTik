//! # MikTik
//!
//! A unified, multi-backend tokenizer library for LLMs
//!
//! MikTik provides a single interface to count tokens across different
//! model families (OpenAI, Claude, LLaMA, Mistral, etc.) by dispatching
//! to the appropriate backend tokenizer (tiktoken-rs, web tokenizer JSON,
//! or SentencePiece `.model`).
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
pub use tokenizer::AutoTokenizer;
