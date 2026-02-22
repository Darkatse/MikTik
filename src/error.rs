use thiserror::Error;

/// Unified error type for all tokenizer operations.
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("failed to load tokenizer: {0}")]
    LoadError(String),

    #[error("encoding failed: {0}")]
    EncodeError(String),

    #[error("decoding failed: {0}")]
    DecodeError(String),
}
