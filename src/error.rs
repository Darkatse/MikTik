/// Errors from MikTik, one variant for each thing a caller does next.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TokenizerError {
    /// No model matches the name. Use another name.
    #[error("unknown model '{0}'")]
    UnknownModel(String),

    /// The model needs a registered source and has none. Register one and retry.
    #[error("model '{0}' has no registered tokenizer source")]
    NotRegistered(&'static str),

    /// A source could not be read, recognized or converted, or its format needs a feature that is
    /// not enabled.
    #[error("failed to load tokenizer: {0}")]
    Load(String),

    /// An artifact is of another version or another model, is truncated, or holds a definition
    /// that cannot be used. Build it again from the model's source.
    #[error("unusable tokenizer artifact: {0}")]
    Artifact(String),

    #[error("encoding failed: {0}")]
    Encode(String),

    /// Decoding failed, or the decoded bytes are not UTF-8.
    #[error("decoding failed: {0}")]
    Decode(String),
}
