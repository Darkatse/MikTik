use crate::error::TokenizerError;
use crate::model::Message;

/// The central abstraction over all tokenizer backends.
///
/// Every backend (Tiktoken, HuggingFace, etc.) implements this trait,
/// enabling the registry to dispatch calls polymorphically.
///
/// All methods are `&self` — tokenizer instances are stateless once loaded.
pub trait AutoTokenizer: Send + Sync {
    /// Encode text into a sequence of token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;

    /// Decode a sequence of token IDs back into text.
    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>;

    /// Count the number of tokens in a plain text string.
    ///
    /// Default implementation delegates to `encode`, but backends may
    /// override this to avoid the intermediate `Vec` allocation.
    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        self.encode(text).map(|ids| ids.len())
    }

    /// Count the total tokens for a list of chat messages.
    ///
    /// This accounts for per-message overhead tokens that chat-completion
    /// APIs add (e.g., role markers, separators).
    fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError>;
}
