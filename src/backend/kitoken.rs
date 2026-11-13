use ::kitoken::Kitoken;

use crate::error::TokenizerError;
use crate::message::Message;
use crate::tokenizer::AutoTokenizer;

/// Shared runtime adapter for tokenizer definitions converted into Kitoken.
///
/// `SPECIAL_TOKENS` is part of the format contract rather than a caller option:
/// Hugging Face added tokens participate in ordinary encoding, while SentencePiece
/// control tokens do not. Keeping that choice in the backend type prevents calls for
/// one loaded model from accidentally changing semantics between operations.
pub struct KitokenBackend<const SPECIAL_TOKENS: bool> {
    tokenizer: Kitoken,
}

impl<const SPECIAL_TOKENS: bool> KitokenBackend<SPECIAL_TOKENS> {
    pub(super) fn new(tokenizer: Kitoken) -> Self {
        Self { tokenizer }
    }
}

impl<const SPECIAL_TOKENS: bool> AutoTokenizer for KitokenBackend<SPECIAL_TOKENS> {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.tokenizer
            .encode(text, SPECIAL_TOKENS)
            .map_err(|err| TokenizerError::Encode(err.to_string()))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let bytes = self
            .tokenizer
            .decode(token_ids, SPECIAL_TOKENS)
            .map_err(|err| TokenizerError::Decode(err.to_string()))?;

        // Kitoken preserves arbitrary token bytes; MikTik deliberately exposes UTF-8 text.
        // A partial byte token therefore remains an explicit decode error for the caller.
        String::from_utf8(bytes).map_err(|err| TokenizerError::Decode(err.to_string()))
    }

    fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        // Non-OpenAI backends have no shared chat-template contract. Preserve MikTik's
        // established behavior by counting message content and leaving prompt rendering
        // to the caller that owns the target model's template.
        messages.iter().try_fold(0, |total, message| {
            self.count_tokens(&message.content)
                .map(|count| total + count)
        })
    }
}
