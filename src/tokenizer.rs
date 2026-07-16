use crate::error::TokenizerError;
use crate::model::Message;

const CUMULATIVE_ESTIMATE_CONTEXT_BYTES: usize = 256;

fn trailing_context(text: &str) -> &str {
    let mut start = text.len().saturating_sub(CUMULATIVE_ESTIMATE_CONTEXT_BYTES);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn apply_count_delta(
    total: usize,
    previous_context: usize,
    next_context: usize,
) -> Result<usize, TokenizerError> {
    let updated = if next_context >= previous_context {
        total.checked_add(next_context - previous_context)
    } else {
        total.checked_sub(previous_context - next_context)
    };

    updated.ok_or_else(|| {
        TokenizerError::EncodeError(
            "cumulative token estimate exceeded the supported count range".to_string(),
        )
    })
}

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

    /// Estimate token counts after cumulatively appending each addition to `base`.
    ///
    /// The returned vector has one count per addition. The default implementation
    /// retokenizes a bounded trailing context, avoiding repeated full-prefix scans
    /// while preserving each backend's existing `count_tokens` semantics.
    ///
    /// Results are estimates because tokenization can depend on text preceding the
    /// retained context. Backends may override this method when they can provide a
    /// better implementation without changing these caller-visible semantics.
    fn estimate_cumulative_token_counts(
        &self,
        base: &str,
        additions: &[&str],
    ) -> Result<Vec<usize>, TokenizerError> {
        if additions.is_empty() {
            return Ok(Vec::new());
        }

        let mut total = self.count_tokens(base)?;
        let initial_context = trailing_context(base);
        let mut context_count = if initial_context.len() == base.len() {
            total
        } else {
            self.count_tokens(initial_context)?
        };
        let mut context = initial_context.to_string();
        let mut counts = Vec::with_capacity(additions.len());

        for addition in additions {
            if addition.is_empty() {
                counts.push(total);
                continue;
            }

            context.push_str(addition);
            let next_context_count = self.count_tokens(&context)?;
            total = apply_count_delta(total, context_count, next_context_count)?;
            counts.push(total);
            context_count = next_context_count;

            if context.len() > CUMULATIVE_ESTIMATE_CONTEXT_BYTES * 2 {
                context = trailing_context(&context).to_string();
                context_count = self.count_tokens(&context)?;
            }
        }

        Ok(counts)
    }

    /// Count the total tokens for a list of chat messages.
    ///
    /// This accounts for per-message overhead tokens that chat-completion
    /// APIs add (e.g., role markers, separators).
    fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CharacterTokenizer;

    impl AutoTokenizer for CharacterTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
            token_ids
                .iter()
                .map(|id| {
                    char::from_u32(*id).ok_or_else(|| {
                        TokenizerError::DecodeError(format!("invalid character token: {id}"))
                    })
                })
                .collect()
        }

        fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
            Ok(text.chars().count())
        }

        fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
            Ok(messages
                .iter()
                .map(|message| message.content.chars().count())
                .sum())
        }
    }

    #[test]
    fn cumulative_estimates_preserve_order_and_unicode_boundaries() {
        let tokenizer = CharacterTokenizer;
        let base = "世".repeat(200);
        let additions = ["界", "", "🙂", "\ne\u{301}"];

        let counts = tokenizer
            .estimate_cumulative_token_counts(&base, &additions)
            .expect("cumulative estimate should succeed");

        assert_eq!(counts, vec![201, 201, 202, 205]);
    }

    #[test]
    fn empty_additions_do_not_invoke_the_tokenizer() {
        struct FailingTokenizer;

        impl AutoTokenizer for FailingTokenizer {
            fn encode(&self, _: &str) -> Result<Vec<u32>, TokenizerError> {
                Err(TokenizerError::EncodeError("unexpected call".to_string()))
            }

            fn decode(&self, _: &[u32]) -> Result<String, TokenizerError> {
                unreachable!()
            }

            fn count_messages(&self, _: &[Message]) -> Result<usize, TokenizerError> {
                unreachable!()
            }
        }

        let counts = FailingTokenizer
            .estimate_cumulative_token_counts("ignored", &[])
            .expect("empty input should not tokenize");

        assert!(counts.is_empty());
    }
}
