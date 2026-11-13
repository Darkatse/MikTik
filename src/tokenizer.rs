use crate::error::TokenizerError;
use crate::message::Message;

const DEFAULT_CUMULATIVE_ESTIMATE_CONTEXT_BYTES: usize = 256;

fn trailing_context(text: &str, context_bytes: usize) -> &str {
    let mut start = text.len().saturating_sub(context_bytes);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// Controls the speed and stopping behavior of cumulative token estimates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CumulativeEstimateOptions {
    /// Maximum trailing UTF-8 bytes retained across additions.
    pub context_bytes: usize,
    /// Stops encoding after the first estimate at or above this token count.
    /// Remaining result slots are filled with the terminal estimate.
    pub stop_at: Option<usize>,
}

impl Default for CumulativeEstimateOptions {
    fn default() -> Self {
        Self {
            context_bytes: DEFAULT_CUMULATIVE_ESTIMATE_CONTEXT_BYTES,
            stop_at: None,
        }
    }
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
        TokenizerError::Encode(
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
        options: CumulativeEstimateOptions,
    ) -> Result<Vec<usize>, TokenizerError> {
        if additions.is_empty() {
            return Ok(Vec::new());
        }

        let mut total = self.count_tokens(base)?;
        let initial_context = trailing_context(base, options.context_bytes);
        let mut context_count = if initial_context.len() == base.len() {
            total
        } else {
            self.count_tokens(initial_context)?
        };
        let mut context = initial_context.to_string();
        let mut counts = Vec::with_capacity(additions.len());

        for addition in additions {
            if !addition.is_empty() {
                context.push_str(addition);
                let next_context_count = self.count_tokens(&context)?;
                total = apply_count_delta(total, context_count, next_context_count)?;
                context_count = next_context_count;
            }

            counts.push(total);
            if options.stop_at.is_some_and(|limit| total >= limit) {
                counts.resize(additions.len(), total);
                break;
            }

            if context.len() > options.context_bytes.saturating_mul(2) {
                context = trailing_context(&context, options.context_bytes).to_string();
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
    use std::sync::Mutex;

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
                        TokenizerError::Decode(format!("invalid character token: {id}"))
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
            .estimate_cumulative_token_counts(
                &base,
                &additions,
                CumulativeEstimateOptions::default(),
            )
            .expect("cumulative estimate should succeed");

        assert_eq!(counts, vec![201, 201, 202, 205]);
    }

    #[test]
    fn empty_additions_do_not_invoke_the_tokenizer() {
        struct FailingTokenizer;

        impl AutoTokenizer for FailingTokenizer {
            fn encode(&self, _: &str) -> Result<Vec<u32>, TokenizerError> {
                Err(TokenizerError::Encode("unexpected call".to_string()))
            }

            fn decode(&self, _: &[u32]) -> Result<String, TokenizerError> {
                unreachable!()
            }

            fn count_messages(&self, _: &[Message]) -> Result<usize, TokenizerError> {
                unreachable!()
            }
        }

        let counts = FailingTokenizer
            .estimate_cumulative_token_counts("ignored", &[], CumulativeEstimateOptions::default())
            .expect("empty input should not tokenize");

        assert!(counts.is_empty());
    }

    #[test]
    fn cumulative_options_bound_context_and_stop_future_encoding() {
        struct RecordingTokenizer(Mutex<Vec<String>>);

        impl AutoTokenizer for RecordingTokenizer {
            fn encode(&self, _: &str) -> Result<Vec<u32>, TokenizerError> {
                unreachable!()
            }

            fn decode(&self, _: &[u32]) -> Result<String, TokenizerError> {
                unreachable!()
            }

            fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
                self.0.lock().unwrap().push(text.to_string());
                Ok(text.chars().count())
            }

            fn count_messages(&self, _: &[Message]) -> Result<usize, TokenizerError> {
                unreachable!()
            }
        }

        assert_eq!(
            CumulativeEstimateOptions::default(),
            CumulativeEstimateOptions {
                context_bytes: 256,
                stop_at: None,
            }
        );

        let tokenizer = RecordingTokenizer(Mutex::new(Vec::new()));
        let counts = tokenizer
            .estimate_cumulative_token_counts(
                "abcdef",
                &["g", "h", "i"],
                CumulativeEstimateOptions {
                    context_bytes: 2,
                    stop_at: Some(8),
                },
            )
            .unwrap();

        assert_eq!(counts, vec![7, 8, 8]);
        assert_eq!(
            *tokenizer.0.lock().unwrap(),
            ["abcdef", "ef", "efg", "efgh"]
        );
    }
}
