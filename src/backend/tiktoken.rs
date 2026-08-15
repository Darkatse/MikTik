use tiktoken_rs::{CoreBPE, bpe_for_model};

use crate::error::TokenizerError;
use crate::model::Message;
use crate::tokenizer::AutoTokenizer;

/// Backend for OpenAI models using the tiktoken BPE tokenizer.
pub struct TiktokenBackend {
    // tiktoken-rs 0.12 owns each parsed encoding in a process-wide singleton.
    // Holding its static reference avoids rebuilding large rank tables per registry.
    bpe: &'static CoreBPE,
    /// Per-message token overhead (e.g., 3 for most GPT models, 4 for gpt-3.5-turbo-0301).
    tokens_per_message: usize,
    /// Extra tokens added at the end of the full prompt (typically 3 for "assistant" priming).
    tokens_per_reply: usize,
}

impl TiktokenBackend {
    /// Create a backend from a tiktoken model name (e.g., `"gpt-4o"`, `"gpt-3.5-turbo"`).
    pub fn from_model(model: &str) -> Result<Self, TokenizerError> {
        let bpe = bpe_for_model(model).map_err(|err| {
            TokenizerError::LoadError(format!("failed to load tiktoken model '{model}': {err}"))
        })?;

        // The 0301 snapshot used one extra ChatML framing token per message.
        let tokens_per_message = if model.starts_with("gpt-3.5-turbo-0301") {
            4
        } else {
            3
        };

        Ok(Self {
            bpe,
            tokens_per_message,
            tokens_per_reply: 3,
        })
    }
}

impl AutoTokenizer for TiktokenBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        Ok(self.bpe.encode_ordinary(text))
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        // Use the counting API so future tiktoken-rs releases can optimize it
        // independently without changing MikTik's backend contract.
        Ok(self.bpe.count_ordinary(text))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        self.bpe
            .decode(token_ids)
            .map_err(|err| TokenizerError::DecodeError(err.to_string()))
    }

    fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        // ChatML accounting is reply priming plus framing and encoded fields per message.
        let mut total = self.tokens_per_reply;
        for message in messages {
            total += self.tokens_per_message;
            total += self.bpe.count_with_special_tokens(&message.role);
            total += self.bpe.count_with_special_tokens(&message.content);
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiktoken_encode_decode_roundtrip() {
        let backend = TiktokenBackend::from_model("gpt-4o").expect("backend should load");
        let input = "hello token world";

        let encoded = backend.encode(input).expect("encode should succeed");
        let decoded = backend.decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded, input);
    }

    #[test]
    fn legacy_gpt35_counts_more_message_overhead() {
        let legacy = TiktokenBackend::from_model("gpt-3.5-turbo-0301").expect("legacy loads");
        let modern = TiktokenBackend::from_model("gpt-4").expect("modern loads");
        let messages = vec![
            Message::new("system", "You are a helpful assistant."),
            Message::new("user", "Explain ownership in Rust."),
        ];

        let legacy_count = legacy.count_messages(&messages).expect("legacy count");
        let modern_count = modern.count_messages(&messages).expect("modern count");

        assert_eq!(legacy_count, modern_count + messages.len());
    }

    #[test]
    fn o1_alias_uses_o200k_compatible_model() {
        let backend = TiktokenBackend::from_model("o1").expect("o1 alias should load");
        let count = backend
            .count_tokens("count me")
            .expect("count should succeed");
        assert!(count > 0);
    }

    #[test]
    fn gpt2_uses_r50k_tokenizer() {
        let backend = TiktokenBackend::from_model("gpt2").expect("gpt2 should load");
        let count = backend
            .count_tokens("count me")
            .expect("count should succeed");
        assert!(count > 0);
    }

    #[test]
    fn decode_invalid_token_id_returns_decode_error() {
        let backend = TiktokenBackend::from_model("gpt-4o").expect("backend should load");
        let err = backend
            .decode(&[u32::MAX])
            .expect_err("invalid token id should fail");

        match err {
            TokenizerError::DecodeError(_) => {}
            other => panic!("unexpected error type: {other}"),
        }
    }
}
