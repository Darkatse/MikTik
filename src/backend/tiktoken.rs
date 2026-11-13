use tiktoken_rs::{CoreBPE, cl100k_base_singleton, o200k_base_singleton, r50k_base_singleton};

use crate::error::TokenizerError;
use crate::message::Message;
use crate::model::{Encoding, Framing};
use crate::tokenizer::AutoTokenizer;

/// Backend for OpenAI models using the tiktoken BPE tokenizer.
pub struct TiktokenBackend {
    // tiktoken-rs 0.12 owns each parsed encoding in a process-wide singleton.
    // Holding its static reference avoids rebuilding large rank tables per registry.
    bpe: &'static CoreBPE,
    framing: Framing,
}

impl TiktokenBackend {
    pub fn new(encoding: Encoding, framing: Framing) -> Self {
        let bpe = match encoding {
            Encoding::O200kBase => o200k_base_singleton(),
            Encoding::Cl100kBase => cl100k_base_singleton(),
            Encoding::R50kBase => r50k_base_singleton(),
        };
        Self { bpe, framing }
    }
}

impl AutoTokenizer for TiktokenBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        Ok(self.bpe.encode_ordinary(text))
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        Ok(self.bpe.count_ordinary(text))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        self.bpe
            .decode(token_ids)
            .map_err(|err| TokenizerError::Decode(err.to_string()))
    }

    fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        let count = |text: &str| self.bpe.count_with_special_tokens(text);
        Ok(match self.framing {
            Framing::ChatMl {
                per_message,
                per_reply,
            } => {
                per_reply
                    + messages
                        .iter()
                        .map(|message| per_message + count(&message.role) + count(&message.content))
                        .sum::<usize>()
            }
            Framing::ContentOnly => messages.iter().map(|message| count(&message.content)).sum(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;

    fn backend(model: &str) -> TiktokenBackend {
        let model = Model::resolve(model).unwrap();
        let crate::model::Source::Builtin(encoding) = model.source() else {
            panic!("{} is not built in", model.id());
        };
        TiktokenBackend::new(encoding, model.framing())
    }

    #[test]
    fn tiktoken_encode_decode_roundtrip() {
        let backend = backend("gpt-4o");
        let input = "hello token world";

        let encoded = backend.encode(input).expect("encode should succeed");
        let decoded = backend.decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded, input);
    }

    #[test]
    fn legacy_gpt35_counts_more_message_overhead() {
        let legacy = backend("gpt-3.5-turbo-0301");
        let modern = backend("gpt-4");
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
        let backend = backend("o1");
        let count = backend
            .count_tokens("count me")
            .expect("count should succeed");
        assert!(count > 0);
    }

    #[test]
    fn gpt2_uses_r50k_tokenizer() {
        let backend = backend("gpt2");
        let count = backend
            .count_tokens("count me")
            .expect("count should succeed");
        assert!(count > 0);
    }

    #[test]
    fn decode_invalid_token_id_returns_decode_error() {
        let backend = backend("gpt-4o");
        let err = backend
            .decode(&[u32::MAX])
            .expect_err("invalid token id should fail");

        match err {
            TokenizerError::Decode(_) => {}
            other => panic!("unexpected error type: {other}"),
        }
    }
}
