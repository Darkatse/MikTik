use std::path::Path;

use sentencepiece_rs::SentencePieceProcessor;

use crate::error::TokenizerError;
use crate::model::Message;
use crate::tokenizer::AutoTokenizer;

/// Backend for SentencePiece `.model` tokenizers.
///
/// Supports the model type and normalization rules embedded in each model.
pub struct SentencePieceBackend {
    tokenizer: SentencePieceProcessor,
}

impl SentencePieceBackend {
    /// Load a SentencePiece tokenizer from a `.model` file.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let data = std::fs::read(path).map_err(|err| {
            TokenizerError::LoadError(format!(
                "failed to read SentencePiece tokenizer from '{}': {err}",
                path.display()
            ))
        })?;
        Self::from_bytes(&data)
    }

    /// Load a SentencePiece tokenizer from serialized protobuf bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TokenizerError> {
        let tokenizer = SentencePieceProcessor::from_serialized_model(data).map_err(|err| {
            TokenizerError::LoadError(format!("failed to parse SentencePiece model bytes: {err}"))
        })?;
        Ok(Self { tokenizer })
    }

    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.tokenizer
            .encode_to_ids(text)
            .map_err(|err| TokenizerError::EncodeError(err.to_string()))?
            .into_iter()
            .map(|id| {
                u32::try_from(id).map_err(|_| {
                    TokenizerError::EncodeError(format!(
                        "SentencePiece token id {id} exceeds the supported range"
                    ))
                })
            })
            .collect()
    }
}

impl AutoTokenizer for SentencePieceBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_ids(text)
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        self.tokenizer
            .encode_to_ids(text)
            .map(|ids| ids.len())
            .map_err(|err| TokenizerError::EncodeError(err.to_string()))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let ids = token_ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        self.tokenizer
            .decode_ids(&ids)
            .map_err(|err| TokenizerError::DecodeError(err.to_string()))
    }

    fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        messages.iter().try_fold(0usize, |acc, message| {
            let count = self.count_tokens(message.content.as_str())?;
            Ok(acc + count)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bpe_model_bytes() -> Vec<u8> {
        let pieces = [
            piece("<unk>", 0.0, 2),
            piece("<s>", 0.0, 3),
            piece("</s>", 0.0, 3),
            piece("▁", 0.0, 1),
            piece("a", 0.0, 1),
            piece("b", 0.0, 1),
            piece("ab", 10.0, 1),
        ];
        let mut model = Vec::new();
        for value in pieces {
            message_field(&mut model, 1, &value);
        }

        let mut trainer = Vec::new();
        varint_field(&mut trainer, 3, 2);
        message_field(&mut model, 2, &trainer);
        model
    }

    fn piece(text: &str, score: f32, kind: u64) -> Vec<u8> {
        let mut value = Vec::new();
        message_field(&mut value, 1, text.as_bytes());
        key(&mut value, 2, 5);
        value.extend_from_slice(&score.to_bits().to_le_bytes());
        varint_field(&mut value, 3, kind);
        value
    }

    fn message_field(output: &mut Vec<u8>, field: u32, value: &[u8]) {
        key(output, field, 2);
        varint(output, value.len() as u64);
        output.extend_from_slice(value);
    }

    fn varint_field(output: &mut Vec<u8>, field: u32, value: u64) {
        key(output, field, 0);
        varint(output, value);
    }

    fn key(output: &mut Vec<u8>, field: u32, wire_type: u8) {
        varint(output, ((field as u64) << 3) | u64::from(wire_type));
    }

    fn varint(output: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            output.push((value as u8) | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    #[test]
    fn bpe_model_uses_bpe_merges() {
        let tokenizer =
            SentencePieceBackend::from_bytes(&bpe_model_bytes()).expect("BPE model should load");

        let ids = tokenizer.encode("ab").expect("text should encode");
        assert_eq!(ids, [3, 6]);
        assert_eq!(tokenizer.count_tokens("ab").unwrap(), 2);
        assert_eq!(tokenizer.decode(&ids).unwrap(), "ab");
    }

    #[test]
    fn from_invalid_bytes_returns_load_error() {
        let err = match SentencePieceBackend::from_bytes(b"not-a-sentencepiece-model") {
            Ok(_) => panic!("invalid bytes should fail"),
            Err(err) => err,
        };
        match err {
            TokenizerError::LoadError(_) => {}
            other => panic!("unexpected error type: {other}"),
        }
    }

    #[test]
    fn from_missing_file_returns_load_error() {
        let missing = std::env::temp_dir().join("miktik-missing-sentencepiece.model");
        let _ = std::fs::remove_file(&missing);

        let err = match SentencePieceBackend::from_file(&missing) {
            Ok(_) => panic!("missing file should fail"),
            Err(err) => err,
        };
        match err {
            TokenizerError::LoadError(message) => {
                assert!(message.contains(&missing.display().to_string()));
            }
            other => panic!("unexpected error type: {other}"),
        }
    }
}
