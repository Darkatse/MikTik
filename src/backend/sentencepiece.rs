use std::path::Path;

use ::kitoken::Kitoken;

use super::kitoken::KitokenBackend;
use crate::error::TokenizerError;

/// Backend for SentencePiece `.model` tokenizers.
///
/// The `false` policy leaves SentencePiece control tokens out of ordinary text
/// encoding and decoding while Kitoken still handles unknown and user-defined pieces.
pub type SentencePieceBackend = KitokenBackend<false>;

impl KitokenBackend<false> {
    /// Load and convert a serialized SentencePiece model from a file.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let tokenizer = Kitoken::from_sentencepiece_file(path).map_err(|err| {
            TokenizerError::Load(format!(
                "failed to load SentencePiece tokenizer from '{}': {err}",
                path.display()
            ))
        })?;
        Ok(Self::new(tokenizer))
    }

    /// Load and convert a serialized SentencePiece model from memory.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TokenizerError> {
        let tokenizer = Kitoken::from_sentencepiece_slice(data).map_err(|err| {
            TokenizerError::Load(format!("failed to parse SentencePiece model bytes: {err}"))
        })?;
        Ok(Self::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::AutoTokenizer;

    use super::*;

    fn bpe_model_bytes() -> Vec<u8> {
        // Build the smallest useful SentencePiece ModelProto in place. Keeping this
        // fixture dependency-free avoids reintroducing a protobuf stack just for tests.
        // Field numbers below follow sentencepiece_model.proto.
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
        // TrainerSpec.model_type = BPE.
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
    fn invalid_source_returns_load_error() {
        let err = match SentencePieceBackend::from_bytes(b"not-a-sentencepiece-model") {
            Ok(_) => panic!("invalid bytes should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, TokenizerError::Load(_)));

        let missing = std::env::temp_dir().join("miktik-missing-sentencepiece.model");
        let _ = std::fs::remove_file(&missing);
        let err = match SentencePieceBackend::from_file(&missing) {
            Ok(_) => panic!("missing file should fail"),
            Err(err) => err,
        };
        assert!(
            matches!(err, TokenizerError::Load(message) if message.contains(&missing.display().to_string()))
        );
    }
}
