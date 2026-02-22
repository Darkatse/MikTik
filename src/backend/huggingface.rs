use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::TokenizerError;
use crate::model::Message;
use crate::tokenizer::AutoTokenizer;

/// Backend for HuggingFace-compatible tokenizers.
///
/// Handles web/fast tokenizer JSON files through the `tokenizers` crate.
pub struct HuggingFaceBackend {
    tokenizer: Tokenizer,
}

impl HuggingFaceBackend {
    /// Load a tokenizer from a tokenizer JSON file path.
    ///
    /// The `tokenizers` crate deserializes a full tokenizer pipeline in JSON format.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let tokenizer = Tokenizer::from_file(path).map_err(|err| {
            TokenizerError::LoadError(format!(
                "failed to load HuggingFace tokenizer from '{}': {err}",
                path.display()
            ))
        })?;
        Ok(Self { tokenizer })
    }

    /// Load a tokenizer from tokenizer JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TokenizerError> {
        let tokenizer = Tokenizer::from_bytes(data).map_err(|err| {
            TokenizerError::LoadError(format!(
                "failed to deserialize HuggingFace tokenizer bytes: {err}"
            ))
        })?;
        Ok(Self { tokenizer })
    }
}

impl AutoTokenizer for HuggingFaceBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| TokenizerError::EncodeError(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(token_ids, false)
            .map_err(|err| TokenizerError::DecodeError(err.to_string()))
    }

    fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        messages.iter().try_fold(0usize, |acc, message| {
            let encoding = self
                .tokenizer
                .encode(message.content.as_str(), false)
                .map_err(|err| TokenizerError::EncodeError(err.to_string()))?;
            Ok(acc + encoding.get_ids().len())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokenizers::models::wordlevel::WordLevel;

    use super::*;

    fn test_tokenizer_json() -> String {
        let vocab = [
            ("<unk>".to_string(), 0),
            ("hello".to_string(), 1),
            ("world".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("wordlevel model should build");
        let tokenizer = Tokenizer::new(model);
        tokenizer.to_string(false).expect("serialize tokenizer")
    }

    fn temp_json_path() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "miktik-hf-test-{}-{nonce}.json",
            std::process::id()
        ))
    }

    #[test]
    fn from_bytes_encode_decode_roundtrip() {
        let json = test_tokenizer_json();
        let backend = HuggingFaceBackend::from_bytes(json.as_bytes())
            .expect("backend should load from bytes");

        let ids = backend.encode("hello").expect("encode should succeed");
        let decoded = backend.decode(&ids).expect("decode should succeed");

        assert_eq!(ids.len(), 1);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn from_file_and_count_messages() {
        let json = test_tokenizer_json();
        let path = temp_json_path();
        fs::write(&path, json.as_bytes()).expect("test tokenizer should be written");

        let backend = HuggingFaceBackend::from_file(&path).expect("backend should load from file");
        let messages = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "world"),
        ];
        let count = backend
            .count_messages(&messages)
            .expect("message counting should succeed");

        let _ = fs::remove_file(path);
        assert_eq!(count, 2);
    }

    #[test]
    fn from_invalid_bytes_returns_load_error() {
        let err = match HuggingFaceBackend::from_bytes(b"{not-valid-json") {
            Ok(_) => panic!("invalid tokenizer json should fail"),
            Err(err) => err,
        };
        match err {
            TokenizerError::LoadError(_) => {}
            other => panic!("unexpected error type: {other}"),
        }
    }
}
