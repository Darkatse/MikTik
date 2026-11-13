use std::path::Path;

use ::kitoken::Kitoken;

use super::kitoken::KitokenBackend;
use crate::error::TokenizerError;

/// Backend for Hugging Face `tokenizer.json` tokenizers.
///
/// The `true` policy keeps added and special tokens active, matching the behavior
/// encoded by Hugging Face tokenizer definitions.
pub type HuggingFaceBackend = KitokenBackend<true>;

impl KitokenBackend<true> {
    /// Load and convert a Hugging Face tokenizer definition from a file.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let tokenizer = Kitoken::from_tokenizers_file(path).map_err(|err| {
            TokenizerError::Load(format!(
                "failed to load Hugging Face tokenizer from '{}': {err}",
                path.display()
            ))
        })?;
        Ok(Self::new(tokenizer))
    }

    /// Load and convert a Hugging Face tokenizer definition from memory.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TokenizerError> {
        let tokenizer = Kitoken::from_tokenizers_slice(data).map_err(|err| {
            TokenizerError::Load(format!(
                "failed to deserialize Hugging Face tokenizer bytes: {err}"
            ))
        })?;
        Ok(Self::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::message::Message;
    use crate::tokenizer::AutoTokenizer;

    use super::*;

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
        let backend =
            HuggingFaceBackend::from_bytes(crate::backend::TEST_TOKENIZER_JSON.as_bytes())
                .expect("backend should load from bytes");

        let ids = backend.encode("hello").expect("encode should succeed");
        assert_eq!(backend.count_tokens("hello").unwrap(), ids.len());
        assert_eq!(backend.decode(&ids).unwrap(), "hello");
    }

    #[test]
    fn from_file_and_count_messages() {
        let path = temp_json_path();
        fs::write(&path, crate::backend::TEST_TOKENIZER_JSON)
            .expect("test tokenizer should be written");

        let backend = HuggingFaceBackend::from_file(&path).expect("backend should load from file");
        let messages = [
            Message::new("user", "hello"),
            Message::new("assistant", "world"),
        ];

        let result = backend.count_messages(&messages);
        let _ = fs::remove_file(path);
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn from_invalid_bytes_returns_load_error() {
        let err = match HuggingFaceBackend::from_bytes(b"{not-valid-json") {
            Ok(_) => panic!("invalid bytes should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, TokenizerError::Load(_)));
    }
}
