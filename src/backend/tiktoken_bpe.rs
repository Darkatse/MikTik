use std::path::Path;

use ::kitoken::Kitoken;

use super::kitoken::KitokenBackend;
use crate::error::TokenizerError;

/// Backend for models published as raw tiktoken BPE rank files
/// (`tiktoken.model`), currently Moonshot's Kimi family.
///
/// A tiktoken rank file carries no split pattern or special token table;
/// kitoken infers both for known vocabulary sizes. Every Kimi release ships
/// the same byte-identical vocab (verified from K2 through K3), so the
/// built-in detection covers all of them.
///
/// Special tokens participate in ordinary encoding, matching the Hugging Face
/// backend's contract.
pub type TiktokenBpeBackend = KitokenBackend<true>;

impl KitokenBackend<true> {
    /// Load a tiktoken BPE rank file from disk.
    pub fn from_bpe_file(path: &Path) -> Result<Self, TokenizerError> {
        let tokenizer = Kitoken::from_tiktoken_file(path).map_err(|err| {
            TokenizerError::Load(format!(
                "failed to load tiktoken model from '{}': {err}",
                path.display()
            ))
        })?;
        Ok(Self::new(tokenizer))
    }

    /// Load a tiktoken BPE rank file from memory.
    pub fn from_bpe_bytes(data: &[u8]) -> Result<Self, TokenizerError> {
        let tokenizer = Kitoken::from_tiktoken_slice(data).map_err(|err| {
            TokenizerError::Load(format!("failed to load tiktoken model bytes: {err}"))
        })?;
        Ok(Self::new(tokenizer))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use base64::Engine;

    use super::*;
    use crate::tokenizer::AutoTokenizer;

    /// Builds a synthetic tiktoken payload with exactly 163,584 ranks so
    /// kitoken's vocabulary-size detection applies the Kimi split pattern.
    /// Layout: all single bytes, then a few known merges, then unique
    /// fillers.
    pub(crate) fn test_kimi_tiktoken_payload() -> Vec<u8> {
        const KIMI_VOCAB_LEN: usize = 163_584;

        let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
        tokens.push(b"ab".to_vec());
        tokens.push("你".as_bytes().to_vec());
        tokens.push("好".as_bytes().to_vec());
        tokens.push("你好".as_bytes().to_vec());
        while tokens.len() < KIMI_VOCAB_LEN {
            tokens.push(format!("~filler-{}~", tokens.len()).into_bytes());
        }

        tokens
            .iter()
            .enumerate()
            .map(|(rank, token)| {
                format!(
                    "{} {rank}",
                    base64::engine::general_purpose::STANDARD.encode(token)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes()
    }

    #[test]
    fn kimi_pattern_groups_han_and_applies_merges() {
        let backend = TiktokenBpeBackend::from_bpe_bytes(&test_kimi_tiktoken_payload())
            .expect("backend should load from bytes");

        // The Kimi split pattern groups consecutive Han characters into one
        // pre-token, so the "你好" merge applies; "ab" merges as ASCII.
        assert_eq!(backend.encode("你好").expect("encode"), vec![259]);
        assert_eq!(backend.encode("ab").expect("encode"), vec![256]);
        assert_eq!(backend.count_tokens("你好ab").unwrap(), 2);

        let ids = backend.encode("你好ab").expect("encode");
        assert_eq!(backend.decode(&ids).expect("decode"), "你好ab");
    }

    #[test]
    fn invalid_payload_fails_fast() {
        match TiktokenBpeBackend::from_bpe_bytes(b"\xff\xfe not a tiktoken file") {
            Err(_) => {}
            Ok(_) => panic!("invalid payload should be rejected"),
        }
    }
}
