use std::path::Path;

use sentencepiece_model::SentencePieceModel;
use tokenizers::Tokenizer;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::precompiled::Precompiled;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

use crate::error::TokenizerError;
use crate::model::Message;
use crate::tokenizer::AutoTokenizer;

/// Backend for SentencePiece `.model` tokenizers.
///
/// This backend parses SentencePiece protobuf models and materializes a
/// `tokenizers::Tokenizer` with a Unigram model and SentencePiece-compatible
/// metaspace preprocessing.
pub struct SentencePieceBackend {
    tokenizer: Tokenizer,
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
        let sp_model = SentencePieceModel::from_slice(data).map_err(|err| {
            TokenizerError::LoadError(format!("failed to parse SentencePiece model bytes: {err}"))
        })?;
        let tokenizer = build_tokenizer(&sp_model)?;
        Ok(Self { tokenizer })
    }
}

fn build_tokenizer(sp_model: &SentencePieceModel) -> Result<Tokenizer, TokenizerError> {
    let vocab = sp_model
        .pieces()
        .iter()
        .map(|piece| {
            (
                piece.piece.clone().unwrap_or_default(),
                f64::from(piece.score.unwrap_or(0.0)),
            )
        })
        .collect::<Vec<_>>();

    let trainer = sp_model.trainer();
    let unk_id = trainer
        .and_then(|spec| spec.unk_id)
        .and_then(|id| (id >= 0).then_some(id as usize));
    let byte_fallback = trainer.and_then(|spec| spec.byte_fallback).unwrap_or(false);

    let model = Unigram::from(vocab, unk_id, byte_fallback).map_err(|err| {
        TokenizerError::LoadError(format!(
            "failed to build unigram model from SentencePiece: {err}"
        ))
    })?;

    let mut tokenizer = Tokenizer::new(model);

    if let Some(precompiled_map) = sp_model
        .normalizer()
        .and_then(|spec| spec.precompiled_charsmap.as_ref())
        .filter(|bytes| !bytes.is_empty())
    {
        let precompiled = Precompiled::from(precompiled_map).map_err(|err| {
            TokenizerError::LoadError(format!(
                "failed to load SentencePiece precompiled normalizer: {err}"
            ))
        })?;
        tokenizer.with_normalizer(Some(precompiled));
    }

    let prepend_scheme = match sp_model
        .normalizer()
        .and_then(|spec| spec.add_dummy_prefix)
        .unwrap_or(true)
    {
        true => PrependScheme::Always,
        false => PrependScheme::Never,
    };
    let metaspace = Metaspace::new('\u{2581}', prepend_scheme, true);
    tokenizer.with_pre_tokenizer(Some(metaspace.clone()));
    tokenizer.with_decoder(Some(metaspace));

    Ok(tokenizer)
}

impl AutoTokenizer for SentencePieceBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode_fast(text, false)
            .map_err(|err| TokenizerError::EncodeError(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode_fast(text, false)
            .map_err(|err| TokenizerError::EncodeError(err.to_string()))?;
        Ok(encoding.get_ids().len())
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
                .encode_fast(message.content.as_str(), false)
                .map_err(|err| TokenizerError::EncodeError(err.to_string()))?;
            Ok(acc + encoding.get_ids().len())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
