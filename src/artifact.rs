//! Artifacts: a tokenizer's definition and estimator, as [`Tokenizer::to_bytes`] writes them, read
//! back without converting a source or deriving the estimator again.
//!
//! The layout, with integers in little-endian:
//!
//! | Field | Bytes |
//! | --- | --- |
//! | Magic | `MIKTIK\0A` |
//! | Version | `u32`, [`ARTIFACT_VERSION`] |
//! | Model id | `u16` length, then UTF-8 |
//! | Definition | `u64` length, then kitoken's serialized definition |
//! | Estimator | `u64` length, then the estimator's flags and trie |
//!
//! [`Tokenizer::to_bytes`]: crate::Tokenizer::to_bytes

use crate::estimate::Estimator;
use crate::model::Model;

/// The version of the artifact format, which [`Tokenizer::to_bytes`](crate::Tokenizer::to_bytes)
/// writes and registration accepts. It changes whenever an artifact would no longer mean the same
/// thing, including with every upgrade of kitoken, whose serialized definition an artifact holds.
/// An artifact of another version fails with
/// [`TokenizerError::Artifact`](crate::TokenizerError::Artifact) and is rebuilt from its source.
pub const ARTIFACT_VERSION: u32 = 1;

const MAGIC: &[u8] = b"MIKTIK\0A";

/// Whether `bytes` start as an artifact does.
pub(crate) fn is_artifact(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

/// An artifact's sections, still to be read.
pub(crate) struct Sections<'a> {
    pub(crate) definition: &'a [u8],
    pub(crate) estimator: &'a [u8],
}

pub(crate) fn write(model: &Model, definition: &[u8], estimator: &Estimator) -> Vec<u8> {
    let mut estimator_bytes = Vec::new();
    estimator.write(&mut estimator_bytes);
    let id = model.id().as_bytes();
    let mut artifact = Vec::with_capacity(
        MAGIC.len() + 4 + 2 + id.len() + 8 + definition.len() + 8 + estimator_bytes.len(),
    );
    artifact.extend_from_slice(MAGIC);
    artifact.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
    // Model ids are short literals of the model table.
    artifact.extend_from_slice(&(id.len() as u16).to_le_bytes());
    artifact.extend_from_slice(id);
    for section in [definition, &estimator_bytes] {
        artifact.extend_from_slice(&(section.len() as u64).to_le_bytes());
        artifact.extend_from_slice(section);
    }
    artifact
}

/// The sections of `model`'s artifact. Fails unless the artifact is of this version and this model
/// and its lengths add up to its size.
pub(crate) fn read<'a>(model: &Model, artifact: &'a [u8]) -> Result<Sections<'a>, String> {
    let mut rest = Rest(artifact);
    if rest.take(MAGIC.len())? != MAGIC {
        return Err("not an artifact".to_string());
    }
    let version = u32::from_le_bytes(rest.array()?);
    if version != ARTIFACT_VERSION {
        return Err(format!(
            "artifact version {version}; this MikTik reads version {ARTIFACT_VERSION}"
        ));
    }
    let id_len = u16::from_le_bytes(rest.array()?);
    let id = rest.take(id_len.into())?;
    if id != model.id().as_bytes() {
        return Err(format!(
            "an artifact of model '{}'",
            String::from_utf8_lossy(id)
        ));
    }
    let definition = rest.section()?;
    let estimator = rest.section()?;
    if !rest.0.is_empty() {
        return Err(format!(
            "{} bytes follow the end of the artifact",
            rest.0.len()
        ));
    }
    Ok(Sections {
        definition,
        estimator,
    })
}

/// The bytes of an artifact not read yet.
struct Rest<'a>(&'a [u8]);

impl<'a> Rest<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        if len > self.0.len() {
            return Err("the artifact is truncated".to_string());
        }
        let (taken, rest) = self.0.split_at(len);
        self.0 = rest;
        Ok(taken)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let bytes = self.take(N)?;
        Ok(std::array::from_fn(|at| bytes[at]))
    }

    fn section(&mut self) -> Result<&'a [u8], String> {
        let len = u64::from_le_bytes(self.array()?);
        self.take(usize::try_from(len).unwrap_or(usize::MAX))
    }
}
