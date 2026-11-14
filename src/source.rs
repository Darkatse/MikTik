//! Tokenizer sources: what a caller registers, how its format is recognized, and how each format
//! becomes a kitoken definition.

use std::borrow::Cow;
use std::fmt;
use std::path::PathBuf;

use kitoken::{Definition, Metadata};

use crate::artifact;
use crate::error::TokenizerError;
use crate::model::{Encoding, Format, Model, Source};

/// A registered source: a file read when the model loads, or bytes held in memory.
pub(crate) enum SourceData {
    File(PathBuf),
    Bytes(Vec<u8>),
}

impl SourceData {
    fn read(&self) -> Result<Cow<'_, [u8]>, String> {
        match self {
            Self::File(path) => std::fs::read(path)
                .map(Cow::Owned)
                .map_err(|err| format!("cannot read it: {err}")),
            Self::Bytes(bytes) => Ok(Cow::Borrowed(bytes)),
        }
    }
}

impl fmt::Display for SourceData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => write!(f, "'{}'", path.display()),
            Self::Bytes(_) => f.write_str("registered bytes"),
        }
    }
}

/// What went wrong with a model's source, naming the model and the source: the registered source
/// when there is one, otherwise the built-in encoding.
pub(crate) fn describe(
    model: &Model,
    source: Option<&SourceData>,
    detail: impl fmt::Display,
) -> String {
    let id = model.id();
    match (source, model.source()) {
        (Some(source), _) => format!("model '{id}', source {source}: {detail}"),
        (None, Source::Builtin(encoding)) => {
            format!("model '{id}', built-in {}: {detail}", encoding.name())
        }
        (None, Source::Registered(_)) => format!("model '{id}': {detail}"),
    }
}

/// What a model's source holds.
pub(crate) enum Content<'a> {
    /// A definition converted from a source format or from the built-in encoding.
    Definition(Box<Definition>),
    /// An artifact, still to be read.
    Artifact(Cow<'a, [u8]>),
}

/// Reads a model's source: the registered source when there is one, otherwise the built-in
/// encoding. Every model accepts an artifact; a source format must be one the model accepts.
pub(crate) fn read<'a>(
    model: &Model,
    source: Option<&'a SourceData>,
) -> Result<Content<'a>, TokenizerError> {
    let Some(source) = source else {
        return match model.source() {
            Source::Builtin(encoding) => builtin(encoding)
                .map(|definition| Content::Definition(Box::new(definition)))
                .map_err(|detail| TokenizerError::Load(describe(model, None, detail))),
            Source::Registered(_) => Err(TokenizerError::NotRegistered(model.id())),
        };
    };
    let error = |detail: String| TokenizerError::Load(describe(model, Some(source), detail));

    let bytes = source.read().map_err(error)?;
    if artifact::is_artifact(&bytes) {
        return Ok(Content::Artifact(bytes));
    }
    let format = sniff(&bytes).ok_or_else(|| {
        error(format!(
            "not a MikTik artifact, {}, {} or {}",
            Format::TokenizerJson,
            Format::SentencePiece,
            Format::TiktokenRanks
        ))
    })?;
    match model.source() {
        Source::Builtin(_) => {
            return Err(error(format!(
                "is a {format}; the model is built in and takes only a MikTik artifact"
            )));
        }
        Source::Registered(accepted) if !accepted.contains(&format) => {
            let accepted = accepted.iter().map(Format::to_string).collect::<Vec<_>>();
            return Err(error(format!(
                "is a {format}; the model accepts a MikTik artifact or {}",
                accepted.join(" or ")
            )));
        }
        Source::Registered(_) => {}
    }
    convert(format, &bytes)
        .map(|definition| Content::Definition(Box::new(definition)))
        .map_err(error)
}

/// Recognizes a source format from its first bytes. Each check accepts only a beginning that the
/// format's converter accepts too, so a recognized source can fail only on its content.
fn sniff(bytes: &[u8]) -> Option<Format> {
    // The whitespace JSON allows before a value.
    let first_non_space = bytes
        .iter()
        .find(|byte| !matches!(byte, b' ' | b'\t' | b'\n' | b'\r'));
    if first_non_space == Some(&b'{') {
        return Some(Format::TokenizerJson);
    }
    // A SentencePiece ModelProto starts with its pieces: field 1, length-delimited, tag 0x0A.
    if bytes.first() == Some(&0x0A) {
        return Some(Format::SentencePiece);
    }
    // A tiktoken rank file starts with a line `<base64 token> <decimal rank>`.
    let line = bytes.split(|&byte| byte == b'\n').next()?;
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let space = line.iter().position(|&byte| byte == b' ')?;
    let (token, rank) = (&line[..space], &line[space + 1..]);
    let is_base64 = |byte: &u8| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=');
    (!token.is_empty()
        && token.iter().all(is_base64)
        && !rank.is_empty()
        && rank.iter().all(u8::is_ascii_digit))
    .then_some(Format::TiktokenRanks)
}

#[cfg_attr(
    not(any(
        feature = "huggingface",
        feature = "sentencepiece",
        feature = "tiktoken-bpe"
    )),
    allow(unused_variables)
)]
fn convert(format: Format, bytes: &[u8]) -> Result<Definition, String> {
    match format {
        #[cfg(feature = "huggingface")]
        Format::TokenizerJson => {
            Definition::from_tokenizers_slice(bytes).map_err(|err| err.to_string())
        }
        #[cfg(not(feature = "huggingface"))]
        Format::TokenizerJson => Err(format!(
            "reading a {format} needs the 'huggingface' feature"
        )),
        #[cfg(feature = "sentencepiece")]
        Format::SentencePiece => {
            Definition::from_sentencepiece_slice(bytes).map_err(|err| err.to_string())
        }
        #[cfg(not(feature = "sentencepiece"))]
        Format::SentencePiece => Err(format!(
            "reading a {format} needs the 'sentencepiece' feature"
        )),
        #[cfg(feature = "tiktoken-bpe")]
        Format::TiktokenRanks => {
            Definition::from_tiktoken_slice(bytes).map_err(|err| err.to_string())
        }
        #[cfg(not(feature = "tiktoken-bpe"))]
        Format::TiktokenRanks => Err(format!(
            "reading a {format} needs the 'tiktoken-bpe' feature"
        )),
    }
}

#[cfg(feature = "openai")]
fn builtin(encoding: Encoding) -> Result<Definition, String> {
    openai::definition(encoding)
}

#[cfg(not(feature = "openai"))]
fn builtin(_: Encoding) -> Result<Definition, String> {
    Err("built-in encodings need the 'openai' feature, or register a MikTik artifact".to_string())
}

/// Whether control tokens such as `<|endoftext|>` are recognized. The source format decides, and
/// kitoken records the format of a definition in its metadata, which artifacts keep.
#[derive(Clone, Copy)]
pub(crate) struct SpecialTokens {
    /// In `encode` and `count_tokens`.
    pub(crate) in_text: bool,
    /// In message roles and content; `decode` writes out the same ones.
    pub(crate) in_messages: bool,
}

impl SpecialTokens {
    pub(crate) fn of(model: &Model, meta: &Metadata) -> Self {
        let (in_text, in_messages) = match model.source() {
            // tiktoken's contract: `encode` reads ordinary text, while messages and decoding know
            // the special tokens.
            Source::Builtin(_) => (false, true),
            // SentencePiece control tokens are never text. User-defined pieces are kitoken priority
            // tokens, which match either way.
            Source::Registered(_) if meta.source == "sentencepiece" => (false, false),
            Source::Registered(_) => (true, true),
        };
        Self {
            in_text,
            in_messages,
        }
    }
}

#[cfg(feature = "openai")]
mod openai {
    use kitoken::{Definition, Regex, SpecialToken, SpecialTokenKind, Split, SplitBehavior};

    use crate::model::Encoding;

    /// The encoding as tiktoken defines it. A rank file carries only the vocabulary, so the split
    /// pattern and the special tokens are set here.
    pub(super) fn definition(encoding: Encoding) -> Result<Definition, String> {
        let (ranks, pattern, specials): (&[u8], &str, &[(&str, u32)]) = match encoding {
            Encoding::O200kBase => (
                include_bytes!("../assets/openai/o200k_base.tiktoken"),
                O200K_PATTERN,
                &[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)],
            ),
            Encoding::Cl100kBase => (
                include_bytes!("../assets/openai/cl100k_base.tiktoken"),
                CL100K_PATTERN,
                &[
                    ("<|endoftext|>", 100257),
                    ("<|fim_prefix|>", 100258),
                    ("<|fim_middle|>", 100259),
                    ("<|fim_suffix|>", 100260),
                    ("<|endofprompt|>", 100276),
                ],
            ),
            Encoding::R50kBase => (
                include_bytes!("../assets/openai/r50k_base.tiktoken"),
                R50K_PATTERN,
                &[("<|endoftext|>", 50256)],
            ),
        };

        let mut definition =
            Definition::from_tiktoken_slice(ranks).map_err(|err| err.to_string())?;
        definition.config.split = vec![Split::Pattern {
            pattern: Regex::new(pattern).map_err(|err| err.0)?.into(),
            behavior: SplitBehavior::Isolate,
        }];
        definition.specials = specials
            .iter()
            .enumerate()
            .map(|(priority, &(text, id))| SpecialToken {
                id,
                bytes: text.as_bytes().to_vec(),
                kind: SpecialTokenKind::Control,
                ident: None,
                score: priority as f32,
                // kitoken splits text at extracted special tokens even when it does not recognize
                // them. tiktoken's ordinary encoding does not split there, so these are matched only
                // where they are recognized.
                extract: false,
            })
            .collect();
        Ok(definition)
    }

    const O200K_PATTERN: &str = concat!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|\p{N}{1,3}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"|\s*[\r\n]+",
        r"|\s+(?!\S)",
        r"|\s+",
    );
    const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
    const R50K_PATTERN: &str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_are_recognized_from_their_first_bytes() {
        for (bytes, format) in [
            (&b"{\"model\": {}}"[..], Some(Format::TokenizerJson)),
            (b"\r\n  {}", Some(Format::TokenizerJson)),
            (b"\x0A\x09\x0A\x05<unk>", Some(Format::SentencePiece)),
            (b"IQ== 0\nIg== 1\n", Some(Format::TiktokenRanks)),
            (b"IQ== 0\r\n", Some(Format::TiktokenRanks)),
            (b"", None),
            (b"hello world", None),
            (b"IQ== zero\n", None),
            (b"kitoken\0\x01", None),
        ] {
            assert_eq!(sniff(bytes), format, "{:?}", String::from_utf8_lossy(bytes));
        }
    }
}
