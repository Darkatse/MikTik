use std::convert::Infallible;
use std::sync::OnceLock;

use kitoken::{Definition, Kitoken};

use crate::artifact;
use crate::error::TokenizerError;
use crate::estimate::Estimator;
use crate::message::Message;
use crate::model::Model;
use crate::source::{self, Content, SourceData, SpecialTokens};

/// A loaded tokenizer for one model. It never changes after it loads.
///
/// The `count_*` methods, [`encode`](Self::encode) and [`decode`](Self::decode) are exact. The
/// `estimate_*` methods are fast approximations that never fail; see the crate documentation.
pub struct Tokenizer {
    model: &'static Model,
    /// kitoken's serialized definition: what an artifact carries, and what the engine is built from.
    definition: Box<[u8]>,
    /// The exact engine. Built when the tokenizer loads from a source, so a source that cannot be
    /// used fails there; built on first use when it loads from an artifact, which was written from
    /// a source that loaded.
    engine: OnceLock<Result<Engine, TokenizerError>>,
    estimator: Estimator,
}

/// kitoken, and which special tokens it recognizes.
struct Engine {
    kitoken: Kitoken,
    specials: SpecialTokens,
}

impl Engine {
    fn build(model: &Model, definition: Definition) -> Result<Self, String> {
        // kitoken panics on an empty vocabulary.
        if definition.model.vocab().is_empty() {
            return Err("the vocabulary is empty".to_string());
        }
        let specials = SpecialTokens::of(model, &definition.meta);
        let kitoken = Kitoken::from_definition(definition).map_err(|err| err.to_string())?;
        Ok(Self { kitoken, specials })
    }
}

impl Tokenizer {
    /// Loads the model's tokenizer from `source`, keeping estimates of up to `segment_cache` bytes
    /// of recent segments.
    pub(crate) fn load(
        model: &'static Model,
        source: Option<&SourceData>,
        segment_cache: usize,
    ) -> Result<Self, TokenizerError> {
        let tokenizer = match source::read(model, source)? {
            Content::Definition(definition) => Self::from_definition(model, *definition)
                .map_err(|detail| TokenizerError::Load(source::describe(model, source, detail)))?,
            Content::Artifact(artifact) => {
                Self::from_artifact(model, &artifact).map_err(|detail| {
                    TokenizerError::Artifact(source::describe(model, source, detail))
                })?
            }
        };
        Ok(Self {
            estimator: tokenizer.estimator.with_segment_cache(segment_cache),
            ..tokenizer
        })
    }

    fn from_definition(model: &'static Model, definition: Definition) -> Result<Self, String> {
        let specials = SpecialTokens::of(model, &definition.meta);
        let estimator = Estimator::from_definition(&definition, specials.in_text)?;
        let serialized = definition.to_vec().into_boxed_slice();
        let engine = Engine::build(model, definition)?;
        Ok(Self {
            model,
            definition: serialized,
            engine: OnceLock::from(Ok(engine)),
            estimator,
        })
    }

    fn from_artifact(model: &'static Model, artifact: &[u8]) -> Result<Self, String> {
        let sections = artifact::read(model, artifact)?;
        Ok(Self {
            model,
            definition: sections.definition.into(),
            engine: OnceLock::new(),
            estimator: Estimator::read(sections.estimator)?,
        })
    }

    /// The model this tokenizer belongs to.
    pub fn model(&self) -> &'static Model {
        self.model
    }

    /// Encodes text into token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let engine = self.engine()?;
        self.encode_with(engine, text, engine.specials.in_text)
    }

    /// Decodes token ids into text. Fails when the ids decode to bytes that are not UTF-8, such as
    /// part of a multi-byte character.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let error =
            |err: String| TokenizerError::Decode(format!("model '{}': {err}", self.model.id()));
        let engine = self.engine()?;
        let bytes = engine
            .kitoken
            .decode(ids, engine.specials.in_messages)
            .map_err(|err| error(err.to_string()))?;
        String::from_utf8(bytes).map_err(|err| error(err.to_string()))
    }

    /// The number of tokens `encode` returns for the text.
    pub fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        self.encode(text).map(|ids| ids.len())
    }

    /// Counts chat messages as the model's [`Framing`](crate::Framing) describes.
    pub fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        let engine = self.engine()?;
        self.model.framing().total(messages, |text| {
            self.encode_with(engine, text, engine.specials.in_messages)
                .map(|ids| ids.len())
        })
    }

    /// Estimates the tokens in the text.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        self.estimator.estimate(text)
    }

    /// Estimates chat messages as the model's [`Framing`](crate::Framing) describes, with each role
    /// and content estimated by [`estimate_tokens`](Self::estimate_tokens).
    pub fn estimate_messages(&self, messages: &[Message]) -> usize {
        let Ok(total) = self.model.framing().total(messages, |text| {
            Ok::<_, Infallible>(self.estimate_tokens(text))
        });
        total
    }

    /// Estimates `base` followed by each prefix of `additions`: the `i`th result is
    /// [`estimate_tokens`](Self::estimate_tokens) of `base + additions[0] + … + additions[i]`. The
    /// cost grows with the total length rather than with the sum of the prefixes' lengths.
    ///
    /// Once a result reaches `stop_at`, the remaining results repeat it.
    pub fn estimate_cumulative_token_counts(
        &self,
        base: &str,
        additions: &[impl AsRef<str>],
        stop_at: Option<usize>,
    ) -> Vec<usize> {
        self.estimator.cumulative(base, additions, stop_at)
    }

    /// The tokenizer as an artifact: register these bytes for the same model, with
    /// [`register_model_bytes`](crate::TokenizerRegistry::register_model_bytes) or as a file, and
    /// the model loads without converting its source. Every method of the loaded tokenizer gives the
    /// same results as this one's. The caller stores the artifact; see
    /// [`ARTIFACT_VERSION`](crate::ARTIFACT_VERSION).
    pub fn to_bytes(&self) -> Vec<u8> {
        artifact::write(self.model, &self.definition, &self.estimator)
    }

    /// The exact engine, built on first use when the tokenizer loaded from an artifact.
    fn engine(&self) -> Result<&Engine, TokenizerError> {
        self.engine
            .get_or_init(|| {
                Definition::from_slice(&self.definition)
                    .map_err(|err| err.to_string())
                    .and_then(|definition| Engine::build(self.model, definition))
                    .map_err(|detail| {
                        TokenizerError::Artifact(format!(
                            "model '{}', the artifact's definition: {detail}",
                            self.model.id()
                        ))
                    })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn encode_with(
        &self,
        engine: &Engine,
        text: &str,
        specials: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        engine
            .kitoken
            .encode(text, specials)
            .map_err(|err| TokenizerError::Encode(format!("model '{}': {err}", self.model.id())))
    }
}
