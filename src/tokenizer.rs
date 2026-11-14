use std::convert::Infallible;

use kitoken::Kitoken;

use crate::error::TokenizerError;
use crate::estimate::Estimator;
use crate::message::Message;
use crate::model::Model;
use crate::source::{self, SourceData, SpecialTokens};

/// A loaded tokenizer for one model. It never changes after it loads.
///
/// The `count_*` methods, [`encode`](Self::encode) and [`decode`](Self::decode) are exact. The
/// `estimate_*` methods are fast approximations that never fail; see the crate documentation.
pub struct Tokenizer {
    model: &'static Model,
    kitoken: Kitoken,
    specials: SpecialTokens,
    estimator: Estimator,
}

impl Tokenizer {
    pub(crate) fn load(
        model: &'static Model,
        source: Option<&SourceData>,
    ) -> Result<Self, TokenizerError> {
        let error = |detail: String| source::load_error(model, source, detail);
        let definition = source::definition(model, source)?;
        let specials = SpecialTokens::of(model, &definition);
        let estimator = Estimator::from_definition(&definition, specials.in_text).map_err(error)?;
        let kitoken = Kitoken::from_definition(definition).map_err(|err| error(err.to_string()))?;
        Ok(Self {
            model,
            kitoken,
            specials,
            estimator,
        })
    }

    /// The model this tokenizer belongs to.
    pub fn model(&self) -> &'static Model {
        self.model
    }

    /// Encodes text into token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_with(text, self.specials.in_text)
    }

    /// Decodes token ids into text. Fails when the ids decode to bytes that are not UTF-8, such as
    /// part of a multi-byte character.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let error =
            |err: String| TokenizerError::Decode(format!("model '{}': {err}", self.model.id()));
        let bytes = self
            .kitoken
            .decode(ids, self.specials.in_messages)
            .map_err(|err| error(err.to_string()))?;
        String::from_utf8(bytes).map_err(|err| error(err.to_string()))
    }

    /// The number of tokens `encode` returns for the text.
    pub fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        self.encode(text).map(|ids| ids.len())
    }

    /// Counts chat messages as the model's [`Framing`](crate::Framing) describes.
    pub fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        self.model.framing().total(messages, |text| {
            self.encode_with(text, self.specials.in_messages)
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

    fn encode_with(&self, text: &str, specials: bool) -> Result<Vec<u32>, TokenizerError> {
        self.kitoken
            .encode(text, specials)
            .map_err(|err| TokenizerError::Encode(format!("model '{}': {err}", self.model.id())))
    }
}
