use kitoken::Kitoken;

use crate::error::TokenizerError;
use crate::message::Message;
use crate::model::{Framing, Model};
use crate::source::{self, SourceData, SpecialTokens};

/// A loaded tokenizer for one model. It never changes after it loads.
pub struct Tokenizer {
    model: &'static Model,
    kitoken: Kitoken,
    specials: SpecialTokens,
}

impl Tokenizer {
    pub(crate) fn load(
        model: &'static Model,
        source: Option<&SourceData>,
    ) -> Result<Self, TokenizerError> {
        let definition = source::definition(model, source)?;
        let specials = SpecialTokens::of(model, &definition);
        let kitoken = Kitoken::from_definition(definition)
            .map_err(|err| source::load_error(model, source, err))?;
        Ok(Self {
            model,
            kitoken,
            specials,
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

    /// Counts chat messages as the model's [`Framing`] describes.
    pub fn count_messages(&self, messages: &[Message]) -> Result<usize, TokenizerError> {
        let count = |text: &str| {
            self.encode_with(text, self.specials.in_messages)
                .map(|ids| ids.len())
        };
        match self.model.framing() {
            Framing::ChatMl {
                per_message,
                per_reply,
            } => messages.iter().try_fold(per_reply, |total, message| {
                Ok(total + per_message + count(&message.role)? + count(&message.content)?)
            }),
            Framing::ContentOnly => messages
                .iter()
                .try_fold(0, |total, message| Ok(total + count(&message.content)?)),
        }
    }

    fn encode_with(&self, text: &str, specials: bool) -> Result<Vec<u32>, TokenizerError> {
        self.kitoken
            .encode(text, specials)
            .map_err(|err| TokenizerError::Encode(format!("model '{}': {err}", self.model.id())))
    }
}
