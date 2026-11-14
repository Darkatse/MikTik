use std::path::PathBuf;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::error::TokenizerError;
use crate::model::Model;
use crate::source::SourceData;
use crate::tokenizer::Tokenizer;

/// One tokenizer per model, loaded on first use.
///
/// Built-in models load without registration. Other models need a registered source; registering
/// only records it, and [`get`](Self::get) reads it. Any model also takes an artifact that
/// [`Tokenizer::to_bytes`] wrote, which loads in milliseconds; for a built-in model it takes the
/// place of the built-in encoding. Each model loads at most once at a time and independently of
/// other models. A failed load is not remembered: the source stays registered and the next `get`
/// tries again.
pub struct TokenizerRegistry {
    /// One slot per row of [`Model::all`], in the same order.
    slots: Box<[Mutex<Slot>]>,
    /// The bytes of segment text each tokenizer's estimate cache holds.
    segment_cache: usize,
}

enum Slot {
    Empty,
    Registered(SourceData),
    Loaded(Arc<Tokenizer>),
}

impl TokenizerRegistry {
    /// A registry whose tokenizers keep no estimates.
    pub fn new() -> Self {
        Self::with_segment_cache(0)
    }

    /// A registry whose tokenizers each keep the estimates of recent segments, up to
    /// `capacity_bytes` of segment text. Estimates are the same with and without the cache; text
    /// that is estimated again, such as a long chat that grows at its end, is estimated faster.
    pub fn with_segment_cache(capacity_bytes: usize) -> Self {
        Self {
            slots: Model::all()
                .iter()
                .map(|_| Mutex::new(Slot::Empty))
                .collect(),
            segment_cache: capacity_bytes,
        }
    }

    /// Registers a file as the model's source, replacing any earlier source. Tokenizers already
    /// returned by [`get`](Self::get) are not affected.
    pub fn register_model_file(
        &self,
        model: &str,
        path: impl Into<PathBuf>,
    ) -> Result<(), TokenizerError> {
        self.register(model, SourceData::File(path.into()))
    }

    /// Registers bytes as the model's source, replacing any earlier source. Tokenizers already
    /// returned by [`get`](Self::get) are not affected.
    pub fn register_model_bytes(
        &self,
        model: &str,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<(), TokenizerError> {
        self.register(model, SourceData::Bytes(bytes.into()))
    }

    /// The tokenizer of the model the name resolves to, loading it on first use.
    pub fn get(&self, model: &str) -> Result<Arc<Tokenizer>, TokenizerError> {
        let model = resolve(model)?;
        let mut slot = self.slot(model);
        let tokenizer = match &*slot {
            Slot::Loaded(tokenizer) => return Ok(Arc::clone(tokenizer)),
            Slot::Empty => Tokenizer::load(model, None, self.segment_cache)?,
            Slot::Registered(source) => Tokenizer::load(model, Some(source), self.segment_cache)?,
        };
        let tokenizer = Arc::new(tokenizer);
        *slot = Slot::Loaded(Arc::clone(&tokenizer));
        Ok(tokenizer)
    }

    fn register(&self, model: &str, source: SourceData) -> Result<(), TokenizerError> {
        *self.slot(resolve(model)?) = Slot::Registered(source);
        Ok(())
    }

    fn slot(&self, model: &'static Model) -> MutexGuard<'_, Slot> {
        let index = Model::all()
            .iter()
            .position(|row| ptr::eq(row, model))
            .expect("every model is a row of the model table");
        // A slot is only ever replaced whole, so a panic while the lock was held cannot leave it
        // half-written, and the value inside a poisoned lock is still valid.
        self.slots[index]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn resolve(name: &str) -> Result<&'static Model, TokenizerError> {
    Model::resolve(name).ok_or_else(|| TokenizerError::UnknownModel(name.to_string()))
}
