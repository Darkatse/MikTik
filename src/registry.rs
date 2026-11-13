use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[cfg(feature = "huggingface")]
use crate::backend::huggingface::HuggingFaceBackend;
#[cfg(feature = "sentencepiece")]
use crate::backend::sentencepiece::SentencePieceBackend;
#[cfg(feature = "openai")]
use crate::backend::tiktoken::TiktokenBackend;
#[cfg(feature = "tiktoken-bpe")]
use crate::backend::tiktoken_bpe::TiktokenBpeBackend;
use crate::error::TokenizerError;
use crate::message::Message;
use crate::model::{Format, Model, Source};
use crate::tokenizer::{AutoTokenizer, CumulativeEstimateOptions};

#[cfg_attr(
    not(any(
        feature = "huggingface",
        feature = "sentencepiece",
        feature = "tiktoken-bpe"
    )),
    allow(dead_code)
)]
#[derive(Clone, Debug)]
enum ModelSource {
    File(PathBuf),
    Bytes(Arc<[u8]>),
}

/// Loads tokenizers by model name and caches them.
pub struct TokenizerRegistry {
    cache: RwLock<HashMap<&'static str, Arc<dyn AutoTokenizer>>>,
    model_sources: RwLock<HashMap<&'static str, ModelSource>>,
}

impl TokenizerRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            model_sources: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tokenizer file for a model.
    pub fn register_model_file(
        &self,
        model: &str,
        path: impl Into<PathBuf>,
    ) -> Result<(), TokenizerError> {
        self.register_model_source(model, ModelSource::File(path.into()))
    }

    /// Register tokenizer bytes for a model.
    pub fn register_model_bytes(
        &self,
        model: &str,
        data: impl Into<Vec<u8>>,
    ) -> Result<(), TokenizerError> {
        let data = Arc::<[u8]>::from(data.into());
        self.register_model_source(model, ModelSource::Bytes(data))
    }

    pub fn register_huggingface_file(
        &self,
        model: &str,
        path: impl Into<PathBuf>,
    ) -> Result<(), TokenizerError> {
        self.register_model_file(model, path)
    }

    pub fn register_huggingface_bytes(
        &self,
        model: &str,
        data: impl Into<Vec<u8>>,
    ) -> Result<(), TokenizerError> {
        self.register_model_bytes(model, data)
    }

    fn register_model_source(
        &self,
        model: &str,
        source: ModelSource,
    ) -> Result<(), TokenizerError> {
        let model = resolve(model)?;
        if let Source::Builtin(_) = model.source() {
            return Err(TokenizerError::Load(format!(
                "model '{}' is built in and does not accept a registered source",
                model.id()
            )));
        }

        {
            let mut sources = self
                .model_sources
                .write()
                .map_err(|e| TokenizerError::Load(format!("model source lock poisoned: {e}")))?;
            sources.insert(model.id(), source);
        }

        let mut cache = self
            .cache
            .write()
            .map_err(|e| TokenizerError::Load(format!("cache lock poisoned: {e}")))?;
        cache.remove(model.id());
        Ok(())
    }

    /// Get (or lazily load) a tokenizer for the given model name.
    pub fn get(&self, model: &str) -> Result<Arc<dyn AutoTokenizer>, TokenizerError> {
        let model = resolve(model)?;
        {
            let cache = self
                .cache
                .read()
                .map_err(|e| TokenizerError::Load(format!("cache lock poisoned: {e}")))?;
            if let Some(tok) = cache.get(model.id()) {
                return Ok(Arc::clone(tok));
            }
        }

        let mut cache = self
            .cache
            .write()
            .map_err(|e| TokenizerError::Load(format!("cache lock poisoned: {e}")))?;
        if let Some(tok) = cache.get(model.id()) {
            return Ok(Arc::clone(tok));
        }

        let tok = Arc::from(self.load_tokenizer(model)?);
        cache.insert(model.id(), Arc::clone(&tok));
        Ok(tok)
    }

    /// Convenience: count tokens for a plain text string.
    pub fn count_tokens(&self, model: &str, text: &str) -> Result<usize, TokenizerError> {
        self.get(model)?.count_tokens(text)
    }

    /// Estimate token counts after cumulatively appending each addition to `base`.
    pub fn estimate_cumulative_token_counts(
        &self,
        model: &str,
        base: &str,
        additions: &[&str],
        options: CumulativeEstimateOptions,
    ) -> Result<Vec<usize>, TokenizerError> {
        self.get(model)?
            .estimate_cumulative_token_counts(base, additions, options)
    }

    /// Convenience: count tokens for a list of chat messages.
    pub fn count_messages(
        &self,
        model: &str,
        messages: &[Message],
    ) -> Result<usize, TokenizerError> {
        self.get(model)?.count_messages(messages)
    }

    fn load_tokenizer(
        &self,
        model: &'static Model,
    ) -> Result<Box<dyn AutoTokenizer>, TokenizerError> {
        match model.source() {
            #[cfg(feature = "openai")]
            Source::Builtin(encoding) => {
                Ok(Box::new(TiktokenBackend::new(encoding, model.framing())))
            }
            #[cfg(not(feature = "openai"))]
            Source::Builtin(_) => Err(TokenizerError::Load(format!(
                "model '{}' needs the 'openai' feature",
                model.id()
            ))),
            Source::Registered(formats) => {
                let source = {
                    let sources = self.model_sources.read().map_err(|e| {
                        TokenizerError::Load(format!("model source lock poisoned: {e}"))
                    })?;
                    sources.get(model.id()).cloned()
                };
                let source = source.ok_or(TokenizerError::NotRegistered(model.id()))?;

                let mut errors = Vec::with_capacity(formats.len());
                for &format in formats {
                    match load_format(format, &source) {
                        Ok(tokenizer) => return Ok(tokenizer),
                        Err(err) => errors.push(format!("{format:?}: {err}")),
                    }
                }
                Err(TokenizerError::Load(format!(
                    "model '{}': {}",
                    model.id(),
                    errors.join("; ")
                )))
            }
        }
    }
}

fn resolve(name: &str) -> Result<&'static Model, TokenizerError> {
    Model::resolve(name).ok_or_else(|| TokenizerError::UnknownModel(name.to_string()))
}

#[allow(unused_variables)]
fn load_format(
    format: Format,
    source: &ModelSource,
) -> Result<Box<dyn AutoTokenizer>, TokenizerError> {
    match format {
        #[cfg(feature = "huggingface")]
        Format::TokenizerJson => Ok(Box::new(match source {
            ModelSource::File(path) => HuggingFaceBackend::from_file(path)?,
            ModelSource::Bytes(bytes) => HuggingFaceBackend::from_bytes(bytes)?,
        })),
        #[cfg(feature = "sentencepiece")]
        Format::SentencePiece => Ok(Box::new(match source {
            ModelSource::File(path) => SentencePieceBackend::from_file(path)?,
            ModelSource::Bytes(bytes) => SentencePieceBackend::from_bytes(bytes)?,
        })),
        #[cfg(feature = "tiktoken-bpe")]
        Format::TiktokenRanks => Ok(Box::new(match source {
            ModelSource::File(path) => TiktokenBpeBackend::from_bpe_file(path)?,
            ModelSource::Bytes(bytes) => TiktokenBpeBackend::from_bpe_bytes(bytes)?,
        })),
        #[allow(unreachable_patterns)]
        _ => Err(TokenizerError::Load(format!(
            "{format:?} needs a feature that is not enabled"
        ))),
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "huggingface")]
    use std::fs;
    #[cfg(feature = "openai")]
    use std::thread;
    #[cfg(feature = "huggingface")]
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[cfg(feature = "huggingface")]
    fn test_tokenizer_json() -> String {
        crate::backend::TEST_TOKENIZER_JSON.to_string()
    }

    #[cfg(feature = "huggingface")]
    fn temp_json_path() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "miktik-registry-hf-test-{}-{nonce}.json",
            std::process::id()
        ))
    }

    #[test]
    fn unknown_names_are_errors() {
        let registry = TokenizerRegistry::new();
        assert_eq!(
            registry.get("my-unknown-model").err(),
            Some(TokenizerError::UnknownModel("my-unknown-model".into()))
        );
        assert_eq!(
            registry.register_model_bytes("my-unknown-model", Vec::new()),
            Err(TokenizerError::UnknownModel("my-unknown-model".into()))
        );
    }

    #[test]
    fn builtin_models_reject_registered_sources() {
        let registry = TokenizerRegistry::new();
        let err = registry
            .register_model_bytes("gpt-4o", Vec::new())
            .expect_err("built-in models take no source");
        assert!(matches!(err, TokenizerError::Load(_)));
    }

    #[test]
    fn unregistered_models_report_the_missing_source() {
        let registry = TokenizerRegistry::new();
        assert_eq!(
            registry.get("claude-3-5-sonnet").err(),
            Some(TokenizerError::NotRegistered("claude"))
        );
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn nerdstash_versions_keep_registered_resources_separate() {
        let registry = TokenizerRegistry::new();
        registry
            .register_model_bytes("nerdstash-v2", test_tokenizer_json().into_bytes())
            .expect("v2 resource registration should succeed");

        assert_eq!(
            registry
                .count_tokens("nerdstash_v2", "hello")
                .expect("v2 counting should succeed"),
            1
        );
        assert_eq!(
            registry.get("nerdstash").err(),
            Some(TokenizerError::NotRegistered("nerdstash"))
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn get_reuses_cached_tokenizer_instances() {
        let registry = TokenizerRegistry::new();
        let first = registry.get("gpt-4o").expect("first load should succeed");
        let second = registry
            .get("chatgpt-4o-latest")
            .expect("alias should resolve to cached instance");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    #[cfg(feature = "openai")]
    fn cumulative_estimates_are_available_for_tiktoken_models() {
        let registry = TokenizerRegistry::new();
        let base = "A stable world description with ordinary words and punctuation.\n".repeat(100);
        let additions = ["First entry.\n", "第二条目。\n", "Spaces  \n🙂\n"];

        let estimates = registry
            .estimate_cumulative_token_counts(
                "gpt-4o",
                &base,
                &additions,
                CumulativeEstimateOptions::default(),
            )
            .expect("cumulative estimate should succeed");

        let mut text = base;
        for (addition, estimate) in additions.iter().zip(estimates) {
            text.push_str(addition);
            let exact = registry
                .count_tokens("gpt-4o", &text)
                .expect("exact count should succeed");
            assert!(estimate.abs_diff(exact) * 20 <= exact.max(1));
        }
    }

    #[test]
    #[cfg(not(any(feature = "huggingface", feature = "sentencepiece")))]
    fn registered_sources_need_their_format_feature() {
        let registry = TokenizerRegistry::new();
        registry
            .register_model_bytes("claude", b"{}".to_vec())
            .expect("registration only records the source");
        let err = registry
            .count_tokens("claude-3-5-sonnet", "hello")
            .expect_err("tokenizer.json support is disabled");
        assert!(matches!(err, TokenizerError::Load(message) if message.contains("feature")));
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn register_hf_bytes_enables_claude_counting() {
        let registry = TokenizerRegistry::new();
        registry
            .register_huggingface_bytes("claude-3-5-sonnet", test_tokenizer_json().into_bytes())
            .expect("resource registration should succeed");

        let count = registry
            .count_tokens("claude", "hello")
            .expect("counting should succeed");
        let cumulative = registry
            .estimate_cumulative_token_counts(
                "claude",
                "",
                &["hello", "world"],
                CumulativeEstimateOptions::default(),
            )
            .expect("cumulative estimate should succeed");
        assert_eq!(count, 1);
        assert_eq!(cumulative, vec![1, 1]);
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn register_hf_file_enables_llama3_counting() {
        let registry = TokenizerRegistry::new();
        let json = test_tokenizer_json();
        let path = temp_json_path();
        fs::write(&path, json.as_bytes()).expect("fixture should be written");

        registry
            .register_huggingface_file("llama3", path.clone())
            .expect("resource registration should succeed");

        let count = registry
            .count_tokens("llama-3.3-70b", "world")
            .expect("counting should succeed");

        let _ = fs::remove_file(path);
        assert_eq!(count, 1);
    }

    #[test]
    #[cfg(feature = "tiktoken-bpe")]
    fn register_tiktoken_bpe_bytes_enables_kimi_counting() {
        let registry = TokenizerRegistry::new();
        registry
            .register_model_bytes(
                "moonshotai/kimi-k3",
                crate::backend::tiktoken_bpe::tests::test_kimi_tiktoken_payload(),
            )
            .expect("kimi resource registration should succeed");

        let count = registry
            .count_tokens("kimi-k3", "你好ab")
            .expect("kimi counting should succeed");
        assert_eq!(count, 2);

        let messages = [Message::new("user", "ab")];
        let total = registry
            .count_messages("kimi", &messages)
            .expect("kimi message counting should succeed");
        assert_eq!(total, 1);
    }

    #[test]
    #[cfg(feature = "openai")]
    fn concurrent_get_reuses_single_cached_instance() {
        let registry = Arc::new(TokenizerRegistry::new());
        let models = [
            "gpt-4o",
            "chatgpt-4o-latest",
            "gpt-4.5-preview",
            "gpt-4o-mini",
            "chatgpt-4o-latest",
            "gpt-4o",
            "gpt-4.5-preview",
            "gpt-4o-mini",
        ];

        let mut handles = Vec::with_capacity(models.len());
        for model in models {
            let registry = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                registry
                    .get(model)
                    .expect("all aliases should resolve and load")
            }));
        }

        let results = handles
            .into_iter()
            .map(|handle| handle.join().expect("thread should not panic"))
            .collect::<Vec<_>>();

        let first = &results[0];
        for tok in &results[1..] {
            assert!(Arc::ptr_eq(first, tok));
        }
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn invalid_hf_resource_bytes_propagate_load_error() {
        let registry = TokenizerRegistry::new();
        registry
            .register_huggingface_bytes("claude", b"{not-json".to_vec())
            .expect("registration should accept raw bytes");

        let err = registry
            .count_tokens("claude", "hello")
            .expect_err("invalid resource should fail at load");
        assert!(matches!(err, TokenizerError::Load(message) if message.contains("deserialize")));
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn sentencepiece_family_can_fallback_to_json_payload() {
        let registry = TokenizerRegistry::new();
        registry
            .register_model_bytes("gemini-2.0-flash", test_tokenizer_json().into_bytes())
            .expect("registration should accept bytes");

        let count = registry
            .count_tokens("gemma", "hello")
            .expect("json fallback should succeed");
        assert_eq!(count, 1);
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn invalid_hf_resource_file_propagates_load_error() {
        let registry = TokenizerRegistry::new();
        let missing_path = std::env::temp_dir().join("miktik-missing-tokenizer.json");
        let _ = fs::remove_file(&missing_path);

        registry
            .register_huggingface_file("claude", missing_path.clone())
            .expect("registration should accept path");

        let err = registry
            .count_tokens("claude", "hello")
            .expect_err("missing file should fail at load");
        assert!(
            matches!(err, TokenizerError::Load(message) if message.contains(&missing_path.display().to_string()))
        );
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn register_hf_source_replaces_cached_instance() {
        let registry = TokenizerRegistry::new();
        registry
            .register_huggingface_bytes("claude", test_tokenizer_json().into_bytes())
            .expect("first resource registration should succeed");

        let first = registry
            .get("claude")
            .expect("first tokenizer instance should load");

        registry
            .register_huggingface_bytes("claude", test_tokenizer_json().into_bytes())
            .expect("second resource registration should succeed");

        let second = registry
            .get("claude")
            .expect("second tokenizer instance should load");

        assert!(!Arc::ptr_eq(&first, &second));
    }
}
