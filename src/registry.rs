use std::borrow::Cow;
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
use crate::model::Message;
use crate::tokenizer::{AutoTokenizer, CumulativeEstimateOptions};

type ModelResolver = fn(&str) -> Option<&'static str>;
type ModelLoader = fn(&ModelSource) -> Result<Box<dyn AutoTokenizer>, TokenizerError>;

const FALLBACK_MODEL: &str = "gpt-3.5-turbo";
const TIKTOKEN_MODELS: &[&str] = &[
    "o1",
    "gpt-4o",
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt2",
];
/// Models that ship a raw tiktoken BPE rank file (e.g. `tiktoken.model`)
/// instead of a Hugging Face `tokenizer.json`.
const TIKTOKEN_BPE_MODELS: &[&str] = &["kimi"];
const HUGGINGFACE_MODELS: &[&str] = &[
    "claude",
    "llama3",
    "llama",
    "mistral",
    "yi",
    "gemma",
    "jamba",
    "nerdstash",
    "nerdstash_v2",
    "command-r",
    "command-a",
    "qwen2",
    "nemo",
    "deepseek",
];
const SENTENCEPIECE_MODELS: &[&str] = &[
    "llama",
    "mistral",
    "yi",
    "gemma",
    "jamba",
    "nerdstash",
    "nerdstash_v2",
];
const WEB_TOKENIZER_MODELS: &[&str] = &[
    "claude",
    "llama3",
    "command-r",
    "command-a",
    "qwen2",
    "nemo",
    "deepseek",
];
const O_SERIES_PREFIXES: &[&str] = &["o1", "o3", "o4", "gpt-5", "gpt-6"];
const GPT4_FAMILY_MATCHERS: &[&str] = &["gpt-4", "chatgpt-4o"];
const GPT4O_MATCHERS: &[&str] = &["4o", "4.1", "4.5", "chatgpt-4o"];
const CLAUDE_ALIASES: &[(&str, &str)] = &[("claude", "claude")];
// All Moonshot Kimi releases (K2 through K3) share one byte-identical
// tiktoken BPE vocab, so every "kimi" model id resolves to "kimi".
const KIMI_ALIASES: &[(&str, &str)] = &[("kimi", "kimi")];
const LLAMA_ALIASES: &[(&str, &str)] = &[
    // Keep llama3 ahead of llama to preserve priority.
    ("llama3", "llama3"),
    ("llama-3", "llama3"),
    ("llama", "llama"),
];
const SENTENCEPIECE_ALIASES: &[(&str, &str)] = &[
    ("mistral", "mistral"),
    ("yi", "yi"),
    ("gemma", "gemma"),
    ("gemini", "gemma"),
    ("learnlm", "gemma"),
    ("jamba", "jamba"),
    ("nerdstash-v2", "nerdstash_v2"),
    ("nerdstash_v2", "nerdstash_v2"),
    ("nerdstash", "nerdstash"),
];
const HF_JSON_ALIASES: &[(&str, &str)] = &[
    ("command-r", "command-r"),
    ("command-a", "command-a"),
    ("qwen2", "qwen2"),
    ("nemo", "nemo"),
    ("deepseek", "deepseek"),
];
const MODEL_RESOLUTION_CHAIN: &[ModelResolver] = &[
    resolve_o_series,
    resolve_gpt4_family,
    resolve_claude,
    resolve_kimi,
    resolve_llama_family,
    resolve_sentencepiece_family,
    resolve_hf_json_family,
];

fn contains_any(name: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| name.contains(needle))
}

fn starts_with_any(name: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| name.starts_with(prefix))
}

fn resolve_from_contains_aliases(
    name: &str,
    aliases: &'static [(&'static str, &'static str)],
) -> Option<&'static str> {
    aliases
        .iter()
        .find_map(|(pattern, canonical)| name.contains(pattern).then_some(*canonical))
}

fn resolve_o_series(name: &str) -> Option<&'static str> {
    starts_with_any(name, O_SERIES_PREFIXES).then_some("o1")
}

fn resolve_gpt4_family(name: &str) -> Option<&'static str> {
    if !contains_any(name, GPT4_FAMILY_MATCHERS) {
        return None;
    }
    if name.contains("32k") {
        return Some("gpt-4-32k");
    }
    if contains_any(name, GPT4O_MATCHERS) {
        return Some("gpt-4o");
    }
    Some("gpt-4")
}

fn resolve_claude(name: &str) -> Option<&'static str> {
    resolve_from_contains_aliases(name, CLAUDE_ALIASES)
}

fn resolve_kimi(name: &str) -> Option<&'static str> {
    resolve_from_contains_aliases(name, KIMI_ALIASES)
        .or_else(|| (name == "k3" || name.starts_with("k3-")).then_some("kimi"))
}

fn resolve_llama_family(name: &str) -> Option<&'static str> {
    resolve_from_contains_aliases(name, LLAMA_ALIASES)
}

fn resolve_sentencepiece_family(name: &str) -> Option<&'static str> {
    resolve_from_contains_aliases(name, SENTENCEPIECE_ALIASES)
}

fn resolve_hf_json_family(name: &str) -> Option<&'static str> {
    resolve_from_contains_aliases(name, HF_JSON_ALIASES)
}

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

/// The public facade for resolving model names and dispatching tokenizer calls.
///
/// Internally caches loaded tokenizer instances behind an `RwLock` for
/// thread-safe lazy loading. Callers interact exclusively through this type.
pub struct TokenizerRegistry {
    cache: RwLock<HashMap<String, Arc<dyn AutoTokenizer>>>,
    model_sources: RwLock<HashMap<String, ModelSource>>,
}

impl TokenizerRegistry {
    pub fn supported_tiktoken_models() -> &'static [&'static str] {
        TIKTOKEN_MODELS
    }

    pub fn supported_tiktoken_bpe_models() -> &'static [&'static str] {
        TIKTOKEN_BPE_MODELS
    }

    pub fn supported_huggingface_models() -> &'static [&'static str] {
        HUGGINGFACE_MODELS
    }

    pub fn supported_sentencepiece_models() -> &'static [&'static str] {
        SENTENCEPIECE_MODELS
    }

    pub fn supported_web_tokenizer_models() -> &'static [&'static str] {
        WEB_TOKENIZER_MODELS
    }

    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            model_sources: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tokenizer model file path for a model family.
    ///
    /// Supports web tokenizer JSON files, SentencePiece `.model` files, and
    /// tiktoken BPE rank files (e.g. Kimi's `tiktoken.model`).
    pub fn register_model_file(
        &self,
        model: &str,
        path: impl Into<PathBuf>,
    ) -> Result<(), TokenizerError> {
        self.register_model_source(model, ModelSource::File(path.into()))
    }

    /// Register raw tokenizer bytes for a model family.
    ///
    /// Supports tokenizer JSON payloads, SentencePiece serialized protobuf
    /// bytes, and tiktoken BPE rank payloads (e.g. Kimi's `tiktoken.model`).
    pub fn register_model_bytes(
        &self,
        model: &str,
        data: impl Into<Vec<u8>>,
    ) -> Result<(), TokenizerError> {
        let data = Arc::<[u8]>::from(data.into());
        self.register_model_source(model, ModelSource::Bytes(data))
    }

    /// Register a HuggingFace tokenizer file path for a model family.
    ///
    /// The provided `model` may be any alias; it will be canonicalized via
    /// `resolve_model` before registration.
    pub fn register_huggingface_file(
        &self,
        model: &str,
        path: impl Into<PathBuf>,
    ) -> Result<(), TokenizerError> {
        self.register_model_file(model, path)
    }

    /// Register raw HuggingFace tokenizer bytes for a model family.
    ///
    /// This is useful for embedded assets (`include_bytes!`) or externally
    /// managed caches that already materialized bytes in memory.
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
        let canonical = Self::resolve_model_ref(model);

        // Each file-backed family gates registration on its own compile-time
        // feature. Everything else (built-in tiktoken encodings, unknown
        // models) accepts no resource and fails fast with the reason.
        if Self::is_huggingface_model_canonical(canonical) {
            if !Self::supports_huggingface_family_backends() {
                return Err(TokenizerError::LoadError(format!(
                    "HuggingFace-family backends for '{canonical}' are disabled at compile time; enable feature 'huggingface' or 'sentencepiece'"
                )));
            }
        } else if Self::is_tiktoken_bpe_model_canonical(canonical) {
            if !cfg!(feature = "tiktoken-bpe") {
                return Err(Self::disabled_backend_error(canonical, "tiktoken-bpe"));
            }
        } else {
            return Err(TokenizerError::ModelNotFound(format!(
                "model '{model}' resolves to '{canonical}', which does not accept registered tokenizer resources"
            )));
        }

        {
            let mut sources = self.model_sources.write().map_err(|e| {
                TokenizerError::LoadError(format!("model source lock poisoned: {e}"))
            })?;
            sources.insert(canonical.to_string(), source);
        }

        self.invalidate_cached_model(canonical)
    }

    fn invalidate_cached_model(&self, canonical: &str) -> Result<(), TokenizerError> {
        let mut cache = self
            .cache
            .write()
            .map_err(|e| TokenizerError::LoadError(format!("cache lock poisoned: {e}")))?;
        cache.remove(canonical);
        Ok(())
    }

    /// Resolve a raw model name (with possible aliases/suffixes) to its
    /// canonical tokenizer identifier.
    ///
    /// Implements the alias-matching chain from the architecture spec:
    /// O-series → GPT-4 family → Claude → LLaMA → open-source → fallback.
    pub fn resolve_model(model_name: &str) -> String {
        Self::resolve_model_ref(model_name).to_string()
    }

    pub fn resolve_model_ref(model_name: &str) -> &'static str {
        let trimmed = model_name.trim();
        if trimmed.is_empty() {
            return FALLBACK_MODEL;
        }

        let normalized = if trimmed
            .as_bytes()
            .iter()
            .any(|byte| byte.is_ascii_uppercase())
        {
            Cow::Owned(trimmed.to_ascii_lowercase())
        } else {
            Cow::Borrowed(trimmed)
        };

        if let Some(canonical) = Self::resolve_model_exact(normalized.as_ref()) {
            return canonical;
        }

        for resolver in MODEL_RESOLUTION_CHAIN {
            if let Some(canonical) = resolver(normalized.as_ref()) {
                return canonical;
            }
        }

        FALLBACK_MODEL
    }

    fn resolve_model_exact(normalized: &str) -> Option<&'static str> {
        match normalized {
            "o1" => Some("o1"),
            "gpt-4o" => Some("gpt-4o"),
            "gpt-4" => Some("gpt-4"),
            "gpt-4-32k" => Some("gpt-4-32k"),
            "gpt-3.5-turbo" => Some("gpt-3.5-turbo"),
            "gpt-3.5-turbo-0301" => Some("gpt-3.5-turbo-0301"),
            "gpt2" | "gpt-2" => Some("gpt2"),
            "claude" => Some("claude"),
            "llama3" => Some("llama3"),
            "llama" => Some("llama"),
            "mistral" => Some("mistral"),
            "yi" => Some("yi"),
            "gemma" => Some("gemma"),
            "jamba" => Some("jamba"),
            "nerdstash" => Some("nerdstash"),
            "nerdstash-v2" | "nerdstash_v2" => Some("nerdstash_v2"),
            "command-r" => Some("command-r"),
            "command-a" => Some("command-a"),
            "qwen2" => Some("qwen2"),
            "nemo" => Some("nemo"),
            "deepseek" => Some("deepseek"),
            _ => None,
        }
    }

    pub fn is_tiktoken_model(model: &str) -> bool {
        Self::is_tiktoken_model_canonical(Self::resolve_model_ref(model))
    }

    pub fn is_tiktoken_bpe_model(model: &str) -> bool {
        Self::is_tiktoken_bpe_model_canonical(Self::resolve_model_ref(model))
    }

    pub fn is_huggingface_model(model: &str) -> bool {
        Self::is_huggingface_model_canonical(Self::resolve_model_ref(model))
    }

    pub fn is_sentencepiece_model(model: &str) -> bool {
        Self::is_sentencepiece_model_canonical(Self::resolve_model_ref(model))
    }

    pub fn is_web_tokenizer_model(model: &str) -> bool {
        Self::is_web_tokenizer_model_canonical(Self::resolve_model_ref(model))
    }

    fn is_tiktoken_model_canonical(canonical: &str) -> bool {
        TIKTOKEN_MODELS.contains(&canonical)
    }

    fn is_tiktoken_bpe_model_canonical(canonical: &str) -> bool {
        TIKTOKEN_BPE_MODELS.contains(&canonical)
    }

    fn is_huggingface_model_canonical(canonical: &str) -> bool {
        HUGGINGFACE_MODELS.contains(&canonical)
    }

    fn is_sentencepiece_model_canonical(canonical: &str) -> bool {
        SENTENCEPIECE_MODELS.contains(&canonical)
    }

    fn is_web_tokenizer_model_canonical(canonical: &str) -> bool {
        WEB_TOKENIZER_MODELS.contains(&canonical)
    }

    fn supports_huggingface_family_backends() -> bool {
        cfg!(any(feature = "huggingface", feature = "sentencepiece"))
    }

    fn disabled_backend_error(canonical: &str, feature_hint: &str) -> TokenizerError {
        TokenizerError::LoadError(format!(
            "canonical model '{canonical}' requires compile-time feature '{feature_hint}'"
        ))
    }

    /// Get (or lazily load) a tokenizer for the given model name.
    pub fn get(&self, model: &str) -> Result<Arc<dyn AutoTokenizer>, TokenizerError> {
        let canonical = Self::resolve_model_ref(model);
        self.get_with_canonical(canonical)
    }

    pub fn get_canonical(&self, canonical: &str) -> Result<Arc<dyn AutoTokenizer>, TokenizerError> {
        self.get_with_canonical(canonical)
    }

    fn get_with_canonical(
        &self,
        canonical: &str,
    ) -> Result<Arc<dyn AutoTokenizer>, TokenizerError> {
        // Fast path: read lock
        {
            let cache = self
                .cache
                .read()
                .map_err(|e| TokenizerError::LoadError(format!("cache lock poisoned: {e}")))?;
            if let Some(tok) = cache.get(canonical) {
                return Ok(Arc::clone(tok));
            }
        }

        // Slow path: write lock + load
        let mut cache = self
            .cache
            .write()
            .map_err(|e| TokenizerError::LoadError(format!("cache lock poisoned: {e}")))?;

        // Double-check after acquiring write lock
        if let Some(tok) = cache.get(canonical) {
            return Ok(Arc::clone(tok));
        }

        let tok = self.load_tokenizer(canonical)?;
        let tok = Arc::from(tok);
        cache.insert(canonical.to_string(), Arc::clone(&tok));
        Ok(tok)
    }

    /// Convenience: count tokens for a plain text string.
    pub fn count_tokens(&self, model: &str, text: &str) -> Result<usize, TokenizerError> {
        self.get(model)?.count_tokens(text)
    }

    pub fn count_tokens_canonical(
        &self,
        canonical: &str,
        text: &str,
    ) -> Result<usize, TokenizerError> {
        self.get_canonical(canonical)?.count_tokens(text)
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

    pub fn estimate_cumulative_token_counts_canonical(
        &self,
        canonical: &str,
        base: &str,
        additions: &[&str],
        options: CumulativeEstimateOptions,
    ) -> Result<Vec<usize>, TokenizerError> {
        self.get_canonical(canonical)?
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

    pub fn count_messages_canonical(
        &self,
        canonical: &str,
        messages: &[Message],
    ) -> Result<usize, TokenizerError> {
        self.get_canonical(canonical)?.count_messages(messages)
    }

    /// Internal: instantiate the appropriate backend for a canonical model name.
    fn load_tokenizer(&self, canonical: &str) -> Result<Box<dyn AutoTokenizer>, TokenizerError> {
        if Self::is_tiktoken_model_canonical(canonical) {
            #[cfg(feature = "openai")]
            {
                return Ok(Box::new(TiktokenBackend::from_model(canonical)?));
            }
            #[cfg(not(feature = "openai"))]
            {
                return Err(Self::disabled_backend_error(canonical, "openai"));
            }
        }
        if Self::is_tiktoken_bpe_model_canonical(canonical) {
            #[cfg(feature = "tiktoken-bpe")]
            {
                return self.load_tiktoken_bpe_tokenizer(canonical);
            }
            #[cfg(not(feature = "tiktoken-bpe"))]
            {
                return Err(Self::disabled_backend_error(canonical, "tiktoken-bpe"));
            }
        }
        if Self::is_huggingface_model_canonical(canonical) {
            return self.load_huggingface_family_tokenizer(canonical);
        }

        Err(TokenizerError::ModelNotFound(format!(
            "no backend is configured for canonical model '{canonical}'"
        )))
    }

    #[cfg(feature = "tiktoken-bpe")]
    fn load_tiktoken_bpe_tokenizer(
        &self,
        canonical: &str,
    ) -> Result<Box<dyn AutoTokenizer>, TokenizerError> {
        let source = {
            let sources = self.model_sources.read().map_err(|e| {
                TokenizerError::LoadError(format!("model source lock poisoned: {e}"))
            })?;
            sources.get(canonical).cloned()
        };

        let source = source.ok_or_else(|| {
            TokenizerError::ModelNotFound(format!(
                "tokenizer resource is not registered for canonical model '{canonical}'"
            ))
        })?;

        let backend = match &source {
            ModelSource::File(path) => TiktokenBpeBackend::from_bpe_file(path)?,
            ModelSource::Bytes(bytes) => TiktokenBpeBackend::from_bpe_bytes(bytes.as_ref())?,
        };
        Ok(Box::new(backend))
    }

    fn load_huggingface_family_tokenizer(
        &self,
        canonical: &str,
    ) -> Result<Box<dyn AutoTokenizer>, TokenizerError> {
        if !Self::supports_huggingface_family_backends() {
            return Err(Self::disabled_backend_error(
                canonical,
                "huggingface or sentencepiece",
            ));
        }

        let source = {
            let sources = self.model_sources.read().map_err(|e| {
                TokenizerError::LoadError(format!("model source lock poisoned: {e}"))
            })?;
            sources.get(canonical).cloned()
        };

        let source = source.ok_or_else(|| {
            TokenizerError::ModelNotFound(format!(
                "tokenizer resource is not registered for canonical model '{canonical}'"
            ))
        })?;

        let loaders = Self::loader_chain_for(canonical);
        if loaders.is_empty() {
            return Err(Self::disabled_backend_error(
                canonical,
                "huggingface or sentencepiece",
            ));
        }

        let mut errors = Vec::with_capacity(loaders.len());
        for (format_name, loader) in loaders {
            match loader(&source) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(err) => errors.push(format!("{format_name}: {err}")),
            }
        }

        if !errors.is_empty() {
            return Err(TokenizerError::LoadError(format!(
                "failed to load canonical model '{canonical}': {}",
                errors.join("; ")
            )));
        }

        Err(TokenizerError::ModelNotFound(format!(
            "no HuggingFace-family loader is configured for canonical model '{canonical}'"
        )))
    }

    fn loader_chain_for(canonical: &str) -> Vec<(&'static str, ModelLoader)> {
        #[allow(unused_mut)]
        let mut loaders = Vec::with_capacity(2);
        if Self::is_sentencepiece_model_canonical(canonical) {
            #[cfg(feature = "sentencepiece")]
            loaders.push((
                "SentencePiece",
                Self::load_sentencepiece_from_source as ModelLoader,
            ));
            #[cfg(feature = "huggingface")]
            loaders.push((
                "tokenizer JSON",
                Self::load_huggingface_json_from_source as ModelLoader,
            ));
        } else if Self::is_web_tokenizer_model_canonical(canonical) {
            #[cfg(feature = "huggingface")]
            loaders.push((
                "tokenizer JSON",
                Self::load_huggingface_json_from_source as ModelLoader,
            ));
            #[cfg(feature = "sentencepiece")]
            loaders.push((
                "SentencePiece",
                Self::load_sentencepiece_from_source as ModelLoader,
            ));
        }
        loaders
    }

    #[cfg(feature = "huggingface")]
    fn load_huggingface_json_from_source(
        source: &ModelSource,
    ) -> Result<Box<dyn AutoTokenizer>, TokenizerError> {
        let backend = match source {
            ModelSource::File(path) => HuggingFaceBackend::from_file(path)?,
            ModelSource::Bytes(bytes) => HuggingFaceBackend::from_bytes(bytes.as_ref())?,
        };
        Ok(Box::new(backend))
    }

    #[cfg(feature = "sentencepiece")]
    fn load_sentencepiece_from_source(
        source: &ModelSource,
    ) -> Result<Box<dyn AutoTokenizer>, TokenizerError> {
        let backend = match source {
            ModelSource::File(path) => SentencePieceBackend::from_file(path)?,
            ModelSource::Bytes(bytes) => SentencePieceBackend::from_bytes(bytes.as_ref())?,
        };
        Ok(Box::new(backend))
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
    fn resolve_model_maps_openai_aliases() {
        assert_eq!(
            TokenizerRegistry::resolve_model("gpt-4.5-preview"),
            "gpt-4o"
        );
        assert_eq!(TokenizerRegistry::resolve_model("gpt-4.1"), "gpt-4o");
        assert_eq!(
            TokenizerRegistry::resolve_model("chatgpt-4o-latest"),
            "gpt-4o"
        );
        assert_eq!(TokenizerRegistry::resolve_model("o3-mini"), "o1");
        assert_eq!(TokenizerRegistry::resolve_model("o4-mini"), "o1");
        assert_eq!(TokenizerRegistry::resolve_model("gpt-6-astra"), "o1");
        assert_eq!(TokenizerRegistry::resolve_model("gpt-6-variant"), "o1");
        assert_eq!(
            TokenizerRegistry::resolve_model("gpt-3.5-turbo-0301"),
            "gpt-3.5-turbo-0301"
        );
        assert_eq!(TokenizerRegistry::resolve_model("gpt-2"), "gpt2");
    }

    #[test]
    fn resolve_model_ref_matches_resolve_model() {
        for model in [
            "gpt-4o",
            "gpt-4.1",
            "chatgpt-4o-latest",
            "claude-3-5-sonnet",
            "llama-3.3-70b",
            "my-unknown-model",
        ] {
            assert_eq!(
                TokenizerRegistry::resolve_model_ref(model),
                TokenizerRegistry::resolve_model(model)
            );
        }
    }

    #[test]
    fn resolve_model_keeps_priority_order() {
        // 32k has higher priority than other GPT-4 variants.
        assert_eq!(
            TokenizerRegistry::resolve_model("gpt-4o-32k-experimental"),
            "gpt-4-32k"
        );
        // llama3 rule must match before generic llama.
        assert_eq!(TokenizerRegistry::resolve_model("llama-3.3-70b"), "llama3");
    }

    #[test]
    fn resolve_model_maps_sentencepiece_aliases() {
        assert_eq!(
            TokenizerRegistry::resolve_model("gemini-2.0-flash"),
            "gemma"
        );
        assert_eq!(TokenizerRegistry::resolve_model("learnlm-pro"), "gemma");
        assert_eq!(
            TokenizerRegistry::resolve_model("nerdstash-v2"),
            "nerdstash_v2"
        );
        assert_eq!(
            TokenizerRegistry::resolve_model("nerdstash_v2-tokenizer"),
            "nerdstash_v2"
        );
        assert_eq!(
            TokenizerRegistry::resolve_model("nerdstash-v1"),
            "nerdstash"
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

        let err = match registry.get("nerdstash") {
            Ok(_) => panic!("v1 should not reuse the registered v2 resource"),
            Err(err) => err,
        };
        assert!(matches!(err, TokenizerError::ModelNotFound(_)));
    }

    #[test]
    fn resolve_model_uses_fallback_for_unknown_models() {
        assert_eq!(
            TokenizerRegistry::resolve_model("my-unknown-model"),
            "gpt-3.5-turbo"
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
    fn huggingface_family_models_report_disabled_feature_error() {
        let registry = TokenizerRegistry::new();
        let err = registry
            .count_tokens("claude-3-5-sonnet", "hello")
            .expect_err("non-openai backend should be disabled");

        match err {
            TokenizerError::LoadError(message) => {
                assert!(message.contains("compile-time feature"));
            }
            other => panic!("unexpected error type: {other}"),
        }
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn unregistered_hf_resource_returns_explicit_error() {
        let registry = TokenizerRegistry::new();
        let err = match registry.get("claude-3-5-sonnet") {
            Ok(_) => panic!("should fail without a registered resource"),
            Err(err) => err,
        };

        match err {
            TokenizerError::ModelNotFound(message) => {
                assert!(message.contains("not registered"));
            }
            other => panic!("unexpected error type: {other}"),
        }
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
    #[cfg(feature = "huggingface")]
    fn register_hf_rejects_non_hf_canonical_models() {
        let registry = TokenizerRegistry::new();
        let err = registry
            .register_huggingface_bytes("gpt-4o", test_tokenizer_json().into_bytes())
            .expect_err("openai canonical should be rejected");

        match err {
            TokenizerError::ModelNotFound(message) => {
                assert!(message.contains("does not accept registered tokenizer resources"));
            }
            other => panic!("unexpected error type: {other}"),
        }
    }

    #[test]
    fn resolve_model_maps_kimi_aliases() {
        for model in [
            "kimi",
            "kimi-k3",
            "Kimi-K2.7-Code",
            "kimi-k2.5-0905-preview",
            "moonshotai/kimi-k2.6",
            "kimi-latest",
            "kimi-for-coding",
            "k3-256k",
        ] {
            assert_eq!(
                TokenizerRegistry::resolve_model(model),
                "kimi",
                "model '{model}' should resolve to the kimi tokenizer"
            );
        }
        assert!(TokenizerRegistry::is_tiktoken_bpe_model("kimi-k3"));
        assert!(!TokenizerRegistry::is_tiktoken_bpe_model("gpt-4o"));
        // Intentionally invalid ID: xAI's retired model ID is `grok-3`, not `grok3`.
        assert!(!TokenizerRegistry::is_tiktoken_bpe_model("grok3"));
        assert!(!TokenizerRegistry::is_tiktoken_bpe_model("work3"));
    }

    #[test]
    #[cfg(feature = "tiktoken-bpe")]
    fn unregistered_kimi_resource_returns_explicit_error() {
        let registry = TokenizerRegistry::new();
        let err = match registry.get("kimi-k3") {
            Ok(_) => panic!("should fail without a registered resource"),
            Err(err) => err,
        };

        match err {
            TokenizerError::ModelNotFound(message) => {
                assert!(message.contains("not registered"));
            }
            other => panic!("unexpected error type: {other}"),
        }
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

        // Kitoken-backed models count message content only; chat framing is
        // owned by the caller (see KitokenBackend::count_messages).
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

        match err {
            TokenizerError::LoadError(message) => {
                assert!(message.contains("deserialize"));
            }
            other => panic!("unexpected error type: {other}"),
        }
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

        match err {
            TokenizerError::LoadError(message) => {
                assert!(message.contains(&missing_path.display().to_string()));
            }
            other => panic!("unexpected error type: {other}"),
        }
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
