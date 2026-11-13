//! Registry behavior: name resolution errors, registration, loading and its failures.

mod support;

use miktik::{TokenizerError, TokenizerRegistry};

fn load_error(registry: &TokenizerRegistry, model: &str) -> String {
    match registry.get(model) {
        Err(TokenizerError::Load(message)) => message,
        Err(other) => panic!("{model}: expected a load error, got {other}"),
        Ok(_) => panic!("{model}: expected a load error"),
    }
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
fn models_without_a_source_are_not_registered() {
    let registry = TokenizerRegistry::new();
    assert_eq!(
        registry.get("claude-sonnet-4-5").err(),
        Some(TokenizerError::NotRegistered("claude"))
    );
}

#[test]
fn sources_must_be_a_format_the_model_accepts() {
    let registry = TokenizerRegistry::new();

    registry
        .register_model_bytes("claude", b"not a tokenizer".to_vec())
        .unwrap();
    assert!(load_error(&registry, "claude").contains("not a tokenizer.json"));

    registry
        .register_model_bytes("claude", support::sentencepiece_model())
        .unwrap();
    assert!(load_error(&registry, "claude").contains("accepts tokenizer.json"));

    registry
        .register_model_bytes("gpt-4o", support::TOKENIZER_JSON)
        .unwrap();
    assert!(load_error(&registry, "gpt-4o").contains("built in"));
}

#[cfg(not(feature = "huggingface"))]
#[test]
fn formats_need_their_feature() {
    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes("claude", support::TOKENIZER_JSON)
        .unwrap();
    assert!(load_error(&registry, "claude").contains("'huggingface' feature"));
}

#[cfg(feature = "tiktoken-bpe")]
#[test]
fn sources_that_cannot_be_built_fail_to_load() {
    let registry = TokenizerRegistry::new();
    // `!` is ranked twice.
    registry
        .register_model_bytes("kimi", b"IQ== 0\nIQ== 1\nIg== 2\n".to_vec())
        .unwrap();
    assert!(load_error(&registry, "kimi").contains("registered bytes"));
}

#[cfg(feature = "huggingface")]
#[test]
fn an_empty_vocabulary_fails_to_load() {
    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes(
            "claude",
            br#"{"model": {"type": "BPE", "vocab": {}, "merges": []}}"#.to_vec(),
        )
        .unwrap();
    assert!(load_error(&registry, "claude").contains("vocabulary is empty"));
}

#[cfg(feature = "huggingface")]
#[test]
fn a_failed_load_keeps_the_source_for_the_next_get() {
    let path = support::temp_path("late-tokenizer.json");
    let _ = std::fs::remove_file(&path);
    let registry = TokenizerRegistry::new();
    registry.register_model_file("claude", &path).unwrap();

    assert!(load_error(&registry, "claude").contains(&path.display().to_string()));
    std::fs::write(&path, support::TOKENIZER_JSON).unwrap();
    let loaded = registry.get("claude");
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded.unwrap().count_tokens("hello").unwrap(), 1);
}

#[cfg(feature = "huggingface")]
#[test]
fn registering_again_replaces_the_source() {
    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes("claude", support::TOKENIZER_JSON)
        .unwrap();
    let first = registry.get("claude").unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &first,
        &registry.get("claude").unwrap()
    ));

    registry
        .register_model_bytes("claude", b"{}".to_vec())
        .unwrap();
    load_error(&registry, "claude");
    assert_eq!(first.encode("hello").unwrap(), [1]);
}

#[cfg(feature = "openai")]
#[test]
fn names_of_one_model_share_one_tokenizer() {
    let registry = TokenizerRegistry::new();
    let names = ["gpt-4o", "o3-mini", "openai/gpt-5", "chatgpt-4o-latest"];
    let start = std::sync::Barrier::new(names.len());
    let tokenizers = std::thread::scope(|scope| {
        let handles = names.map(|name| {
            let (registry, start) = (&registry, &start);
            scope.spawn(move || {
                start.wait();
                registry.get(name).unwrap()
            })
        });
        handles.map(|handle| handle.join().unwrap())
    });
    for tokenizer in &tokenizers[1..] {
        assert!(std::sync::Arc::ptr_eq(&tokenizers[0], tokenizer));
    }
}
