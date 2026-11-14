//! Artifacts: a tokenizer loaded from its artifact behaves as the one that wrote it, and an
//! artifact that does not fit fails with `TokenizerError::Artifact`.

#![cfg(any(feature = "openai", feature = "huggingface", feature = "sentencepiece"))]

mod support;

use miktik::{Message, Tokenizer, TokenizerRegistry};

/// Texts with whitespace runs, folded characters and special-token text.
const FRAGMENTS: &[&str] = &[
    "a",
    "ab",
    "hello",
    " world",
    " ",
    "   ",
    "\n",
    "\r\n",
    "\n\n",
    "中文",
    "ｈｅｌｌｏ",
    "\u{3000}",
    "<|end|>",
    "<|endoftext|>",
    "🙂",
];

/// Loads `tokenizer`'s artifact into a registry of its own and compares every method.
fn assert_round_trip(tokenizer: &Tokenizer) {
    let model = tokenizer.model().id();
    let artifact = tokenizer.to_bytes();
    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes(model, artifact.clone())
        .unwrap();
    let loaded = registry.get(model).unwrap();
    assert_eq!(loaded.to_bytes(), artifact, "{model}");

    let texts = support::random_texts(FRAGMENTS, 300, 12);
    for text in &texts {
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(loaded.encode(text).unwrap(), ids, "{model}: {text:?}");
        assert_eq!(loaded.decode(&ids).ok(), tokenizer.decode(&ids).ok());
        let messages = [Message::new("user", text.as_str())];
        assert_eq!(
            loaded.count_messages(&messages).unwrap(),
            tokenizer.count_messages(&messages).unwrap()
        );
        assert_eq!(
            loaded.estimate_tokens(text),
            tokenizer.estimate_tokens(text),
            "{model}: {text:?}"
        );
    }
    assert_eq!(
        loaded.estimate_cumulative_token_counts("", &texts, None),
        tokenizer.estimate_cumulative_token_counts("", &texts, None)
    );
}

#[cfg(feature = "openai")]
#[test]
fn built_in_tokenizers_round_trip() {
    let registry = TokenizerRegistry::new();
    // gpt2's split starts a part at the last newline of a whitespace run; gpt-4o's does not.
    for model in ["gpt-4o", "gpt2"] {
        assert_round_trip(&registry.get(model).unwrap());
    }
}

#[cfg(feature = "huggingface")]
#[test]
fn folding_tokenizers_round_trip() {
    let json = support::TOKENIZER_JSON.replace(
        r#""normalizer": null"#,
        r#""normalizer": { "type": "NFKC" }"#,
    );
    let registry = TokenizerRegistry::new();
    registry.register_model_bytes("claude", json).unwrap();
    assert_round_trip(&registry.get("claude").unwrap());
}

#[cfg(feature = "sentencepiece")]
#[test]
fn sentencepiece_tokenizers_round_trip() {
    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes("gemma", support::sentencepiece_model())
        .unwrap();
    assert_round_trip(&registry.get("gemma").unwrap());
}

#[cfg(feature = "openai")]
mod refused {
    use miktik::{TokenizerError, TokenizerRegistry};

    fn artifact() -> Vec<u8> {
        TokenizerRegistry::new().get("gpt-4o").unwrap().to_bytes()
    }

    fn artifact_error(model: &str, artifact: Vec<u8>) -> String {
        let registry = TokenizerRegistry::new();
        registry.register_model_bytes(model, artifact).unwrap();
        match registry.get(model) {
            Err(TokenizerError::Artifact(message)) => message,
            Err(other) => panic!("expected an artifact error, got {other}"),
            Ok(_) => panic!("expected an artifact error"),
        }
    }

    #[test]
    fn artifacts_must_be_of_this_version_and_model_and_whole() {
        let artifact = artifact();

        let mut other_version = artifact.clone();
        // The version follows the eight bytes of the magic.
        other_version[8] ^= 0xFF;
        assert!(artifact_error("gpt-4o", other_version).contains("version"));

        assert!(artifact_error("gpt-4", artifact.clone()).contains("model 'gpt-4o'"));

        let truncated = artifact[..artifact.len() - 1].to_vec();
        assert!(artifact_error("gpt-4o", truncated).contains("truncated"));

        let mut longer = artifact;
        longer.push(0);
        assert!(artifact_error("gpt-4o", longer).contains("follow the end"));
    }

    #[test]
    fn a_damaged_definition_fails_on_first_exact_use() {
        let mut artifact = artifact();
        // The definition, after the magic, the version, the model id and the section's length,
        // starts with kitoken's own magic.
        artifact[8 + 4 + 2 + "gpt-4o".len() + 8] ^= 0xFF;
        let registry = TokenizerRegistry::new();
        registry.register_model_bytes("gpt-4o", artifact).unwrap();
        let gpt = registry.get("gpt-4o").unwrap();

        assert!(gpt.estimate_tokens("Hello, world!") > 0);
        assert!(matches!(
            gpt.count_tokens("Hello, world!"),
            Err(TokenizerError::Artifact(_))
        ));
    }
}
