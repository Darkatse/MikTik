use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use miktik::{Message, TokenizerRegistry};

const PRODUCTION_CORPUS_ENV: &str = "MIKTIK_PRODUCTION_CORPUS";
const DEFAULT_PRODUCTION_CORPUS_PATH: &str = ".local/The Divine Comedy.txt";
const PRODUCTION_MODEL_DIR_ENV: &str = "MIKTIK_PRODUCTION_MODEL_DIR";
const DEFAULT_PRODUCTION_MODEL_DIR: &str = ".local/models";
const CLAUDE_TOKENIZER_FILE: &str = "claude.json";
const DEEPSEEK_TOKENIZER_FILE: &str = "deepseek-v3.tokenizer.json";
const GEMMA_SENTENCEPIECE_FILE: &str = "gemma.model";

fn production_corpus_path() -> PathBuf {
    std::env::var_os(PRODUCTION_CORPUS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PRODUCTION_CORPUS_PATH))
}

fn load_production_corpus() -> String {
    let path = production_corpus_path();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read production corpus from '{}': {err}",
            path.display()
        )
    });

    // Normalize CRLF/LF to keep assertions stable across platforms.
    raw.replace("\r\n", "\n")
}

fn production_model_dir() -> PathBuf {
    std::env::var_os(PRODUCTION_MODEL_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PRODUCTION_MODEL_DIR))
}

fn required_model_path(file_name: &str) -> PathBuf {
    let path = production_model_dir().join(file_name);
    assert!(
        path.is_file(),
        "required model resource is missing: '{}'. Run scripts/download_production_models.ps1 first.",
        path.display()
    );
    path
}

fn chunk_messages(corpus: &str, chunk_count: usize) -> Vec<Message> {
    assert!(chunk_count > 0);
    let lines: Vec<&str> = corpus.lines().collect();
    let chunk_size = lines.len().div_ceil(chunk_count);

    lines
        .chunks(chunk_size.max(1))
        .enumerate()
        .map(|(index, chunk)| {
            let role = if index % 2 == 0 { "user" } else { "assistant" };
            Message::new(role, chunk.join("\n"))
        })
        .collect()
}

fn assert_message_count_matches_content_sum(
    registry: &TokenizerRegistry,
    model: &str,
    messages: &[Message],
) {
    let message_total = registry
        .count_messages(model, messages)
        .unwrap_or_else(|err| panic!("message counting failed for '{model}': {err}"));
    let content_total: usize = messages
        .iter()
        .map(|message| {
            registry
                .count_tokens(model, &message.content)
                .unwrap_or_else(|err| {
                    panic!("per-message content counting failed for '{model}': {err}")
                })
        })
        .sum();

    assert_eq!(
        message_total, content_total,
        "non-OpenAI backends currently count message content only"
    );
}

#[test]
#[ignore = "production-like corpus test; run explicitly with `cargo test --test production_corpus -- --ignored`"]
fn openai_aliases_match_canonical_counts_on_divine_comedy() {
    let registry = TokenizerRegistry::new();
    let corpus = load_production_corpus();

    let pairs = [
        ("chatgpt-4o-latest", "gpt-4o"),
        ("gpt-4.5-preview", "gpt-4o"),
        ("o3-mini", "o1"),
        ("my-unknown-model", "gpt-3.5-turbo"),
    ];

    for (alias, canonical) in pairs {
        let alias_count = registry
            .count_tokens(alias, &corpus)
            .unwrap_or_else(|err| panic!("failed to count tokens for alias '{alias}': {err}"));
        let canonical_count = registry
            .count_tokens(canonical, &corpus)
            .unwrap_or_else(|err| {
                panic!("failed to count tokens for canonical '{canonical}': {err}")
            });

        assert_eq!(
            alias_count, canonical_count,
            "alias '{alias}' should match canonical '{canonical}'"
        );
        assert!(alias_count > 0, "count should be positive for '{alias}'");
    }
}

#[test]
#[ignore = "production-like corpus test; run explicitly with `cargo test --test production_corpus -- --ignored`"]
fn openai_message_count_is_deterministic_for_large_conversation() {
    let registry = TokenizerRegistry::new();
    let corpus = load_production_corpus();
    let messages = chunk_messages(&corpus, 24);

    let first = registry
        .count_messages("gpt-4o", &messages)
        .expect("first message count should succeed");
    let second = registry
        .count_messages("chatgpt-4o-latest", &messages)
        .expect("alias message count should succeed");
    let plain_text = registry
        .count_tokens("gpt-4o", &corpus)
        .expect("plain text count should succeed");

    assert_eq!(
        first, second,
        "alias and canonical message counts must match"
    );
    assert!(
        first > plain_text,
        "chat message accounting should exceed plain text token count"
    );
}

#[test]
#[ignore = "production-like corpus test; run explicitly with `cargo test --test production_corpus -- --ignored`"]
fn concurrent_openai_counting_is_consistent_on_large_corpus() {
    let registry = Arc::new(TokenizerRegistry::new());
    let corpus = Arc::new(load_production_corpus());

    let models = [
        "gpt-4o",
        "chatgpt-4o-latest",
        "gpt-4.5-preview",
        "o1",
        "o3-mini",
        "my-unknown-model",
        "gpt-3.5-turbo",
    ];

    let mut handles = Vec::with_capacity(models.len());
    for model in models {
        let registry = Arc::clone(&registry);
        let corpus = Arc::clone(&corpus);
        handles.push(thread::spawn(move || {
            let count = registry
                .count_tokens(model, &corpus)
                .unwrap_or_else(|err| panic!("count failed for '{model}': {err}"));
            (model, count)
        }));
    }

    let mut results = HashMap::new();
    for handle in handles {
        let (model, count) = handle.join().expect("worker thread should not panic");
        results.insert(model, count);
    }

    assert_eq!(results["chatgpt-4o-latest"], results["gpt-4o"]);
    assert_eq!(results["gpt-4.5-preview"], results["gpt-4o"]);
    assert_eq!(results["o3-mini"], results["o1"]);
    assert_eq!(results["my-unknown-model"], results["gpt-3.5-turbo"]);
}

#[test]
#[ignore = "production-like corpus test; run explicitly with `cargo test --test production_corpus -- --ignored`"]
fn web_tokenizers_claude_and_deepseek_handle_full_corpus() {
    let registry = TokenizerRegistry::new();
    let corpus = load_production_corpus();
    let claude_path = required_model_path(CLAUDE_TOKENIZER_FILE);
    let deepseek_path = required_model_path(DEEPSEEK_TOKENIZER_FILE);

    registry
        .register_model_file("claude-3-5-sonnet", claude_path)
        .expect("claude tokenizer registration should succeed");
    registry
        .register_model_file("deepseek-r1", deepseek_path)
        .expect("deepseek tokenizer registration should succeed");

    let claude_count = registry
        .count_tokens("claude-3-7-sonnet", &corpus)
        .expect("claude count should succeed");
    let claude_canonical = registry
        .count_tokens("claude", &corpus)
        .expect("claude canonical count should succeed");
    assert_eq!(claude_count, claude_canonical);

    let deepseek_count = registry
        .count_tokens("deepseek-chat", &corpus)
        .expect("deepseek count should succeed");
    let deepseek_canonical = registry
        .count_tokens("deepseek", &corpus)
        .expect("deepseek canonical count should succeed");
    assert_eq!(deepseek_count, deepseek_canonical);

    assert!(claude_count > 0);
    assert!(deepseek_count > 0);

    let messages = chunk_messages(&corpus, 32);
    assert_message_count_matches_content_sum(&registry, "claude", &messages);
    assert_message_count_matches_content_sum(&registry, "deepseek", &messages);
}

#[test]
#[ignore = "production-like corpus test; run explicitly with `cargo test --test production_corpus -- --ignored`"]
fn sentencepiece_gemini_alias_handles_full_corpus() {
    let registry = TokenizerRegistry::new();
    let corpus = load_production_corpus();
    let gemma_model_path = required_model_path(GEMMA_SENTENCEPIECE_FILE);

    registry
        .register_model_file("gemini-2.0-flash", gemma_model_path)
        .expect("gemma sentencepiece registration should succeed");

    let gemini_count = registry
        .count_tokens("gemini-2.0-flash", &corpus)
        .expect("gemini alias count should succeed");
    let gemma_count = registry
        .count_tokens("gemma-2-9b", &corpus)
        .expect("gemma canonical count should succeed");
    let learnlm_count = registry
        .count_tokens("learnlm-pro", &corpus)
        .expect("learnlm alias count should succeed");

    assert!(gemini_count > 0);
    assert_eq!(gemini_count, gemma_count);
    assert_eq!(gemini_count, learnlm_count);

    let messages = chunk_messages(&corpus, 32);
    assert_eq!(
        registry
            .count_tokens("gemini-2.0-flash", &corpus)
            .expect("second gemini count should succeed"),
        gemini_count,
        "sentencepiece counting should be deterministic"
    );
    assert_message_count_matches_content_sum(&registry, "gemini-2.0-flash", &messages);
}
