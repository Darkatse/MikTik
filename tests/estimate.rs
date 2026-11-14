//! Estimates: how they add up across cuts and prefixes, and which pieces they match.

mod support;

#[cfg(any(feature = "openai", feature = "huggingface"))]
use miktik::Tokenizer;
#[cfg(any(feature = "openai", feature = "huggingface", feature = "sentencepiece"))]
use miktik::TokenizerRegistry;

/// Fragments that meet at cuts, runs of whitespace and folded characters.
#[cfg(any(feature = "openai", feature = "huggingface"))]
const FRAGMENTS: &[&str] = &[
    "a",
    "the",
    " the",
    "Hello",
    "hello",
    "world",
    " ",
    "  ",
    "\t",
    "\n",
    "\n\n",
    "\r\n",
    " \n",
    "\n ",
    "1234",
    ",",
    "!?",
    "中文",
    "汉字测试",
    "，",
    "。",
    "\u{3000}",
    "ｈｅｌｌｏ",
    "Ａ",
    "🙂",
    "<|end|>",
    "<|endoftext|>",
];

/// Cumulative estimates equal the estimates of each prefix, `stop_at` cuts them short, and the
/// estimate of a text cut in two is the sum of the parts' estimates.
#[cfg(any(feature = "openai", feature = "huggingface"))]
fn assert_estimates_add_up(tokenizer: &Tokenizer) {
    let model = tokenizer.model().id();
    let texts = support::random_texts(FRAGMENTS, 400, 20);
    for group in texts.chunks(8) {
        let (base, additions) = group.split_first().unwrap();
        let mut text = base.clone();
        let per_prefix: Vec<usize> = additions
            .iter()
            .map(|addition| {
                text.push_str(addition);
                tokenizer.estimate_tokens(&text)
            })
            .collect();
        assert_eq!(
            tokenizer.estimate_cumulative_token_counts(base, additions, None),
            per_prefix,
            "{model}: {base:?} + {additions:?}"
        );

        let stop_at = per_prefix[per_prefix.len() / 2];
        let reached = per_prefix
            .iter()
            .position(|&estimate| estimate >= stop_at)
            .unwrap();
        let mut stopped = per_prefix.clone();
        stopped[reached..].fill(per_prefix[reached]);
        assert_eq!(
            tokenizer.estimate_cumulative_token_counts(base, additions, Some(stop_at)),
            stopped,
            "{model}: {base:?} + {additions:?} stopping at {stop_at}"
        );

        // A newline followed by a letter is a cut.
        let before = format!("{}\n", group[0]);
        let after = format!("x{}", group[1]);
        assert_eq!(
            tokenizer.estimate_tokens(&format!("{before}{after}")),
            tokenizer.estimate_tokens(&before) + tokenizer.estimate_tokens(&after),
            "{model}: {before:?} + {after:?}"
        );
    }
}

#[cfg(feature = "openai")]
#[test]
fn estimates_add_up() {
    let registry = TokenizerRegistry::new();
    // gpt2's split starts a part at the last newline of a run of whitespace; gpt-4o's does not.
    for model in ["gpt-4o", "gpt2"] {
        assert_estimates_add_up(&registry.get(model).unwrap());
    }
}

#[cfg(feature = "openai")]
#[test]
fn special_tokens_that_text_does_not_produce_are_not_pieces() {
    let gpt = TokenizerRegistry::new().get("gpt-4o").unwrap();
    assert!(gpt.estimate_tokens("<|endoftext|>") > 1);
}

/// The fixture `tokenizer.json`, normalized to NFKC as Claude's is.
#[cfg(feature = "huggingface")]
fn nfkc_tokenizer() -> std::sync::Arc<Tokenizer> {
    let json = support::TOKENIZER_JSON.replace(
        r#""normalizer": null"#,
        r#""normalizer": { "type": "NFKC" }"#,
    );
    let registry = TokenizerRegistry::new();
    registry.register_model_bytes("claude", json).unwrap();
    registry.get("claude").unwrap()
}

#[cfg(feature = "huggingface")]
#[test]
fn folded_estimates_add_up() {
    assert_estimates_add_up(&nfkc_tokenizer());
}

#[cfg(feature = "huggingface")]
#[test]
fn tokenizer_json_pieces_include_special_tokens_and_fold_full_width_forms() {
    let claude = nfkc_tokenizer();
    assert_eq!(claude.estimate_tokens("hello<|end|>"), 2);
    assert_eq!(claude.estimate_tokens("ｈｅｌｌｏ"), 1);
}

#[cfg(feature = "sentencepiece")]
#[test]
fn sentencepiece_pieces_are_spaced_and_include_user_defined_pieces() {
    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes("gemma", support::sentencepiece_model())
        .unwrap();
    let gemma = registry.get("gemma").unwrap();
    // `ab`, the user-defined `▁▁▁` and `ab`.
    assert_eq!(gemma.estimate_tokens("ab   ab"), 3);
    // Control tokens are not text: three bytes that no piece covers.
    assert_eq!(gemma.estimate_tokens("<s>"), 3);
}
