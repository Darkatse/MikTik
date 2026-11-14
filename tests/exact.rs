//! Exact results. The built-in encodings match tiktoken-rs id for id, and each registered format
//! follows its special-token rule.

mod support;

#[cfg(feature = "openai")]
mod openai {
    use miktik::{Framing, Message, TokenizerError, TokenizerRegistry};
    use tiktoken_rs::CoreBPE;

    /// Inputs where the split patterns and special tokens matter.
    const FIXED: &[&str] = &[
        "",
        " ",
        "\n",
        "  \n  ",
        "\t \n\r\n ",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog. I'm sure you'll see it's fine; they've gone.",
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "日本語のテキストと한국어 텍스트, and ÉLAN vs élan vs e\u{301}lan.",
        "fn main() {\n    let x = 1234567 + 3.14159;\n    println!(\"{x}\");\n}\n",
        "trailing spaces   \n\n   ",
        "<|endoftext|>",
        "before<|endoftext|>after <|endofprompt|> <|im_start|>user<|im_end|> <|fim_prefix|>",
        "<|endoftext",
        "🙂👍🏽🏳️‍🌈 \u{3000}全角　ＡＢＣ　１２３",
    ];

    /// Pieces that random texts are built from, chosen to meet at pattern boundaries.
    const FRAGMENTS: &[&str] = &[
        "a",
        "the",
        " the",
        "The",
        "HTTP",
        "I'm",
        "'s",
        "'LL",
        "don't",
        " ",
        "  ",
        "   ",
        "\t",
        "\n",
        "\n\n",
        "\r\n",
        " \n ",
        "\u{a0}",
        "1",
        "12",
        "1234",
        "3.14",
        "中文",
        "汉字测试",
        "の",
        "한국어",
        "é",
        "e\u{301}",
        "ß",
        "Ω",
        "🙂",
        "👍🏽",
        "\u{3000}",
        "Ａ",
        "!",
        "?!",
        "...",
        "//",
        "\\",
        "$",
        "—",
        "\u{200b}",
        "<|endoftext|>",
        "<|endofprompt|>",
        "<|im_start|>",
        "<|fim_prefix|>",
        "<|",
        "|>",
    ];

    /// `encode` treats special tokens as text; message counting and `decode` know them.
    fn assert_matches(model: &str, reference: &CoreBPE) {
        let tokenizer = TokenizerRegistry::new().get(model).unwrap();
        let Framing::ChatMl {
            per_message,
            per_reply,
        } = tokenizer.model().framing()
        else {
            panic!("{model} is counted as ChatML");
        };
        let role = reference.encode_with_special_tokens("user").len();

        let texts = FIXED.iter().map(|text| text.to_string());
        for text in texts.chain(crate::support::random_texts(FRAGMENTS, 2_000, 40)) {
            assert_eq!(
                tokenizer.encode(&text).unwrap(),
                reference.encode_ordinary(&text),
                "{model}: encode {text:?}"
            );
            let ids = reference.encode_with_special_tokens(&text);
            assert_eq!(
                tokenizer
                    .count_messages(&[Message::new("user", text.as_str())])
                    .unwrap(),
                per_reply + per_message + role + ids.len(),
                "{model}: count_messages {text:?}"
            );
            assert_eq!(
                tokenizer.decode(&ids).unwrap(),
                text,
                "{model}: decode {text:?}"
            );
        }
        assert!(matches!(
            tokenizer.decode(&[u32::MAX]),
            Err(TokenizerError::Decode(_))
        ));
    }

    #[test]
    fn o200k_base_matches_tiktoken() {
        assert_matches("gpt-4o", tiktoken_rs::o200k_base_singleton());
    }

    #[test]
    fn cl100k_base_matches_tiktoken() {
        assert_matches("gpt-4", tiktoken_rs::cl100k_base_singleton());
    }

    #[test]
    fn r50k_base_matches_tiktoken() {
        assert_matches("gpt2", tiktoken_rs::r50k_base_singleton());
    }

    #[test]
    fn decoding_part_of_a_character_is_an_error() {
        let gpt2 = TokenizerRegistry::new().get("gpt2").unwrap();
        let ids = gpt2.encode("𝔘").unwrap();
        assert!(ids.len() > 1, "the character spans several byte tokens");
        assert!(matches!(
            gpt2.decode(&ids[..1]),
            Err(TokenizerError::Decode(_))
        ));
    }
}

#[cfg(feature = "huggingface")]
#[test]
fn tokenizer_json_recognizes_special_tokens() {
    use miktik::{Message, TokenizerRegistry};

    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes("claude", support::TOKENIZER_JSON)
        .unwrap();
    let claude = registry.get("claude").unwrap();

    assert_eq!(claude.encode("hello<|end|>").unwrap(), [1, 3]);
    assert_eq!(claude.decode(&[1, 3]).unwrap(), "hello <|end|>");
    let count = |content| {
        claude
            .count_messages(&[Message::new("user", content)])
            .unwrap()
    };
    assert_eq!(count("hello<|end|>"), count("hello") + 1);
}

#[cfg(feature = "sentencepiece")]
#[test]
fn sentencepiece_control_tokens_are_not_text() {
    use miktik::TokenizerRegistry;

    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes("gemma", support::sentencepiece_model())
        .unwrap();
    let gemma = registry.get("gemma").unwrap();

    assert_eq!(gemma.encode("ab").unwrap(), [3, 6]);
    assert_ne!(gemma.encode("<s>").unwrap(), [1]);
    assert_eq!(gemma.decode(&[1, 3, 6, 2]).unwrap(), "ab");
}

#[cfg(feature = "tiktoken-bpe")]
#[test]
fn tiktoken_rank_files_recognize_special_tokens() {
    use miktik::{Message, TokenizerRegistry};

    let registry = TokenizerRegistry::new();
    registry
        .register_model_bytes("kimi", support::kimi_ranks())
        .unwrap();
    let kimi = registry.get("kimi-k2.6").unwrap();

    // Kimi's pattern keeps runs of Han characters together, so `你好` merges.
    assert_eq!(kimi.encode("你好ab").unwrap(), [259, 256]);
    assert_eq!(kimi.decode(&[259, 256]).unwrap(), "你好ab");
    // `<|im_end|>` is Kimi's first special token.
    assert_eq!(
        kimi.encode("<|im_end|>").unwrap(),
        [support::KIMI_VOCAB_LEN]
    );
    let count = |content| {
        kimi.count_messages(&[Message::new("user", content)])
            .unwrap()
    };
    assert_eq!(count("ab<|im_end|>"), count("ab") + 1);
}
