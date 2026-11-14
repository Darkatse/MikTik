//! Small tokenizer sources built in place, one per registered format.

// Each test crate uses only some of the fixtures.
#![allow(dead_code)]

use base64::Engine;

/// A WordPiece `tokenizer.json`: `[UNK]` 0, `hello` 1, `world` 2 and the special token `<|end|>` 3.
pub const TOKENIZER_JSON: &str = r###"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [
        {
            "id": 0,
            "content": "[UNK]",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        },
        {
            "id": 3,
            "content": "<|end|>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }
    ],
    "normalizer": null,
    "pre_tokenizer": { "type": "WhitespaceSplit" },
    "post_processor": null,
    "decoder": {
        "type": "WordPiece",
        "prefix": "##",
        "cleanup": true
    },
    "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "max_input_chars_per_word": 100,
        "continuing_subword_prefix": "##",
        "vocab": { "[UNK]": 0, "hello": 1, "world": 2, "<|end|>": 3 }
    }
}"###;

/// A BPE SentencePiece model: `<unk>` 0, the control tokens `<s>` 1 and `</s>` 2, then `▁` 3, `a` 4,
/// `b` 5, the merge `ab` 6 and the user-defined piece `▁▁▁` 7, like Gemma's runs of spaces.
pub fn sentencepiece_model() -> Vec<u8> {
    // The ModelProto is written field by field, following sentencepiece_model.proto, so the tests
    // need no protobuf library.
    let pieces = [
        piece("<unk>", 0.0, 2),
        piece("<s>", 0.0, 3),
        piece("</s>", 0.0, 3),
        piece("▁", 0.0, 1),
        piece("a", 0.0, 1),
        piece("b", 0.0, 1),
        piece("ab", 10.0, 1),
        piece("▁▁▁", 0.0, 4),
    ];
    let mut model = Vec::new();
    for value in pieces {
        message_field(&mut model, 1, &value);
    }
    let mut trainer = Vec::new();
    // TrainerSpec.model_type = BPE.
    varint_field(&mut trainer, 3, 2);
    message_field(&mut model, 2, &trainer);
    model
}

fn piece(text: &str, score: f32, kind: u64) -> Vec<u8> {
    let mut value = Vec::new();
    message_field(&mut value, 1, text.as_bytes());
    key(&mut value, 2, 5);
    value.extend_from_slice(&score.to_bits().to_le_bytes());
    varint_field(&mut value, 3, kind);
    value
}

fn message_field(output: &mut Vec<u8>, field: u32, value: &[u8]) {
    key(output, field, 2);
    varint(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn varint_field(output: &mut Vec<u8>, field: u32, value: u64) {
    key(output, field, 0);
    varint(output, value);
}

fn key(output: &mut Vec<u8>, field: u32, wire_type: u8) {
    varint(output, ((field as u64) << 3) | u64::from(wire_type));
}

fn varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        output.push((value as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

/// The size of Kimi's vocabulary, from which kitoken infers Kimi's split pattern and special tokens.
pub const KIMI_VOCAB_LEN: u32 = 163_584;

/// A tiktoken rank file the size of Kimi's: every byte, then the merges `ab` 256, `你` 257, `好` 258
/// and `你好` 259, then unique fillers.
pub fn kimi_ranks() -> Vec<u8> {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
    tokens.push(b"ab".to_vec());
    tokens.push("你".as_bytes().to_vec());
    tokens.push("好".as_bytes().to_vec());
    tokens.push("你好".as_bytes().to_vec());
    while tokens.len() < KIMI_VOCAB_LEN as usize {
        tokens.push(format!("~filler-{}~", tokens.len()).into_bytes());
    }
    tokens
        .iter()
        .enumerate()
        .map(|(rank, token)| {
            format!(
                "{} {rank}",
                base64::engine::general_purpose::STANDARD.encode(token)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes()
}

/// `count` texts of 1 to `max_len` fragments, drawn from a fixed seed so failures reproduce.
pub fn random_texts(fragments: &[&str], count: usize, max_len: usize) -> Vec<String> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut below = move |bound: usize| {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % bound
    };
    (0..count)
        .map(|_| {
            let len = 1 + below(max_len);
            (0..len)
                .map(|_| fragments[below(fragments.len())])
                .collect()
        })
        .collect()
}

/// A path in the system temp directory that no other test uses.
pub fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("miktik-{}-{name}", std::process::id()))
}
