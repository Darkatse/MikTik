#[cfg(feature = "huggingface")]
pub mod huggingface;
// Both external formats use the same runtime adapter; their public modules only own conversion.
#[cfg(any(feature = "huggingface", feature = "sentencepiece"))]
mod kitoken;
#[cfg(feature = "sentencepiece")]
pub mod sentencepiece;
#[cfg(feature = "openai")]
pub mod tiktoken;

#[cfg(all(test, feature = "huggingface"))]
pub(crate) const TEST_TOKENIZER_JSON: &str = r###"{
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
        "vocab": { "[UNK]": 0, "hello": 1, "world": 2 }
    }
}"###;
