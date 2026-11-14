# MikTik

Tokenizers for LLMs in Rust: exact token ids and counts, and estimates several times faster, for
OpenAI's encodings and for the `tokenizer.json`, SentencePiece and tiktoken files of other models.
[kitoken](https://github.com/Systemcluster/kitoken) is the one engine behind every format.

```toml
[dependencies]
miktik = { version = "0.5", features = ["full"] }
```

```rust
use miktik::{Message, TokenizerRegistry};

let registry = TokenizerRegistry::new();

// Built in. Names such as "chatgpt-4o-latest", "o3-mini" or "openai/gpt-5" resolve to it.
let gpt = registry.get("gpt-4o")?;
let ids = gpt.encode("Hello, world!")?;
let tokens = gpt.count_tokens("Hello, world!")?;
let estimate = gpt.estimate_tokens("Hello, world!");

// Other models need a source.
registry.register_model_file("claude", "tokenizers/claude.json")?;
let claude = registry.get("claude-sonnet-4-5")?;
let estimate = claude.estimate_messages(&[Message::new("user", "What is Rust?")]);
```

## Models and names

`Model::all()` is the table of tokenizers MikTik knows. Each row has an id, a source and a message
framing:

- The OpenAI encodings ship with the crate: `gpt-4o` (o200k), `gpt-4` and `gpt-3.5-turbo-0301`
  (cl100k) and `gpt2` (r50k).
- The other rows, for Claude, Llama, Mistral, Gemma and Gemini, Qwen, DeepSeek, Kimi, Command R
  and A, Nemo, Yi, Jamba and NerdStash, take a source you register with `register_model_file` or
  `register_model_bytes`: a `tokenizer.json`, a SentencePiece model or a tiktoken rank file, as
  the row accepts. The format is recognized from the content. SillyTavern ships suitable files,
  and the repository's `scripts/fetch-test-assets.sh` names a pinned source for every row.

`Model::resolve` maps a model name to a row: it trims the name, compares it without regard to case,
drops a provider prefix such as `openai/`, and then tries ordered rules, so `qwen3-max` resolves to
`qwen2` and `gemini-3.1-pro` to `gemma`. A name no rule knows resolves to nothing, and
`TokenizerRegistry::get` returns `TokenizerError::UnknownModel`; which tokenizer to use instead is
the caller's choice.

## Exact counts and estimates

`encode`, `decode`, `count_tokens` and `count_messages` are exact. They return `Result`, since the
engine may still have to be built.

The `estimate_*` methods never fail and are 3 to 50 times faster than exact counting on one core,
depending on the model, the text and the processor. An
estimate cuts the text into segments after each newline followed by something other than
whitespace, and covers each segment with the longest pieces of the tokenizer's vocabulary, one token
per piece. Everything it uses is derived from the tokenizer's own definition. On two books in
English and Chinese and a role-play transcript, every model's estimate of a whole text is within 2%
of the exact count, and 95% of 1 KiB pieces are within 5%.

Estimates depend only on the text and add up across segments, which makes three things cheap:

- `estimate_cumulative_token_counts(base, additions, stop_at)` estimates `base` followed by each
  prefix of `additions` in time proportional to the total length.
- `TokenizerRegistry::with_segment_cache(capacity_bytes)` keeps the estimates of recent segments,
  so text estimated again, such as a chat that grows at its end, costs only its new segments.
- The `parallel` feature estimates texts of 256 KiB or more on rayon's threads.

None of them changes a result.

## Artifacts

Converting a source can take hundreds of milliseconds on a phone. `Tokenizer::to_bytes` writes an
artifact that loads in milliseconds:

```rust
use miktik::{TokenizerError, TokenizerRegistry};

let registry = TokenizerRegistry::new();
registry.register_model_file("claude", "claude.mtk")?;
let claude = match registry.get("claude") {
    Err(TokenizerError::Artifact(_)) | Err(TokenizerError::Load(_)) => {
        // No usable artifact: load from the source and store a new one.
        registry.register_model_file("claude", "tokenizers/claude.json")?;
        let claude = registry.get("claude")?;
        std::fs::write("claude.mtk.tmp", claude.to_bytes())?;
        std::fs::rename("claude.mtk.tmp", "claude.mtk")?;
        claude
    }
    loaded => loaded?,
};
```

A tokenizer loaded from an artifact gives the same results as the one that wrote it. Its exact
engine is built on the first exact call, so a caller that only estimates never pays for it in time
or memory. MikTik writes no files: the caller stores artifacts, atomically, since they carry no
checksum. An artifact of another `ARTIFACT_VERSION`, another model or cut short fails with
`TokenizerError::Artifact`.

## Features

Features decide which source formats can be converted; artifacts load with any features.

| Feature | |
| --- | --- |
| `openai` (default) | The built-in OpenAI encodings |
| `huggingface` | `tokenizer.json` sources |
| `sentencepiece` | SentencePiece model sources |
| `tiktoken-bpe` | tiktoken rank file sources |
| `full` | All of the above |
| `parallel` | Estimates of long texts on several threads, with rayon |

MikTik needs Rust 1.92. It is used by [TauriTavern](https://github.com/Darkatse/TauriTavern).

## License

MIT
