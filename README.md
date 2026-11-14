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

- The OpenAI encodings ship with the crate: `gpt-4o` and `gpt-5` (o200k), `gpt-4` and
  `gpt-3.5-turbo-0301` (cl100k) and `gpt2` (r50k).
- The other rows, for Claude, Llama, Mistral, Gemini and Gemma, Qwen and Qwen 3.5, DeepSeek, Kimi
  and Kimi K3, GLM, MiniMax, Command R and A, Nemo, Yi, Jamba and NerdStash, take a source you
  register with `register_model_file` or `register_model_bytes`: a `tokenizer.json`, a
  SentencePiece model or a tiktoken rank file, as the row accepts. The format is recognized from the content. The
  repository's `scripts/fetch-test-assets.sh` names a pinned source for every row.

`Model::resolve` maps a model name to a row: it trims the name, compares it without regard to case,
drops a provider prefix such as `openai/`, and then tries ordered rules, so `qwen3-max` resolves to
`qwen2`, `qwen3.6-plus` to `qwen3.5` and `gemini-3.1-pro` to `gemini`. A name no rule knows resolves
to nothing, and `TokenizerRegistry::get` returns `TokenizerError::UnknownModel`; which tokenizer to
use instead is the caller's choice.

A row's exact counts equal those of the models it serves, as their providers count them; the
repository's `scripts/provider-count-check.py` compares them. A few providers publish no tokenizer:
Gemini 4 and Grok resolve to the closest row, within 1% and 3.1% on the test texts. Claude's row is
the tokenizer Anthropic published for Claude 2; Claude 3 to 4.6 count within 2% of it on English
and 11% more on Chinese, and Claude 4.7 and later about 25% fewer on English.

A row's framing is what its provider adds around chat messages, measured the same way: the script
counts a user, an assistant and a user message against the first one alone. `count_messages` and
`estimate_messages` count each such pair of messages as the provider does, or one token over. Some
templates frame the two roles differently (Kimi K3 adds 11 tokens to a user message and 23 to an
assistant one), so a single message is off by half the difference. A default system prompt that
some providers add to every request (DeepSeek, MiniMax, Kimi K3, Grok) is not counted. Rows whose framing no provider has been measured for count only message content:
Llama, Mistral, Nemo, Yi, Gemma, Jamba, Command R and A, and NerdStash.

## Exact counts and estimates

`encode`, `decode`, `count_tokens` and `count_messages` are exact; `count_messages` adds the row's
framing to the exact counts of each role and content. They return `Result`, since the engine may
still have to be built.

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
