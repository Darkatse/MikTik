# MikTik

Unified, multi-backend tokenizer library for LLMs, written in Rust.

MikTik provides one interface for token counting across model families:
- OpenAI models via `tiktoken-rs`
- Web tokenizer JSON models via `tokenizers`
- SentencePiece `.model` tokenizers via `sentencepiece-model` + `tokenizers`

## Design Goals
- Unified API for `encode` / `decode` / message token counting
- Lazy loading with thread-safe registry cache
- Explicit error propagation (no hidden panic paths)
- No network I/O in core library (resource ownership stays in caller)

## Installation

```toml
[dependencies]
miktik = "0.1.2"
```

Default build enables only OpenAI (`tiktoken-rs`) for a minimal footprint.

Enable additional backends explicitly when needed:

```toml
[dependencies]
miktik = { version = "0.1", features = ["huggingface", "sentencepiece"] }
```

Feature matrix:
- `openai` (default): OpenAI-compatible counting via `tiktoken-rs`
- `huggingface`: tokenizer JSON loading via `tokenizers`
- `sentencepiece`: SentencePiece `.model` loading (`huggingface` implied)
- `full`: convenience bundle (`openai + huggingface + sentencepiece`)

## Quick Start

```rust
use miktik::{Message, TokenizerRegistry};

let registry = TokenizerRegistry::new();

let text_tokens = registry.count_tokens("gpt-4o", "Hello, world!")?;

// Requires `huggingface` feature.
registry.register_model_file("claude", "/path/to/claude-tokenizer.json")?;
let chat_tokens = registry.count_messages(
    "claude",
    &[
        Message::new("user", "What is Rust?"),
        Message::new("assistant", "A systems programming language."),
    ],
)?;
# Ok::<(), miktik::TokenizerError>(())
```

## Model Resolution

Raw model names are canonicalized by rule chain:
- O-series (`o1`/`o3`/`gpt-5`) -> `o1`
- GPT-4 family -> `gpt-4o` / `gpt-4-32k` / `gpt-4`
- Claude / LLaMA / open-source aliases -> canonical family id
- Unknown models fallback to `gpt-3.5-turbo`

## Model Resource Registration

For non-OpenAI families, register resources before counting:
- `register_model_file(model, path)`
- `register_model_bytes(model, bytes)`
- Compatibility aliases:
  - `register_huggingface_file(model, path)`
  - `register_huggingface_bytes(model, bytes)`

Supported formats:
- Web tokenizer: `tokenizer.json`
- SentencePiece: `tokenizer.model`

## Thread Safety

`TokenizerRegistry` is safe for concurrent use:
- Uses `RwLock<HashMap<...>>` for lazy cache
- Uses double-check locking to avoid duplicate instantiation

## Integration

MikTik is designed for general Rust LLM projects and is actively used in
`TauriTavern`.

- TauriTavern: `https://github.com/Darkatse/TauriTavern`

## License

MIT
