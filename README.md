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
miktik = "0.2"
```

Default build enables only OpenAI (`tiktoken-rs`) for a minimal footprint.

Enable additional backends explicitly when needed:

```toml
[dependencies]
miktik = { version = "0.2", features = ["huggingface", "sentencepiece"] }
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
- O-series (`o1`/`o3`/`o4`/`gpt-5`) -> `o1`
- GPT-4 family -> `gpt-4o` / `gpt-4-32k` / `gpt-4` (e.g. `gpt-4.1` -> `gpt-4o`)
- Legacy model variants are preserved when they affect counting (e.g. `gpt-3.5-turbo-0301`)
- Claude / LLaMA / open-source aliases -> canonical family id
- Unknown models fallback to `gpt-3.5-turbo`

For performance-sensitive callers, prefer non-allocating resolution:

```rust
use miktik::TokenizerRegistry;

let canonical = TokenizerRegistry::resolve_model_ref("chatgpt-4o-latest");
assert_eq!(canonical, "gpt-4o");
```

If you already keep a canonical model string around, you can bypass resolution entirely:

```rust
use miktik::TokenizerRegistry;

let registry = TokenizerRegistry::new();
let canonical = "gpt-4o";
let count = registry.count_tokens_canonical(canonical, "Hello!")?;
# Ok::<(), miktik::TokenizerError>(())
```

You can also query model families (resolution-aware):

```rust
use miktik::TokenizerRegistry;

assert!(TokenizerRegistry::is_tiktoken_model("gpt-4.1"));
assert!(TokenizerRegistry::is_huggingface_model("claude-3-5-sonnet"));
```

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
