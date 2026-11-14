# Changelog

## 0.5.0

MikTik now reads every format through kitoken, describes its models in one public table, and adds
estimates, artifacts and a segment cache. The API changed; the table at the end maps 0.4 to 0.5.

### Added

- `Tokenizer::estimate_tokens`, `estimate_messages` and `estimate_cumulative_token_counts`:
  estimates 3 to 50 times faster than exact counting, derived from each tokenizer's definition.
  Every model's estimate of a whole text is within 2% of the exact count on the test texts.
- `TokenizerRegistry::with_segment_cache`, which keeps the estimates of recent segments.
- The `parallel` feature, which estimates texts of 256 KiB or more on rayon's threads.
- Artifacts: `Tokenizer::to_bytes`, `ARTIFACT_VERSION` and `TokenizerError::Artifact`. Every model
  takes an artifact as its source and loads it in milliseconds, so built-in models load without the
  `openai` feature too.
- `Model`, with `Model::all`, `Model::resolve`, `id`, `source` and `framing`, and the `Source`,
  `Encoding`, `Format` and `Framing` types.

### Changed

- kitoken is the only engine. The built-in OpenAI encodings ship as rank files and give the same
  token ids as tiktoken-rs, which is no longer a dependency. kitoken is pinned to an exact version,
  since artifacts hold its serialized definitions.
- A name no rule knows is an error, `UnknownModel`, instead of falling back to `gpt-3.5-turbo`.
- Name resolution drops provider prefixes such as `openai/`, and resolves `qwen`, `qwq`,
  `gpt-3.5` and `gpt-35` names.
- The model ids `o1`, `gpt-4-32k` and `gpt-3.5-turbo` are gone: those names resolve to `gpt-4o` and
  `gpt-4`, with the same encoding and framing.
- `kimi` counts messages with ChatML framing: three tokens per message and three for the reply.
- Sources are recognized by their content. A source that cannot be used fails at `get` with an
  error naming the model and the source, and stays registered for the next try.
- The minimum Rust version is 1.92.

### Removed

- The `backend` module, `AutoTokenizer` and `CumulativeEstimateOptions`.
- The registry's methods that forward to a tokenizer by model name, the `*_canonical` variants,
  `supported_*_models`, `is_*_model`, `resolve_model`, `resolve_model_ref` and
  `register_huggingface_*`.

### Migrating from 0.4

| 0.4 | 0.5 |
| --- | --- |
| `registry.count_tokens(model, text)` and the other methods that take a model name | `registry.get(model)?.count_tokens(text)` |
| `get_canonical`, `*_canonical` | `get`; a model id resolves to itself |
| `resolve_model`, `resolve_model_ref`, falling back to `gpt-3.5-turbo` | `Model::resolve(name)`, which returns `None` for an unknown name; `get` returns `UnknownModel` |
| `supported_*_models()`, `is_*_model()` | `Model::all()`, `Model::source()`, `Model::framing()` |
| `Arc<dyn AutoTokenizer>` | `Arc<Tokenizer>`, with the same method names |
| `estimate_cumulative_token_counts(.., CumulativeEstimateOptions { context_bytes, stop_at })` | `estimate_cumulative_token_counts(.., stop_at)`, equal to the estimate of each prefix |
| `register_huggingface_file`, `register_huggingface_bytes` | `register_model_file`, `register_model_bytes` |
| `TokenizerError::{ModelNotFound, LoadError, EncodeError, DecodeError}` | `UnknownModel`, `NotRegistered`, `Load`, `Artifact`, `Encode`, `Decode` |
| Counting with `count_*` where an approximation is enough | `estimate_*` |
