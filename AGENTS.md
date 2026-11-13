# MikTik maintainer guide

MikTik is a general-purpose tokenizer library for LLMs. TauriTavern is one of its users; the API is not shaped around any single caller.

## Principles

- **One path per fact.** Every fact about a model or a source format is written in one place, and every operation has one entry point. Do not add parallel lists, aliases of public functions, or compatibility wrappers.
- **Fail fast.** Unknown model names, missing sources and malformed data return a typed error that names the model and the source. Never fall back silently.
- **Measure before adding complexity.** A performance claim or a design choice that rests on speed cites a measurement, and the method is written down so it can be repeated.
- **Tests cover behavior.** Test observable behavior, non-trivial invariants and defects that were fixed. Do not test literals, direct delegation or removed features.
- **Comments state invariants.** Code comments explain what must hold and why non-obvious code is correct. The history of a decision belongs in the commit message.

## Invariants

1. **Model facts are written once.** A model's id, source and message framing live only in the model table (`src/model.rs`), and name resolution only produces rows of that table. Nothing else branches on a model id. Which special tokens each operation recognizes follows from the source format (`SpecialTokens` in `src/source.rs`).
2. **A source is checked when it loads.** `get` converts the source and builds the engine, so a source that cannot be used fails there, with an error naming the model and the source, and stays registered for the next try. A loaded `Tokenizer` never changes.
3. **Exact results have a reference.** The built-in encodings match tiktoken-rs id for id (`tests/exact.rs`). Their rank files are OpenAI's (hashes in `assets/openai/README.md`); their split patterns and special tokens are tiktoken's, written in `src/source.rs`.
4. **No `unsafe`.** The crate is `#![forbid(unsafe_code)]`. No input bytes can cause undefined behavior.
5. **The only I/O is reading files the caller registers.** No network access, no writes.

## Adding a model

1. Add a row to `MODELS` in `src/model.rs`: id, source (built-in encoding or accepted formats) and framing.
2. If names other than the id should resolve to it, add a rule to `RULES`, before the first rule that would match those names wrongly.
3. Add a representative name to the resolution tests.

## Upgrading kitoken

kitoken is the only engine, so an upgrade can change any exact result. Before taking one:

- Run the full test matrix; `tests/exact.rs` compares the built-in encodings with tiktoken-rs and checks each format's special-token rule.
- Read the changes to the converters and to how `encode` splits around special tokens. MikTik's built-in definitions depend on special tokens with `extract: false` being split only where they are recognized.
- Raise `rust-version` if kitoken's rises.

## Commands

```bash
cargo fmt --check
cargo clippy --all-targets --no-default-features -- -D warnings  # also with no flag and with --all-features
cargo test --no-default-features                                 # also with no flag and with --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo +1.92 check --locked --all-features                        # MSRV: rust-version in Cargo.toml
cargo package --locked                                           # the .crate must stay under 10 MB
```

CI (`.github/workflows/ci.yml`) runs the same set.

## MSRV

`rust-version` is the highest `rust-version` among the required dependencies (kitoken 0.11.1 needs 1.92). Raise it in the same commit as the dependency that needs it.
