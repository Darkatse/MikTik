# MikTik maintainer guide

MikTik is a general-purpose tokenizer library for LLMs. TauriTavern is one of its users; the API is not shaped around any single caller.

## Principles

- **One path per fact.** Every fact about a model or a source format is written in one place, and every operation has one entry point. Do not add parallel lists, aliases of public functions, or compatibility wrappers.
- **Fail fast.** Unknown model names, missing sources and malformed data return a typed error that names the model and the source. Never fall back silently.
- **Measure before adding complexity.** A performance claim or a design choice that rests on speed cites a measurement, and the method is written down so it can be repeated.
- **Tests cover behavior.** Test observable behavior, non-trivial invariants and defects that were fixed. Do not test literals, direct delegation or removed features.
- **Comments state invariants.** Code comments explain what must hold and why non-obvious code is correct. The history of a decision belongs in the commit message.

## Invariants

1. **No `unsafe`.** The crate is `#![forbid(unsafe_code)]`. No input bytes can cause undefined behavior.
2. **The only I/O is reading files the caller registers.** No network access, no writes.

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
