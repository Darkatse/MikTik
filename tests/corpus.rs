//! Estimates against exact counts on real tokenizers and texts, for every model in the table.
//!
//! `scripts/fetch-test-assets.sh` fetches the inputs into `.local/`; then
//!
//! ```text
//! MIKTIK_MODEL_DIR=.local/models MIKTIK_CORPUS_DIR=.local/corpora \
//!     cargo test --release --all-features --test corpus -- --ignored --nocapture
//! ```
//!
//! The model directory holds a source for every model that needs one, named after the model id
//! (`claude.json`, `gemma.model`, `kimi.tiktoken`); every `.txt` in the corpus directory is a text.
//! A text with CRLF line ends is also checked with LF ones, since tokenizers split them differently.
//! Each model's artifact must give the same exact counts and estimates as its source.

#![cfg(all(
    feature = "openai",
    feature = "huggingface",
    feature = "sentencepiece",
    feature = "tiktoken-bpe"
))]

use std::path::{Path, PathBuf};

use miktik::{Encoding, Model, Source, Tokenizer, TokenizerRegistry};

/// The largest error of a whole text's estimate, in percent.
const WHOLE_LIMIT: f64 = 2.0;
/// The largest 95th percentile of the errors on pieces of [`CHUNK`] bytes, in percent.
const CHUNK_P95_LIMIT: f64 = 5.0;
const CHUNK: usize = 1024;

#[test]
#[ignore = "needs MIKTIK_MODEL_DIR and MIKTIK_CORPUS_DIR; see the module documentation"]
fn estimates_stay_close_to_exact_counts() {
    let models = directory("MIKTIK_MODEL_DIR");
    let texts = texts(&directory("MIKTIK_CORPUS_DIR"));
    let registry = TokenizerRegistry::new();
    let artifacts = TokenizerRegistry::new();
    let mut failures = Vec::new();
    for model in Model::all() {
        if let Source::Registered(_) = model.source() {
            registry
                .register_model_file(model.id(), source_file(&models, model.id()))
                .unwrap();
        }
        let tokenizer = registry.get(model.id()).unwrap();
        artifacts
            .register_model_bytes(model.id(), tokenizer.to_bytes())
            .unwrap();
        let from_artifact = artifacts.get(model.id()).unwrap();
        for (name, text) in &texts {
            if let Source::Builtin(encoding) = model.source() {
                assert_matches_tiktoken(&tokenizer, encoding, name, text);
            }
            let (exact, estimate) = (
                tokenizer.count_tokens(text).unwrap(),
                tokenizer.estimate_tokens(text),
            );
            assert_eq!(
                (
                    from_artifact.count_tokens(text).unwrap(),
                    from_artifact.estimate_tokens(text)
                ),
                (exact, estimate),
                "{} on {name}: the artifact differs from the source",
                model.id()
            );
            let whole = error(estimate, exact);
            let p95 = chunk_p95(&tokenizer, text);
            println!(
                "{:<18} {name:<6} whole {whole:+6.2}%  {CHUNK} B p95 {p95:5.2}%",
                model.id()
            );
            if whole.abs() > WHOLE_LIMIT || p95 > CHUNK_P95_LIMIT {
                failures.push(format!(
                    "{} on {name}: whole {whole:+.2}%, p95 {p95:.2}%",
                    model.id()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

fn directory(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable} is not set; see the module documentation"))
}

/// Every `.txt` in the directory, by file stem, and the LF version of each with CRLF line ends.
fn texts(directory: &Path) -> Vec<(String, String)> {
    let mut texts: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "txt"))
        .flat_map(|path| {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&path).unwrap();
            let lf = text
                .contains("\r\n")
                .then(|| (format!("{name}-lf"), text.replace("\r\n", "\n")));
            std::iter::once((name, text)).chain(lf)
        })
        .collect();
    texts.sort();
    assert!(!texts.is_empty(), "no .txt in {}", directory.display());
    texts
}

/// The one file named after the model.
fn source_file(directory: &Path, id: &str) -> PathBuf {
    let files: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.file_stem().is_some_and(|stem| stem == id))
        .collect();
    match files.as_slice() {
        [file] => file.clone(),
        [] => panic!("no source for '{id}' in {}", directory.display()),
        _ => panic!("several sources for '{id}': {files:?}"),
    }
}

fn assert_matches_tiktoken(tokenizer: &Tokenizer, encoding: Encoding, name: &str, text: &str) {
    let reference = match encoding {
        Encoding::O200kBase => tiktoken_rs::o200k_base_singleton(),
        Encoding::Cl100kBase => tiktoken_rs::cl100k_base_singleton(),
        Encoding::R50kBase => tiktoken_rs::r50k_base_singleton(),
        other => panic!("no reference for {other:?}"),
    };
    assert!(
        tokenizer.encode(text).unwrap() == reference.encode_ordinary(text),
        "{} differs from tiktoken on {name}",
        tokenizer.model().id()
    );
}

/// The 95th percentile of the absolute errors on consecutive pieces of about [`CHUNK`] bytes.
fn chunk_p95(tokenizer: &Tokenizer, text: &str) -> f64 {
    let mut errors = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + CHUNK).min(text.len());
        while !text.is_char_boundary(end) {
            end += 1;
        }
        let chunk = &text[start..end];
        let exact = tokenizer.count_tokens(chunk).unwrap();
        errors.push(error(tokenizer.estimate_tokens(chunk), exact).abs());
        start = end;
    }
    errors.sort_by(f64::total_cmp);
    errors[errors.len() * 95 / 100]
}

/// The error of an estimate, in percent of the exact count.
fn error(estimate: usize, exact: usize) -> f64 {
    (estimate as f64 - exact as f64) / exact.max(1) as f64 * 100.0
}
