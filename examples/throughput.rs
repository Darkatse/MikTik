//! Load time and throughput of each model, exact and estimated.
//!
//! ```text
//! cargo run --release --all-features --example throughput -- <model dir> <corpus dir> [model id…]
//! ```
//!
//! The directories are laid out as `tests/corpus.rs` describes; `scripts/fetch-test-assets.sh`
//! fills them. Each figure is the best of several runs on one thread. Pin the process to one kind of
//! core (`taskset` on Android) to measure that core.

use std::path::Path;
use std::time::{Duration, Instant};

use miktik::{Model, Source, TokenizerRegistry};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [models, corpora, only @ ..] = args.as_slice() else {
        eprintln!("usage: throughput <model dir> <corpus dir> [model id…]");
        std::process::exit(2);
    };
    let texts = texts(Path::new(corpora));

    println!(
        "{:<18} {:>8}  {:<6} {:>12} {:>14} {:>8}",
        "model", "load ms", "text", "exact MB/s", "estimate MB/s", "speedup"
    );
    let registry = TokenizerRegistry::new();
    for model in Model::all() {
        if !only.is_empty() && !only.iter().any(|id| id == model.id()) {
            continue;
        }
        if let Source::Registered(_) = model.source() {
            let Some(source) = source_file(Path::new(models), model.id()) else {
                println!("{:<18} no source", model.id());
                continue;
            };
            registry.register_model_file(model.id(), source).unwrap();
        }
        let (tokenizer, load) = timed(|| registry.get(model.id()).unwrap());
        for (name, text) in &texts {
            let exact = best(3, || tokenizer.count_tokens(text).unwrap());
            let estimate = best(5, || tokenizer.estimate_tokens(text));
            println!(
                "{:<18} {:>8.1}  {name:<6} {:>12.1} {:>14.1} {:>7.1}x",
                model.id(),
                millis(load),
                megabytes_per_second(text, exact),
                megabytes_per_second(text, estimate),
                exact.as_secs_f64() / estimate.as_secs_f64(),
            );
        }
    }
}

/// Every `.txt` in the directory, by file stem.
fn texts(directory: &Path) -> Vec<(String, String)> {
    let mut texts: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "txt"))
        .map(|path| {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            (name, std::fs::read_to_string(&path).unwrap())
        })
        .collect();
    texts.sort();
    texts
}

fn source_file(directory: &Path, id: &str) -> Option<std::path::PathBuf> {
    std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.file_stem().is_some_and(|stem| stem == id))
}

fn timed<T>(run: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let value = run();
    (value, start.elapsed())
}

fn best<T>(runs: usize, mut run: impl FnMut() -> T) -> Duration {
    (0..runs)
        .map(|_| timed(|| std::hint::black_box(run())).1)
        .min()
        .unwrap()
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn megabytes_per_second(text: &str, duration: Duration) -> f64 {
    text.len() as f64 / 1e6 / duration.as_secs_f64()
}
