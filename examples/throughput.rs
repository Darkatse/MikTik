//! Load time and throughput of each model, exact and estimated.
//!
//! ```text
//! cargo run --release --all-features --example throughput -- <model dir> <corpus dir> [model id…]
//! ```
//!
//! The directories are laid out as `tests/corpus.rs` describes; `scripts/fetch-test-assets.sh`
//! fills them. Each figure is the best of several runs on one thread. Pin the process to one kind of
//! core (`taskset` on Android) to measure that core. `cached` is how much faster the estimate of the
//! first 100 KiB of the text is when the segment cache holds all its segments. With the `parallel`
//! feature, `threads` is how much faster the estimate of the whole text is on the threads of
//! rayon's global pool, one per core the process may use.

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
    #[cfg(feature = "parallel")]
    println!("rayon threads: {}", rayon::current_num_threads());

    println!(
        "{:<18} {:>8}  {:<6} {:>12} {:>14} {:>8} {:>8} {:>8}",
        "model", "load ms", "text", "exact MB/s", "estimate MB/s", "speedup", "cached", "threads"
    );
    let registry = TokenizerRegistry::new();
    let cached_registry = TokenizerRegistry::with_segment_cache(1 << 30);
    for model in Model::all() {
        if !only.is_empty() && !only.iter().any(|id| id == model.id()) {
            continue;
        }
        if let Source::Registered(_) = model.source() {
            let Some(source) = source_file(Path::new(models), model.id()) else {
                println!("{:<18} no source", model.id());
                continue;
            };
            registry.register_model_file(model.id(), &source).unwrap();
            cached_registry
                .register_model_file(model.id(), source)
                .unwrap();
        }
        let (tokenizer, load) = timed(|| registry.get(model.id()).unwrap());
        let cached = cached_registry.get(model.id()).unwrap();
        for (name, text) in &texts {
            let exact = best(3, || tokenizer.count_tokens(text).unwrap());
            let estimate = best(5, || one_thread(|| tokenizer.estimate_tokens(text)));
            let head = head(text, 100 * 1024);
            let miss = best(5, || tokenizer.estimate_tokens(head));
            cached.estimate_tokens(head);
            let hit = best(5, || cached.estimate_tokens(head));
            let threads = best(5, || tokenizer.estimate_tokens(text));
            println!(
                "{:<18} {:>8.1}  {name:<6} {:>12.1} {:>14.1} {:>7.1}x {:>7.1}x {:>7.1}x",
                model.id(),
                millis(load),
                megabytes_per_second(text, exact),
                megabytes_per_second(text, estimate),
                exact.as_secs_f64() / estimate.as_secs_f64(),
                miss.as_secs_f64() / hit.as_secs_f64(),
                estimate.as_secs_f64() / threads.as_secs_f64(),
            );
        }
    }
}

/// Runs `run` on one thread, where an estimate of a long text would otherwise use several.
#[cfg(feature = "parallel")]
fn one_thread<T: Send>(run: impl FnOnce() -> T + Send) -> T {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
    })
    .install(run)
}

#[cfg(not(feature = "parallel"))]
fn one_thread<T>(run: impl FnOnce() -> T) -> T {
    run()
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

/// The longest prefix of `text` of at most `len` bytes.
fn head(text: &str, len: usize) -> &str {
    &text[..text.floor_char_boundary(len.min(text.len()))]
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
