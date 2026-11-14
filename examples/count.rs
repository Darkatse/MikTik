//! Exact and estimated token counts of text files.
//!
//! ```text
//! cargo run --release --all-features --example count -- <model> [--source <file>] <text file>…
//! ```
//!
//! Prints one line per text file: the exact count, the estimate and the file. `--source` registers
//! a source for the model first.

use miktik::TokenizerRegistry;

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = || -> ! {
        eprintln!("usage: count <model> [--source <file>] <text file>…");
        std::process::exit(2);
    };
    let model = args.next().unwrap_or_else(|| usage());
    let mut files: Vec<String> = args.collect();
    let registry = TokenizerRegistry::new();
    if files.first().is_some_and(|arg| arg == "--source") {
        let source = files.get(1).cloned().unwrap_or_else(|| usage());
        registry.register_model_file(&model, source).unwrap();
        files.drain(..2);
    }
    if files.is_empty() {
        usage();
    }
    let tokenizer = registry.get(&model).unwrap();
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        let exact = tokenizer.count_tokens(&text).unwrap();
        println!("{exact} {} {file}", tokenizer.estimate_tokens(&text));
    }
}
