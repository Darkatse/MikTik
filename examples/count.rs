//! Exact and estimated token counts of text files and of chat messages.
//!
//! ```text
//! cargo run --release --all-features --example count -- <model> [--source <file>] <file>…
//! ```
//!
//! Prints one line per file: the exact count, the estimate and the file. A `.json` file holds a
//! list of messages, `[{"role": "user", "content": "…"}, …]`, counted with the model's framing;
//! any other file is text. `--source` registers a source for the model first.

use miktik::{Message, TokenizerRegistry};

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = || -> ! {
        eprintln!("usage: count <model> [--source <file>] <file>…");
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
        let (exact, estimate) = if file.ends_with(".json") {
            let messages = messages(&text);
            (
                tokenizer.count_messages(&messages).unwrap(),
                tokenizer.estimate_messages(&messages),
            )
        } else {
            (
                tokenizer.count_tokens(&text).unwrap(),
                tokenizer.estimate_tokens(&text),
            )
        };
        println!("{exact} {estimate} {file}");
    }
}

fn messages(json: &str) -> Vec<Message> {
    let values: Vec<serde_json::Value> = serde_json::from_str(json).unwrap();
    values
        .iter()
        .map(|value| {
            let field = |key: &str| value[key].as_str().unwrap().to_owned();
            Message::new(field("role"), field("content"))
        })
        .collect()
}
