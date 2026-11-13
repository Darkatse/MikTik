use std::borrow::Cow;

/// A tokenizer identity, such as `gpt-4o`, `claude` or `gemma`: one row of MikTik's model table.
///
/// Models exist only as `&'static Model`, obtained from [`Model::all`] or [`Model::resolve`].
#[derive(Debug, PartialEq, Eq)]
pub struct Model {
    id: &'static str,
    source: Source,
    framing: Framing,
}

/// Where a model's tokenizer comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// An OpenAI encoding embedded in the crate (`openai` feature).
    Builtin(Encoding),
    /// A source the caller registers, in one of these formats.
    Registered(&'static [Format]),
}

/// An OpenAI encoding embedded in the crate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    O200kBase,
    Cl100kBase,
    R50kBase,
}

impl Encoding {
    /// The tiktoken name of the encoding, such as `o200k_base`.
    pub fn name(self) -> &'static str {
        match self {
            Self::O200kBase => "o200k_base",
            Self::Cl100kBase => "cl100k_base",
            Self::R50kBase => "r50k_base",
        }
    }
}

/// A tokenizer source format.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// A Hugging Face `tokenizer.json`.
    TokenizerJson,
    /// A serialized SentencePiece model (`tokenizer.model`).
    SentencePiece,
    /// A tiktoken rank file: one base64 token and its rank per line.
    TiktokenRanks,
}

/// How chat messages are counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// ChatML: a fixed number of tokens for each message and for the reply primer, on top of each
    /// message's role and content.
    ChatMl {
        per_message: usize,
        per_reply: usize,
    },
    /// Only message content counts.
    ContentOnly,
}

impl Model {
    /// Every model, in table order.
    pub fn all() -> &'static [Model] {
        MODELS
    }

    /// Resolves a model name such as `chatgpt-4o-latest` or `qwen3-max` to a model.
    ///
    /// The name is trimmed and compared case-insensitively. A model id resolves to itself; any other
    /// name goes through the resolution rules, and the first rule that matches decides. Returns
    /// `None` when no rule matches; the caller chooses what to fall back to.
    pub fn resolve(name: &str) -> Option<&'static Model> {
        let name = name.trim();
        let name = if name.bytes().any(|byte| byte.is_ascii_uppercase()) {
            Cow::Owned(name.to_ascii_lowercase())
        } else {
            Cow::Borrowed(name)
        };
        if let Some(model) = MODELS.iter().find(|model| model.id == name) {
            return Some(model);
        }
        let (id, _) = RULES.iter().find(|(_, matches)| matches(&name))?;
        MODELS.iter().find(|model| model.id == *id)
    }

    /// The model id, such as `gpt-4o`.
    pub fn id(&self) -> &'static str {
        self.id
    }

    pub fn source(&self) -> Source {
        self.source
    }

    pub fn framing(&self) -> Framing {
        self.framing
    }

    const fn builtin(id: &'static str, encoding: Encoding, framing: Framing) -> Self {
        Self {
            id,
            source: Source::Builtin(encoding),
            framing,
        }
    }

    const fn registered(id: &'static str, formats: &'static [Format]) -> Self {
        Self {
            id,
            source: Source::Registered(formats),
            framing: Framing::ContentOnly,
        }
    }
}

const CHAT_ML: Framing = Framing::ChatMl {
    per_message: 3,
    per_reply: 3,
};
const JSON: &[Format] = &[Format::TokenizerJson];
const SENTENCEPIECE_OR_JSON: &[Format] = &[Format::SentencePiece, Format::TokenizerJson];
const TIKTOKEN_RANKS: &[Format] = &[Format::TiktokenRanks];

const MODELS: &[Model] = &[
    Model::builtin("o1", Encoding::O200kBase, CHAT_ML),
    Model::builtin("gpt-4o", Encoding::O200kBase, CHAT_ML),
    Model::builtin("gpt-4", Encoding::Cl100kBase, CHAT_ML),
    Model::builtin("gpt-4-32k", Encoding::Cl100kBase, CHAT_ML),
    Model::builtin("gpt-3.5-turbo", Encoding::Cl100kBase, CHAT_ML),
    // This snapshot framed each message with one more token.
    Model::builtin(
        "gpt-3.5-turbo-0301",
        Encoding::Cl100kBase,
        Framing::ChatMl {
            per_message: 4,
            per_reply: 3,
        },
    ),
    Model::builtin("gpt2", Encoding::R50kBase, CHAT_ML),
    Model::registered("claude", JSON),
    Model::registered("llama3", JSON),
    Model::registered("command-r", JSON),
    Model::registered("command-a", JSON),
    Model::registered("qwen2", JSON),
    Model::registered("nemo", JSON),
    Model::registered("deepseek", JSON),
    Model::registered("kimi", TIKTOKEN_RANKS),
    Model::registered("llama", SENTENCEPIECE_OR_JSON),
    Model::registered("mistral", SENTENCEPIECE_OR_JSON),
    Model::registered("yi", SENTENCEPIECE_OR_JSON),
    Model::registered("gemma", SENTENCEPIECE_OR_JSON),
    Model::registered("jamba", SENTENCEPIECE_OR_JSON),
    Model::registered("nerdstash", SENTENCEPIECE_OR_JSON),
    Model::registered("nerdstash_v2", SENTENCEPIECE_OR_JSON),
];

/// A model id and a test of normalized names.
type Rule = (&'static str, fn(&str) -> bool);

/// Resolution rules for names that are not model ids, in priority order: the first rule that
/// matches the trimmed, lowercased name decides. A new rule goes before the first existing rule
/// that would wrongly match its names.
const RULES: &[Rule] = &[
    ("gpt2", |name| name == "gpt-2"),
    ("o1", |name| {
        ["o1", "o3", "o4", "gpt-5", "gpt-6"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
    }),
    ("gpt-4-32k", |name| is_gpt_4(name) && name.contains("32k")),
    ("gpt-4o", |name| {
        is_gpt_4(name) && contains_any(name, &["4o", "4.1", "4.5"])
    }),
    ("gpt-4", is_gpt_4),
    ("gpt-3.5-turbo", |name| {
        contains_any(name, &["gpt-3.5", "gpt-35"])
    }),
    ("claude", |name| name.contains("claude")),
    ("kimi", |name| {
        name.contains("kimi") || name == "k3" || name.starts_with("k3-")
    }),
    ("llama3", |name| contains_any(name, &["llama3", "llama-3"])),
    ("llama", |name| name.contains("llama")),
    ("mistral", |name| name.contains("mistral")),
    ("yi", |name| name.contains("yi")),
    ("gemma", |name| {
        contains_any(name, &["gemma", "gemini", "learnlm"])
    }),
    ("jamba", |name| name.contains("jamba")),
    ("nerdstash_v2", |name| {
        contains_any(name, &["nerdstash-v2", "nerdstash_v2"])
    }),
    ("nerdstash", |name| name.contains("nerdstash")),
    ("command-r", |name| name.contains("command-r")),
    ("command-a", |name| name.contains("command-a")),
    ("qwen2", |name| contains_any(name, &["qwen", "qwq"])),
    ("nemo", |name| name.contains("nemo")),
    ("deepseek", |name| name.contains("deepseek")),
];

fn is_gpt_4(name: &str) -> bool {
    contains_any(name, &["gpt-4", "chatgpt-4o"])
}

fn contains_any(name: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| name.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved_id(name: &str) -> Option<&'static str> {
        Model::resolve(name).map(Model::id)
    }

    #[test]
    fn every_id_resolves_to_its_own_model() {
        for model in Model::all() {
            assert_eq!(Model::resolve(model.id()), Some(model));
        }
    }

    #[test]
    fn rules_map_names_to_models() {
        for (name, id) in [
            ("gpt-2", "gpt2"),
            ("o3-mini", "o1"),
            ("gpt-5.2", "o1"),
            ("gpt-6-astra", "o1"),
            ("chatgpt-4o-latest", "gpt-4o"),
            ("gpt-4.1-mini", "gpt-4o"),
            ("gpt-4.5-preview", "gpt-4o"),
            ("gpt-4-turbo", "gpt-4"),
            ("gpt-3.5-turbo-1106", "gpt-3.5-turbo"),
            ("gpt-35-turbo", "gpt-3.5-turbo"),
            ("claude-3-5-sonnet", "claude"),
            ("moonshotai/kimi-k2.6", "kimi"),
            ("k3-256k", "kimi"),
            ("llama-2-7b", "llama"),
            ("mistral-large", "mistral"),
            ("yi-large", "yi"),
            ("gemini-3.8-flash", "gemma"),
            ("learnlm-pro", "gemma"),
            ("jamba-1.5-large", "jamba"),
            ("nerdstash-v1", "nerdstash"),
            ("command-r-plus", "command-r"),
            ("command-a-03-2025", "command-a"),
            ("qwen3-max", "qwen2"),
            ("qwq-32b", "qwen2"),
            ("nemo-12b", "nemo"),
            ("deepseek-chat", "deepseek"),
            ("  GPT-4O  ", "gpt-4o"),
        ] {
            assert_eq!(resolved_id(name), Some(id), "{name}");
        }
    }

    #[test]
    fn earlier_rules_win() {
        for (name, id) in [
            ("gpt-4o-32k", "gpt-4-32k"),
            ("llama-3.3-70b", "llama3"),
            ("nerdstash-v2", "nerdstash_v2"),
            ("mistral-nemo", "mistral"),
            ("deepseek-r1-distill-qwen-32b", "qwen2"),
        ] {
            assert_eq!(resolved_id(name), Some(id), "{name}");
        }
    }

    #[test]
    fn unknown_names_resolve_to_nothing() {
        for name in ["", "my-unknown-model", "glm-5", "grok3", "work3"] {
            assert_eq!(resolved_id(name), None, "{name}");
        }
    }
}
