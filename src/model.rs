use std::borrow::Cow;
use std::fmt;

use crate::message::Message;

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
    /// An OpenAI encoding embedded in the crate (`openai` feature). A registered artifact takes its
    /// place.
    Builtin(Encoding),
    /// A source the caller registers: an artifact, or a source in one of these formats.
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

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TokenizerJson => "tokenizer.json",
            Self::SentencePiece => "SentencePiece model",
            Self::TiktokenRanks => "tiktoken rank file",
        })
    }
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

impl Framing {
    /// The tokens of `messages`, given the tokens of a role or a content.
    pub(crate) fn total<E>(
        self,
        messages: &[Message],
        mut count: impl FnMut(&str) -> Result<usize, E>,
    ) -> Result<usize, E> {
        match self {
            Self::ChatMl {
                per_message,
                per_reply,
            } => messages.iter().try_fold(per_reply, |total, message| {
                Ok(total + per_message + count(&message.role)? + count(&message.content)?)
            }),
            Self::ContentOnly => messages
                .iter()
                .try_fold(0, |total, message| Ok(total + count(&message.content)?)),
        }
    }
}

impl Model {
    /// Every model, in table order.
    pub fn all() -> &'static [Model] {
        MODELS
    }

    /// Resolves a model name such as `chatgpt-4o-latest` or `qwen3-max` to a model.
    ///
    /// The name is trimmed and compared case-insensitively, and a provider prefix such as `openai/`
    /// is dropped. A model id resolves to itself; any other name goes through the resolution rules,
    /// and the first rule that matches decides. Returns `None` when no rule matches; the caller
    /// chooses what to fall back to.
    pub fn resolve(name: &str) -> Option<&'static Model> {
        let name = name.trim();
        let name = name.rsplit_once('/').map_or(name, |(_, model)| model);
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

    /// Where the model's tokenizer comes from.
    pub fn source(&self) -> Source {
        self.source
    }

    /// How the model counts chat messages.
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

    const fn registered(id: &'static str, formats: &'static [Format], framing: Framing) -> Self {
        Self {
            id,
            source: Source::Registered(formats),
            framing,
        }
    }
}

const CHAT_ML: Framing = Framing::ChatMl {
    per_message: 3,
    per_reply: 3,
};
const CONTENT_ONLY: Framing = Framing::ContentOnly;
const JSON: &[Format] = &[Format::TokenizerJson];
const SENTENCEPIECE_OR_JSON: &[Format] = &[Format::SentencePiece, Format::TokenizerJson];
const TIKTOKEN_RANKS: &[Format] = &[Format::TiktokenRanks];

/// A static, so every model has one address; the registry finds a model's slot by it.
///
/// Each row is a distinct tokenizer, so no two rows share both source and framing. A built-in row
/// is named after one model that uses its encoding; the names of the others resolve to it.
static MODELS: &[Model] = &[
    Model::builtin("gpt-4o", Encoding::O200kBase, CHAT_ML),
    Model::builtin("gpt-4", Encoding::Cl100kBase, CHAT_ML),
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
    Model::registered("claude", JSON, CONTENT_ONLY),
    Model::registered("llama3", JSON, CONTENT_ONLY),
    Model::registered("command-r", JSON, CONTENT_ONLY),
    Model::registered("command-a", JSON, CONTENT_ONLY),
    Model::registered("qwen2", JSON, CONTENT_ONLY),
    Model::registered("qwen3.5", JSON, CONTENT_ONLY),
    Model::registered("nemo", JSON, CONTENT_ONLY),
    Model::registered("deepseek", JSON, CONTENT_ONLY),
    Model::registered("glm", JSON, CONTENT_ONLY),
    Model::registered("minimax", JSON, CONTENT_ONLY),
    // Kimi's chat template frames each message with three special tokens and primes the reply with
    // three tokens, as ChatML does.
    Model::registered("kimi", TIKTOKEN_RANKS, CHAT_ML),
    Model::registered("llama", SENTENCEPIECE_OR_JSON, CONTENT_ONLY),
    Model::registered("mistral", SENTENCEPIECE_OR_JSON, CONTENT_ONLY),
    Model::registered("yi", SENTENCEPIECE_OR_JSON, CONTENT_ONLY),
    Model::registered("gemma", SENTENCEPIECE_OR_JSON, CONTENT_ONLY),
    Model::registered("jamba", SENTENCEPIECE_OR_JSON, CONTENT_ONLY),
    Model::registered("nerdstash", SENTENCEPIECE_OR_JSON, CONTENT_ONLY),
    Model::registered("nerdstash_v2", SENTENCEPIECE_OR_JSON, CONTENT_ONLY),
];

/// A model id and a test of normalized names.
type Rule = (&'static str, fn(&str) -> bool);

/// Resolution rules for names that are not model ids, in priority order: the first rule that
/// matches the trimmed, lowercased name decides. A new rule goes before the first existing rule
/// that would wrongly match its names.
const RULES: &[Rule] = &[
    ("gpt2", |name| name == "gpt-2"),
    ("gpt-4o", |name| {
        ["o1", "o3", "o4", "gpt-5", "gpt-6"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            || (name.contains("gpt-4") && contains_any(name, &["4o", "4.1", "4.5"]))
    }),
    ("gpt-4", |name| {
        contains_any(name, &["gpt-4", "gpt-3.5", "gpt-35"])
    }),
    ("claude", |name| name.contains("claude")),
    ("kimi", |name| {
        name.contains("kimi") || name == "k3" || name.starts_with("k3-")
    }),
    ("llama3", |name| contains_any(name, &["llama3", "llama-3"])),
    ("llama", |name| name.contains("llama")),
    ("nemo", |name| contains_any(name, &["nemo", "pixtral"])),
    ("mistral", |name| {
        contains_any(name, &["mistral", "mixtral"])
    }),
    // Qwen 3.5 changed the vocabulary; Tongyi models are built on Qwen 3.
    ("qwen3.5", |name| {
        qwen_version(name).is_some_and(|version| version >= (3, 5))
    }),
    ("qwen2", |name| {
        contains_any(name, &["qwen", "qwq", "tongyi"])
    }),
    ("yi", |name| name.contains("yi")),
    // Gemini 3 counts exactly as Gemma 3; Gemini 4 publishes no tokenizer and counts within about
    // 1% of it on the public texts.
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
    // MAI-DS-R1, R1 1776 and Sonar Reasoning are built on DeepSeek R1.
    ("deepseek", |name| {
        contains_any(name, &["deepseek", "mai-ds", "r1-1776", "sonar-reasoning"])
    }),
    ("glm", |name| name.contains("glm")),
    ("minimax", |name| name.contains("minimax")),
    // xAI publishes no tokenizer for its current models. On the public texts Qwen2's counts are
    // within 3.1% of Grok 4.3 and 4.20 and 2.6% of Grok 4.5 to 4.7; o200k's are 16% off on Chinese.
    ("qwen2", |name| name.contains("grok")),
];

fn contains_any(name: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| name.contains(needle))
}

/// The version right after `qwen` in a name, such as `(3, 5)` for `qwen3.5-plus` and `(3, 0)` for
/// `qwen3-max`; `None` for an unversioned name such as `qwen-plus`.
fn qwen_version(name: &str) -> Option<(u32, u32)> {
    let (_, rest) = name.split_once("qwen")?;
    let (major, rest) = leading_number(rest)?;
    let minor = rest
        .strip_prefix('.')
        .and_then(leading_number)
        .map_or(0, |(minor, _)| minor);
    Some((major, minor))
}

/// The number a text starts with, and the rest of the text.
fn leading_number(text: &str) -> Option<(u32, &str)> {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    Some((text[..end].parse().ok()?, &text[end..]))
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
    fn every_rule_names_a_model() {
        for (id, _) in RULES {
            assert!(Model::all().iter().any(|model| model.id() == *id), "{id}");
        }
    }

    #[test]
    fn rules_map_names_to_models() {
        for (name, id) in [
            ("gpt-2", "gpt2"),
            ("o1", "gpt-4o"),
            ("o3-mini", "gpt-4o"),
            ("openai/gpt-5", "gpt-4o"),
            ("gpt-5.2", "gpt-4o"),
            ("gpt-6-astra", "gpt-4o"),
            ("chatgpt-4o-latest", "gpt-4o"),
            ("gpt-4.1-mini", "gpt-4o"),
            ("gpt-4.5-preview", "gpt-4o"),
            ("gpt-4-turbo", "gpt-4"),
            ("gpt-4-32k", "gpt-4"),
            ("gpt-3.5-turbo", "gpt-4"),
            ("gpt-3.5-turbo-1106", "gpt-4"),
            ("gpt-35-turbo", "gpt-4"),
            ("claude-3-5-sonnet", "claude"),
            ("moonshotai/kimi-k2.6", "kimi"),
            ("k3-256k", "kimi"),
            ("llama-2-7b", "llama"),
            ("mistral-large", "mistral"),
            ("yi-large", "yi"),
            ("gemini-3.8-flash", "gemma"),
            ("models/gemini-3.1-pro-preview", "gemma"),
            ("learnlm-pro", "gemma"),
            ("jamba-1.5-large", "jamba"),
            ("nerdstash-v1", "nerdstash"),
            ("command-r-plus", "command-r"),
            ("command-a-03-2025", "command-a"),
            ("qwen3-max", "qwen2"),
            ("qwen-plus", "qwen2"),
            ("qwen2.5-72b-instruct", "qwen2"),
            ("qwq-32b", "qwen2"),
            ("qwen3.5-plus", "qwen3.5"),
            ("Qwen/Qwen3.5-397B-A17B", "qwen3.5"),
            ("qwen3.8-max", "qwen3.5"),
            ("z-ai/glm-5.3", "glm"),
            ("glm-4.5-air", "glm"),
            ("MiniMax-M3", "minimax"),
            ("minimax/minimax-m3.1-flash", "minimax"),
            ("pixtral-large-latest", "nemo"),
            ("mixtral-8x7b-instruct", "mistral"),
            ("mai-ds-r1", "deepseek"),
            ("r1-1776", "deepseek"),
            ("sonar-reasoning-pro", "deepseek"),
            ("x-ai/grok-4.7", "qwen2"),
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
            ("llama-3.3-70b", "llama3"),
            ("nerdstash-v2", "nerdstash_v2"),
            ("open-mistral-nemo", "nemo"),
            ("deepseek-r1-distill-qwen-32b", "qwen2"),
            ("tongyi-deepresearch-30b-a3b", "qwen2"),
        ] {
            assert_eq!(resolved_id(name), Some(id), "{name}");
        }
    }

    #[test]
    fn unknown_names_resolve_to_nothing() {
        for name in ["", "my-unknown-model", "work3"] {
            assert_eq!(resolved_id(name), None, "{name}");
        }
    }
}
