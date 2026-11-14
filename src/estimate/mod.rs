//! Estimates. A text is cut into segments after each newline that is followed by something other
//! than whitespace. A segment is folded as the tokenizer normalizes it, then covered from its start:
//! each step takes the longest piece the rest of the segment starts with, or one byte where no piece
//! does, and counts one token. A text's estimate is the sum of its segments' estimates, so it
//! depends only on the text and adds up across cuts.

mod cache;
#[cfg(feature = "parallel")]
mod parallel;
mod trie;

use std::borrow::Cow;

use kitoken::{Definition, Normalization, SpecialTokenKinds, UnicodeNormalization};

use cache::SegmentCache;
use trie::Trie;

/// The estimate side of a tokenizer, derived from its definition.
pub(crate) struct Estimator {
    trie: Trie,
    /// Full-width ASCII forms and the ideographic space are matched as the ASCII that NFKC writes
    /// for them.
    fold: bool,
    /// The split starts a part at the last newline of a whitespace run that other text follows.
    split_before_last_newline: bool,
    /// Estimates of recent segments, when the registry keeps them.
    cache: Option<SegmentCache>,
}

/// The character SentencePiece writes for a space.
const SPACE_MARK: &str = "\u{2581}";

impl Estimator {
    /// The pieces are the vocabulary and the special tokens that `count_tokens` recognizes
    /// (`specials_in_text`), spelled as the text they match:
    ///
    /// - A definition whose normalization replaces spaces with `▁` spells spaces that way, so `▁`
    ///   in a piece stands for a space.
    /// - A definition that normalizes to NFKC or NFKD matches full-width forms as ASCII, so the
    ///   text is folded the same way first.
    /// - A definition whose split starts a part at the last newline of a whitespace run that other
    ///   text follows (GPT-2's pattern) never joins that newline with the whitespace before it.
    ///
    /// Other normalization and splitting are not reproduced.
    pub(crate) fn from_definition(
        definition: &Definition,
        specials_in_text: bool,
    ) -> Result<Self, String> {
        let normalization = &definition.config.normalization;
        let marked = normalization.iter().any(|rule| {
            matches!(rule, Normalization::Replace { replacement, .. } if replacement == SPACE_MARK)
        });
        let fold = normalization.iter().any(|rule| {
            matches!(
                rule,
                Normalization::Unicode {
                    scheme: UnicodeNormalization::NFKC | UnicodeNormalization::NFKD
                }
            )
        });

        // GPT-2's pattern starts a part at the second newline here; patterns that keep runs of
        // newlines together do not.
        let split_before_last_newline = definition
            .config
            .split("a\n\nb")
            .iter()
            .any(|&(start, _)| start == 2);

        let kinds = specials_in_text.as_kinds(&definition.meta);
        let specials = definition
            .specials
            .iter()
            .filter(|special| kinds.contains(&special.kind))
            .map(|special| special.bytes.as_slice());
        let mut pieces: Vec<Cow<'_, [u8]>> = definition
            .model
            .vocab()
            .iter()
            .map(|token| token.bytes.as_slice())
            .chain(specials)
            .filter(|piece| !piece.is_empty())
            .map(|piece| {
                if marked {
                    unmark(piece)
                } else {
                    Cow::Borrowed(piece)
                }
            })
            .collect();
        pieces.sort_unstable();
        pieces.dedup();
        Ok(Self {
            trie: Trie::build(&pieces)?,
            fold,
            split_before_last_newline,
            cache: None,
        })
    }

    /// Keeps estimates of up to `capacity` bytes of recent segments; 0 keeps none.
    pub(crate) fn with_segment_cache(self, capacity: usize) -> Self {
        Self {
            cache: (capacity > 0).then(|| SegmentCache::new(capacity)),
            ..self
        }
    }

    /// The estimate of `text`.
    pub(crate) fn estimate(&self, text: &str) -> usize {
        #[cfg(feature = "parallel")]
        if text.len() >= parallel::MIN_LEN {
            return parallel::sum(text, |group| self.serial(group));
        }
        self.serial(text)
    }

    fn serial(&self, text: &str) -> usize {
        let mut folded = String::new();
        segments(text)
            .map(|segment| self.segment(segment, &mut folded))
            .sum()
    }

    /// The estimates of `base + additions[..=i]` for each `i`, each equal to [`estimate`] of that
    /// text. Once an estimate reaches `stop_at`, it fills the remaining places.
    ///
    /// [`estimate`]: Self::estimate
    pub(crate) fn cumulative(
        &self,
        base: &str,
        additions: &[impl AsRef<str>],
        stop_at: Option<usize>,
    ) -> Vec<usize> {
        let added: usize = additions
            .iter()
            .map(|addition| addition.as_ref().len())
            .sum();
        let mut text = String::with_capacity(base.len() + added);
        text.push_str(base);
        // The estimate of the segments before `open`, where the last segment starts.
        let mut closed = 0;
        let mut open = 0;
        // Cuts before this position are counted in `closed`.
        let mut from = 0;
        let mut estimates = Vec::with_capacity(additions.len());
        let mut folded = String::new();
        for addition in additions {
            text.push_str(addition.as_ref());
            while let Some(cut) = next_cut(&text, from) {
                closed += self.segment(&text[open..cut], &mut folded);
                (open, from) = (cut, cut);
            }
            // A newline that ends the text becomes a cut when the next addition starts with
            // something other than whitespace.
            from = if text.ends_with('\n') {
                text.len() - 1
            } else {
                text.len()
            };

            let estimate = closed + self.segment(&text[open..], &mut folded);
            estimates.push(estimate);
            if stop_at.is_some_and(|limit| estimate >= limit) {
                estimates.resize(additions.len(), estimate);
                break;
            }
        }
        estimates
    }

    /// The estimate of a segment. `folded` is room for the folded segment, reused from segment to
    /// segment.
    fn segment(&self, segment: &str, folded: &mut String) -> usize {
        match &self.cache {
            Some(cache) => cache.get_or(segment, || self.cover_segment(segment, folded)),
            None => self.cover_segment(segment, folded),
        }
    }

    fn cover_segment(&self, segment: &str, folded: &mut String) -> usize {
        let segment = if self.fold {
            fold_full_width(segment, folded)
        } else {
            segment
        };
        let bytes = segment.as_bytes();
        match bytes {
            // Other text follows the segment, unless it ends the text, so the newline is a token of
            // its own: whatever starts the next segment rarely forms one token with it.
            [.., space, b'\n'] if self.split_before_last_newline && space.is_ascii_whitespace() => {
                self.cover(&bytes[..bytes.len() - 1]) + 1
            }
            _ => self.cover(bytes),
        }
    }

    fn cover(&self, mut text: &[u8]) -> usize {
        let mut tokens = 0;
        while !text.is_empty() {
            text = &text[self.trie.longest(text).max(1)..];
            tokens += 1;
        }
        tokens
    }
}

/// The segments of `text`, in order.
fn segments(text: &str) -> impl Iterator<Item = &str> {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start == text.len() {
            return None;
        }
        let end = next_cut(text, start).unwrap_or(text.len());
        let segment = &text[start..end];
        start = end;
        Some(segment)
    })
}

/// The first cut after `from`: a position after a newline and before a character other than
/// whitespace. `from` is a character boundary.
fn next_cut(text: &str, mut from: usize) -> Option<usize> {
    loop {
        let cut = from + text[from..].find('\n')? + 1;
        let next = match *text.as_bytes().get(cut)? {
            byte if byte.is_ascii() => char::from(byte),
            _ => text[cut..].chars().next()?,
        };
        if !next.is_whitespace() {
            return Some(cut);
        }
        from = cut;
    }
}

/// `piece` with each `▁` written as a space.
fn unmark(piece: &[u8]) -> Cow<'_, [u8]> {
    let mark = SPACE_MARK.as_bytes();
    if !piece.windows(mark.len()).any(|window| window == mark) {
        return Cow::Borrowed(piece);
    }
    let mut unmarked = Vec::with_capacity(piece.len());
    let mut rest = piece;
    while let Some((&byte, after)) = rest.split_first() {
        if let Some(after_mark) = rest.strip_prefix(mark) {
            unmarked.push(b' ');
            rest = after_mark;
        } else {
            unmarked.push(byte);
            rest = after;
        }
    }
    Cow::Owned(unmarked)
}

/// `text` with full-width ASCII forms and the ideographic space written as ASCII, as NFKC does,
/// written into `folded` when anything changes. Whitespace stays whitespace and other characters
/// stay other characters, so folding moves no cut.
fn fold_full_width<'a>(text: &'a str, folded: &'a mut String) -> &'a str {
    let ascii = |c: char| match c {
        '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0),
        '\u{3000}' => Some(' '),
        _ => None,
    };
    // In UTF-8 these characters start with 0xEF or 0xE3, which only ever start three-byte
    // characters, so the text between them is copied as it is.
    let bytes = text.as_bytes();
    folded.clear();
    let mut copied = 0;
    let mut at = 0;
    while let Some(offset) = bytes[at..]
        .iter()
        .position(|&byte| byte == 0xEF || byte == 0xE3)
    {
        at += offset;
        if let Some(ascii) = text[at..].chars().next().and_then(ascii) {
            folded.push_str(&text[copied..at]);
            folded.push(ascii);
            copied = at + 3;
        }
        at += 3;
    }
    if copied == 0 {
        return text;
    }
    folded.push_str(&text[copied..]);
    folded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_start_after_a_newline_and_before_other_than_whitespace() {
        let text = "a\nb\n\nc\n d\r\ne\n\u{3000}f\n";
        let cut: Vec<_> = segments(text).collect();
        assert_eq!(cut, ["a\n", "b\n\n", "c\n d\r\n", "e\n\u{3000}f\n"]);
        assert_eq!(segments("").count(), 0);
    }
}
