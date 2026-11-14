//! Estimates of long texts on several threads. The text is grouped at cuts, so the sum of the
//! groups' estimates is the estimate of the text.

use rayon::prelude::*;

use super::next_cut;

/// Texts at least this long are estimated in groups.
pub(super) const MIN_LEN: usize = 256 * 1024;

/// A group ends at the first cut after this many bytes.
const GROUP_LEN: usize = 64 * 1024;

/// The sum of `estimate` over the groups of `text`, on the threads of the current rayon pool, or of
/// the global pool outside one.
pub(super) fn sum(text: &str, estimate: impl Fn(&str) -> usize + Send + Sync) -> usize {
    let mut groups = Vec::with_capacity(text.len() / GROUP_LEN + 1);
    let mut start = 0;
    while start < text.len() {
        let end = if start + GROUP_LEN >= text.len() {
            text.len()
        } else {
            next_cut(text, text.ceil_char_boundary(start + GROUP_LEN)).unwrap_or(text.len())
        };
        groups.push(&text[start..end]);
        start = end;
    }
    groups.into_par_iter().map(estimate).sum()
}
