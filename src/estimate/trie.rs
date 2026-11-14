//! A byte-wise double-array trie of pieces: the longest piece a text starts with, in one pass over
//! the text.
//!
//! Every lookup is a checked `get`, so a walk never panics, whatever the units hold.

/// One state. Its children on bytes `b` are the units `base + b` whose `check` is the state's index
/// plus one; the root is unit 0, and free units have `check` 0.
#[derive(Clone, Copy, Default)]
struct Unit {
    /// Bits 0–30: where the children start. Bit 31: a piece ends at this state.
    base: u32,
    check: u32,
}

const END: u32 = 1 << 31;
const BASE: u32 = END - 1;

pub(crate) struct Trie {
    units: Box<[Unit]>,
}

impl Trie {
    /// The trie of `pieces`, which must be sorted and unique; empty pieces are never matched. Fails
    /// only when the pieces need more units than a base can address.
    pub(crate) fn build(pieces: &[impl AsRef<[u8]>]) -> Result<Self, String> {
        let mut units = vec![Unit::default(); 256];
        // Free units at and after this index; every unit before it is in use.
        let mut first_free = 1;
        // Depth first, so a state's children sit near the states placed just before them. Each
        // entry is a state, its pieces and how many of their bytes lead to the state.
        let mut pending = vec![(0, 0..pieces.len(), 0)];
        let mut labels = Vec::new();
        while let Some((state, mut range, depth)) = pending.pop() {
            // Sorted, so a piece that ends here comes first.
            if !range.is_empty() && pieces[range.start].as_ref().len() == depth {
                units[state].base |= END;
                range.start += 1;
            }
            if range.is_empty() {
                continue;
            }
            // The children: one per distinct next byte, each with the run of pieces under it.
            labels.clear();
            let mut start = range.start;
            for at in range.start + 1..=range.end {
                let label = pieces[start].as_ref()[depth];
                if at == range.end || pieces[at].as_ref()[depth] != label {
                    labels.push((label, start..at));
                    start = at;
                }
            }

            while units[first_free].check != 0 {
                first_free += 1;
                grow(&mut units, first_free + 1);
            }
            // The first base, from the first free unit up, whose child units are all free.
            let first = labels[0].0 as usize;
            let mut base = first_free.saturating_sub(first);
            loop {
                grow(&mut units, base + 256);
                let free = |&(label, _): &(u8, _)| {
                    let unit = base + label as usize;
                    unit != 0 && units[unit].check == 0
                };
                if labels.iter().all(free) {
                    break;
                }
                base += 1;
            }
            units[state].base |= u32::try_from(base)
                .ok()
                .filter(|&base| base <= BASE)
                .ok_or_else(|| "the vocabulary has too many pieces to estimate with".to_string())?;
            // Children are popped in label order.
            for (label, range) in labels.drain(..).rev() {
                let child = base + label as usize;
                // Below 2^31 + 256, so the index plus one fits.
                units[child].check = state as u32 + 1;
                pending.push((child, range, depth + 1));
            }
        }
        let used = units
            .iter()
            .rposition(|unit| unit.check != 0)
            .map_or(1, |last| last + 1);
        units.truncate(used);
        Ok(Self {
            units: units.into_boxed_slice(),
        })
    }

    /// The length of the longest piece `text` starts with; 0 when none does.
    #[inline]
    pub(crate) fn longest(&self, text: &[u8]) -> usize {
        let mut longest = 0;
        let mut state = 0;
        let mut unit = self.units.first().copied().unwrap_or_default();
        for (at, &byte) in text.iter().enumerate() {
            let next = (unit.base & BASE) as usize + byte as usize;
            match self.units.get(next) {
                Some(&child) if child.check as usize == state + 1 => {
                    state = next;
                    unit = child;
                    if unit.base & END != 0 {
                        longest = at + 1;
                    }
                }
                _ => break,
            }
        }
        longest
    }
}

fn grow(units: &mut Vec<Unit>, len: usize) {
    if units.len() < len {
        units.resize(len.max(units.len() * 2), Unit::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator, so failures reproduce.
    struct Random(u64);

    impl Random {
        fn below(&mut self, bound: usize) -> usize {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % bound
        }

        /// Bytes from a small alphabet, so pieces share prefixes, plus the extremes 0 and 255.
        fn bytes(&mut self, max_len: usize) -> Vec<u8> {
            const ALPHABET: &[u8] = b"\0ab\xff";
            let len = self.below(max_len) + 1;
            (0..len)
                .map(|_| ALPHABET[self.below(ALPHABET.len())])
                .collect()
        }
    }

    fn naive_longest(pieces: &[Vec<u8>], text: &[u8]) -> usize {
        pieces
            .iter()
            .filter(|piece| text.starts_with(piece))
            .map(Vec::len)
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn finds_the_longest_piece_a_text_starts_with() {
        let mut random = Random(0x9E37_79B9_7F4A_7C15);
        for _ in 0..200 {
            let count = random.below(60) + 1;
            let mut pieces: Vec<Vec<u8>> = (0..count).map(|_| random.bytes(6)).collect();
            pieces.sort();
            pieces.dedup();
            let trie = Trie::build(&pieces).unwrap();
            for _ in 0..50 {
                let text = random.bytes(10);
                assert_eq!(
                    trie.longest(&text),
                    naive_longest(&pieces, &text),
                    "{pieces:?} {text:?}"
                );
            }
        }
    }
}
