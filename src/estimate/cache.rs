//! Estimates of recent segments, keyed by the segment's text, so a hit is always the same segment
//! and a cached estimate equals the computed one.

use std::hash::BuildHasher;
use std::sync::{Mutex, MutexGuard, PoisonError};

use foldhash::fast::FixedState;
use hashbrown::HashTable;

/// Shards, each behind its own lock, so threads estimating different segments rarely wait.
const SHARDS: usize = 16;

pub(crate) struct SegmentCache {
    shards: Box<[Mutex<Generations>]>,
    /// The segment text one generation of a shard holds, in bytes.
    generation_bytes: usize,
}

/// Two generations: segments enter the new one, and a hit in the old one moves the segment to the
/// new one. When the new one is full it becomes the old one, and the old one is dropped.
#[derive(Default)]
struct Generations {
    new: HashTable<Entry>,
    old: HashTable<Entry>,
    new_bytes: usize,
}

/// A segment's estimate, with the segment's hash so it is computed once.
struct Entry {
    hash: u64,
    segment: Box<str>,
    estimate: usize,
}

impl Entry {
    fn is(&self, hash: u64, segment: &str) -> bool {
        self.hash == hash && *self.segment == *segment
    }
}

impl SegmentCache {
    /// A cache holding at most `capacity` bytes of segment text.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::default()).collect(),
            generation_bytes: capacity / SHARDS / 2,
        }
    }

    /// The estimate of `segment`, cached or computed by `estimate`.
    pub(crate) fn get_or(&self, segment: &str, estimate: impl FnOnce() -> usize) -> usize {
        // A segment larger than a generation would not fit in one.
        if segment.len() > self.generation_bytes {
            return estimate();
        }
        // The tables index by the lowest and highest bits of the same hash, so the shard comes from
        // bits in between.
        let hash = FixedState::default().hash_one(segment);
        let shard = &self.shards[(hash >> 32) as usize % SHARDS];
        if let Some(cached) = lock(shard).get(hash, segment, self.generation_bytes) {
            return cached;
        }
        // Estimated without the lock; another thread may estimate the same segment meanwhile, with
        // the same result.
        let entry = Entry {
            hash,
            segment: segment.into(),
            estimate: estimate(),
        };
        let estimated = entry.estimate;
        lock(shard).insert(entry, self.generation_bytes);
        estimated
    }
}

impl Generations {
    fn get(&mut self, hash: u64, segment: &str, generation_bytes: usize) -> Option<usize> {
        if let Some(entry) = self.new.find(hash, |entry| entry.is(hash, segment)) {
            return Some(entry.estimate);
        }
        let (entry, _) = self
            .old
            .find_entry(hash, |entry| entry.is(hash, segment))
            .ok()?
            .remove();
        let estimate = entry.estimate;
        self.insert(entry, generation_bytes);
        Some(estimate)
    }

    fn insert(&mut self, entry: Entry, generation_bytes: usize) {
        let (hash, segment) = (entry.hash, &*entry.segment);
        if self
            .new
            .find(hash, |cached| cached.is(hash, segment))
            .is_some()
        {
            return;
        }
        if self.new_bytes + entry.segment.len() > generation_bytes {
            self.old = std::mem::take(&mut self.new);
            self.new_bytes = 0;
        }
        self.new_bytes += entry.segment.len();
        self.new
            .insert_unique(entry.hash, entry, |entry| entry.hash);
    }
}

fn lock(shard: &Mutex<Generations>) -> MutexGuard<'_, Generations> {
    // Nothing that can panic runs under the lock, and every entry is a complete estimate.
    shard.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_cache_returns_what_it_was_given() {
        // Each shard's generation holds 4 bytes, so the cache keeps changing generations.
        let cache = SegmentCache::new(SHARDS * 2 * 4);
        let estimate =
            |segment: &str| segment.len() * 7 + segment.bytes().map(usize::from).sum::<usize>();
        let segments: Vec<String> = (0..2_000).map(|i| format!("{}", i % 97)).collect();
        for segment in segments.iter().chain(&segments) {
            assert_eq!(
                cache.get_or(segment, || estimate(segment)),
                estimate(segment)
            );
        }
        for shard in &cache.shards {
            let shard = lock(shard);
            let held: usize = shard
                .new
                .iter()
                .chain(shard.old.iter())
                .map(|entry| entry.segment.len())
                .sum();
            assert!(held <= 2 * cache.generation_bytes);
        }
    }
}
