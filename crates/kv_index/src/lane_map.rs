//! The lane map: one worker's blocks by engine hash, as the event lane that owns the worker
//! keeps them for the run index.
//!
//! A slot carries a place only, `(run, offset)` in 8 bytes. The engine hash of a block lives once
//! in the run index, beside the block's content hash, so a probe that reaches a candidate slot
//! asks the index whether the hash at that place is the key: the `verify` closure of every
//! lookup, and the `key_of` closure growth uses to rehash. Beside the slots a 16-bit word per
//! slot holds a seven-bit fingerprint of the key (a slot is read only when it agrees, one false
//! candidate in 128) and the slot's displacement from its home. Robin Hood hashing keeps the
//! displacements short and bounded at seven-eighths load: an insert that has travelled farther
//! than the entry it meets takes that entry's place and the entry moves on, and a lookup stops as
//! soon as it meets an entry closer to its home than the lookup is to its own. Deletion shifts
//! the entries after the hole back by one until one sits at home: metadata alone decides, no
//! neighbour's key is read, which is what makes key-less slots possible at all (a neighbour's
//! key would be a read into the index per shifted entry).
//!
//! Engine-hash invariant: the index carries one engine hash per distinct block, the one its first
//! holder stored. A worker whose engine names the same block by another hash cannot be found
//! through the place, so the index puts that worker's key into the map's `overflow`, an exact side
//! table consulted when the probe finds nothing; a fleet whose engines disagree pays a hash-map
//! entry for those blocks and nothing else changes.
//!
//! Semantics match a hash map: a key maps to at most one place, `insert` replaces, `remove` of an
//! absent key is a no-op, iteration yields every place once. A differential test against a hash
//! map model under random operations keeps the Robin Hood moves and the backshift right.

use rustc_hash::FxHashMap;

use crate::{event_tree::SequenceHash, run_index::BlockRef};

/// Smallest table a non-empty map allocates.
const MIN_SLOTS: usize = 16;
/// How many keys ahead a batch touches.
const AHEAD: usize = 8;
/// Metadata of an empty slot. A full slot's low byte is the key's fingerprint with the top bit
/// set, its high byte the displacement from the home slot.
const VACANT: u16 = 0;
/// Largest displacement the metadata holds; an insert that would pass it grows the table first.
const MAX_DISPLACEMENT: usize = 255;
/// One displacement step in the metadata word.
const STEP: u16 = 0x100;

const NOWHERE: BlockRef = BlockRef {
    run: u32::MAX,
    offset: 0,
};

/// Seven bits of the key the home slot does not use, with the top bit set so it is never
/// `VACANT`.
#[inline]
fn fingerprint(key: u64) -> u16 {
    u16::from(((key >> 25) as u8) | 0x80)
}

#[inline]
fn meta(fingerprint: u16, displacement: usize) -> u16 {
    fingerprint | ((displacement as u16) << 8)
}

#[inline]
fn displacement(meta: u16) -> usize {
    usize::from(meta >> 8)
}

#[inline]
fn fingerprint_of(meta: u16) -> u16 {
    meta & 0xFF
}

/// One worker's blocks by engine hash: where each lives in the run index.
pub struct RunBlockMap {
    /// Power-of-two length, or empty before the first insert.
    slots: Box<[BlockRef]>,
    /// One word per slot: `VACANT`, or fingerprint and displacement.
    meta: Box<[u16]>,
    /// Entries in the slots.
    len: usize,
    shift: u32,
    /// Keys the index cannot verify through their place (engine-hash conflicts), exact.
    overflow: Option<Box<FxHashMap<u64, BlockRef>>>,
}

impl Default for RunBlockMap {
    fn default() -> Self {
        Self {
            slots: Box::default(),
            meta: Box::default(),
            len: 0,
            shift: 64,
            overflow: None,
        }
    }
}

impl RunBlockMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Entries held, overflow included.
    pub fn len(&self) -> usize {
        self.len + self.overflow.as_ref().map_or(0, |table| table.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Slots allocated (8 bytes each, plus a 2-byte metadata word).
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Bytes held from the process allocator: slots, metadata and the overflow table.
    pub fn memory_bytes(&self) -> usize {
        let overflow = self.overflow.as_ref().map_or(0, |table| {
            table.capacity() * (size_of::<u64>() + size_of::<BlockRef>() + 1)
        });
        self.slots.len() * size_of::<BlockRef>() + self.meta.len() * size_of::<u16>() + overflow
    }

    #[inline]
    fn home(&self, key: u64) -> usize {
        // Fibonacci hashing spreads structured keys; engine hashes are already uniform.
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// The slot holding `key`: walks the metadata from the home slot, reads a slot only when its
    /// fingerprint agrees, and stops at a vacant slot or at an entry closer to its home than the
    /// walk is to `key`'s.
    #[inline]
    fn find(&self, key: u64, mut verify: impl FnMut(BlockRef) -> bool) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        let mask = self.mask();
        let wanted = fingerprint(key);
        let mut index = self.home(key);
        let mut travelled = 0;
        loop {
            let found = self.meta[index];
            if found == VACANT || displacement(found) < travelled {
                return None;
            }
            if fingerprint_of(found) == wanted && verify(self.slots[index]) {
                return Some(index);
            }
            index = (index + 1) & mask;
            travelled += 1;
        }
    }

    fn in_overflow(&self, key: u64) -> Option<BlockRef> {
        self.overflow
            .as_ref()
            .and_then(|table| table.get(&key).copied())
    }

    /// Whether `key` is held; `verify` tells whether a place holds the key.
    pub fn contains_key(&self, key: SequenceHash, verify: impl FnMut(BlockRef) -> bool) -> bool {
        self.find(key.0, verify).is_some() || self.in_overflow(key.0).is_some()
    }

    /// Where `key` lives; `verify` tells whether a place holds the key.
    pub fn get(&self, key: SequenceHash, verify: impl FnMut(BlockRef) -> bool) -> Option<BlockRef> {
        self.find(key.0, verify)
            .map(|index| self.slots[index])
            .or_else(|| self.in_overflow(key.0))
    }

    /// Room for `additional` more entries under seven-eighths load; `key_of` names the key held
    /// at a place so the entries can find their new homes.
    fn reserve(&mut self, additional: usize, key_of: &mut impl FnMut(BlockRef) -> Option<u64>) {
        let needed = self.len + additional;
        if needed * 8 <= self.slots.len() * 7 {
            return;
        }
        let mut capacity = self.slots.len().max(MIN_SLOTS);
        while needed * 8 > capacity * 7 {
            capacity *= 2;
        }
        self.rehash(capacity, key_of);
    }

    /// Move every entry into a table of `capacity` slots. An entry whose place no longer names
    /// a block (the index has let it go) is dropped.
    fn rehash(&mut self, capacity: usize, key_of: &mut impl FnMut(BlockRef) -> Option<u64>) {
        let old_slots =
            std::mem::replace(&mut self.slots, (0..capacity).map(|_| NOWHERE).collect());
        let old_meta = std::mem::replace(&mut self.meta, (0..capacity).map(|_| VACANT).collect());
        self.shift = 64 - capacity.trailing_zeros();
        self.len = 0;
        for (slot, meta) in old_slots.iter().zip(old_meta.iter()) {
            if *meta == VACANT {
                continue;
            }
            if let Some(key) = key_of(*slot) {
                self.place(key, *slot, key_of);
            }
        }
    }

    /// Put a key known to be absent at its Robin Hood position, moving entries that are closer
    /// to their homes along; grows the table (and starts over) when a displacement would pass the
    /// metadata's range.
    fn place(&mut self, key: u64, at: BlockRef, key_of: &mut impl FnMut(BlockRef) -> Option<u64>) {
        let mask = self.mask();
        let mut carried = (fingerprint(key), at);
        let mut index = self.home(key);
        let mut travelled = 0usize;
        loop {
            let found = self.meta[index];
            if found == VACANT {
                self.meta[index] = meta(carried.0, travelled);
                self.slots[index] = carried.1;
                self.len += 1;
                return;
            }
            let resident = displacement(found);
            if resident < travelled {
                // Take the slot; the resident moves on with its own displacement.
                let evicted = (fingerprint_of(found), self.slots[index]);
                self.meta[index] = meta(carried.0, travelled);
                self.slots[index] = carried.1;
                carried = evicted;
                travelled = resident;
            }
            index = (index + 1) & mask;
            travelled += 1;
            if travelled > MAX_DISPLACEMENT {
                // Out of metadata range: double, rehash everything placed so far, and place the
                // carried entry by its key.
                let Some(carried_key) = key_of(carried.1) else {
                    return;
                };
                self.rehash(self.slots.len() * 2, key_of);
                self.place(carried_key, carried.1, key_of);
                return;
            }
        }
    }

    /// Insert or replace; the previous place when the key was present. `verify` tells whether a
    /// place holds `key`, `key_of` names the key at a place (growth).
    pub fn insert(
        &mut self,
        key: SequenceHash,
        at: BlockRef,
        verify: impl FnMut(BlockRef) -> bool,
        mut key_of: impl FnMut(BlockRef) -> Option<u64>,
    ) -> Option<BlockRef> {
        if let Some(table) = self.overflow.as_mut() {
            if let Some(previous) = table.get_mut(&key.0) {
                return Some(std::mem::replace(previous, at));
            }
        }
        if let Some(index) = self.find(key.0, verify) {
            return Some(std::mem::replace(&mut self.slots[index], at));
        }
        self.reserve(1, &mut key_of);
        self.place(key.0, at, &mut key_of);
        None
    }

    /// Keep `key` in the exact side table: the index cannot verify it through its place.
    pub fn insert_overflow(&mut self, key: SequenceHash, at: BlockRef) -> Option<BlockRef> {
        self.overflow
            .get_or_insert_with(Box::default)
            .insert(key.0, at)
    }

    /// Entries in the exact side table.
    pub fn overflow_len(&self) -> usize {
        self.overflow.as_ref().map_or(0, |table| table.len())
    }

    /// Close the hole at `index`: pull the following entries of the probe run back one slot
    /// each until one sits at its home (or the run ends).
    fn backshift(&mut self, mut hole: usize) {
        let mask = self.mask();
        loop {
            let next = (hole + 1) & mask;
            let found = self.meta[next];
            if found == VACANT || displacement(found) == 0 {
                self.meta[hole] = VACANT;
                self.slots[hole] = NOWHERE;
                return;
            }
            self.slots[hole] = self.slots[next];
            self.meta[hole] = found - STEP;
            hole = next;
        }
    }

    /// Remove `key`; its place when it was present.
    pub fn remove(
        &mut self,
        key: SequenceHash,
        verify: impl FnMut(BlockRef) -> bool,
    ) -> Option<BlockRef> {
        if let Some(index) = self.find(key.0, verify) {
            let removed = self.slots[index];
            self.len -= 1;
            self.backshift(index);
            return Some(removed);
        }
        let table = self.overflow.as_mut()?;
        let removed = table.remove(&key.0);
        if table.is_empty() {
            self.overflow = None;
        }
        removed
    }

    /// Ask the cache for the home slot of `key` before the key is used: a prefetch hint, so the
    /// misses of a batch overlap instead of serialising.
    #[inline]
    fn touch(&self, key: u64) {
        if !self.slots.is_empty() {
            let home = self.home(key);
            prefetch_hint::prefetch_read(&self.meta[home]);
            prefetch_hint::prefetch_read(&self.slots[home]);
        }
    }

    /// Remove every key of a batch, reporting each present one's place, with the home slots
    /// touched `AHEAD` keys early. `verify(at, key)` tells whether `at` holds `key`.
    pub fn remove_all(
        &mut self,
        keys: &[SequenceHash],
        mut verify: impl FnMut(BlockRef, SequenceHash) -> bool,
        mut on_removed: impl FnMut(BlockRef),
    ) {
        if self.is_empty() {
            return;
        }
        for key in &keys[..keys.len().min(AHEAD)] {
            self.touch(key.0);
        }
        for (index, key) in keys.iter().enumerate() {
            if let Some(next) = keys.get(index + AHEAD) {
                self.touch(next.0);
            }
            if let Some(at) = self.remove(*key, |at| verify(at, *key)) {
                on_removed(at);
            }
        }
    }

    /// Point `count` consecutive keys at the places from `first` on, with the home slots touched
    /// `AHEAD` keys early: the lane map writes of one store. A key already present (a block the
    /// worker re-stored) moves to its new place. `verify(at, key)` tells whether `at` holds
    /// `key`; `key_of` names the key at a place (growth).
    pub fn insert_run(
        &mut self,
        keys: impl ExactSizeIterator<Item = SequenceHash> + Clone,
        first: BlockRef,
        mut verify: impl FnMut(BlockRef, SequenceHash) -> bool,
        mut key_of: impl FnMut(BlockRef) -> Option<u64>,
    ) {
        self.reserve(keys.len(), &mut key_of);
        let mut ahead = keys.clone();
        for key in ahead.by_ref().take(AHEAD) {
            self.touch(key.0);
        }
        for (index, key) in keys.enumerate() {
            if let Some(next) = ahead.next() {
                self.touch(next.0);
            }
            let at = BlockRef {
                run: first.run,
                offset: first.offset + index as u32,
            };
            if let Some(table) = self.overflow.as_mut() {
                if let Some(place) = table.get_mut(&key.0) {
                    *place = at;
                    continue;
                }
            }
            match self.find(key.0, |place| verify(place, key)) {
                Some(slot) => self.slots[slot] = at,
                None => self.place(key.0, at, &mut key_of),
            }
        }
    }

    /// Every place, in table order, then the overflow's.
    pub fn iter(&self) -> impl Iterator<Item = BlockRef> + '_ {
        self.slots
            .iter()
            .zip(self.meta.iter())
            .filter(|(_, meta)| **meta != VACANT)
            .map(|(slot, _)| *slot)
            .chain(
                self.overflow
                    .iter()
                    .flat_map(|table| table.values().copied()),
            )
    }
}

impl IntoIterator for RunBlockMap {
    type Item = BlockRef;
    type IntoIter = std::vec::IntoIter<BlockRef>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}

impl std::fmt::Debug for RunBlockMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunBlockMap")
            .field("len", &self.len())
            .field("slots", &self.slots.len())
            .field("overflow", &self.overflow_len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// The index's side of the map in miniature: which key each place holds.
    #[derive(Default)]
    struct Places {
        key_at: HashMap<BlockRef, u64>,
    }

    impl Places {
        fn verify(&self, at: BlockRef, key: u64) -> bool {
            self.key_at.get(&at) == Some(&key)
        }

        fn key_of(&self, at: BlockRef) -> Option<u64> {
            self.key_at.get(&at).copied()
        }
    }

    fn place(step: u64, key: u64) -> BlockRef {
        BlockRef {
            run: (key % 1_000) as u32 + 1,
            offset: step as u32,
        }
    }

    /// Randomized differential test against a hash map. A small key space forces long probe
    /// runs, Robin Hood moves, wraparound and backshifts across them; places are unique per
    /// insertion so the model can play the index's verifier.
    #[test]
    fn matches_a_hash_map_under_random_operations() {
        for seed in 1..=12u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let key_space = 8 + seed * 23;
            let mut map = RunBlockMap::new();
            let mut model: HashMap<u64, BlockRef> = HashMap::new();
            let mut places = Places::default();
            for step in 0..30_000u64 {
                let key = rng.next() % key_space;
                match rng.next() % 7 {
                    0..=2 => {
                        let at = place(step, key);
                        places.key_at.insert(at, key);
                        let previous = map.insert(
                            SequenceHash(key),
                            at,
                            |at| places.verify(at, key),
                            |at| places.key_of(at),
                        );
                        assert_eq!(previous, model.insert(key, at), "seed {seed} step {step}");
                    }
                    3..=4 => {
                        let removed = map.remove(SequenceHash(key), |at| places.verify(at, key));
                        assert_eq!(removed, model.remove(&key), "seed {seed} step {step}");
                    }
                    5 => {
                        let keys: Vec<SequenceHash> = (0..(rng.next() % 12))
                            .map(|_| SequenceHash(rng.next() % key_space))
                            .collect();
                        let mut seen = Vec::new();
                        map.remove_all(
                            &keys,
                            |at, key| places.verify(at, key.0),
                            |at| seen.push(at),
                        );
                        let expected: Vec<BlockRef> =
                            keys.iter().filter_map(|key| model.remove(&key.0)).collect();
                        assert_eq!(seen, expected, "seed {seed} step {step}");
                    }
                    6 => {
                        // A run of absent keys, consecutive places.
                        let mut keys = Vec::new();
                        for _ in 0..(rng.next() % 12) {
                            let key = rng.next() % key_space;
                            if !model.contains_key(&key) && !keys.contains(&SequenceHash(key)) {
                                keys.push(SequenceHash(key));
                            }
                        }
                        let first = BlockRef {
                            run: 100_000 + step as u32,
                            offset: 0,
                        };
                        for (index, key) in keys.iter().enumerate() {
                            let at = BlockRef {
                                run: first.run,
                                offset: index as u32,
                            };
                            places.key_at.insert(at, key.0);
                            model.insert(key.0, at);
                        }
                        map.insert_run(
                            keys.iter().copied(),
                            first,
                            |at, key| places.verify(at, key.0),
                            |at| places.key_of(at),
                        );
                    }
                    _ => {
                        let keys: Vec<SequenceHash> = (0..(rng.next() % 6))
                            .map(|_| SequenceHash(rng.next() % key_space))
                            .collect();
                        for key in &keys {
                            assert_eq!(
                                map.contains_key(*key, |at| places.verify(at, key.0)),
                                model.contains_key(&key.0),
                                "seed {seed} step {step}"
                            );
                        }
                    }
                }
                assert_eq!(map.len(), model.len(), "seed {seed} step {step}");
                assert_eq!(
                    map.get(SequenceHash(key), |at| places.verify(at, key)),
                    model.get(&key).copied(),
                    "seed {seed} step {step}"
                );
            }
            let mut entries: Vec<BlockRef> = map.iter().collect();
            let mut expected: Vec<BlockRef> = model.into_values().collect();
            entries.sort();
            expected.sort();
            assert_eq!(entries, expected, "seed {seed}");
        }
    }

    /// Keys the index cannot verify through their place live in the exact side table and take
    /// part in every operation.
    #[test]
    fn conflicting_keys_live_in_the_overflow() {
        let mut map = RunBlockMap::new();
        let mut places = Places::default();
        let at = |run: u32| BlockRef { run, offset: 0 };
        places.key_at.insert(at(1), 10);
        map.insert_run(
            std::iter::once(SequenceHash(10)),
            at(1),
            |at, key| places.verify(at, key.0),
            |at| places.key_of(at),
        );
        // Key 20 names the block at run 2, whose engine hash the index carries as 21.
        places.key_at.insert(at(2), 21);
        assert_eq!(map.insert_overflow(SequenceHash(20), at(2)), None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.overflow_len(), 1);
        assert_eq!(
            map.get(SequenceHash(20), |at| places.verify(at, 20)),
            Some(at(2))
        );
        assert!(map.contains_key(SequenceHash(20), |at| places.verify(at, 20)));
        let mut seen: Vec<BlockRef> = map.iter().collect();
        seen.sort();
        assert_eq!(seen, vec![at(1), at(2)]);
        assert_eq!(map.insert_overflow(SequenceHash(20), at(3)), Some(at(2)));
        assert_eq!(
            map.insert(
                SequenceHash(20),
                at(4),
                |at| places.verify(at, 20),
                |at| places.key_of(at)
            ),
            Some(at(3))
        );
        let mut removed = Vec::new();
        map.remove_all(
            &[SequenceHash(20), SequenceHash(10), SequenceHash(30)],
            |at, key| places.verify(at, key.0),
            |at| removed.push(at),
        );
        assert_eq!(removed, vec![at(4), at(1)]);
        assert!(map.is_empty());
        assert_eq!(map.overflow_len(), 0);
        assert!(map.memory_bytes() > 0);
    }

    /// Displacements stay short at seven-eighths load.
    #[test]
    fn displacements_stay_short_at_full_load() {
        let mut map = RunBlockMap::new();
        let mut places = Places::default();
        let mut rng = Rng(7);
        let keys: Vec<SequenceHash> = (0..7 * 4096).map(|_| SequenceHash(rng.next())).collect();
        for (index, key) in keys.iter().enumerate() {
            let at = BlockRef {
                run: 1,
                offset: index as u32,
            };
            places.key_at.insert(at, key.0);
            map.insert_run(
                std::iter::once(*key),
                at,
                |at, key| places.verify(at, key.0),
                |at| places.key_of(at),
            );
        }
        assert_eq!(map.capacity(), 32768);
        let longest = map.meta.iter().map(|m| displacement(*m)).max().unwrap();
        let mean = map
            .meta
            .iter()
            .filter(|m| **m != VACANT)
            .map(|m| displacement(*m))
            .sum::<usize>() as f64
            / map.len() as f64;
        assert!(longest < 64, "longest displacement {longest}");
        assert!(mean < 4.0, "mean displacement {mean}");
        for key in &keys {
            assert!(map.contains_key(*key, |at| places.verify(at, key.0)));
        }
    }

    #[test]
    fn empty_map_answers_without_slots() {
        let mut map = RunBlockMap::new();
        assert!(map.is_empty());
        assert!(!map.contains_key(SequenceHash(3), |_| true));
        assert_eq!(map.remove(SequenceHash(3), |_| true), None);
        map.remove_all(
            &[SequenceHash(1)],
            |_, _| true,
            |_| panic!("nothing to remove"),
        );
        assert_eq!(map.capacity(), 0);
        assert_eq!(map.memory_bytes(), 0);
    }
}
