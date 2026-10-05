//! A run-compressed KV index: the event-driven prefix index as a tree of runs with per-run
//! worker coverage bitsets, lock-free and allocation-free for readers, bounded in memory.
//!
//! The index answers the same question as [`PositionalIndexer`](crate::PositionalIndexer): for a
//! request given as its per-block content hashes, how many leading blocks does each worker hold,
//! where "holds" means the worker stored, at every position up to there, the block that sits on
//! the request's chain. It is fed by the same engine events (stored / removed / cleared, keyed by
//! the engine's block hashes) and keeps the same per-worker block map the gateway's event monitor
//! owns, so it drops into the same call sites. Its results are checked against
//! [`ReferenceIndexer`](crate::ReferenceIndexer) by `tests/exactness_run.rs` (including evictions
//! that leave holes in a chain) and under concurrent lanes by `tests/concurrency_run.rs`.
//!
//! Shape:
//! - A **run** is a maximal stretch of consecutive positions on one chain whose set of holding
//!   workers is the same at every position. It stores one content hash per position and one
//!   coverage bit per worker. Runs form a tree: a run's children continue it with different next
//!   blocks. A store appends to a run or adds a child; a divergence or a coverage change splits a
//!   run into a prefix and a suffix; positions never move except by a split, which forwards them.
//!   Because coverage is uniform within a run, a worker that evicts a block in the middle of a
//!   chain simply stops covering the piece that holds it, and a lookup stops there for that
//!   worker and nowhere else: holes are exact.
//! - A **lookup** walks from the root, comparing the request's content hashes against each run on
//!   the path (one compare loop per run, not one probe per block) and ANDing the alive set with
//!   the run's coverage; a worker that drops out scores the position where it dropped. Readers
//!   take no locks, allocate nothing but the result map and write to no memory at all: every run
//!   header, hash array and child table lives in an arena addressed by integer ids, a run's
//!   window `(hash array, base, length, children)` is read under a seqlock version that is
//!   checked again after the run's hashes, coverage and child entry have been read, so a split,
//!   a growth, an unlink or a reuse of the run is atomic to a reader.
//! - **Writers** (the event lanes, one per engine worker) lock one run at a time, plus its parent
//!   for the moment it takes to unlink an empty run. A decode extension of the worker's own leaf
//!   appends in place: one lock, no allocation. A split shares the hash array between prefix and
//!   suffix (no copy) and leaves a forwarding record so map entries written before it still
//!   resolve, which keeps other lanes' maps untouched. The lane's own map takes an engine hash to
//!   `(run, offset)`.
//! - **Memory is recycled.** Run headers, hash arrays (reference-counted across the runs a split
//!   leaves sharing one) and child tables return to free lists when they die; a run id carries a
//!   generation so a stale child entry or forwarding record to a reused id is recognised.
//!   Children are an open-addressing table (linear probing, tombstones, rebuilt at 3/4 load), so
//!   a node with many children, the root above all, inserts in constant time.
//!
//! Engine hashes: the index trusts the engine's parent pointers and block identities, as the
//! positional indexer does. Nothing is shared between workers through the maps, so one engine
//! reusing a hash cannot corrupt another worker's view.
//!
//! Memory: 8 bytes per distinct block on a chain (its content hash, shared by every worker that
//! holds it, with about 12% slack for growth) plus a 64-byte run header, the coverage words and a
//! child table per branching run, against the per-worker map entry each lane keeps for removals.

use std::{
    collections::BTreeSet,
    sync::{
        atomic::{fence, AtomicIsize, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;
use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::event_tree::{
    chain_prefix_hash, ApplyError, ContentHash, OverlapScores, SequenceHash, StoredBlock,
    WorkerIdExhausted,
};

/// The virtual root: position 0's parent, holds no blocks, never dies.
const ROOT: u32 = 0;
/// "No table" / "no array": word 0 of the arena is never handed out.
const NONE: u32 = 0;
/// Forwarding target of blocks whose last holder evicted them: nowhere.
const GONE: u32 = u32::MAX;
/// A table slot whose child was unlinked.
const TOMB: u64 = u64::MAX;
/// Runs per slab chunk (1024) and chunks in the directory (64 Mi runs in all).
const RUN_CHUNK_BITS: u32 = 10;
const RUN_CHUNK: usize = 1 << RUN_CHUNK_BITS;
const RUN_DIR: usize = 1 << 16;
/// Words per arena chunk (1 Mi, 8 MiB) and chunks in the directory (4 Gi words in all).
const WORD_CHUNK_BITS: u32 = 20;
const WORD_CHUNK: usize = 1 << WORD_CHUNK_BITS;
const WORD_DIR: usize = 1 << 12;
/// Hash array capacities: multiples of 8 up to 128, then powers of two.
const SMALL_ARRAY_CLASSES: usize = 16;
const ARRAY_CLASSES: usize = SMALL_ARRAY_CLASSES + 13;
/// Child table slot counts: powers of two from 2.
const MIN_TABLE_SLOTS: usize = 2;
const TABLE_CLASSES: usize = 27;
/// Coverage words per run at most: 1024 workers.
const MAX_WORDS: usize = 16;

/// Where one of a worker's blocks lives: the run and the offset of the block within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockRef {
    pub run: u32,
    pub offset: u32,
}

/// One worker's blocks by engine hash, owned by that worker's event lane (the gateway's
/// `KvEventMonitor` task). Entries may point at a run that has since been split; the index
/// forwards them on use.
pub type RunBlockMap = FxHashMap<SequenceHash, BlockRef>;

/// Hash array capacity class: `8, 16, .., 128, 256, 512, ..`.
fn array_class(capacity: usize) -> usize {
    if capacity <= 8 * SMALL_ARRAY_CLASSES {
        capacity.div_ceil(8).max(1) - 1
    } else {
        let bits = usize::BITS - (capacity - 1).leading_zeros();
        SMALL_ARRAY_CLASSES + (bits as usize - 8)
    }
}

fn class_capacity(class: usize) -> usize {
    if class < SMALL_ARRAY_CLASSES {
        8 * (class + 1)
    } else {
        1 << (class - SMALL_ARRAY_CLASSES + 8)
    }
}

/// Room for `len` hashes plus a little for decode extensions.
fn capacity_for(len: usize) -> usize {
    len + (len / 8).max(2)
}

fn table_class(slots: usize) -> usize {
    (slots.trailing_zeros() as usize).saturating_sub(MIN_TABLE_SLOTS.trailing_zeros() as usize)
}

/// Append-only storage of 64-bit words in fixed chunks with free lists per size class. Holds
/// hash arrays (`used | capacity << 32`, `refs`, then the hashes) and child tables
/// (`slots | used << 32`, `live`, then `(head hash, run id | generation << 32)` slots). An
/// allocation never crosses a chunk, so any array is one slice.
struct WordArena {
    dir: Box<[OnceLock<Box<[AtomicU64]>>]>,
    next: AtomicU64,
    free_arrays: Vec<SegQueue<u32>>,
    free_tables: Vec<SegQueue<u32>>,
}

impl WordArena {
    fn new() -> Self {
        Self {
            dir: (0..WORD_DIR).map(|_| OnceLock::new()).collect(),
            next: AtomicU64::new(1),
            free_arrays: (0..ARRAY_CLASSES).map(|_| SegQueue::new()).collect(),
            free_tables: (0..TABLE_CLASSES).map(|_| SegQueue::new()).collect(),
        }
    }

    #[inline]
    fn chunk(&self, index: usize) -> &[AtomicU64] {
        self.dir[index].get_or_init(|| (0..WORD_CHUNK).map(|_| AtomicU64::new(0)).collect())
    }

    /// `count` consecutive words starting at `start` (all within one chunk, as allocated).
    #[inline]
    fn words(&self, start: u32, count: usize) -> &[AtomicU64] {
        let start = start as usize;
        let offset = start & (WORD_CHUNK - 1);
        &self.chunk(start >> WORD_CHUNK_BITS)[offset..offset + count]
    }

    #[inline]
    fn word(&self, at: u32) -> &AtomicU64 {
        &self.words(at, 1)[0]
    }

    /// Fresh words inside one chunk.
    fn bump(&self, count: usize) -> u32 {
        debug_assert!(0 < count && count <= WORD_CHUNK);
        loop {
            let current = self.next.load(Ordering::Relaxed);
            let mut start = current as usize;
            if start >> WORD_CHUNK_BITS != (start + count - 1) >> WORD_CHUNK_BITS {
                start = ((start >> WORD_CHUNK_BITS) + 1) << WORD_CHUNK_BITS;
            }
            let end = start + count;
            assert!(
                end <= WORD_DIR * WORD_CHUNK,
                "run index arena exhausted: more than 2^32 hash words"
            );
            if self
                .next
                .compare_exchange_weak(current, end as u64, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.chunk(start >> WORD_CHUNK_BITS);
                return start as u32;
            }
        }
    }

    fn used(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    /// Words sitting in free lists.
    fn free_words(&self) -> usize {
        let arrays: usize = self
            .free_arrays
            .iter()
            .enumerate()
            .map(|(class, list)| list.len() * (class_capacity(class) + 2))
            .sum();
        let tables: usize = self
            .free_tables
            .iter()
            .enumerate()
            .map(|(class, list)| list.len() * (2 + 2 * (MIN_TABLE_SLOTS << class)))
            .sum();
        arrays + tables
    }

    // ---- hash arrays: [used | capacity << 32][refs][hash; capacity], data = start + 2 ----

    /// A hash array holding `contents` with room for at least `capacity`; returns the data start.
    fn alloc_array(&self, contents: &[u64], capacity: usize) -> u32 {
        let class = array_class(capacity.max(contents.len()).max(8));
        let capacity = class_capacity(class);
        let recycled = self.free_arrays[class].pop();
        let start = recycled.unwrap_or_else(|| self.bump(capacity + 2));
        let data = start + 2;
        for (slot, &hash) in self.words(data, contents.len()).iter().zip(contents) {
            slot.store(hash, Ordering::Relaxed);
        }
        self.word(start + 1).store(1, Ordering::Relaxed);
        self.word(start).store(
            pack(contents.len() as u32, capacity as u32),
            Ordering::Release,
        );
        data
    }

    #[inline]
    fn array_header(&self, data: u32) -> &AtomicU64 {
        self.word(data - 2)
    }

    /// One more run shares this array.
    fn array_retain(&self, data: u32) {
        self.word(data - 1).fetch_add(1, Ordering::Relaxed);
    }

    /// One run fewer uses this array; the last one frees it.
    fn array_release(&self, data: u32) {
        if data == NONE {
            return;
        }
        if self.word(data - 1).fetch_sub(1, Ordering::AcqRel) == 1 {
            let (_, capacity) = unpack(self.array_header(data).load(Ordering::Relaxed));
            self.free_arrays[array_class(capacity as usize)].push(data - 2);
        }
    }

    // ---- child tables: [slots | used << 32][live][(head, run | gen << 32); slots] ----

    fn alloc_table(&self, slots: usize) -> u32 {
        let class = table_class(slots);
        let recycled = self.free_tables[class].pop();
        let table = recycled.unwrap_or_else(|| self.bump(2 + 2 * slots));
        for word in self.words(table + 2, 2 * slots) {
            word.store(0, Ordering::Relaxed);
        }
        self.word(table + 1).store(0, Ordering::Relaxed);
        self.word(table).store(slots as u64, Ordering::Release);
        table
    }

    fn free_table(&self, table: u32) {
        if table == NONE {
            return;
        }
        let slots = self.table_slots(table);
        self.free_tables[table_class(slots)].push(table);
    }

    #[inline]
    fn table_slots(&self, table: u32) -> usize {
        self.word(table).load(Ordering::Relaxed) as u32 as usize
    }

    /// The child continuing with `head`, as `(run, generation)`.
    #[inline]
    fn table_find(&self, table: u32, head: u64) -> Option<(u32, u32)> {
        if table == NONE {
            return None;
        }
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        let mask = slots - 1;
        let mut index = head as usize & mask;
        for _ in 0..slots {
            let entry = words[2 * index + 1].load(Ordering::Acquire);
            if entry == 0 {
                return None;
            }
            if entry != TOMB && words[2 * index].load(Ordering::Relaxed) == head {
                return Some((entry as u32, (entry >> 32) as u32));
            }
            index = (index + 1) & mask;
        }
        None
    }

    /// Live `(head, run, generation)` entries of a table.
    fn table_entries(&self, table: u32) -> Vec<(u64, u32, u32)> {
        if table == NONE {
            return Vec::new();
        }
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        (0..slots)
            .filter_map(|index| {
                let entry = words[2 * index + 1].load(Ordering::Acquire);
                (entry != 0 && entry != TOMB).then(|| {
                    (
                        words[2 * index].load(Ordering::Relaxed),
                        entry as u32,
                        (entry >> 32) as u32,
                    )
                })
            })
            .collect()
    }

    /// Visit the live children of a table without allocating.
    fn for_each_child(&self, table: u32, mut visit: impl FnMut(u32)) {
        if table == NONE {
            return;
        }
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        for index in 0..slots {
            let entry = words[2 * index + 1].load(Ordering::Acquire);
            if entry != 0 && entry != TOMB {
                visit(entry as u32);
            }
        }
    }

    /// Write an entry into a free or tombstoned slot of a table that has room (writer side).
    fn table_put(&self, table: u32, head: u64, run: u32, generation: u32) {
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        let mask = slots - 1;
        let mut index = head as usize & mask;
        loop {
            let entry = words[2 * index + 1].load(Ordering::Relaxed);
            if entry == 0 || entry == TOMB {
                words[2 * index].store(head, Ordering::Relaxed);
                words[2 * index + 1].store(
                    u64::from(run) | (u64::from(generation) << 32),
                    Ordering::Release,
                );
                let header = self.word(table);
                let used = (header.load(Ordering::Relaxed) >> 32) + u64::from(entry == 0);
                header.store(slots as u64 | (used << 32), Ordering::Relaxed);
                self.word(table + 1).fetch_add(1, Ordering::Relaxed);
                return;
            }
            index = (index + 1) & mask;
        }
    }

    /// Tombstone the entry of `run`; returns the live count left.
    fn table_take(&self, table: u32, run: u32) -> u64 {
        let slots = self.table_slots(table);
        let words = self.words(table + 2, 2 * slots);
        for index in 0..slots {
            let entry = words[2 * index + 1].load(Ordering::Relaxed);
            if entry != 0 && entry != TOMB && entry as u32 == run {
                words[2 * index + 1].store(TOMB, Ordering::Release);
                return self.word(table + 1).fetch_sub(1, Ordering::Relaxed) - 1;
            }
        }
        self.word(table + 1).load(Ordering::Relaxed)
    }

    /// Whether another entry fits under 3/4 load (tombstones count).
    fn table_has_room(&self, table: u32) -> bool {
        if table == NONE {
            return false;
        }
        let header = self.word(table).load(Ordering::Relaxed);
        let (slots, used) = (header as u32 as usize, (header >> 32) as usize);
        (used + 1) * 4 <= slots * 3
    }

    /// A new table holding the live entries of `table` plus room for one more.
    fn table_grown(&self, table: u32) -> u32 {
        let entries = self.table_entries(table);
        let needed = entries.len() + 1;
        let mut slots = MIN_TABLE_SLOTS;
        while needed * 4 > slots * 3 {
            slots *= 2;
        }
        let grown = self.alloc_table(slots);
        for (head, run, generation) in entries {
            self.table_put(grown, head, run, generation);
        }
        grown
    }
}

#[inline]
fn pack(used: u32, capacity: u32) -> u64 {
    (u64::from(capacity) << 32) | u64::from(used)
}

#[inline]
fn unpack(header: u64) -> (u32, u32) {
    (header as u32, (header >> 32) as u32)
}

/// Writer-side bookkeeping of a run, under its lock.
#[derive(Default)]
struct RunMeta {
    /// Splits this run has undergone, oldest first: `(offset, suffix run, suffix generation)`.
    /// A block that sat at `offset >= o` before the split lives in the suffix at `offset - o`
    /// (and may have been forwarded again from there); `GONE` means nobody holds it any more.
    /// Offsets decrease along the vector: a run never grows after a split.
    splits: Vec<(u32, u32, u32)>,
    /// Unlinked from the tree; its id may be reused (with the next generation).
    dead: bool,
}

/// A reader's consistent view of a run's window.
#[derive(Clone, Copy)]
struct Window {
    /// Data start of the hash array, or `NONE`.
    block: u32,
    /// Offset of the run's first hash within the array.
    base: u32,
    len: u32,
    /// Child table, or `NONE`.
    children: u32,
}

struct Run {
    /// Absolute position of the run's first block.
    start: AtomicU32,
    /// Run id of the parent (the root's parent is itself).
    parent: AtomicU32,
    /// Generation in the high half, seqlock in the low half: odd while an update is in flight.
    version: AtomicU64,
    block: AtomicU32,
    base: AtomicU32,
    len: AtomicU32,
    children: AtomicU32,
    meta: Mutex<RunMeta>,
}

impl Run {
    fn blank() -> Self {
        Self {
            start: AtomicU32::new(0),
            parent: AtomicU32::new(ROOT),
            version: AtomicU64::new(0),
            block: AtomicU32::new(NONE),
            base: AtomicU32::new(0),
            len: AtomicU32::new(0),
            children: AtomicU32::new(NONE),
            meta: Mutex::new(RunMeta::default()),
        }
    }

    #[inline]
    fn start(&self) -> usize {
        self.start.load(Ordering::Relaxed) as usize
    }

    #[inline]
    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    #[inline]
    fn generation(&self) -> u32 {
        (self.version.load(Ordering::Relaxed) >> 32) as u32
    }

    /// The window as a reader sees it, with the version to confirm afterwards. An in-place
    /// append only grows `len`, published with a release store after its hashes, and needs no
    /// version; everything else that changes the window goes through `begin_update`.
    #[inline]
    fn snapshot(&self) -> (Window, u64) {
        loop {
            let before = self.version.load(Ordering::Acquire);
            if before & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let window = Window {
                block: self.block.load(Ordering::Relaxed),
                base: self.base.load(Ordering::Relaxed),
                len: self.len.load(Ordering::Acquire),
                children: self.children.load(Ordering::Relaxed),
            };
            fence(Ordering::Acquire);
            if self.version.load(Ordering::Relaxed) == before {
                return (window, before);
            }
        }
    }

    /// Whether everything read since the snapshot belongs to it.
    #[inline]
    fn confirm(&self, version: u64) -> bool {
        fence(Ordering::Acquire);
        self.version.load(Ordering::Relaxed) == version
    }

    fn begin_update(&self) {
        self.version.fetch_add(1, Ordering::Acquire);
    }

    fn end_update(&self) {
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Start a new life of this header: next generation, fields reset.
    fn reincarnate(&self, start: usize, parent: u32, window: Window) {
        self.version.fetch_add((1 << 32) | 1, Ordering::Acquire);
        self.start.store(start as u32, Ordering::Relaxed);
        self.parent.store(parent, Ordering::Relaxed);
        self.block.store(window.block, Ordering::Relaxed);
        self.base.store(window.base, Ordering::Relaxed);
        self.len.store(window.len, Ordering::Relaxed);
        self.children.store(window.children, Ordering::Relaxed);
        self.end_update();
    }
}

struct RunChunk {
    runs: Box<[Run]>,
    /// `words` coverage words per run, run-major.
    coverage: Box<[AtomicU64]>,
}

/// Run storage: a directory of fixed-size chunks created on first use, with a free list of dead
/// ids.
struct RunSlab {
    dir: Box<[OnceLock<RunChunk>]>,
    next: AtomicU32,
    free: SegQueue<u32>,
    words: usize,
}

impl RunSlab {
    fn new(words: usize) -> Self {
        Self {
            dir: (0..RUN_DIR).map(|_| OnceLock::new()).collect(),
            next: AtomicU32::new(0),
            free: SegQueue::new(),
            words,
        }
    }

    #[inline]
    fn chunk(&self, index: usize) -> &RunChunk {
        self.dir[index].get_or_init(|| RunChunk {
            runs: (0..RUN_CHUNK).map(|_| Run::blank()).collect(),
            coverage: (0..RUN_CHUNK * self.words)
                .map(|_| AtomicU64::new(0))
                .collect(),
        })
    }

    #[inline]
    fn run(&self, id: u32) -> &Run {
        let id = id as usize;
        &self.chunk(id >> RUN_CHUNK_BITS).runs[id & (RUN_CHUNK - 1)]
    }

    #[inline]
    fn coverage(&self, id: u32) -> &[AtomicU64] {
        let id = id as usize;
        let chunk = self.chunk(id >> RUN_CHUNK_BITS);
        let first = (id & (RUN_CHUNK - 1)) * self.words;
        &chunk.coverage[first..first + self.words]
    }

    /// A run that is not yet reachable from the tree: a dead header given its next life, or a
    /// fresh one.
    fn alloc(&self, start: usize, parent: u32, window: Window) -> u32 {
        if let Some(id) = self.free.pop() {
            let run = self.run(id);
            let mut meta = run.meta.lock();
            debug_assert!(meta.dead && coverage_is_empty(self.coverage(id)));
            meta.splits.clear();
            meta.dead = false;
            run.reincarnate(start, parent, window);
            return id;
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        assert!(
            (id as usize) < RUN_DIR * RUN_CHUNK,
            "run index slab exhausted: more than 2^26 runs"
        );
        let run = self.run(id);
        run.start.store(start as u32, Ordering::Relaxed);
        run.parent.store(parent, Ordering::Relaxed);
        run.block.store(window.block, Ordering::Relaxed);
        run.base.store(window.base, Ordering::Relaxed);
        run.len.store(window.len, Ordering::Relaxed);
        run.children.store(window.children, Ordering::Relaxed);
        id
    }

    fn allocated(&self) -> usize {
        self.next.load(Ordering::Relaxed) as usize
    }
}

#[inline]
fn has(coverage: &[AtomicU64], worker: u32) -> bool {
    coverage[(worker / 64) as usize].load(Ordering::Relaxed) & (1u64 << (worker % 64)) != 0
}

fn set(coverage: &[AtomicU64], worker: u32) {
    coverage[(worker / 64) as usize].fetch_or(1u64 << (worker % 64), Ordering::Relaxed);
}

fn clear(coverage: &[AtomicU64], worker: u32) {
    coverage[(worker / 64) as usize].fetch_and(!(1u64 << (worker % 64)), Ordering::Relaxed);
}

fn coverage_is_empty(coverage: &[AtomicU64]) -> bool {
    coverage
        .iter()
        .all(|word| word.load(Ordering::Relaxed) == 0)
}

/// Exactly `worker` and nobody else.
fn covered_only_by(coverage: &[AtomicU64], worker: u32) -> bool {
    let word = (worker / 64) as usize;
    let bit = 1u64 << (worker % 64);
    coverage.iter().enumerate().all(|(index, slot)| {
        let value = slot.load(Ordering::Relaxed);
        if index == word {
            value == bit
        } else {
            value == 0
        }
    })
}

fn workers(coverage: &[AtomicU64]) -> Vec<u32> {
    coverage
        .iter()
        .enumerate()
        .flat_map(|(index, word)| {
            let value = word.load(Ordering::Relaxed);
            (0..64)
                .filter(move |bit| value & (1u64 << bit) != 0)
                .map(move |bit| (index * 64 + bit) as u32)
        })
        .collect()
}

/// Where a block that sat at `at` before this run's splits lives now, one hop, with the
/// generation the suffix had when the split happened.
fn forward(splits: &[(u32, u32, u32)], at: BlockRef) -> Option<(BlockRef, u32)> {
    splits
        .iter()
        .find(|(offset, _, _)| *offset <= at.offset)
        .map(|&(offset, suffix, generation)| {
            (
                BlockRef {
                    run: suffix,
                    offset: at.offset - offset,
                },
                generation,
            )
        })
}

/// Memory and shape counters, for the scoreboard.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunIndexStats {
    /// Run headers ever created (resident).
    pub runs_allocated: usize,
    /// Dead headers waiting for reuse.
    pub runs_free: usize,
    /// Runs linked in the tree.
    pub runs_live: usize,
    /// Content hashes held by live runs.
    pub blocks_live: usize,
    /// Bytes of the word arena handed out so far (hash arrays, child tables, free lists
    /// included).
    pub arena_bytes: usize,
    /// Bytes of the word arena sitting in free lists.
    pub arena_free_bytes: usize,
    /// Bytes of run headers, coverage words included (all allocated runs).
    pub header_bytes: usize,
}

/// Worker slots: a slot is in use from `intern_worker` until `remove_worker`, after which it is
/// handed out again (every coverage bit of a removed worker is clear by then).
#[derive(Default)]
struct WorkerRegistry {
    names: Vec<Option<Arc<str>>>,
    free: Vec<u32>,
}

/// The run-compressed index. Worker ids are interned `u32`s, as in the positional indexer.
pub struct RunIndex {
    slab: RunSlab,
    arena: WordArena,
    words: usize,
    max_workers: usize,
    worker_to_id: DashMap<Arc<str>, u32, FxBuildHasher>,
    registry: Mutex<WorkerRegistry>,
    /// Blocks held per worker, one cache line each: the lanes write these on every event.
    worker_blocks: Box<[CachePadded<AtomicUsize>]>,
    /// Per-worker signed contributions to the distinct-block count, summed on demand.
    distinct_blocks: Box<[CachePadded<AtomicIsize>]>,
}

impl Default for RunIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of one attempt to walk a store into the tree.
enum Walk {
    Done,
    /// The walk met a run another lane unlinked meanwhile; start over from the parent block.
    Restart,
    /// The parent block is not held after all (its run died since the map was written).
    NoParent,
}

/// What [`RunIndex::store_in_run`] found.
enum InRun<'b> {
    /// Every block is placed.
    Done,
    /// The blocks still to place start right after the run's last block.
    Continue(&'b [StoredBlock]),
    /// Carry on in this run (id, generation) from its first block.
    MoveTo(u32, u32),
}

/// Blocks a store walk placed: `count` blocks from `start` in the event sit in `run` from
/// `offset`. Recorded under the run lock, written into the lane map after it.
struct Placed {
    run: u32,
    offset: u32,
    start: usize,
    count: usize,
}

/// A batch of a worker's offsets in one run, with the generation the run must still have.
struct Removal {
    run: u32,
    generation: Option<u32>,
    offsets: Vec<u32>,
}

impl RunIndex {
    /// An index for up to 256 workers.
    pub fn new() -> Self {
        Self::with_max_workers(256)
    }

    /// An index for up to `max_workers` interned workers (at most 1024); coverage costs one bit
    /// per worker per run, rounded up to 64.
    pub fn with_max_workers(max_workers: usize) -> Self {
        let max_workers = max_workers.clamp(1, MAX_WORDS * 64);
        let words = max_workers.div_ceil(64);
        let slab = RunSlab::new(words);
        // The root is run 0: position 0's parent, no blocks, no coverage.
        let root = slab.alloc(
            0,
            ROOT,
            Window {
                block: NONE,
                base: 0,
                len: 0,
                children: NONE,
            },
        );
        debug_assert_eq!(root, ROOT);
        Self {
            slab,
            arena: WordArena::new(),
            words,
            max_workers,
            worker_to_id: DashMap::with_hasher(FxBuildHasher),
            registry: Mutex::new(WorkerRegistry::default()),
            worker_blocks: (0..max_workers)
                .map(|_| CachePadded::new(AtomicUsize::new(0)))
                .collect(),
            distinct_blocks: (0..max_workers)
                .map(|_| CachePadded::new(AtomicIsize::new(0)))
                .collect(),
        }
    }

    /// Intern a worker name; the same name maps to the same id until the worker is removed.
    pub fn intern_worker(&self, worker: &str) -> Result<u32, WorkerIdExhausted> {
        if let Some(entry) = self.worker_to_id.get(worker) {
            return Ok(*entry.value());
        }
        let name: Arc<str> = Arc::from(worker);
        match self.worker_to_id.entry(name.clone()) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let mut registry = self.registry.lock();
                let id = match registry.free.pop() {
                    Some(id) => id,
                    None if registry.names.len() < self.max_workers => {
                        registry.names.push(None);
                        (registry.names.len() - 1) as u32
                    }
                    None => return Err(WorkerIdExhausted),
                };
                registry.names[id as usize] = Some(name);
                entry.insert(id);
                Ok(id)
            }
        }
    }

    /// Hand a removed worker's slot back once nothing refers to it any more.
    fn release_worker(&self, worker: u32) {
        let mut registry = self.registry.lock();
        let Some(name) = registry
            .names
            .get_mut(worker as usize)
            .and_then(Option::take)
        else {
            return;
        };
        self.worker_to_id.remove(&*name);
        registry.free.push(worker);
    }

    pub fn worker_id(&self, worker: &str) -> Option<u32> {
        self.worker_to_id.get(worker).map(|entry| *entry.value())
    }

    /// Blocks held across all workers (a block two workers hold counts twice).
    pub fn current_size(&self) -> usize {
        self.worker_blocks
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum()
    }

    /// Distinct blocks held by at least one worker.
    pub fn entry_count(&self) -> usize {
        let total: isize = self
            .distinct_blocks
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum();
        total.max(0) as usize
    }

    pub fn worker_block_count(&self, worker: u32) -> usize {
        self.worker_blocks
            .get(worker as usize)
            .map_or(0, |count| count.load(Ordering::Relaxed))
    }

    fn credit(&self, worker: u32, blocks: usize) {
        if blocks == 0 {
            return;
        }
        self.worker_blocks[worker as usize].fetch_add(blocks, Ordering::Relaxed);
    }

    fn debit(&self, worker: u32, blocks: usize) {
        if blocks == 0 {
            return;
        }
        self.worker_blocks[worker as usize].fetch_sub(blocks, Ordering::Relaxed);
    }

    /// `worker` became the first holder of `blocks` distinct blocks.
    fn distinct_add(&self, worker: u32, blocks: usize) {
        if blocks != 0 {
            self.distinct_blocks[worker as usize].fetch_add(blocks as isize, Ordering::Relaxed);
        }
    }

    /// `worker` was the last holder of `blocks` distinct blocks.
    fn distinct_sub(&self, worker: u32, blocks: usize) {
        if blocks != 0 {
            self.distinct_blocks[worker as usize].fetch_sub(blocks as isize, Ordering::Relaxed);
        }
    }

    /// The hash at `offset` of a run, from the writer's side (the run is locked).
    #[inline]
    fn hash_at(&self, run: &Run, offset: usize) -> u64 {
        let data = run.block.load(Ordering::Relaxed) + run.base.load(Ordering::Relaxed);
        self.arena
            .word(data + offset as u32)
            .load(Ordering::Relaxed)
    }

    /// Follow split forwarding records to where a block lives now, returning with the final run
    /// locked so nothing can move the block before the caller uses it. `None` when the block is
    /// not held any more (its run died, or it was forwarded to nowhere).
    fn resolve_locked(&self, mut at: BlockRef) -> Option<(BlockRef, MutexGuard<'_, RunMeta>)> {
        let mut expected: Option<u32> = None;
        loop {
            if at.run == GONE {
                return None;
            }
            let run = self.slab.run(at.run);
            let meta = run.meta.lock();
            if meta.dead || expected.is_some_and(|generation| generation != run.generation()) {
                return None;
            }
            match forward(&meta.splits, at) {
                Some((next, generation)) => {
                    at = next;
                    expected = Some(generation);
                }
                None => return Some((at, meta)),
            }
        }
    }

    /// Add `child` to the run's table (the run is locked), growing the table when it is 3/4
    /// full; the old table is freed after the new one is published.
    fn link_child(&self, run: &Run, head: u64, child: u32) {
        let table = run.children.load(Ordering::Relaxed);
        let generation = self.slab.run(child).generation();
        if self.arena.table_has_room(table) {
            self.arena.table_put(table, head, child, generation);
            return;
        }
        let grown = if table == NONE {
            self.arena.alloc_table(MIN_TABLE_SLOTS)
        } else {
            self.arena.table_grown(table)
        };
        self.arena.table_put(grown, head, child, generation);
        run.begin_update();
        run.children.store(grown, Ordering::Relaxed);
        run.end_update();
        self.arena.free_table(table);
    }

    /// Take `child` out of the run's table (the run is locked); an emptied table goes away.
    fn unlink_child(&self, run: &Run, child: u32) {
        let table = run.children.load(Ordering::Relaxed);
        if table == NONE {
            return;
        }
        if self.arena.table_take(table, child) == 0 {
            run.begin_update();
            run.children.store(NONE, Ordering::Relaxed);
            run.end_update();
            self.arena.free_table(table);
        }
    }

    /// Retire a run that is unlinked (locked by the caller): its array reference goes, its
    /// window empties, and its id is queued for reuse once the caller has dropped the lock.
    fn kill(&self, run_id: u32, meta: &mut RunMeta, freed: &mut Vec<u32>) {
        let run = self.slab.run(run_id);
        let block = run.block.load(Ordering::Relaxed);
        run.begin_update();
        run.block.store(NONE, Ordering::Relaxed);
        run.base.store(0, Ordering::Relaxed);
        run.len.store(0, Ordering::Relaxed);
        run.children.store(NONE, Ordering::Relaxed);
        run.end_update();
        self.arena.array_release(block);
        meta.dead = true;
        freed.push(run_id);
    }

    /// Dead ids go back to the slab only after their locks are released, so a thread holding a
    /// live run's lock and reviving a dead id never waits on a thread that holds the dead id's
    /// lock and wants the live run.
    fn recycle(&self, freed: &mut Vec<u32>) {
        for id in freed.drain(..) {
            self.slab.free.push(id);
        }
    }

    /// Split `run` (locked by the caller) at `at`: the run keeps `[0, at)`, a new suffix run takes
    /// `[at, len)` on the same hash array, the run's children and its coverage minus `exclude`.
    /// A suffix nobody covers and that has no children is not created: the forwarding record
    /// says the blocks are gone.
    fn split_locked(
        &self,
        worker: u32,
        run_id: u32,
        meta: &mut RunMeta,
        at: usize,
        exclude: Option<u32>,
    ) -> u32 {
        let run = self.slab.run(run_id);
        let coverage = self.slab.coverage(run_id);
        let len = run.len();
        debug_assert!(at > 0 && at < len, "split inside the run: 0 < {at} < {len}");
        let block = run.block.load(Ordering::Relaxed);
        let base = run.base.load(Ordering::Relaxed);
        let children = run.children.load(Ordering::Relaxed);
        let mut suffix_words = [0u64; MAX_WORDS];
        for (index, word) in coverage.iter().enumerate() {
            let mut value = word.load(Ordering::Relaxed);
            if let Some(worker) = exclude {
                if (worker / 64) as usize == index {
                    value &= !(1u64 << (worker % 64));
                }
            }
            suffix_words[index] = value;
        }
        let uncovered = suffix_words[..self.words].iter().all(|word| *word == 0);
        if uncovered && !coverage_is_empty(coverage) {
            // The excluded worker was the last holder of these blocks.
            self.distinct_sub(worker, len - at);
        }
        if uncovered && children == NONE {
            run.begin_update();
            run.len.store(at as u32, Ordering::Relaxed);
            run.end_update();
            meta.splits.push((at as u32, GONE, 0));
            return GONE;
        }
        self.arena.array_retain(block);
        let suffix_id = self.slab.alloc(
            run.start() + at,
            run_id,
            Window {
                block,
                base: base + at as u32,
                len: (len - at) as u32,
                children,
            },
        );
        let suffix = self.slab.run(suffix_id);
        for (slot, value) in self.slab.coverage(suffix_id).iter().zip(suffix_words) {
            slot.store(value, Ordering::Relaxed);
        }
        self.arena.for_each_child(children, |child| {
            self.slab
                .run(child)
                .parent
                .store(suffix_id, Ordering::Release);
        });
        let table = self.arena.alloc_table(MIN_TABLE_SLOTS);
        self.arena
            .table_put(table, self.hash_at(run, at), suffix_id, suffix.generation());
        run.begin_update();
        run.len.store(at as u32, Ordering::Relaxed);
        run.children.store(table, Ordering::Relaxed);
        run.end_update();
        meta.splits
            .push((at as u32, suffix_id, suffix.generation()));
        suffix_id
    }

    /// Unlink `run` (locked by the caller, known to be an uncovered leaf) from its parent and
    /// retire it, then the parent if that leaves it an uncovered leaf too.
    fn unlink_locked(&self, run_id: u32, meta: &mut RunMeta, freed: &mut Vec<u32>) {
        if run_id == ROOT || meta.dead {
            return;
        }
        let run = self.slab.run(run_id);
        loop {
            let parent_id = run.parent.load(Ordering::Acquire);
            let parent = self.slab.run(parent_id);
            // Child-then-parent is the only order in which two run locks are ever held. A split
            // of the parent may have re-parented this run while we waited; check and retry.
            let mut parent_meta = parent.meta.lock();
            if run.parent.load(Ordering::Acquire) != parent_id {
                continue;
            }
            self.unlink_child(parent, run_id);
            self.kill(run_id, meta, freed);
            if parent_id != ROOT
                && parent.children.load(Ordering::Relaxed) == NONE
                && coverage_is_empty(self.slab.coverage(parent_id))
            {
                self.unlink_locked(parent_id, &mut parent_meta, freed);
            }
            return;
        }
    }

    /// Store `blocks` for `worker` after `parent` (position 0 when `None`).
    pub fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        map: &mut RunBlockMap,
    ) -> Result<(), ApplyError> {
        if blocks.is_empty() {
            return Ok(());
        }
        let origin = match parent {
            None => None,
            Some(hash) => {
                if map.is_empty() {
                    return Err(ApplyError::WorkerNotTracked);
                }
                match map.get(&hash) {
                    Some(&at) => Some((hash, at)),
                    None => return Err(ApplyError::ParentBlockNotFound),
                }
            }
        };
        loop {
            match self.store_walk(worker, blocks, origin, map) {
                Walk::Done => return Ok(()),
                Walk::Restart => {}
                Walk::NoParent => return Err(ApplyError::ParentBlockNotFound),
            }
        }
    }

    /// One attempt to place a store, from the parent block (or the root) down the tree. Map
    /// entries for the placed blocks are written after the run locks are released: the lane map
    /// is private, and a split meanwhile is covered by the forwarding records.
    fn store_walk(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        origin: Option<(SequenceHash, BlockRef)>,
        map: &mut RunBlockMap,
    ) -> Walk {
        let mut pending: Vec<Placed> = Vec::new();
        let outcome = self.store_walk_locked(worker, blocks, origin, map, &mut pending);
        for placed in pending {
            for (index, stored) in blocks[placed.start..placed.start + placed.count]
                .iter()
                .enumerate()
            {
                map.insert(
                    stored.seq_hash,
                    BlockRef {
                        run: placed.run,
                        offset: placed.offset + index as u32,
                    },
                );
            }
        }
        outcome
    }

    fn store_walk_locked(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        origin: Option<(SequenceHash, BlockRef)>,
        map: &mut RunBlockMap,
        pending: &mut Vec<Placed>,
    ) -> Walk {
        let (mut run_id, mut offset, mut meta) = match origin {
            None => (ROOT, 0usize, self.slab.run(ROOT).meta.lock()),
            Some((hash, at)) => {
                let Some((at, meta)) = self.resolve_locked(at) else {
                    map.remove(&hash);
                    return Walk::NoParent;
                };
                map.insert(hash, at);
                (at.run, at.offset as usize + 1, meta)
            }
        };
        let mut remaining = blocks;
        loop {
            let next = match self.store_in_run(
                worker,
                run_id,
                &mut meta,
                offset,
                remaining,
                blocks.len() - remaining.len(),
                pending,
            ) {
                InRun::Done => return Walk::Done,
                InRun::Continue(rest) => {
                    remaining = rest;
                    self.store_at_end(
                        worker,
                        run_id,
                        &meta,
                        remaining,
                        blocks.len() - remaining.len(),
                        pending,
                    )
                }
                InRun::MoveTo(child, generation) => Some((child, generation)),
            };
            let Some((child, generation)) = next else {
                return Walk::Done;
            };
            drop(meta);
            // Between the parent's lock and the child's, the child may have been unlinked and
            // its id given to another run: start over from the parent block if so.
            run_id = child;
            offset = 0;
            meta = self.slab.run(run_id).meta.lock();
            if meta.dead || self.slab.run(run_id).generation() != generation {
                return Walk::Restart;
            }
        }
    }

    /// Match `remaining` against the run from `offset`: join or split the run as needed, record
    /// the matched blocks, and say how to go on.
    #[expect(clippy::too_many_arguments)]
    fn store_in_run<'b>(
        &self,
        worker: u32,
        run_id: u32,
        meta: &mut RunMeta,
        offset: usize,
        remaining: &'b [StoredBlock],
        block_start: usize,
        pending: &mut Vec<Placed>,
    ) -> InRun<'b> {
        let run = self.slab.run(run_id);
        let coverage = self.slab.coverage(run_id);
        let len = run.len();
        if offset >= len {
            return InRun::Continue(remaining);
        }
        let covered = has(coverage, worker);
        if !covered && offset > 0 {
            // The parent entry pointed into a run this worker does not cover (it cannot, unless
            // the engine re-stored under a stale parent). Cut here and join the suffix instead.
            let suffix = self.split_locked(worker, run_id, meta, offset, None);
            return InRun::MoveTo(suffix, self.slab.run(suffix).generation());
        }
        let data = run.block.load(Ordering::Relaxed) + run.base.load(Ordering::Relaxed);
        let hashes = self.arena.words(data + offset as u32, len - offset);
        let matched = remaining
            .iter()
            .zip(hashes)
            .take_while(|(stored, slot)| stored.content_hash.0 == slot.load(Ordering::Relaxed))
            .count();
        let available = len - offset;
        if matched < available && (matched < remaining.len() || !covered) {
            // A divergence inside the run (both branches stay held), or a worker that holds only
            // a prefix of it: either way the run ends here.
            self.split_locked(worker, run_id, meta, offset + matched, None);
        }
        if !covered {
            if coverage_is_empty(coverage) {
                self.distinct_add(worker, run.len());
            }
            set(coverage, worker);
            self.credit(worker, run.len());
        }
        pending.push(Placed {
            run: run_id,
            offset: offset as u32,
            start: block_start,
            count: matched,
        });
        if matched == remaining.len() {
            return InRun::Done;
        }
        InRun::Continue(&remaining[matched..])
    }

    /// `remaining` starts right after the run's last block: descend into the child that continues
    /// it, append to the worker's own leaf, or open a new run. Returns the child (id, generation)
    /// to descend into.
    fn store_at_end(
        &self,
        worker: u32,
        run_id: u32,
        meta: &RunMeta,
        remaining: &[StoredBlock],
        block_start: usize,
        pending: &mut Vec<Placed>,
    ) -> Option<(u32, u32)> {
        let run = self.slab.run(run_id);
        let coverage = self.slab.coverage(run_id);
        let head = remaining[0].content_hash.0;
        let children = run.children.load(Ordering::Relaxed);
        if let Some(found) = self.arena.table_find(children, head) {
            return Some(found);
        }
        let len = run.len();
        let contents: Vec<u64> = remaining
            .iter()
            .map(|stored| stored.content_hash.0)
            .collect();
        let own_leaf = run_id != ROOT
            && children == NONE
            && meta.splits.is_empty()
            && covered_only_by(coverage, worker);
        let (target, first) = if own_leaf {
            self.append(run, &contents);
            (run_id, len)
        } else {
            let block = self
                .arena
                .alloc_array(&contents, capacity_for(contents.len()));
            let new_id = self.slab.alloc(
                run.start() + len,
                run_id,
                Window {
                    block,
                    base: 0,
                    len: contents.len() as u32,
                    children: NONE,
                },
            );
            set(self.slab.coverage(new_id), worker);
            self.link_child(run, head, new_id);
            (new_id, 0)
        };
        pending.push(Placed {
            run: target,
            offset: first as u32,
            start: block_start,
            count: remaining.len(),
        });
        self.credit(worker, contents.len());
        self.distinct_add(worker, contents.len());
        None
    }

    /// Extend a leaf in place when its window ends the hash array and the array has room;
    /// otherwise move it to a larger array.
    fn append(&self, run: &Run, contents: &[u64]) {
        let block = run.block.load(Ordering::Relaxed);
        let base = run.base.load(Ordering::Relaxed) as usize;
        let len = run.len();
        let header = self.arena.array_header(block);
        let (used, capacity) = unpack(header.load(Ordering::Acquire));
        let end = base + len;
        let claimed = end == used as usize
            && end + contents.len() <= capacity as usize
            && header
                .compare_exchange(
                    pack(used, capacity),
                    pack((end + contents.len()) as u32, capacity),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok();
        if claimed {
            let slots = self.arena.words(block + end as u32, contents.len());
            for (slot, &hash) in slots.iter().zip(contents) {
                slot.store(hash, Ordering::Relaxed);
            }
            run.len
                .store((len + contents.len()) as u32, Ordering::Release);
            return;
        }
        let mut grown: Vec<u64> = self
            .arena
            .words(block + base as u32, len)
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .collect();
        grown.extend_from_slice(contents);
        let new_block = self.arena.alloc_array(&grown, capacity_for(grown.len()));
        run.begin_update();
        run.block.store(new_block, Ordering::Relaxed);
        run.base.store(0, Ordering::Relaxed);
        run.len.store(grown.len() as u32, Ordering::Release);
        run.end_update();
        self.arena.array_release(block);
    }

    /// Forget the named blocks of `worker`; unknown hashes are ignored.
    pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], map: &mut RunBlockMap) {
        let mut refs: Vec<BlockRef> = hashes.iter().filter_map(|hash| map.remove(hash)).collect();
        refs.sort_unstable_by_key(|at| at.run);
        let mut work: Vec<Removal> = Vec::new();
        let mut index = 0;
        while index < refs.len() {
            let run = refs[index].run;
            let end = refs[index..]
                .iter()
                .position(|at| at.run != run)
                .map_or(refs.len(), |count| index + count);
            work.push(Removal {
                run,
                generation: None,
                offsets: refs[index..end].iter().map(|at| at.offset).collect(),
            });
            index = end;
        }
        let mut freed = Vec::new();
        while let Some(removal) = work.pop() {
            self.remove_from_run(worker, removal, &mut work, &mut freed);
            self.recycle(&mut freed);
        }
    }

    /// Drop `worker` from the offsets of one run, forwarding offsets a split moved on.
    fn remove_from_run(
        &self,
        worker: u32,
        removal: Removal,
        work: &mut Vec<Removal>,
        freed: &mut Vec<u32>,
    ) {
        if removal.run == GONE {
            return;
        }
        let run = self.slab.run(removal.run);
        let coverage = self.slab.coverage(removal.run);
        let mut meta = run.meta.lock();
        if meta.dead
            || removal
                .generation
                .is_some_and(|generation| generation != run.generation())
        {
            return;
        }
        let offsets = reforward(&meta, removal.offsets, work);
        if offsets.is_empty() || !has(coverage, worker) {
            return;
        }
        self.remove_ranges(worker, removal.run, &mut meta, offsets);
        if coverage_is_empty(coverage) && run.children.load(Ordering::Relaxed) == NONE {
            self.unlink_locked(removal.run, &mut meta, freed);
        }
    }

    /// Drop `worker` from the given offsets of a run it covers, splitting the run so that the
    /// pieces it still covers keep their offsets.
    fn remove_ranges(&self, worker: u32, run_id: u32, meta: &mut RunMeta, mut offsets: Vec<u32>) {
        let run = self.slab.run(run_id);
        offsets.sort_unstable();
        offsets.dedup();
        let len = run.len();
        offsets.retain(|&offset| (offset as usize) < len);
        // Contiguous ranges, highest first, so earlier ranges keep their offsets after the splits
        // a later range causes.
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &offset in &offsets {
            let offset = offset as usize;
            match ranges.last_mut() {
                Some((_, high)) if *high + 1 == offset => *high = offset,
                _ => ranges.push((offset, offset)),
            }
        }
        for (low, high) in ranges.into_iter().rev() {
            if high + 1 < run.len() {
                // The tail beyond the range keeps every worker, this one included.
                self.split_locked(worker, run_id, meta, high + 1, None);
            }
            if low > 0 {
                // The range becomes its own run without this worker.
                self.split_locked(worker, run_id, meta, low, Some(worker));
            } else {
                self.clear_holder(run_id, worker);
            }
            self.debit(worker, high + 1 - low);
        }
    }

    /// Drop `worker` from a run it covers, keeping the distinct-block count in step.
    fn clear_holder(&self, run_id: u32, worker: u32) {
        let coverage = self.slab.coverage(run_id);
        clear(coverage, worker);
        if coverage_is_empty(coverage) {
            self.distinct_sub(worker, self.slab.run(run_id).len());
        }
    }

    /// Forget every block of `worker` (the engine cleared its cache); the map is emptied.
    pub fn apply_cleared(&self, worker: u32, map: &mut RunBlockMap) {
        let drained = std::mem::take(map);
        self.drop_worker(worker, drained);
    }

    /// Forget every block of `worker` (the worker left) and free its slot; the same name interns
    /// afresh afterwards.
    pub fn remove_worker(&self, worker: u32, map: RunBlockMap) {
        self.drop_worker(worker, map);
        self.release_worker(worker);
    }

    fn drop_worker(&self, worker: u32, map: RunBlockMap) {
        // Keyed by (run, generation): a forwarding record to a dead generation of an id must not
        // shadow the live run that reused the id.
        let mut seen: FxHashSet<(u32, Option<u32>)> = FxHashSet::default();
        let mut work: Vec<(u32, Option<u32>)> = Vec::new();
        for (_, at) in map {
            if seen.insert((at.run, None)) {
                work.push((at.run, None));
            }
        }
        let mut freed = Vec::new();
        while let Some((run_id, generation)) = work.pop() {
            if run_id == GONE {
                continue;
            }
            let run = self.slab.run(run_id);
            let coverage = self.slab.coverage(run_id);
            let mut meta = run.meta.lock();
            if meta.dead || generation.is_some_and(|generation| generation != run.generation()) {
                continue;
            }
            // Blocks of this worker may have moved into suffixes since the map was written.
            for &(_, suffix, suffix_generation) in &meta.splits {
                if suffix != GONE && seen.insert((suffix, Some(suffix_generation))) {
                    work.push((suffix, Some(suffix_generation)));
                }
            }
            if has(coverage, worker) {
                self.clear_holder(run_id, worker);
                self.debit(worker, run.len());
                if coverage_is_empty(coverage) && run.children.load(Ordering::Relaxed) == NONE {
                    self.unlink_locked(run_id, &mut meta, &mut freed);
                }
            }
            drop(meta);
            self.recycle(&mut freed);
        }
    }

    /// Score every worker by how many leading blocks of the request it holds. With `early_exit`,
    /// report the workers holding the first block, each scored 1.
    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        let mut out = OverlapScores::default();
        let Some(&first) = content_hashes.first() else {
            return out;
        };
        let root = self.slab.run(ROOT);
        let (mut run_id, mut expected) = loop {
            let (window, version) = root.snapshot();
            let found = self.arena.table_find(window.children, first.0);
            if root.confirm(version) {
                match found {
                    Some(entry) => break entry,
                    None => return out,
                }
            }
        };
        let words = self.words;
        let mut alive = [0u64; MAX_WORDS];
        let mut position = 0usize;
        loop {
            let run = self.slab.run(run_id);
            let (window, version) = run.snapshot();
            if (version >> 32) as u32 != expected {
                break;
            }
            let len = window.len as usize;
            let available = len.min(content_hashes.len() - position);
            let hashes = self.arena.words(window.block + window.base, available);
            let matched = content_hashes[position..position + available]
                .iter()
                .zip(hashes)
                .take_while(|(content, slot)| content.0 == slot.load(Ordering::Relaxed))
                .count();
            let coverage = self.slab.coverage(run_id);
            let mut held = [0u64; MAX_WORDS];
            for (word, slot) in held[..words].iter_mut().zip(coverage) {
                *word = slot.load(Ordering::Relaxed);
            }
            let next = if matched == len && position + matched < content_hashes.len() {
                self.arena
                    .table_find(window.children, content_hashes[position + matched].0)
            } else {
                None
            };
            if !run.confirm(version) {
                continue;
            }
            if matched == 0 {
                break;
            }
            if position == 0 {
                alive = held;
                if early_exit {
                    emit(&alive[..words], 1, &mut out);
                    return out;
                }
            } else {
                for (index, word) in alive[..words].iter_mut().enumerate() {
                    let dropped = *word & !held[index];
                    if dropped != 0 {
                        emit_word(index, dropped, position as u32, &mut out);
                    }
                    *word &= held[index];
                }
            }
            if alive[..words].iter().all(|word| *word == 0) {
                return out;
            }
            position += matched;
            match next {
                Some((child, generation)) => {
                    run_id = child;
                    expected = generation;
                }
                None => break,
            }
        }
        emit(&alive[..words], position as u32, &mut out);
        out
    }

    /// Every block every worker holds, as `(worker, position, content hash, prefix hash)`;
    /// for tests and for comparing against the reference indexer. Not consistent under
    /// concurrent writes.
    #[doc(hidden)]
    pub fn debug_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        let mut out = BTreeSet::new();
        let mut stack: Vec<(u32, Option<SequenceHash>)> = Vec::new();
        let (root, _) = self.slab.run(ROOT).snapshot();
        for (_, child, _) in self.arena.table_entries(root.children) {
            stack.push((child, None));
        }
        while let Some((run_id, mut prefix)) = stack.pop() {
            let run = self.slab.run(run_id);
            let (window, _) = run.snapshot();
            let start = run.start();
            let holders = workers(self.slab.coverage(run_id));
            let hashes = self
                .arena
                .words(window.block + window.base, window.len as usize);
            for (offset, slot) in hashes.iter().enumerate() {
                let content = ContentHash(slot.load(Ordering::Relaxed));
                let next = match prefix {
                    Some(previous) => chain_prefix_hash(previous, content),
                    None => SequenceHash(content.0),
                };
                for &worker in &holders {
                    out.insert((worker, start + offset, content, next));
                }
                prefix = Some(next);
            }
            for (_, child, _) in self.arena.table_entries(window.children) {
                stack.push((child, prefix));
            }
        }
        out
    }

    /// Shape and memory counters.
    pub fn stats(&self) -> RunIndexStats {
        let allocated = self.slab.allocated();
        let mut stats = RunIndexStats {
            runs_allocated: allocated,
            runs_free: self.slab.free.len(),
            arena_bytes: self.arena.used() as usize * size_of::<AtomicU64>(),
            arena_free_bytes: self.arena.free_words() * size_of::<AtomicU64>(),
            header_bytes: allocated * (size_of::<Run>() + self.words * size_of::<AtomicU64>()),
            ..RunIndexStats::default()
        };
        for id in 1..allocated as u32 {
            let run = self.slab.run(id);
            if run.meta.lock().dead {
                continue;
            }
            stats.runs_live += 1;
            stats.blocks_live += run.len();
        }
        stats
    }
}

/// Offsets taken from the map may have moved into suffixes since they were written: send those
/// on, tagged with the generation the suffix had at the split.
fn reforward(meta: &RunMeta, mut offsets: Vec<u32>, work: &mut Vec<Removal>) -> Vec<u32> {
    if meta.splits.is_empty() {
        return offsets;
    }
    let mut forwarded: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
    offsets.retain(
        |&offset| match forward(&meta.splits, BlockRef { run: ROOT, offset }) {
            Some((next, generation)) => {
                if next.run != GONE {
                    forwarded
                        .entry((next.run, generation))
                        .or_default()
                        .push(next.offset);
                }
                false
            }
            None => true,
        },
    );
    work.extend(
        forwarded
            .into_iter()
            .map(|((run, generation), offsets)| Removal {
                run,
                generation: Some(generation),
                offsets,
            }),
    );
    offsets
}

fn emit(alive: &[u64], score: u32, out: &mut OverlapScores) {
    if score == 0 {
        return;
    }
    let count: u32 = alive.iter().map(|word| word.count_ones()).sum();
    out.scores.reserve(count as usize);
    for (index, &word) in alive.iter().enumerate() {
        if word != 0 {
            emit_word(index, word, score, out);
        }
    }
}

#[inline]
fn emit_word(index: usize, mut word: u64, score: u32, out: &mut OverlapScores) {
    while word != 0 {
        let bit = word.trailing_zeros();
        out.scores.insert((index * 64 + bit as usize) as u32, score);
        word &= word - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{request_prefix_hashes, ReferenceIndexer};

    fn content(stream: u64, position: usize) -> ContentHash {
        crate::compute_content_hash(&[stream as u32, (stream >> 32) as u32, position as u32])
    }

    fn blocks_of(contents: &[ContentHash]) -> Vec<StoredBlock> {
        contents
            .iter()
            .zip(request_prefix_hashes(contents))
            .map(|(&content_hash, seq_hash)| StoredBlock {
                seq_hash,
                content_hash,
            })
            .collect()
    }

    fn scores(index: &RunIndex, query: &[ContentHash]) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = index
            .find_matches(query, false)
            .scores
            .into_iter()
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn array_classes_round_up_and_back() {
        for capacity in [1usize, 8, 9, 16, 100, 128, 129, 256, 257, 1000, 4096, 5000] {
            let class = array_class(capacity);
            assert!(class_capacity(class) >= capacity, "capacity {capacity}");
            assert_eq!(array_class(class_capacity(class)), class);
        }
        assert_eq!(table_class(2), 0);
        assert_eq!(table_class(4), 1);
        assert_eq!(table_class(1024), 9);
    }

    #[test]
    fn store_lookup_and_divergence() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let mut map = RunBlockMap::default();
        let held: Vec<ContentHash> = (0..10).map(|p| content(1, p)).collect();
        index
            .apply_stored(w, &blocks_of(&held), None, &mut map)
            .expect("store");
        assert_eq!(scores(&index, &held), vec![(w, 10)]);
        assert_eq!(scores(&index, &held[..4]), vec![(w, 4)]);
        let mut diverged = held.clone();
        diverged[5] = content(2, 0);
        assert_eq!(scores(&index, &diverged), vec![(w, 5)]);
        let mut extended = held.clone();
        extended.push(content(3, 0));
        assert_eq!(scores(&index, &extended), vec![(w, 10)]);
        assert_eq!(scores(&index, &[content(9, 0)]), vec![]);
        assert_eq!(index.current_size(), 10);
        assert_eq!(index.entry_count(), 10);
    }

    #[test]
    fn two_workers_share_a_prefix_and_split_at_the_fork() {
        let index = RunIndex::with_max_workers(8);
        let a = index.intern_worker("a").expect("id");
        let b = index.intern_worker("b").expect("id");
        let (mut ma, mut mb) = (RunBlockMap::default(), RunBlockMap::default());
        let base: Vec<ContentHash> = (0..6).map(|p| content(1, p)).collect();
        let mut fork = base[..3].to_vec();
        fork.extend((0..4).map(|p| content(2, p)));
        index
            .apply_stored(a, &blocks_of(&base), None, &mut ma)
            .expect("store a");
        index
            .apply_stored(b, &blocks_of(&fork), None, &mut mb)
            .expect("store b");
        assert_eq!(scores(&index, &base), vec![(a, 6), (b, 3)]);
        assert_eq!(scores(&index, &fork), vec![(a, 3), (b, 7)]);
        let mut early: Vec<(u32, u32)> =
            index.find_matches(&base, true).scores.into_iter().collect();
        early.sort_unstable();
        assert_eq!(early, vec![(a, 1), (b, 1)]);
        let mut reference = ReferenceIndexer::new();
        reference
            .apply_stored(a, &blocks_of(&base), None)
            .expect("ref a");
        reference
            .apply_stored(b, &blocks_of(&fork), None)
            .expect("ref b");
        assert_eq!(index.debug_blocks(), reference.blocks());
        assert_eq!(index.entry_count(), 10);
    }

    #[test]
    fn a_worker_holds_both_sides_of_its_own_divergence() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let mut map = RunBlockMap::default();
        let base: Vec<ContentHash> = (0..6).map(|p| content(1, p)).collect();
        let mut fork = base[..3].to_vec();
        fork.extend((0..2).map(|p| content(2, p)));
        let blocks = blocks_of(&base);
        index
            .apply_stored(w, &blocks, None, &mut map)
            .expect("base");
        let fork_blocks = blocks_of(&fork);
        index
            .apply_stored(w, &fork_blocks[3..], Some(blocks[2].seq_hash), &mut map)
            .expect("fork");
        assert_eq!(scores(&index, &base), vec![(w, 6)]);
        assert_eq!(scores(&index, &fork), vec![(w, 5)]);
        assert_eq!(index.current_size(), 8);
    }

    #[test]
    fn a_hole_stops_the_match_at_the_hole() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let mut map = RunBlockMap::default();
        let held: Vec<ContentHash> = (0..8).map(|p| content(1, p)).collect();
        let blocks = blocks_of(&held);
        index
            .apply_stored(w, &blocks, None, &mut map)
            .expect("store");
        index.apply_removed(w, &[blocks[3].seq_hash], &mut map);
        assert_eq!(scores(&index, &held), vec![(w, 3)]);
        assert_eq!(index.current_size(), 7);
        // Re-storing the missing block after its parent heals the hole.
        index
            .apply_stored(w, &blocks[3..4], Some(blocks[2].seq_hash), &mut map)
            .expect("heal");
        assert_eq!(scores(&index, &held), vec![(w, 8)]);
        let mut reference = ReferenceIndexer::new();
        reference.apply_stored(w, &blocks, None).expect("ref");
        assert_eq!(index.debug_blocks(), reference.blocks());
    }

    #[test]
    fn a_hole_in_a_shared_run_affects_only_the_evicting_worker() {
        let index = RunIndex::with_max_workers(8);
        let v = index.intern_worker("v").expect("id");
        let w = index.intern_worker("w").expect("id");
        let (mut mv, mut mw) = (RunBlockMap::default(), RunBlockMap::default());
        let held: Vec<ContentHash> = (0..10).map(|p| content(1, p)).collect();
        let blocks = blocks_of(&held);
        index.apply_stored(v, &blocks, None, &mut mv).expect("v");
        index.apply_stored(w, &blocks, None, &mut mw).expect("w");
        index.apply_removed(w, &[blocks[5].seq_hash], &mut mw);
        assert_eq!(scores(&index, &held), vec![(v, 10), (w, 5)]);
        index.apply_removed(v, &[blocks[7].seq_hash, blocks[8].seq_hash], &mut mv);
        assert_eq!(scores(&index, &held), vec![(v, 7), (w, 5)]);
        index
            .apply_stored(w, &blocks[5..6], Some(blocks[4].seq_hash), &mut mw)
            .expect("heal");
        assert_eq!(scores(&index, &held), vec![(v, 7), (w, 10)]);
        let mut reference = ReferenceIndexer::new();
        reference.apply_stored(v, &blocks, None).expect("ref v");
        reference.apply_stored(w, &blocks, None).expect("ref w");
        reference.apply_removed(v, &[blocks[7].seq_hash, blocks[8].seq_hash]);
        assert_eq!(index.debug_blocks(), reference.blocks());
    }

    #[test]
    fn tail_removal_truncates_and_a_clear_empties() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let mut map = RunBlockMap::default();
        let held: Vec<ContentHash> = (0..8).map(|p| content(1, p)).collect();
        let blocks = blocks_of(&held);
        index
            .apply_stored(w, &blocks, None, &mut map)
            .expect("store");
        let tail: Vec<SequenceHash> = blocks[5..].iter().map(|b| b.seq_hash).collect();
        index.apply_removed(w, &tail, &mut map);
        assert_eq!(scores(&index, &held), vec![(w, 5)]);
        assert_eq!(map.len(), 5);
        assert_eq!(index.entry_count(), 5);
        index.apply_cleared(w, &mut map);
        assert!(map.is_empty());
        assert_eq!(scores(&index, &held), vec![]);
        assert_eq!(index.current_size(), 0);
        assert_eq!(index.entry_count(), 0);
        assert!(index.debug_blocks().is_empty());
        let stats = index.stats();
        assert_eq!(stats.runs_live, 0);
        assert_eq!(stats.runs_free, 1, "the dead run waits for reuse");
        assert_eq!(
            stats.arena_free_bytes,
            stats.arena_bytes - 8,
            "every array and table is back in a free list"
        );
    }

    #[test]
    fn dead_runs_and_arrays_are_reused() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let mut map = RunBlockMap::default();
        for round in 0..200u64 {
            let held: Vec<ContentHash> = (0..12).map(|p| content(10 + round, p)).collect();
            let blocks = blocks_of(&held);
            index
                .apply_stored(w, &blocks, None, &mut map)
                .expect("store");
            assert_eq!(scores(&index, &held), vec![(w, 12)]);
            let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.seq_hash).collect();
            index.apply_removed(w, &hashes, &mut map);
            assert_eq!(scores(&index, &held), vec![]);
        }
        let stats = index.stats();
        assert!(
            stats.runs_allocated <= 3,
            "runs were not recycled: {stats:?}"
        );
        assert!(
            stats.arena_bytes < 4096,
            "arena words were not recycled: {stats:?}"
        );
        assert_eq!(index.current_size(), 0);
    }

    #[test]
    fn appends_reuse_the_array_until_another_worker_joins() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let v = index.intern_worker("v").expect("id");
        let (mut mw, mut mv) = (RunBlockMap::default(), RunBlockMap::default());
        let held: Vec<ContentHash> = (0..40).map(|p| content(1, p)).collect();
        let blocks = blocks_of(&held);
        index
            .apply_stored(w, &blocks[..4], None, &mut mw)
            .expect("first");
        for step in 1..10 {
            let from = step * 4;
            index
                .apply_stored(
                    w,
                    &blocks[from..from + 4],
                    Some(blocks[from - 1].seq_hash),
                    &mut mw,
                )
                .expect("extend");
        }
        assert_eq!(
            index.stats().runs_live,
            1,
            "decode extensions stay in one run"
        );
        assert_eq!(scores(&index, &held), vec![(w, 40)]);
        index
            .apply_stored(v, &blocks[..20], None, &mut mv)
            .expect("join");
        assert_eq!(scores(&index, &held), vec![(w, 40), (v, 20)]);
        assert_eq!(index.stats().runs_live, 2);
        let more: Vec<ContentHash> = (0..3).map(|p| content(2, p)).collect();
        let mut long = held.clone();
        long.extend(more);
        let long_blocks = blocks_of(&long);
        index
            .apply_stored(w, &long_blocks[40..], Some(blocks[39].seq_hash), &mut mw)
            .expect("extend after join");
        assert_eq!(scores(&index, &long), vec![(w, 43), (v, 20)]);
        assert_eq!(index.stats().runs_live, 2, "the tail is still w's own leaf");
    }

    #[test]
    fn many_children_grow_the_table_and_stay_findable() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let mut map = RunBlockMap::default();
        let prompt: Vec<ContentHash> = (0..3).map(|p| content(1, p)).collect();
        index
            .apply_stored(w, &blocks_of(&prompt), None, &mut map)
            .expect("prompt");
        let anchor = blocks_of(&prompt)[2].seq_hash;
        let mut chains = Vec::new();
        for branch in 0..300u64 {
            let mut chain = prompt.clone();
            chain.extend((0..2).map(|p| content(100 + branch, p)));
            index
                .apply_stored(w, &blocks_of(&chain)[3..], Some(anchor), &mut map)
                .expect("branch");
            chains.push(chain);
        }
        for chain in &chains {
            assert_eq!(scores(&index, chain), vec![(w, 5)]);
        }
        let mut unknown = prompt.clone();
        unknown.push(content(999, 0));
        assert_eq!(scores(&index, &unknown), vec![(w, 3)]);
        // Unlinking every other branch tombstones its slot; the rest stay findable.
        for chain in chains.iter().step_by(2) {
            let hashes: Vec<SequenceHash> =
                blocks_of(chain)[3..].iter().map(|b| b.seq_hash).collect();
            index.apply_removed(w, &hashes, &mut map);
        }
        for (branch, chain) in chains.iter().enumerate() {
            let expected = if branch % 2 == 0 { 3 } else { 5 };
            assert_eq!(
                scores(&index, chain),
                vec![(w, expected)],
                "branch {branch}"
            );
        }
        let mut reference = ReferenceIndexer::new();
        reference
            .apply_stored(w, &blocks_of(&prompt), None)
            .expect("ref prompt");
        for chain in chains.iter().skip(1).step_by(2) {
            reference
                .apply_stored(w, &blocks_of(chain)[3..], Some(anchor))
                .expect("ref branch");
        }
        assert_eq!(index.debug_blocks(), reference.blocks());
    }

    #[test]
    fn parent_errors_match_the_positional_indexer() {
        let index = RunIndex::with_max_workers(8);
        let w = index.intern_worker("w").expect("id");
        let mut map = RunBlockMap::default();
        let held: Vec<ContentHash> = (0..3).map(|p| content(1, p)).collect();
        let blocks = blocks_of(&held);
        assert!(matches!(
            index.apply_stored(w, &blocks[1..], Some(blocks[0].seq_hash), &mut map),
            Err(ApplyError::WorkerNotTracked)
        ));
        index
            .apply_stored(w, &blocks[..1], None, &mut map)
            .expect("store");
        assert!(matches!(
            index.apply_stored(w, &blocks[2..], Some(blocks[1].seq_hash), &mut map),
            Err(ApplyError::ParentBlockNotFound)
        ));
    }

    #[test]
    fn worker_slots_are_bounded_and_reused_after_removal() {
        let index = RunIndex::with_max_workers(2);
        assert_eq!(index.intern_worker("a"), Ok(0));
        assert_eq!(index.intern_worker("b"), Ok(1));
        assert_eq!(index.intern_worker("a"), Ok(0));
        assert_eq!(index.intern_worker("c"), Err(WorkerIdExhausted));
        let mut map = RunBlockMap::default();
        let held: Vec<ContentHash> = (0..4).map(|p| content(1, p)).collect();
        index
            .apply_stored(0, &blocks_of(&held), None, &mut map)
            .expect("store");
        index.remove_worker(0, map);
        assert_eq!(index.worker_id("a"), None);
        assert_eq!(
            index.intern_worker("c"),
            Ok(0),
            "the freed slot is handed out again"
        );
        assert_eq!(
            scores(&index, &held),
            vec![],
            "nothing of the old holder survives"
        );
        assert_eq!(index.intern_worker("a"), Err(WorkerIdExhausted));
    }
}
