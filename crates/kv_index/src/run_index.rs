//! A run-compressed KV index: the event-driven prefix index as a tree of runs with per-run
//! worker coverage bitsets, lock-free and allocation-free for readers.
//!
//! The index answers the same question as [`PositionalIndexer`](crate::PositionalIndexer): for a
//! request given as its per-block content hashes, how many leading blocks does each worker hold,
//! where "holds" means the worker stored, at every position up to there, the block that sits on
//! the request's chain. It is fed by the same engine events (stored / removed / cleared, keyed by
//! the engine's block hashes) and keeps the same per-worker block map the gateway's event monitor
//! owns, so it drops into the same call sites. Its results are checked against
//! [`ReferenceIndexer`](crate::ReferenceIndexer) by `tests/exactness_run.rs`, including evictions
//! that leave holes in a chain.
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
//!   take no locks, allocate nothing but the result map and touch no shared cache line in write
//!   mode: every pointer they follow is an [`ArcSwap`] load (a store into a thread-local debt
//!   slot), coverage words are plain atomic loads, and a run's hashes, length and children are
//!   published together as one body that only grows in place, so a split is atomic to a reader.
//! - **Writers** (the event lanes, one per engine worker) lock one run at a time, plus its parent
//!   for the moment it takes to unlink an empty run. A decode extension of the worker's own leaf
//!   appends in place: one lock, no allocation. The lane's own map takes an engine hash to
//!   `(run, offset)`; a split leaves a forwarding record on the prefix run so entries written
//!   before the split still resolve, which keeps other lanes' maps untouched.
//!
//! Engine hashes: the index trusts the engine's parent pointers and block identities, as the
//! positional indexer does. Nothing is shared between workers through the maps, so one engine
//! reusing a hash cannot corrupt another worker's view.
//!
//! Memory: 8 bytes per distinct block on a chain (its content hash, shared by every worker that
//! holds it) plus a run header, against the per-worker map entry each lane keeps for removals.
//! Runs are allocated from an append-only slab and are not recycled yet; a split or an emptied run
//! leaves its header in place (see [`RunIndex::stats`]).

use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use arc_swap::{ArcSwap, Guard};
use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::event_tree::{
    chain_prefix_hash, ApplyError, ContentHash, OverlapScores, SequenceHash, StoredBlock,
    WorkerIdExhausted,
};

/// The virtual root: position 0's parent, holds no blocks.
const ROOT: u32 = 0;
/// Runs per slab chunk.
const CHUNK_RUNS: usize = 1024;
/// Smallest hash array a run allocates.
const MIN_HASH_CAP: usize = 8;
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

/// Backing store for content hashes, shared by the runs a split produces: each run reads its own
/// window `[base, base + len)`; only the run whose window ends at `used` may append in place.
struct HashBlock {
    data: Box<[AtomicU64]>,
    used: AtomicU32,
}

impl HashBlock {
    fn new(contents: &[u64], capacity: usize) -> Arc<Self> {
        let capacity = capacity.max(contents.len()).max(MIN_HASH_CAP);
        let data: Box<[AtomicU64]> = (0..capacity).map(|_| AtomicU64::new(0)).collect();
        for (slot, &hash) in data.iter().zip(contents) {
            slot.store(hash, Ordering::Relaxed);
        }
        Arc::new(Self {
            data,
            used: AtomicU32::new(contents.len() as u32),
        })
    }

    #[inline]
    fn get(&self, index: usize) -> u64 {
        self.data[index].load(Ordering::Relaxed)
    }

    fn capacity(&self) -> usize {
        self.data.len()
    }
}

/// A run's children, sorted by the content hash of their first block. Replaced wholesale on
/// change (children are few, inserts are rare next to lookups).
struct ChildTable {
    entries: Box<[(u64, u32)]>,
}

impl ChildTable {
    fn one(head: u64, run: u32) -> Arc<Self> {
        Arc::new(Self {
            entries: vec![(head, run)].into_boxed_slice(),
        })
    }

    #[inline]
    fn find(&self, head: u64) -> Option<u32> {
        if self.entries.len() <= 4 {
            return self
                .entries
                .iter()
                .find(|entry| entry.0 == head)
                .map(|entry| entry.1);
        }
        self.entries
            .binary_search_by_key(&head, |entry| entry.0)
            .ok()
            .map(|index| self.entries[index].1)
    }

    fn with(&self, head: u64, run: u32) -> Arc<Self> {
        let mut entries: Vec<(u64, u32)> = self.entries.to_vec();
        match entries.binary_search_by_key(&head, |entry| entry.0) {
            Ok(index) => entries[index] = (head, run),
            Err(index) => entries.insert(index, (head, run)),
        }
        Arc::new(Self {
            entries: entries.into_boxed_slice(),
        })
    }

    fn without(&self, run: u32) -> Option<Arc<Self>> {
        let entries: Box<[(u64, u32)]> = self
            .entries
            .iter()
            .copied()
            .filter(|entry| entry.1 != run)
            .collect();
        if entries.is_empty() {
            None
        } else {
            Some(Arc::new(Self { entries }))
        }
    }
}

/// What a reader needs of a run, replaced as one unit so a reader never sees a half-applied
/// split. `len` grows in place on append (published with a release store after the hashes);
/// everything else is immutable once published.
struct RunBody {
    block: Arc<HashBlock>,
    base: u32,
    len: AtomicU32,
    children: Option<Arc<ChildTable>>,
}

impl RunBody {
    fn new(
        block: Arc<HashBlock>,
        base: usize,
        len: usize,
        children: Option<Arc<ChildTable>>,
    ) -> Self {
        Self {
            block,
            base: base as u32,
            len: AtomicU32::new(len as u32),
            children,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    #[inline]
    fn hash(&self, offset: usize) -> u64 {
        self.block.get(self.base as usize + offset)
    }

    fn contents(&self, from: usize, to: usize) -> Vec<u64> {
        (from..to).map(|offset| self.hash(offset)).collect()
    }

    #[inline]
    fn child(&self, head: u64) -> Option<u32> {
        self.children.as_ref().and_then(|table| table.find(head))
    }

    fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// The same window with other children.
    fn with_children(&self, children: Option<Arc<ChildTable>>) -> Self {
        Self::new(self.block.clone(), self.base as usize, self.len(), children)
    }
}

/// Writer-side bookkeeping of a run, under its lock.
#[derive(Default)]
struct RunMeta {
    /// Splits this run has undergone, oldest first: `(offset, suffix run)`. A block that sat at
    /// `offset >= o` before the split lives in the suffix at `offset - o` (and may have been
    /// forwarded again from there). Offsets decrease along the vector: a run never grows after a
    /// split.
    splits: Vec<(u32, u32)>,
    /// Unlinked from the tree; kept so forwarding records still resolve.
    dead: bool,
}

struct Run {
    /// Absolute position of the run's first block.
    start: AtomicU32,
    /// Run id of the parent (the root's parent is itself).
    parent: AtomicU32,
    body: ArcSwap<RunBody>,
    /// One bit per interned worker.
    coverage: Box<[AtomicU64]>,
    meta: Mutex<RunMeta>,
}

impl Run {
    fn blank(words: usize, empty: &Arc<HashBlock>) -> Self {
        Self {
            start: AtomicU32::new(0),
            parent: AtomicU32::new(ROOT),
            body: ArcSwap::from_pointee(RunBody::new(empty.clone(), 0, 0, None)),
            coverage: (0..words).map(|_| AtomicU64::new(0)).collect(),
            meta: Mutex::new(RunMeta::default()),
        }
    }

    #[inline]
    fn start(&self) -> usize {
        self.start.load(Ordering::Relaxed) as usize
    }

    #[inline]
    fn has(&self, worker: u32) -> bool {
        self.coverage[(worker / 64) as usize].load(Ordering::Relaxed) & (1u64 << (worker % 64)) != 0
    }

    fn set(&self, worker: u32) {
        self.coverage[(worker / 64) as usize].fetch_or(1u64 << (worker % 64), Ordering::Relaxed);
    }

    fn clear(&self, worker: u32) {
        self.coverage[(worker / 64) as usize]
            .fetch_and(!(1u64 << (worker % 64)), Ordering::Relaxed);
    }

    fn coverage_is_empty(&self) -> bool {
        self.coverage
            .iter()
            .all(|word| word.load(Ordering::Relaxed) == 0)
    }

    /// Exactly `worker` and nobody else.
    fn covered_only_by(&self, worker: u32) -> bool {
        let word = (worker / 64) as usize;
        let bit = 1u64 << (worker % 64);
        self.coverage.iter().enumerate().all(|(index, slot)| {
            let value = slot.load(Ordering::Relaxed);
            if index == word {
                value == bit
            } else {
                value == 0
            }
        })
    }

    fn workers(&self) -> Vec<u32> {
        self.coverage
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
}

struct RunChunk {
    runs: Box<[Run]>,
}

type Chunks = Guard<Arc<Vec<Arc<RunChunk>>>>;

/// Append-only run storage: a vector of fixed-size chunks, replaced copy-on-write when a chunk is
/// added (rare), each chunk's runs addressed in place. Run ids are never reused.
struct RunSlab {
    chunks: ArcSwap<Vec<Arc<RunChunk>>>,
    next: AtomicU32,
    grow: Mutex<()>,
    words: usize,
    empty: Arc<HashBlock>,
}

impl RunSlab {
    fn new(words: usize) -> Self {
        let empty = HashBlock::new(&[], MIN_HASH_CAP);
        // Nobody may append into the shared empty block.
        empty.used.store(MIN_HASH_CAP as u32, Ordering::Relaxed);
        let slab = Self {
            chunks: ArcSwap::from_pointee(Vec::new()),
            next: AtomicU32::new(0),
            grow: Mutex::new(()),
            words,
            empty,
        };
        slab.ensure_chunk(0);
        slab
    }

    fn ensure_chunk(&self, chunk: usize) {
        if self.chunks.load().len() > chunk {
            return;
        }
        let _grow = self.grow.lock();
        let current = self.chunks.load_full();
        if current.len() > chunk {
            return;
        }
        let mut next: Vec<Arc<RunChunk>> = (*current).clone();
        while next.len() <= chunk {
            let runs: Box<[Run]> = (0..CHUNK_RUNS)
                .map(|_| Run::blank(self.words, &self.empty))
                .collect();
            next.push(Arc::new(RunChunk { runs }));
        }
        self.chunks.store(Arc::new(next));
    }

    /// A fresh run, not yet reachable from the tree.
    fn alloc(&self, start: usize, parent: u32, body: RunBody) -> u32 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.ensure_chunk(id as usize / CHUNK_RUNS);
        let chunks = self.chunks.load();
        let run = slab_run(&chunks, id);
        run.start.store(start as u32, Ordering::Relaxed);
        run.parent.store(parent, Ordering::Relaxed);
        run.body.store(Arc::new(body));
        id
    }

    fn allocated(&self) -> usize {
        self.next.load(Ordering::Relaxed) as usize
    }
}

#[inline]
fn slab_run(chunks: &Chunks, id: u32) -> &Run {
    let id = id as usize;
    &chunks[id / CHUNK_RUNS].runs[id % CHUNK_RUNS]
}

/// Reload the chunk vector if `id` was allocated after it was loaded.
#[inline]
fn ensure(slab: &RunSlab, chunks: &mut Chunks, id: u32) {
    if id as usize / CHUNK_RUNS >= chunks.len() {
        *chunks = slab.chunks.load();
    }
}

/// Like [`RunIndex::resolve`], returning with the final run locked so nothing can move the block
/// before the caller uses it. `None` when a run lies outside `chunks`.
fn resolve_locked(
    chunks: &Chunks,
    mut at: BlockRef,
) -> Option<(BlockRef, MutexGuard<'_, RunMeta>)> {
    loop {
        if at.run as usize / CHUNK_RUNS >= chunks.len() {
            return None;
        }
        let meta = slab_run(chunks, at.run).meta.lock();
        match forward(&meta.splits, at) {
            Some(next) => at = next,
            None => return Some((at, meta)),
        }
    }
}

/// Where a block that sat at `at` before this run's splits lives now, one hop.
fn forward(splits: &[(u32, u32)], at: BlockRef) -> Option<BlockRef> {
    splits
        .iter()
        .find(|(offset, _)| *offset <= at.offset)
        .map(|&(offset, suffix)| BlockRef {
            run: suffix,
            offset: at.offset - offset,
        })
}

/// Memory and shape counters, for the scoreboard.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunIndexStats {
    /// Runs ever allocated (headers resident).
    pub runs_allocated: usize,
    /// Runs still linked in the tree.
    pub runs_live: usize,
    /// Content hashes held by live runs.
    pub blocks_live: usize,
    /// Bytes of hash arrays referenced by live runs (capacity, counted once per shared array).
    pub hash_bytes: usize,
    /// Bytes of run headers, coverage words included (all allocated runs).
    pub header_bytes: usize,
}

/// The run-compressed index. Worker ids are interned `u32`s, as in the positional indexer.
pub struct RunIndex {
    slab: RunSlab,
    words: usize,
    max_workers: usize,
    worker_to_id: DashMap<Arc<str>, u32, FxBuildHasher>,
    registry: Mutex<WorkerRegistry>,
    worker_blocks: Box<[AtomicUsize]>,
    total_blocks: AtomicUsize,
    distinct_blocks: AtomicUsize,
}

impl Default for RunIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Worker slots: a slot is in use from `intern_worker` until `remove_worker`, after which it is
/// handed out again (every coverage bit of a removed worker is clear by then).
#[derive(Default)]
struct WorkerRegistry {
    names: Vec<Option<Arc<str>>>,
    free: Vec<u32>,
}

/// Outcome of one attempt to walk a store into the tree.
enum Walk {
    Done,
    /// The walk met a run another lane unlinked meanwhile; start over from the parent block.
    Restart,
}

/// What [`RunIndex::store_in_run`] found.
enum InRun<'b> {
    /// Every block is placed.
    Done,
    /// The blocks still to place start right after the run's last block.
    Continue(&'b [StoredBlock]),
    /// Carry on in this run from its first block.
    MoveTo(u32),
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
        let root = slab.alloc(0, ROOT, RunBody::new(slab.empty.clone(), 0, 0, None));
        debug_assert_eq!(root, ROOT);
        Self {
            slab,
            words,
            max_workers,
            worker_to_id: DashMap::with_hasher(FxBuildHasher),
            registry: Mutex::new(WorkerRegistry::default()),
            worker_blocks: (0..max_workers).map(|_| AtomicUsize::new(0)).collect(),
            total_blocks: AtomicUsize::new(0),
            distinct_blocks: AtomicUsize::new(0),
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
        self.total_blocks.load(Ordering::Relaxed)
    }

    /// Distinct blocks held by at least one worker.
    pub fn entry_count(&self) -> usize {
        self.distinct_blocks.load(Ordering::Relaxed)
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
        self.total_blocks.fetch_add(blocks, Ordering::Relaxed);
    }

    fn debit(&self, worker: u32, blocks: usize) {
        if blocks == 0 {
            return;
        }
        self.worker_blocks[worker as usize].fetch_sub(blocks, Ordering::Relaxed);
        self.total_blocks.fetch_sub(blocks, Ordering::Relaxed);
    }

    /// Follow split forwarding records to where a block lives now, without keeping any lock.
    fn resolve(&self, chunks: &mut Chunks, mut at: BlockRef) -> BlockRef {
        loop {
            ensure(&self.slab, chunks, at.run);
            let meta = slab_run(chunks, at.run).meta.lock();
            match forward(&meta.splits, at) {
                Some(next) => at = next,
                None => return at,
            }
        }
    }

    /// Split `run` (locked by the caller) at `at`: the run keeps `[0, at)`, a new suffix run takes
    /// `[at, len)` on the same hash array, the run's children and its coverage minus `exclude`.
    /// The suffix is published as the run's only child when it holds anything; a suffix nobody
    /// covers and that has no children is dead on arrival but still receives the forwarding so
    /// entries written before the split resolve to a run that reports no coverage.
    fn split_locked(
        &self,
        run_id: u32,
        meta: &mut RunMeta,
        at: usize,
        exclude: Option<u32>,
    ) -> u32 {
        let chunks = self.slab.chunks.load();
        let run = slab_run(&chunks, run_id);
        let body = run.body.load_full();
        let len = body.len();
        debug_assert!(at > 0 && at < len, "split inside the run: 0 < {at} < {len}");
        let suffix_body = RunBody::new(
            body.block.clone(),
            body.base as usize + at,
            len - at,
            body.children.clone(),
        );
        let suffix_id = self.slab.alloc(run.start() + at, run_id, suffix_body);
        let chunks = self.slab.chunks.load();
        let run = slab_run(&chunks, run_id);
        let suffix = slab_run(&chunks, suffix_id);
        for (index, word) in run.coverage.iter().enumerate() {
            let mut value = word.load(Ordering::Relaxed);
            if let Some(worker) = exclude {
                if (worker / 64) as usize == index {
                    value &= !(1u64 << (worker % 64));
                }
            }
            suffix.coverage[index].store(value, Ordering::Relaxed);
        }
        if let Some(children) = &body.children {
            for &(_, child) in &children.entries {
                slab_run(&chunks, child)
                    .parent
                    .store(suffix_id, Ordering::Release);
            }
        }
        let uncovered = suffix.coverage_is_empty();
        if uncovered && !run.coverage_is_empty() {
            // The excluded worker was the last holder of these blocks.
            self.distinct_blocks.fetch_sub(len - at, Ordering::Relaxed);
        }
        let children = if !uncovered || body.children.is_some() {
            Some(ChildTable::one(body.hash(at), suffix_id))
        } else {
            suffix.meta.lock().dead = true;
            None
        };
        run.body.store(Arc::new(RunBody::new(
            body.block.clone(),
            body.base as usize,
            at,
            children,
        )));
        meta.splits.push((at as u32, suffix_id));
        suffix_id
    }

    /// Unlink `run` (locked by the caller, known to be an uncovered leaf) from its parent, and
    /// then the parent if that leaves it an uncovered leaf too.
    fn unlink_locked(&self, run_id: u32, meta: &mut RunMeta) {
        if run_id == ROOT || meta.dead {
            return;
        }
        let chunks = self.slab.chunks.load();
        let run = slab_run(&chunks, run_id);
        loop {
            let parent_id = run.parent.load(Ordering::Acquire);
            let parent = slab_run(&chunks, parent_id);
            // Child-then-parent is the only order in which two run locks are ever held. A split
            // of the parent may have re-parented this run while we waited; check and retry.
            let mut parent_meta = parent.meta.lock();
            if run.parent.load(Ordering::Acquire) != parent_id {
                continue;
            }
            meta.dead = true;
            let body = parent.body.load_full();
            let children = body
                .children
                .as_ref()
                .and_then(|table| table.without(run_id));
            let parent_is_leaf = children.is_none();
            parent.body.store(Arc::new(body.with_children(children)));
            if parent_id != ROOT && parent_is_leaf && parent.coverage_is_empty() {
                self.unlink_locked(parent_id, &mut parent_meta);
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
            if let Walk::Done = self.store_walk(worker, blocks, origin, map) {
                return Ok(());
            }
        }
    }

    /// One attempt to place a store, from the parent block (or the root) down the tree.
    fn store_walk(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        origin: Option<(SequenceHash, BlockRef)>,
        map: &mut RunBlockMap,
    ) -> Walk {
        let mut chunks = self.slab.chunks.load();
        let (mut run_id, mut offset, mut meta) = match origin {
            None => (ROOT, 0usize, slab_run(&chunks, ROOT).meta.lock()),
            Some((hash, at)) => {
                let (at, meta) = loop {
                    // A statement, so the `None` temporary is gone before the reload below.
                    if let Some(found) = resolve_locked(&chunks, at) {
                        break found;
                    }
                    chunks = self.slab.chunks.load();
                };
                map.insert(hash, at);
                (at.run, at.offset as usize + 1, meta)
            }
        };
        let mut remaining = blocks;
        loop {
            if meta.dead {
                return Walk::Restart;
            }
            let run = slab_run(&chunks, run_id);
            let next =
                match self.store_in_run(worker, run_id, run, &mut meta, offset, remaining, map) {
                    InRun::Done => return Walk::Done,
                    InRun::Continue(rest) => {
                        remaining = rest;
                        self.store_at_end(worker, run_id, run, &meta, remaining, map)
                    }
                    InRun::MoveTo(child) => Some(child),
                };
            let Some(child) = next else {
                return Walk::Done;
            };
            drop(meta);
            run_id = child;
            offset = 0;
            ensure(&self.slab, &mut chunks, run_id);
            meta = slab_run(&chunks, run_id).meta.lock();
        }
    }

    /// Match `remaining` against the run from `offset`: join or split the run as needed, record
    /// the matched blocks, and say how to go on.
    #[expect(clippy::too_many_arguments)]
    fn store_in_run<'b>(
        &self,
        worker: u32,
        run_id: u32,
        run: &Run,
        meta: &mut RunMeta,
        offset: usize,
        remaining: &'b [StoredBlock],
        map: &mut RunBlockMap,
    ) -> InRun<'b> {
        let mut body = run.body.load_full();
        let len = body.len();
        if offset >= len {
            return InRun::Continue(remaining);
        }
        let covered = run.has(worker);
        if !covered && offset > 0 {
            // The parent entry pointed into a run this worker does not cover (it cannot, unless
            // the engine re-stored under a stale parent). Cut here and join the suffix instead.
            let suffix = self.split_locked(run_id, meta, offset, None);
            return InRun::MoveTo(suffix);
        }
        let available = len - offset;
        let matched = remaining
            .iter()
            .zip(offset..len)
            .take_while(|(stored, index)| stored.content_hash.0 == body.hash(*index))
            .count();
        if matched < available && (matched < remaining.len() || !covered) {
            // A divergence inside the run (both branches stay held), or a worker that holds only
            // a prefix of it: either way the run ends here.
            self.split_locked(run_id, meta, offset + matched, None);
            body = run.body.load_full();
        }
        if !covered {
            if run.coverage_is_empty() {
                self.distinct_blocks
                    .fetch_add(body.len(), Ordering::Relaxed);
            }
            run.set(worker);
            self.credit(worker, body.len());
        }
        for (index, stored) in remaining[..matched].iter().enumerate() {
            map.insert(
                stored.seq_hash,
                BlockRef {
                    run: run_id,
                    offset: (offset + index) as u32,
                },
            );
        }
        if matched == remaining.len() {
            return InRun::Done;
        }
        InRun::Continue(&remaining[matched..])
    }

    /// `remaining` starts right after the run's last block: descend into the child that continues
    /// it, append to the worker's own leaf, or open a new run. Returns the child to descend into.
    fn store_at_end(
        &self,
        worker: u32,
        run_id: u32,
        run: &Run,
        meta: &RunMeta,
        remaining: &[StoredBlock],
        map: &mut RunBlockMap,
    ) -> Option<u32> {
        let body = run.body.load_full();
        let head = remaining[0].content_hash.0;
        if let Some(child) = body.child(head) {
            return Some(child);
        }
        let len = body.len();
        let contents: Vec<u64> = remaining
            .iter()
            .map(|stored| stored.content_hash.0)
            .collect();
        let own_leaf = run_id != ROOT
            && body.is_leaf()
            && meta.splits.is_empty()
            && run.covered_only_by(worker);
        let (target, first) = if own_leaf {
            append(run, &body, &contents);
            (run_id, len)
        } else {
            let capacity = (contents.len() * 2).max(MIN_HASH_CAP);
            let new_body =
                RunBody::new(HashBlock::new(&contents, capacity), 0, contents.len(), None);
            let new_id = self.slab.alloc(run.start() + len, run_id, new_body);
            let chunks = self.slab.chunks.load();
            slab_run(&chunks, new_id).set(worker);
            let children = match &body.children {
                Some(table) => table.with(head, new_id),
                None => ChildTable::one(head, new_id),
            };
            run.body.store(Arc::new(body.with_children(Some(children))));
            (new_id, 0)
        };
        for (index, stored) in remaining.iter().enumerate() {
            map.insert(
                stored.seq_hash,
                BlockRef {
                    run: target,
                    offset: (first + index) as u32,
                },
            );
        }
        self.credit(worker, contents.len());
        self.distinct_blocks
            .fetch_add(contents.len(), Ordering::Relaxed);
        None
    }

    /// Forget the named blocks of `worker`; unknown hashes are ignored.
    pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], map: &mut RunBlockMap) {
        let mut chunks = self.slab.chunks.load();
        let mut groups: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        for hash in hashes {
            if let Some(at) = map.remove(hash) {
                let at = self.resolve(&mut chunks, at);
                groups.entry(at.run).or_default().push(at.offset);
            }
        }
        let mut work: Vec<(u32, Vec<u32>)> = groups.into_iter().collect();
        while let Some((run_id, offsets)) = work.pop() {
            ensure(&self.slab, &mut chunks, run_id);
            let run = slab_run(&chunks, run_id);
            let mut meta = run.meta.lock();
            let offsets = reforward(&meta, offsets, &mut work);
            if offsets.is_empty() || !run.has(worker) {
                continue;
            }
            self.remove_ranges(worker, run_id, run, &mut meta, offsets);
            if run.coverage_is_empty() && run.body.load().is_leaf() {
                self.unlink_locked(run_id, &mut meta);
            }
        }
    }

    /// Drop `worker` from the given offsets of a run it covers, splitting the run so that the
    /// pieces it still covers keep their offsets.
    fn remove_ranges(
        &self,
        worker: u32,
        run_id: u32,
        run: &Run,
        meta: &mut RunMeta,
        mut offsets: Vec<u32>,
    ) {
        offsets.sort_unstable();
        offsets.dedup();
        let len = run.body.load().len();
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
            let len = run.body.load().len();
            if high + 1 < len {
                // The tail beyond the range keeps every worker, this one included.
                self.split_locked(run_id, meta, high + 1, None);
            }
            if low > 0 {
                // The range becomes its own run without this worker.
                self.split_locked(run_id, meta, low, Some(worker));
            } else {
                self.clear_holder(run, worker);
            }
            self.debit(worker, high + 1 - low);
        }
    }

    /// Drop `worker` from a run it covers, keeping the distinct-block count in step.
    fn clear_holder(&self, run: &Run, worker: u32) {
        run.clear(worker);
        if run.coverage_is_empty() {
            self.distinct_blocks
                .fetch_sub(run.body.load().len(), Ordering::Relaxed);
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
        let mut chunks = self.slab.chunks.load();
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut work: Vec<u32> = Vec::new();
        for (_, at) in map {
            let at = self.resolve(&mut chunks, at);
            if seen.insert(at.run) {
                work.push(at.run);
            }
        }
        while let Some(run_id) = work.pop() {
            ensure(&self.slab, &mut chunks, run_id);
            let run = slab_run(&chunks, run_id);
            let mut meta = run.meta.lock();
            // Blocks of this worker may have moved into suffixes since the map was resolved.
            for &(_, suffix) in &meta.splits {
                if seen.insert(suffix) {
                    work.push(suffix);
                }
            }
            if run.has(worker) {
                self.clear_holder(run, worker);
                self.debit(worker, run.body.load().len());
                if run.coverage_is_empty() && run.body.load().is_leaf() {
                    self.unlink_locked(run_id, &mut meta);
                }
            }
        }
    }

    /// Score every worker by how many leading blocks of the request it holds. With `early_exit`,
    /// report the workers holding the first block, each scored 1.
    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        let mut out = OverlapScores::default();
        let Some(&first) = content_hashes.first() else {
            return out;
        };
        let mut chunks = self.slab.chunks.load();
        let Some(mut run_id) = slab_run(&chunks, ROOT).body.load().child(first.0) else {
            return out;
        };
        let words = self.words;
        let mut alive = [0u64; MAX_WORDS];
        let mut position = 0usize;
        loop {
            ensure(&self.slab, &mut chunks, run_id);
            let run = slab_run(&chunks, run_id);
            let body = run.body.load();
            let len = body.len();
            let available = len.min(content_hashes.len() - position);
            let matched = (0..available)
                .take_while(|&index| content_hashes[position + index].0 == body.hash(index))
                .count();
            if matched == 0 {
                break;
            }
            if position == 0 {
                for (word, slot) in alive[..words].iter_mut().zip(run.coverage.iter()) {
                    *word = slot.load(Ordering::Relaxed);
                }
                if early_exit {
                    emit(&alive[..words], 1, &mut out);
                    return out;
                }
            } else {
                for (index, word) in alive[..words].iter_mut().enumerate() {
                    let coverage = run.coverage[index].load(Ordering::Relaxed);
                    let dropped = *word & !coverage;
                    if dropped != 0 {
                        emit_word(index, dropped, position as u32, &mut out);
                    }
                    *word &= coverage;
                }
            }
            if alive[..words].iter().all(|word| *word == 0) {
                return out;
            }
            position += matched;
            if matched < len || position == content_hashes.len() {
                break;
            }
            match body.child(content_hashes[position].0) {
                Some(child) => run_id = child,
                None => break,
            }
        }
        emit(&alive[..words], position as u32, &mut out);
        out
    }

    /// Every block every worker holds, as `(worker, position, content hash, prefix hash)`;
    /// for tests and for comparing against the reference indexer.
    #[doc(hidden)]
    pub fn debug_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        let mut out = BTreeSet::new();
        let mut chunks = self.slab.chunks.load();
        let mut stack: Vec<(u32, Option<SequenceHash>)> = Vec::new();
        if let Some(table) = &slab_run(&chunks, ROOT).body.load().children {
            for &(_, child) in &table.entries {
                stack.push((child, None));
            }
        }
        while let Some((run_id, mut prefix)) = stack.pop() {
            ensure(&self.slab, &mut chunks, run_id);
            let run = slab_run(&chunks, run_id);
            let body = run.body.load_full();
            let start = run.start();
            let workers = run.workers();
            for offset in 0..body.len() {
                let content = ContentHash(body.hash(offset));
                let next = match prefix {
                    Some(previous) => chain_prefix_hash(previous, content),
                    None => SequenceHash(content.0),
                };
                for &worker in &workers {
                    out.insert((worker, start + offset, content, next));
                }
                prefix = Some(next);
            }
            if let Some(table) = &body.children {
                for &(_, child) in &table.entries {
                    stack.push((child, prefix));
                }
            }
        }
        out
    }

    /// Shape and memory counters.
    pub fn stats(&self) -> RunIndexStats {
        let chunks = self.slab.chunks.load();
        let allocated = self.slab.allocated();
        let mut stats = RunIndexStats {
            runs_allocated: allocated,
            header_bytes: allocated
                * (size_of::<Run>() + size_of::<RunBody>() + self.words * size_of::<AtomicU64>()),
            ..RunIndexStats::default()
        };
        let mut arrays: FxHashSet<usize> = FxHashSet::default();
        for id in 1..allocated as u32 {
            let run = slab_run(&chunks, id);
            if run.meta.lock().dead {
                continue;
            }
            let body = run.body.load();
            stats.runs_live += 1;
            stats.blocks_live += body.len();
            if arrays.insert(Arc::as_ptr(&body.block) as usize) {
                stats.hash_bytes += body.block.capacity() * size_of::<AtomicU64>();
            }
        }
        stats
    }
}

/// Extend a leaf in place when its window ends the shared array and the array has room; otherwise
/// move it to a larger array.
fn append(run: &Run, body: &Arc<RunBody>, contents: &[u64]) {
    let len = body.len();
    let end = body.base as usize + len;
    let fits = end + contents.len() <= body.block.capacity();
    let claimed = fits
        && body
            .block
            .used
            .compare_exchange(
                end as u32,
                (end + contents.len()) as u32,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
    if claimed {
        for (index, &hash) in contents.iter().enumerate() {
            body.block.data[end + index].store(hash, Ordering::Relaxed);
        }
        body.len
            .store((len + contents.len()) as u32, Ordering::Release);
        return;
    }
    let mut grown = body.contents(0, len);
    grown.extend_from_slice(contents);
    let capacity = grown.len() * 2;
    run.body.store(Arc::new(RunBody::new(
        HashBlock::new(&grown, capacity),
        0,
        grown.len(),
        None,
    )));
}

/// Offsets resolved without the lock may have moved into suffixes since: send those on.
fn reforward(meta: &RunMeta, mut offsets: Vec<u32>, work: &mut Vec<(u32, Vec<u32>)>) -> Vec<u32> {
    if meta.splits.is_empty() {
        return offsets;
    }
    let mut forwarded: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    offsets.retain(
        |&offset| match forward(&meta.splits, BlockRef { run: ROOT, offset }) {
            Some(next) => {
                forwarded.entry(next.run).or_default().push(next.offset);
                false
            }
            None => true,
        },
    );
    work.extend(forwarded);
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
        assert_eq!(index.stats().runs_live, 0);
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
