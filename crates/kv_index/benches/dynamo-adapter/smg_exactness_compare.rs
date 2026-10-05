// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How often `ConcurrentRadixTreeCompressed` (CRTC) disagrees with SMG's single-threaded
//! `ReferenceIndexer` on identical event streams, with and without holes (a worker evicts a
//! block from the middle of a chain and keeps the blocks after it).
//!
//! The corpus generator is a port of SMG's exactness harness
//! (`crates/kv_index/tests/exactness.rs`): the same xorshift stream, the same operations in
//! the same proportions (new chains, extensions, divergent siblings, tail / whole-chain /
//! middle removals, clears, worker removals) and the same lookup kinds, so a seed produces
//! the same corpus here as there. Every event goes to three indexers:
//!
//! - CRTC, through `SyncIndexer::worker` running on a dedicated lane thread. Each event is
//!   a `WorkerTask::EventWithAck` and the test waits for its ack (the thread pool's
//!   `apply_event_and_wait` path); worker removal is `WorkerTask::RemoveWorker` without and
//!   then with `sweep_tree`, as `ThreadPoolIndexer` sends it; a `WorkerTask::Flush` precedes
//!   every lookup round. Lookups run on the test thread through
//!   `SyncIndexer::find_matches(.., false)`, the thread pool's inline read path. With every
//!   event acked before the next one is sent, the tree is quiescent at every lookup.
//! - SMG's `ReferenceIndexer`, the oracle.
//! - SMG's production `PositionalIndexer`, a control that must report zero mismatches.
//!
//! Only full score maps (worker -> matched blocks) with `early_exit = false` are compared.
//! CRTC's `early_exit` stops the walk once one rank remains; SMG's reports every position-0
//! holder with score 1; the two contracts are not comparable, so early exit is left out.
//!
//! Scale with `KV_INDEX_EXACTNESS_EVENTS` (default 20000), `SMG_EXACTNESS_SEEDS`
//! (comma-separated, default `20261005,20261006`) and `SMG_EXACTNESS_LARGE_EVENTS` (default
//! 80000, hole corpus, first seed only).
#![cfg(feature = "smg-backend")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    sync::Arc,
    thread::JoinHandle,
};

use dynamo_kv_router::{
    indexer::{SyncIndexer, WorkerTask},
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
    },
    ConcurrentRadixTreeCompressed,
};
use kv_index::{
    request_prefix_hashes, ContentHash, PositionalIndexer, ReferenceIndexer, SequenceHash,
    StoredBlock, WorkerBlockMap,
};
use rustc_hash::FxHashMap;
use tokio::sync::oneshot;

// ---------------------------------------------------------------------------------------
// Corpus primitives, identical to SMG's harness
// ---------------------------------------------------------------------------------------

/// xorshift64*, the same generator as SMG's harness, so a seed yields the same corpus.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn range(&mut self, lo: usize, hi_inclusive: usize) -> usize {
        lo + self.below(hi_inclusive - lo + 1)
    }

    fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.next() % denominator < numerator
    }
}

fn content(stream: u64, position: usize) -> ContentHash {
    kv_index::compute_content_hash(&[
        (stream & 0xffff_ffff) as u32,
        (stream >> 32) as u32,
        position as u32,
    ])
}

/// Blocks of a content sequence as an engine would hash them: the engine hash is the chain
/// hash of the contents so far, unique per distinct prefix and shared by every worker that
/// stores the same prefix. SMG `SequenceHash` = Dynamo `ExternalSequenceBlockHash`, SMG
/// `ContentHash` = Dynamo `LocalBlockHash`.
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

fn stored_event(parent: Option<SequenceHash>, blocks: &[StoredBlock]) -> KvCacheEventData {
    KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash: parent.map(|hash| ExternalSequenceBlockHash(hash.0)),
        start_position: None,
        blocks: blocks
            .iter()
            .map(|block| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(block.seq_hash.0),
                tokens_hash: LocalBlockHash(block.content_hash.0),
                mm_extra_info: None,
            })
            .collect(),
    })
}

fn removed_event(hashes: &[SequenceHash]) -> KvCacheEventData {
    KvCacheEventData::Removed(KvCacheRemoveData {
        block_hashes: hashes
            .iter()
            .map(|hash| ExternalSequenceBlockHash(hash.0))
            .collect(),
    })
}

// ---------------------------------------------------------------------------------------
// One CRTC event lane
// ---------------------------------------------------------------------------------------

/// A CRTC tree plus the thread running `SyncIndexer::worker` for its single lane. SMG's
/// interned `u32` worker ids are used as Dynamo worker ids with `dp_rank` 0; SMG never
/// recycles an id, so a removed worker never comes back under the same Dynamo id either.
struct CrtcLane {
    tree: Arc<ConcurrentRadixTreeCompressed>,
    sender: flume::Sender<WorkerTask>,
    thread: Option<JoinHandle<()>>,
    next_event_id: u64,
}

impl CrtcLane {
    fn new() -> Self {
        let tree = Arc::new(ConcurrentRadixTreeCompressed::new());
        let (sender, receiver) = flume::unbounded();
        let thread = {
            let tree = Arc::clone(&tree);
            std::thread::spawn(move || tree.worker(receiver, None).expect("CRTC lane"))
        };
        Self {
            tree,
            sender,
            thread: Some(thread),
            next_event_id: 0,
        }
    }

    /// Applies one event on the lane and waits for its ack: whether CRTC applied it.
    fn apply(&mut self, worker: u32, data: KvCacheEventData) -> bool {
        let event_id = self.next_event_id;
        self.next_event_id += 1;
        let event = RouterEvent::new(
            u64::from(worker),
            KvCacheEvent {
                event_id,
                data,
                dp_rank: 0,
            },
        );
        let (resp, ack) = oneshot::channel();
        self.sender
            .send(WorkerTask::EventWithAck { event, resp })
            .expect("CRTC lane alive");
        ack.blocking_recv().expect("CRTC lane acked the event")
    }

    fn flush(&self) {
        let (resp, ack) = oneshot::channel();
        self.sender
            .send(WorkerTask::Flush(resp))
            .expect("CRTC lane alive");
        ack.blocking_recv().expect("CRTC lane flushed");
    }

    /// `ThreadPoolIndexer::remove_worker` sends `RemoveWorker` without `sweep_tree` to every
    /// lane and then with it to lane 0; with one lane that is the same lane twice.
    fn remove_worker(&self, worker: u32) {
        for sweep_tree in [false, true] {
            let (resp, ack) = oneshot::channel();
            self.sender
                .send(WorkerTask::RemoveWorker {
                    worker_id: u64::from(worker),
                    sweep_tree,
                    resp,
                })
                .expect("CRTC lane alive");
            ack.blocking_recv().expect("CRTC lane removed the worker");
        }
    }

    fn scores(&self, query: &[ContentHash]) -> BTreeMap<u32, u32> {
        let hashes: Vec<LocalBlockHash> = query.iter().map(|hash| LocalBlockHash(hash.0)).collect();
        self.tree
            .find_matches(&hashes, false)
            .scores
            .into_iter()
            .map(|(worker, score)| {
                assert_eq!(worker.dp_rank, 0, "every event was sent with dp_rank 0");
                (
                    u32::try_from(worker.worker_id).expect("worker ids are SMG u32 ids"),
                    score,
                )
            })
            .collect()
    }
}

impl Drop for CrtcLane {
    fn drop(&mut self) {
        let _ = self.sender.send(WorkerTask::Terminate);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

// ---------------------------------------------------------------------------------------
// Tallies
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QueryKind {
    Exact,
    Prefix,
    MiddleReplaced,
    SuffixReplaced,
    Extended,
    Unknown,
    /// A full chain through a block the worker evicted from its middle.
    AcrossHole,
    /// A prefix of such a chain that ends after the hole.
    PastHole,
}

const ALL_KINDS: [QueryKind; 8] = [
    QueryKind::Exact,
    QueryKind::Prefix,
    QueryKind::MiddleReplaced,
    QueryKind::SuffixReplaced,
    QueryKind::Extended,
    QueryKind::Unknown,
    QueryKind::AcrossHole,
    QueryKind::PastHole,
];

/// Per-kind CRTC tally. `under` and `over` count lookups with at least one under-counted
/// (CRTC lower, or worker missing) or over-counted (CRTC higher, or phantom worker) worker;
/// a lookup with both counts in both.
#[derive(Default, Clone, Copy)]
struct KindTally {
    issued: usize,
    mismatched: usize,
    under: usize,
    over: usize,
}

impl KindTally {
    fn add(&mut self, other: &KindTally) {
        self.issued += other.issued;
        self.mismatched += other.mismatched;
        self.under += other.under;
        self.over += other.over;
    }
}

#[derive(Default)]
struct Mismatches {
    lookups: usize,
    crtc: BTreeMap<QueryKind, KindTally>,
    /// (lookup, worker) pairs CRTC under- or over-counted.
    crtc_worker_under: usize,
    crtc_worker_over: usize,
    /// Blocks CRTC failed to credit, summed over those pairs, and blocks it over-credited.
    crtc_blocks_under: usize,
    crtc_blocks_over: usize,
    /// Across-hole lookups whose reference score passes the hole: the worker re-stored the
    /// evicted block, so the chain is reachable again in the engine. CRTC under-counts the
    /// worker on `refilled_under` of them.
    refilled_holes: usize,
    refilled_under: usize,
    over_examples: Vec<String>,
    under_examples: Vec<String>,
    control_mismatched: usize,
    control_by_kind: BTreeMap<QueryKind, usize>,
    control_examples: Vec<String>,
}

impl Mismatches {
    fn crtc_total(&self) -> (usize, usize, usize) {
        self.crtc.values().fold((0, 0, 0), |(m, u, o), tally| {
            (m + tally.mismatched, u + tally.under, o + tally.over)
        })
    }
}

/// How a CRTC score map differs from the reference: workers under-counted (CRTC lower, or
/// missing) and over-counted (CRTC higher, or phantom), and the blocks lost and gained.
#[derive(Default, Clone, Copy)]
struct Diff {
    under_workers: usize,
    over_workers: usize,
    under_blocks: usize,
    over_blocks: usize,
}

fn classify(expected: &BTreeMap<u32, u32>, got: &BTreeMap<u32, u32>) -> Diff {
    let mut diff = Diff::default();
    for worker in expected.keys().chain(got.keys()).collect::<BTreeSet<_>>() {
        let want = expected.get(worker).copied().unwrap_or(0);
        let have = got.get(worker).copied().unwrap_or(0);
        if have < want {
            diff.under_workers += 1;
            diff.under_blocks += (want - have) as usize;
        } else if have > want {
            diff.over_workers += 1;
            diff.over_blocks += (have - want) as usize;
        }
    }
    diff
}

fn diff_line(expected: &BTreeMap<u32, u32>, got: &BTreeMap<u32, u32>, label: &str) -> String {
    expected
        .keys()
        .chain(got.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|worker| expected.get(worker) != got.get(worker))
        .map(|worker| {
            format!(
                "worker {worker}: {label} {:?}, reference {:?}",
                got.get(worker),
                expected.get(worker)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

// ---------------------------------------------------------------------------------------
// The harness, ported from SMG with CRTC added as a third indexer
// ---------------------------------------------------------------------------------------

struct Held {
    worker: u32,
    contents: Vec<ContentHash>,
}

/// A chain as it was when one of its middle blocks was evicted; its blocks after `hole` stay
/// in the index but are unreachable through the chain, so lookups along it must stop at
/// `hole`.
struct Holed {
    worker: u32,
    contents: Vec<ContentHash>,
    hole: usize,
}

struct Harness {
    production: PositionalIndexer,
    reference: ReferenceIndexer,
    crtc: CrtcLane,
    maps: FxHashMap<u32, WorkerBlockMap>,
    workers: Vec<u32>,
    held: Vec<Held>,
    holes: bool,
    holed: Vec<Holed>,
    prompts: Vec<Vec<ContentHash>>,
    next_stream: u64,
    next_worker: u32,
    rng: Rng,
    events: usize,
    stored_blocks: usize,
    crtc_stores: usize,
    /// Store events CRTC rejected (every rejection on this path is `ParentBlockNotFound`:
    /// `store.rs` returns no other error once the rank has a slot).
    crtc_store_errors: usize,
    crtc_removes: usize,
    /// Remove events CRTC rejected (`BlockNotFound`: the rank is unknown to the lane).
    crtc_remove_errors: usize,
}

impl Harness {
    fn new(seed: u64, jump_size: usize, workers: usize, holes: bool) -> Self {
        let mut rng = Rng::new(seed);
        let production = PositionalIndexer::new(jump_size);
        let mut harness = Self {
            production,
            reference: ReferenceIndexer::new(),
            crtc: CrtcLane::new(),
            maps: FxHashMap::default(),
            workers: Vec::new(),
            held: Vec::new(),
            holes,
            holed: Vec::new(),
            prompts: Vec::new(),
            next_stream: 1,
            next_worker: 0,
            rng: Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15),
            events: 0,
            stored_blocks: 0,
            crtc_stores: 0,
            crtc_store_errors: 0,
            crtc_removes: 0,
            crtc_remove_errors: 0,
        };
        for _ in 0..workers {
            harness.add_worker();
        }
        let prompt_count = rng.range(3, 8);
        for _ in 0..prompt_count {
            let len = rng.range(8, 48);
            let stream = harness.fresh_stream();
            harness
                .prompts
                .push((0..len).map(|p| content(stream, p)).collect());
        }
        harness
    }

    fn fresh_stream(&mut self) -> u64 {
        self.next_stream += 1;
        self.next_stream
    }

    fn add_worker(&mut self) -> u32 {
        let url = format!("http://worker-{}:8000", self.next_worker);
        self.next_worker += 1;
        let id = self.production.intern_worker(&url).expect("worker id");
        self.maps.insert(id, WorkerBlockMap::default());
        self.workers.push(id);
        id
    }

    fn random_worker(&mut self) -> u32 {
        self.workers[self.rng.below(self.workers.len())]
    }

    /// A new user turn: a prompt prefix plus fresh blocks.
    fn new_chain(&mut self) -> Vec<ContentHash> {
        let prompt = &self.prompts[self.rng.below(self.prompts.len())];
        let mut contents = prompt.clone();
        let turn = self.rng.range(4, 32);
        let stream = self.fresh_stream();
        contents.extend((0..turn).map(|p| content(stream, p)));
        contents
    }

    /// Store `contents` on `worker` as the engine would: only the suffix the worker does not
    /// hold, after the last block it does hold. A fully held chain is re-stored for its last
    /// block, which exercises the duplicate-store path. The engine's view of what the worker
    /// holds is SMG's per-worker block map, maintained by the production indexer; CRTC gets
    /// the same event, so a store CRTC rejects leaves the engine and CRTC disagreeing from
    /// then on, as it would in production.
    fn store(&mut self, worker: u32, contents: Vec<ContentHash>) {
        let blocks = blocks_of(&contents);
        let held = self.maps.get(&worker).expect("worker map");
        let mut known = 0;
        while known < blocks.len() && held.contains_key(&blocks[known].seq_hash) {
            known += 1;
        }
        let start = if known == blocks.len() {
            blocks.len() - 1
        } else {
            known
        };
        let parent = if start == 0 {
            None
        } else {
            Some(blocks[start - 1].seq_hash)
        };
        let map = self.maps.get_mut(&worker).expect("worker map");
        let produced = self
            .production
            .apply_stored(worker, &blocks[start..], parent, map);
        let referenced = self
            .reference
            .apply_stored(worker, &blocks[start..], parent);
        assert_eq!(
            produced.is_ok(),
            referenced.is_ok(),
            "store outcome differs: production {produced:?}, reference {referenced:?}"
        );
        let applied = self
            .crtc
            .apply(worker, stored_event(parent, &blocks[start..]));
        self.crtc_stores += 1;
        if !applied {
            self.crtc_store_errors += 1;
        }
        if produced.is_ok() {
            self.stored_blocks += blocks.len() - start;
            self.held.push(Held { worker, contents });
        }
        self.events += 1;
    }

    fn remove(&mut self, worker: u32, hashes: &[SequenceHash]) {
        let map = self.maps.get_mut(&worker).expect("worker map");
        self.production.apply_removed(worker, hashes, map);
        self.reference.apply_removed(worker, hashes);
        let applied = self.crtc.apply(worker, removed_event(hashes));
        self.crtc_removes += 1;
        if !applied {
            self.crtc_remove_errors += 1;
        }
    }

    fn pick_held(&mut self) -> Option<usize> {
        if self.held.is_empty() {
            return None;
        }
        let index = self.rng.below(self.held.len());
        if self.maps.contains_key(&self.held[index].worker) {
            Some(index)
        } else {
            self.held.swap_remove(index);
            None
        }
    }

    /// Evict a tail: the chain's blocks from a random position on, together with the same
    /// positions of every other held chain of that worker that runs through them.
    fn remove_tail(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let keep = self.rng.below(contents.len());
        let mut hashes: Vec<SequenceHash> = Vec::new();
        for held in self.held.iter_mut().filter(|h| h.worker == worker) {
            let shared = held
                .contents
                .iter()
                .zip(&contents)
                .take_while(|(a, b)| a == b)
                .count();
            if shared > keep {
                let blocks = blocks_of(&held.contents);
                hashes.extend(blocks[keep..].iter().map(|b| b.seq_hash));
                held.contents.truncate(keep);
            }
        }
        hashes.sort_unstable_by_key(|h| h.0);
        hashes.dedup();
        self.remove(worker, &hashes);
        self.held.retain(|h| !h.contents.is_empty());
        self.events += 1;
    }

    /// Evict a whole conversation: the chain's blocks beyond the longest prefix it shares
    /// with another held chain of the same worker.
    fn remove_chain(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let shared = self
            .held
            .iter()
            .enumerate()
            .filter(|(i, h)| *i != index && h.worker == worker)
            .map(|(_, h)| {
                h.contents
                    .iter()
                    .zip(&contents)
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .max()
            .unwrap_or(0);
        let blocks = blocks_of(&contents);
        let hashes: Vec<SequenceHash> = blocks[shared..].iter().map(|b| b.seq_hash).collect();
        self.remove(worker, &hashes);
        self.held.swap_remove(index);
        self.events += 1;
    }

    /// Evict one block from the middle of a chain while the blocks after it stay (the vLLM
    /// case). Every held chain of the worker that runs through the block loses it; the full
    /// chains are kept aside so lookups can run across the hole.
    fn remove_middle(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        if self.held[index].contents.len() < 3 {
            return;
        }
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let hole = self.rng.range(1, contents.len() - 2);
        let evicted = blocks_of(&contents)[hole].seq_hash;
        for held in self.held.iter_mut().filter(|h| h.worker == worker) {
            let shared = held
                .contents
                .iter()
                .zip(&contents)
                .take_while(|(a, b)| a == b)
                .count();
            if shared > hole {
                self.holed.push(Holed {
                    worker,
                    contents: held.contents.clone(),
                    hole,
                });
                held.contents.truncate(hole);
            }
        }
        self.remove(worker, &[evicted]);
        self.events += 1;
    }

    fn pick_holed(&mut self) -> Option<usize> {
        if self.holed.is_empty() {
            return None;
        }
        let index = self.rng.below(self.holed.len());
        if self.maps.contains_key(&self.holed[index].worker) {
            Some(index)
        } else {
            self.holed.swap_remove(index);
            None
        }
    }

    fn clear_worker(&mut self) {
        let worker = self.random_worker();
        let map = self.maps.get_mut(&worker).expect("worker map");
        self.production.apply_cleared(worker, map);
        self.reference.apply_cleared(worker);
        let applied = self.crtc.apply(worker, KvCacheEventData::Cleared);
        assert!(applied, "CRTC never rejects a clear");
        self.held.retain(|h| h.worker != worker);
        self.events += 1;
    }

    fn remove_worker(&mut self) {
        if self.workers.len() < 2 {
            return;
        }
        let position = self.rng.below(self.workers.len());
        let worker = self.workers.swap_remove(position);
        let map = self.maps.remove(&worker).expect("worker map");
        self.production.remove_worker(worker, map);
        self.reference.remove_worker(worker);
        self.crtc.remove_worker(worker);
        self.held.retain(|h| h.worker != worker);
        self.add_worker();
        self.events += 1;
    }

    fn step(&mut self) {
        let roll = self.rng.below(1000);
        match roll {
            0..=399 => {
                let chain = self.new_chain();
                let worker = self.random_worker();
                self.store(worker, chain);
            }
            400..=649 => {
                // Extend a held chain by another turn on the same worker.
                let Some(index) = self.pick_held() else {
                    return;
                };
                let worker = self.held[index].worker;
                let mut contents = self.held[index].contents.clone();
                let turn = self.rng.range(4, 32);
                let stream = self.fresh_stream();
                contents.extend((0..turn).map(|p| content(stream, p)));
                self.store(worker, contents);
            }
            650..=799 => {
                // A sibling that diverges at a random position, including 1 and the last
                // block, stored on a random worker (often another one, which shares the
                // prefix blocks).
                let Some(index) = self.pick_held() else {
                    return;
                };
                let base = &self.held[index].contents;
                if base.len() < 2 {
                    return;
                }
                let divergence = match self.rng.below(10) {
                    0 => 1,
                    1 => base.len() - 1,
                    2 => base.len(),
                    _ => self.rng.range(1, base.len()),
                };
                let mut contents: Vec<ContentHash> = base[..divergence].to_vec();
                let turn = self.rng.range(1, 24);
                let stream = self.fresh_stream();
                contents.extend((0..turn).map(|p| content(stream, p)));
                let worker = self.random_worker();
                self.store(worker, contents);
            }
            800..=899 => {
                if self.holes && self.rng.chance(1, 2) {
                    self.remove_middle();
                } else {
                    self.remove_tail();
                }
            }
            900..=949 => self.remove_chain(),
            950..=984 => {
                if self.workers.len() < 64 && self.rng.chance(1, 4) {
                    self.add_worker();
                } else {
                    let Some(index) = self.pick_held() else {
                        return;
                    };
                    let contents = self.held[index].contents.clone();
                    let worker = self.random_worker();
                    self.store(worker, contents);
                }
            }
            985..=994 => self.clear_worker(),
            _ => self.remove_worker(),
        }
    }

    fn production_scores(&self, query: &[ContentHash]) -> BTreeMap<u32, u32> {
        self.production
            .find_matches(query, false)
            .scores
            .into_iter()
            .collect()
    }

    /// Checks one lookup against the reference and returns the reference and CRTC maps.
    fn check_lookup(
        &mut self,
        kind: QueryKind,
        query: Vec<ContentHash>,
        out: &mut Mismatches,
    ) -> (BTreeMap<u32, u32>, BTreeMap<u32, u32>) {
        if query.is_empty() {
            return (BTreeMap::new(), BTreeMap::new());
        }
        out.lookups += 1;
        let expected = self.reference.find_matches(&query);
        let control = self.production_scores(&query);
        if control != expected {
            out.control_mismatched += 1;
            *out.control_by_kind.entry(kind).or_default() += 1;
            if out.control_examples.len() < 5 {
                out.control_examples.push(format!(
                    "{kind:?} query of {} blocks after {} events: {}",
                    query.len(),
                    self.events,
                    diff_line(&expected, &control, "PositionalIndexer")
                ));
            }
        }
        let crtc = self.crtc.scores(&query);
        let diff = if crtc == expected {
            Diff::default()
        } else {
            classify(&expected, &crtc)
        };
        let (under, over) = (diff.under_workers, diff.over_workers);
        if over > 0 && out.over_examples.len() < 3 {
            let hashes: Vec<String> = query
                .iter()
                .map(|hash| format!("{:#018x}", hash.0))
                .collect();
            out.over_examples.push(format!(
                "{kind:?} query of {} blocks after {} events\n    query hashes: [{}]\n    reference: {expected:?}\n    crtc:      {crtc:?}",
                query.len(),
                self.events,
                hashes.join(", ")
            ));
        }
        if under > 0 && out.under_examples.len() < 3 {
            out.under_examples.push(format!(
                "{kind:?} query of {} blocks after {} events: {}",
                query.len(),
                self.events,
                diff_line(&expected, &crtc, "crtc")
            ));
        }
        out.crtc_worker_under += under;
        out.crtc_worker_over += over;
        out.crtc_blocks_under += diff.under_blocks;
        out.crtc_blocks_over += diff.over_blocks;
        let tally = out.crtc.entry(kind).or_default();
        tally.issued += 1;
        if under > 0 || over > 0 {
            tally.mismatched += 1;
        }
        if under > 0 {
            tally.under += 1;
        }
        if over > 0 {
            tally.over += 1;
        }
        (expected, crtc)
    }

    fn lookups(&mut self, out: &mut Mismatches) {
        // The thread pool's flush barrier; every event is already acked, so this only
        // drains the lane's epoch garbage before the reads.
        self.crtc.flush();
        for _ in 0..8 {
            let Some(index) = self.pick_held() else {
                return;
            };
            let base = self.held[index].contents.clone();
            if base.is_empty() {
                continue;
            }
            self.check_lookup(QueryKind::Exact, base.clone(), out);
            let prefix_len = self.rng.range(1, base.len());
            self.check_lookup(QueryKind::Prefix, base[..prefix_len].to_vec(), out);
            if base.len() >= 2 {
                let mut middle = base.clone();
                let at = self.rng.range(1, base.len() - 1);
                let stream = self.fresh_stream();
                middle[at] = content(stream, 0);
                self.check_lookup(QueryKind::MiddleReplaced, middle, out);
                let mut suffix = base[..self.rng.range(1, base.len() - 1)].to_vec();
                let stream = self.fresh_stream();
                let extra = self.rng.range(1, 16);
                suffix.extend((0..extra).map(|p| content(stream, p)));
                self.check_lookup(QueryKind::SuffixReplaced, suffix, out);
            }
            let mut extended = base.clone();
            let stream = self.fresh_stream();
            let extra = self.rng.range(1, 40);
            extended.extend((0..extra).map(|p| content(stream, p)));
            self.check_lookup(QueryKind::Extended, extended, out);
        }
        if self.holes {
            for _ in 0..4 {
                let Some(index) = self.pick_holed() else {
                    break;
                };
                let (full, hole, worker) = {
                    let holed = &self.holed[index];
                    (holed.contents.clone(), holed.hole, holed.worker)
                };
                let past = self.rng.range(hole + 1, full.len());
                self.check_lookup(QueryKind::PastHole, full[..past].to_vec(), out);
                let (expected, crtc) = self.check_lookup(QueryKind::AcrossHole, full, out);
                let want = expected.get(&worker).copied().unwrap_or(0) as usize;
                if want > hole {
                    out.refilled_holes += 1;
                    if (crtc.get(&worker).copied().unwrap_or(0) as usize) < want {
                        out.refilled_under += 1;
                    }
                }
            }
        }
        let stream = self.fresh_stream();
        let len = self.rng.range(1, 32);
        let unknown: Vec<ContentHash> = (0..len).map(|p| content(stream, p)).collect();
        self.check_lookup(QueryKind::Unknown, unknown, out);
    }
}

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn seeds() -> Vec<u64> {
    std::env::var("SMG_EXACTNESS_SEEDS")
        .ok()
        .map(|list| {
            list.split(',')
                .map(|seed| seed.trim().parse().expect("SMG_EXACTNESS_SEEDS: u64 list"))
                .collect()
        })
        .unwrap_or_else(|| vec![20_261_005, 20_261_006])
}

fn run_corpus(
    seed: u64,
    jump_size: usize,
    workers: usize,
    events: usize,
    holes: bool,
) -> (Harness, Mismatches) {
    let mut harness = Harness::new(seed, jump_size, workers, holes);
    let mut mismatches = Mismatches::default();
    while harness.events < events {
        harness.step();
        if harness.events.is_multiple_of(256) {
            harness.lookups(&mut mismatches);
        }
    }
    harness.lookups(&mut mismatches);
    (harness, mismatches)
}

// ---------------------------------------------------------------------------------------
// Result rows and rendering
// ---------------------------------------------------------------------------------------

struct Row {
    corpus: &'static str,
    seed: u64,
    jump: usize,
    workers: usize,
    events: usize,
    stored_blocks: usize,
    lookups: usize,
    crtc_mismatched: usize,
    crtc_under: usize,
    crtc_over: usize,
    crtc_worker_under: usize,
    crtc_worker_over: usize,
    crtc_blocks_under: usize,
    crtc_blocks_over: usize,
    refilled_holes: usize,
    refilled_under: usize,
    crtc_stores: usize,
    crtc_store_errors: usize,
    crtc_removes: usize,
    crtc_remove_errors: usize,
    control_mismatched: usize,
    by_kind: BTreeMap<QueryKind, KindTally>,
    over_examples: Vec<String>,
    under_examples: Vec<String>,
    control_examples: Vec<String>,
}

impl Row {
    fn new(
        corpus: &'static str,
        seed: u64,
        jump: usize,
        workers: usize,
        harness: &Harness,
        mismatches: Mismatches,
    ) -> Self {
        let (crtc_mismatched, crtc_under, crtc_over) = mismatches.crtc_total();
        Self {
            corpus,
            seed,
            jump,
            workers,
            events: harness.events,
            stored_blocks: harness.stored_blocks,
            lookups: mismatches.lookups,
            crtc_mismatched,
            crtc_under,
            crtc_over,
            crtc_worker_under: mismatches.crtc_worker_under,
            crtc_worker_over: mismatches.crtc_worker_over,
            crtc_blocks_under: mismatches.crtc_blocks_under,
            crtc_blocks_over: mismatches.crtc_blocks_over,
            refilled_holes: mismatches.refilled_holes,
            refilled_under: mismatches.refilled_under,
            crtc_stores: harness.crtc_stores,
            crtc_store_errors: harness.crtc_store_errors,
            crtc_removes: harness.crtc_removes,
            crtc_remove_errors: harness.crtc_remove_errors,
            control_mismatched: mismatches.control_mismatched,
            by_kind: mismatches.crtc,
            over_examples: mismatches.over_examples,
            under_examples: mismatches.under_examples,
            control_examples: mismatches.control_examples,
        }
    }

    fn label(&self) -> String {
        format!(
            "{} seed {} jump {} workers {} events {}",
            self.corpus, self.seed, self.jump, self.workers, self.events
        )
    }
}

fn pct(part: usize, whole: usize) -> String {
    if whole == 0 {
        "-".to_string()
    } else {
        format!("{:.2}%", 100.0 * part as f64 / whole as f64)
    }
}

fn render(title: &str, rows: &[Row]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\n## {title}\n");
    let _ = writeln!(
        out,
        "| corpus | seed | jump/workers | events | stored blocks | lookups | CRTC mismatched (under / over) | CRTC (lookup,worker) under / over | CRTC blocks lost / gained | re-filled holes seen / CRTC under on them | CRTC store errors (ParentBlockNotFound) | CRTC remove errors | SMG PositionalIndexer mismatched |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for row in rows {
        let _ = writeln!(
            out,
            "| {} | {} | {}/{} | {} | {} | {} | {} ({}) ({} / {}) | {} / {} | {} / {} | {} / {} | {} / {} ({}) | {} / {} | {} |",
            row.corpus,
            row.seed,
            row.jump,
            row.workers,
            row.events,
            row.stored_blocks,
            row.lookups,
            row.crtc_mismatched,
            pct(row.crtc_mismatched, row.lookups),
            row.crtc_under,
            row.crtc_over,
            row.crtc_worker_under,
            row.crtc_worker_over,
            row.crtc_blocks_under,
            row.crtc_blocks_over,
            row.refilled_holes,
            row.refilled_under,
            row.crtc_store_errors,
            row.crtc_stores,
            pct(row.crtc_store_errors, row.crtc_stores),
            row.crtc_remove_errors,
            row.crtc_removes,
            row.control_mismatched,
        );
    }

    // Per kind, aggregated over the (jump, workers) configurations of one corpus x seed x
    // events cell.
    let mut cells: BTreeMap<(&'static str, u64, usize), BTreeMap<QueryKind, KindTally>> =
        BTreeMap::new();
    for row in rows {
        let cell = cells.entry((row.corpus, row.seed, row.events)).or_default();
        for (kind, tally) in &row.by_kind {
            cell.entry(*kind).or_default().add(tally);
        }
    }
    let _ = writeln!(
        out,
        "\n### CRTC mismatches by lookup kind (summed over the three jump/workers configurations)\n"
    );
    let _ = write!(out, "| corpus | seed | events |");
    for kind in ALL_KINDS {
        let _ = write!(out, " {kind:?} issued / mismatched (under / over) |");
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "|---|---|---|{}", "---|".repeat(ALL_KINDS.len()));
    for ((corpus, seed, events), cell) in &cells {
        let _ = write!(out, "| {corpus} | {seed} | {events} |");
        for kind in ALL_KINDS {
            let tally = cell.get(&kind).copied().unwrap_or_default();
            if tally.issued == 0 {
                let _ = write!(out, " - |");
            } else {
                let _ = write!(
                    out,
                    " {} / {} ({} / {}) |",
                    tally.issued, tally.mismatched, tally.under, tally.over
                );
            }
        }
        let _ = writeln!(out);
    }

    for row in rows {
        if !row.over_examples.is_empty() {
            let _ = writeln!(
                out,
                "\n### CRTC over-counts, {} (first {})\n",
                row.label(),
                row.over_examples.len()
            );
            for example in &row.over_examples {
                let _ = writeln!(out, "- {example}");
            }
        }
    }
    for row in rows {
        if !row.under_examples.is_empty() {
            let _ = writeln!(
                out,
                "\n### CRTC under-count examples, {} (first {})\n",
                row.label(),
                row.under_examples.len()
            );
            for example in &row.under_examples {
                let _ = writeln!(out, "- {example}");
            }
        }
    }
    for row in rows {
        if !row.control_examples.is_empty() {
            let _ = writeln!(
                out,
                "\n### SMG PositionalIndexer mismatches, {} (first {})\n",
                row.label(),
                row.control_examples.len()
            );
            for example in &row.control_examples {
                let _ = writeln!(out, "- {example}");
            }
        }
    }
    out
}

const CONFIGS: [(usize, usize); 3] = [(8, 16), (64, 2), (3, 64)];

fn run_rows(corpus: &'static str, holes: bool, seeds: &[u64], events: usize) -> Vec<Row> {
    let mut rows = Vec::new();
    for &seed in seeds {
        for (jump, workers) in CONFIGS {
            let (harness, mismatches) =
                run_corpus(seed ^ jump as u64, jump, workers, events, holes);
            assert!(
                harness.stored_blocks > events,
                "corpus too small to mean anything: {} blocks for {} events",
                harness.stored_blocks,
                events
            );
            if holes {
                let across = mismatches
                    .crtc
                    .get(&QueryKind::AcrossHole)
                    .map_or(0, |tally| tally.issued);
                assert!(
                    across >= 64,
                    "hole corpus too small to mean anything: {across} lookups across holes"
                );
            }
            rows.push(Row::new(corpus, seed, jump, workers, &harness, mismatches));
        }
    }
    rows
}

fn assert_control_exact(rows: &[Row]) {
    for row in rows {
        assert_eq!(
            row.control_mismatched,
            0,
            "SMG PositionalIndexer must match the reference ({}): {:?}",
            row.label(),
            row.control_examples
        );
    }
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

/// Tail and whole-chain evictions only (what a radix-tree cache such as SGLang's produces).
/// CRTC is expected to be exact here.
#[test]
fn hole_free_corpus_crtc_matches_the_reference() {
    let events = env_or("KV_INDEX_EXACTNESS_EVENTS", 20_000) as usize;
    let rows = run_rows("hole-free", false, &seeds(), events);
    println!("{}", render("Hole-free corpus", &rows));
    assert_control_exact(&rows);
    for row in &rows {
        assert_eq!(
            row.crtc_mismatched,
            0,
            "CRTC disagreed with the reference on the hole-free corpus ({}): under {}, over {}; {:?} {:?}",
            row.label(),
            row.crtc_under,
            row.crtc_over,
            row.under_examples,
            row.over_examples
        );
    }
}

/// The vLLM corpus: middle blocks get evicted while later blocks stay, and lookups run
/// across and past the holes. Mismatches are measured, not asserted away; the only
/// assertions are the control's exactness and CRTC's own rule that it never over-counts.
#[test]
fn hole_corpus_crtc_versus_the_reference() {
    let events = env_or("KV_INDEX_EXACTNESS_EVENTS", 20_000) as usize;
    let large = env_or("SMG_EXACTNESS_LARGE_EVENTS", 80_000) as usize;
    let seeds = seeds();
    let mut rows = run_rows("holes", true, &seeds, events);
    if large > events {
        rows.extend(run_rows("holes", true, &seeds[..1], large));
    }
    println!("{}", render("Hole corpus", &rows));
    assert_control_exact(&rows);
    for row in &rows {
        assert_eq!(
            row.crtc_over,
            0,
            "CRTC over-counted on the hole corpus ({}): {:?}",
            row.label(),
            row.over_examples
        );
    }
}

/// One lookup of the focused case: a table row plus the control and over-count checks.
#[allow(clippy::too_many_arguments)]
fn check_focused(
    out: &mut String,
    failures: &mut Vec<String>,
    worker: u32,
    phase: &str,
    label: &str,
    query: &[ContentHash],
    lane: &CrtcLane,
    production: &PositionalIndexer,
    reference: &ReferenceIndexer,
) {
    let expected = reference.find_matches(query);
    let control: BTreeMap<u32, u32> = production
        .find_matches(query, false)
        .scores
        .into_iter()
        .collect();
    let crtc = lane.scores(query);
    let _ = writeln!(
        out,
        "| {phase} | {label} ({} blocks) | {:?} | {:?} | {:?} |",
        query.len(),
        expected.get(&worker),
        control.get(&worker),
        crtc.get(&worker)
    );
    assert_eq!(
        control, expected,
        "{phase} {label}: SMG PositionalIndexer must match the reference"
    );
    if classify(&expected, &crtc).over_workers > 0 {
        failures.push(format!(
            "{phase} {label}: CRTC over-counted: reference {expected:?}, crtc {crtc:?}"
        ));
    }
}

/// A worker holds [A ..= J] (ten blocks) and evicts E while F ..= J stay, as vLLM's free
/// queue can order it. A request for the full chain hits A ..= D in the engine and
/// recomputes the rest, so the score is 4. Then the engine re-stores E (a request re-hit
/// A ..= D and recomputed E; F ..= J were still cached), which makes the whole chain
/// reachable again: the reference scores 10. Finally the engine appends K after J.
#[test]
fn evicted_middle_block_ends_the_match() {
    let mut lane = CrtcLane::new();
    let production = PositionalIndexer::new(8);
    let mut reference = ReferenceIndexer::new();
    let worker = production
        .intern_worker("http://worker-0:8000")
        .expect("worker id");
    let mut map = WorkerBlockMap::default();
    let contents: Vec<ContentHash> = (0..11).map(|p| content(7, p)).collect();
    let blocks = blocks_of(&contents);
    let ten = &blocks[..10];

    let mut out = String::new();
    let _ = writeln!(out, "\n## Focused case: ten blocks, E evicted, F..J kept\n");
    let _ = writeln!(
        out,
        "| phase | query | reference | SMG PositionalIndexer | CRTC |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|");
    let mut failures = Vec::new();

    production
        .apply_stored(worker, ten, None, &mut map)
        .expect("store");
    reference.apply_stored(worker, ten, None).expect("store");
    assert!(lane.apply(worker, stored_event(None, ten)));
    production.apply_removed(worker, &[ten[4].seq_hash], &mut map);
    reference.apply_removed(worker, &[ten[4].seq_hash]);
    assert!(lane.apply(worker, removed_event(&[ten[4].seq_hash])));
    lane.flush();
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "1 after evicting E",
        "A..J",
        &contents[..10],
        &lane,
        &production,
        &reference,
    );
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "1 after evicting E",
        "A..G",
        &contents[..7],
        &lane,
        &production,
        &reference,
    );
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "1 after evicting E",
        "A..D",
        &contents[..4],
        &lane,
        &production,
        &reference,
    );

    // The engine re-stores E after D: a request matched A..D, recomputed E, and F..J were
    // still cached, so only E is new.
    production
        .apply_stored(worker, &ten[4..5], Some(ten[3].seq_hash), &mut map)
        .expect("re-store E");
    reference
        .apply_stored(worker, &ten[4..5], Some(ten[3].seq_hash))
        .expect("re-store E");
    let restore_applied = lane.apply(worker, stored_event(Some(ten[3].seq_hash), &ten[4..5]));
    lane.flush();
    let _ = writeln!(
        out,
        "| 2 re-store E after D | CRTC applied the store: {restore_applied} | | | |"
    );
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "2 after re-storing E",
        "A..J",
        &contents[..10],
        &lane,
        &production,
        &reference,
    );
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "2 after re-storing E",
        "A..G",
        &contents[..7],
        &lane,
        &production,
        &reference,
    );
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "2 after re-storing E",
        "A..D",
        &contents[..4],
        &lane,
        &production,
        &reference,
    );

    // The engine appends K after J, which it still holds.
    production
        .apply_stored(worker, &blocks[10..11], Some(ten[9].seq_hash), &mut map)
        .expect("store K");
    reference
        .apply_stored(worker, &blocks[10..11], Some(ten[9].seq_hash))
        .expect("store K");
    let append_applied = lane.apply(worker, stored_event(Some(ten[9].seq_hash), &blocks[10..11]));
    lane.flush();
    let _ = writeln!(
        out,
        "| 3 store K after J | CRTC applied the store: {append_applied} | | | |"
    );
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "3 after storing K",
        "A..K",
        &contents,
        &lane,
        &production,
        &reference,
    );
    check_focused(
        &mut out,
        &mut failures,
        worker,
        "3 after storing K",
        "A..J",
        &contents[..10],
        &lane,
        &production,
        &reference,
    );

    println!("{out}");
    assert!(failures.is_empty(), "{failures:?}");
}

/// Why most re-filled holes in the corpus score exactly and a few do not. Both cases
/// evict E from a chain the worker keeps the rest of and then re-store E..H through a new
/// chain that diverges after H.
///
/// - `edge ends at H`: another worker's divergent store had already split the edge after
///   H, so the re-fill covers the rest of the holed edge and CRTC promotes the worker back
///   to full coverage there; the child edge [I ..= L] still carries the worker's full bit
///   from before the hole, so the whole chain scores.
/// - `edge runs past H`: the holed edge is [A ..= L] in one node; the re-fill diverges
///   inside it, so the split suffix [I ..= L] inherits no coverage for the worker and the
///   chain scores 8 of 12 for good.
#[test]
fn refilled_hole_scores_fully_only_when_the_refill_reaches_the_edge_end() {
    let mut out = String::new();
    let _ = writeln!(out, "\n## Re-filled holes: edge boundary decides\n");
    let _ = writeln!(
        out,
        "| case | phase | query | reference | SMG PositionalIndexer | CRTC |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|");
    let mut failures = Vec::new();

    for case in ["edge ends at H", "edge runs past H"] {
        let mut lane = CrtcLane::new();
        let production = PositionalIndexer::new(8);
        let mut reference = ReferenceIndexer::new();
        let w = production
            .intern_worker("http://worker-0:8000")
            .expect("worker id");
        let v = production
            .intern_worker("http://worker-1:8000")
            .expect("worker id");
        let (mut map_w, mut map_v) = (WorkerBlockMap::default(), WorkerBlockMap::default());
        // A ..= L, then a divergent sibling X after J, then a new turn Y after H.
        let chain: Vec<ContentHash> = (0..12).map(|p| content(11, p)).collect();
        let blocks = blocks_of(&chain);
        let mut sibling: Vec<ContentHash> = chain[..8].to_vec();
        sibling.push(content(12, 0));
        let sibling_blocks = blocks_of(&sibling);
        let mut refill: Vec<ContentHash> = chain[..8].to_vec();
        refill.push(content(13, 0));
        let refill_blocks = blocks_of(&refill);

        let store = |worker: u32,
                     map: &mut WorkerBlockMap,
                     parent: Option<SequenceHash>,
                     stored: &[StoredBlock],
                     lane: &mut CrtcLane,
                     reference: &mut ReferenceIndexer| {
            production
                .apply_stored(worker, stored, parent, map)
                .expect("store");
            reference
                .apply_stored(worker, stored, parent)
                .expect("store");
            lane.apply(worker, stored_event(parent, stored))
        };

        // w holds A ..= J, then extends with K, L (one compressed edge in CRTC).
        assert!(store(
            w,
            &mut map_w,
            None,
            &blocks[..10],
            &mut lane,
            &mut reference
        ));
        assert!(store(
            w,
            &mut map_w,
            Some(blocks[9].seq_hash),
            &blocks[10..12],
            &mut lane,
            &mut reference
        ));
        if case == "edge ends at H" {
            // v's sibling splits the edge after H: [A..H] -> {[I..L] (w), [X] (v)}.
            assert!(store(
                v,
                &mut map_v,
                None,
                &sibling_blocks,
                &mut lane,
                &mut reference
            ));
        }
        // w evicts E and keeps everything after it.
        production.apply_removed(w, &[blocks[4].seq_hash], &mut map_w);
        reference.apply_removed(w, &[blocks[4].seq_hash]);
        assert!(lane.apply(w, removed_event(&[blocks[4].seq_hash])));
        lane.flush();
        let label = |phase: &str| format!("{case} | {phase}");
        check_focused(
            &mut out,
            &mut failures,
            w,
            &label("after evicting E"),
            "A..L",
            &chain,
            &lane,
            &production,
            &reference,
        );
        // A new turn on w re-hits A ..= D, recomputes E ..= H and continues with Y: the
        // engine stores E ..= H, Y after D.
        let applied = store(
            w,
            &mut map_w,
            Some(blocks[3].seq_hash),
            &refill_blocks[4..],
            &mut lane,
            &mut reference,
        );
        lane.flush();
        let _ = writeln!(
            out,
            "| {case} | re-store E..H,Y after D | CRTC applied the store: {applied} | | | |"
        );
        check_focused(
            &mut out,
            &mut failures,
            w,
            &label("after the re-fill"),
            "A..L",
            &chain,
            &lane,
            &production,
            &reference,
        );
        check_focused(
            &mut out,
            &mut failures,
            w,
            &label("after the re-fill"),
            "A..H,Y",
            &refill,
            &lane,
            &production,
            &reference,
        );
        // The engine extends K, L with M, which it holds the parent of.
        let mut longer = chain.clone();
        longer.push(content(14, 0));
        let longer_blocks = blocks_of(&longer);
        let applied = store(
            w,
            &mut map_w,
            Some(blocks[11].seq_hash),
            &longer_blocks[12..],
            &mut lane,
            &mut reference,
        );
        lane.flush();
        let _ = writeln!(
            out,
            "| {case} | store M after L | CRTC applied the store: {applied} | | | |"
        );
        check_focused(
            &mut out,
            &mut failures,
            w,
            &label("after storing M"),
            "A..M",
            &longer,
            &lane,
            &production,
            &reference,
        );
    }

    println!("{out}");
    assert!(failures.is_empty(), "{failures:?}");
}
