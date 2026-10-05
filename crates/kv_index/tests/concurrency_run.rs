//! Concurrency harness for `RunIndex`: many event lanes writing shared chains at once while
//! readers look them up, then the end state against the `ReferenceIndexer`.
//!
//! Lanes own one worker each and replay random stores, extensions, tail and middle evictions,
//! clears and worker replacements over a shared pool of conversations, so runs are joined, split,
//! truncated and unlinked by different threads at the same time. Every lane logs what it applied;
//! after the lanes stop, the logs are replayed in application order into the reference, which
//! must hold exactly the same blocks and score every pool chain the same way. Readers run
//! throughout and check the invariants that hold under any interleaving: no score exceeds the
//! request, early-exit scores are 1, and nothing panics.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use kv_index::{
    request_prefix_hashes, ContentHash, ReferenceIndexer, RunBlockMap, RunIndex, SequenceHash,
    StoredBlock,
};

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
}

fn content(stream: u64, position: usize) -> ContentHash {
    kv_index::compute_content_hash(&[
        (stream & 0xffff_ffff) as u32,
        (stream >> 32) as u32,
        position as u32,
    ])
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

/// A shared pool of conversations: a few prompts, each continued by several turns, with sibling
/// turns branching off at various depths.
fn pool(rng: &mut Rng) -> Vec<Vec<ContentHash>> {
    let mut chains: Vec<Vec<ContentHash>> = Vec::new();
    let mut stream = 1u64;
    for _ in 0..6 {
        stream += 1;
        let prompt: Vec<ContentHash> = (0..rng.range(8, 40)).map(|p| content(stream, p)).collect();
        for _ in 0..12 {
            let base = if chains.is_empty() || rng.below(3) == 0 {
                prompt.clone()
            } else {
                let parent = &chains[chains.len() - 1 - rng.below(chains.len().min(12))];
                parent[..rng.range(1, parent.len())].to_vec()
            };
            stream += 1;
            let mut chain = base;
            chain.extend((0..rng.range(1, 24)).map(|p| content(stream, p)));
            chains.push(chain);
        }
    }
    chains
}

enum Event {
    Stored {
        blocks: Vec<StoredBlock>,
        parent: Option<SequenceHash>,
    },
    Removed(Vec<SequenceHash>),
    Cleared,
    WorkerRemoved,
}

struct Logged {
    seq: u64,
    worker: u32,
    event: Event,
}

struct Lane<'a> {
    index: &'a RunIndex,
    pool: &'a [Vec<ContentHash>],
    clock: &'a AtomicU64,
    worker: u32,
    name_counter: u64,
    lane: usize,
    map: RunBlockMap,
    rng: Rng,
    log: Vec<Logged>,
}

impl Lane<'_> {
    fn record(&mut self, event: Event) {
        let seq = self.clock.fetch_add(1, Ordering::Relaxed);
        self.log.push(Logged {
            seq,
            worker: self.worker,
            event,
        });
    }

    fn store(&mut self, contents: &[ContentHash]) {
        let blocks = blocks_of(contents);
        let mut known = 0;
        while known < blocks.len() && self.map.contains_key(&blocks[known].seq_hash) {
            known += 1;
        }
        let start = if known == blocks.len() {
            blocks.len() - 1
        } else {
            known
        };
        let parent = (start > 0).then(|| blocks[start - 1].seq_hash);
        let outcome = self
            .index
            .apply_stored(self.worker, &blocks[start..], parent, &mut self.map);
        assert!(
            outcome.is_ok(),
            "lane {}: store after a held parent failed: {outcome:?}",
            self.lane
        );
        self.record(Event::Stored {
            blocks: blocks[start..].to_vec(),
            parent,
        });
    }

    /// Remove a contiguous range of the blocks this lane holds on a pool chain.
    fn remove_some(&mut self, contents: &[ContentHash]) {
        let blocks = blocks_of(contents);
        let held: Vec<usize> = (0..blocks.len())
            .filter(|&i| self.map.contains_key(&blocks[i].seq_hash))
            .collect();
        if held.is_empty() {
            return;
        }
        let from = self.rng.below(held.len());
        let to = match self.rng.below(4) {
            0 => held.len(),
            _ => (from + self.rng.range(1, 3)).min(held.len()),
        };
        let hashes: Vec<SequenceHash> = held[from..to].iter().map(|&i| blocks[i].seq_hash).collect();
        self.index
            .apply_removed(self.worker, &hashes, &mut self.map);
        self.record(Event::Removed(hashes));
    }

    fn step(&mut self) {
        let chain = self.pool[self.rng.below(self.pool.len())].clone();
        match self.rng.below(1000) {
            0..=549 => {
                let len = if self.rng.below(2) == 0 {
                    chain.len()
                } else {
                    self.rng.range(1, chain.len())
                };
                self.store(&chain[..len]);
            }
            550..=899 => self.remove_some(&chain),
            900..=994 => {
                // Walk a chain in two turns: a prefix now, the rest right after (decode extension).
                let cut = self.rng.range(1, chain.len());
                self.store(&chain[..cut]);
                self.store(&chain);
            }
            995..=998 => {
                self.index.apply_cleared(self.worker, &mut self.map);
                self.record(Event::Cleared);
            }
            _ => {
                let map = std::mem::take(&mut self.map);
                self.index.remove_worker(self.worker, map);
                self.record(Event::WorkerRemoved);
                self.name_counter += 1;
                let name = format!("lane-{}-{}", self.lane, self.name_counter);
                self.worker = self.index.intern_worker(&name).expect("worker slot");
            }
        }
    }
}

fn scores(index: &RunIndex, query: &[ContentHash], early_exit: bool) -> BTreeMap<u32, u32> {
    index
        .find_matches(query, early_exit)
        .scores
        .into_iter()
        .collect()
}

#[test]
fn concurrent_lanes_and_readers_end_in_the_reference_state() {
    let lanes = 16usize;
    let steps = std::env::var("KV_INDEX_CONCURRENCY_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000usize);
    let mut rng = Rng::new(20261005);
    let pool = pool(&mut rng);
    let index = RunIndex::with_max_workers(64);
    let clock = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let logs: Mutex<Vec<Logged>> = Mutex::new(Vec::new());
    let reader_lookups = AtomicU64::new(0);

    thread::scope(|scope| {
        for lane in 0..lanes {
            let (index, pool, clock, logs) = (&index, &pool, &clock, &logs);
            let seed = rng.next();
            scope.spawn(move || {
                let worker = index
                    .intern_worker(&format!("lane-{lane}-0"))
                    .expect("worker slot");
                let mut state = Lane {
                    index,
                    pool,
                    clock,
                    worker,
                    name_counter: 0,
                    lane,
                    map: RunBlockMap::default(),
                    rng: Rng::new(seed),
                    log: Vec::new(),
                };
                for _ in 0..steps {
                    state.step();
                }
                logs.lock().unwrap().extend(state.log);
            });
        }
        for reader in 0..4usize {
            let (index, pool, stop, reader_lookups) = (&index, &pool, &stop, &reader_lookups);
            let seed = rng.next() ^ reader as u64;
            scope.spawn(move || {
                let mut rng = Rng::new(seed);
                while !stop.load(Ordering::Relaxed) {
                    let chain = &pool[rng.below(pool.len())];
                    let mut query = chain.clone();
                    if rng.below(3) == 0 {
                        let at = rng.below(query.len());
                        query[at] = content(u64::MAX - reader as u64, at);
                    }
                    let full = scores(index, &query, false);
                    for (&worker, &score) in &full {
                        assert!(
                            score as usize <= query.len(),
                            "worker {worker} scored {score} on a {}-block request",
                            query.len()
                        );
                        assert!(score > 0, "worker {worker} reported with score 0");
                    }
                    let early = scores(index, &query, true);
                    assert!(early.values().all(|&score| score == 1));
                    reader_lookups.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        // Lanes are finite; readers stop once the lanes have logged.
        loop {
            thread::sleep(std::time::Duration::from_millis(20));
            if logs.lock().unwrap().len() >= lanes * steps {
                break;
            }
        }
        stop.store(true, Ordering::Relaxed);
    });

    let mut logs = logs.into_inner().unwrap();
    logs.sort_by_key(|entry| entry.seq);
    let mut reference = ReferenceIndexer::new();
    for entry in &logs {
        match &entry.event {
            Event::Stored { blocks, parent } => {
                reference
                    .apply_stored(entry.worker, blocks, *parent)
                    .expect("the lane stored after a held parent");
            }
            Event::Removed(hashes) => reference.apply_removed(entry.worker, hashes),
            Event::Cleared => reference.apply_cleared(entry.worker),
            Event::WorkerRemoved => reference.remove_worker(entry.worker),
        }
    }
    assert!(
        reader_lookups.load(Ordering::Relaxed) > 1000,
        "readers barely ran: {} lookups",
        reader_lookups.load(Ordering::Relaxed)
    );

    let produced = index.debug_blocks();
    let expected = reference.blocks();
    let missing: Vec<_> = expected.difference(&produced).take(5).collect();
    let phantom: Vec<_> = produced.difference(&expected).take(5).collect();
    assert!(
        missing.is_empty() && phantom.is_empty(),
        "end state differs after {} events: {} reference blocks, {} index blocks; missing e.g. \
         {missing:?}; phantom e.g. {phantom:?}",
        logs.len(),
        expected.len(),
        produced.len()
    );
    let mut checked = 0;
    for chain in &pool {
        for query in [chain.clone(), chain[..chain.len().div_ceil(2)].to_vec()] {
            assert_eq!(
                scores(&index, &query, false),
                reference.find_matches(&query),
                "lookup differs for a {}-block query",
                query.len()
            );
            checked += 1;
        }
    }
    assert!(checked > 100);
    assert_eq!(
        index.entry_count(),
        expected
            .iter()
            .map(|(_, position, content, prefix)| (*position, *content, *prefix))
            .collect::<BTreeSet<_>>()
            .len(),
        "distinct block counter"
    );
}
