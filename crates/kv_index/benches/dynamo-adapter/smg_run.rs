//! SMG's run-compressed `RunIndex` (`crates/kv_index/src/run_index.rs` in smg-project/smg) behind
//! the `SyncIndexer` interface, so the Mooncake replay measures it through the same lanes, queues,
//! observation records and accounting as the Dynamo backends.
//!
//! The adapter charges SMG the same translation costs as `smg_positional.rs`: one
//! `Vec<StoredBlock>` per stored event, one `Vec<SequenceHash>` per removal, one
//! `Vec<ContentHash>` per lookup, and the score translation from SMG's interned `u32` worker ids
//! back to `WorkerWithDpRank`.
//!
//! Semantics follow SMG's gateway (`KvEventMonitor`): one per-worker block map per lane, stores
//! placed by the parent's tracked position (SMG ignores `start_position`), removals and clears by
//! engine block hash, worker removal through the lane's map.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use arc_swap::ArcSwap;

use flume::Receiver;
use kv_index::{ApplyError, ContentHash, RunBlockMap, RunIndex, SequenceHash, StoredBlock};
use rustc_hash::FxHashMap;

use super::KvRouterError;
use super::metrics::{EventKind, KvIndexerMetrics, PreBoundEventCounters};
#[cfg(feature = "bench")]
use super::observation::WorkerObservationState;
use super::traits::SyncIndexer;
use super::types::{WorkerLookupStats, WorkerTask};
use crate::protocols::{
    KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, WorkerId,
    WorkerWithDpRank,
};

/// Bytes per slot of the lane maps (`FxHashMap<SequenceHash, BlockRef>`: 16-byte entry plus a
/// control byte), for the memory report.
const MAP_SLOT_BYTES: usize = 17;

/// One lane's view of a worker: SMG's interned id and SMG's per-worker block map.
struct LaneWorker {
    smg_id: u32,
    blocks: RunBlockMap,
    /// Capacity last charged to `SmgRun::map_slots`.
    charged_slots: usize,
}

pub struct SmgRun {
    inner: RunIndex,
    /// SMG worker id -> Dynamo worker, for translating lookup scores back. Replaced wholesale
    /// when a worker is interned (rare) so lookups read it without a lock: a read lock taken by
    /// 128 query lanes is a shared cache line every lookup writes.
    workers: ArcSwap<Vec<Option<WorkerWithDpRank>>>,
    intern_lock: Mutex<()>,
    /// Allocated slots across every lane map, kept current by the lanes.
    map_slots: AtomicUsize,
    /// Lookups served and runs walked by them, for the fragmentation line of the report.
    lookups: AtomicUsize,
    runs_walked: AtomicUsize,
}

impl SmgRun {
    pub fn new(max_workers: usize) -> Self {
        Self {
            inner: RunIndex::with_max_workers(max_workers),
            workers: ArcSwap::from_pointee(Vec::new()),
            intern_lock: Mutex::new(()),
            map_slots: AtomicUsize::new(0),
            lookups: AtomicUsize::new(0),
            runs_walked: AtomicUsize::new(0),
        }
    }

    /// Shape and memory counters of the index itself (lane maps excluded).
    pub fn stats(&self) -> kv_index::RunIndexStats {
        self.inner.stats()
    }

    fn charge_map(&self, entry: &mut LaneWorker) {
        let slots = entry.blocks.capacity();
        if slots > entry.charged_slots {
            self.map_slots
                .fetch_add(slots - entry.charged_slots, Ordering::Relaxed);
        } else if slots < entry.charged_slots {
            self.map_slots
                .fetch_sub(entry.charged_slots - slots, Ordering::Relaxed);
        }
        entry.charged_slots = slots;
    }

    fn release_map(&self, entry: &LaneWorker) {
        self.map_slots
            .fetch_sub(entry.charged_slots, Ordering::Relaxed);
    }

    fn intern(&self, worker: WorkerWithDpRank) -> u32 {
        let key = format!("{}:{}", worker.worker_id, worker.dp_rank);
        let id = self
            .inner
            .intern_worker(&key)
            .expect("SMG run index worker slots exhausted; raise --max-workers");
        let _guard = self.intern_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut table: Vec<Option<WorkerWithDpRank>> = self.workers.load().as_ref().clone();
        if table.len() <= id as usize {
            table.resize(id as usize + 1, None);
        }
        table[id as usize] = Some(worker);
        self.workers.store(std::sync::Arc::new(table));
        id
    }

    fn apply_event(
        &self,
        lane: &mut FxHashMap<WorkerWithDpRank, LaneWorker>,
        event: RouterEvent,
    ) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        match event.event.data {
            KvCacheEventData::Stored(store) => {
                let entry = lane.entry(worker).or_insert_with(|| LaneWorker {
                    smg_id: self.intern(worker),
                    blocks: RunBlockMap::default(),
                    charged_slots: 0,
                });
                let blocks: Vec<StoredBlock> = store
                    .blocks
                    .iter()
                    .map(|block| StoredBlock {
                        seq_hash: SequenceHash(block.block_hash.0),
                        content_hash: ContentHash(block.tokens_hash.0),
                    })
                    .collect();
                let parent = store.parent_hash.map(|hash| SequenceHash(hash.0));
                let outcome = self
                    .inner
                    .apply_stored(entry.smg_id, &blocks, parent, &mut entry.blocks)
                    .map_err(|error| match error {
                        ApplyError::ParentBlockNotFound | ApplyError::WorkerNotTracked => {
                            KvCacheEventError::ParentBlockNotFound
                        }
                    });
                self.charge_map(entry);
                outcome
            }
            KvCacheEventData::Removed(remove) => {
                let Some(entry) = lane.get_mut(&worker) else {
                    return Err(KvCacheEventError::BlockNotFound);
                };
                let hashes: Vec<SequenceHash> = remove
                    .block_hashes
                    .iter()
                    .map(|hash| SequenceHash(hash.0))
                    .collect();
                self.inner
                    .apply_removed(entry.smg_id, &hashes, &mut entry.blocks);
                self.charge_map(entry);
                Ok(())
            }
            KvCacheEventData::Cleared => {
                if let Some(entry) = lane.get_mut(&worker) {
                    self.inner.apply_cleared(entry.smg_id, &mut entry.blocks);
                    self.charge_map(entry);
                }
                Ok(())
            }
        }
    }

    fn remove_worker(&self, lane: &mut FxHashMap<WorkerWithDpRank, LaneWorker>, worker_id: WorkerId) {
        let gone: Vec<WorkerWithDpRank> = lane
            .keys()
            .filter(|worker| worker.worker_id == worker_id)
            .copied()
            .collect();
        for worker in gone {
            if let Some(entry) = lane.remove(&worker) {
                self.release_map(&entry);
                self.inner.remove_worker(entry.smg_id, entry.blocks);
            }
        }
    }

    fn remove_worker_dp_rank(
        &self,
        lane: &mut FxHashMap<WorkerWithDpRank, LaneWorker>,
        worker: WorkerWithDpRank,
    ) {
        if let Some(entry) = lane.remove(&worker) {
            self.release_map(&entry);
            self.inner.remove_worker(entry.smg_id, entry.blocks);
        }
    }

    fn record(
        counters: Option<&PreBoundEventCounters>,
        kind: EventKind,
        result: &Result<(), KvCacheEventError>,
    ) {
        if result.is_err() {
            tracing::warn!("Failed to apply event: {:?}", result.as_ref().err());
        }
        if let Some(counters) = counters {
            counters.inc(kind, *result);
        }
    }
}

impl SyncIndexer for SmgRun {
    fn worker(
        &self,
        event_receiver: Receiver<WorkerTask>,
        metrics: Option<std::sync::Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let mut lane: FxHashMap<WorkerWithDpRank, LaneWorker> = FxHashMap::default();
        let counters = metrics.as_ref().map(|m| m.prebind());
        #[cfg(feature = "bench")]
        let mut observation = WorkerObservationState::default();
        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(event) => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut lane, event);
                    Self::record(counters.as_ref(), kind, &result);
                }
                WorkerTask::EventWithAck { event, resp } => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut lane, event);
                    Self::record(counters.as_ref(), kind, &result);
                    let _ = resp.send(result.is_ok());
                }
                WorkerTask::ApproximateLru(task) => task.complete(Err(KvRouterError::Unsupported(
                    "approximate LRU requires ConcurrentRadixTreeCompressed".to_string(),
                ))),
                #[cfg(feature = "bench")]
                WorkerTask::InstallObservation { writer, resp } => {
                    observation.install(writer, resp);
                }
                #[cfg(feature = "bench")]
                WorkerTask::ObservedEvent {
                    event,
                    correlation_id,
                } => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut lane, event);
                    observation.record(correlation_id, result.is_ok());
                    Self::record(counters.as_ref(), kind, &result);
                }
                #[cfg(feature = "bench")]
                WorkerTask::SealObservation(resp) => observation.seal(resp),
                #[cfg(feature = "bench")]
                WorkerTask::HarvestObservation(resp) => observation.harvest(resp),
                WorkerTask::Anchor { .. } => {
                    tracing::warn!("anchored lookups are not supported by the SMG backend");
                }
                WorkerTask::RemoveWorker {
                    worker_id, resp, ..
                } => {
                    self.remove_worker(&mut lane, worker_id);
                    let _ = resp.send(());
                }
                WorkerTask::RemoveWorkerDpRank {
                    worker_id, dp_rank, ..
                } => {
                    self.remove_worker_dp_rank(&mut lane, WorkerWithDpRank::new(worker_id, dp_rank));
                }
                WorkerTask::CleanupStaleChildren => {}
                WorkerTask::DumpEvents(sender) => {
                    let _ = sender.send(Err(anyhow::anyhow!(
                        "event dumps are not supported by the SMG backend"
                    )));
                }
                WorkerTask::Stats(sender) => {
                    let stats = WorkerLookupStats::from_worker_block_counts(
                        lane.iter()
                            .map(|(worker, entry)| (*worker, entry.blocks.len())),
                    );
                    let _ = sender.send(stats);
                }
                WorkerTask::ContainsWorkerBlock {
                    worker,
                    block_hash,
                    resp,
                } => {
                    let resident = lane
                        .get(&worker)
                        .is_some_and(|entry| entry.blocks.contains_key(&SequenceHash(block_hash.0)));
                    let _ = resp.send(resident);
                }
                WorkerTask::Flush(sender) => {
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => break,
            }
        }
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        let mut scores = OverlapScores::new();
        let table = self.workers.load();
        // Every interned worker may score; size the map once instead of growing it per insert.
        scores.scores.reserve(table.len());
        let walked = self
            .inner
            .score_into(sequence, |hash| hash.0, early_exit, |smg_id, score| {
                if let Some(Some(worker)) = table.get(smg_id as usize) {
                    scores.scores.insert(*worker, score);
                }
            });
        self.lookups.fetch_add(1, Ordering::Relaxed);
        self.runs_walked.fetch_add(walked, Ordering::Relaxed);
        scores
    }

    fn supports_event_dump(&self) -> bool {
        false
    }

    fn supports_routing_decision_pruning(&self) -> bool {
        false
    }

    fn timing_report(&self) -> String {
        let stats = self.inner.stats();
        let memberships = self.inner.current_size();
        let distinct = self.inner.entry_count();
        let map_slots = self.map_slots.load(Ordering::Relaxed);
        let map_bytes = map_slots * MAP_SLOT_BYTES;
        let index_bytes = stats.arena_bytes + stats.header_bytes;
        let per_membership = |bytes: usize| {
            if memberships == 0 {
                0.0
            } else {
                bytes as f64 / memberships as f64
            }
        };
        format!(
            "SmgRun memory report:\n  \
             memberships (worker, block) = {memberships}\n  \
             distinct blocks = {distinct}\n  \
             runs allocated = {} live = {} blocks in live runs = {}\n  \
             arena bytes = {} header bytes = {} (index {:.1} B per membership)\n  \
             lane map slots = {map_slots} bytes = {map_bytes} ({:.1} B per membership)\n  \
             total {:.1} B per membership\n  \
             lookups = {} runs walked per lookup = {:.2}",
            stats.runs_allocated,
            stats.runs_live,
            stats.blocks_live,
            stats.arena_bytes,
            stats.header_bytes,
            per_membership(index_bytes),
            per_membership(map_bytes),
            per_membership(index_bytes + map_bytes),
            self.lookups.load(Ordering::Relaxed),
            self.runs_walked.load(Ordering::Relaxed) as f64
                / self.lookups.load(Ordering::Relaxed).max(1) as f64,
        )
    }
}
