//! SMG's event-driven `PositionalIndexer` (`crates/kv_index` in smg-project/smg) behind the
//! `SyncIndexer` interface, so the Mooncake replay measures it through the same lanes, queues,
//! observation records and accounting as the Dynamo backends.
//!
//! What the adapter adds on top of SMG's own cost, and therefore charges to it: a score
//! translation from SMG's interned `u32` worker ids back to `WorkerWithDpRank` (one lock-free
//! `ArcSwap` load per lookup and one map insert per matching worker). Stores, removals and
//! lookups hand SMG the event's own buffers through its iterator and `ContentSeq` entry points,
//! so no per-event or per-lookup `Vec` is built here.
//!
//! Semantics follow SMG's gateway (`KvEventMonitor`): one per-worker block map per lane, stores
//! placed by the parent's tracked position (SMG ignores `start_position`), removals and clears by
//! engine block hash, worker removal through SMG's reverse map.

use std::sync::Arc;

use arc_swap::ArcSwap;
use flume::Receiver;
use kv_index::{
    ApplyError, ContentHash, ContentSeq, PositionalIndexer, SequenceHash, StoredBlock,
    WorkerBlockMap,
};
use rustc_hash::FxHashMap;

use super::metrics::{EventKind, KvIndexerMetrics, PreBoundEventCounters};
#[cfg(feature = "bench")]
use super::observation::WorkerObservationState;
use super::traits::SyncIndexer;
use super::types::{WorkerLookupStats, WorkerTask};
use super::KvRouterError;
use crate::protocols::{
    KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, WorkerId,
    WorkerWithDpRank,
};

/// One lane's view of a worker: SMG's interned id and SMG's per-worker block map.
struct LaneWorker {
    smg_id: u32,
    blocks: WorkerBlockMap,
}

/// A request's block hashes as SMG reads them, without copying them into SMG's newtype.
struct RequestHashes<'a>(&'a [LocalBlockHash]);

impl ContentSeq for RequestHashes<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn at(&self, position: usize) -> ContentHash {
        ContentHash(self.0[position].0)
    }
}

pub struct SmgPositional {
    inner: PositionalIndexer,
    /// SMG worker id -> Dynamo worker, for translating lookup scores back. Replaced wholesale
    /// when a worker is interned (rare); lookups take one lock-free load.
    workers: ArcSwap<Vec<Option<WorkerWithDpRank>>>,
}

impl SmgPositional {
    pub fn new(jump_size: usize) -> Self {
        Self {
            inner: PositionalIndexer::new(jump_size),
            workers: ArcSwap::from_pointee(Vec::new()),
        }
    }

    fn intern(&self, worker: WorkerWithDpRank) -> u32 {
        let key = format!("{}:{}", worker.worker_id, worker.dp_rank);
        let id = self
            .inner
            .intern_worker(&key)
            .expect("SMG worker id space (u32) exhausted");
        self.workers.rcu(|table| {
            let mut table = Vec::clone(table);
            if table.len() <= id as usize {
                table.resize(id as usize + 1, None);
            }
            table[id as usize] = Some(worker);
            table
        });
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
                    blocks: WorkerBlockMap::default(),
                });
                let blocks = store.blocks.iter().map(|block| StoredBlock {
                    seq_hash: SequenceHash(block.block_hash.0),
                    content_hash: ContentHash(block.tokens_hash.0),
                });
                let parent = store.parent_hash.map(|hash| SequenceHash(hash.0));
                self.inner
                    .apply_stored_iter(entry.smg_id, blocks, parent, &mut entry.blocks)
                    .map_err(|error| match error {
                        ApplyError::ParentBlockNotFound | ApplyError::WorkerNotTracked => {
                            KvCacheEventError::ParentBlockNotFound
                        }
                    })
            }
            KvCacheEventData::Removed(remove) => {
                let Some(entry) = lane.get_mut(&worker) else {
                    return Err(KvCacheEventError::BlockNotFound);
                };
                self.inner.apply_removed_iter(
                    entry.smg_id,
                    remove.block_hashes.iter().map(|hash| SequenceHash(hash.0)),
                    &mut entry.blocks,
                );
                Ok(())
            }
            KvCacheEventData::Cleared => {
                if let Some(entry) = lane.get_mut(&worker) {
                    self.inner.apply_cleared(entry.smg_id, &mut entry.blocks);
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

impl SyncIndexer for SmgPositional {
    fn worker(
        &self,
        event_receiver: Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
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
        let smg = self.inner.find_matches_in(&RequestHashes(sequence), early_exit);
        let mut scores = OverlapScores::new();
        let table = self.workers.load();
        for (smg_id, score) in smg.scores {
            if let Some(Some(worker)) = table.get(smg_id as usize) {
                scores.scores.insert(*worker, score);
            }
        }
        scores
    }

    fn supports_event_dump(&self) -> bool {
        false
    }

    fn supports_routing_decision_pruning(&self) -> bool {
        false
    }
}
