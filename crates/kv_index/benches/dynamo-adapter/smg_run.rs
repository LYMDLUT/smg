//! SMG's run-compressed `RunIndex` (`crates/kv_index/src/run_index.rs` in smg-project/smg) behind
//! the `SyncIndexer` interface, so the Mooncake replay measures it through the same lanes, queues,
//! observation records and accounting as the Dynamo backends.
//!
//! The adapter charges SMG the same translation costs as `smg_positional.rs`: one
//! `Vec<StoredBlock>` per stored event, one `Vec<SequenceHash>` per removal, one
//! `Vec<ContentHash>` per lookup, and the score translation from SMG's interned `u32` worker ids
//! back to `WorkerWithDpRank`.
//!
//! Semantics follow SMG's gateway (`KvEventMonitor`): one per-worker block map, stores placed by
//! the parent's tracked position (SMG ignores `start_position`), removals and clears by engine
//! block hash, worker removal through the worker's map.
//!
//! # Lanes
//!
//! The harness hands every lane thread one channel with a sticky worker-to-lane assignment. Here
//! the lane threads are the lanes of SMG's `LanePool`: each drains its channel into per-worker
//! queues and serves ready workers from any lane, so a lane whose workers are quiet works off the
//! backlog of a busy one (whole workers at a time; events of one worker never leave their order).
//! A worker's queue holds at most `DEPTH_CAP` events; past that its lane keeps the next events in
//! a backlog and reads nothing further from its channel until they are in, so the harness's own
//! queue is the overflow and its queue-depth row still measures it. Flush, seal, stats and worker
//! removal are barriers queued behind every worker the channel has fed, answered by whichever
//! lane applies the last one; observation records go to the writer of the channel the event came
//! in on, so each lane's completion buffer keeps the capacity the harness planned for it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use arc_swap::ArcSwap;
use flume::{Receiver, RecvTimeoutError, TryRecvError};
use kv_index::{
    ApplyError, Claimed, ContentHash, Control, LaneHooks, LanePool, LanePoolConfig, QueueFull,
    RunBlockMap, RunIndex, SequenceHash, StoredBlock,
};
use rustc_hash::FxHashMap;
use tokio::sync::oneshot;

use super::KvRouterError;
use super::metrics::{EventKind, KvIndexerMetrics, PreBoundEventCounters};
#[cfg(feature = "bench")]
use super::observation::{ObservationSeal, WorkerObservationState};
use super::traits::SyncIndexer;
use super::types::{WorkerLookupStats, WorkerTask};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, LocalBlockHash,
    OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank,
};

/// Queued events one worker may hold before its lane holds the next ones back.
const DEPTH_CAP: usize = 2048;
/// Events a lane applies from one worker before letting another ready worker in.
const BATCH: usize = 32;
/// How long an idle lane waits on its channel before looking for work to steal again. Shorter
/// waits cost more than they gain: every wake-up is a syscall on a core shared with the query
/// lanes, and preempting a lane that holds a run lock stalls every lane waiting for it.
const WAIT: Duration = Duration::from_millis(1);
/// How long a lane whose refused events are waiting on a worker another lane is serving sleeps.
const BACKLOG_WAIT: Duration = Duration::from_micros(100);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One worker's state, owned by whichever lane is applying its events: SMG's interned id and
/// SMG's per-worker block map.
struct LaneWorker {
    worker: WorkerWithDpRank,
    smg_id: u32,
    blocks: RunBlockMap,
    /// Capacity last charged to `SmgRun::map_slots`.
    charged_slots: usize,
    /// Bytes last charged to `SmgRun::map_bytes`.
    charged_bytes: usize,
}

/// What each worker does when it reaches a barrier.
#[derive(Clone, Copy)]
enum Step {
    Pass,
    /// Report the worker's block count.
    Count,
    /// Drop the worker's state and free its slot.
    Remove,
}

/// What the last worker through a barrier does.
enum Finish {
    Flush(oneshot::Sender<()>),
    Removed(Option<oneshot::Sender<()>>),
    Stats(oneshot::Sender<WorkerLookupStats>),
    #[cfg(feature = "bench")]
    Seal {
        home: usize,
        resp: oneshot::Sender<Option<ObservationSeal>>,
    },
}

struct Barrier {
    remaining: AtomicUsize,
    rows: Mutex<Vec<(WorkerWithDpRank, usize)>>,
    finish: Mutex<Option<Finish>>,
}

/// What a worker's queue carries.
enum LaneEvent {
    Apply(RouterEvent),
    ApplyAck(RouterEvent, oneshot::Sender<bool>),
    #[cfg(feature = "bench")]
    Observed {
        event: RouterEvent,
        correlation_id: u32,
        /// Lane whose channel delivered the event and whose completion writer records it.
        home: usize,
    },
    Contains {
        block_hash: ExternalSequenceBlockHash,
        resp: oneshot::Sender<bool>,
    },
    Barrier {
        barrier: Arc<Barrier>,
        step: Step,
    },
}

/// Dynamo worker -> pool slot, shared by the lanes (consulted when a lane first sees a worker).
#[derive(Default)]
struct SlotRegistry {
    slots: FxHashMap<WorkerWithDpRank, u32>,
    free: Vec<u32>,
    next: u32,
}

pub struct SmgRun {
    inner: RunIndex,
    pool: LanePool<LaneWorker, LaneEvent>,
    registry: Mutex<SlotRegistry>,
    /// Lane indices handed out to `worker` calls.
    next_lane: AtomicUsize,
    /// Completion writers by the lane whose channel installed them.
    #[cfg(feature = "bench")]
    observations: Box<[Mutex<WorkerObservationState>]>,
    /// SMG worker id -> Dynamo worker, for translating lookup scores back. Replaced wholesale
    /// when a worker is interned (rare) so lookups read it without a lock: a read lock taken by
    /// 128 query lanes is a shared cache line every lookup writes.
    workers: ArcSwap<Vec<Option<WorkerWithDpRank>>>,
    intern_lock: Mutex<()>,
    /// Allocated slots across every lane map, kept current by the lanes.
    map_slots: AtomicUsize,
    /// Bytes the lane maps hold from the process allocator (slots and tags).
    map_bytes: AtomicUsize,
    /// Lookups served and runs walked by them, for the fragmentation line of the report.
    lookups: AtomicUsize,
    runs_walked: AtomicUsize,
}

impl SmgRun {
    /// `lanes` is the harness's event-worker count: every lane thread the harness starts takes one
    /// lane of the pool.
    pub fn new(max_workers: usize, lanes: usize) -> Self {
        let lanes = lanes.max(1);
        Self {
            inner: RunIndex::with_max_workers(max_workers),
            pool: LanePool::new(LanePoolConfig {
                lanes,
                max_workers,
                depth_cap: DEPTH_CAP,
                batch: BATCH,
            }),
            registry: Mutex::new(SlotRegistry::default()),
            next_lane: AtomicUsize::new(0),
            #[cfg(feature = "bench")]
            observations: (0..lanes)
                .map(|_| Mutex::new(WorkerObservationState::default()))
                .collect(),
            workers: ArcSwap::from_pointee(Vec::new()),
            intern_lock: Mutex::new(()),
            map_slots: AtomicUsize::new(0),
            map_bytes: AtomicUsize::new(0),
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
        let bytes = entry.blocks.memory_bytes();
        if bytes > entry.charged_bytes {
            self.map_bytes
                .fetch_add(bytes - entry.charged_bytes, Ordering::Relaxed);
        } else if bytes < entry.charged_bytes {
            self.map_bytes
                .fetch_sub(entry.charged_bytes - bytes, Ordering::Relaxed);
        }
        entry.charged_bytes = bytes;
    }

    fn release_map(&self, entry: &LaneWorker) {
        self.map_slots
            .fetch_sub(entry.charged_slots, Ordering::Relaxed);
        self.map_bytes
            .fetch_sub(entry.charged_bytes, Ordering::Relaxed);
    }

    fn intern(&self, worker: WorkerWithDpRank) -> u32 {
        let key = format!("{}:{}", worker.worker_id, worker.dp_rank);
        let id = self
            .inner
            .intern_worker(&key)
            .expect("SMG run index worker slots exhausted; raise --max-workers");
        let _guard = lock(&self.intern_lock);
        let mut table: Vec<Option<WorkerWithDpRank>> = self.workers.load().as_ref().clone();
        if table.len() <= id as usize {
            table.resize(id as usize + 1, None);
        }
        table[id as usize] = Some(worker);
        self.workers.store(Arc::new(table));
        id
    }

    fn apply_event(
        &self,
        state: &mut Option<LaneWorker>,
        event: RouterEvent,
    ) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        match event.event.data {
            KvCacheEventData::Stored(store) => {
                let entry = state.get_or_insert_with(|| LaneWorker {
                    worker,
                    smg_id: self.intern(worker),
                    blocks: RunBlockMap::default(),
                    charged_slots: 0,
                    charged_bytes: 0,
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
                let Some(entry) = state.as_mut() else {
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
                if let Some(entry) = state.as_mut() {
                    self.inner.apply_cleared(entry.smg_id, &mut entry.blocks);
                    self.charge_map(entry);
                }
                Ok(())
            }
        }
    }

    /// The answer of an event for a worker no lane has a state for: a removal has nothing to
    /// remove, a clear nothing to clear.
    fn unknown_worker_result(event: &RouterEvent) -> Result<(), KvCacheEventError> {
        match event.event.data {
            KvCacheEventData::Removed(_) => Err(KvCacheEventError::BlockNotFound),
            KvCacheEventData::Stored(_) | KvCacheEventData::Cleared => Ok(()),
        }
    }

    fn finish(&self, finish: Finish, rows: Vec<(WorkerWithDpRank, usize)>) {
        match finish {
            Finish::Flush(resp) | Finish::Removed(Some(resp)) => {
                let _ = resp.send(());
            }
            Finish::Removed(None) => {}
            Finish::Stats(resp) => {
                let _ = resp.send(WorkerLookupStats::from_worker_block_counts(
                    rows.into_iter(),
                ));
            }
            #[cfg(feature = "bench")]
            Finish::Seal { home, resp } => lock(&self.observations[home]).seal(resp),
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

/// One harness lane thread as a pool lane: its channel, the workers that channel has fed, and
/// the events the pool refused.
struct Lane<'a> {
    run: &'a SmgRun,
    lane: usize,
    receiver: Receiver<WorkerTask>,
    counters: Option<PreBoundEventCounters>,
    /// Workers this lane's channel has fed: Dynamo worker -> pool slot.
    fed: FxHashMap<WorkerWithDpRank, u32>,
    /// Events refused by the depth cap, in channel order; nothing later leaves the channel until
    /// these are in.
    backlog: VecDeque<(u32, LaneEvent)>,
}

impl Lane<'_> {
    /// The pool slot of `worker`, allocated on first sight when `create` (a store).
    fn slot_for(&mut self, worker: WorkerWithDpRank, create: bool) -> Option<u32> {
        if let Some(&slot) = self.fed.get(&worker) {
            return Some(slot);
        }
        let mut registry = lock(&self.run.registry);
        let slot = match registry.slots.get(&worker) {
            Some(&slot) => slot,
            None => {
                if !create {
                    return None;
                }
                let slot = registry.free.pop().unwrap_or_else(|| {
                    let slot = registry.next;
                    registry.next += 1;
                    slot
                });
                assert!(
                    (slot as usize) < self.run.pool.config().max_workers,
                    "SMG run backend worker slots exhausted; raise --max-workers"
                );
                registry.slots.insert(worker, slot);
                slot
            }
        };
        drop(registry);
        self.fed.insert(worker, slot);
        Some(slot)
    }

    fn slot_for_event(&mut self, event: &RouterEvent) -> Option<u32> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let create = matches!(event.event.data, KvCacheEventData::Stored(_));
        self.slot_for(worker, create)
    }

    /// Stop routing to these workers: their slots free once their removal barrier is applied.
    fn forget(&mut self, gone: &[(WorkerWithDpRank, u32)]) {
        let mut registry = lock(&self.run.registry);
        for (worker, slot) in gone {
            self.fed.remove(worker);
            if registry.slots.get(worker) == Some(slot) {
                registry.slots.remove(worker);
            }
        }
    }

    fn fed_slots(&self) -> Vec<u32> {
        self.fed.values().copied().collect()
    }

    /// Queue `event` for `slot`, behind anything the pool refused earlier.
    fn push(&mut self, slot: u32, event: LaneEvent) {
        if !self.backlog.is_empty() {
            self.backlog.push_back((slot, event));
            return;
        }
        if let Err(QueueFull(event)) = self.run.pool.enqueue(self.lane, slot, event) {
            self.backlog.push_back((slot, event));
        }
    }

    /// Move refused events into the pool in order; `false` while the head is still refused.
    fn flush_backlog(&mut self) -> bool {
        while let Some((slot, event)) = self.backlog.pop_front() {
            if let Err(QueueFull(event)) = self.run.pool.enqueue(self.lane, slot, event) {
                self.backlog.push_front((slot, event));
                return false;
            }
        }
        true
    }

    /// Queue a barrier behind every target; the last to apply it runs `finish`. No target:
    /// finish now.
    fn fan_out(&mut self, targets: Vec<u32>, step: Step, finish: Finish) {
        if targets.is_empty() {
            self.run.finish(finish, Vec::new());
            return;
        }
        let barrier = Arc::new(Barrier {
            remaining: AtomicUsize::new(targets.len()),
            rows: Mutex::new(Vec::new()),
            finish: Mutex::new(Some(finish)),
        });
        for slot in targets {
            self.push(
                slot,
                LaneEvent::Barrier {
                    barrier: Arc::clone(&barrier),
                    step,
                },
            );
        }
    }

    fn ingest(&mut self, task: WorkerTask) -> Control {
        match task {
            WorkerTask::Event(event) => {
                let kind = EventKind::of(&event.event.data);
                match self.slot_for_event(&event) {
                    Some(slot) => self.push(slot, LaneEvent::Apply(event)),
                    None => SmgRun::record(
                        self.counters.as_ref(),
                        kind,
                        &SmgRun::unknown_worker_result(&event),
                    ),
                }
            }
            WorkerTask::EventWithAck { event, resp } => {
                let kind = EventKind::of(&event.event.data);
                match self.slot_for_event(&event) {
                    Some(slot) => self.push(slot, LaneEvent::ApplyAck(event, resp)),
                    None => {
                        let result = SmgRun::unknown_worker_result(&event);
                        SmgRun::record(self.counters.as_ref(), kind, &result);
                        let _ = resp.send(result.is_ok());
                    }
                }
            }
            WorkerTask::ApproximateLru(task) => task.complete(Err(KvRouterError::Unsupported(
                "approximate LRU requires ConcurrentRadixTreeCompressed".to_string(),
            ))),
            #[cfg(feature = "bench")]
            WorkerTask::InstallObservation { writer, resp } => {
                lock(&self.run.observations[self.lane]).install(writer, resp);
            }
            #[cfg(feature = "bench")]
            WorkerTask::ObservedEvent {
                event,
                correlation_id,
            } => {
                let kind = EventKind::of(&event.event.data);
                match self.slot_for_event(&event) {
                    Some(slot) => self.push(
                        slot,
                        LaneEvent::Observed {
                            event,
                            correlation_id,
                            home: self.lane,
                        },
                    ),
                    None => {
                        let result = SmgRun::unknown_worker_result(&event);
                        lock(&self.run.observations[self.lane])
                            .record(correlation_id, result.is_ok());
                        SmgRun::record(self.counters.as_ref(), kind, &result);
                    }
                }
            }
            #[cfg(feature = "bench")]
            WorkerTask::SealObservation(resp) => {
                let targets = self.fed_slots();
                self.fan_out(
                    targets,
                    Step::Pass,
                    Finish::Seal {
                        home: self.lane,
                        resp,
                    },
                );
            }
            #[cfg(feature = "bench")]
            WorkerTask::HarvestObservation(resp) => {
                lock(&self.run.observations[self.lane]).harvest(resp);
            }
            WorkerTask::Anchor { .. } => {
                tracing::warn!("anchored lookups are not supported by the SMG backend");
            }
            WorkerTask::RemoveWorker {
                worker_id, resp, ..
            } => {
                let gone: Vec<(WorkerWithDpRank, u32)> = self
                    .fed
                    .iter()
                    .filter(|(worker, _)| worker.worker_id == worker_id)
                    .map(|(worker, slot)| (*worker, *slot))
                    .collect();
                self.forget(&gone);
                let targets = gone.iter().map(|(_, slot)| *slot).collect();
                self.fan_out(targets, Step::Remove, Finish::Removed(Some(resp)));
            }
            WorkerTask::RemoveWorkerDpRank {
                worker_id, dp_rank, ..
            } => {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                let gone: Vec<(WorkerWithDpRank, u32)> = self
                    .fed
                    .get(&worker)
                    .map(|slot| (worker, *slot))
                    .into_iter()
                    .collect();
                self.forget(&gone);
                let targets = gone.iter().map(|(_, slot)| *slot).collect();
                self.fan_out(targets, Step::Remove, Finish::Removed(None));
            }
            WorkerTask::CleanupStaleChildren => {}
            WorkerTask::DumpEvents(sender) => {
                let _ = sender.send(Err(anyhow::anyhow!(
                    "event dumps are not supported by the SMG backend"
                )));
            }
            WorkerTask::Stats(sender) => {
                let targets = self.fed_slots();
                self.fan_out(targets, Step::Count, Finish::Stats(sender));
            }
            WorkerTask::ContainsWorkerBlock {
                worker,
                block_hash,
                resp,
            } => match self.fed.get(&worker).copied() {
                Some(slot) => self.push(slot, LaneEvent::Contains { block_hash, resp }),
                None => {
                    let _ = resp.send(false);
                }
            },
            WorkerTask::Flush(sender) => {
                let targets = self.fed_slots();
                self.fan_out(targets, Step::Pass, Finish::Flush(sender));
            }
            WorkerTask::Terminate => return Control::Stop,
        }
        Control::Continue
    }
}

impl LaneHooks<LaneWorker, LaneEvent> for Lane<'_> {
    fn apply(&mut self, claimed: Claimed<'_, LaneWorker>, event: LaneEvent) {
        match event {
            LaneEvent::Apply(event) => {
                let kind = EventKind::of(&event.event.data);
                let result = self.run.apply_event(claimed.state, event);
                SmgRun::record(self.counters.as_ref(), kind, &result);
            }
            LaneEvent::ApplyAck(event, resp) => {
                let kind = EventKind::of(&event.event.data);
                let result = self.run.apply_event(claimed.state, event);
                SmgRun::record(self.counters.as_ref(), kind, &result);
                let _ = resp.send(result.is_ok());
            }
            #[cfg(feature = "bench")]
            LaneEvent::Observed {
                event,
                correlation_id,
                home,
            } => {
                let kind = EventKind::of(&event.event.data);
                let result = self.run.apply_event(claimed.state, event);
                lock(&self.run.observations[home]).record(correlation_id, result.is_ok());
                SmgRun::record(self.counters.as_ref(), kind, &result);
            }
            LaneEvent::Contains { block_hash, resp } => {
                let resident = claimed
                    .state
                    .as_ref()
                    .is_some_and(|entry| entry.blocks.contains_key(SequenceHash(block_hash.0)));
                let _ = resp.send(resident);
            }
            LaneEvent::Barrier { barrier, step } => {
                match step {
                    Step::Pass => {}
                    Step::Count => {
                        if let Some(entry) = claimed.state.as_ref() {
                            lock(&barrier.rows).push((entry.worker, entry.blocks.len()));
                        }
                    }
                    Step::Remove => {
                        if let Some(entry) = claimed.state.take() {
                            self.run.release_map(&entry);
                            self.run.inner.remove_worker(entry.smg_id, entry.blocks);
                        }
                        lock(&self.run.registry).free.push(claimed.worker);
                    }
                }
                if barrier.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                    let finish = lock(&barrier.finish).take();
                    let rows = std::mem::take(&mut *lock(&barrier.rows));
                    if let Some(finish) = finish {
                        self.run.finish(finish, rows);
                    }
                }
            }
        }
    }

    fn pump(&mut self) -> Control {
        if !self.flush_backlog() {
            return Control::Continue;
        }
        loop {
            match self.receiver.try_recv() {
                Ok(task) => {
                    if self.ingest(task) == Control::Stop {
                        return Control::Stop;
                    }
                    if !self.backlog.is_empty() {
                        return Control::Continue;
                    }
                }
                Err(TryRecvError::Empty) => return Control::Continue,
                Err(TryRecvError::Disconnected) => return Control::Stop,
            }
        }
    }

    fn wait(&mut self) -> Control {
        if !self.backlog.is_empty() {
            // The refused worker is being served by another lane; its queue will have room soon.
            std::thread::sleep(BACKLOG_WAIT);
            return Control::Continue;
        }
        match self.receiver.recv_timeout(WAIT) {
            Ok(task) => self.ingest(task),
            Err(RecvTimeoutError::Timeout) => Control::Continue,
            Err(RecvTimeoutError::Disconnected) => Control::Stop,
        }
    }
}

impl SyncIndexer for SmgRun {
    fn worker(
        &self,
        event_receiver: Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let lane = self.next_lane.fetch_add(1, Ordering::Relaxed);
        anyhow::ensure!(
            lane < self.pool.config().lanes,
            "SMG run backend was built for {} lanes; lane {lane} is one too many",
            self.pool.config().lanes
        );
        let mut hooks = Lane {
            run: self,
            lane,
            receiver: event_receiver,
            counters: metrics.as_ref().map(|m| m.prebind()),
            fed: FxHashMap::default(),
            backlog: VecDeque::new(),
        };
        self.pool.run_lane(lane, &mut hooks);
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
        let lane = self.inner.lane_stats();
        let pool = self.pool.metrics();
        let memberships = self.inner.current_size();
        let distinct = self.inner.entry_count();
        let map_slots = self.map_slots.load(Ordering::Relaxed);
        let map_bytes = self.map_bytes.load(Ordering::Relaxed);
        let index_bytes = stats.arena_bytes + stats.header_bytes;
        // What the process allocator sees: whole arena and slab chunks plus the maps.
        let allocated_bytes = stats.arena_chunk_bytes + stats.slab_bytes + map_bytes;
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
             allocated: arena chunks {} (free-listed {}) slab {} maps {map_bytes} = {} ({:.1} B per membership)\n  \
             lookups = {} runs walked per lookup = {:.2}\n  \
             lanes: busy {:.3} s idle {:.3} s\n  \
             pool: enqueued {} applied {} refused {} steals {} max depth {} (cap {}) max queued {}\n  \
             locks root/own/shared = {:?} contended = {:?} wait_ms = {:?}\n  \
             restarts = {} splits divergence/hole/stale-parent = {}/{}/{} lock-free inserts = {}\n  \
             stores = {} ({:.1} blocks each): resolve {:.0} ns, walk {:.0} ns, lane map {:.0} ns per event\n  \
             removals = {} ({:.1} blocks each): lane map {:.0} ns, grouping {:.0} ns, runs {:.0} ns per event",
            stats.runs_allocated,
            stats.runs_live,
            stats.blocks_live,
            stats.arena_bytes,
            stats.header_bytes,
            per_membership(index_bytes),
            per_membership(map_bytes),
            per_membership(index_bytes + map_bytes),
            stats.arena_chunk_bytes,
            stats.arena_free_bytes,
            stats.slab_bytes,
            allocated_bytes,
            per_membership(allocated_bytes),
            self.lookups.load(Ordering::Relaxed),
            self.runs_walked.load(Ordering::Relaxed) as f64
                / self.lookups.load(Ordering::Relaxed).max(1) as f64,
            pool.busy_ns as f64 / 1e9,
            pool.idle_ns as f64 / 1e9,
            pool.enqueued,
            pool.applied,
            pool.rejected,
            pool.steals,
            pool.max_depth,
            DEPTH_CAP,
            pool.max_queued,
            lane.locks,
            lane.contended,
            [
                lane.wait_ns[0] as f64 / 1e6,
                lane.wait_ns[1] as f64 / 1e6,
                lane.wait_ns[2] as f64 / 1e6,
            ],
            lane.restarts,
            lane.splits_divergence,
            lane.splits_hole,
            lane.splits_stale_parent,
            lane.inserts,
            lane.stores,
            lane.store_blocks as f64 / lane.stores.max(1) as f64,
            lane.store_ns[0] as f64 / lane.stores.max(1) as f64,
            lane.store_ns[1] as f64 / lane.stores.max(1) as f64,
            lane.store_ns[2] as f64 / lane.stores.max(1) as f64,
            lane.removes,
            lane.remove_blocks as f64 / lane.removes.max(1) as f64,
            lane.remove_ns[0] as f64 / lane.removes.max(1) as f64,
            lane.remove_ns[1] as f64 / lane.removes.max(1) as f64,
            lane.remove_ns[2] as f64 / lane.removes.max(1) as f64,
        )
    }
}
