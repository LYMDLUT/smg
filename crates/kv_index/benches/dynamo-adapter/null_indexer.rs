//! A backend that stores nothing and matches nothing: the memory and scheduling floor of the
//! Mooncake replay (corpus, lanes, queues, observation records) against which the resident bytes
//! of a real backend are read.

use flume::Receiver;

use super::metrics::KvIndexerMetrics;
use super::traits::SyncIndexer;
use super::types::{WorkerLookupStats, WorkerTask};
use super::KvRouterError;
use crate::protocols::{LocalBlockHash, OverlapScores, WorkerWithDpRank};

pub struct NullIndexer;

impl SyncIndexer for NullIndexer {
    fn worker(
        &self,
        event_receiver: Receiver<WorkerTask>,
        _metrics: Option<std::sync::Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        #[cfg(feature = "bench")]
        let mut observation = super::observation::WorkerObservationState::default();
        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(_) => {}
                WorkerTask::EventWithAck { resp, .. } => {
                    let _ = resp.send(true);
                }
                WorkerTask::ApproximateLru(task) => task.complete(Err(KvRouterError::Unsupported(
                    "approximate LRU requires ConcurrentRadixTreeCompressed".to_string(),
                ))),
                #[cfg(feature = "bench")]
                WorkerTask::InstallObservation { writer, resp } => {
                    observation.install(writer, resp);
                }
                #[cfg(feature = "bench")]
                WorkerTask::ObservedEvent { correlation_id, .. } => {
                    observation.record(correlation_id, true);
                }
                #[cfg(feature = "bench")]
                WorkerTask::SealObservation(resp) => observation.seal(resp),
                #[cfg(feature = "bench")]
                WorkerTask::HarvestObservation(resp) => observation.harvest(resp),
                WorkerTask::Anchor { .. } => {}
                WorkerTask::RemoveWorker { resp, .. } => {
                    let _ = resp.send(());
                }
                WorkerTask::RemoveWorkerDpRank { .. } => {}
                WorkerTask::CleanupStaleChildren => {}
                WorkerTask::DumpEvents(sender) => {
                    let _ = sender.send(Err(anyhow::anyhow!(
                        "event dumps are not supported by the null backend"
                    )));
                }
                WorkerTask::Stats(sender) => {
                    let _ = sender.send(WorkerLookupStats::from_worker_block_counts(
                        std::iter::empty::<(WorkerWithDpRank, usize)>(),
                    ));
                }
                WorkerTask::ContainsWorkerBlock { resp, .. } => {
                    let _ = resp.send(false);
                }
                WorkerTask::Flush(sender) => {
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => break,
            }
        }
        Ok(())
    }

    fn find_matches(&self, _sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
        OverlapScores::new()
    }

    fn supports_event_dump(&self) -> bool {
        false
    }

    fn supports_routing_decision_pruning(&self) -> bool {
        false
    }
}
