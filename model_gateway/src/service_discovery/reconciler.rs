//! Level-triggered worker reconciliation.
//!
//! Takes a desired-worker snapshot, diffs it against the registry entries this
//! reconciler owns, and submits the gap as `AddWorker`/`RemoveWorker` jobs.
//! Failed or missed work is retried on the next pass by construction.
//!
//! The module boundary is provider-neutral — it never sees a Kubernetes `Pod`
//! — but the ownership model is still Kubernetes-specific: identity is a Pod
//! UID carried in [`POD_UID_LABEL`]. Generalizing that to a provider-agnostic
//! `(provider, id, spec-hash)` triple is deliberately left to a follow-up so
//! this extraction stays behavior-preserving.

use std::{collections::HashMap, sync::Arc};

use openai_protocol::worker::{WorkerSpec, WorkerType};
use tokio::time;
use tracing::{error, info, warn};

use super::provider::DiscoveryKind;
use crate::{
    app_context::AppContext,
    observability::metrics::{metrics_labels, Metrics},
    worker::{endpoint::Endpoint, registry::WorkerId, EndpointKey, WorkerOrigin},
    workflow::{Job, WorkerRegistrationMode},
};

/// Labels stamped on every discovery-created worker; workers carrying
/// [`POD_UID_LABEL`] are owned (added/removed) by the K8s reconciler.
pub const POD_NAME_LABEL: &str = "smg.ai/pod-name";
pub const POD_UID_LABEL: &str = "smg.ai/pod-uid";

/// Which provider owns the worker: `kubernetes`, `file`, `slurm` or `consul`.
pub const DISCOVERY_PROVIDER_LABEL: &str = "smg.ai/discovery-provider";
/// The provider's stable identity for the worker.
pub const DISCOVERY_ID_LABEL: &str = "smg.ai/discovery-id";
/// BLAKE3 fingerprint of the worker's [`DiscoveredWorkerSpec`].
pub const DISCOVERY_SPEC_HASH_LABEL: &str = "smg.ai/discovery-spec-hash";

/// The portable description of one discovered worker.
///
/// Every provider fills this the same way, and it is exactly what the
/// desired-spec fingerprint covers: nothing provider-specific, and nothing the
/// router injects afterwards (API key, ownership labels, retry budget) or the
/// registration workflow discovers at runtime (connection mode, backend,
/// served models, DP ranks). Keeping it that narrow is what makes a changed
/// fingerprint mean "the source changed this worker" rather than noise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiscoveredWorkerSpec {
    /// Parsed once by the provider, so a bad address is reported against the
    /// record that published it. Held parsed rather than as a `String` so the
    /// derived `Debug` here goes through [`Endpoint`]'s credential masking.
    pub(super) endpoint: Endpoint,
    pub(super) worker_type: WorkerType,
    pub(super) bootstrap_port: Option<u16>,
    pub(super) model_id_override: Option<String>,
    pub(super) kv_connector: Option<String>,
    pub(super) kv_role: Option<String>,
    pub(super) kv_engine_id: Option<String>,
}

impl DiscoveredWorkerSpec {
    /// A deterministic BLAKE3 fingerprint of every field.
    ///
    /// The encoding is explicit rather than derived: versioned, with each field
    /// written as present-or-absent plus a length prefix, so no value can forge
    /// a field boundary and a later change to the projection cannot silently
    /// collide with this one. Field order is fixed by this function, not by the
    /// struct.
    ///
    /// The endpoint is hashed as rendered — scheme included — because the
    /// spelling decides how the worker is registered (`h:p` dual-probes HTTP
    /// and gRPC; `grpc://h:p` does not). Two spellings that parse to the same
    /// endpoint render, and so hash, identically.
    ///
    /// Never log this next to the endpoint: the rendered form can carry
    /// credentials, and the pair would let a reader test guesses offline.
    pub(super) fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"smg-discovered-worker-spec/v1");
        hash_field(&mut hasher, Some(&self.endpoint.render()));
        hash_field(&mut hasher, Some(&self.worker_type.to_string()));
        hash_field(
            &mut hasher,
            self.bootstrap_port.map(|p| p.to_string()).as_deref(),
        );
        hash_field(&mut hasher, self.model_id_override.as_deref());
        hash_field(&mut hasher, self.kv_connector.as_deref());
        hash_field(&mut hasher, self.kv_role.as_deref());
        hash_field(&mut hasher, self.kv_engine_id.as_deref());
        hasher.finalize().to_hex().to_string()
    }
}

fn hash_field(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
    }
}

/// One worker the reconciler wants registered.
#[derive(Debug, Clone)]
pub(super) struct DesiredWorker {
    /// The provider's stable identity for this worker. Unique within a
    /// snapshot: Kubernetes uses `{pod_uid}:{port}`, because one Pod can
    /// serve several ports and each is its own worker.
    pub(super) id: String,
    pub(super) spec: DiscoveredWorkerSpec,
    /// Kubernetes compatibility labels, written as before so existing
    /// selectors and dashboards keep working. Deliberately outside
    /// [`DiscoveredWorkerSpec`]: they are neither portable nor fingerprinted.
    pub(super) pod_name: String,
    pub(super) pod_uid: String,
}

/// Desired view of the cluster derived from the store snapshot.
#[derive(Debug, Default)]
pub(super) struct DesiredState {
    /// Owning pod uid per canonical endpoint for Ready, non-terminating Pods.
    /// Registered workers whose URL is absent — or owned by a different Pod
    /// uid — enter the existing drain/remove workflow.
    pub(super) uid_by_url: HashMap<EndpointKey, String>,
    /// Workers on Running, Ready Pods — registration candidates.
    pub(super) addable: Vec<DesiredWorker>,
}

/// A registry worker owned by K8s discovery (stamped with [`POD_UID_LABEL`]).
#[derive(Debug, Clone)]
pub(super) struct OwnedWorker {
    /// The registry id. A DP group shares one canonical [`Self::key`], so the
    /// id is what distinguishes its ranks — and what the removal guard needs
    /// to pin each rank to its own revision.
    pub(super) id: WorkerId,
    /// The registered address, parsed. Its [`Endpoint::key`] is the identity
    /// used for grouping; the endpoint itself is kept so a removal can submit
    /// a form that parses back (an IPC key is a bare socket path).
    pub(super) endpoint: Endpoint,
    pub(super) pod_uid: String,
    /// Revision guard for removal: a concurrently replaced worker is skipped
    /// and re-evaluated on the next pass instead of removed blindly.
    pub(super) revision: u64,
}

/// One canonical URL to remove, carrying every registry worker that shares it.
///
/// DP-rank expansions collapse to one canonical URL but hold independent
/// revisions, so a single scalar cannot guard the group: it would silently
/// drop the ranks whose revision differs, leaving them registered against a
/// Pod that is already gone.
#[derive(Debug, Clone)]
pub(super) struct RemovalTarget {
    /// One member's parsed address; every member shares its key.
    pub(super) endpoint: Endpoint,
    /// Pod uid of whichever member the registry happened to yield first.
    /// Ranks of one DP group do share it, but a stale-scheme sibling can not:
    /// `grpc://h:p` and `http://h:p` canonicalize alike, so two registrations
    /// from different Pods can land in one target. Logged only, never matched
    /// on — the removal is decided by [`Self::guards`].
    pub(super) pod_uid: String,
    /// `(id, revision)` as observed in this snapshot, one entry per rank.
    pub(super) guards: Vec<(WorkerId, u64)>,
}

/// Snapshot the registry workers this reconciler owns: locally registered
/// (never mesh-imported) and stamped with the pod-uid label. Manually added
/// workers lack the label and are never touched.
fn k8s_owned_workers(app_context: &AppContext) -> Vec<OwnedWorker> {
    app_context
        .worker_registry
        .get_all_with_ids()
        .into_iter()
        .filter_map(|(id, worker)| {
            if app_context.worker_registry.origin_of(&id) != Some(WorkerOrigin::Local) {
                return None;
            }
            let pod_uid = worker.metadata().spec.labels.get(POD_UID_LABEL)?.clone();
            // A registered address the shared parser rejects is one this
            // reconciler cannot safely match against a desired endpoint, so it
            // is left alone rather than guessed at.
            let endpoint = match Endpoint::parse_with_rank(worker.url()) {
                Ok((endpoint, _)) => endpoint,
                Err(e) => {
                    warn!(
                        worker_url = %worker.url(),
                        error = %e,
                        "Skipping discovery-owned worker with an unparsable address"
                    );
                    return None;
                }
            };
            Some(OwnedWorker {
                id,
                endpoint,
                pod_uid,
                revision: worker.revision(),
            })
        })
        .collect()
}

#[derive(Debug, Default)]
pub(super) struct ReconcileActions {
    /// Workers to register: new URLs, plus same-URL pods whose uid changed
    /// (restart with a stable IP).
    pub(super) add: Vec<DesiredWorker>,
    /// Workers to remove: URL gone from the desired set, or owned by a pod
    /// uid that no longer holds the URL (covers a stale-scheme sibling the
    /// same-URL Upsert cannot replace). One entry per canonical URL, carrying
    /// every rank that shares it.
    pub(super) remove: Vec<RemovalTarget>,
}

pub(super) fn compute_actions(
    desired: &DesiredState,
    registered: &[OwnedWorker],
) -> ReconcileActions {
    let mut actions = ReconcileActions::default();

    let mut registered_uid: HashMap<EndpointKey, &str> = HashMap::new();
    // DP-rank expansions share one canonical URL: remove it once, but keep
    // every rank's own `(id, revision)` so the guard cannot drop the ranks
    // whose revision happens to differ from an arbitrarily chosen one.
    let mut remove_by_key: HashMap<EndpointKey, RemovalTarget> = HashMap::new();
    for worker in registered {
        let key = worker.endpoint.key();
        registered_uid.insert(key.clone(), worker.pod_uid.as_str());
        match desired.uid_by_url.get(&key) {
            Some(uid) if *uid == worker.pod_uid => {}
            _ => remove_by_key
                .entry(key)
                .or_insert_with(|| RemovalTarget {
                    endpoint: worker.endpoint.clone(),
                    pod_uid: worker.pod_uid.clone(),
                    guards: Vec::new(),
                })
                .guards
                .push((worker.id.clone(), worker.revision)),
        }
    }
    actions.remove = remove_by_key.into_values().collect();
    // `HashMap` iteration order is unspecified; sort so a pass submits jobs
    // and logs them in a stable order.
    actions
        .remove
        .sort_unstable_by_key(|target| target.endpoint.key());
    for target in &mut actions.remove {
        target
            .guards
            .sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    }

    for worker in &desired.addable {
        match registered_uid.get(&worker.spec.endpoint.key()) {
            Some(uid) if *uid == worker.pod_uid => {}
            _ => actions.add.push(worker.clone()),
        }
    }
    actions
}

fn build_worker_spec(
    desired: &DesiredWorker,
    kind: DiscoveryKind,
    app_context: &AppContext,
) -> WorkerSpec {
    let discovered = &desired.spec;
    let mut spec = WorkerSpec::new(discovered.endpoint.render());
    spec.worker_type = discovered.worker_type;
    spec.bootstrap_port = discovered.bootstrap_port;
    // Provenance first, from the spec alone: the fingerprint must not see the
    // API key, retry budget or labels injected below.
    spec.labels.insert(
        DISCOVERY_PROVIDER_LABEL.to_string(),
        kind.as_label().to_string(),
    );
    spec.labels
        .insert(DISCOVERY_ID_LABEL.to_string(), desired.id.clone());
    spec.labels.insert(
        DISCOVERY_SPEC_HASH_LABEL.to_string(),
        discovered.fingerprint(),
    );
    spec.labels
        .insert(POD_NAME_LABEL.to_string(), desired.pod_name.clone());
    spec.labels
        .insert(POD_UID_LABEL.to_string(), desired.pod_uid.clone());
    // served_model_name is priority #2 in create_worker's model_id chain.
    if let Some(ref model_id) = discovered.model_id_override {
        spec.labels
            .insert("served_model_name".to_string(), model_id.clone());
    }
    spec.kv_connector.clone_from(&discovered.kv_connector);
    spec.kv_role.clone_from(&discovered.kv_role);
    spec.kv_engine_id.clone_from(&discovered.kv_engine_id);
    spec.api_key.clone_from(&app_context.router_config.api_key);
    spec.max_connection_attempts = app_context
        .router_config
        .health_check
        .success_threshold
        .max(1)
        * 20;
    spec
}

/// One reconcile pass: diff the desired workers a provider published against
/// the registry entries this reconciler owns and submit Add/Remove jobs for
/// the gap. Failed or missed work is retried on the next pass by construction.
///
/// `started_at` is taken by the caller so the sync-duration metric still
/// covers the provider's own snapshot conversion, as it did when this
/// function read the reflector store itself.
pub(super) async fn reconcile(
    desired: &DesiredState,
    kind: DiscoveryKind,
    app_context: &Arc<AppContext>,
    started_at: time::Instant,
) {
    let registered = k8s_owned_workers(app_context);
    let actions = compute_actions(desired, &registered);

    let desired_count = desired.uid_by_url.len();
    if actions.add.is_empty() && actions.remove.is_empty() {
        Metrics::set_discovery_workers_discovered(kind.metric_label(), desired_count);
        return;
    }

    let Some(job_queue) = app_context.worker_job_queue.get() else {
        warn!(
            "JobQueue not initialized; deferring {} addition(s), {} removal(s)",
            actions.add.len(),
            actions.remove.len()
        );
        return;
    };

    // One in-flight job per URL: a pending/processing job owns that worker's
    // transition (a duplicate removal would find it already Draining, skip
    // the settle sleep, and collapse the drain window). Completed/failed
    // statuses do not block, so failures retry on the next pass.
    let in_flight = |url: &str| {
        job_queue
            .get_status(url)
            .is_some_and(|status| status.status == "pending" || status.status == "processing")
    };
    let removals: Vec<&RemovalTarget> = actions
        .remove
        .iter()
        .filter(|target| !in_flight(&target.endpoint.lookup_form()))
        .collect();
    let additions: Vec<&DesiredWorker> = actions
        .add
        .iter()
        .filter(|worker| !in_flight(&worker.spec.endpoint.render()))
        .collect();

    if removals.is_empty() && additions.is_empty() {
        Metrics::set_discovery_workers_discovered(kind.metric_label(), desired_count);
        return;
    }

    info!(
        "Reconciling workers: {} to add, {} to remove ({} desired)",
        additions.len(),
        removals.len(),
        desired_count
    );

    for target in removals {
        info!(
            "Removing worker {} ({} registration(s), pod {}): pod unready, gone, \
             terminating, or replaced",
            target.endpoint.redacted(),
            target.guards.len(),
            target.pod_uid
        );
        // Scheme-less for a network address, so `find_workers_by_url` reaches
        // every spelling the group was registered under. An IPC endpoint keeps
        // its scheme, because its key is a bare socket path that would not
        // parse back.
        let job = Job::RemoveWorker {
            url: target.endpoint.lookup_form(),
            expected_revisions: Some(
                target
                    .guards
                    .iter()
                    .map(|(id, revision)| (id.as_str().to_string(), *revision))
                    .collect(),
            ),
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_deregistration(
                kind.metric_label(),
                metrics_labels::DEREGISTRATION_RECONCILED,
            ),
            Err(e) => error!(
                "Failed to submit worker removal for {}: {}",
                target.endpoint.redacted(),
                e
            ),
        }
    }

    for worker in additions {
        info!(
            "Registering worker {} ({:?}) for pod {}",
            worker.spec.endpoint, worker.spec.worker_type, worker.pod_name
        );
        let job = Job::AddWorker {
            config: Box::new(build_worker_spec(worker, kind, app_context)),
            registration_mode: WorkerRegistrationMode::Upsert,
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_registration(
                kind.metric_label(),
                metrics_labels::REGISTRATION_SUCCESS,
            ),
            Err(e) => {
                error!(
                    "Failed to submit worker addition for {}: {}",
                    worker.spec.endpoint, e
                );
                Metrics::record_discovery_registration(
                    kind.metric_label(),
                    metrics_labels::REGISTRATION_FAILED,
                );
            }
        }
    }

    Metrics::set_discovery_workers_discovered(kind.metric_label(), desired_count);
    Metrics::record_discovery_sync_duration(kind.metric_label(), started_at.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_discovery::testing::create_test_app_context;

    fn key(url: &str) -> EndpointKey {
        crate::worker::endpoint_key(url).expect(url)
    }

    /// What `canonical_host_port` used to assert, now served by the shared
    /// parser — plus the spellings it got wrong: an uppercase scheme kept its
    /// prefix, and `@` inside a path or userinfo truncated the key.
    #[test]
    fn ownership_grouping_uses_the_shared_canonical_key() {
        for spelling in [
            "10.0.0.1:8080",
            "http://10.0.0.1:8080",
            "grpc://10.0.0.1:8080@2",
            "10.0.0.1:8080@0",
            "HTTPS://10.0.0.1:8080",
        ] {
            assert_eq!(key(spelling).as_str(), "10.0.0.1:8080", "{spelling}");
        }
        // Both were truncated by `canonical_host_port`'s split on the first
        // `@`: to `/tmp/a` and to `user`.
        assert_eq!(key("ipc:///tmp/a@b.sock").as_str(), "/tmp/a@b.sock");
        assert_eq!(key("http://user@host:8080").as_str(), "user@host:8080");
    }

    fn spec_of(url: &str) -> DiscoveredWorkerSpec {
        DiscoveredWorkerSpec {
            endpoint: Endpoint::parse_with_rank(url).expect(url).0,
            worker_type: WorkerType::Regular,
            bootstrap_port: None,
            model_id_override: None,
            kv_connector: None,
            kv_role: None,
            kv_engine_id: None,
        }
    }

    #[test]
    fn fingerprint_is_deterministic_and_spelling_insensitive() {
        let a = spec_of("10.0.0.1:8080");
        assert_eq!(a.fingerprint(), a.clone().fingerprint());
        // Two spellings that parse to one endpoint register identically.
        assert_eq!(
            spec_of("[::1]:8080").fingerprint(),
            spec_of("[0:0:0:0:0:0:0:1]:8080").fingerprint()
        );
    }

    /// Every field is part of the fingerprint, so a change to any of them is a
    /// change the source made to this worker and warrants a replacement.
    #[test]
    fn every_field_moves_the_fingerprint() {
        let base = spec_of("10.0.0.1:8080");
        let variants = [
            DiscoveredWorkerSpec {
                endpoint: Endpoint::parse("grpc://10.0.0.1:8080").unwrap(),
                ..base.clone()
            },
            DiscoveredWorkerSpec {
                worker_type: WorkerType::Prefill,
                ..base.clone()
            },
            DiscoveredWorkerSpec {
                bootstrap_port: Some(9000),
                ..base.clone()
            },
            DiscoveredWorkerSpec {
                model_id_override: Some("m".to_string()),
                ..base.clone()
            },
            DiscoveredWorkerSpec {
                kv_connector: Some("c".to_string()),
                ..base.clone()
            },
            DiscoveredWorkerSpec {
                kv_role: Some("r".to_string()),
                ..base.clone()
            },
            DiscoveredWorkerSpec {
                kv_engine_id: Some("e".to_string()),
                ..base.clone()
            },
        ];
        let mut seen = vec![base.fingerprint()];
        for variant in &variants {
            let hash = variant.fingerprint();
            assert!(!seen.contains(&hash), "{variant:?} collided");
            seen.push(hash);
        }
    }

    /// The length prefix is what keeps a value from forging a field boundary.
    /// The presence byte alone separates `"ab"|"c"` from `"a"|"bc"`, but not a
    /// value that contains the presence byte itself: without lengths, both of
    /// these encode as `\x01 a \x01 b \x01 c`. Provider input is operator-written
    /// text, so a control byte in it is not hypothetical.
    #[test]
    fn a_value_cannot_forge_a_field_boundary() {
        let left = DiscoveredWorkerSpec {
            kv_connector: Some("a\u{1}b".to_string()),
            kv_role: Some("c".to_string()),
            ..spec_of("10.0.0.1:8080")
        };
        let right = DiscoveredWorkerSpec {
            kv_connector: Some("a".to_string()),
            kv_role: Some("b\u{1}c".to_string()),
            ..spec_of("10.0.0.1:8080")
        };
        assert_ne!(left.fingerprint(), right.fingerprint());

        // Absent and empty are different answers, which the presence byte
        // (not the length) guarantees.
        let absent = spec_of("10.0.0.1:8080");
        let empty = DiscoveredWorkerSpec {
            kv_role: Some(String::new()),
            ..absent.clone()
        };
        assert_ne!(absent.fingerprint(), empty.fingerprint());
    }

    /// The hash is taken from the portable spec alone, so rotating the router's
    /// API key or changing the retry budget cannot trigger a replacement.
    #[test]
    fn build_worker_spec_stamps_provenance_from_the_spec_alone() {
        let app_context = create_test_app_context();
        let desired = desired_worker("10.0.0.1:8080", "uid-1");
        let spec = build_worker_spec(&desired, DiscoveryKind::Kubernetes, &app_context);

        assert_eq!(
            spec.labels
                .get(DISCOVERY_PROVIDER_LABEL)
                .map(String::as_str),
            Some("kubernetes")
        );
        assert_eq!(spec.labels.get(DISCOVERY_ID_LABEL), Some(&desired.id));
        assert_eq!(
            spec.labels.get(DISCOVERY_SPEC_HASH_LABEL),
            Some(&desired.spec.fingerprint())
        );
        // The compatibility labels are still written.
        assert_eq!(
            spec.labels.get(POD_UID_LABEL).map(String::as_str),
            Some("uid-1")
        );
    }

    fn desired_worker(url: &str, uid: &str) -> DesiredWorker {
        DesiredWorker {
            id: format!("{uid}:{url}"),
            spec: spec_of(url),
            pod_name: "w".to_string(),
            pod_uid: uid.to_string(),
        }
    }

    fn desired_state_of(workers: &[DesiredWorker]) -> DesiredState {
        let mut state = DesiredState::default();
        for worker in workers {
            state
                .uid_by_url
                .insert(worker.spec.endpoint.key(), worker.pod_uid.clone());
            state.addable.push(worker.clone());
        }
        state
    }

    fn owned(url: &str, uid: &str) -> OwnedWorker {
        owned_rank(url, uid, url, 1)
    }

    /// One rank of a DP group: same canonical key, distinct id and revision.
    fn owned_rank(url: &str, uid: &str, id: &str, revision: u64) -> OwnedWorker {
        OwnedWorker {
            id: WorkerId::from_string(id.to_string()),
            endpoint: Endpoint::parse_with_rank(url).expect(url).0,
            pod_uid: uid.to_string(),
            revision,
        }
    }

    #[test]
    fn test_compute_actions_adds_missing_workers() {
        let desired = desired_state_of(&[
            desired_worker("10.0.0.1:8080", "u1"),
            desired_worker("10.0.0.1:8081", "u1"),
        ]);
        let actions = compute_actions(&desired, &[]);
        assert_eq!(actions.add.len(), 2);
        assert!(actions.remove.is_empty());
    }

    #[test]
    fn test_compute_actions_same_uid_metadata_change_is_noop() {
        let mut worker = desired_worker("10.0.0.1:8080", "u1");
        worker.spec.kv_connector = Some("NixlConnector".to_string());
        let desired = desired_state_of(&[worker]);
        let registered = [owned("10.0.0.1:8080", "u1")];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert!(actions.remove.is_empty());
    }

    #[test]
    fn test_compute_actions_removes_workers_for_gone_pods() {
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "u1")]);
        let registered = [owned("10.0.0.1:8080", "u1"), owned("10.0.0.2:8080", "u2")];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].endpoint.key().as_str(), "10.0.0.2:8080");
    }

    #[test]
    fn test_compute_actions_uid_change_removes_and_reregisters_same_url() {
        // Same-IP pod restart (hostNetwork / stable IP): URL unchanged but
        // uid differs → the stale worker is removed (covers a scheme-flipped
        // sibling the Upsert cannot replace) and the new one registered.
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "uid-new")]);
        let registered = [owned("10.0.0.1:8080", "uid-old")];
        let actions = compute_actions(&desired, &registered);
        assert_eq!(actions.add.len(), 1);
        assert_eq!(actions.add[0].pod_uid, "uid-new");
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].pod_uid, "uid-old");
    }

    #[test]
    fn test_compute_actions_dp_ranks_removed_once() {
        let registered = [
            owned_rank("10.0.0.1:8080", "u1", "w@0", 1),
            owned_rank("10.0.0.1:8080", "u1", "w@1", 1),
        ];
        let actions = compute_actions(&DesiredState::default(), &registered);
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].endpoint.key().as_str(), "10.0.0.1:8080");
        assert_eq!(actions.remove[0].guards.len(), 2);
        assert!(actions.add.is_empty());
    }

    /// The group collapses to one removal job, but every rank must keep its
    /// own revision. Carrying a single revision retained only the ranks that
    /// happened to share it and left the others registered against a Pod that
    /// was already gone.
    #[test]
    fn dp_ranks_keep_their_own_revision_when_diverged() {
        let registered = [
            owned_rank("10.0.0.1:8080", "u1", "w@0", 7),
            owned_rank("10.0.0.1:8080", "u1", "w@1", 2),
        ];
        let actions = compute_actions(&DesiredState::default(), &registered);
        assert_eq!(actions.remove.len(), 1, "one job per canonical URL");
        let guards: Vec<(&str, u64)> = actions.remove[0]
            .guards
            .iter()
            .map(|(id, revision)| (id.as_str(), *revision))
            .collect();
        assert_eq!(guards, vec![("w@0", 7), ("w@1", 2)]);
    }

    /// Two pods behind one canonical URL keep separate guards per rank, so a
    /// shared revision value cannot make one pod's rank stand in for another's.
    #[test]
    fn diverged_ranks_do_not_collapse_on_equal_revisions() {
        let registered = [
            owned_rank("10.0.0.1:8080", "u1", "w@0", 3),
            owned_rank("10.0.0.1:8080", "u1", "w@1", 3),
            owned_rank("10.0.0.1:8081", "u1", "x@0", 3),
        ];
        let actions = compute_actions(&DesiredState::default(), &registered);
        assert_eq!(actions.remove.len(), 2, "one job per canonical URL");
        assert_eq!(actions.remove[0].guards.len(), 2);
        assert_eq!(actions.remove[1].guards.len(), 1);
    }

    #[test]
    fn test_k8s_owned_workers_scoped_by_label_and_origin() {
        use openai_protocol::model_card::ModelCard;

        use crate::worker::BasicWorkerBuilder;

        let app_context = create_test_app_context();

        let mut labels = HashMap::new();
        labels.insert(POD_UID_LABEL.to_string(), "uid-1".to_string());
        let discovered = Arc::new(
            BasicWorkerBuilder::new("http://10.0.0.1:8080")
                .model(ModelCard::new("m"))
                .labels(labels)
                .build(),
        );
        let manual = Arc::new(
            BasicWorkerBuilder::new("http://10.0.0.2:8080")
                .model(ModelCard::new("m"))
                .build(),
        );
        app_context.worker_registry.register(discovered).unwrap();
        app_context.worker_registry.register(manual).unwrap();

        let owned = k8s_owned_workers(&app_context);
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].endpoint.key().as_str(), "10.0.0.1:8080");
        assert_eq!(owned[0].pod_uid, "uid-1");
    }

    #[test]
    fn test_k8s_owned_workers_excludes_mesh_imported_workers() {
        // A peer's discovered worker arrives via mesh sync carrying the
        // pod-uid label; its pod is absent from this node's store, so without
        // the Local-origin filter the reconciler would remove it every pass.
        let app_context = create_test_app_context();

        let mut spec = WorkerSpec::new("http://10.0.0.3:8080");
        spec.labels
            .insert(POD_UID_LABEL.to_string(), "uid-mesh".to_string());
        let state = smg_mesh::WorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "m".to_string(),
            url: "http://10.0.0.3:8080".to_string(),
            health: true,
            load: 0.0,
            version: 0,
            spec: serde_json::to_vec(&spec).unwrap(),
        };
        app_context.worker_registry.on_remote_worker_state(&state);
        assert!(app_context
            .worker_registry
            .get_by_url("http://10.0.0.3:8080")
            .is_some());

        assert!(k8s_owned_workers(&app_context).is_empty());
    }

    #[test]
    fn test_build_worker_spec_stamps_ownership_labels() {
        let app_context = create_test_app_context();
        let desired = DesiredWorker {
            id: "uid-1:8081".to_string(),
            spec: DiscoveredWorkerSpec {
                worker_type: WorkerType::Prefill,
                bootstrap_port: Some(9080),
                model_id_override: Some("llama".to_string()),
                kv_connector: Some("MooncakeConnector".to_string()),
                kv_engine_id: Some("engine-1".to_string()),
                ..spec_of("10.0.0.1:8081")
            },
            pod_name: "prefill-0".to_string(),
            pod_uid: "uid-1".to_string(),
        };
        let spec = build_worker_spec(&desired, DiscoveryKind::Kubernetes, &app_context);
        assert_eq!(spec.url, "10.0.0.1:8081");
        assert_eq!(spec.worker_type, WorkerType::Prefill);
        assert_eq!(spec.bootstrap_port, Some(9080));
        assert_eq!(spec.kv_connector.as_deref(), Some("MooncakeConnector"));
        assert_eq!(spec.kv_engine_id.as_deref(), Some("engine-1"));
        assert_eq!(
            spec.labels.get(POD_NAME_LABEL),
            Some(&"prefill-0".to_string())
        );
        assert_eq!(spec.labels.get(POD_UID_LABEL), Some(&"uid-1".to_string()));
        assert_eq!(
            spec.labels.get("served_model_name"),
            Some(&"llama".to_string())
        );
    }

    #[test]
    fn test_deregistration_reconciled_metric_label() {
        // Verify the metric label constant exists and has expected value
        assert_eq!(metrics_labels::DEREGISTRATION_RECONCILED, "reconciled");
    }

    #[tokio::test]
    async fn test_reconcile_without_job_queue_is_safe() {
        let app_context = create_test_app_context();
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "u1")]);
        reconcile(
            &desired,
            DiscoveryKind::Kubernetes,
            &app_context,
            time::Instant::now(),
        )
        .await;
    }
}
