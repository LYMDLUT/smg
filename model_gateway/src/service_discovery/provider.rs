//! The contract between a worker-discovery provider and the shared reconciler.
//!
//! A provider turns its source — Kubernetes Pods today; files, Slurm and Consul
//! next — into these types and nothing else. The reconciler sees only these, so
//! it never learns which kind of source produced a worker beyond the
//! [`DiscoveryKind`] it is handed.

use crate::observability::metrics::metrics_labels;

/// Which provider produced a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum DiscoveryKind {
    Kubernetes,
}

impl DiscoveryKind {
    /// The value written to the provider label on every worker this provider
    /// registers, and matched to decide which workers it owns.
    ///
    /// A literal rather than the metric label, even though the two agree today:
    /// this is an identity stamped on registered workers, and renaming a metric
    /// must not silently orphan them.
    pub(super) fn as_label(self) -> &'static str {
        match self {
            DiscoveryKind::Kubernetes => "kubernetes",
        }
    }

    /// The `discovery` value on discovery metrics.
    pub(super) fn metric_label(self) -> &'static str {
        match self {
            DiscoveryKind::Kubernetes => metrics_labels::DISCOVERY_KUBERNETES,
        }
    }
}
