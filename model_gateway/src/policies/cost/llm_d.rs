//! llm-d inference scheduler (EPP) profiles as SMG policies.
//!
//! The EPP scores every endpoint with a clamped weighted sum `S = Σ w · clamp01(s)` and picks the
//! maximum, rotating a global counter through exact ties (`scheduler_profile.go`). Three profiles
//! from the repository are expressed here; `--selection-policy-params` overlays any field of the
//! chosen preset.
//!
//! | policy | scorers (weight) | extras |
//! |---|---|---|
//! | `llm-d-optimized-baseline` | queue 2, kv-utilization 2, prefix 3, no-hit-lru 2 | saturation gate |
//! | `llm-d-precise-prefix` | active-request 4, kv-utilization 2, prefix 2, no-hit-lru 2 | saturation gate |
//! | `llm-d-sticky-until-saturated` | token-load 1 | sticky filter (`prefix-cache-affinity-filter`) |
//!
//! Scorers (`pkg/epp/framework/plugins/scheduling/scorer/`):
//! - prefix: `(1−λ)·match/total + λ·min(1, match·bs/8192)²`, `match` the matched blocks weighted
//!   by tier (device 1.0, host 0.8, shared 0.4) and truncated, `total` the prompt's full blocks;
//! - queue: `(q_max − q)/(q_max − q_min)`, 1 when all are equal;
//! - kv-utilization: `1 − kv_usage`;
//! - active-request: 1 at or under the idle threshold, else `(max − c)/max · max_busy_score`;
//! - no-hit-lru: when any endpoint holds a block of the prompt, 0.5 for all; otherwise never-used
//!   endpoints rank first by position, then used ones from least to most recently used, so cold
//!   prefixes spread instead of piling on one endpoint;
//! - token-load: `1 − min(1, (in-flight prefill tokens + uncached prompt tokens) / T)`.
//!
//! Sticky-until-saturated (`filter/prefixcacheaffinity`): endpoints holding at least
//! `affinity_threshold` of the prompt's blocks are *sticky*; if none, all compete. The request
//! stays with the sticky set unless the best sticky endpoint's prefill backlog, as time at
//! `peak_prefill_tokens_per_second`, exceeds the best cold endpoint's by more than
//! `max_ttft_penalty_ms`. Departure: the EPP compares backlogs alone; this port adds the request's
//! own uncached tokens on each side, since a cold endpoint must prefill the whole prompt.
//!
//! Saturation (`flowcontrol/saturationdetector`): an endpoint is saturated when
//! `q_eff = waiting + max(0, in_flight − waiting − running) ≥ 5·(1+headroom)` or
//! `kv_usage ≥ 0.8·(1+headroom)`; saturated endpoints are skipped while any other remains
//! (fail-open). The EPP also counts a report older than 200 ms as saturated; SMG's load snapshot
//! carries no age and is polled on a seconds cadence, so staleness is not applied.
//!
//! Signals: prefix overlap comes from SMG's index rather than the EPP's producers; in-flight prefill
//! tokens come from the backend's waiting-uncached report plus the router's optimistic bookings
//! when accounting is on (the EPP keeps them router-local, released at the first response chunk);
//! `speculative indexing` of the precise profile is the same accounting layer.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;

use super::{
    inputs::{CandidateInputs, RequestInputs},
    policy::{Needs, Pick, WorkerFilter, WorkerPicker, WorkerScorer, WorkerSelectionPolicy},
};

pub const OPTIMIZED_BASELINE: &str = "llm-d-optimized-baseline";
pub const PRECISE_PREFIX: &str = "llm-d-precise-prefix";
pub const STICKY_UNTIL_SATURATED: &str = "llm-d-sticky-until-saturated";

#[derive(Debug, Clone, Copy)]
pub struct LlmDParams {
    pub queue_weight: f64,
    pub kv_utilization_weight: f64,
    pub prefix_weight: f64,
    pub no_hit_lru_weight: f64,
    pub active_request_weight: f64,
    pub token_load_weight: f64,
    pub sticky_filter: bool,
    pub affinity_threshold: f64,
    pub max_ttft_penalty_ms: f64,
    pub peak_prefill_tokens_per_second: f64,
    pub queue_threshold_tokens: f64,
    /// λ of the prefix scorer's match-length blend.
    pub match_length_blend: f64,
    pub host_tier_weight: f64,
    pub disk_tier_weight: f64,
    pub active_request_idle_threshold: usize,
    pub active_request_max_busy_score: f64,
    pub saturation_filter: bool,
    pub saturation_queue_threshold: f64,
    pub saturation_kv_threshold: f64,
    pub saturation_headroom: f64,
    pub no_hit_lru_capacity: usize,
}

/// Overlay for a preset: every field optional, unknown names rejected.
#[derive(Debug, Clone, Copy, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmDParamsPatch {
    pub queue_weight: Option<f64>,
    pub kv_utilization_weight: Option<f64>,
    pub prefix_weight: Option<f64>,
    pub no_hit_lru_weight: Option<f64>,
    pub active_request_weight: Option<f64>,
    pub token_load_weight: Option<f64>,
    pub sticky_filter: Option<bool>,
    pub affinity_threshold: Option<f64>,
    pub max_ttft_penalty_ms: Option<f64>,
    pub peak_prefill_tokens_per_second: Option<f64>,
    pub queue_threshold_tokens: Option<f64>,
    pub match_length_blend: Option<f64>,
    pub host_tier_weight: Option<f64>,
    pub disk_tier_weight: Option<f64>,
    pub active_request_idle_threshold: Option<usize>,
    pub active_request_max_busy_score: Option<f64>,
    pub saturation_filter: Option<bool>,
    pub saturation_queue_threshold: Option<f64>,
    pub saturation_kv_threshold: Option<f64>,
    pub saturation_headroom: Option<f64>,
    pub no_hit_lru_capacity: Option<usize>,
}

impl LlmDParams {
    /// The repository's `optimized-baseline.yaml` profile.
    pub fn optimized_baseline() -> Self {
        Self {
            queue_weight: 2.0,
            kv_utilization_weight: 2.0,
            prefix_weight: 3.0,
            no_hit_lru_weight: 2.0,
            active_request_weight: 0.0,
            token_load_weight: 0.0,
            sticky_filter: false,
            affinity_threshold: 0.8,
            max_ttft_penalty_ms: 18_000.0,
            peak_prefill_tokens_per_second: 15_928.0,
            queue_threshold_tokens: 4_194_304.0,
            match_length_blend: 0.0,
            host_tier_weight: 0.8,
            disk_tier_weight: 0.4,
            active_request_idle_threshold: 0,
            active_request_max_busy_score: 0.5,
            saturation_filter: true,
            saturation_queue_threshold: 5.0,
            saturation_kv_threshold: 0.8,
            saturation_headroom: 0.0,
            no_hit_lru_capacity: 1024,
        }
    }

    /// The repository's `precise-prefix.yaml` profile.
    pub fn precise_prefix() -> Self {
        Self {
            queue_weight: 0.0,
            active_request_weight: 4.0,
            prefix_weight: 2.0,
            ..Self::optimized_baseline()
        }
    }

    /// `prefix-cache-affinity-filter` + `token-load-scorer`, the composition the EPP README
    /// recommends for sticky sessions.
    pub fn sticky_until_saturated() -> Self {
        Self {
            queue_weight: 0.0,
            kv_utilization_weight: 0.0,
            prefix_weight: 0.0,
            no_hit_lru_weight: 0.0,
            token_load_weight: 1.0,
            sticky_filter: true,
            saturation_filter: false,
            ..Self::optimized_baseline()
        }
    }

    pub fn apply(mut self, patch: &LlmDParamsPatch) -> Self {
        macro_rules! overlay {
            ($($field:ident),* $(,)?) => {
                $( if let Some(value) = patch.$field { self.$field = value; } )*
            };
        }
        overlay!(
            queue_weight,
            kv_utilization_weight,
            prefix_weight,
            no_hit_lru_weight,
            active_request_weight,
            token_load_weight,
            sticky_filter,
            affinity_threshold,
            max_ttft_penalty_ms,
            peak_prefill_tokens_per_second,
            queue_threshold_tokens,
            match_length_blend,
            host_tier_weight,
            disk_tier_weight,
            active_request_idle_threshold,
            active_request_max_busy_score,
            saturation_filter,
            saturation_queue_threshold,
            saturation_kv_threshold,
            saturation_headroom,
            no_hit_lru_capacity,
        );
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("queue_weight", self.queue_weight),
            ("kv_utilization_weight", self.kv_utilization_weight),
            ("prefix_weight", self.prefix_weight),
            ("no_hit_lru_weight", self.no_hit_lru_weight),
            ("active_request_weight", self.active_request_weight),
            ("token_load_weight", self.token_load_weight),
            ("max_ttft_penalty_ms", self.max_ttft_penalty_ms),
            ("saturation_headroom", self.saturation_headroom),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be a finite non-negative number"));
            }
        }
        for (name, value) in [
            ("affinity_threshold", self.affinity_threshold),
            ("match_length_blend", self.match_length_blend),
            ("host_tier_weight", self.host_tier_weight),
            ("disk_tier_weight", self.disk_tier_weight),
            (
                "active_request_max_busy_score",
                self.active_request_max_busy_score,
            ),
            ("saturation_kv_threshold", self.saturation_kv_threshold),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be between 0.0 and 1.0"));
            }
        }
        for (name, value) in [
            (
                "peak_prefill_tokens_per_second",
                self.peak_prefill_tokens_per_second,
            ),
            ("queue_threshold_tokens", self.queue_threshold_tokens),
            (
                "saturation_queue_threshold",
                self.saturation_queue_threshold,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be a finite positive number"));
            }
        }
        if self.no_hit_lru_capacity == 0 {
            return Err("no_hit_lru_capacity must be positive".into());
        }
        Ok(())
    }

    /// Matched blocks weighted by tier and truncated, as the precise producer reports them.
    fn matched_blocks(&self, candidate: &CandidateInputs<'_>) -> f64 {
        (candidate.device_blocks.max(0.0)
            + self.host_tier_weight * candidate.host_blocks.max(0.0)
            + self.disk_tier_weight * candidate.disk_blocks.max(0.0))
        .floor()
    }

    fn uncached_tokens(&self, request: &RequestInputs<'_>, candidate: &CandidateInputs<'_>) -> f64 {
        let cached = self.matched_blocks(candidate) * request.block_size as f64;
        (request.prompt_tokens as f64 - cached).max(0.0)
    }

    /// The EPP's `TTFT_e`, with this port's addition of the request's own uncached tokens.
    fn ttft_ms(&self, request: &RequestInputs<'_>, candidate: &CandidateInputs<'_>) -> f64 {
        (candidate.active_prefill_or_zero() as f64 + self.uncached_tokens(request, candidate))
            / self.peak_prefill_tokens_per_second
            * 1_000.0
    }

    fn is_saturated(&self, candidate: &CandidateInputs<'_>) -> bool {
        let headroom = 1.0 + self.saturation_headroom;
        if candidate
            .kv_usage
            .is_some_and(|kv| kv >= self.saturation_kv_threshold * headroom)
        {
            return true;
        }
        let Some(waiting) = candidate.queue_depth else {
            return false;
        };
        let running = candidate.running_requests.unwrap_or(0);
        let in_flight = candidate.active_requests as u64;
        let effective_queue = waiting + in_flight.saturating_sub(waiting + running);
        effective_queue as f64 >= self.saturation_queue_threshold * headroom
    }
}

fn clamp01(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

/// Endpoints in least-recently-used order for cold (no-hit) requests.
#[derive(Debug)]
struct NoHitLru {
    capacity: usize,
    /// Least recently used first.
    order: Mutex<VecDeque<Arc<str>>>,
}

impl NoHitLru {
    fn scores(&self, candidates: &[CandidateInputs<'_>]) -> Vec<f64> {
        let n = candidates.len();
        if n <= 1 {
            return vec![1.0; n];
        }
        let order = self.order.lock();
        let position = |url: &str| order.iter().position(|used| &**used == url);
        let never_used = candidates
            .iter()
            .filter(|c| position(c.url).is_none())
            .count();
        let denominator = (n - 1) as f64;
        let mut next_never_used = 0usize;
        candidates
            .iter()
            .map(|c| match position(c.url) {
                None => {
                    let rank = next_never_used;
                    next_never_used += 1;
                    1.0 - rank as f64 / denominator
                }
                Some(lru_position) => 1.0 - (never_used + lru_position) as f64 / denominator,
            })
            .collect()
    }

    fn touch(&self, url: &str) {
        let mut order = self.order.lock();
        if let Some(position) = order.iter().position(|used| &**used == url) {
            if let Some(used) = order.remove(position) {
                order.push_back(used);
            }
        } else {
            if order.len() == self.capacity {
                order.pop_front();
            }
            order.push_back(Arc::from(url));
        }
    }

    fn forget(&self, url: &str) {
        self.order.lock().retain(|used| &**used != url);
    }
}

/// The clamped weighted sum, negated into a cost.
#[derive(Debug)]
struct WeightedSum {
    params: LlmDParams,
    no_hit: Arc<NoHitLru>,
}

impl WorkerScorer for WeightedSum {
    fn score(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        costs: &mut [f64],
    ) {
        let p = &self.params;
        let n = candidates.len();
        let queue_of = |c: &CandidateInputs<'_>| c.queue_depth.unwrap_or(0) as f64;
        let (q_min, q_max) = candidates
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), c| {
                (lo.min(queue_of(c)), hi.max(queue_of(c)))
            });
        let max_active = candidates
            .iter()
            .map(|c| c.active_requests)
            .max()
            .unwrap_or(0);
        let full_blocks = (request.prompt_tokens / request.block_size.max(1)) as f64;
        let any_hit = candidates.iter().any(|c| p.matched_blocks(c) >= 1.0);
        let no_hit_scores = if p.no_hit_lru_weight > 0.0 {
            if any_hit {
                vec![0.5; n]
            } else {
                self.no_hit.scores(candidates)
            }
        } else {
            Vec::new()
        };
        for (row, (candidate, cost)) in candidates.iter().zip(costs).enumerate() {
            let mut sum = 0.0;
            if p.queue_weight > 0.0 {
                let score = if q_max > q_min {
                    (q_max - queue_of(candidate)) / (q_max - q_min)
                } else {
                    1.0
                };
                sum += p.queue_weight * clamp01(score);
            }
            if p.kv_utilization_weight > 0.0 {
                sum += p.kv_utilization_weight * clamp01(1.0 - candidate.kv_usage.unwrap_or(0.0));
            }
            if p.prefix_weight > 0.0 && full_blocks > 0.0 {
                let matched = p.matched_blocks(candidate);
                let ratio = (matched / full_blocks).min(1.0);
                let length = (matched * request.block_size as f64 / 8_192.0).min(1.0);
                let score =
                    (1.0 - p.match_length_blend) * ratio + p.match_length_blend * length * length;
                sum += p.prefix_weight * clamp01(score);
            }
            if p.active_request_weight > 0.0 {
                let count = candidate.active_requests;
                let score = if count <= p.active_request_idle_threshold {
                    1.0
                } else if max_active > 0 {
                    (max_active - count) as f64 / max_active as f64
                        * p.active_request_max_busy_score
                } else {
                    1.0
                };
                sum += p.active_request_weight * clamp01(score);
            }
            if p.no_hit_lru_weight > 0.0 {
                sum += p.no_hit_lru_weight * clamp01(no_hit_scores[row]);
            }
            if p.token_load_weight > 0.0 {
                let tokens = candidate.active_prefill_or_zero() as f64
                    + p.uncached_tokens(request, candidate);
                sum += p.token_load_weight
                    * clamp01(1.0 - (tokens / p.queue_threshold_tokens).min(1.0));
            }
            *cost += -sum;
        }
    }
}

/// Lowest cost among the eligible rows, rotating through exact ties. Eligibility is the
/// saturation gate and the sticky filter, both fail-open.
#[derive(Debug)]
struct LlmDPicker {
    params: LlmDParams,
    rotation: AtomicUsize,
    no_hit: Arc<NoHitLru>,
}

impl LlmDPicker {
    fn eligible_rows(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
    ) -> Vec<usize> {
        let p = &self.params;
        let mut rows: Vec<usize> = (0..candidates.len()).collect();
        if p.saturation_filter {
            let open: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&row| !p.is_saturated(&candidates[row]))
                .collect();
            if !open.is_empty() {
                rows = open;
            }
        }
        if p.sticky_filter && p.affinity_threshold > 0.0 {
            let full_blocks = (request.prompt_tokens / request.block_size.max(1)) as f64;
            let sticky = |row: usize| {
                full_blocks > 0.0
                    && p.matched_blocks(&candidates[row]) / full_blocks >= p.affinity_threshold
            };
            let best_ttft = |want_sticky: bool| {
                rows.iter()
                    .copied()
                    .filter(|&row| sticky(row) == want_sticky)
                    .map(|row| p.ttft_ms(request, &candidates[row]))
                    .min_by(f64::total_cmp)
            };
            let keep_sticky = match (best_ttft(true), best_ttft(false)) {
                (None, _) => false,
                (Some(_), None) => true,
                (Some(sticky_ttft), Some(cold_ttft)) => {
                    sticky_ttft - cold_ttft <= p.max_ttft_penalty_ms
                }
            };
            if keep_sticky {
                rows.retain(|&row| sticky(row));
            }
        }
        rows
    }
}

impl WorkerPicker for LlmDPicker {
    fn pick(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        costs: &[f64],
    ) -> Pick {
        let rows = self.eligible_rows(request, candidates);
        let Some(best) = rows
            .iter()
            .map(|&row| costs[row])
            .filter(|cost| !cost.is_nan())
            .min_by(f64::total_cmp)
        else {
            return Pick::None;
        };
        let mut tied: Vec<usize> = rows.into_iter().filter(|&row| costs[row] == best).collect();
        tied.sort_by(|&a, &b| candidates[a].url.cmp(candidates[b].url));
        let rotation = self.rotation.fetch_add(1, Ordering::Relaxed);
        Pick::Final(tied[rotation % tied.len()])
    }

    fn on_dispatch(&self, _request: &RequestInputs<'_>, candidate: &CandidateInputs<'_>) {
        // A cold dispatch: the picked endpoint held none of the prompt.
        if self.params.no_hit_lru_weight > 0.0 && self.params.matched_blocks(candidate) < 1.0 {
            self.no_hit.touch(candidate.url);
        }
    }

    fn on_worker_removed(&self, url: &str) {
        self.no_hit.forget(url);
    }
}

/// Only for the sticky profile without scorers: nothing to drop before scoring.
#[derive(Debug)]
struct KeepAll;

impl WorkerFilter for KeepAll {
    fn keep(&self, _request: &RequestInputs<'_>, _candidate: &CandidateInputs<'_>) -> bool {
        true
    }
}

pub(super) fn policy(name: &'static str, params: LlmDParams) -> WorkerSelectionPolicy {
    let no_hit = Arc::new(NoHitLru {
        capacity: params.no_hit_lru_capacity,
        order: Mutex::new(VecDeque::new()),
    });
    let filters: Vec<Box<dyn WorkerFilter>> = if params.sticky_filter {
        vec![Box::new(KeepAll)]
    } else {
        Vec::new()
    };
    WorkerSelectionPolicy::new(
        name,
        Needs::FLEET_WITH_LOADS,
        filters,
        vec![Box::new(WeightedSum {
            params,
            no_hit: Arc::clone(&no_hit),
        })],
        Box::new(LlmDPicker {
            params,
            rotation: AtomicUsize::new(0),
            no_hit,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(idx: usize, url: &'static str, device_blocks: f64) -> CandidateInputs<'static> {
        CandidateInputs {
            idx,
            url,
            device_blocks,
            host_blocks: 0.0,
            disk_blocks: 0.0,
            effective_score: device_blocks,
            active_requests: 0,
            active_prefill_tokens: Some(0),
            decode_blocks: None,
            kv_usage: Some(0.0),
            queue_depth: Some(0),
            running_requests: Some(0),
            taint: 1.0,
        }
    }

    fn request(prompt_tokens: usize) -> RequestInputs<'static> {
        RequestInputs {
            prompt_tokens,
            block_size: 16,
            request_blocks: (prompt_tokens / 16).max(1),
            avg_load: 0.0,
            prefix_hashes: None,
        }
    }

    #[test]
    fn presets_apply_patches_and_reject_unknown_fields() {
        let patch: LlmDParamsPatch =
            serde_yaml::from_str("{prefix_weight: 5, sticky_filter: true}").unwrap();
        let params = LlmDParams::optimized_baseline().apply(&patch);
        assert_eq!(params.prefix_weight, 5.0);
        assert!(params.sticky_filter);
        assert_eq!(params.queue_weight, 2.0);
        assert!(serde_yaml::from_str::<LlmDParamsPatch>("{prefix: 5}").is_err());
        assert!(LlmDParams::optimized_baseline().validate().is_ok());
        assert!(LlmDParams::precise_prefix().validate().is_ok());
        assert!(LlmDParams::sticky_until_saturated().validate().is_ok());
    }

    #[test]
    fn optimized_baseline_prefers_the_holder_then_spreads_cold_prefixes() {
        let policy = policy(OPTIMIZED_BASELINE, LlmDParams::optimized_baseline());
        let req = request(1_024);
        let cands = [
            candidate(0, "http://w1", 0.0),
            candidate(1, "http://w2", 64.0),
            candidate(2, "http://w3", 0.0),
        ];
        assert_eq!(policy.select(&req, &cands), Pick::Final(1));

        // No holder anywhere: the never-used endpoints are preferred in order, and each cold
        // dispatch moves an endpoint to the back of the LRU.
        let cold = [
            candidate(0, "http://w1", 0.0),
            candidate(1, "http://w2", 0.0),
            candidate(2, "http://w3", 0.0),
        ];
        let first = policy.select(&req, &cold);
        let Pick::Final(first_row) = first else {
            panic!("expected a pick")
        };
        policy.on_dispatch(&req, &cold[first_row]);
        let second = policy.select(&req, &cold);
        assert_ne!(
            first, second,
            "a cold dispatch must move the endpoint back in the LRU"
        );
    }

    #[test]
    fn saturation_gate_is_fail_open() {
        let policy = policy(OPTIMIZED_BASELINE, LlmDParams::optimized_baseline());
        let req = request(1_024);
        let mut saturated_holder = candidate(0, "http://w1", 64.0);
        saturated_holder.queue_depth = Some(7);
        let cands = [saturated_holder.clone(), candidate(1, "http://w2", 0.0)];
        assert_eq!(policy.select(&req, &cands), Pick::Final(1));
        let mut other = candidate(1, "http://w2", 0.0);
        other.kv_usage = Some(0.95);
        let both = [saturated_holder, other];
        // Both saturated: the gate opens and the holder wins on prefix.
        assert_eq!(policy.select(&req, &both), Pick::Final(0));
    }

    #[test]
    fn sticky_until_saturated_breaks_stickiness_past_the_penalty() {
        let params = LlmDParams::sticky_until_saturated().apply(&LlmDParamsPatch {
            peak_prefill_tokens_per_second: Some(1_000.0),
            max_ttft_penalty_ms: Some(1_000.0),
            ..Default::default()
        });
        let policy = policy(STICKY_UNTIL_SATURATED, params);
        let req = request(1_600);
        let mut holder = candidate(0, "http://w1", 100.0);
        let cold = candidate(1, "http://w2", 0.0);
        // Holder backlog 1,000 tokens → 1 s; cold must prefill 1,600 tokens → 1.6 s: stay sticky.
        holder.active_prefill_tokens = Some(1_000);
        assert_eq!(
            policy.select(&req, &[holder.clone(), cold.clone()]),
            Pick::Final(0)
        );
        // Holder backlog 3,000 → 3 s against 1.6 s: penalty exceeded, the lighter token load wins.
        holder.active_prefill_tokens = Some(3_000);
        assert_eq!(policy.select(&req, &[holder, cold]), Pick::Final(1));
    }
}
