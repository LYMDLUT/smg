//! Ramjet (helixml/ramjet v0.7.0) as an SMG policy.
//!
//! Rule (`src/router.rs:379-404`, `:786-803`): maximise capped prefix affinity minus α times the
//! replica's live reserved load.
//!
//! - **Overlap** `ov` is the leading cached run in Ramjet blocks of 2,048 canonical prompt bytes
//!   (about 512 tokens). Ramjet reads it from its own fingerprint LRU, written on a complete
//!   response; this port reads it from SMG's KV index, so sibling requests see each other's blocks
//!   as soon as the engine reports them. `affinity_block_tokens` converts KV blocks to Ramjet
//!   blocks.
//! - **Affinity** on one of three bases over the serving peers: `absolute = min(ov, cap)`,
//!   `marginal = min(max(0, ov − floor), cap)`, `relative = max(0, min(lead, cap) − min(lead − ov,
//!   cap))`, with `floor`/`lead` the coldest/warmest overlap among candidates and `cap` 32 blocks.
//!   Relative is Ramjet's default and the basis that keeps sessions sticky at scale
//!   (`docs/multi-node.md`).
//! - **Load units**: each dispatched request reserves
//!   `clamp(ceil(max(prompt_tokens − ov × affinity_block_tokens, 0) / load_unit_tokens), 1,
//!   max_request_load_units)` units on its worker until it completes (one unit is 32 KiB of raw
//!   body, about 8k tokens). The request's own reservation is not in its score unless
//!   `projected_load` is set. Reservations are made on the host's final dispatch, not on the pick.
//! - **Order**: score descending, overlap descending, then a rotation through the remaining ties
//!   so a hot prefix spreads instead of pinning.
//!
//! Not expressed: the decode floor from a max-tokens bucket (the host has no max-tokens input yet,
//! so `floor` is one unit), the phase-aware cut of the reservation at the first token (no
//! first-token hook yet), the long-prompt lane, session shadowing and prefix single-flight. A
//! reservation older than `reservation_ttl_secs` is dropped as lost.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{
    inputs::{CandidateInputs, RequestInputs},
    policy::{Needs, Pick, WorkerPicker, WorkerSelectionPolicy},
};

pub const POLICY_NAME: &str = "ramjet";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AffinityBasis {
    Absolute,
    Marginal,
    Relative,
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RamjetParams {
    /// Load-unit weight against affinity blocks (`RJ_ROUTE_LOAD_ALPHA`).
    pub alpha: f64,
    /// Tokens per Ramjet affinity block (2,048 canonical bytes).
    pub affinity_block_tokens: usize,
    /// Affinity cap in blocks (`RJ_ROUTE_MAX_OVERLAP_BLOCKS`).
    pub max_affinity_blocks: usize,
    /// Tokens per load unit (32 KiB of raw body).
    pub load_unit_tokens: usize,
    /// Largest reservation one request can make, in load units.
    pub max_request_load_units: usize,
    pub basis: AffinityBasis,
    /// Count the request's own reservation (minus one) in its score (`RJ_ROUTE_PROJECTED_LOAD`).
    pub projected_load: bool,
    /// Age after which a reservation without a completion is dropped.
    pub reservation_ttl_secs: u64,
}

impl Default for RamjetParams {
    fn default() -> Self {
        Self {
            alpha: 4.0,
            affinity_block_tokens: 512,
            max_affinity_blocks: 32,
            load_unit_tokens: 8_192,
            max_request_load_units: 8,
            basis: AffinityBasis::Relative,
            projected_load: false,
            reservation_ttl_secs: 600,
        }
    }
}

impl RamjetParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err("alpha must be a finite non-negative number".into());
        }
        if self.affinity_block_tokens == 0
            || self.max_affinity_blocks == 0
            || self.load_unit_tokens == 0
            || self.max_request_load_units == 0
            || self.reservation_ttl_secs == 0
        {
            return Err(
                "affinity_block_tokens, max_affinity_blocks, load_unit_tokens, \
                        max_request_load_units, and reservation_ttl_secs must be positive"
                    .into(),
            );
        }
        Ok(())
    }

    fn credited_affinity(&self, overlap: usize, floor: usize, lead: usize) -> usize {
        let cap = self.max_affinity_blocks;
        match self.basis {
            AffinityBasis::Absolute => overlap.min(cap),
            AffinityBasis::Marginal => overlap.saturating_sub(floor).min(cap),
            AffinityBasis::Relative => lead
                .min(cap)
                .saturating_sub(lead.saturating_sub(overlap).min(cap)),
        }
    }

    /// Overlap in Ramjet blocks from device blocks of `block_size` tokens.
    fn overlap_blocks(&self, candidate: &CandidateInputs<'_>, block_size: usize) -> usize {
        let tokens = candidate.device_blocks.max(0.0) * block_size as f64;
        (tokens / self.affinity_block_tokens as f64) as usize
    }

    /// Load units a request reserves on a worker holding `overlap` blocks of it.
    fn request_load_units(&self, prompt_tokens: usize, overlap: usize) -> usize {
        let uncached = prompt_tokens.saturating_sub(overlap * self.affinity_block_tokens);
        uncached
            .div_ceil(self.load_unit_tokens)
            .clamp(1, self.max_request_load_units)
    }
}

#[derive(Debug, Default)]
struct Reservations {
    /// Per worker: in-flight reservations, oldest first.
    inflight: HashMap<Arc<str>, VecDeque<(usize, Instant)>>,
}

#[derive(Debug)]
struct RamjetPicker {
    params: RamjetParams,
    rotation: AtomicUsize,
    reservations: Mutex<Reservations>,
}

impl RamjetPicker {
    fn reserved_load(&self, candidates: &[CandidateInputs<'_>]) -> Vec<usize> {
        let ttl = Duration::from_secs(self.params.reservation_ttl_secs);
        let now = Instant::now();
        let mut state = self.reservations.lock();
        candidates
            .iter()
            .map(|candidate| {
                let Some(queue) = state.inflight.get_mut(candidate.url) else {
                    return 0;
                };
                while queue
                    .front()
                    .is_some_and(|(_, since)| now.duration_since(*since) > ttl)
                {
                    queue.pop_front();
                }
                queue.iter().map(|(units, _)| *units).sum()
            })
            .collect()
    }
}

impl WorkerPicker for RamjetPicker {
    fn pick(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        _costs: &[f64],
    ) -> Pick {
        if candidates.is_empty() {
            return Pick::None;
        }
        let p = &self.params;
        let overlaps: Vec<usize> = candidates
            .iter()
            .map(|c| p.overlap_blocks(c, request.block_size))
            .collect();
        let floor = overlaps.iter().copied().min().unwrap_or(0);
        let lead = overlaps.iter().copied().max().unwrap_or(0);
        let load = self.reserved_load(candidates);
        let score = |row: usize| {
            let projected = if p.projected_load {
                p.request_load_units(request.prompt_tokens, overlaps[row]) - 1
            } else {
                0
            };
            let affinity = p.credited_affinity(overlaps[row], floor, lead);
            (
                affinity as f64 - p.alpha * (load[row] + projected) as f64,
                overlaps[row],
            )
        };
        let Some(best) = (0..candidates.len())
            .map(score)
            .max_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
        else {
            return Pick::None;
        };
        let mut tied: Vec<usize> = (0..candidates.len())
            .filter(|&row| score(row) == best)
            .collect();
        tied.sort_by(|&a, &b| candidates[a].url.cmp(candidates[b].url));
        let rotation = self.rotation.fetch_add(1, Ordering::Relaxed);
        Pick::Final(tied[rotation % tied.len()])
    }

    fn on_dispatch(&self, request: &RequestInputs<'_>, candidate: &CandidateInputs<'_>) {
        let overlap = self.params.overlap_blocks(candidate, request.block_size);
        let units = self
            .params
            .request_load_units(request.prompt_tokens, overlap);
        let mut state = self.reservations.lock();
        let key = state
            .inflight
            .get_key_value(candidate.url)
            .map(|(key, _)| Arc::clone(key))
            .unwrap_or_else(|| Arc::from(candidate.url));
        state
            .inflight
            .entry(key)
            .or_default()
            .push_back((units, Instant::now()));
    }

    fn on_request_complete(&self, url: &str) {
        let mut state = self.reservations.lock();
        if let Some(queue) = state.inflight.get_mut(url) {
            queue.pop_front();
            if queue.is_empty() {
                state.inflight.remove(url);
            }
        }
    }

    fn on_worker_removed(&self, url: &str) {
        self.reservations.lock().inflight.remove(url);
    }
}

pub(super) fn policy(params: RamjetParams) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        POLICY_NAME,
        Needs::FLEET_WITH_LOADS,
        Vec::new(),
        Vec::new(),
        Box::new(RamjetPicker {
            params,
            rotation: AtomicUsize::new(0),
            reservations: Mutex::new(Reservations::default()),
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
            active_prefill_tokens: None,
            decode_blocks: None,
            kv_usage: None,
            queue_depth: None,
            running_requests: None,
            taint: 1.0,
        }
    }

    fn request(prompt_tokens: usize) -> RequestInputs<'static> {
        RequestInputs {
            prompt_tokens,
            block_size: 16,
            request_blocks: prompt_tokens / 16,
            avg_load: 0.0,
            prefix_hashes: None,
        }
    }

    #[test]
    fn affinity_bases_match_ramjet() {
        let p = RamjetParams::default();
        // ov = 40 blocks, floor 4, lead 40, cap 32.
        assert_eq!(
            RamjetParams {
                basis: AffinityBasis::Absolute,
                ..p
            }
            .credited_affinity(40, 4, 40),
            32
        );
        assert_eq!(
            RamjetParams {
                basis: AffinityBasis::Marginal,
                ..p
            }
            .credited_affinity(40, 4, 40),
            32
        );
        assert_eq!(
            RamjetParams {
                basis: AffinityBasis::Marginal,
                ..p
            }
            .credited_affinity(10, 4, 40),
            6
        );
        // relative: min(40,32) - min(40-10, 32) = 32 - 30 = 2
        assert_eq!(p.credited_affinity(10, 4, 40), 2);
        assert_eq!(p.credited_affinity(40, 4, 40), 32);
        assert_eq!(p.credited_affinity(0, 0, 40), 0);
    }

    #[test]
    fn load_units_are_size_weighted_and_clamped() {
        let p = RamjetParams::default();
        assert_eq!(p.request_load_units(100, 0), 1);
        assert_eq!(p.request_load_units(8_192, 0), 1);
        assert_eq!(p.request_load_units(8_193, 0), 2);
        assert_eq!(p.request_load_units(1_000_000, 0), 8);
        // 20 blocks of 512 tokens cached out of 16,384 → 6,144 uncached → 1 unit.
        assert_eq!(p.request_load_units(16_384, 20), 1);
    }

    #[test]
    fn reservations_shift_the_pick_until_released() {
        let policy = policy(RamjetParams::default());
        // Both hold the whole 16k-token prompt (32 blocks); w1 sorts first on the rotation.
        let cands = [
            candidate(0, "http://w1", 1024.0),
            candidate(1, "http://w2", 1024.0),
        ];
        let req = request(16_384);
        assert_eq!(policy.select(&req, &cands), Pick::Final(0));
        // A tie rotates.
        assert_eq!(policy.select(&req, &cands), Pick::Final(1));
        // Nine units reserved on w1 (> 32 / 4 = 8 units of affinity) send even a cold request away.
        for _ in 0..9 {
            policy.on_dispatch(&req, &cands[0]);
        }
        let cold = [
            candidate(0, "http://w1", 1024.0),
            candidate(1, "http://w2", 0.0),
        ];
        assert_eq!(policy.select(&req, &cold), Pick::Final(1));
        for _ in 0..9 {
            policy.on_request_complete("http://w1");
        }
        assert_eq!(policy.select(&req, &cold), Pick::Final(0));
    }
}
