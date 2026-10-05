//! Dynamo's default cost function (`dynamo-default-cost-fn`) as an SMG policy.
//!
//! From `lib/router-plugins/builtin/src/default/{scorer,picker}.rs` at Dynamo `main`:
//!
//! ```text
//! credit  = c · decay · device + 0.75 · host + 0.25 · disk
//! prefill = max(0, (active_prefill_tokens + prompt_tokens) / block_size − credit)
//! cost    = prefill_load_scale · prefill + decode_blocks + w_req · active_requests, × taint
//! decay   = 1 / (1 + k · ((active_prefill − min_active_prefill) / block_size) / request_blocks)
//! ```
//!
//! Departures from the source, all forced by what SMG's host can supply today:
//! - `decode_blocks` is the host's estimate of the blocks held by the worker's in-flight requests
//!   (each taken to hold this request's blocks, plus credited output blocks); Dynamo tracks every
//!   active sequence's blocks itself. The backend's KV usage is deliberately not used: it counts
//!   reusable cached blocks and lags by a poll interval, which routes by cache fill, not load.
//! - no shared-cache credit term (SMG indexes no shared pool yet).
//! - ties at temperature zero go to the smallest worker URL unless `tie_break: uniform`; Dynamo
//!   draws uniformly in production and deterministically only in replay.

use super::{
    inputs::{CandidateInputs, RequestInputs},
    policy::{Needs, Pick, WorkerPicker, WorkerScorer, WorkerSelectionPolicy},
    softmax::{pick_lowest, TieBreak},
};

pub const POLICY_NAME: &str = "dynamo-default";

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DynamoDefaultParams {
    pub overlap_score_credit: f64,
    pub overlap_score_credit_decay: f64,
    pub prefill_load_scale: f64,
    pub decode_active_request_weight: f64,
    pub host_cache_hit_weight: f64,
    pub disk_cache_hit_weight: f64,
    pub router_temperature: f64,
    pub tie_break: TieBreak,
}

impl Default for DynamoDefaultParams {
    fn default() -> Self {
        Self {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            decode_active_request_weight: 0.0,
            host_cache_hit_weight: 0.75,
            disk_cache_hit_weight: 0.25,
            router_temperature: 0.0,
            tie_break: TieBreak::Deterministic,
        }
    }
}

impl DynamoDefaultParams {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("overlap_score_credit", self.overlap_score_credit),
            (
                "overlap_score_credit_decay",
                self.overlap_score_credit_decay,
            ),
            ("prefill_load_scale", self.prefill_load_scale),
            (
                "decode_active_request_weight",
                self.decode_active_request_weight,
            ),
            ("host_cache_hit_weight", self.host_cache_hit_weight),
            ("disk_cache_hit_weight", self.disk_cache_hit_weight),
            ("router_temperature", self.router_temperature),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be a finite non-negative number"));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct DynamoDefaultScorer {
    params: DynamoDefaultParams,
}

impl WorkerScorer for DynamoDefaultScorer {
    fn score(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        costs: &mut [f64],
    ) {
        let p = &self.params;
        let block = request.block_size.max(1) as f64;
        let request_blocks = request.request_blocks.max(1) as f64;
        let needs_decay = p.overlap_score_credit_decay > 0.0;
        let min_prefill = if needs_decay {
            candidates
                .iter()
                .map(CandidateInputs::active_prefill_or_zero)
                .min()
                .unwrap_or(0)
        } else {
            0
        };
        for (candidate, cost) in candidates.iter().zip(costs) {
            let decay = if needs_decay {
                let excess_blocks = candidate
                    .active_prefill_or_zero()
                    .saturating_sub(min_prefill) as f64
                    / block;
                1.0 / (1.0 + p.overlap_score_credit_decay * excess_blocks / request_blocks)
            } else {
                1.0
            };
            let credit = p.overlap_score_credit * decay * candidate.device_blocks
                + p.host_cache_hit_weight * candidate.host_blocks
                + p.disk_cache_hit_weight * candidate.disk_blocks;
            let raw_tokens = match candidate.active_prefill_tokens {
                Some(active) => active as f64 + request.prompt_tokens as f64,
                None => request.prompt_tokens as f64,
            };
            let prefill = (raw_tokens / block - credit).max(0.0);
            let decode = candidate
                .decode_blocks
                .unwrap_or(candidate.active_requests as f64 * request_blocks);
            let logit = p.prefill_load_scale * prefill
                + decode
                + p.decode_active_request_weight * candidate.active_requests as f64;
            *cost += logit * candidate.taint;
        }
    }
}

/// Lowest cost, with the configured tie-break at temperature zero and a cost-softmax draw above.
#[derive(Debug)]
pub(super) struct LowestCostPicker {
    pub temperature: f64,
    pub tie_break: TieBreak,
}

impl WorkerPicker for LowestCostPicker {
    fn pick(
        &self,
        _request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        costs: &[f64],
    ) -> Pick {
        pick_lowest(candidates, costs, self.temperature, self.tie_break)
            .map_or(Pick::None, Pick::Final)
    }
}

pub(super) fn policy(params: DynamoDefaultParams) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        POLICY_NAME,
        Needs::FLEET_WITH_LOADS,
        Vec::new(),
        vec![Box::new(DynamoDefaultScorer { params })],
        Box::new(LowestCostPicker {
            temperature: params.router_temperature,
            tie_break: params.tie_break,
        }),
    )
}
