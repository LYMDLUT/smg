//! DualMap (arXiv 2602.06502) as an SMG policy, after Dynamo's port in PR ai-dynamo/dynamo#15450.
//!
//! Each request's prefix is keyed at the block depths `hash_prefix_blocks · 2^k`; the shallowest
//! key that is not hot (seen by more than `2 / workers` of the last `window_requests` requests)
//! is hashed with rendezvous hashing onto two candidate workers. The warmer of the two (more
//! device overlap) takes the request unless its prefill backlog after this request would exceed
//! `pending_prefill_token_budget`; then the lesser backlog of the pair wins. Without prefix
//! hashes, or with one worker, the least backlog over the fleet wins.
//!
//! Departures: worker identity for rendezvous is the XXH3 of the worker URL (Dynamo uses its
//! numeric worker and DP-rank ids); backlog ties go to the smallest URL.

use std::collections::{HashMap, VecDeque};

use parking_lot::Mutex;

use super::{
    inputs::{CandidateInputs, RequestInputs},
    policy::{Needs, Pick, WorkerPicker, WorkerSelectionPolicy},
};

pub const POLICY_NAME: &str = "dualmap";

/// Prefix depths tracked per request: `hash_prefix_blocks` doubled up to seven times.
const MAX_LEVELS: usize = 8;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DualMapParams {
    pub hash_prefix_blocks: usize,
    pub pending_prefill_token_budget: usize,
    pub window_requests: usize,
}

impl Default for DualMapParams {
    fn default() -> Self {
        Self {
            hash_prefix_blocks: 4,
            pending_prefill_token_budget: 65_536,
            window_requests: 1_000,
        }
    }
}

impl DualMapParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.hash_prefix_blocks == 0
            || self.pending_prefill_token_budget == 0
            || self.window_requests == 0
        {
            return Err(
                "hash_prefix_blocks, pending_prefill_token_budget, and window_requests must be positive"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct Window {
    /// Prefix keys at every tracked depth for each request in the window.
    requests: VecDeque<([u64; MAX_LEVELS], usize)>,
    arrivals: HashMap<u64, usize>,
}

#[derive(Debug)]
struct DualMapPicker {
    params: DualMapParams,
    window: Mutex<Window>,
}

fn mix(mut value: u64) -> u64 {
    // SplitMix64 finalizer.
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Rendezvous (highest-random-weight) hash of `key` onto a worker.
fn rendezvous(key: u64, worker: u64, seed: u64) -> u64 {
    mix(key ^ mix(worker ^ mix(seed)))
}

fn worker_identity(url: &str) -> u64 {
    xxhash_rust::xxh3::xxh3_64(url.as_bytes())
}

impl DualMapPicker {
    /// Record this request's prefix keys, then return the shallowest key that is not hot.
    fn hash_key(&self, hashes: &[u64], workers: usize) -> Option<u64> {
        let mut keys = [0u64; MAX_LEVELS];
        let mut levels = 0;
        let mut depth = self.params.hash_prefix_blocks;
        while levels < MAX_LEVELS && depth <= hashes.len() {
            keys[levels] = hashes[depth - 1];
            levels += 1;
            depth *= 2;
        }
        if levels == 0 {
            return None;
        }
        let mut window = self.window.lock();
        if window.requests.len() == self.params.window_requests {
            if let Some((expired, expired_levels)) = window.requests.pop_front() {
                for key in &expired[..expired_levels] {
                    if let Some(count) = window.arrivals.get_mut(key) {
                        *count -= 1;
                        if *count == 0 {
                            window.arrivals.remove(key);
                        }
                    }
                }
            }
        }
        for key in &keys[..levels] {
            *window.arrivals.entry(*key).or_default() += 1;
        }
        window.requests.push_back((keys, levels));
        let hot_share = 2.0 / workers as f64;
        let window_len = window.requests.len() as f64;
        keys[..levels]
            .iter()
            .copied()
            .find(|key| {
                window.arrivals.get(key).copied().unwrap_or(0) as f64 / window_len <= hot_share
            })
            .or(Some(keys[levels - 1]))
    }
}

impl WorkerPicker for DualMapPicker {
    fn pick(
        &self,
        request: &RequestInputs<'_>,
        candidates: &[CandidateInputs<'_>],
        _costs: &[f64],
    ) -> Pick {
        if candidates.is_empty() {
            return Pick::None;
        }
        // Prefill tokens a candidate would carry after taking this request.
        let backlog = |row: usize| {
            candidates[row].active_prefill_or_zero() as usize
                + candidates[row].uncached_prompt_tokens(request)
        };
        let least_backlog = |rows: &mut dyn Iterator<Item = usize>| {
            rows.min_by(|&a, &b| {
                backlog(a)
                    .cmp(&backlog(b))
                    .then_with(|| candidates[a].url.cmp(candidates[b].url))
            })
            .map_or(Pick::None, Pick::Final)
        };
        let key = match request.prefix_hashes {
            Some(hashes) if candidates.len() > 1 => self.hash_key(hashes, candidates.len()),
            _ => None,
        };
        let Some(key) = key else {
            return least_backlog(&mut (0..candidates.len()));
        };
        let identities: Vec<u64> = candidates.iter().map(|c| worker_identity(c.url)).collect();
        let draw = |seed: u64, skip: Option<usize>| {
            (0..candidates.len())
                .filter(|&row| Some(row) != skip)
                .max_by_key(|&row| rendezvous(key, identities[row], seed))
        };
        let first = draw(1, None);
        let (Some(first), Some(second)) = (first, first.and_then(|first| draw(2, Some(first))))
        else {
            return least_backlog(&mut (0..candidates.len()));
        };
        let cached = |row: usize| candidates[row].device_blocks;
        let (warm, cold) = match cached(first).total_cmp(&cached(second)) {
            std::cmp::Ordering::Equal => return least_backlog(&mut [first, second].into_iter()),
            std::cmp::Ordering::Greater => (first, second),
            std::cmp::Ordering::Less => (second, first),
        };
        if backlog(warm) > self.params.pending_prefill_token_budget {
            return least_backlog(&mut [warm, cold].into_iter());
        }
        Pick::Final(warm)
    }
}

pub(super) fn policy(params: DualMapParams) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        POLICY_NAME,
        Needs {
            prefix_hashes: true,
            ..Needs::FLEET_WITH_LOADS
        },
        Vec::new(),
        Vec::new(),
        Box::new(DualMapPicker {
            params,
            window: Mutex::new(Window::default()),
        }),
    )
}
