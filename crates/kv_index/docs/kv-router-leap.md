# KV router leap: goals, guardrails and method

Working document for the `perf/kv-router-leap` effort. It is the contract for the work: the goals, the
gates a change must pass, how everything is measured, and the rule that nothing is published until all of
it holds. Status, scoreboard rows and decisions are appended at the bottom as the loop runs.

## 1. Why

NVIDIA's Dynamo team published a year-over-year KV indexer benchmark (branch `rupei/indexer-yoy-data` in
`ai-dynamo/dynamo`, data for the post of October 2026) that measures SMG's `PositionalIndexer` from
`crates/kv_index` at commit `0f9f219` directly against their indexer, on an AMD EPYC 9654P (96 cores, one
socket; 8 cores issue events, 1 issues lookups, the indexer runs on 87), replaying the public Mooncake trace
(128 inference workers, trace duplication 20, trace length factor 4, 128-token blocks, 472,160 requests and
1,974,035 KV events, 320M block operations per replay, open loop, three fresh-process repetitions per window):

| Indexer (their build) | Highest sustained block ops/s (every rep kept up) | Overloaded achieved | Lookup p50 / p99 at 51.8M offered |
|---|---|---|---|
| Dynamo CRTC, top of the four-PR stack (#15606–#15611), mimalloc, 64 event workers | 1.68 B | 1.78 B | 2.2 µs / 8.1 µs |
| Dynamo CRTC as shipped on `main` (glibc), 16 event workers | 49.9 M | ~62 M | 2.5 µs / 9.8 µs |
| SMG `PositionalIndexer` @ 0f9f219, opt-level 3, mimalloc, 64 event workers | 203 M | 224–238 M | 7.4 µs / 30 µs |
| llm-d precise prefix index v0.11.0 | 0.66 M | 0.88 M | cannot keep up |

Their score check: SMG returned the same overlap scores as CRTC for every lookup on a 1,000-request fixture.
So the gap is throughput and latency, not correctness. The gap is 8× on sustained throughput and 3–4× on
lookup p99, and their shipped `main` is 4× slower than SMG; what beat SMG is unmerged work. That is the
target, and it moves.

Block op definition (their `lib/bench/kv_router/INDEXER_BENCH.md`): a requested block hash, a stored block
hash, or a removed block hash; `Cleared` counts one logical op and zero blocks. Achieved rate = block ops /
(last completion − start), drain included. A trial is valid only if everything was issued within 1.01× the
window; it "kept up" if replay plus drain finished within 1.10×. We adopt this definition unchanged.

Hashing is compatible: SMG's `compute_content_hash` and `chain_prefix_hash` are byte-identical to Dynamo's
per-block and chain hashes for the base model with no LoRA, salt or multimodal input (XXH3-64, seed 1337,
little-endian token ids), so corpora and fixtures can be shared.

## 2. Goals

All numbers on one host, both systems built and run the same day, Dynamo's benchmark definition and trace.
The reference competitor is the top of their CRTC stack as of the day of measurement, not their `main`.

| Track | Goal | Minimum to publish |
|---|---|---|
| T1 Indexer throughput | ≥ 10× the competitor's highest sustained block ops/s, at equal backend cores, with the harness's issuers scaled until the indexer, not the generator, is the bottleneck | ≥ 3× |
| T2 Indexer latency | lookup p99 ≤ competitor's at every offered load both sustain; lookup p50 ≤ 2 µs | p99 ≤ competitor's at 51.8M offered |
| T3 Index memory | ≤ 1/5 of the competitor's bytes per indexed block at 128 workers; RSS flat over a 24 h replay | ≤ 1/2 |
| T4 Routing accuracy | achieved hit rate ≥ 0.98 of the oracle; predicted vs actual cached tokens exact at p99 on the kept-up path; convergence after a gap or reconnect ≤ 500 ms | ≥ 0.95, ≤ 1 block, ≤ 1 s |
| T5 End-to-end latency (mock engine replay) | mean TTFT ≥ 40% lower and goodput at SLO ≥ 25% higher than Dynamo's default cost function run as an SMG policy at 3× Mooncake arrival; p99 TTFT not worse; prefix reuse ≥ 0.52 | 25% / 15% |
| T6 Routing decision cost | ≤ 10 µs p99 per request at 10k req/s per core, including the lookup | ≤ 20 µs |
| T7 Scaling | linear sustained throughput from 8 to 128 workers and from 1 to 64 event lanes; no cliff across the two NUMA sockets of this host | within 20% of linear |
| T8 Fault tolerance | every drill in §5 passes; index after recovery identical to a fresh snapshot | all pass |
| T9 Hardware agreement | simulation and GB300 runs agree on direction and within 15% on magnitude for T4 and T5 | within 25% |

## 3. Guardrails (hard, every change, no exceptions)

A change that violates any line below is reverted, not argued about.

1. **Exactness.** The index is a function of the event stream. After any replay, the set of (worker,
   position, block) in the index equals the reference built from the same events by the single-threaded
   reference indexer; zero phantom blocks, zero missing blocks. Every lookup score equals the reference
   score. Approximations that trade correctness for speed (equal-size skips, single-entry shortcuts that
   skip the prefix-hash check) are forbidden. The current jump search over-counts on divergent prompts
   (`event_tree.rs` `workers_if_single` / `count_workers_at`: worker `[A,B,C]`, query `[A,X,C]` scores 3);
   that is a bug to fix, not a baseline to keep.
2. **No reads that write.** The lookup path performs no stores to shared memory (no `touch`, no refcount
   bumps per hop, no lock acquisition that writes a cache line other workers read). Verified by `perf c2c`
   or an allocation-and-store counting harness on the lookup path, not by inspection.
3. **Bounded memory.** Every queue, buffer and cache has a bound and a metric; a 24 h chaos replay ends
   within 5% of its 1 h RSS. Unbounded `flume`-style lanes are not accepted.
4. **Latency before throughput.** A throughput gain that raises lookup p99 at any sustained load is
   rejected. Overloaded headline numbers are reported together with the sustained-load latency curve, never
   alone (Dynamo's own bench doc shows the overloaded number cannot detect read-path regressions).
5. **Measurement discipline.** Fresh process per trial; ≥ 20 trials for any published number, 3 for loop
   iterations; medians with bootstrap 95% CIs; same-binary control pairs before calling any difference under
   5%; cores pinned with issuers disjoint from the backend mask; `numactl --interleave=all`; trials that
   overlapped another job on the host are discarded and said so.
6. **No regression anywhere else.** The gateway harness matrix (HTTP, gRPC, PD, 64 and 512 tokens, 16/32
   thread runtimes) stays monotone; `cargo test --workspace`, clippy with `-D warnings`, nightly fmt and
   pre-commit stay green on every commit of the branch.
7. **Engine truth over router belief.** Accuracy is measured against the engines' own `cached_tokens` and
   reuse counters, never against the router's own prediction.
8. **One branch, no partial PRs.** All work lands on `perf/kv-router-leap` as reviewable commits. **No pull
   request is opened until every goal in §2 is met at the "goal" level (not the minimum)**, the drills in
   §5 pass, and T9 holds. The PR then publishes the complete benchmark: both systems' commits and build
   flags, the host, every CSV, the scripts, the scoreboard history, and what did not work. Partial results go
   into this document, not into PRs.

## 4. What has to be built

### 4.1 Wire and proto: follow the engines' KV-event schema

vLLM `main` (`vllm/distributed/kv_events.py`) and SGLang `main` (`sglang/srt/disaggregation/kv_events.py`)
now carry fields the SMG relay drops. `crates/engine_servicer/src/kv_events.rs` decodes only `block_hashes`,
`parent_block_hash`, `token_ids`, `block_size`, `lora_id`; `crates/grpc_client/proto/common.proto` cannot
carry the rest; `cache_level` is never set. Required:

- `KvBlocksStored`: `medium` (GPU / CPU / CPU_PINNED / STORAGE; map to a tier enum), `group_idx`,
  `kv_cache_spec_kind`, `kv_cache_spec_sliding_window`, `locality` (LOCAL / REMOTE), `ownership`,
  `session_id`, `lora_name`, `cache_salt` (SGLang) and per-block `extra_keys` (vLLM; LoRA name,
  multimodal `(identifier, offset)` pairs, cache salt, prompt-embedding digests).
- `KvBlocksRemoved`: `medium`, `group_idx`, `locality`, `ownership`. `KvCacheCleared`: `ownership`.
- Relay rules (what Dynamo's normalizer does, and what we must match or better): index only full-attention
  cache groups (full, MLA, sink-full); drop `locality = REMOTE` and unknown media; fold LoRA name and cache
  salt into the per-block hash seed and propagate the salt down the parent chain; drop events whose block
  count does not match `token_ids.len() / block_size`; drop self-referencing hash chains; refcount duplicate
  store/remove pairs per (dp_rank, tier) so a removal is forwarded only when the count reaches zero; support
  msgspec `array_like` tuples and tagged maps for both engines; accept signed and unsigned 64-bit hashes.
- Gateway: tier-aware credits (device > host > disk) from `cache_level`; a continuation index for lower
  tiers; `session_id` carried for diagnostics.
- Tests: recorded event batches from current vLLM and SGLang builds (both layouts), one fixture per field.

### 4.2 Indexer

Design direction (to be validated by the harness, not assumed):

- Lookups in O(log D) probes: galloping or binary probing over the request's chain hashes to find the
  longest matched position, with the prefix hash checked at every landing (guardrail 1).
- Events in O(1) amortized: run-compressed storage keyed on the hash chain (start position, head hash,
  length, worker coverage) so a stored run appends and an evicted tail truncates one run.
- Coverage as dense bitsets over interned worker ids; set intersection and counts by AND and popcount.
- Write lanes as dedicated OS threads, each (worker, dp_rank) pinned to one lane, FIFO per rank, bounded
  queues with backpressure and a metric; reads inline on the caller, lock-free (epoch or atomic-slot
  tables, no shard read locks, no `Arc` per hop).
- Grouped removals; a worker clear in O(that worker's blocks) via the reverse map (keep this: Dynamo's clear
  sweeps the whole tree).
- Allocator A/B on this host (jemalloc, which the gateway already uses, vs mimalloc) with RSS reported.
- A reference single-threaded indexer kept in the crate for guardrail 1.

### 4.3 Routing

- Cost-function policy interface in the gateway (filter / score / pick), with Dynamo's default cost
  function, `llm-d-optimized-baseline`, `ramjet` and `dualmap` as policies, so T5 compares policies under
  one host and one mock.
- Optimistic self-accounting of a dispatched request's uncached blocks (predicted), reconciled when the
  engine's events arrive; short-TTL predict-on-route side index merged by maximum.
- Decode-slot and prefill-backlog terms from `SchedulerLoad` and SGLang's load publication; softmax
  tie-breaking with temperature; output-block tracking.

### 4.4 Recovery

- Gap detection with engine replay over the publisher's replay socket; a bounded live-tail buffer while a
  rank recovers; per-rank cursors and fencing; a full per-worker resync (tree dump) when replay cannot cover
  the gap; a new router replica bootstraps from a peer's dump. The index after recovery must equal a fresh
  snapshot (guardrail 1).

### 4.5 Harness

- Dynamo's `mooncake_bench` with an SMG backend (their restriction to `nested-map` and CRTC is a few lines),
  so both systems are measured by one binary; and the same replay ported into `crates/kv_index/benches` so
  the loop does not depend on their repo.
- A KV-event-emitting mock engine: `mock-worker` with a per-worker simulated radix cache (capacity, LRU
  eviction, prefix hits), publishing vLLM-format events over ZMQ and returning `cached_tokens`; Mooncake
  trace replay through the gateway at configurable arrival speedups.
- Chaos scripts for the drills in §5.
- GB300 validation: vLLM and SGLang arm64 containers, 8–16 instances across the four GPUs with KV events on;
  weights for Qwen3-8B, gpt-oss-20b and Qwen3-0.6B are on the host.

## 5. Fault-tolerance drills

| Drill | Pass |
|---|---|
| engine killed mid-stream | its in-flight streams end with a clean error within 2 s; no hung stream; the worker is excluded within one health interval; its index state is cleared |
| engine restart with cache cleared | `AllBlocksCleared` applied; cursor reset; hit rate back to steady state within 60 s |
| event stream gap (N batches dropped) | replay restores an index identical to a fresh snapshot |
| slow subscriber | memory bounded; routing falls back to the approximate tree; a metric says so; automatic recovery |
| router restart | peers' dumps rebuild the index; p99 TTFT during the first 30 s ≤ 2× steady state |
| worker partitioned (packets dropped) | circuit breaker opens; no request waits past the connect timeout; it closes after recovery |
| overload (8× arrival) | admitted requests keep bounded TTFT; shed load is reported, not timed out |
| PD prefill worker lost during bootstrap | decode fails fast; the request is retried on another prefill worker |

## 6. The loop

### 6.1 Parallel structure (from 2026-10-05 22:55)

Phase 0 established the shared harness and the exactness tooling; from here the work fans out. Each
workstream owns a branch `leap/<topic>` cut from this branch, a worktree `/tmp/wt-leap-<topic>`, a
target directory, and a disjoint set of files; the orchestrator (the loop) harvests their commits,
integrates them here in dependency order, re-measures, appends the scoreboard, and re-dispatches.
Measurements on the quiet cores (0–63) are serialised through `flock /tmp/leap-measure.lock`;
builds and tests stay on cores 72–143.

| Workstream | Owns | Deliverable |
|---|---|---|
| indexer-run | `crates/kv_index/src/run_index*`, `tests/exactness_run.rs` | run-compressed, chain-hash-keyed index with O(log D) lookups and O(1) amortised events, bitset coverage, lock-free reads, measured in the shared harness |
| indexer-fast | `crates/kv_index/src/event_tree.rs`, `tests/exactness.rs` | PositionalIndexer: store-free lookups, bitset coverage, batched write path, allocation-free probes, each change measured |
| servicer-schema | proto `common.proto`, `engine_servicer` relay and tests, monitor field plumbing, `kv_index/src/salt.rs` | section 4.1: every engine field carried, normaliser rules, fixtures for both layouts and engines |
| mock-engine | `crates/mock_worker/**`, new `crates/replay/`, `~/smg-perf/replay/` | KV-event-emitting mock with a real prefix cache and timing model; Mooncake replayer with TTFT/TPOT/goodput/hit-rate/oracle |
| policy | `model_gateway/src/policies/**`, policy flags, selection bench | filter/score/pick interface, Dynamo default + llm-d-optimized-baseline + ramjet + dualmap, optimistic self-accounting, softmax |
| recovery | `model_gateway/src/worker/kv_event_monitor.rs` control flow, `~/smg-perf/chaos/`, `docs/recovery-protocol.md` | cursors, live-tail buffer, fencing, transactional resync, metrics, synthetic-stream tests, section 5 drills |
| bench-port | `crates/kv_index/benches/**` (not `dynamo-adapter/`), `~/smg-perf/indexer/` | corpus export from Dynamo's bench, SMG-side open-loop drain-inclusive replay, issuer scaling, parity check |
| gpu-harness | `~/smg-perf/gpu/**`, `docs/gpu-harness.md`, captured fixtures | vLLM and SGLang containers with KV events on the GB300s, real event captures, first real-engine baseline |

Integration rule: a workstream's commits land here only after its own gates and the exactness harness
pass on the integrated tree; conflicts in shared files (`lib.rs` exports, flags, workspace members) are
resolved by the orchestrator, never by a workstream editing another's files.


Each iteration: measure (scoreboard row) → profile → one change → gates (§3) → re-measure → append the row
and the decision here. Weekly: the full matrix with 20 trials and CIs, both systems rebuilt at their current
heads. The competitor's number is re-measured whenever their stack moves.

Phase 0, baselines and harness (first): both indexers in one binary on this host; mock engine with events;
reference indexer; chaos scripts. Phase 1, exactness and schema (§4.1, the jump-search fix, read-only
lookups). Phase 2, the indexer (§4.2). Phase 3, routing (§4.3) and recovery (§4.4). Phase 4, drills, soak,
GB300 validation, publication.

## 7. Scoreboard and decisions

| Date | Track | What | Result | Decision |
|---|---|---|---|---|
| 2026-10-05 | T1 | Dynamo `main` CRTC (glibc), this host, 59 backend cores, 8 / 32 event workers, overloaded 750 ms, 3 trials each | 49–58 M / 55–60 M block ops/s; their `nested-map` 34–38 M | baseline for their shipped code on this host |
| 2026-10-05 | T1 | SMG `throughput_bench` on the same trace and factors, SMG's own replay method, 72 cores, bench profile | peak 146 M (synthetic default workload: 176 M) | not comparable until the SMG backend runs inside their harness |
| 2026-10-05 | T1/T2 | Dynamo top-of-stack (`rupei/crtc-writer-lookup` @ 50bdb355f8, features `mooncake,router-bench` = mimalloc), this host, 59 backend cores (5–63), 64 event lanes, 128 query lanes | keeps up at 640 M (500 ms window) and 426 M (750 ms); overloaded 858–900 M (300 and 200 ms windows); lookup service p50 3.4 µs / p99 13–14 µs sustained, p50 2.0–2.6 µs / p99 8.5–10.4 µs when overloaded; 8 lanes: 177 M overloaded | **this is the reference on this host**: 640 M sustained, ~0.9 B capacity, p99 14 µs |
| 2026-10-05 | method | their harness's own ceiling on this host | at windows ≤ 150 ms the generator cannot issue on schedule (4 issuer cores + 1 query issuer offer at most ~2 B block ops/s) | a 10× result (≥ 6–9 B) is not measurable with their issuer layout; the SMG port of the harness must scale issuers (more issuer cores and lanes) and every result must report block ops/s per backend core and the sustained-latency curve alongside the headline |
| 2026-10-05 | T1/T2 | **SMG `PositionalIndexer` (this branch, 16bb22ee) inside Dynamo's `mooncake_bench`** via the adapter in `benches/dynamo-adapter/`, same binary, same masks (backend cores 5–63), jump 8 | 64 lanes: 142 M overloaded (does not keep up at 427 M offered), lookup service p50 14 µs / p99 345–377 µs; 8 lanes: 25 M, p50 11.7 µs / p99 33 µs; jump 64: 135–138 M, p99 416–476 µs; no event failures | first like-for-like number: 6× behind the competitor's capacity (142 M vs 860 M) and 25× behind on loaded lookup p99 (350 µs vs 14 µs); ingestion barely scales with lanes (8 → 64 lanes gives 25 → 142 M), so the per-block write path (shard write lock + `touch` per block) is the first bottleneck |
| 2026-10-05 | T1/T2 | SMG `PositionalIndexer` sustained-window sweep inside Dynamo's harness, 64 lanes, same masks | keeps up at 106 M (3 s window), fails at 213 M (1.5 s: 135 M achieved); lookup service p50 12–16 µs, p99 46–75 µs when keeping up, 320 µs at the edge. Competitor on the same windows: keeps up through 640 M, p50 3.3 µs, p99 13 µs | **on this host the baseline ratios are 6× on sustained throughput (106 M vs 640 M), 6× on capacity (142 M vs 860 M), 4× on lookup p50 and 3.5–25× on p99**; the 10× goal therefore means ≥ 6.4 B sustained here, beyond their harness's issuer ceiling, so the SMG-side port of the replay must scale issuers |
| 2026-10-05 | method | Phase 0 step 1 done: one binary measures both | adapter + wiring kept in `crates/kv_index/benches/dynamo-adapter/` | next: SMG's sustained (kept-up) window sweep, then the reference indexer and the exactness harness (Phase 1 cannot start without them) |
