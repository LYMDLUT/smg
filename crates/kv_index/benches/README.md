# kv_index benchmarks

| File | What it is |
| --- | --- |
| `throughput_bench.rs` | Criterion micro-benchmarks of the indexers (see its module doc). |
| `mooncake_replay.rs` | Open-loop replay of a Mooncake indexer corpus by Dynamo's method, against this crate's indexers (`cargo bench -p kv-index --bench mooncake_replay -- --help`). |
| `corpus_export.patch` | The `--export-corpus` option for Dynamo's `lib/bench/kv_router/mooncake_bench` (against ai-dynamo/dynamo `50bdb355f8`), which writes the corpus the replay consumes. |
| `dynamo-adapter/` | `PositionalIndexer` behind Dynamo's `SyncIndexer`, so Dynamo's own binary can measure it (its README explains the wiring). |
| `protocol/bisect_sustained.py` | Threshold search for sustained throughput: brackets the highest offered rate at which trials keep up (3 fresh-process trials per point, geometric bisection to within 10%), for either harness. |
| `protocol/publish_protocol.py` | Guardrail 5 runner: N fresh-process trials with an interleaved same-binary control pair, the lock held per trial, a foreign-load check on the measurement cores before and after each trial, medians with bootstrap 95% confidence intervals, markdown output. |
| `protocol/hostload.py` | The foreign-load sampler the two scripts share (per-process CPU on a core set over a short interval). |

## Why a corpus and a replay

Dynamo's indexer benchmark (`lib/bench/kv_router/INDEXER_BENCH.md` in ai-dynamo/dynamo) defines
the measurement this crate compares itself against: the Mooncake trace replayed open loop, with
deadlines scaled into a window, 128 query lanes, events sharded by worker, and throughput counted
in block operations (requested, stored and removed block hashes) over the time from the start of
issue to the last completion, drain included. Reproducing that definition here has two steps:

1. Dynamo's binary prepares the schedule exactly as it would run it (trace parsing, duplication,
   deadline sort, dense ids) and, with the patch above, writes it to a file instead of running.
2. `mooncake_replay` loads that file and runs the same open-loop protocol against a backend behind
   a small trait, with no dependency on Dynamo's crates at measurement time.

The same file replays byte-for-byte against both harnesses, so the two can be checked against each
other (the parity table below) and the loop can measure new indexers here while Dynamo's binary
remains the arbiter for published comparisons.

## Corpus format: `SMGMCK01`, version 1

All integers are little-endian. One file holds one prepared schedule for one window; deadlines
are already scaled to that window. The replay rescales linearly when asked for another window
(`--benchmark-duration-ms`), which is what Dynamo does per trial.

| Section | Layout |
| --- | --- |
| Magic | 8 bytes `SMGMCK01` |
| Header | u32 version (1), u32 block_size, u64 reference_window_ns, u64 trace_duplication_factor, u64 trace_length_factor, u64 inference_worker_duplication_factor, u64 logical_workers (max worker id + 1) |
| Totals | 7 × u64: requests, stored_events, removed_events, cleared_events, request_blocks, stored_blocks, removed_blocks |
| Trace path | u64 length, UTF-8 bytes (informational) |
| Query hashes | u64 count, count × u64 local block hash |
| Stored blocks | u64 count, count × (u64 block_hash, u64 tokens_hash) |
| Removed hashes | u64 count, count × u64 block_hash |
| Operations | u64 count, then per operation: u32 id, u64 deadline_ns, u64 worker_id, u8 kind, kind-specific fields |

Kind-specific fields:

| kind | Fields |
| --- | --- |
| 0 query | u64 start into the query hash slab, u32 length |
| 1 stored | u32 dp_rank, u64 event_id, u8 has_parent, u64 parent (0 when absent), u8 has_start_position, u32 start_position (0 when absent), u64 start into the stored-block slab, u32 length |
| 2 removed | u32 dp_rank, u64 event_id, u64 start into the removed-hash slab, u32 length |
| 3 cleared | u32 dp_rank, u64 event_id |

Operations appear in Dynamo's issue order (deadline, queries before events at equal deadlines,
worker id, trace order) and ids are their dense positions; the loader verifies both, and the
totals. The export is deterministic: two exports of the same arguments hash identically.

## What the replay does

The protocol follows `mooncake_open_loop.rs` in Dynamo step by step:

- Query lane = `worker_id % query_lanes`. Event lane = round robin over `(worker_id, dp_rank)` in
  order of first appearance (`ThreadPoolIndexer`'s assignment). Event issuers own contiguous
  worker ranges (`contiguous_worker_issuer`); query lanes are sharded over the query issuers in
  contiguous ranges (one issuer by default, as in Dynamo).
- Before the trial: `malloc_trim`, a quiescence sleep (`--pre-run-quiescence-ms`, 5000), a pass
  over the corpus pages, thread pinning (`--issuer-cpus`, `--query-issuer-cpu`, `--backend-cpus`).
- Issue: start is 20 ms after the barrier; each issuer sleeps to the absolute monotonic deadline
  (`clock_nanosleep`, `TIMER_ABSTIME`) and spins the last `--issuer-spin-us`; the queries of a
  deadline are published before its events; `accepted_ns` is stamped after the handoff.
- Lanes record started/finished for lookups and finished for events. Completion edges give the
  maximum queue depth and outstanding updates; FIFO is checked per worker.
- Validity, as Dynamo: the generator is valid when nothing failed and the issue span is at most
  1.01 × window; `kept_up` additionally needs the last completion within 1.10 × window.
- Rates: `achieved_block_ops_per_sec = total_block_ops / (last_completion − start)`, drain
  included; `offered` divides by the window; `actual_issue` by the issue span. Percentiles are
  nearest-rank (p50, p99, p99.9, max). The result JSON uses Dynamo's field names, plus `harness`,
  `corpus`, `mirror_dynamo_costs`, `rejected_events` and a `provenance` object (argv, binary and
  corpus blake3, trace parameters).

Differences from Dynamo's runner, all on the harness side:

- Lanes are OS threads (query lanes park and are unparked by the issuer; event lanes block on
  `std::sync::mpsc`) instead of tokio tasks on a `Notify` and `flume` channels. Both are
  unbounded with one consumer per lane.
- `--issuer-threads` sets the number of event issuers (Dynamo derives it from `--issuer-cpus`),
  and `--query-issuer-threads` the number of query issuers. Dynamo publishes every lookup from one
  thread and its event issuers wait for that thread at each deadline; on this host that single
  thread caps the generator near 1.5B block ops/s whatever the number of event issuers (table
  below). With several query issuers the per-deadline flag becomes a pending count that each
  issuer retires for its share, so events still wait for every query of their deadline.
- `--mirror-dynamo-costs` (default on) charges the lanes what Dynamo's harness charges every
  backend: each event arrives as an owned payload in Dynamo's block layout (40 bytes per block,
  allocated before the trial) that the lane converts into this crate's 16-byte blocks and frees
  after the apply, and each lookup copies its hashes into this crate's hash type, as the adapter
  does inside Dynamo's binary. With it off, lanes read the corpus slabs and copy nothing; that is
  the cheapest way to drive a backend here, not a number comparable with Dynamo's.
- Backends: `positional` (this crate's `PositionalIndexer`), `reference` (the single-threaded
  exactness reference; small corpora only) and `null` (no indexer: the harness's own ceiling on a
  layout). A new index plugs in by implementing `ReplayBackend` (four slice-based methods).

## Commands

Export the standard corpus from a Dynamo checkout with `corpus_export.patch` applied (the
`dynamo-adapter` wiring adds the `smg-positional` subcommand; any backend subcommand works):

```
mooncake_bench lib/kv-router/traces/mooncake_trace.jsonl \
  --num-unique-inference-workers 128 --trace-duplication-factor 20 --trace-length-factor 4 \
  --query-lanes 128 --benchmark-duration-ms 3000 --export-corpus mooncake-w128-dup20-len4-3000ms.smgmck \
  positional
```

Replay it on Dynamo's competitor layout (issuers on 0-3, query issuer on 4, lanes on 5-63):

```
cargo bench -p kv-index --bench mooncake_replay --no-run
numactl --interleave=all target/release/deps/mooncake_replay-<hash> mooncake-w128-dup20-len4-3000ms.smgmck \
  --backend positional --query-lanes 128 --event-lanes 64 --issuer-threads 4 \
  --issuer-cpus 0-3 --query-issuer-cpu 4 --backend-cpus 5-63 --result-json-output result.json
```

Harness ceiling for a layout (no indexer): 16 event issuers, 4 query issuers, 50 ms window from
the same corpus:

```
numactl --interleave=all target/release/deps/mooncake_replay-<hash> mooncake-w128-dup20-len4-3000ms.smgmck \
  --backend null --benchmark-duration-ms 50 --issuer-threads 16 --issuer-cpus 0-15 \
  --query-issuer-threads 4 --query-issuer-cpus 16-19 --backend-cpus 24-63 --result-json-output ceiling.json
```

`--offered-block-ops-per-sec <rate>` sets the window from the corpus's block-op total instead of
`--benchmark-duration-ms`, which is what a threshold search moves. One process per trial, as
Dynamo's method requires.

## Sustained throughput: the threshold search

The contract defines sustained throughput as the highest offered rate at which a trial keeps up
(generator valid, achieved at least 99% of offered). A window-driven replay only says "keeps up
at this window", so the threshold has to be bracketed:

```
python3 benches/protocol/bisect_sustained.py --lock /tmp/measure.lock --out out/bisect \
  --lo 107e6 --hi 427e6 --trials 3 --tolerance 0.10 \
  --command "numactl --interleave=all <mooncake_replay> <corpus> --backend positional ... \
             --offered-block-ops-per-sec {rate} --result-json-output {json}"
```

`--lo` must keep up and `--hi` must fail (`--verify-ends` checks both first); each point runs
three fresh processes and passes only if all three keep up; the search moves the geometric
midpoint until the bracket is within the tolerance and writes `bracket.json` and `bracket.md`.
For Dynamo's binary use `{window_ms}` in the template with `--total-block-ops` (the corpus
total, 320,105,993 for the standard corpus), and the script derives the window per point.

## Publication protocol (guardrail 5)

```
python3 benches/protocol/publish_protocol.py --name "<system, harness>" --trials 20 \
  --lock /tmp/measure.lock --cores 0-63 --allow '<background daemon regex>' --out out/protocol/<tag> \
  --command "<one trial, with {json} for the result path>"
```

Each subject trial is followed by a control trial of the same command (or `--control-command`),
so the pair shows the noise floor an A/A comparison would show before any difference under 5% is
called. Every trial holds the lock, and the measurement cores are sampled for one second before
and after it: a process above 5% CPU that is neither the trial nor allow-listed marks the trial
discarded (kept and listed with the offender); allow-listed daemons are recorded as background.
The summary gives medians with percentile-bootstrap 95% intervals (10,000 resamples) of achieved
block ops/s and lookup p50/p99 per series, the subject-minus-control difference with its own
interval, the discarded trials and why, and the background processes seen. Finished trials are
skipped on re-run, so an interrupted run resumes.

## Plugging in a new index

`ReplayBackend` is four slice-based methods plus a per-lane state type. The run-compressed index
(`RunIndex`: `intern_worker`, `apply_stored(worker, &[StoredBlock], parent, &mut RunBlockMap)`,
`apply_removed`, `apply_cleared`, `find_matches(&[ContentHash], early_exit)`) maps onto it exactly
as `Positional` does, with `RunBlockMap` as the per-worker map held in the lane; add a
`BackendKind` variant and a `run()` arm in `main`.

## Parity

The same corpus (128 workers, duplication 20, length factor 4; 2,446,195 operations, 320,105,993
block ops) was replayed against this crate's `PositionalIndexer` through both harnesses on one
144-CPU Neoverse-V2 host, Dynamo's competitor layout (event issuers on CPUs 0-3, query issuer on
4, 64 event lanes and 128 query lanes on 5-63, `numactl --interleave=all`, one process per trial,
3000 ms and 750 ms windows). Dynamo's binary ran the indexer through `dynamo-adapter/`.

| Window | Harness | Achieved per trial (M block ops/s) | Lookup p50 (us) | Lookup p99 (us) | Scheduled to finished p99 (us) | Kept up |
| --- | --- | --- | --- | --- | --- | --- |
| 3000 ms | Dynamo `mooncake_bench`, SMG adapter | 106.1, 106.2, 106.2 (mean 106.2) | 17.3-18.2 | 113-138 | 3278-6972 | yes |
| 3000 ms | SMG `mooncake_replay` | 106.1, 106.2, 106.4 (mean 106.2) | 14.0-16.9 | 58-114 | 517-2350 | yes |
| 750 ms | Dynamo `mooncake_bench`, SMG adapter | 131.6, 132.4, 134.7, 136.9, 137.7, 140.9, 142.7, 144.6 (mean 137.7) | 15.5-21.1 | 378-487 | 1234-119188 | no |
| 750 ms | SMG `mooncake_replay` | 106.0, 135.4, 135.9, 141.1, 141.3, 143.5, 147.7, 148.2, 169.6, 172.2, 184.5 (mean 147.8) | 13.2-18.3 | 199-302 | 498-10019 | no |

Diagnostics at 750 ms, SMG harness: 64 lanes pinned round robin: 121.4; 59 lanes pinned one per core: 115.8, 113.1 (M block ops/s).

At the sustained window the two harnesses agree on throughput to 0.2%, with lookup latency a
little lower here (parked OS-thread lanes against tokio tasks). At the overloaded window, which
measures capacity, the SMG harness spreads wider: most trials sit in or just above Dynamo's
band, three ran 20-35% faster and one slower. The per-lane diagnostics explain the spread: an
event lane spent 36-41 us of CPU per event in a trial where it kept a core to itself and 50-74 us
across floating trials, and with 64 lanes on 59 cores the drain is set by whichever lanes end up
sharing cores and by what else the host is doing. Pinning does not help on this layout: 64 pinned
lanes double up five cores, and 59 lanes leave 128 workers unevenly spread (three workers on some
lanes, two on others). Dynamo's harness, whose query lanes are tokio workers on the same cores,
sits consistently in the slow mode. Capacity under overload is therefore a property of the lane
layout and the host as much as of the indexer, in both harnesses: sustained numbers are
comparable across harnesses, capacity numbers should be compared within one harness and over
many trials, and published comparisons keep using Dynamo's binary.

## Issuer scaling and the measurable ceiling

The generator has to issue the whole corpus within 1.01 x the window for a trial to count, so the
harness has a ceiling of its own. Measured with the `null` backend (lanes do no work) by
rescaling the 3000 ms corpus to shorter windows:

One query issuer on CPU 16, event issuers on CPUs 0..N-1, lanes on 17-63, null backend; cells give the generator verdict, the rate actually issued and the issue-lag p99 for reads (r) and updates (u):

| Event issuers | 300 ms (1.07B offered) | 150 ms (2.13B offered) | 100 ms (3.20B offered) | 75 ms (4.27B offered) | 50 ms (6.40B offered) |
| --- | --- | --- | --- | --- | --- |
| 1 | INVALID 0.32B issued, lag p99 r 87 / u 679963 us | INVALID 0.29B issued, lag p99 r 45467 / u 928753 us | INVALID 0.30B issued, lag p99 r 108645 / u 964914 us | INVALID 0.31B issued, lag p99 r 125718 / u 934179 us | INVALID 0.29B issued, lag p99 r 149054 / u 1028555 us |
| 2 | INVALID 0.77B issued, lag p99 r 85 / u 113491 us | INVALID 0.74B issued, lag p99 r 44837 / u 280203 us | INVALID 0.70B issued, lag p99 r 110769 / u 354107 us | INVALID 0.73B issued, lag p99 r 116020 / u 357057 us | INVALID 0.67B issued, lag p99 r 159181 / u 420190 us |
| 4 | valid 1.07B issued, lag p99 r 88 / u 141 us | INVALID 1.57B issued, lag p99 r 52848 / u 52953 us | INVALID 1.58B issued, lag p99 r 101456 / u 101607 us | INVALID 1.48B issued, lag p99 r 139067 / u 139189 us | INVALID 1.55B issued, lag p99 r 155040 / u 155227 us |
| 8 | valid 1.07B issued, lag p99 r 99 / u 94 us | INVALID 1.54B issued, lag p99 r 56819 / u 56868 us | INVALID 1.61B issued, lag p99 r 98170 / u 98234 us | INVALID 1.47B issued, lag p99 r 140528 / u 140618 us | INVALID 1.55B issued, lag p99 r 154695 / u 154785 us |
| 16 | valid 1.07B issued, lag p99 r 168 / u 152 us | INVALID 1.55B issued, lag p99 r 56348 / u 56333 us | INVALID 1.58B issued, lag p99 r 101028 / u 101059 us | INVALID 1.47B issued, lag p99 r 140846 / u 140902 us | INVALID 1.50B issued, lag p99 r 161154 / u 161223 us |

Ceiling (highest offered rate with a valid generator):
- event issuers 1: no valid window in the grid
- event issuers 2: no valid window in the grid
- event issuers 4: 1.07B block ops/s (300 ms window)
- event issuers 8: 1.07B block ops/s (300 ms window)
- event issuers 16: 1.07B block ops/s (300 ms window)

Same layouts with the overloaded `PositionalIndexer` (lanes busy, so publishes rarely wake a parked lane):

- t16 q1 150 ms: valid True, achieved 115.0M, lookup p50 9.8 us p99 396 us, drain 2634 ms
- t16 q1 50 ms: valid True, achieved 118.0M, lookup p50 7.8 us p99 323 us, drain 2661 ms

Two facts matter for measuring indexers far beyond today's numbers. First, event issuers scale
(about 0.3-0.4B block ops/s per thread here), but with one query issuer the generator stalls near
1.5B block ops/s whatever their number, because event issuers wait at every deadline for the
query issuer, which pays a wake-up per published query when the lanes are idle; the query issuer
is now sharded (`--query-issuer-threads`). Second, the null backend is the worst case for that
cost: with a busy backend the lanes rarely park, and 16 event issuers with a single query issuer
issued the whole corpus on schedule even at the 50 ms window (6.4B block ops/s offered) against
the overloaded `PositionalIndexer`.
