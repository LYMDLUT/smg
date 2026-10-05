# SMG backend for Dynamo's Mooncake indexer replay

`smg_positional.rs` implements Dynamo's `SyncIndexer` over this crate's `PositionalIndexer`,
`smg_run.rs` does the same over `RunIndex`, and `null_indexer.rs` is a backend that stores and
matches nothing (the replay's memory and scheduling floor, so a backend's resident bytes per block
can be read as its RSS above the null run's). `wiring.patch` adds the `smg-positional`, `smg-run`
and `null-indexer` backends to `lib/bench/kv_router/mooncake_bench` in `ai-dynamo/dynamo` (applied
on top of `rupei/crtc-writer-lookup`, the top of their CRTC stack). Together they let one binary
measure every indexer with identical lanes, queues, observation records and accounting, which is
the T1/T2/T3 comparison in `../../docs/kv-router-leap.md`.

Apply: copy the three `.rs` files to `lib/kv-router/src/indexer/`, `git apply wiring.patch`, point
the `kv-index` path dependency at this crate's checkout, then

```
cargo bench --package dynamo-bench --bench mooncake_bench --no-default-features \
  --features mooncake,router-bench --no-run
```

Overheads the adapter charges to SMG (not present for the Dynamo backends): the translation of
SMG's interned worker ids back to `WorkerWithDpRank` in each score (one lock-free `ArcSwap` load per
lookup, one map insert per matching worker). Stores, removals and lookups pass the event's own
buffers through SMG's `apply_stored_iter`, `apply_removed_iter` and `find_matches_in`, so the
adapter builds no per-event or per-lookup `Vec`.
The patch also adds `end_rss_bytes` to the replay's result JSON: the process's `VmRSS` once every
lane has drained, which is the corpus plus the backend's index. Read a backend's resident bytes
as its figure minus the `null-indexer` run's at the same settings; peak RSS from `time` is taken
during corpus generation and says nothing about the index.

`lane_cpu.patch` (applied after `wiring.patch`) names the thread-pool lanes `kv-lane-<n>` and adds
`backend_lane_cpu_ns` to the result JSON: the user plus system time of those threads read from
`/proc/self/task` when the run ends, so a backend's lane cost per block op is a measured number
rather than a share of profile samples.

`smg-run` takes `--max-workers` (default 256): the run index keeps one coverage bit per worker
slot per run, so the slot count is fixed up front (at most 1024); slots are reused after a worker
is removed.

## Lanes: `kv_index::LanePool`

`smg_run.rs` runs the harness's lane threads as lanes of SMG's `LanePool` (`src/lane_pool.rs`):
each lane drains its channel into per-worker queues and serves ready workers from any lane's list,
whole workers at a time, so events of one worker keep their order while a quiet lane works off a
busy lane's backlog. `SmgRun::new(max_workers, lanes)` takes the harness's `--num-event-workers`
as `lanes`; thread counts, pinning and the per-lane CPU accounting stay the harness's, and the
other backends' dispatch is untouched. A worker's queue holds at most 2048 events; past that the
lane keeps the next ones back and reads no further from its channel until they are in, so the
harness's channel is the overflow and its `queue_depth_at_stop` row still measures it. Flush, seal,
stats and worker removal travel as barriers behind every worker the channel has fed, answered by
whichever lane applies the last one; observation records go to the completion writer of the
channel the event arrived on, which keeps each lane's buffer at the capacity the harness planned.
The report's `pool:` line gives events enqueued, applied and refused by the cap, whole-worker
steals, the deepest worker queue and the most events queued at once; `lanes:` gives apply time and
wait time summed over lanes.

## Lane-map slots: 8 bytes and the engine-hash invariant

A lane-map slot is the block's place in the run index, `(run, offset)`, plus a 16-bit word of
fingerprint and Robin Hood displacement; the engine hash itself lives once per distinct block in
the index, beside the content hash, and every lookup checks a candidate slot against it. The
invariant is that every engine names a block by the same hash (true of the replay, whose events
derive both hashes from the trace, and of a fleet of one engine build and hashing seed). A worker
whose engine disagrees is still exact: the index counts the block under `engine-hash conflicts`
and keeps that worker's key in the map's side table at a hash-map entry's cost.

## Memory rows: `memory_accounting.patch`

Whole-process RSS says nothing about an indexer inside this harness (the generator's simulated
engines hold 13 to 17 GB). `memory_accounting.patch`, applied on top of `wiring.patch`, adds a
`memory-accounting` feature to `dynamo-bench` that wraps the same mimalloc global allocator in
counters of live bytes and calls, snapshots them after the pre-run quiescence (before the backend
exists), after the final flush (the backend holds the replay's end state) and after the backend is
dropped (its lanes terminated and joined), and writes a `memory_accounting` object into the result
JSON. The field is `null` when the feature is off, so timing binaries are untouched; the counters
are shared atomics and perturb timing, so never read throughput or latency from an accounting run.

Build and run one accounting trial per backend with the same command lines as the timing runs:

```
cargo bench --package dynamo-bench --bench mooncake_bench --no-default-features \
  --features mooncake,router-bench,memory-accounting --no-run
```

Fields: `indexer_live_bytes` is what dropping the backend freed, which is the index, its lanes'
per-worker state and queues and nothing of the harness; `live_bytes_delta` (baseline to final
flush) also contains the harness's own run buffers and is there for cross-checking;
`resident_blocks_at_end` is the sum of the backend's own per-worker block counts after the flush
(`WorkerTask::Stats`, the writer lookup for CRTC, the per-lane maps for SMG) and
`net_stored_blocks` the harness's stored-minus-removed count; `bytes_per_resident_block` is
`indexer_live_bytes / resident_blocks_at_end`, the T3 figure; `indexer_references_at_drop` must be
1 for the drop to have freed the backend. Bytes are the sizes the program requested through
`GlobalAlloc` layouts, not the allocator's internal overhead, so the figure does not depend on the
allocator.
