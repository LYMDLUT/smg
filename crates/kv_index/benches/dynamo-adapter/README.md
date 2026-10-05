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
Overheads the adapters charge to SMG (not present for the Dynamo backends): one `Vec<StoredBlock>`
per stored event, one `Vec<SequenceHash>` per removal, one `Vec<ContentHash>` per lookup, and the
translation of SMG's interned worker ids back to `WorkerWithDpRank` in each score.
`smg-run` takes `--max-workers` (default 256): the run index keeps one coverage bit per worker
slot per run, so the slot count is fixed up front (at most 1024); slots are reused after a worker
is removed.
