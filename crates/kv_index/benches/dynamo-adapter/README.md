# SMG backend for Dynamo's Mooncake indexer replay

`smg_positional.rs` implements Dynamo's `SyncIndexer` over this crate's `PositionalIndexer`, and
`wiring.patch` adds the `smg-positional` backend to `lib/bench/kv_router/mooncake_bench` in
`ai-dynamo/dynamo` (applied on top of `rupei/crtc-writer-lookup`, the top of their CRTC stack).
Together they let one binary measure both indexers with identical lanes, queues, observation
records and accounting, which is the T1/T2 comparison in `../../docs/kv-router-leap.md`.

Apply: copy `smg_positional.rs` to `lib/kv-router/src/indexer/`, `git apply wiring.patch`, point
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
