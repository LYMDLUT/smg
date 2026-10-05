# Exactness of the competitor's index against SMG's reference

`smg_exactness_compare.rs` is an integration test for Dynamo's `kv-router` crate. It replays the
seeded corpora of `crates/kv_index/tests/exactness.rs` (hole-free, with holes, and the focused
ten-block case) into three indexers fed by the same events: Dynamo's `ConcurrentRadixTreeCompressed`
through one `SyncIndexer` lane (`EventWithAck` per event, `Flush` before each lookup round, lookups
on the test thread with `early_exit = false`, exactly the thread pool's paths), SMG's
`ReferenceIndexer` as the oracle, and SMG's production `PositionalIndexer` as a control that must
report zero mismatches. Every lookup's full score map is compared; CRTC disagreements are split into
under-counts and over-counts, and CRTC's per-event apply errors are counted.

## Placing and running it

```
cp smg_exactness_compare.rs <dynamo>/lib/kv-router/tests/
export CARGO_TARGET_DIR=...   # keep it off the measurement target dirs
cd <dynamo>/lib/kv-router && cargo test --release --features smg-backend --test smg_exactness_compare -- --nocapture
```

`<dynamo>` is a checkout of Dynamo's `rupei/crtc-writer-lookup` (top of their unmerged CRTC stack,
50bdb355 at the time of writing) with the `smg-backend` feature from `wiring.patch`, which gives the
crate a path dependency on this `kv_index`. Knobs: `KV_INDEX_EXACTNESS_EVENTS` (20000),
`SMG_EXACTNESS_SEEDS` (`20261005,20261006`), `SMG_EXACTNESS_LARGE_EVENTS` (80000, hole corpus,
first seed). Each corpus runs the three (jump, workers) configurations SMG's own harness uses.

## Result (2026-10-05, five identical runs)

| corpus | lookups | CRTC mismatches | over-counts | CRTC store errors (ParentBlockNotFound) | SMG `PositionalIndexer` |
|---|---|---|---|---|---|
| hole-free, 2 seeds × 20k events × 3 configs | 9,656 + 9,613 | 0 | 0 | 0 of 100k | 0 |
| holes, 2 seeds × 20k × 3 | 10,485 + 10,666 | 7 + 5 (0.07% / 0.05%), all under-counts | 0 | 6 of 100k | 0 |
| holes, 80k × 3 | 39,686 | 7 (0.02%), the same events as the 20k run | 0 | 6 of 201k | 0 |

Under-count deficits per hit were 1 to 28 blocks (for example a 46-block across-hole query scored
21). Focused case (worker holds A..J, evicts E, keeps F..J): after the eviction all three agree on 4;
after the engine re-stores E behind D the reference and SMG score A..J as 10, CRTC as 5; a later
store of K behind J is rejected by CRTC with ParentBlockNotFound and the chain stays at 5 for good.

Mechanism, in Dynamo's `lib/kv-router/src/indexer/concurrent_radix_tree_compressed/`: a removal sets
the rank's single cutoff to the lowest removed position and clears its full bit (`node.rs:710-750`,
`state.rs:321-344`), then scrubs the lane's lookup entries for every hash past the hole
(`remove.rs:114`; the `TODO(CORRECTNESS)` at `remove.rs:105-110` names the gap). A later store whose
parent lies past the hole finds no entry (`store.rs:143-178`) or is rejected by `parent_coverage`
(`node.rs:441`, `store.rs:181-201`); re-storing the hole credits only what that store carries
(`node.rs:460-493`), and a re-fill that diverges inside the holed edge splits it so the suffix
inherits only full bits (`state.rs:263-293`). Lookups stop at the cutoff (`node.rs:812, 829`). The
corpus rate is low because a re-fill that reaches the end of the edge promotes the rank back to full
and the child edges still carry its pre-hole bits; the loss is permanent only when the re-fill
diverges strictly inside one compressed edge. CRTC never over-counted in any run.

This measures the data structure under one lane with the tree quiescent at every lookup, not the
thread pool's documented best-effort behaviour under concurrent writers.
