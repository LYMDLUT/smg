# HiCache against the loop-head gateway (tier-aware monitor), 2026-10-05

Setup (container-free): two SGLang 0.5.21 workers through the loop-head Rust SGLang servicer
(`scripts/run-sglang-grpc-host.sh`, `SMG_SGLANG_SERVICER_IMPL=rust`, `--grpc-mode`; the servicer listens on
`--port`, here 21081/21082), KV events on, page size 16:
`sgl-hc` = `--enable-hierarchical-cache --hicache-ratio 8 --hicache-write-policy write_through --max-total-tokens 4096`
(4 k-token device pool, 32 k-token pinned host tier) on GPU 1; `sgl-plain` = no HiCache, 9 k-token pool, on GPU 2.
Gateway: loop-head `smg` (f4dc134b) `--backend sglang --policy cache_aware --log-level debug` on cores 64-71,
port 30300. Workload (`workload_kv.py --text --stream`, sequential, 5 s pauses): `shared` (8 prompts sharing a
512-token prefix), `repeat` (same 8), `evict` (16 unique 1024-token prompts), `recheck` (the shared 8 again).
The HiCache worker's own event stream was captured in parallel (`capture*.jsonl`, `hash_sequence.py`).

Run 2 (both workers healthy, `workload-run2.jsonl`, `gateway-debug-run2.log`, `capture-run2.jsonl`):

| phase | gateway decision (debug log, in order) | engine `cached_tokens` |
|---|---|---|
| shared | tree path (the event index was still empty for these workers): fallback -> 21082, 6 x `tree_match` -> 21082, 1 x `spill` -> 21081 | 0, then 512 x6 |
| repeat | event path: `event_hit` -> 21081 (x8, plus 6 more hits in the next phase's warm-up) | 512, 624 x7 |
| evict (16 unique) | `event_miss` -> 21081 x16; the HiCache worker's device pool overflows: its stream shows `Remove GPU` for the 96 prefix hashes with the `CPU_PINNED` copies kept | 0 x16 |
| recheck | **`event_hit` -> 21081 x8** (the demoted worker, still credited); its stream then shows `Store GPU` for the same 96 hashes (load-back) | **624 x8** |

Per-hash transitions on the HiCache worker for the shared prefix: `Store GPU -> Store CPU_PINNED -> Remove GPU -> Store GPU`
(96 hashes). The plain worker also held the prefix on its device (it served six of the eight `shared` requests) and was
healthy, so the recheck choice was between a device copy on 21082 and a host-only copy on 21081; the loop-head gateway kept
21081's blocks after `Remove GPU` because the host tier still had them and routed there with full credit, and the engine
confirmed with `cached_tokens` 624 (the host hit counts toward the cache report). Before the tier-aware monitor a `Remove GPU`
dropped the block from the index, so the demoted worker would have lost its credit.

Run 1 (`workload.jsonl`, `gateway-debug.log`, `capture.jsonl`): same sequence with only the HiCache worker healthy in the
gateway's view (the plain worker had been marked unhealthy by a stale probe before it was ready and the gateway had not
re-admitted it); 15 x `event_hit`, 16 x `event_miss`, then 8 x `event_hit` on 21081 with `cached_tokens` 624 x8.

Pitfalls met: the headless HiCache scheduler died once with `Scheduler watchdog timeout (300 s)` on its very first
1-token request after start (its servicer logged `Headless engine exited with code 1`); a relaunch served normally.
`--grpc-mode` is deprecated in SGLang 0.5.21 in favour of `--smg-grpc-mode`; the gateway must be launched with
`--backend sglang` once (the launcher's default is vllm).
