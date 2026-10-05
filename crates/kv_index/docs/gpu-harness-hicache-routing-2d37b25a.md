# HiCache against the pushed head 2d37b25a (tier-aware monitor with copy counts), 2026-10-05

Servicer wheel, `smg` and `replay` built from 2d37b25a; SGLang 0.5.21 workers from the host venv through the Rust SGLang
servicer (`scripts/run-sglang-grpc-host.sh`): `sgl-hc` (HiCache write-through, ratio 8, 4 k-token device pool, GPU 1,
servicer 21081) and `sgl-plain` (9 k pool, GPU 2, 21082). Gateway `--backend sglang --policy cache_aware --log-level debug`,
port 30300, cores 64-71. Workload `workload_kv.py --text --stream`, sequential.

## Run A (`2d37b25a/`): both workers registered from the start

shared 8 -> tree fallback alternated the two workers (4 each); repeat 8 and the following requests: `event_hit` -> 21082
(the plain worker, which also held the prefix on its device); evict 16 -> alternated; recheck 8 -> `event_hit` -> 21082,
`cached_tokens` 512 (prefix hit on the plain worker). The HiCache worker's prefix blocks had gone `Store GPU -> Store
CPU_PINNED -> Remove GPU` (320 removals) but the event path preferred the device copy elsewhere, so this run does not
test credit for a host-only copy.

## Run B (`2d37b25a-strict/`): the HiCache worker alone, then the plain worker added before the recheck

1. Gateway registered with 21081 only: shared 8 -> `[0, 512 x7]`, repeat 8 -> `624 x8` (`event_hit` x15), evict 16 unique
   1024-token prompts -> `event_miss` x16; the HiCache stream shows the 96 prefix hashes `Store GPU -> Store CPU_PINNED ->
   Remove GPU` (demotion, host copies kept). (A first recheck at this point, still single-worker, was `event_hit` x8 with 624.)
2. Second eviction round (16 new unique prompts, seed 32) -> the prefix is demoted again (`Remove GPU` on the 96 hashes).
3. `POST /workers {"url":"grpc://127.0.0.1:21082", ...}` -> the plain worker is admitted and healthy (empty cache).
4. recheck 8 (same shared prompts): **`event_hit` -> 21081 x8** and **`cached_tokens` 624 x8**; the HiCache stream then shows
   `Remove GPU -> Store GPU` for the 96 prefix hashes (load-back from the pinned host tier).

So with a real alternative available (a healthy, idle worker), the loop-head gateway kept crediting the demoted blocks on
the HiCache worker as host-resident and routed the repeats there; the engine confirmed with the host hit counted in
`cached_tokens`. Before the tier-aware monitor a `Remove GPU` dropped the block from the index and the recheck would have
gone through the no-overlap fallback.

Files: `gateway-debug*.log` (decision lines), `workload-*.jsonl` (engine truth per request), `capture*.jsonl` (HiCache
worker's event stream), `hash_sequence.py` output in the console log. Pitfall: the admin route is `POST /workers` with a
JSON body (`/add_worker` is not a route on this head).
