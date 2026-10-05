# mock-worker

Multi-port mock HTTP/gRPC inference workers for scale-testing the SMG gateway's
routing and async-runtime behavior. One process hosts many protocol-accurate
stand-ins for vLLM/SGLang engines.

Two modes:

- **canned** (default) — every response is fixed (no model); a single optional
  `--gen-ms` delay. Cheap enough to run thousands of idle workers for measuring
  the gateway's own CPU/registration/routing cost.
- **realistic** (`--engine realistic`) — each worker runs a continuous-batching
  **engine simulator** that behaves like a real LLM engine on CPU, so the whole
  gateway (including load- and cache-aware routing) can be exercised without
  GPUs. See [Realistic engine](#realistic-engine).

## What it implements

**HTTP** (vLLM/SGLang HTTP surface the gateway probes and routes to):
- `GET /health` → `200 OK` (gates registration + health promotion)
- `GET /v1/models` → one model, `owned_by: sglang` (backend/model detection)
- `POST /v1/chat/completions` · `/v1/completions` · `/generate` — non-stream JSON
  or SSE (`data: {chunk}\n\n … data: [DONE]\n\n`)
- `GET /v1/loads?include=core` → `WorkerLoadResponse` (load-aware policies)

**gRPC** (TokenSpeed scheduler — the gateway tokenizes, the worker speaks token
ids): `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `Generate` (streamed
chunks + complete), `GetLoads`, `Abort`, and (realistic mode only)
`SubscribeKvEvents`; other admin RPCs return `unimplemented`.

## Run (canned)

```bash
cargo run --release -p mock-worker -- \
  --http-base-port 9000 --http-count 2000 \
  --grpc-base-port 19000 --grpc-count 0 \
  --model mock-model --gen-ms 5
```

Each worker is one port. Register them against an IGW gateway with
`POST /workers` (`{"url":"http://127.0.0.1:9000"}`, or `grpc://…` with
`connection_mode`/`runtime`/`models` for gRPC).

## Realistic engine

`--engine realistic` backs each worker with a continuous-batching simulator
([`src/engine.rs`](src/engine.rs)) built the way vLLM schedules, so routing
experiments against it transfer to engines (and compare with Dynamo's offline
replay, which models the same loop):

- **pass loop** — every pass has a token budget (`--max-batched-tokens`, 8192)
  and a sequence cap (`--max-running`, 256). Running requests go first, each
  taking one decode token or a prefill chunk; then the queue is admitted FCFS
  while budget and KV room remain. With `--prefill-first true` (SGLang) a pass
  that contains prefill runs prefill only.
- **block-level KV pool** — `--kv-blocks`/`--kv-tokens` physical blocks of
  `--block-size` tokens, keyed by content hash with reference counts; idle
  cached blocks sit in an LRU and are evicted head-first when an allocation
  needs room; a running request that still cannot get a block preempts the most
  recently admitted request (LIFO), which recomputes later. A fully cached
  prompt recomputes its last block; a prefix hit on an idle block references it
  again.
- **KV events** (`SubscribeKvEvents`) — per request within a pass, `Removed`
  for the blocks evicted by its allocation, then one `Stored` per contiguous
  run of blocks it completed (parent-chained, with token ids). `Removed` fires
  only when the last copy of a hash leaves the pool, `Stored` only when a hash
  first appears; completion and preemption emit nothing. Every event of a pass
  becomes visible at the pass end, so a request arriving mid-pass cannot see
  that pass's blocks. A reset (`POST /admin/reset`) publishes
  `AllBlocksCleared`.
- **timing** — `--timing polynomial` (default) uses AISimulate's baseline:
  prefill `16.50142 + 1.518344e-2·T + 4.209989e-7·T²` ms over the uncached
  tokens `T` of the pass, decode `max(1, 5.74 + 54.01·u − 25.74·u²)` ms over
  the KV utilisation `u` of the decoding requests; a pass lasts prefill plus
  decode, and the first token of a prefill adds no decode time.
  `--timing linear` keeps the older `prefill_tps` / `base + slope·batch` model.
- **cached tokens** — a request sharing a prefix with cached blocks pays less
  prefill and reports `cached_tokens` (gRPC chunks, HTTP
  `usage.prompt_tokens_details.cached_tokens`).

| Flag | Default | Meaning |
|------|---------|---------|
| `--timing` | polynomial | `polynomial` or `linear` pass-duration model |
| `--prefill-poly a,b,c` | AISimulate | prefill ms = a + b·T + c·T² |
| `--decode-poly a,b,c` | AISimulate | decode ms = max(1, a + b·u + c·u²) |
| `--prefill-tps` | 8000 | linear model: prefill tokens/s (selects `linear`) |
| `--decode-base-ms` | 6.0 | linear model: fixed decode-pass ms |
| `--decode-per-req-ms` | 0.35 | linear model: decode ms per running request |
| `--max-batched-tokens` | 8192 | token budget per pass (`--prefill-chunk` is an alias) |
| `--max-running` | 256 | sequences per pass |
| `--kv-tokens` / `--kv-blocks` | 524288 tokens | KV pool capacity |
| `--block-size` | 16 | cache block/page size (tokens); must match the worker's `kv_block_size` |
| `--prefix-cache` | true | prefix caching + KV events |
| `--prefill-first` | false | SGLang-style prefill-only passes |
| `--context-length` | 32768 | advertised context length |
| `--admin-port` | off | process-wide admin API (below) |

```bash
cargo run --release -p mock-worker -- \
  --engine realistic --grpc-base-port 19000 --grpc-count 8 --model mock-model --admin-port 19100
```

Agreement with hardware is the caller's problem: AISimulate's published
agreement for these polynomials (mean absolute percentage error 48.5% on TTFT,
28.9% on TPOT) was measured with prefix caching disabled, so cache-hit and
routing effects have no published validation. Treat the simulator as a relative
A/B harness for policies and validate absolute numbers on GPUs.

### Admin API

`--admin-port` serves the ground truth a routing benchmark needs and real
engines do not expose:

- `GET /admin/fleet` — every engine (`grpc:<port>` / `http:<port>`) with cache
  size, load, cached blocks and preemptions;
- `GET /admin/requests?since=<seq>&limit=<n>` — admitted requests with the
  serving worker, prompt/cached tokens, queue wait and the **arrival-time
  oracle**: the most cached tokens any worker of the process held when the
  request arrived (the best a router could have obtained). Join on the
  gateway's response `id`;
- `GET /admin/cache/{worker}` — the worker's cached block keys;
- `POST /admin/reset[/{worker}]` — clear caches and publish `AllBlocksCleared`
  (an engine restart, to the index).

**Tokenizer note (gRPC):** the gateway tokenizes prompts before routing, so it
needs a real tokenizer for the model. Register each worker with a tokenizer
label, e.g. `"labels":{"tokenizer_path":"gpt2"}`, and a `"kv_block_size":16`, and
do **not** pass `--disable-tokenizer-autoload`. (HTTP workers need no tokenizer
but cannot drive event-driven `cache_aware`, which requires token ids.)

## Scale-test rig (gateway CPU)

`scripts/scale_test.sh` launches an IGW gateway, starts a canned mock fleet,
REST-registers it, and samples the gateway PID's CPU + `/health` latency:

```bash
scripts/scale_test.sh --http 2000 --policy cache_aware --rps 500 --duration 30
scripts/scale_test.sh --grpc 1000 --policy least_load
```

## Policy A/B rig (no-GPU routing fidelity)

`scripts/sim_ab.sh` launches the gateway + a **realistic** mock fleet, drives the
same Poisson workload (`scripts/sim_load.py`, with a tunable shared-prefix
fraction) under each routing policy, and prints a side-by-side table of
TTFT / ITL / E2E / throughput:

```bash
# Full fidelity: gateway tokenizes (downloads gpt2) and routes on token ids,
# so both least_load and event-driven cache_aware engage.
scripts/sim_ab.sh --mode grpc --workers 8 --rps 120 --duration 30

# Offline-friendly: no tokenizer; drives least_load + latency + approximate cache_aware.
scripts/sim_ab.sh --mode http --workers 16 --policies "random least_load"
```
