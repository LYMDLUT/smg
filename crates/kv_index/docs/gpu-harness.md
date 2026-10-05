# GPU harness: real engines on devgpu044 (GB300)

Companion to the KV-router leap contract (`kv-router-leap.md`, 3.T4/T9 and 4.5). Everything here ran on
devgpu044 (arm64 Grace, 4x NVIDIA GB300 284 GB, driver 580.126.20, podman 5.8.5 rootless, no
nvidia-container-toolkit, no root). Scripts, fixtures and results live under `~/smg-perf/gpu/`; the
captured streams are checked in under `crates/engine_servicer/tests/fixtures/captured/`.

## 1. GPU access without root

No CDI spec exists on the host and nothing can be installed, so containers get the GPUs the manual way
(`scripts/podman-gpu.sh`, path (a) of the three candidates; it worked first time, so (b) a bare venv with the
aarch64 vLLM wheel and (c) the SGLang image were not needed as fallbacks):

- device nodes passed with `--device`: `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`,
  `/dev/nvidia-caps-imex-channels/channel0` and `/dev/nvidia<N>` per GPU (all `crw-rw-rw-`; the
  `/dev/nvidia-caps/*` nodes are root-only and are not needed);
- the driver's user-space libraries copied once from `/usr/lib64` into `~/smg-perf/gpu/nvlibs`
  (`scripts/stage-nvlibs.sh`: `libcuda`, `libnvidia-ml`, `libnvidia-ptxjitcompiler`, `libnvidia-nvvm`,
  `libnvidia-gpucomp`, `libnvidia-cfg`, `libnvidia-allocator`, `libcudadebugger`, `libnvidia-nscq`, with their
  SONAME symlinks, plus `nvidia-smi`) and bind-mounted read-only at `/usr/local/nvidia/lib64`, which the
  `nvidia/cuda` base images already have on `LD_LIBRARY_PATH`;
- `--cpuset-cpus=72-143` (cores 0-63 are the measurement partition of the other workstreams), `--network host`,
  `--ipc host` (do not combine with `--shm-size`), `--log-driver k8s-file` (the rootless default is journald,
  which the user cannot read).

Check: `podman-gpu.sh 0 --rm --entrypoint python3 docker.io/vllm/vllm-openai:latest -c 'import torch; print(torch.cuda.is_available())'`
gives torch 2.13.0+cu130, vLLM 0.31.0, `NVIDIA GB300`.

Pitfalls: `podman run` occasionally fails with `crun: sd-bus call: Remote peer disconnected` (rootless cgroup
setup through the user bus); re-running works. Pulls and pip need `http_proxy=http://fwdproxy:8080`.
`HF_HUB_OFFLINE=1` with the HF cache mounted at `/root/.cache/huggingface` keeps the engines from reaching out
(weights copied from other users' caches into `~/smg-perf/gpu/models/hub`).

## 2. Images and versions

| component | version |
|---|---|
| `docker.io/vllm/vllm-openai:latest` (arm64) | vLLM 0.31.0, torch 2.13.0+cu130, Python 3.12, grpcio 1.84, msgspec 0.22, pyzmq 27.2 |
| `docker.io/lmsysorg/sglang:latest` (arm64, 34 GB) | SGLang 0.5.21 |
| `localhost/smg-vllm:local` | the vLLM image plus `smg-grpc-proto`, `smg-grpc-servicer` and the `smg` binding wheel from this worktree (`image-ctx/Containerfile`, `scripts/build-image.sh`) |
| gateway | `smg 1.11.0` built from this branch into `~/.cargo/target-leap-gpu-harness/release/smg` |

The binding wheel is abi3 (`pyo3 abi3-py38`), built on the host with
`maturin build --release --features vendored-openssl --compatibility linux` (host glibc 2.34 is older than the
image's, so the wheel loads inside the container).

## 3. KV-event capture (deliverable 1)

`scripts/capture_kv_events.py` connects a SUB socket to the engine's publisher (optionally fetching the replay
buffer from seq 0 first) and appends one JSON line per batch with the three raw frames (`topic`, `seq`,
base64 msgpack payload). `scripts/summarize_kv_events.py` prints the field census; `scripts/workload_kv.py` drives
the scripted workload (token-id prompts, shared prefixes, concurrent sharers, 120 unique 1 k prompts through a
9 k-token cache, reset) and records the engine's own `cached_tokens` per request.

Engines were sized to evict: vLLM `--kv-cache-memory-bytes 1073741824` (585 blocks of 16), SGLang
`--max-total-tokens 9216 --page-size 16`. The reset is `POST /reset_prefix_cache` on vLLM (needs
`VLLM_SERVER_DEV_MODE=1`; there is no CLI flag for it in 0.31) and `POST /flush_cache` on SGLang. SGLang's
default attention backend on GB300 forces page size 64, so `--attention-backend triton` is used to keep
`--page-size 16` equal to vLLM's block size.

Commands:

```
scripts/run-vllm.sh vllm-smoke 0 8101 5557 Qwen/Qwen3-0.6B --max-model-len 4096 --max-num-seqs 16 \
    --kv-cache-memory-bytes 1073741824 --enforce-eager --gpu-memory-utilization 0.3
scripts/run-sglang.sh sglang-smoke 1 8201 5567 Qwen/Qwen3-0.6B --mem-fraction-static 0.3 \
    --max-total-tokens 9216 --max-running-requests 16 --context-length 4096 --attention-backend triton
venv/bin/python scripts/capture_kv_events.py --endpoint tcp://127.0.0.1:5557 --topic kv-events \
    --replay tcp://127.0.0.1:5558 --from-seq 0 --out fixtures/vllm/capture.jsonl --stop-file fixtures/vllm/.stop &
venv/bin/python scripts/workload_kv.py --base http://127.0.0.1:8101 --model Qwen/Qwen3-0.6B --out fixtures/vllm/workload.jsonl
touch fixtures/vllm/.stop; venv/bin/python scripts/summarize_kv_events.py fixtures/vllm/capture.jsonl
```

What the real streams contain (full census in the fixtures' `README.md` and `*-summary.json`):

| | vLLM 0.31.0 | SGLang 0.5.21 |
|---|---|---|
| envelope | `[ts, events, dp_rank=0]`, seq from 0, no gaps | same, seq from 0 (first batch = startup `AllBlocksCleared`, only in the replay buffer) |
| `BlockStored` keys | `block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium, lora_name, extra_keys, group_idx, kv_cache_spec_kind` | `block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium` |
| `BlockRemoved` keys | `block_hashes, medium, group_idx` | `block_hashes, medium` |
| `medium` | `"GPU"` always | `"GPU"` always (no HiCache) |
| `group_idx` / `kv_cache_spec_kind` | 0 / `full_attention` | absent |
| `locality`, `ownership`, `session_id`, `cache_salt`, `kv_cache_spec_sliding_window` | absent (omitted when None) | absent |
| `extra_keys` | list of nulls aligned with `block_hashes` | absent |
| hashes | u64, never negative | i64, half negative |
| removals | one event per hash, up to 256 per batch | one event per node, all its pages listed |
| `cached_tokens` | `prompt_tokens_details` always present; 624 for a fully cached 640-token prompt | `prompt_tokens_details` omitted when 0; concurrent sharers of a new prefix all miss |

## 4. Fleet through the Rust servicer (deliverable 2)

Image `localhost/smg-vllm:local` = the vLLM image plus `smg-grpc-proto` (built from `crates/grpc_client/python`;
its `smg_grpc_proto/proto` is a symlink to `../../proto`, which must be dereferenced in the build context),
`smg-grpc-servicer` (`grpc_servicer/`) and the `smg` binding wheel. Build the wheel on the host first
(`scripts/build-wheel.sh`), then `scripts/build-image.sh` (`podman build --network host` so pip reaches the proxy).

Eight workers, two per GPU (`scripts/launch-fleet.sh Qwen/Qwen3-0.6B 8 --max-model-len 4096 --max-num-seqs 64
--gpu-memory-utilization 0.4`; each is `scripts/run-vllm-grpc.sh`: `vllm serve <model> --grpc --port <p>
--servicer-impl rust --kv-events-config {...zmq...} --prefix-caching-hash-algo sha256_cbor --block-size 16
--enable-prompt-tokens-details`). Startup: ~2 min (torch.compile 37 s, CUDA graphs) until the servicer logs
`Engine connected; the servicer is SERVING` and `SubscribeKvEvents: connected to ZMQ endpoint`.

Gateway (`scripts/launch-gateway.sh <policy> 8 <tokenizer-dir>`): `smg launch --backend vllm --worker-urls
grpc://127.0.0.1:<p>... --policy cache_aware --model-path <tokenizer-dir> --port 30100 --prometheus-port 29100`.
The connection mode comes from the `grpc://` scheme; with `cache_aware` the gateway creates the KV event monitor
and subscribes to every gRPC worker by itself (log lines `Starting KV event subscription`, `KV event stream
connected worker_url=... start_seq=0`, one per worker; the model's block size is learned from the first event).
`GET /workers` lists the eight workers as healthy with `connection_mode: grpc`.

T4 ground truth, by hand (`scripts/workload_kv.py --text --stream` through the gateway at `--log-level debug`,
`scripts/t4_compare.py` joins the engine's `prompt_tokens_details.cached_tokens` with the gateway's per-request
decision lines; `~/smg-perf/gpu/results/t4-cache_aware-*`):

| phase | routing decision (gateway debug log) | engine `cached_tokens` | router-implied overlap |
|---|---|---|---|
| shared, request 0 (new 512-token prefix) | `no overlap, expected-wait fallback` -> worker 50051 | 0 | 0 |
| shared, requests 1-7 (same prefix, new suffixes) | `overlap match branch=event_hit` -> 50051 | 512 each | 512 |
| repeat (the same 8 prompts again) | `event_hit` -> 50051 | 624 each (640-token prompt, last block recomputed) | 624 |
| burst (8 concurrent prompts on a new prefix) | no overlap yet -> spread over 8 workers | 0 x7, 512 once (two landed on 50054, the second hit the in-flight prefill) | n/a |
| recheck (shared prompts again) | `event_hit` -> 50051 | 624 each | 624 |

Agreement 24/24 on the sequential phases: every routing decision that claimed overlap landed on a worker where
the engine then reported exactly the block-floored prefix. The gateway exposes the engine's count only on
streaming completions (`stream_options.include_usage`) and chat completions; non-streaming `/v1/completions`
responses omit `prompt_tokens_details` on this path (the servicer does send `cached_tokens`). There is no
per-request overlap number in metrics or headers for gRPC workers (`x-smg-routed-worker-id` is HTTP-only;
`smg_cache_aware_match_ratio` is recorded by the tree path, not the event path), so the comparison needs the
debug log.

Qwen3-8B was not deployed: the time went into the gateway anomaly below; the scripts take the model id and
`--gpu-memory-utilization 0.4` leaves room for two 8B instances per GPU (16 GB weights each).

## 5. Baseline (deliverable 3)

Workload: prefix repetition, 16 prefixes x 32 prompts, 1024 prefix + 128 suffix tokens, 64 output tokens,
Poisson 16 req/s, concurrency 64, 512 requests (`scripts/run-bench.sh` = `vllm bench serve --dataset-name
prefix_repetition` from the vLLM image; `scripts/bench_prefix.py` = the same shape without a container).

| run | client | TTFT p50 / p90 / p99 (ms) | TPOT p50 / p99 (ms) | E2E p50 (ms) | req/s |
|---|---|---|---|---|---|
| gateway cache_aware (first run of the session) | vllm bench | 26.4 / 42.8 / 154.3 | 2.51 / 5.39 | 187 | 15.9 |
| gateway round_robin | vllm bench | 1324 / 2641 / 3084 | 0.00 / 3.03 | 1331 | 15.3 |
| gateway round_robin, repeat | vllm bench | 3733 / 5785 / 6403 | 0.00 / 4.69 | 3750 | 13.5 |
| gateway round_robin | bench_prefix.py | 1567 / 3291 / 4180 | 0.04 / 8.4 | 1617 | 16.2 |
| gateway random | bench_prefix.py | 1222 / 2730 / 3723 | 0.02 / 4.6 | 1352 | 14.2 (72 x HTTP 500) |
| gateway cache_aware (later run) | bench_prefix.py | 732 / 3082 / 3698 | 0.04 / 4.1 | 801 | 16.0 (28 x HTTP 500) |
| one plain vLLM HTTP instance, same 16 req/s | bench_prefix.py | 28.5 / 44.1 / 56.9 | 1.99 / 4.06 | 155 | 17.0 |
| gateway cache_aware, **warn** level, 4 workers w0-w3, cores 64-71 | bench_prefix.py | 18.4 / 22.4 / 28.1 | 1.49 / 1.93 | 113 | 17.0 |
| gateway round_robin, **warn** level, 4 workers | bench_prefix.py | 17.6 / 20.8 / 27.0 | 1.47 / 1.85 | 110 | 17.0 |
| gateway round_robin, info level, same 4 workers | bench_prefix.py | 333 / 910 / 1744 | 1.44 / 22.6 | 463 | 16.9 |
| one plain vLLM HTTP instance, all-unique prompts, 2 req/s | vllm bench | 23.5 / - / 33.5 | 1.68 / 2.08 | 129 | 2.0 |

Resolved the same day: the stall is the gateway's per-request info logging written to a file (warn level or stdout to /dev/null gives 18-31 ms TTFT and smooth streaming on the same workers; table in the anomaly note). The harness runs the gateway at `--log-level warn` for every measurement; the info-level rows above are kept as the record of that defect. Starvation was tested and ruled out first (gateway and clients on the idle cores 64-71, scheduler wait time 0 ms, 8 or 72 runtime workers, round robin and cache_aware, HTTP and gRPC workers all show the same stall; details and a syscall timeline in `~/smg-perf/gpu/results/gateway-stream-anomaly.md`). The gateway and the load clients now run on cores 64-71 for every GPU-side run (`launch-gateway.sh`, `GATEWAY_CPUS`), engines and builds on 72-143. Read with that note: the direct gRPC path to any of the eight
servicers answers a 2048-token prompt in 15-41 ms and streams at 1.4 ms/token, while the gateway's HTTP path
adds 30-600 ms and often delivers the whole response as one burst (`smg_router_request_duration_seconds` mean
3.6 ms vs `smg_http_request_duration_seconds` mean 140 ms over the same requests). The first cache_aware run is
the only gateway run that streamed normally; the round-robin runs never did. So the only numbers above that
qualify as an engine-truth reference for T9 are the direct-engine rows and the direct-servicer probes; the
gateway rows record the state of the HTTP path on this branch and must be re-taken once it is fixed. The HTTP
500s are `Tokenizer not found for model` during the first seconds after the gateway reports its workers
healthy: wait for the `Tokenizer '<model>' ... registered` log line before loading it.

## 6. Pitfalls collected

- Keep worker ports below 32768: `net.ipv4.ip_local_port_range` is 32768-65535 here, and two servicers failed
  with `failed to bind 0.0.0.0:50056: Address already in use` because an outgoing connection of another
  process already held the port. The fleet scripts default to 20051+i (gRPC), 21051+i (handshake), 5600+2i (ZMQ).
- `pkill -f` matching your own command line kills your shell; use a pid file (`launch-gateway.sh` + `gateway.pid`).
- Rootless podman wedges under host load: `podman ps`/`run`/`rm` hung for minutes while the containers kept
  running; `timeout` on an attached `podman run` leaves the container running. Run benchmarks detached with a
  log file, or without containers (`bench_prefix.py`, `probe_grpc.py` from the venv with `grpcio` and the local
  `smg-grpc-proto`).
- Each container has its own torch.compile cache (`/root/.cache/vllm`); a ninth instance recompiled for 285 s
  under load. Mount a shared cache directory if startup time matters.
- The gateway rejects token-id prompts (`prompt: data did not match any variant of untagged enum StringOrArray`);
  send text and size it with the tokenizer (`workload_kv.py --text`).
- SGLang omits `prompt_tokens_details` when nothing was cached; vLLM always sends it (0).
- `nvidia-smi topo -m` puts GPUs 0/1 on cores 0-17 and GPUs 2/3 on 18-33; our cpuset 72-143 is remote to all
  four, which did not matter for these runs (SM utilisation stayed near zero).

## 7. What is running / where things are

- `~/smg-perf/gpu/scripts`: everything above; `fixtures/{vllm,sglang}`: raw captures and summaries;
  `results/`: bench JSONs, T4 table, anomaly note; `logs/`: container and gateway logs; `models/hub`: copies
  of Qwen3-0.6B and Qwen3-8B; `wheels/`: the binding wheel; `image-ctx/`: the image build context.
- Containers `vllm-w0..w7` (fleet), `vllm-http` (plain HTTP instance on GPU 0, port 8102) and the gateway
  (`logs/gateway.pid`) were left running for the next workstream; `podman rm -f vllm-w0 ... vllm-http` and
  `kill $(cat logs/gateway.pid)` stop them.
