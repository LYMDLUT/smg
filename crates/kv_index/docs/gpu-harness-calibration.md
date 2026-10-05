# Engine timing calibration: Qwen3-8B on one GB300 (vLLM 0.31.0, Rust servicer, no gateway)

`calibration.json` is what the mock engine's `--timing` mode should read. Produced by `scripts/calibrate_engine.py`
against worker `8b-w0` (gRPC 20061) warm, token-id prompts of fresh random ids (no prefix hits), two passes
(`*-pass1.json`, `*.json` = pass 2 with steady-window decode and more prefill sizes).

- `capacity`: KV tokens/blocks at the fleet's settings (`--gpu-memory-utilization 0.4 --max-model-len 40960`:
  676 128 tokens = 42 258 blocks, 92.85 GiB, ~144 KiB per token), the 0.3 reference, the HTTP reference, and
  the restricted pool used for the eviction replay (`--num-gpu-blocks-override 12000` = 192 000 tokens).
- `prefill_ms_minus_overhead`: per prompt size, median (and min) time-to-first-chunk of a single request minus
  the 1-token prompt's TTFC (fixed overhead), 5 repeats, and the least-squares `a + b*T + c*T^2`.
  Use the points: the curve is a plateau (~75 ms from 512 to 3072 tokens), a step at 4096, then linear
  (~13 ms per 1 k tokens beyond 8 k); the quadratic carries +-20 ms residuals.
- `decode_step_ms`: mean inter-token latency per setting (prompt 512 / 2048, concurrency 8 / 32 / 64 / 128,
  128 output tokens, `ignore_eos`), measured in the steady window after the last stream's first token and before
  the first stream's last token, plus fits `d + e*u + f*u^2` against KV utilisation `u` (tokens in use / capacity,
  computed), against batch size, and against KV tokens in use. Whole-stream means (`pass1`) include the
  prefill-interleaving of simultaneous starts and are higher.
- `notes`: the stalls, the servicer's empty usage fields, and the pass-to-pass drift.

Reproduce: `venv/bin/python scripts/calibrate_engine.py --target 127.0.0.1:20061 --kv-tokens 676128 --out out.json`
(needs `grpcio`, `numpy`, `smg-grpc-proto` in the venv; worker from `scripts/run-vllm-grpc-host.sh`).

## Added on the orchestrator's request: scheduler settings and batched prefill

- `scheduler-settings.json`: effective settings of the fleet workers and the HTTP reference (CLI values, vLLM 0.31 log
  lines, and the installed SchedulerConfig defaults, because 0.31 does not dump the scheduler config at startup):
  `max_num_seqs 128`, `max_num_batched_tokens 16384`, chunked prefill on, `long_prefill_token_threshold 0` (default,
  no derivation beyond the cap in 0.31), `max_model_len 40960`, block 16, `num_gpu_blocks` 42 258 / 29 669 / 12 000.
- `prefill-batch-sweep.json` and `calibration.json["batched_prefill_grpc_path"]`: 4/8/16 concurrent 1024-token and
  8 concurrent 4096-token fresh prompts, 3 repeats, first and last first-token time of the batch. Over the gRPC servicer
  the batch is admitted together: one ~78 ms pass for 4096 tokens, ~113 ms for 8192, ~200 ms for 16384, two passes
  (~335 ms) for 32768. The HTTP reference admitted one request per engine step (`engine_steps == N` from
  `vllm:iteration_tokens_total_count`), so its batch TTFTs (0.4-3.2 s) describe the HTTP front end, not the scheduler.
