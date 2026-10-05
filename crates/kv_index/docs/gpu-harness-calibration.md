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
