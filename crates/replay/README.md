# replay

Replays a Mooncake-format trace (`{timestamp, input_length, output_length,
hash_ids}` per line, one `hash_id` per 512-token block) through the gateway and
scores routing quality end to end.

- Every `hash_id` becomes a deterministic text block (`--words-per-block`, 480
  words, about one token each), so rows that share ids share prompt prefixes
  after the gateway tokenizes them.
- Requests are sent open-loop at `timestamp / --speedup` as streaming chat
  completions with `stream_options.include_usage`, recording TTFT, inter-token
  latencies, end-to-end latency, the serving worker (`system_fingerprint`, which
  the gateway sets from the worker's `weight_version` label), and the
  engine-reported `cached_tokens`.
- With `--admin <mock admin url>` each request is joined with the mock fleet's
  record of it (`GET /admin/requests`), adding the arrival-time oracle (the most
  cached tokens any worker held when it arrived) and the queue wait.

Output: `summary.json` (mean/p50/p90/p99 TTFT, per-request mean ITL (TPOT)
distribution, e2e latency, goodput at the SLO `--slo-ttft-ms` / `--slo-itl-ms`
(default 500 ms TTFT and 50 ms per-request mean ITL; a strict variant uses the
per-request p99), prefix reuse = cached / prompt tokens, oracle prefix reuse,
hit-over-oracle, per-worker request and uncached-token counts, balance) and
`requests.csv` with one row per request.

```bash
replay --trace mooncake_trace.jsonl --gateway http://127.0.0.1:31000 \
  --model mock-model --speedup 4 --limit 5000 --admin http://127.0.0.1:31002 --out out/
```

The mock fleet it is meant for is `mock-worker --engine realistic --admin-port`;
see `crates/mock_worker/README.md` for the engine's scheduler, KV pool and
timing model (and the caveat that its timing polynomials were validated against
hardware only with prefix caching off).
