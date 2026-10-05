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

## Soak (guardrail 3)

`~/smg-perf/replay/soak.sh --label L --hours 24` keeps one gateway and one mock
fleet up for the whole run and replays the trace window by window (one
measurement-lock acquisition per window, released between windows), while:

- `soak-faults.py` fires the mock's fault hooks on a schedule, cycling over
  the workers: drop 20 batches, 1000 ms publishing delay for 60 s, publisher
  restart, pause 30 s then resume, and a worker restart (cache reset plus
  publisher restart, what the index sees when an engine restarts);
- `soak-sampler.py` appends one row per minute to `samples.csv`: gateway RSS,
  its cache-aware branch counters, match-ratio mean, engine cache-hit gauge
  and KV-subscription failures from `/metrics`, and from the mock's admin API
  the last minute's requests, hit/oracle, prefix reuse, per-worker balance,
  preemptions and KV batches.

`soak-report.py DIR` prints RSS at hour 1 and at the end with the ratio the
guardrail asks for (within 5%), hit/oracle, reuse and balance in the five
minutes before and after each fault, the counters, and the per-window table.
The run is restartable: `--resume` continues the window position and the fault
cycle in a new segment (a restarted gateway is a new RSS baseline).
