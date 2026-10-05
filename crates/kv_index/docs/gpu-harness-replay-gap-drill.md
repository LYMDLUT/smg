# Replay-gap drill on the pushed head b4943d69 (per-rank stream recovery, KV-event recovery counters)

Setup: gateway built from b4943d69 (`cache_aware`, `--log-level info` to a file, health probe every 20 s, timeout 30 s,
threshold 5, cores 64-71, port 30400) over the four 0.6B drill workers (host venv, 2d37b25a servicer wheel, gRPC 20071-20074,
`--max-model-len 4096`, 0.3 of a GPU each). Load: `bench_prefix.py` at 8 req/s for 2400 prompts (~293 s), 16 prefixes of
1024 tokens + 128-token suffixes, 32 output tokens, concurrency 32, per-request rows with the engine's `cached_tokens`.
At bench+40 s worker `drill-w1` (20072) was killed by its pid file and relaunched on the same ports, so its process, its
servicer and its KV publisher restarted (sequence back to 0). Gateway `/metrics` scraped every 2 s (`metrics.txt`).
Metric names read: `smg_kv_event_batches_total{worker,disposition}`, `smg_kv_event_gaps_total`, `smg_kv_event_missed_batches_total`,
`smg_kv_event_resyncs_total{worker,reason}`, `smg_kv_event_lag_seconds`, `smg_kv_event_degraded_ranks`, `smg_kv_event_tail_depth`,
`smg_kv_event_subscription_failures_total`.

## Timeline (pass 2, `b4943d69-pass2/`)

| event | wall clock | relative |
|---|---|---|
| kill of drill-w1 | 17:49:36 | bench+40.0 s |
| gateway: `KV event stream error, reconnecting` for 20072, then `Failed to subscribe ... retrying` (tcp connect refused) | 17:49:36-17:49:42 | kill+0-6 s |
| gateway: `KV event stream connected worker_url=...20072 start_seq=233` (the new servicer process answers before its engine is up; the relay has no replay buffer, the cursor is just carried over) | 17:49:47 | kill+11 s |
| health probes fail for 20072 (20 s cadence) | 17:49:52, 17:50:12, 17:50:32 | |
| worker back: `Engine connected; the servicer is SERVING` | 17:50:51 | kill+75.5 s |
| first batch from the restarted publisher reaches the gateway: `KV event publisher restarted; clearing the worker's index state worker_url=...20072 rank=0 received=0`; `smg_kv_event_resyncs_total{worker="grpc://127.0.0.1:20072",reason="publisher_restart"}` 0 -> 1 | 17:51:37 | kill+133.7 s, SERVING+59.1 s |
| applied batches for 20072 after the resync | until bench end | 234 -> 384 (+150); the other three workers applied 1200-1565 each over the run |

Time to detection is one batch: the restart is recognised on the first event the new publisher sends (the sequence
number goes backwards from the carried cursor 233 to 0), with no `gaps`, `missed_batches` or `degraded_ranks` counted
(`smg_kv_event_gaps_total` and `smg_kv_event_missed_batches_total` stayed 0). What takes time is not the detection but the
first event: the worker spends 75 s restarting, and after it is SERVING another 59 s pass before it produces an event,
because the gateway only routes new or spilled work to a worker whose index is empty and the health probe re-admits it on
its 20 s cadence. Engine-truth hit fraction (cached_tokens > 0, per 10 s bucket) stayed 0.96-1.00 across the outage and
after the resync (the 16 prefixes were cached on the other three workers; one bucket at 0.99 when the returned worker took
its first cold requests); TTFT p50 stayed 17-19 ms throughout; 1 of 2400 requests failed (an over-length 400, unrelated).
`oracle` is empty on hardware; `cached_tokens` is the truth.

Pass 1 (`b4943d69/`, kept): bench 89 s, kill at +25 s, default 60 s health probe; the worker needed 126 s to come back,
so the sequence restart fell after the load and no resync was counted inside the scrape; the 70-80 s bucket showed TTFT
p50 17.9 s because requests kept being routed to the dead worker until the 60 s probe excluded it. Lesson for the recipe:
keep the probe at 20 s (the published launcher does) and make the load window longer than the worker's restart.

Reproduce: `scripts/gap-drill.sh <label>` then `venv/bin/python scripts/gap-drill-analyze.py results/gap-drill/<label>`.
