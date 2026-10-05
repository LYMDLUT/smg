# Mooncake replay, Qwen3-8B fleet with the KV pool restricted to 12 000 blocks per worker (192 k tokens; fleet 768 k = 7 % of the rows 0-1999 working set of 11.02 M tokens), f4dc134b binaries, 2026-10-05

Same trace, speedup, gateway flags and workers as the full-pool table; the gateway was **not** restarted between the three runs of a policy (its index and placement state carried over), which matters for cache_aware, see below.

| run | ok / errors | req/s | goodput req/s | within SLO | strict | prefix reuse | TTFT mean / p50 / p90 / p99 (ms) | mean ITL mean / p90 / p99 (ms) | e2e p50 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| round_robin run 1 | 1950 / 50 | 16.73 | 7.57 | 45.2 % | 20.9 % | 0.331 | 4011 / 542 / 12874 / 21378 | 25.0 / 47.2 / 224.5 | 3827 / 26414 |
| round_robin run 2 | 1950 / 50 | 15.90 | 8.01 | 50.4 % | 24.9 % | 0.343 | 2569 / 348 / 7745 / 15721 | 29.3 / 77.6 / 222.0 | 3015 / 22303 |
| round_robin run 3 | 1950 / 50 | 16.44 | 10.20 | 62.1 % | 28.8 % | 0.361 | 1056 / 267 / 3766 / 7377 | 35.1 / 99.0 / 225.4 | 1884 / 13572 |
| cache_aware run 1 | 1950 / 50 | 16.12 | 7.93 | 49.2 % | 32.8 % | 0.367 | 3889 / 484 / 12103 / 19753 | 28.1 / 58.9 / 256.5 | 3668 / 25221 |
| cache_aware run 2 | 1950 / 50 | 15.68 | 5.52 | 35.2 % | 24.7 % | 0.365 | 3855 / 996 / 12758 / 20829 | 35.3 / 88.1 / 224.1 | 3294 / 28073 |
| cache_aware run 3 | 1950 / 50 | 15.17 | 1.09 | 7.2 % | 4.3 % | 0.374 | 4812 / 3746 / 10502 / 16853 | 135.6 / 232.2 / 4260.4 | 8925 / 20908 |
| **round_robin mean of 3** | 1950 / 50 | 16.36 | 8.59 | 52.5 % | 24.8 % | 0.345 | 2545 / 385 / 8128 / 14826 | 29.8 / 74.6 / 224.0 | 2909 / 20763 |
| **cache_aware mean of 3** | 1950 / 50 | 15.66 | 4.84 | 30.5 % | 20.6 % | 0.369 | 4185 / 1742 / 11788 / 19145 | 66.3 / 126.4 / 1580.3 | 5296 / 24734 |

Reading: with the pool at 7 % of the working set both policies lose about half their goodput against the full pool (14.5 / 13.9 req/s there). Round robin is stable to slightly improving run over run (7.6, 8.0, 10.2 req/s). cache_aware starts level with it (7.9) and then collapses (5.5, then 1.1 req/s with TTFT p50 3.7 s) while its reported prefix reuse stays at 0.37: the gateway keeps routing repeats to the worker its index credits, but with a 192 k-token pool those blocks are long evicted (vLLM announces a removal only when the block is reused, so the index lags) and the credited worker queues. This is the setting where the T9 agreement check and the policy work have something to separate; the d2179cd0 rerun restarts the gateway between runs to split state carry-over from the policy itself.
