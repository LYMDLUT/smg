# Mooncake replay, 8B fleet with 12 000-block pools (192 k tokens/worker), gateway d2179cd0 (tier-aware monitor, run index), fresh gateway per run, warm engines, servicer wheel f4dc134b

| run | ok / errors | req/s | goodput req/s | within SLO | strict | prefix reuse | TTFT mean / p50 / p90 / p99 (ms) | mean ITL mean / p90 / p99 (ms) | e2e p50 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| cache_aware run 2 | 1950 / 50 | 12.28 | 2.31 | 18.8 % | 12.6 % | 0.360 | 15000 / 15682 / 31170 / 46499 | 89.5 / 190.1 / 1520.3 | 20186 / 53758 |
| cache_aware run 3 | 1950 / 50 | 13.96 | 7.80 | 55.8 % | 42.8 % | 0.371 | 2942 / 292 / 10588 / 20070 | 17.2 / 32.6 / 180.8 | 1475 / 26156 |
| round_robin run 2 | 1950 / 50 | 16.68 | 13.64 | 81.8 % | 38.1 % | 0.371 | 286 / 222 / 488 / 2143 | 23.5 / 50.6 / 219.1 | 695 / 7358 |
| round_robin run 3 | 1950 / 50 | 16.46 | 9.07 | 55.1 % | 24.6 % | 0.349 | 2382 / 305 / 7878 / 13356 | 26.5 / 53.4 / 218.3 | 3002 / 18393 |
| **cache_aware mean of 2** | 1950 / 50 | 13.12 | 5.05 | 37.3 % | 27.7 % | 0.365 | 8971 / 7987 / 20879 / 33284 | 53.4 / 111.4 / 850.6 | 10831 / 39957 |
| **round_robin mean of 2** | 1950 / 50 | 16.57 | 11.35 | 68.4 % | 31.3 % | 0.360 | 1334 / 263 / 4183 / 7750 | 25.0 / 52.0 / 218.7 | 1849 / 12875 |

Reading: on warm engines round robin holds the full-pool numbers even at 7 % of the working set (goodput 13.6 req/s, within-SLO 82 %), while cache_aware is far worse and erratic (goodput 8.0 / 2.3 / 7.8 req/s, TTFT p50 0.3-15.7 s) at the same prefix reuse (0.37): with 192 k-token pools and prompts up to 40 k tokens, concentrating repeats on the credited worker queues it (TTFT p90 10-16 s) while the credit itself is stale (vLLM announces a removal only when the block is reused: 180 k-590 k BlockRemoved per worker per run against 2 k-21 k stores). The earlier f4dc134b restricted series ran on cold engines right after the relaunch and is not comparable run for run.
