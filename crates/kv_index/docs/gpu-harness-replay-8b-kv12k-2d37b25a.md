# Mooncake replay, 8B fleet with 12 000-block pools (192 k tokens/worker, fleet 7 % of the working set), gateway and replay built from 2d37b25a, fresh gateway per run, warm engines, servicer wheel f4dc134b (the 2d37b25a wheel was still building)

| run | ok / errors | req/s | goodput req/s | within SLO | strict | prefix reuse | TTFT mean / p50 / p90 / p99 (ms) | mean ITL mean / p90 / p99 (ms) | e2e p50 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| cache_aware run 1 | 1950 / 50 | 13.88 | 8.61 | 62.1 % | 39.5 % | 0.365 | 2688 / 253 / 10253 / 24217 | 20.0 / 32.8 / 196.4 | 1805 / 30568 |
| cache_aware run 2 | 1950 / 50 | 14.21 | 5.01 | 35.2 % | 22.6 % | 0.361 | 3861 / 1171 / 12056 / 24442 | 36.3 / 127.6 / 198.5 | 3781 / 27484 |
| cache_aware run 3 | 1950 / 50 | 16.31 | 7.51 | 46.1 % | 36.9 % | 0.371 | 5645 / 1038 / 17342 / 24777 | 21.9 / 37.9 / 199.5 | 5212 / 32130 |
| round_robin run 1 | 1950 / 50 | 16.69 | 14.24 | 85.3 % | 40.2 % | 0.371 | 237 / 200 / 427 / 870 | 21.0 / 42.1 / 209.4 | 644 / 7530 |
| round_robin run 2 | 1950 / 50 | 16.68 | 12.77 | 76.6 % | 36.7 % | 0.362 | 407 / 245 / 771 / 3892 | 22.9 / 45.9 / 215.5 | 791 / 9484 |
| round_robin run 3 | 1950 / 50 | 16.42 | 8.62 | 52.5 % | 22.3 % | 0.357 | 1280 / 291 / 4315 / 9599 | 36.8 / 117.7 / 243.9 | 2338 / 13936 |
| **cache_aware mean of 3** | 1950 / 50 | 14.80 | 7.04 | 47.8 % | 33.0 % | 0.366 | 4065 / 821 / 13217 / 24479 | 26.1 / 66.1 / 198.1 | 3599 / 30061 |
| **round_robin mean of 3** | 1950 / 50 | 16.60 | 11.88 | 71.5 % | 33.0 % | 0.363 | 641 / 245 / 1838 / 4787 | 26.9 / 68.5 / 223.0 | 1258 / 10317 |

Reading: the picture of the d2179cd0 table holds on 2d37b25a. Round robin keeps most of its full-pool goodput (12.8-14.2 req/s, within-SLO 77-85 %, TTFT p50 200-245 ms) while cache_aware delivers 5.0-8.6 req/s with TTFT p50 0.25-1.5 s and p90 10-17 s at the same prefix reuse (0.36-0.37). With 192 k-token pools and prompts up to 40 k tokens the credited worker is the bottleneck: cache_aware keeps sending a prefix's repeats to the worker whose pool has long evicted them (vLLM publishes a removal only when the block is reused again, 180 k-590 k removals per worker per run), so the hit does not materialise and the queue does. A capacity- or load-aware fallback at low pool-to-working-set ratios is what the policy work needs to add; at the full pool (676 k tokens/worker) cache_aware led by 0.6 req/s goodput and 4 points of within-SLO.
