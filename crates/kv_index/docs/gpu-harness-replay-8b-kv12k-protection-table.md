# cache_aware with worker protection / spill on the 12 000-block 8B fleet (2d37b25a gateway, 3 runs each, fresh gateway per run)

| row | flags | run | goodput req/s | within SLO | strict | TTFT p50 / p90 / p99 (ms) | prefix reuse | shed (`smg_worker_overload_shed_total`) | req/s |
|---|---|---|---|---|---|---|---|---|---|
| ca+prot | --worker-overload-protection (KV usage 0.9) | 1 | 1.81 | 19.8 % | 11.7 % | 8222 / 32392 / 38251 | 0.332 | 588 | 9.13 |
| ca+prot | --worker-overload-protection (KV usage 0.9) | 2 | 9.84 | 61.1 % | 29.5 % | 284 / 5107 / 9321 | 0.363 | 0 | 16.12 |
| ca+prot | --worker-overload-protection (KV usage 0.9) | 3 | 8.84 | 67.8 % | 38.1 % | 261 / 16258 / 32750 | 0.370 | 0 | 13.03 |
| ca+prot-tu0.8 | --worker-overload-protection --worker-overload-token-usage 0.8 | 1 | 7.40 | 51.5 % | 27.2 % | 338 / 22312 / 25370 | 0.359 | 16 | 14.37 |
| ca+prot-tu0.8 | --worker-overload-protection --worker-overload-token-usage 0.8 | 2 | 11.70 | 73.1 % | 38.8 % | 246 / 2748 / 5528 | 0.369 | 0 | 16.00 |
| ca+prot-tu0.8 | --worker-overload-protection --worker-overload-token-usage 0.8 | 3 | 12.01 | 73.3 % | 45.3 % | 232 / 3110 / 8417 | 0.374 | 0 | 16.39 |
| ca+prot-wq8 | --worker-overload-protection --worker-overload-waiting-requests 8 | 1 | 13.32 | 81.3 % | 45.1 % | 220 / 1201 / 9896 | 0.373 | 0 | 16.38 |
| ca+prot-wq8 | --worker-overload-protection --worker-overload-waiting-requests 8 | 2 | 12.38 | 76.6 % | 42.9 % | 224 / 1193 / 4378 | 0.373 | 0 | 16.16 |
| ca+prot-wq8 | --worker-overload-protection --worker-overload-waiting-requests 8 | 3 | 13.06 | 81.3 % | 46.7 % | 228 / 1079 / 5728 | 0.376 | 0 | 16.07 |
| ca+spill | --balance-abs-threshold 4 --balance-rel-threshold 1.25 | 1 | 14.41 | 87.6 % | 49.9 % | 184 / 498 / 2199 | 0.373 | 0 | 16.44 |
| ca+spill | --balance-abs-threshold 4 --balance-rel-threshold 1.25 | 2 | 13.33 | 81.0 % | 49.5 % | 176 / 966 / 4936 | 0.375 | 0 | 16.46 |
| ca+spill | --balance-abs-threshold 4 --balance-rel-threshold 1.25 | 3 | 5.98 | 41.9 % | 24.9 % | 964 / 23349 / 36628 | 0.363 | 0 | 14.28 |
| ca+kvtrig | --overload-token-usage-threshold 0.9 --balance-token-usage-threshold 0.5 | 1 | 13.76 | 82.8 % | 35.5 % | 227 / 466 / 1707 | 0.365 | 0 | 16.62 |
| ca+kvtrig | --overload-token-usage-threshold 0.9 --balance-token-usage-threshold 0.5 | 2 | 13.32 | 80.3 % | 34.6 % | 224 / 590 / 2251 | 0.363 | 0 | 16.58 |
| ca+kvtrig | --overload-token-usage-threshold 0.9 --balance-token-usage-threshold 0.5 | 3 | 6.81 | 43.3 % | 21.3 % | 620 / 16572 / 29503 | 0.352 | 0 | 15.74 |
| round_robin | round robin | 1 | 13.43 | 80.5 % | 36.5 % | 239 / 562 / 4460 | 0.368 | 0 | 16.70 |
| round_robin | round robin | 2 | 14.33 | 85.9 % | 39.4 % | 188 / 434 / 995 | 0.368 | 0 | 16.68 |
| round_robin | round robin | 3 | 12.44 | 75.4 % | 34.4 % | 242 / 843 / 3448 | 0.366 | 0 | 16.50 |
| **ca+prot** | --worker-overload-protection (KV usage 0.9) | mean of 3 | **6.83** | **49.6 %** | 26.4 % | 2922 / 17919 / 26774 | 0.355 | 196 | 12.76 |
| **ca+prot-tu0.8** | --worker-overload-protection --worker-overload-token-usage 0.8 | mean of 3 | **10.37** | **66.0 %** | 37.1 % | 272 / 9390 / 13105 | 0.367 | 5 | 15.58 |
| **ca+prot-wq8** | --worker-overload-protection --worker-overload-waiting-requests 8 | mean of 3 | **12.92** | **79.7 %** | 44.9 % | 224 / 1158 / 6668 | 0.374 | 0 | 16.20 |
| **ca+spill** | --balance-abs-threshold 4 --balance-rel-threshold 1.25 | mean of 3 | **11.24** | **70.2 %** | 41.5 % | 441 / 8271 / 14588 | 0.370 | 0 | 15.73 |
| **ca+kvtrig** | --overload-token-usage-threshold 0.9 --balance-token-usage-threshold 0.5 | mean of 3 | **11.30** | **68.8 %** | 30.5 % | 357 / 5876 / 11154 | 0.360 | 0 | 16.32 |
| **round_robin** | round robin | mean of 3 | **13.40** | **80.6 %** | 36.8 % | 223 / 613 / 2968 | 0.367 | 0 | 16.62 |
