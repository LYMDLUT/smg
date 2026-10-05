# cache_aware with worker protection / spill at the restricted pool (hardware twin of the mock matrix rows)

8B fleet with 12 000-block pools (192 k tokens per worker, 7 % of the rows 0-1999 working set), 2d37b25a gateway and servicer
wheel, Mooncake rows 0-1999 at speedup 3, fresh gateway before every run, warm engines, gateway and client on cores 64-71,
`--log-level warn --reasoning-parser passthrough`, health probe 30 s / 20 s / 5. Rows are named as in the mock matrix.
`shed` is `smg_worker_overload_shed_total` scraped from the gateway right after each run (before its restart); it counts requests rejected because every worker was over the threshold, and the replay counts those as errors (ca+prot run 1: 627 errors = 50 over-length + 577 shed; ca+prot-tu0.8 run 1: 64 = 50 + 14), which is why goodput drops with the shed count. Exclusions that merely re-route a request do not increment it. The gateway's
load monitor polls `GetLoads` on each servicer; the 2d37b25a vLLM servicer fills `token_usage`, `num_running_reqs` and
`num_waiting_reqs`, so every row had its signal. Table: `protection-table.md` (per run and mean of 3).

| row (mean of 3) | goodput req/s | within SLO | TTFT p50 / p90 / p99 ms | reuse | shed |
|---|---|---|---|---|---|
| ca+prot (`--worker-overload-protection`, KV 0.9) | 6.83 | 49.6 % | 2922 / 17919 / 26774 | 0.355 | 588, 0, 0 |
| ca+prot-tu0.8 | 10.37 | 66.0 % | 272 / 9390 / 13105 | 0.367 | 16, 0, 0 |
| ca+prot-wq8 | 12.92 | 79.7 % | 224 / 1158 / 6668 | 0.374 | 0 |
| ca+spill (abs 4, rel 1.25) | 11.24 | 70.2 % | 441 / 8271 / 14588 | 0.370 | 0 |
| ca+kvtrig (overload 0.9, balance 0.5) | 11.30 | 68.8 % | 357 / 5876 / 11154 | 0.360 | 0 |
| round_robin | 13.40 | 80.6 % | 223 / 613 / 2968 | 0.367 | 0 |
| plain cache_aware (previous round, same fleet) | 7.04 | 47.8 % | 821 / 13217 / 24479 | 0.366 | - |

Reading. Every guard helps against plain cache_aware (7.0 req/s), none beats round robin at this pool size, and the two
waiting-queue-shaped guards are the ones that close the gap: `ca+prot-wq8` (exclude a worker at 8 queued requests) reaches
12.9 req/s / 79.7 % within-SLO with TTFT p50 224 ms, run after run (13.3 / 12.4 / 13.1), and the spill gate at abs 4 / rel 1.25
and the KV triggers are as good as round robin in two runs of three (14.4 / 13.3 and 13.8 / 13.3) and collapse in the third
(6.0 and 6.8, TTFT p90 17-23 s). The KV-usage ceilings are the weak ones here: at 0.9 the pools of a 192 k-token worker sit
near the ceiling most of the time, so run 1 shed 588 requests and still queued (1.8 req/s, TTFT p50 8.2 s), and runs 2-3 shed
nothing and behaved like plain cache_aware with a better tail; at 0.8 one run shed 16 and the other two reached 11.7-12.0.
The signal that predicts the queueing on this fleet is the waiting count, not KV usage: with 40 k-token prompts a worker is
saturated by a handful of requests long before its KV usage reads 0.9, and KV usage barely moves between hot and cold
workers (all pools are full), so the usage-spread trigger (balance 0.5) rarely fires. Prefix reuse is the same for every
row (0.35-0.37): the guards change where the queue forms, not how much is cached. The strict SLO column (per-request p99
ITL under 50 ms) favours the cache-aware rows (37-50 %) over round robin (37 %) because fewer of their requests share a
decode batch with cold prefills.
Variance: one collapsed run in ca+prot, ca+spill and ca+kvtrig each; the collapsed runs are the ones where the first few
seconds concentrated the 40 k-token prompts on one worker before any signal crossed its threshold.
