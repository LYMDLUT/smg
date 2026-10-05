# KV event recovery: what the gateway does and what the wire protocol needs

Companion to `kv-router-leap.md` section 4.4. The gateway side (per-rank cursors, bounded gap
recovery, publisher-restart detection, the snapshot tail buffer) lives in
`model_gateway/src/worker/kv_event_recovery.rs` and is driven from
`model_gateway/src/worker/kv_event_monitor.rs`. This note records the wire semantics the gateway
relies on, the state machine, the fault drills, and the additions the servicers and
`crates/grpc_client/proto/common.proto` need so recovery converges to an exact index instead of a
best-effort one. It is written for the schema workstream to fold into its proto change; nothing in
the "proposed" section is implemented on the server yet.

## Wire semantics today

`SubscribeKvEventsRequest.start_sequence_number` is the **last sequence the client applied**, not
the first one it wants: the SGLang servicer asks its engine's replay socket for `cursor + 1`, the
mock engine streams buffered batches `> cursor`, and `0` means "no cursor, live only". Two
consequences shape the gateway's rules:

- a server never legitimately sends a sequence below the cursor as the first batch of a fresh
  connection, so one there is a restarted publisher (vLLM and SGLang count from 0 per process);
- a cursor of exactly 0 is indistinguishable from "no cursor". A rank whose only applied batch was
  sequence 0 resubscribes live, and a batch lost during that reconnect is settled as a gap.

| Server | `start_sequence_number != 0` | History gone | Snapshot | Ranks |
|---|---|---|---|---|
| Rust `engine_servicer` relay (vLLM, SGLang, TokenSpeed) | ignored: streams live from the publisher's current position | `DATA_LOSS` when the publisher's sequence goes backwards (a restart); gaps never signalled | none | relays rank 0 only |
| Python servicers (SGLang, vLLM) | replay when the engine offers one; `OUT_OF_RANGE` when it cannot honour a non-zero start | `DATA_LOSS` on a publisher restart or an unverifiable replay | none ("zero starts live rebuilding") | every DP rank, `dp_rank` set, a cursor per rank |
| `mock-worker --engine realistic` | replays its buffer `> cursor`, then live | never signalled (the publisher-restart fault hook keeps the stream up) | none | rank 0 |

Engine facts the rules lean on (`~/smg-perf/research/{vllm,sglang}-kv-events.md`): both
publishers number batches from 0 per process and restart at 0; SGLang's first batch after startup
carries `AllBlocksCleared`, vLLM's does not; SGLang's replay socket silently truncates large
replays at libzmq's default send high-water mark (1,000 frames) and has no snapshot; an idle engine
publishes nothing, so a quiet stream is not a dead one.

## The gateway's state machine

One `RankState` (cursor, pending replay, degraded flag, snapshot tail) per `(worker, dp_rank)`,
all of them over the worker's one `WorkerIndexState` (block map, physical copies per tier,
counters), which stays pooled across ranks as the monitor already does: the worker URL is the
routing target and its copies are interchangeable for a prefix hit. Cursors and copy counts never
mix, and nothing about blocks is tracked twice.

| Batch sequence `seq` against the rank's cursor `last` | Decision |
|---|---|
| no cursor yet | apply, cursor = `seq` |
| `last + 1` | apply |
| `<= last`, and the batch carries the engine's own `Cleared`, or it is the first batch on a fresh connection with `seq < last`, or `seq` is 0 or 1 under a larger cursor, or `seq + 1024 <= last` | **publisher restart**: clear the worker's index state, start the other ranks' cursors over, apply, cursor = `seq` |
| `<= last` otherwise | duplicate (replay overlap): skip |
| `> last + 1`, no replay pending | **gap**: remember `expected = last + 1`, reconnect with the cursor; the batch is not applied |
| `> last + 1`, replay pending | **unrecoverable gap**: the server skipped ahead anyway. `missed <= 1024`: keep the blocks, mark the rank degraded (an engine still holds most of them and never re-sends stores); more: clear the worker. Apply, cursor = `seq` |
| snapshot in flight | hold in the rank's tail (bounded at 1,024 batches; overflow forces another snapshot) |

A restart on one rank clears the whole worker because the pooled copies cannot be attributed to a
rank; the other ranks' cursors start over so their next batch is taken as a first one instead of
clearing the index a second time. `OUT_OF_RANGE` / `DATA_LOSS` from the server do the same for
every rank and resubscribe live. Worker removal hands the pooled state to one `remove_worker`
pass. A reconnect sends rank 0's cursor (the servicers replay rank 0); other ranks dedup what
arrives. The reconnect backoff caps at 5 s: a worker that restarts is healthy again within seconds,
and until its stream is back the blocks it stores are invisible to routing, because the servicers
resume after the cursor and never resend them.

Every decision is a metric: `smg_kv_event_batches_total{disposition}` (applied, stale,
tail_overflow), `smg_kv_event_gaps_total{outcome}` (replay_requested, unrecovered_kept,
unrecovered_cleared), `smg_kv_event_missed_batches_total`, `smg_kv_event_resyncs_total{reason}`
(out_of_range, data_loss, publisher_restart, gap_cleared), `smg_kv_event_lag_seconds` (publisher
stamp to apply), `smg_kv_event_degraded_ranks`, `smg_kv_event_tail_depth`.

What the gateway still cannot do without server help: learn which blocks changed during an
unreplayable gap, replace a worker's state atomically, tell a restarted publisher that kept its
cache from one that lost it, or know a worker's resident blocks after a gateway restart. All of
these need an epoch or a snapshot.

## Fault drills

`~/smg-perf/chaos/` runs the contract's section 5 against a fleet of `mock-worker --engine
realistic` gRPC workers that publish KV events, with health checks on. **No root is needed**: a
crash is `SIGKILL`, a frozen worker is `SIGSTOP`/`SIGCONT`, a partition or an unreachable peer is a
user-space TCP proxy (`tcp-proxy.py`) between gateway and worker switched to `blackhole` or
`refuse`, CPU starvation is busy loops pinned to the worker's cores, overload is eight times the
harness's stream count, and the gateway restart is a kill plus re-registration. Four more drills
use the mock engine's admin fault hooks (`crates/mock_worker/README.md`:
`POST /admin/fault/{worker}/drop?batches=N`, `delay?ms=D`, `restart-publisher`, `pause`,
`resume`) and skip when the build lacks them. Each drill prints `RESULT: PASS|FAIL|SKIP` with the
measured numbers and appends to `~/smg-perf/results/chaos.tsv`.

## Comparison with Dynamo (`lib/llm/src/kv_router/indexer/recovery/`)

| Property | Dynamo | SMG after this branch |
|---|---|---|
| Cursor | per (worker, dp_rank), `Initial`/`Live` | same |
| Gap | one recovery request; the worker answers `Events` or a full `TreeDump`; live tail buffered <= 1,024 during recovery | one replay request; settled if the server skips ahead (kept or cleared by size); tail buffer ready, snapshot pending server support |
| Publisher restart | new incarnation from discovery triggers a rank reset with a barrier | detected from the stream (fresh-connection rule, clear below cursor, counter at its start, far-below window) and from the servicers' `DATA_LOSS`; epoch proposed below |
| Resync | transactional per-rank replacement from a tree dump | clear + live rebuild today; atomic replacement once snapshots exist |
| Worker removal | broadcast to all lanes, full-tree sweep | O(worker's blocks) via the reverse map |
| Lag metric | none found | `smg_kv_event_lag_seconds` |

## Proposed additions

### 1. Publisher identity on every batch

```proto
message KvEventBatch {
  uint64 sequence_number = 1;
  double timestamp = 2;
  repeated KvCacheEvent events = 3;
  optional int32 dp_rank = 4;
  // New: changes whenever the publisher (engine process) restarts. Sequence numbers are only
  // comparable within one epoch. Servicers derive it from the engine process start time or a
  // random 64-bit value chosen at publisher creation; the mock engine's `generation` is one.
  optional uint64 publisher_epoch = 5;
  // New: set on the first batch of an epoch whose cache survived the restart (a publisher
  // restart without an engine restart), so the subscriber keeps its blocks and only renumbers.
  optional bool cache_retained = 6;
}
```

With an epoch the gateway no longer infers restarts from the stream, and an epoch change is a
precise "this rank's cache is empty" signal unless `cache_retained` says otherwise.

### 2. Resume with intent

```proto
message SubscribeKvEventsRequest {
  // Last sequence applied (unchanged semantics); meaningful only with has_cursor.
  uint64 start_sequence_number = 1;
  // New: distinguishes a cursor of 0 from "no cursor".
  bool has_cursor = 2;
  // New: the epoch the cursor belongs to; the server replays only if it matches.
  optional uint64 publisher_epoch = 3;
  // New: when the server cannot replay after start_sequence_number, send a snapshot of the
  // rank's current blocks first (see 3) instead of failing with OUT_OF_RANGE / DATA_LOSS.
  bool snapshot_if_unreplayable = 4;
  // New: which data-parallel rank to subscribe to (today: rank 0 only, or all ranks on one
  // stream). A separate stream per rank keeps each publisher's sequence space contiguous and
  // lets a cursor name one rank.
  optional int32 dp_rank = 5;
}
```

### 3. Snapshots on the stream

```proto
message KvEventBatch {
  ...
  // New: this batch is part of a snapshot of the rank's resident blocks, not a live event.
  // The first snapshot batch is preceded by a Cleared event; snapshot batches carry Stored
  // events in parent order; the batch with snapshot_complete = true carries the sequence
  // number of the last live event the snapshot includes (its watermark). Live events resume
  // after it with sequence_number = watermark + 1.
  optional bool snapshot = 7;
  optional bool snapshot_complete = 8;
}
```

A snapshot is produced from the engine's own view (SGLang's radix cache, vLLM's block pool via
the KV-event replay buffer plus the connector's resident set) so it is a consistent cut at the
watermark. If the server cannot build one, it answers `FAILED_PRECONDITION` with a message and the
gateway falls back to today's live rebuild, marked degraded.

Gateway behaviour with this in place (already coded, behind `RankState::begin_snapshot` /
`finish_snapshot`): on a gap the server cannot replay, request a snapshot; hold live batches in the
bounded tail while it streams (a tail overflow restarts the snapshot); replace the worker's blocks
atomically when the snapshot completes (needs `PositionalIndexer::replace_worker`, a swap of the
block map and index entries under one guard, to be added in the indexer workstream); then apply
the tail from the watermark.

### 4. Capabilities

Rather than probing with a request and interpreting an error, the first message of a stream (or a
tiny `KvEventStreamInfo`) states `replay_supported`, `snapshot_supported`, `retained_batches`
(replay buffer size) and `block_size`. The gateway uses them to pick a recovery path without a
failed round trip and exposes them as metrics.

### 5. Servicer-side changes that need no proto change

- The Rust relay should answer a non-zero cursor it cannot honour with `OUT_OF_RANGE` instead of
  silently streaming live, and keep a bounded replay buffer (vLLM's own `buffer_steps` is 10,000
  batches; the relay sits next to it and can mirror the last N).
- The SGLang servicer should request replays in chunks below libzmq's send high-water mark so a
  long replay is not truncated into a second gap.
