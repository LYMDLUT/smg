# KV-event fixtures

Engine-encoded `KVEventBatch` payloads for `tests/kv_event_fixtures.rs`, which
decodes and normalizes them with the relay in `src/kv_wire.rs` and round-trips
the result into the gateway's positional index.

## Generated sets

`generate.py` encodes four scenarios with msgspec through mirrors of the
engines' own structs (sources named in the script): vLLM and SGLang, each in
the current tagged-map layout and the legacy tag-first array layout. One file
is one publisher message (`<set>-<nn>.msgpack`); `manifest.json` lists the
sets, their files in order, and the normalizer's expected output: every
forwarded event with its tier, cache level, namespace, tokens and extra keys,
and the counters (`DropReason::as_str` names). Regenerate after changing the
script:

```
python3 -m venv /tmp/kv-fixtures && /tmp/kv-fixtures/bin/pip install msgspec
/tmp/kv-fixtures/bin/python crates/engine_servicer/tests/fixtures/kv_events/generate.py
```

The scenarios cover: both hash forms (vLLM's unsigned low 64 bits and raw
digests, SGLang's signed high 64 bits); a sliding-window group with no
hashes; a CPU-offload placeholder and its eager removal; STORAGE blocks that
are remote, agent-owned (`kvcr`) and local; unknown media; unaligned and
self-referencing stores; an unknown event type; a malformed event; a pool
reset; a LoRA request with a multimodal item, a cache salt and a prompt
embeddings digest, with a child that omits the salt; vLLM's duplicate physical
copies and per-copy removals; SGLang's startup clear, HiCache write-through
(store GPU, store CPU_PINNED, remove GPU, store GPU, remove CPU_PINNED), a
salted chain, an Eagle bigram page, DISK and EXTERNAL media; and a second DP
rank for both engines.

## Recorded captures

Captures from a live engine are checked by the same test. Record one file per
ZMQ message from the publisher (vLLM `--kv-events-config`, SGLang
`--kv-events-config`; the frame is `[topic, sequence as 8-byte big-endian,
payload]`), for example:

```python
import zmq
sock = zmq.Context().socket(zmq.SUB)
sock.subscribe(b"")
sock.connect("tcp://127.0.0.1:5557")
while True:
    topic, seq, payload = sock.recv_multipart()
    n = int.from_bytes(seq, "big")
    open(f"{n:012d}.msgpack", "wb").write(payload)
```

Then run the test against the directory:

```
SMG_KV_EVENT_CAPTURES=/path/to/captures cargo test -p engine-servicer --test kv_event_fixtures -- --nocapture
```

Without a manifest every `*.msgpack` file in name order is one stream that
must decode without losing a batch, and the counters are printed. With a
`manifest.json` of the generated sets' shape (`files`, optional `expect`) the
captures are checked like the generated ones; omit `expect` to only require
decoding and print the counts.
