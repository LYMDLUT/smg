#!/usr/bin/env python3
"""Generate the KV-event fixtures the relay tests decode and normalize.

The batches are encoded with msgspec through struct mirrors of the engines'
own definitions, so the bytes are what the publishers put on the wire:

- vLLM `vllm/distributed/kv_events.py` (main @ 0c16eee3f1): `EventBatch` is
  `array_like` (`[ts, events, data_parallel_rank]`), events are tagged maps
  (`type`) with `omit_defaults`; required fields are present even when nil.
- SGLang `python/sglang/srt/disaggregation/kv_events.py` (b1bbd74f28): the
  same shape, `attn_dp_rank` in the batch's third slot (nil when unset,
  SGLang's batch does not omit defaults), `medium`, `cache_salt` and
  `session_id` omitted when None, bigram pages as `[t, t+1]` cells.
- The legacy layout both engines used before the map encoding: the same
  structs with `array_like=True`, tag first, trailing defaults omitted.

Each fixture is one scenario as a sequence of batches, one file per batch
(one ZMQ message), and the normalizer's expected output for the sequence:
the forwarded events in order and the counters, keyed by the relay's
`DropReason::as_str` names. Expectations are written by hand next to the
events, as the rules in `crates/engine_servicer/src/kv_wire.rs` say; the
Rust test (`tests/kv_event_fixtures.rs`) checks the implementation agrees.

Usage: `python3 generate.py` (needs `msgspec`; the engines pin 0.22).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any

import msgspec

HERE = pathlib.Path(__file__).resolve().parent
U64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Struct mirrors
# ---------------------------------------------------------------------------


def vllm_structs(array_like: bool):
    """vLLM's event structs; `array_like` selects the legacy layout."""

    class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
        ts: float
        events: list[Any]
        data_parallel_rank: int | None = None

    class KVCacheEvent(
        msgspec.Struct, array_like=array_like, omit_defaults=True, gc=False, tag=True
    ):
        pass

    class BlockStored(KVCacheEvent):
        block_hashes: list[int | bytes]
        parent_block_hash: int | bytes | None
        token_ids: list[int]
        block_size: int
        lora_id: int | None
        medium: str | None
        lora_name: str | None
        extra_keys: list[tuple[Any, ...] | None] | None = None
        group_idx: int | None = None
        kv_cache_spec_kind: str | None = None
        kv_cache_spec_sliding_window: int | None = None
        locality: str | None = None
        ownership: str | None = None
        session_id: str | None = None

    class BlockRemoved(KVCacheEvent):
        block_hashes: list[int | bytes]
        medium: str | None
        group_idx: int | None = None
        locality: str | None = None
        ownership: str | None = None

    class AllBlocksCleared(KVCacheEvent):
        pass

    class BlockMigrated(KVCacheEvent):
        """An event type the relay does not know (stands in for a future one)."""

        block_hashes: list[int]
        destination: str

    return EventBatch, BlockStored, BlockRemoved, AllBlocksCleared, BlockMigrated


def sglang_structs(array_like: bool):
    """SGLang's event structs; `array_like` selects the legacy layout."""

    class EventBatch(msgspec.Struct, array_like=True, gc=False):
        ts: float
        events: list[Any]
        attn_dp_rank: int | None = None

    class KVCacheEvent(
        msgspec.Struct, array_like=array_like, omit_defaults=True, gc=False, tag=True
    ):
        pass

    class BlockStored(KVCacheEvent):
        block_hashes: list[int]
        parent_block_hash: int | None
        token_ids: list[Any]  # ints, or [t, t+1] pairs under bigram hashing
        block_size: int
        lora_id: int | None
        medium: str | None = None
        cache_salt: str | None = None
        session_id: str | None = None

    class BlockRemoved(KVCacheEvent):
        block_hashes: list[int]
        medium: str | None = None

    class AllBlocksCleared(KVCacheEvent):
        pass

    class BlockMigrated(KVCacheEvent):
        block_hashes: list[int]
        destination: str

    return EventBatch, BlockStored, BlockRemoved, AllBlocksCleared, BlockMigrated


# ---------------------------------------------------------------------------
# Hash identities
# ---------------------------------------------------------------------------


def digest(label: str) -> bytes:
    return hashlib.sha256(label.encode()).digest()


def as_i64(value: int) -> int:
    value &= U64
    return value - (1 << 64) if value >= 1 << 63 else value


def vllm_int(label: str) -> int:
    """vLLM's integer form: the low 64 bits of the digest, unsigned."""
    return int.from_bytes(digest(label), "big") & U64


def vllm_expected(label: str) -> int:
    """What the relay forwards for either form of a vLLM hash."""
    return as_i64(vllm_int(label))


def sglang_int(label: str) -> int:
    """SGLang's integer form: the high 64 bits of the digest, signed."""
    return int.from_bytes(digest(label)[:8], "big", signed=True)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def stored_expect(
    hashes,
    tokens,
    *,
    dp_rank=0,
    parent=None,
    tier="device",
    cache_level=None,
    lora_name=None,
    cache_salt=None,
    group_idx=None,
    session_id=None,
    extra_keys=None,
):
    return {
        "kind": "stored",
        "dp_rank": dp_rank,
        "hashes": hashes,
        "parent": parent,
        "tier": tier,
        "cache_level": cache_level,
        "tokens": tokens,
        "lora_name": lora_name,
        "cache_salt": cache_salt,
        "group_idx": group_idx,
        "session_id": session_id,
        "extra_keys": extra_keys,
    }


def removed_expect(hashes, *, dp_rank=0, tier="device", cache_level=None):
    return {
        "kind": "removed",
        "dp_rank": dp_rank,
        "hashes": hashes,
        "tier": tier,
        "cache_level": cache_level,
    }


def cleared_expect(*, dp_rank=0):
    return {"kind": "cleared", "dp_rank": dp_rank}


def vllm_scenario(array_like: bool):
    EventBatch, Stored, Removed, Cleared, Migrated = vllm_structs(array_like)
    bs = 4
    h = {name: vllm_int(f"vllm-{name}") for name in "abcdefghijk"}
    e = {name: vllm_expected(f"vllm-{name}") for name in "abcdefghijk"}
    d1, d2 = digest("vllm-digest-1"), digest("vllm-digest-2")
    embeds = digest("vllm-prompt-embeds")
    # The low 64 bits of a digest can exceed i64::MAX; make sure one does.
    assert any(v >= 1 << 63 for v in h.values()), "pick labels with a high bit set"

    def gpu_stored(hashes, parent, tokens, **kw):
        kw.setdefault("medium", "GPU")
        kw.setdefault("group_idx", 0)
        kw.setdefault("kv_cache_spec_kind", "full_attention")
        return Stored(
            block_hashes=hashes,
            parent_block_hash=parent,
            token_ids=tokens,
            block_size=kw.pop("block_size", bs),
            lora_id=kw.pop("lora_id", None),
            lora_name=kw.pop("lora_name", None),
            **kw,
        )

    batches = []
    forwarded = []

    # Batch 0: a plain chain in both hash forms; a sliding-window group store.
    batches.append(
        EventBatch(
            ts=1700000000.0,
            data_parallel_rank=0,
            events=[
                gpu_stored([h["a"], h["b"]], None, list(range(1, 9)), session_id="req-1"),
                # Sliding-window group: more tokens than hashes x block size,
                # no hashes at all. Dropped by the group gate.
                gpu_stored(
                    [],
                    None,
                    list(range(1, 9)),
                    group_idx=1,
                    kv_cache_spec_kind="sliding_window",
                    kv_cache_spec_sliding_window=128,
                ),
                # Raw digests (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0), the
                # parent given as an int, no extra keys on either block.
                gpu_stored([d1, d2], h["b"], list(range(9, 17)), extra_keys=[None, None]),
            ],
        )
    )
    forwarded += [
        stored_expect(
            [e["a"], e["b"]],
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            group_idx=0,
            session_id="req-1",
        ),
        stored_expect(
            [as_i64(int.from_bytes(d1[-8:], "big")), as_i64(int.from_bytes(d2[-8:], "big"))],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
            parent=e["b"],
            group_idx=0,
        ),
    ]

    # Batch 1: a second physical copy, per-copy removals, offload tiers and
    # every other drop rule.
    batches.append(
        EventBatch(
            ts=1700000001.0,
            data_parallel_rank=0,
            events=[
                gpu_stored([h["a"], h["b"]], None, list(range(1, 9))),  # duplicate copy
                Removed(block_hashes=[h["a"]], medium="GPU", group_idx=0),
                Removed(block_hashes=[h["a"]], medium="GPU", group_idx=0),  # other copy
                # CPU offload placeholder: a chunk key, no tokens, block_size 0.
                gpu_stored([h["c"]], None, [], medium="CPU", block_size=0, kv_cache_spec_kind=None),
                Removed(block_hashes=[h["c"]], medium="CPU", group_idx=0),
                gpu_stored([h["d"]], None, [1, 2, 3, 4], medium="STORAGE", locality="REMOTE"),
                gpu_stored(
                    [h["d"]],
                    None,
                    [1, 2, 3, 4],
                    medium="STORAGE",
                    locality="LOCAL",
                    ownership="kvcr",
                ),
                gpu_stored([h["d"]], None, [1, 2, 3, 4], medium="STORAGE", locality="LOCAL"),
                gpu_stored([h["f"]], None, [1, 2, 3, 4], medium="MARS"),
                gpu_stored([h["g"]], None, [1, 2, 3, 4, 5, 6]),  # unaligned
                gpu_stored([h["i"]], h["i"], [1, 2, 3, 4]),  # parent is itself
                Migrated(block_hashes=[h["j"]], destination="peer"),
                # Malformed: hashes are not a list. Encoded as a raw map/array
                # because no struct produces it.
                (
                    ["BlockStored", "nope", None, [1, 2, 3, 4], bs]
                    if array_like
                    else {
                        "type": "BlockStored",
                        "block_hashes": "nope",
                        "parent_block_hash": None,
                        "token_ids": [1, 2, 3, 4],
                        "block_size": bs,
                    }
                ),
            ],
        )
    )
    forwarded += [
        stored_expect([e["a"], e["b"]], [[1, 2, 3, 4], [5, 6, 7, 8]], group_idx=0),
        removed_expect([e["a"]]),
        removed_expect([e["a"]]),
        removed_expect([e["c"]], tier="host", cache_level=1),
        stored_expect([e["d"]], [[1, 2, 3, 4]], tier="disk", cache_level=2, group_idx=0),
    ]

    # Batch 2: the pool reset, then the chain again (not a duplicate any more).
    batches.append(
        EventBatch(
            ts=1700000002.0,
            data_parallel_rank=0,
            events=[
                Cleared(),
                gpu_stored([h["a"], h["b"]], None, list(range(1, 9))),
            ],
        )
    )
    forwarded += [
        cleared_expect(),
        stored_expect([e["a"], e["b"]], [[1, 2, 3, 4], [5, 6, 7, 8]], group_idx=0),
    ]

    # Batch 3: a LoRA request with a multimodal item, a cache salt and prompt
    # embeddings; the salt rides in block 0's extra keys only and the child
    # inherits it.
    batches.append(
        EventBatch(
            ts=1700000003.0,
            data_parallel_rank=0,
            events=[
                gpu_stored(
                    [h["e"]],
                    None,
                    [1, 2, 3, 4],
                    lora_id=7,
                    lora_name="adapter",
                    extra_keys=[("adapter", ("mm-abc", 0), "salt-1", embeds)],
                    session_id="req-2",
                ),
                gpu_stored(
                    [h["h"]],
                    h["e"],
                    [5, 6, 7, 8],
                    lora_id=7,
                    lora_name="adapter",
                    extra_keys=[("adapter",)],
                    session_id="req-2",
                ),
            ],
        )
    )
    forwarded += [
        stored_expect(
            [e["e"]],
            [[1, 2, 3, 4]],
            lora_name="adapter",
            cache_salt="salt-1",
            group_idx=0,
            session_id="req-2",
            extra_keys=[
                [
                    {"text": "adapter"},
                    {"multimodal": ["mm-abc", 0]},
                    {"text": "salt-1"},
                    {"blob_len": 32},
                ]
            ],
        ),
        stored_expect(
            [e["h"]],
            [[5, 6, 7, 8]],
            parent=e["e"],
            lora_name="adapter",
            cache_salt="salt-1",
            group_idx=0,
            session_id="req-2",
            extra_keys=[[{"text": "adapter"}]],
        ),
    ]

    # Batch 4: another DP rank stores the same hashes; seen-sets are per rank.
    batches.append(
        EventBatch(
            ts=1700000004.0,
            data_parallel_rank=1,
            events=[gpu_stored([h["a"], h["b"]], None, list(range(1, 9)))],
        )
    )
    forwarded += [
        stored_expect([e["a"], e["b"]], [[1, 2, 3, 4], [5, 6, 7, 8]], dp_rank=1, group_idx=0),
    ]

    counts = {
        "forwarded_stored": 8,
        "forwarded_removed": 3,
        "forwarded_cleared": 1,
        "duplicate_stores": 1,
        "bigram_stores": 0,
        "dropped": {
            "non_main_attention_group": 1,
            "placeholder": 1,
            "non_local_locality": 1,
            "unsupported_ownership": 1,
            "unknown_medium": 1,
            "unaligned_blocks": 1,
            "self_referencing_hashes": 1,
            "unknown_type": 1,
            "malformed": 1,
        },
    }
    return batches, {"forwarded": forwarded, "counts": counts}


def sglang_scenario(array_like: bool):
    EventBatch, Stored, Removed, Cleared, Migrated = sglang_structs(array_like)
    bs = 4
    s = {name: sglang_int(f"sglang-{name}") for name in "abcdefgh"}
    assert any(v < 0 for v in s.values()), "pick labels with a negative i64"

    def stored(hashes, parent, tokens, **kw):
        kw.setdefault("medium", "GPU")
        return Stored(
            block_hashes=hashes,
            parent_block_hash=parent,
            token_ids=tokens,
            block_size=bs,
            lora_id=None,
            **kw,
        )

    batches = []
    forwarded = []

    # Batch 0: the first batch after startup clears.
    batches.append(EventBatch(ts=1700000000.0, attn_dp_rank=0, events=[Cleared()]))
    forwarded += [cleared_expect()]

    # Batch 1: a chain; the second store is coalesced over two pages.
    batches.append(
        EventBatch(
            ts=1700000001.0,
            attn_dp_rank=0,
            events=[
                stored([s["a"]], None, [1, 2, 3, 4], session_id="req-1"),
                stored([s["b"], s["c"]], s["a"], list(range(5, 13)), session_id="req-1"),
            ],
        )
    )
    forwarded += [
        stored_expect([s["a"]], [[1, 2, 3, 4]], session_id="req-1"),
        stored_expect(
            [s["b"], s["c"]], [[5, 6, 7, 8], [9, 10, 11, 12]], parent=s["a"], session_id="req-1"
        ),
    ]

    # Batch 2: HiCache write-through: back up to host, demote (device copy
    # goes, host stays), load back, evict the host copy.
    batches.append(
        EventBatch(
            ts=1700000002.0,
            attn_dp_rank=0,
            events=[
                stored([s["a"]], None, [1, 2, 3, 4], medium="CPU_PINNED"),
                Removed(block_hashes=[s["a"]], medium="GPU"),
                stored([s["a"]], None, [1, 2, 3, 4]),
                Removed(block_hashes=[s["a"]], medium="CPU_PINNED"),
            ],
        )
    )
    forwarded += [
        stored_expect([s["a"]], [[1, 2, 3, 4]], tier="host", cache_level=1),
        removed_expect([s["a"]]),
        stored_expect([s["a"]], [[1, 2, 3, 4]]),
        removed_expect([s["a"]], tier="host", cache_level=1),
    ]

    # Batch 3: a salted request's chain (the legacy array layout has no
    # readable salt slot, so that variant stores the chain unsalted).
    salt = None if array_like else "tenant-a"
    batches.append(
        EventBatch(
            ts=1700000003.0,
            attn_dp_rank=0,
            events=[
                stored([s["d"]], None, [1, 2, 3, 4], cache_salt=salt),
                stored([s["e"]], s["d"], [5, 6, 7, 8], cache_salt=salt),
            ],
        )
    )
    forwarded += [
        stored_expect([s["d"]], [[1, 2, 3, 4]], cache_salt=salt),
        stored_expect([s["e"]], [[5, 6, 7, 8]], parent=s["d"], cache_salt=salt),
    ]

    # Batch 4: an Eagle bigram page, removed again in the same batch. (Its
    # tokens differ from the plain chain's: two engine hashes with the same
    # tokens at the same position share one index membership per worker.)
    batches.append(
        EventBatch(
            ts=1700000004.0,
            attn_dp_rank=0,
            events=[
                stored([s["f"]], None, [[21, 22], [22, 23], [23, 24], [24, 25]]),
                Removed(block_hashes=[s["f"]], medium="GPU"),
            ],
        )
    )
    forwarded += [
        stored_expect([s["f"]], [[21, 22, 23, 24]]),
        removed_expect([s["f"]]),
    ]

    # Batch 5: the tiers the default core never emits but defines; an event
    # type the relay does not know.
    batches.append(
        EventBatch(
            ts=1700000005.0,
            attn_dp_rank=0,
            events=[
                stored([s["g"]], None, [1, 2, 3, 4], medium="DISK"),
                stored([s["h"]], None, [1, 2, 3, 4], medium="EXTERNAL"),
                Migrated(block_hashes=[s["h"]], destination="peer"),
            ],
        )
    )
    forwarded += [
        stored_expect([s["g"]], [[1, 2, 3, 4]], tier="disk", cache_level=2),
        stored_expect([s["h"]], [[1, 2, 3, 4]], tier="external", cache_level=3),
    ]

    # Batch 6: another attention DP rank; its batch carries its rank.
    batches.append(
        EventBatch(
            ts=1700000006.0,
            attn_dp_rank=1,
            events=[stored([s["a"]], None, [1, 2, 3, 4])],
        )
    )
    forwarded += [stored_expect([s["a"]], [[1, 2, 3, 4]], dp_rank=1)]

    if array_like:
        # SGLang's legacy arrays put session_id where vLLM has extra_keys; the
        # relay cannot read it there.
        for item in forwarded:
            if "session_id" in item:
                item["session_id"] = None

    counts = {
        "forwarded_stored": 10,
        "forwarded_removed": 3,
        "forwarded_cleared": 1,
        "duplicate_stores": 0,
        "bigram_stores": 1,
        "dropped": {"unknown_type": 1},
    }
    return batches, {"forwarded": forwarded, "counts": counts}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def main() -> int:
    encoder = msgspec.msgpack.Encoder()
    fixtures = []
    for engine, scenario in (("vllm", vllm_scenario), ("sglang", sglang_scenario)):
        for layout, array_like in (("map", False), ("array", True)):
            batches, expect = scenario(array_like)
            name = f"{engine}-{layout}"
            files = []
            for index, batch in enumerate(batches):
                file = f"{name}-{index:02d}.msgpack"
                (HERE / file).write_bytes(encoder.encode(batch))
                files.append(file)
            fixtures.append(
                {
                    "name": name,
                    "engine": engine,
                    "layout": layout,
                    "block_size": 4,
                    "files": files,
                    "expect": expect,
                }
            )
    manifest = {
        "generator": "generate.py",
        "msgspec": msgspec.__version__,
        "fixtures": fixtures,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {sum(len(f['files']) for f in fixtures)} batches for {len(fixtures)} fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
