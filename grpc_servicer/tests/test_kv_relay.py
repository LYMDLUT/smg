"""The KV-event relay: parity with the Rust normalizer on the recorded fixtures,
lenient decoding, and multi-rank streaming with per-rank replay over real ZMQ
and gRPC (no engine needed)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytest.importorskip("smg_grpc_proto")
grpc = pytest.importorskip("grpc")
zmq = pytest.importorskip("zmq")
msgspec = pytest.importorskip("msgspec")
import zmq.asyncio  # noqa: E402, F811
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer import kv_relay  # noqa: E402

FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "engine_servicer"
    / "tests"
    / "fixtures"
    / "kv_events"
)
TIERS = {
    "device": common_pb2.KV_CACHE_TIER_DEVICE,
    "host": common_pb2.KV_CACHE_TIER_HOST,
    "disk": common_pb2.KV_CACHE_TIER_DISK,
    "external": common_pb2.KV_CACHE_TIER_EXTERNAL,
}


def _manifest():
    return json.loads((FIXTURES / "manifest.json").read_text())


def _optional(message, name):
    return getattr(message, name) if message.HasField(name) else None


def _key_shape(key):
    which = key.WhichOneof("key")
    if which == "blob":
        return {"blob_len": len(key.blob)}
    if which == "multimodal":
        return {"multimodal": [key.multimodal.identifier, key.multimodal.offset]}
    return {which: getattr(key, which)}


# ---------------------------------------------------------------------------
# Parity with the Rust relay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", _manifest()["fixtures"], ids=lambda f: f["name"])
def test_generated_fixtures_normalize_like_the_rust_relay(fixture):
    normalizer = kv_relay.Normalizer()
    batches = [
        normalizer.normalize_batch(kv_relay.decode_batch((FIXTURES / name).read_bytes()), seq)
        for seq, name in enumerate(fixture["files"])
    ]
    forwarded = [
        (_optional(batch, "dp_rank"), event) for batch in batches for event in batch.events
    ]
    expect = fixture["expect"]
    assert len(forwarded) == len(expect["forwarded"]), fixture["name"]
    for index, ((rank, event), want) in enumerate(zip(forwarded, expect["forwarded"])):
        at = f"{fixture['name']} forwarded event {index}"
        assert rank == want["dp_rank"], at
        kind = event.WhichOneof("data")
        assert kind == want["kind"], at
        if kind == "stored":
            stored = event.stored
            assert [b.block_hash for b in stored.blocks] == want["hashes"], at
            assert _optional(stored, "parent_block_hash") == want["parent"], at
            assert stored.tier == TIERS[want["tier"]], at
            assert [list(b.token_ids) for b in stored.blocks] == want["tokens"], at
            for block in stored.blocks:
                assert _optional(block, "cache_level") == want["cache_level"], at
                assert block.block_size == len(block.token_ids), at
            assert _optional(stored, "lora_name") == want["lora_name"], at
            assert _optional(stored, "cache_salt") == want["cache_salt"], at
            assert _optional(stored, "group_idx") == want["group_idx"], at
            assert _optional(stored, "session_id") == want.get("session_id"), at
            if want.get("extra_keys") is not None:
                got = [[_key_shape(key) for key in block.extra_keys] for block in stored.blocks]
                assert got == want["extra_keys"], at
        elif kind == "removed":
            removed = event.removed
            assert list(removed.block_hashes) == want["hashes"], at
            assert removed.tier == TIERS[want["tier"]], at
            assert _optional(removed, "cache_level") == want["cache_level"], at
    counts = normalizer.counts
    want = expect["counts"]
    assert counts.forwarded_stored == want["forwarded_stored"], fixture["name"]
    assert counts.forwarded_removed == want["forwarded_removed"], fixture["name"]
    assert counts.forwarded_cleared == want["forwarded_cleared"], fixture["name"]
    assert counts.duplicate_stores == want["duplicate_stores"], fixture["name"]
    assert counts.bigram_stores == want["bigram_stores"], fixture["name"]
    assert counts.dropped == want["dropped"], fixture["name"]


# ---------------------------------------------------------------------------
# Lenient decoding
# ---------------------------------------------------------------------------


def _store(hashes, tokens, **extra):
    event = {
        "type": "BlockStored",
        "block_hashes": hashes,
        "parent_block_hash": None,
        "token_ids": tokens,
        "block_size": 4,
        "lora_id": None,
        "medium": "GPU",
        "lora_name": None,
    }
    event.update(extra)
    return event


def _batch(events, rank=0, ts=1.5):
    return msgspec.msgpack.encode([ts, events, rank])


def _normalize(payloads, rank=None):
    normalizer = kv_relay.Normalizer()
    batches = [
        normalizer.normalize_batch(kv_relay.decode_batch(payload), seq + 1, rank)
        for seq, payload in enumerate(payloads)
    ]
    return batches, normalizer.counts


def test_hashes_fold_like_the_engines_send_them():
    assert kv_relay.fold_hash(7) == 7
    assert kv_relay.fold_hash(2**63) == -(2**63)
    assert kv_relay.fold_hash(-3) == -3
    digest = bytes(range(32))
    assert kv_relay.fold_hash(digest) == int.from_bytes(digest[-8:], "big", signed=True)
    assert kv_relay.fold_hash("nope") is None


def test_unknown_keys_and_bigram_cells_decode_and_a_bad_event_costs_itself():
    payload = _batch(
        [
            _store([1], [[1, 2], [2, 3], [3, 4], [4, 5]], future_field={"nested": True}),
            {"type": "BlockStored", "block_hashes": "nope", "token_ids": [1], "block_size": 1},
            {"type": "BlockMigrated", "block_hashes": [1]},
            ["BlockRemoved", [1], "GPU"],
        ]
    )
    batches, counts = _normalize([payload])
    events = batches[0].events
    assert [event.WhichOneof("data") for event in events] == ["stored", "removed"]
    assert list(events[0].stored.blocks[0].token_ids) == [1, 2, 3, 4]
    assert events[1].event_id == 4, "ids advance for dropped events too"
    assert counts.bigram_stores == 1
    assert counts.dropped == {"malformed": 1, "unknown_type": 1}


def test_socket_rank_wins_over_the_payload_rank():
    batches, _ = _normalize([_batch([_store([1], [1, 2, 3, 4])], rank=0)], rank=3)
    assert batches[0].dp_rank == 3
    batches, _ = _normalize([msgspec.msgpack.encode([1.5, [_store([1], [1, 2, 3, 4])]])])
    assert not batches[0].HasField("dp_rank")


def test_a_non_batch_payload_is_an_error():
    with pytest.raises(ValueError):
        kv_relay.decode_batch(msgspec.msgpack.encode({"not": "a batch"}))


# ---------------------------------------------------------------------------
# Streaming: ranks, cursors, replay
# ---------------------------------------------------------------------------


def _bind_consecutive(ctx, kind, count, attempts=20):
    """``count`` sockets of ``kind`` on consecutive ports, as the engines lay out ranks."""
    for _ in range(attempts):
        sockets = []
        try:
            first = ctx.socket(kind)
            if kind == zmq.XPUB:
                first.setsockopt(zmq.XPUB_VERBOSE, 1)
            base = first.bind_to_random_port("tcp://127.0.0.1")
            sockets.append(first)
            for offset in range(1, count):
                sock = ctx.socket(kind)
                if kind == zmq.XPUB:
                    sock.setsockopt(zmq.XPUB_VERBOSE, 1)
                sock.bind(f"tcp://127.0.0.1:{base + offset}")
                sockets.append(sock)
            return base, sockets
        except zmq.ZMQError:
            for sock in sockets:
                sock.close(linger=0)
    raise RuntimeError("no consecutive ports available")


@pytest_asyncio.fixture
async def bridge():
    ctx = zmq.asyncio.Context()
    base, pubs = _bind_consecutive(ctx, zmq.XPUB, 2)
    replay_base, routers = _bind_consecutive(ctx, zmq.ROUTER, 2)
    config = SimpleNamespace(
        endpoint=f"tcp://127.0.0.1:{base}",
        replay_endpoint=f"tcp://127.0.0.1:{replay_base}",
        topic="kv",
    )
    options = {"replay_timeout": 0.3, "recv_timeout": 0.1}

    async def handler(request, context):
        sources = kv_relay.rank_sources(config, range(len(pubs)))
        async for batch in kv_relay.relay(
            sources,
            kv_relay.Engine.SGLANG,
            request.start_sequence_number,
            context,
            topic=config.topic,
            zmq_context=ctx,
            **options,
        ):
            yield batch

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "test.KvEvents",
                {
                    "Subscribe": grpc.unary_stream_rpc_method_handler(
                        handler,
                        request_deserializer=common_pb2.SubscribeKvEventsRequest.FromString,
                        response_serializer=common_pb2.KvEventBatch.SerializeToString,
                    )
                },
            ),
        )
    )
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{grpc_port}")
    rpc = channel.unary_stream(
        "/test.KvEvents/Subscribe",
        request_serializer=common_pb2.SubscribeKvEventsRequest.SerializeToString,
        response_deserializer=common_pb2.KvEventBatch.FromString,
    )

    def subscribe(cursor=0):
        return rpc(common_pb2.SubscribeKvEventsRequest(start_sequence_number=cursor))

    async def subscribed():
        # XPUB acknowledges each actual subscription; no timing sleeps needed.
        for pub in pubs:
            while await asyncio.wait_for(pub.recv(), 3) != b"\x01kv":
                pass

    async def publish(rank, seq, payload=None):
        payload = (
            _batch([_store([seq + 1], [1, 2, 3, 4])], rank=None) if payload is None else payload
        )
        await pubs[rank].send_multipart([b"kv", seq.to_bytes(8, "big"), payload])

    async def replay_request(rank):
        frames = await asyncio.wait_for(routers[rank].recv_multipart(), 3)
        assert frames[1] == b""
        return frames[0], int.from_bytes(frames[2], "big")

    async def replay_send(rank, identity, seq, payload=None, framing="sglang"):
        payload = _batch([_store([seq + 1], [1, 2, 3, 4])]) if payload is None else payload
        seq_bytes = kv_relay._END_SEQ if seq == -1 else seq.to_bytes(8, "big")
        frames = [identity, b"", seq_bytes, payload if seq != -1 else b""]
        if framing == "vllm":
            frames.insert(2, b"kv" if seq != -1 else b"")
        await routers[rank].send_multipart(frames)

    try:
        yield SimpleNamespace(
            subscribe=subscribe,
            subscribed=subscribed,
            publish=publish,
            replay_request=replay_request,
            replay_send=replay_send,
            pubs=pubs,
            routers=routers,
            config=config,
            options=options,
        )
    finally:
        await channel.close()
        await server.stop(None)
        for sock in pubs + routers:
            sock.close(linger=0)
        ctx.term()


async def read(call):
    return await asyncio.wait_for(call.read(), 3)


async def read_error(call, after_at_most=3):
    """The status the stream ends with, allowing a few batches before it."""
    with pytest.raises(grpc.aio.AioRpcError) as error:
        for _ in range(after_at_most + 1):
            await read(call)
    return error.value.code()


@pytest.mark.asyncio
async def test_ranks_are_tagged_from_their_socket_and_numbered_contiguously(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    await bridge.publish(0, 1)
    await bridge.publish(1, 0)  # rank 1 starts its own count; not stale
    await bridge.publish(1, 1)
    seen = [await read(call) for _ in range(4)]
    assert [batch.sequence_number for batch in seen] == [1, 2, 3, 4]
    # The two sockets are polled together, so ranks interleave; each rank's
    # own order holds and every batch carries its socket's rank.
    per_rank = {0: [], 1: []}
    for batch in seen:
        per_rank[batch.dp_rank].append(batch.events[0].stored.blocks[0].block_hash)
    assert per_rank == {0: [1, 2], 1: [1, 2]}
    call.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["sglang", "vllm"])
async def test_a_gap_is_filled_from_that_ranks_replay_before_later_batches(bridge, framing):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(1, 0)
    assert (await read(call)).dp_rank == 1
    # Rank 1 skips 1 and 2; rank 0 keeps publishing meanwhile.
    await bridge.publish(1, 3)
    identity, start = await bridge.replay_request(1)
    assert start == 1
    await bridge.publish(0, 0)
    for seq in (1, 2, 3):  # the replay overlaps the live batch 3
        await bridge.replay_send(1, identity, seq, framing=framing)
    await bridge.replay_send(1, identity, -1, framing=framing)
    hashes = []
    for _ in range(4):
        batch = await read(call)
        hashes.append((batch.dp_rank, batch.events[0].stored.blocks[0].block_hash))
    # Replayed 1, 2, 3 for rank 1 in order, the live 3 deduplicated, then rank 0.
    assert hashes[:3] == [(1, 2), (1, 3), (1, 4)]
    assert hashes[3] == (0, 1)
    await bridge.publish(1, 4)
    assert (await read(call)).events[0].stored.blocks[0].block_hash == 5
    call.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["truncated", "empty", "timeout", "short", "malformed"])
async def test_an_unverifiable_replay_ends_the_stream_with_data_loss(bridge, fault):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 4)
    identity, start = await bridge.replay_request(0)
    assert start == 1
    if fault == "truncated":
        await bridge.replay_send(0, identity, 2)  # history no longer holds 1
    elif fault == "empty":
        await bridge.replay_send(0, identity, -1)
    elif fault == "short":
        await bridge.replay_send(0, identity, 1)
        await bridge.replay_send(0, identity, -1)  # ends before 3
    elif fault == "malformed":
        await bridge.routers[0].send_multipart([identity, b"", b"bad"])
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_a_gap_without_a_replay_endpoint_ends_the_stream(bridge):
    bridge.config.replay_endpoint = None
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 2)
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_a_publisher_restart_ends_the_stream_with_data_loss(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 5)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 0)
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_a_nonzero_cursor_is_refused_before_subscribing(bridge):
    call = bridge.subscribe(100)
    assert await read_error(call) == grpc.StatusCode.OUT_OF_RANGE
    assert not await bridge.pubs[0].poll(timeout=50)


@pytest.mark.asyncio
async def test_bad_payloads_and_duplicates_are_skipped_without_replay(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    await bridge.publish(0, 1, b"not msgpack")
    await bridge.publish(0, 1, b"not msgpack")
    await bridge.pubs[0].send_multipart([b"kv", b"short"])
    await bridge.publish(0, 2)
    await bridge.publish(0, 2)
    first = await read(call)
    second = await read(call)
    assert (first.sequence_number, second.sequence_number) == (1, 2)
    assert second.events[0].stored.blocks[0].block_hash == 3
    assert not await bridge.routers[0].poll(timeout=50), "no replay for a consumed sequence"
    call.cancel()


@pytest.mark.asyncio
async def test_cancellation_releases_every_subscription(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    call.cancel()
    for pub in bridge.pubs:
        assert await asyncio.wait_for(pub.recv(), 3) == b"\x00kv"
