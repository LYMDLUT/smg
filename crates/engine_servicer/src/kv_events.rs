//! `SubscribeKvEvents`: an engine's ZMQ KV-cache event publisher relayed into
//! the gRPC stream: the publisher's own sequence numbers, one event-id
//! counter per stream, bad frames skipped, no replay, and the stream ending
//! with the client. The wire format and the normalization each event goes
//! through live in [`crate::kv_wire`].
//!
//! `SMG_KV_EVENT_HASH_CHECK=sglang|vllm-sha256-cbor` turns on the relay's
//! engine-hash verification ([`crate::engine_hash`]): every store whose
//! parent is known is rehashed the worker's way and mismatches are counted
//! in the stream's counters, never dropped. The counters are logged when the
//! stream closes.
//!
//! A publisher restart is visible only as its sequence counter starting over
//! (the SUB socket reconnects silently). The relay ends the stream with
//! `DATA_LOSS` when a sequence goes backwards, so the gateway clears what it
//! held for the worker and resubscribes from zero instead of discarding the
//! restarted publisher's batches as stale.
//!
//! Framing (`ZmqEventPublisher` in both engines): one PUB multipart message
//! per scheduler step, `[topic, sequence as u64 big-endian, msgpack batch]`.

use engine_zmq_client::codec::TrailingTolerant;
use futures::stream;
use smg_grpc_client::common_proto::{self as common};
use tonic::Status;
use tracing::{debug, info, warn};
use zeromq::{
    prelude::{Socket, SocketRecv},
    SocketOptions, SubSocket, ZmqError, ZmqMessage,
};

use crate::{
    kv_wire::{low64_big_endian, Normalizer, WireBatch},
    BoxStream,
};

/// The Python vLLM servicer's refusal when vLLM runs without a ZMQ publisher.
pub(crate) const VLLM_DISABLED_MESSAGE: &str = "KV cache events not enabled. Start vLLM with \
     --kv-events-config '{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\"}'";

/// The Python SGLang servicer's refusal when SGLang runs without a ZMQ publisher.
pub(crate) const SGLANG_DISABLED_MESSAGE: &str = "KV cache events not enabled. Start SGLang \
     with --kv-events-config '{\"publisher\": \"zmq\"}'";

/// The Python TokenSpeed servicer's refusal without a publisher.
pub(crate) const TOKENSPEED_DISABLED_MESSAGE: &str = "KV cache events not enabled. Start \
     TokenSpeed with --kv-events-config '{\"enable_kv_cache_events\": true, \"publisher\": \
     \"zmq\"}'";

/// Handle one `SubscribeKvEvents` call against a configured publisher: a
/// relay of rank 0's publisher at `kv_events_endpoint`. Both engines publish
/// the same batch shape (`[ts, events, rank]`, the rank named
/// `data_parallel_rank` by vLLM and `attn_dp_rank` by TokenSpeed; positional
/// on the wire).
pub(crate) fn subscribe(
    kv_events_endpoint: &str,
    topic: String,
    request: common::SubscribeKvEventsRequest,
) -> BoxStream<common::KvEventBatch> {
    // For DP attention each rank publishes on port + rank with independent
    // sequence counters; subscribing to several on one socket interleaves
    // them and breaks gap detection. Subscribe to rank 0 only for now.
    let endpoint = endpoint_for_rank(kv_events_endpoint, 0);
    if request.start_sequence_number != 0 {
        // As on the Python relay: no replay, the stream starts at the
        // publisher's current position and the Router dedups by sequence.
        debug!(
            start_sequence_number = request.start_sequence_number,
            "SubscribeKvEvents: replay is not supported; streaming live events"
        );
    }
    relay(endpoint, topic)
}

/// Resolve a KV-events PUB endpoint to a connectable SUB address: bind
/// wildcards become loopback, and under data parallelism rank `dp_rank`
/// publishes on `base_port + dp_rank` (tcp only; ipc/inproc get no port
/// arithmetic).
pub(crate) fn endpoint_for_rank(endpoint: &str, dp_rank: u32) -> String {
    let resolved = endpoint
        .replace('*', "127.0.0.1")
        .replace("0.0.0.0", "127.0.0.1");
    if dp_rank == 0 || !resolved.starts_with("tcp://") {
        return resolved;
    }
    let Some((host, port)) = resolved.rsplit_once(':') else {
        return resolved;
    };
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return resolved;
    }
    match port.parse::<u64>() {
        Ok(port) => format!("{host}:{}", port.saturating_add(u64::from(dp_rank))),
        Err(_) => resolved,
    }
}

type Item = Result<common::KvEventBatch, Status>;

/// The relay as a response stream: it connects on its first poll, so the
/// RPC's headers go out as soon as the handler returns (the Python relay
/// sends its initial metadata before its first receive), then yields one
/// proto batch per publisher message. Dropping it closes the socket.
fn relay(endpoint: String, topic: String) -> BoxStream<common::KvEventBatch> {
    Box::pin(stream::unfold(
        Relay::Connecting { endpoint, topic },
        |relay| async move { relay.step().await },
    ))
}

enum Relay {
    Connecting { endpoint: String, topic: String },
    Live(Box<Live>),
    Ended,
}

impl Relay {
    /// The next stream item and the state after it; `None` ends the stream.
    async fn step(self) -> Option<(Item, Self)> {
        let mut live = match self {
            Self::Ended => return None,
            Self::Connecting { endpoint, topic } => match connect(&endpoint, &topic).await {
                Ok(socket) => Box::new(Live {
                    endpoint,
                    socket,
                    event_id: 0,
                    last_sequence: None,
                    normalizer: Normalizer::from_env(),
                }),
                Err(status) => return Some((Err(status), Self::Ended)),
            },
            Self::Live(live) => live,
        };
        match live.next_batch().await {
            Ok(batch) => Some((Ok(batch), Self::Live(live))),
            Err(status) => Some((Err(status), Self::Ended)),
        }
    }
}

/// A connected subscription and the stream's event-id counter.
struct Live {
    endpoint: String,
    socket: SubSocket,
    /// Advances once per publisher event, convertible or not, so ids stay
    /// monotonic as the Python relay's do.
    event_id: u64,
    /// The last sequence number relayed, to notice a publisher restart.
    last_sequence: Option<u64>,
    /// Per-stream normalization state (seen hashes, cache groups, salts).
    normalizer: Normalizer,
}

impl Drop for Live {
    fn drop(&mut self) {
        info!(
            endpoint = %self.endpoint,
            counts = ?self.normalizer.counts(),
            "SubscribeKvEvents: stream closed"
        );
    }
}

impl Live {
    /// The next decodable batch; what the Python relay skips (fewer than
    /// three frames, undecodable payloads) is skipped here too.
    async fn next_batch(&mut self) -> Item {
        loop {
            let message = self.socket.recv().await.map_err(|error| {
                Status::internal(format!(
                    "SubscribeKvEvents: receive from {} failed: {error}",
                    self.endpoint
                ))
            })?;
            let Some((sequence_number, payload)) = split_frames(&message) else {
                continue;
            };
            if let Some(last) = self.last_sequence.filter(|&last| sequence_number < last) {
                return Err(Status::data_loss(format!(
                    "SubscribeKvEvents: publisher at {} restarted its sequence at \
                     {sequence_number} after {last}; resubscribe from zero",
                    self.endpoint
                )));
            }
            self.last_sequence = Some(sequence_number);
            let batch = match rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(payload) {
                Ok(batch) => batch.0,
                Err(error) => {
                    warn!(%error, sequence_number, "Failed to decode KV event batch");
                    continue;
                }
            };
            return Ok(self
                .normalizer
                .normalize_batch(batch, sequence_number, &mut self.event_id));
        }
    }
}

/// A SUB socket subscribed to `topic` and connected to `endpoint`. The
/// subscription is recorded first and sent on connect (and on the crate's
/// reconnects), as libzmq does; a refused publisher is retried until the
/// stream is dropped, as libzmq's background connect would.
async fn connect(endpoint: &str, topic: &str) -> Result<SubSocket, Status> {
    let mut options = SocketOptions::default();
    options.no_connect_timeout();
    let mut socket = SubSocket::with_options(options);
    let failed = |step: &str, error: ZmqError| {
        Status::internal(format!("SubscribeKvEvents: {step} {endpoint}: {error}"))
    };
    socket
        .subscribe(topic)
        .await
        .map_err(|error| failed("could not subscribe to", error))?;
    socket
        .connect(endpoint)
        .await
        .map_err(|error| failed("could not connect to", error))?;
    info!(%endpoint, "SubscribeKvEvents: connected to ZMQ endpoint");
    Ok(socket)
}

/// A publisher message's `[topic, sequence, payload, ...]` as the sequence
/// number and payload, or `None` for fewer than three frames.
fn split_frames(message: &ZmqMessage) -> Option<(u64, &[u8])> {
    if message.len() < 3 {
        return None;
    }
    let sequence = message.get(1)?;
    let payload = message.get(2)?;
    Some((low64_big_endian(sequence), payload.as_ref()))
}

/// Golden publisher payloads encoded by vLLM 0.30.1rc1 (msgspec 0.22) with
/// `crates/engine_servicer/scripts/generate_kv_events_golden.py`, which also
/// prints the Python relay's conversion of them (the expected protos below).
#[cfg(test)]
pub(crate) mod golden {
    use zeromq::ZmqMessage;

    /// `KVEventBatch(ts=1700000000.5, data_parallel_rank=None)` with a
    /// `BlockStored` of two sha256-byte hashes (`00..00 80 00..00`,
    /// `00..00 ff..fe`), parent 7, tokens 1..=8, block size 4, medium GPU; a
    /// `BlockStored` of int hash 0x1234, no parent, tokens [9, 10], block
    /// size 2, lora_id 3, group_idx 0, kv_cache_spec_kind full_attention; an
    /// unaligned `BlockStored` (one hash, block size 4, three tokens); a
    /// `BlockRemoved` of [0x1234, ff..ff]; an `AllBlocksCleared`.
    pub(crate) const BATCH1: &str = "93cb41d954fc402000009588a474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657392c4200000000000000000000000000000000000000000000000008000000000000000c420000000000000000000000000000000000000000000000000fffffffffffffffeb1706172656e745f626c6f636b5f6861736807a9746f6b656e5f696473980102030405060708aa626c6f636b5f73697a6504a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c08aa474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657391cd1234b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647392090aaa626c6f636b5f73697a6502a76c6f72615f696403a66d656469756dc0a96c6f72615f6e616d65c0a967726f75705f69647800b26b765f63616368655f737065635f6b696e64ae66756c6c5f617474656e74696f6e88a474797065ab426c6f636b53746f726564ac626c6f636b5f6861736865739105b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647393010203aa626c6f636b5f73697a6504a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c083a474797065ac426c6f636b52656d6f766564ac626c6f636b5f68617368657392cd1234c420ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa66d656469756da347505581a474797065b0416c6c426c6f636b73436c6561726564c0";

    /// `KVEventBatch(ts=1700000001.0, data_parallel_rank=1)` with one
    /// `BlockStored`: int hash 42, parent 41, tokens [100, 101], block size 2.
    pub(crate) const BATCH2: &str = "93cb41d954fc404000009188a474797065ab426c6f636b53746f726564ac626c6f636b5f686173686573912ab1706172656e745f626c6f636b5f6861736829a9746f6b656e5f696473926465aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c001";

    pub(crate) fn bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
            .collect()
    }

    /// A publisher message: `[topic, sequence (u64 big-endian), payload]`.
    pub(crate) fn frame(topic: &[u8], sequence: u64, payload: &[u8]) -> ZmqMessage {
        let mut message = ZmqMessage::from(topic.to_vec());
        message.push_back(sequence.to_be_bytes().to_vec().into());
        message.push_back(payload.to_vec().into());
        message
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::StreamExt;
    use smg_grpc_client::common_proto::{kv_cache_event, KvCacheLocality, KvCacheTier};
    use tokio::time::timeout;
    use zeromq::{prelude::*, PubSocket, SocketEvent};

    use super::{golden, *};
    use crate::kv_wire::WireBatch;

    fn decode(hex: &str) -> WireBatch {
        rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&golden::bytes(hex))
            .expect("golden batch decodes")
            .0
    }

    fn convert_batch(
        batch: WireBatch,
        sequence_number: u64,
        event_id: &mut u64,
    ) -> common::KvEventBatch {
        Normalizer::new().normalize_batch(batch, sequence_number, event_id)
    }

    fn stored(event: &common::KvCacheEvent) -> &common::KvBlocksStored {
        match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => stored,
            other => panic!("expected a stored event, got {other:?}"),
        }
    }

    fn block(block_hash: i64, token_ids: Vec<u32>, lora_id: Option<i64>) -> common::KvBlock {
        common::KvBlock {
            block_hash,
            block_size: i32::try_from(token_ids.len()).unwrap(),
            token_ids,
            lora_id,
            cache_level: None,
            ..Default::default()
        }
    }

    #[test]
    fn endpoint_for_rank_mirrors_the_python_helper() {
        assert_eq!(endpoint_for_rank("tcp://*:5557", 0), "tcp://127.0.0.1:5557");
        assert_eq!(
            endpoint_for_rank("tcp://0.0.0.0:5557", 0),
            "tcp://127.0.0.1:5557"
        );
        assert_eq!(endpoint_for_rank("tcp://*:5557", 2), "tcp://127.0.0.1:5559");
        assert_eq!(
            endpoint_for_rank("tcp://10.0.0.1:5557", 1),
            "tcp://10.0.0.1:5558"
        );
        assert_eq!(endpoint_for_rank("tcp://host:port", 1), "tcp://host:port");
        assert_eq!(endpoint_for_rank("ipc:///tmp/kv", 1), "ipc:///tmp/kv");
    }

    /// The golden batches convert to exactly what the Python relay produced
    /// for them: sha256 hashes reduced to their low 64 bits, an unaligned
    /// store skipped but its event id consumed, `lora_id` and the parent
    /// carried, `dp_rank` set only when the publisher set it.
    #[test]
    fn golden_batches_convert_like_the_python_relay() {
        let mut event_id = 0;
        let batch = convert_batch(decode(golden::BATCH1), 9, &mut event_id);
        assert_eq!(event_id, 5);
        assert_eq!(batch.sequence_number, 9);
        assert!((batch.timestamp - 1_700_000_000.5).abs() < f64::EPSILON);
        assert_eq!(batch.dp_rank, None);
        assert_eq!(
            batch
                .events
                .iter()
                .map(|event| event.event_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 5]
        );
        let first = stored(&batch.events[0]);
        assert_eq!(first.parent_block_hash, Some(7));
        assert_eq!(
            first.blocks,
            vec![
                block(i64::MIN, vec![1, 2, 3, 4], None),
                block(-2, vec![5, 6, 7, 8], None),
            ]
        );
        let second = stored(&batch.events[1]);
        assert_eq!(second.parent_block_hash, None);
        assert_eq!(second.blocks, vec![block(0x1234, vec![9, 10], Some(3))]);
        assert_eq!(
            batch.events[2].data,
            Some(kv_cache_event::Data::Removed(common::KvBlocksRemoved {
                block_hashes: vec![0x1234, -1],
                cache_level: None,
                tier: Some(KvCacheTier::Device as i32),
                medium: Some("GPU".to_string()),
                locality: Some(KvCacheLocality::Local as i32),
                ..Default::default()
            }))
        );
        assert_eq!(
            batch.events[3].data,
            Some(kv_cache_event::Data::Cleared(
                common::KvCacheCleared::default()
            ))
        );

        let batch = convert_batch(decode(golden::BATCH2), 10, &mut event_id);
        assert_eq!(event_id, 6);
        assert_eq!(batch.sequence_number, 10);
        assert_eq!(batch.dp_rank, Some(1));
        assert_eq!(batch.events[0].event_id, 6);
        let only = stored(&batch.events[0]);
        assert_eq!(only.parent_block_hash, Some(41));
        assert_eq!(only.blocks, vec![block(42, vec![100, 101], None)]);
    }

    /// The batch array may omit the trailing rank and may grow new fields;
    /// an event of a type this relay does not convert is skipped on its own
    /// (consuming its event id), as the Python relay skips unknown types.
    #[test]
    fn batch_layout_tolerates_an_omitted_rank_and_trailing_fields() {
        let short = rmp_serde::to_vec(&(1.5f64, Vec::<u8>::new())).unwrap();
        let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&short)
            .unwrap()
            .0;
        assert!(batch.events.is_empty());
        assert_eq!(batch.dp_rank, None);

        let long = rmp_serde::to_vec(&(1.5f64, Vec::<u8>::new(), 2i32, "future")).unwrap();
        let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&long)
            .unwrap()
            .0;
        assert_eq!(batch.dp_rank, Some(2));

        let unknown = rmp_serde::to_vec(&serde_json::json!([
            1.5,
            [{"type": "Mystery", "x": 1}, {"type": "AllBlocksCleared"}]
        ]))
        .unwrap();
        let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&unknown)
            .unwrap()
            .0;
        let mut event_id = 4;
        let converted = convert_batch(batch, 7, &mut event_id);
        assert_eq!(event_id, 6, "the skipped event still consumed an id");
        assert_eq!(converted.events.len(), 1);
        assert_eq!(converted.events[0].event_id, 6);
        assert!(matches!(
            converted.events[0].data,
            Some(kv_cache_event::Data::Cleared(_))
        ));
    }

    #[test]
    fn frames_are_split_like_the_python_relay() {
        let payload = golden::bytes(golden::BATCH2);
        let message = golden::frame(b"kv", 5, &payload);
        let (sequence, body) = split_frames(&message).expect("three frames");
        assert_eq!(sequence, 5);
        assert_eq!(body, payload.as_slice());

        let mut short = ZmqMessage::from(b"kv".to_vec());
        short.push_back(5u64.to_be_bytes().to_vec().into());
        assert!(split_frames(&short).is_none());

        // A shorter sequence frame zero-extends, as `int.from_bytes` does.
        let mut narrow = ZmqMessage::from(b"kv".to_vec());
        narrow.push_back(vec![1, 2].into());
        narrow.push_back(payload.into());
        assert_eq!(
            split_frames(&narrow).map(|(sequence, _)| sequence),
            Some(0x0102)
        );
    }

    /// A local publisher's frames come out as proto batches under the
    /// publisher's sequence numbers; short frames, undecodable payloads and
    /// other topics are skipped; dropping the stream closes the connection.
    #[tokio::test]
    async fn relays_a_local_publisher_and_closes_with_the_stream() {
        let mut publisher = PubSocket::new();
        let mut monitor = publisher.monitor();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string();
        let mut stream = relay(endpoint, "kv".to_string());
        let batch1 = golden::bytes(golden::BATCH1);
        let batch2 = golden::bytes(golden::BATCH2);

        // The subscription reaches the publisher a moment after the connect;
        // probe with sequence 0 until a batch comes through.
        let mut first = None;
        for _ in 0..200 {
            publisher
                .send(golden::frame(b"kv", 0, &batch1))
                .await
                .expect("publish");
            if let Ok(item) = timeout(Duration::from_millis(50), stream.next()).await {
                first = Some(item.expect("stream open").expect("a batch"));
                break;
            }
        }
        let first = first.expect("the subscription went live");
        assert_eq!(first.sequence_number, 0);
        assert_eq!(first.events.len(), 4);

        let mut short = ZmqMessage::from(b"kv".to_vec());
        short.push_back(1u64.to_be_bytes().to_vec().into());
        publisher.send(short).await.expect("publish");
        publisher
            .send(golden::frame(b"kv", 2, b"not msgpack"))
            .await
            .expect("publish");
        publisher
            .send(golden::frame(b"other", 3, &batch2))
            .await
            .expect("publish");
        publisher
            .send(golden::frame(b"kv", 4, &batch2))
            .await
            .expect("publish");
        // Probe duplicates may still be queued; the next new batch is 4.
        let next = loop {
            let batch = timeout(Duration::from_secs(5), stream.next())
                .await
                .expect("a batch in time")
                .expect("stream open")
                .expect("a batch");
            if batch.sequence_number != 0 {
                break batch;
            }
        };
        assert_eq!(next.sequence_number, 4);
        assert_eq!(next.dp_rank, Some(1));
        assert_eq!(stored(&next.events[0]).blocks[0].block_hash, 42);

        // A publisher restart (its counter starts over) ends the stream with
        // DATA_LOSS, which the gateway answers by clearing and resubscribing.
        publisher
            .send(golden::frame(b"kv", 1, &batch2))
            .await
            .expect("publish");
        let status = timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("an item in time")
            .expect("stream open")
            .expect_err("a restart is an error");
        assert_eq!(status.code(), tonic::Code::DataLoss);
        assert!(status
            .message()
            .contains("restarted its sequence at 1 after 4"));
        assert!(timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("the end in time")
            .is_none());

        drop(stream);
        let disconnected = timeout(Duration::from_secs(5), async {
            while let Some(event) = monitor.next().await {
                if matches!(event, SocketEvent::Disconnected(_)) {
                    return true;
                }
            }
            false
        })
        .await
        .expect("the publisher notices in time");
        assert!(disconnected);
    }
}
