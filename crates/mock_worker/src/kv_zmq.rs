//! A vLLM-wire KV-event publisher fed by a simulated engine: what
//! `ZmqEventPublisher` in `vllm/distributed/kv_events.py` puts on the wire,
//! so the servicers' relays (Rust and Python) can be exercised end to end
//! without a GPU.
//!
//! - One PUB socket per worker; one multipart message per pass:
//!   `[topic, sequence as u64 big-endian, msgpack payload]`, the sequence
//!   counting from 0 per publisher.
//! - The payload is vLLM's `EventBatch`, array-like: `[ts, events,
//!   data_parallel_rank]`. Events are tagged maps in vLLM's field order with
//!   its `omit_defaults`: a `BlockStored` carries `type`, `block_hashes`,
//!   `parent_block_hash`, `token_ids`, `block_size`, `lora_id`, `medium`,
//!   `lora_name` (required, nil when unset) and then `group_idx` and
//!   `kv_cache_spec_kind`; a `BlockRemoved` carries `type`, `block_hashes`,
//!   `medium`, `group_idx`; `AllBlocksCleared` only its tag. Block hashes
//!   are unsigned 64-bit integers, as vLLM's int form.
//! - Optional replay on a ROUTER socket at the next port: a request's last
//!   frame is the start sequence (8 bytes big-endian); the reply is every
//!   buffered batch from it as `[routing…, topic, seq, payload]` and then
//!   `[routing…, b"", END, b""]`, END being eight 0xff bytes ((-1) signed);
//!   the last `buffer_steps` batches are kept.

use std::{
    collections::VecDeque,
    time::{SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use rmpv::Value;
use smg_grpc_client::common_proto as common;
use zeromq::{
    prelude::{Socket, SocketRecv, SocketSend},
    PubSocket, RouterSocket, ZmqError, ZmqMessage,
};

use crate::engine::Engine;

/// vLLM's end-of-replay marker: `(-1).to_bytes(8, "big", signed=True)`.
pub const END_SEQ: [u8; 8] = [0xff; 8];

/// Where and how one worker publishes.
#[derive(Clone, Debug)]
pub struct KvZmqConfig {
    pub host: String,
    /// PUB port; the replay ROUTER, when enabled, binds `port + 1`.
    pub port: u16,
    pub replay: bool,
    pub topic: String,
    pub buffer_steps: usize,
    pub dp_rank: i32,
}

/// Bind the sockets and publish the engine's events until its event channel
/// closes.
pub async fn serve(engine: Engine, cfg: KvZmqConfig) {
    let mut publisher = PubSocket::new();
    let endpoint = format!("tcp://{}:{}", cfg.host, cfg.port);
    if let Err(e) = publisher.bind(&endpoint).await {
        tracing::error!("kv-events publisher failed to bind {endpoint}: {e}");
        return;
    }
    let mut replay = None;
    if cfg.replay {
        let mut router = RouterSocket::new();
        let replay_endpoint = format!("tcp://{}:{}", cfg.host, cfg.port.saturating_add(1));
        if let Err(e) = router.bind(&replay_endpoint).await {
            tracing::error!("kv-events replay failed to bind {replay_endpoint}: {e}");
            return;
        }
        replay = Some(router);
    }
    tracing::info!(
        "kv-events publisher {} on {endpoint} (replay {}, topic {:?})",
        engine.name(),
        if cfg.replay { "on" } else { "off" },
        cfg.topic
    );
    run(engine, cfg, publisher, replay).await;
}

/// Publish on already-bound sockets (tests bind port 0 and read it back).
pub async fn run(
    engine: Engine,
    cfg: KvZmqConfig,
    mut publisher: PubSocket,
    mut replay: Option<RouterSocket>,
) {
    let mut events = engine.subscribe_kv(0);
    let mut state = Publisher::new(cfg.topic.into_bytes(), cfg.buffer_steps, cfg.dp_rank);
    loop {
        tokio::select! {
            batch = events.next() => {
                let Some(Ok(batch)) = batch else { break };
                let message = state.publish(&batch);
                if let Err(e) = publisher.send(message).await {
                    tracing::warn!("kv-events publish failed: {e}");
                }
            }
            request = recv_replay(&mut replay) => {
                let Ok(request) = request else { continue };
                let Some(router) = replay.as_mut() else { continue };
                for reply in state.replay(&request) {
                    if let Err(e) = router.send(reply).await {
                        tracing::warn!("kv-events replay send failed: {e}");
                        break;
                    }
                }
            }
        }
    }
}

/// The next replay request, or never when replay is off.
async fn recv_replay(replay: &mut Option<RouterSocket>) -> Result<ZmqMessage, ZmqError> {
    match replay {
        Some(router) => router.recv().await,
        None => std::future::pending().await,
    }
}

/// The publisher's own state: sequence counter and replay buffer.
pub struct Publisher {
    topic: Vec<u8>,
    seq: u64,
    buffer: VecDeque<(u64, Vec<u8>)>,
    buffer_steps: usize,
    dp_rank: i32,
}

impl Publisher {
    pub fn new(topic: Vec<u8>, buffer_steps: usize, dp_rank: i32) -> Self {
        Self {
            topic,
            seq: 0,
            buffer: VecDeque::new(),
            buffer_steps: buffer_steps.max(1),
            dp_rank,
        }
    }

    /// The next sequence number to be published.
    pub fn next_seq(&self) -> u64 {
        self.seq
    }

    /// Encode `batch`, assign it the next sequence, keep it for replay and
    /// return the PUB message.
    pub fn publish(&mut self, batch: &common::KvEventBatch) -> ZmqMessage {
        let payload = encode_batch(batch, self.dp_rank);
        let seq = self.seq;
        self.seq += 1;
        self.buffer.push_back((seq, payload.clone()));
        while self.buffer.len() > self.buffer_steps {
            self.buffer.pop_front();
        }
        frame(&self.topic, seq, payload)
    }

    /// A publisher restart: the sequence starts over and the buffer is gone.
    pub fn restart(&mut self) {
        self.seq = 0;
        self.buffer.clear();
    }

    /// Replies to a replay request as vLLM frames them (see the module doc);
    /// a request without an 8-byte start sequence gets no reply.
    pub fn replay(&self, request: &ZmqMessage) -> Vec<ZmqMessage> {
        let frames: Vec<Vec<u8>> = request.iter().map(|f| f.to_vec()).collect();
        let Some((start, routing)) = frames.split_last() else {
            return Vec::new();
        };
        if start.len() != 8 || routing.is_empty() {
            return Vec::new();
        }
        let start = u64::from_be_bytes([
            start[0], start[1], start[2], start[3], start[4], start[5], start[6], start[7],
        ]);
        let mut replies = Vec::new();
        for (seq, payload) in self.buffer.iter().filter(|(seq, _)| *seq >= start) {
            let mut message = routed(routing);
            message.push_back(self.topic.clone().into());
            message.push_back(seq.to_be_bytes().to_vec().into());
            message.push_back(payload.clone().into());
            replies.push(message);
        }
        let mut end = routed(routing);
        end.push_back(Vec::new().into());
        end.push_back(END_SEQ.to_vec().into());
        end.push_back(Vec::new().into());
        replies.push(end);
        replies
    }
}

/// A reply's routing prefix (the request's frames before the start sequence:
/// the peer identity and the empty delimiter).
fn routed(routing: &[Vec<u8>]) -> ZmqMessage {
    let mut message = ZmqMessage::from(routing[0].clone());
    for frame in &routing[1..] {
        message.push_back(frame.clone().into());
    }
    message
}

/// A PUB message: `[topic, sequence (u64 big-endian), payload]`.
pub fn frame(topic: &[u8], seq: u64, payload: Vec<u8>) -> ZmqMessage {
    let mut message = ZmqMessage::from(topic.to_vec());
    message.push_back(seq.to_be_bytes().to_vec().into());
    message.push_back(payload.into());
    message
}

fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// vLLM's `EventBatch` for a proto batch: `[ts, events, data_parallel_rank]`.
pub fn encode_batch(batch: &common::KvEventBatch, dp_rank: i32) -> Vec<u8> {
    let events: Vec<Value> = batch
        .events
        .iter()
        .filter_map(|event| match &event.data {
            Some(common::kv_cache_event::Data::Stored(stored)) => Some(stored_map(stored)),
            Some(common::kv_cache_event::Data::Removed(removed)) => Some(removed_map(removed)),
            Some(common::kv_cache_event::Data::Cleared(_)) => Some(cleared_map()),
            None => None,
        })
        .collect();
    let value = Value::Array(vec![
        Value::F64(unix_seconds()),
        Value::Array(events),
        Value::from(dp_rank),
    ]);
    let mut buf = Vec::new();
    // Writing into a Vec cannot fail.
    let _ = rmpv::encode::write_value(&mut buf, &value);
    buf
}

fn key(name: &str) -> Value {
    Value::String(name.into())
}

fn hash_value(hash: i64) -> Value {
    // vLLM's int form is the unsigned low 64 bits; the proto carries the same
    // bits as a signed value.
    Value::from(hash as u64)
}

/// `BlockStored` as vLLM encodes it: the block_size of the first block, the
/// token ids of every block in order.
fn stored_map(stored: &common::KvBlocksStored) -> Value {
    let block_size = stored
        .blocks
        .first()
        .map(|b| i64::from(b.block_size))
        .unwrap_or(0);
    let token_ids: Vec<Value> = stored
        .blocks
        .iter()
        .flat_map(|b| b.token_ids.iter().map(|t| Value::from(*t)))
        .collect();
    Value::Map(vec![
        (key("type"), Value::String("BlockStored".into())),
        (
            key("block_hashes"),
            Value::Array(
                stored
                    .blocks
                    .iter()
                    .map(|b| hash_value(b.block_hash))
                    .collect(),
            ),
        ),
        (
            key("parent_block_hash"),
            stored
                .parent_block_hash
                .map(hash_value)
                .unwrap_or(Value::Nil),
        ),
        (key("token_ids"), Value::Array(token_ids)),
        (key("block_size"), Value::from(block_size)),
        (key("lora_id"), Value::Nil),
        (key("medium"), Value::String("GPU".into())),
        (key("lora_name"), Value::Nil),
        (key("group_idx"), Value::from(0)),
        (
            key("kv_cache_spec_kind"),
            Value::String("full_attention".into()),
        ),
    ])
}

fn removed_map(removed: &common::KvBlocksRemoved) -> Value {
    Value::Map(vec![
        (key("type"), Value::String("BlockRemoved".into())),
        (
            key("block_hashes"),
            Value::Array(
                removed
                    .block_hashes
                    .iter()
                    .map(|h| hash_value(*h))
                    .collect(),
            ),
        ),
        (key("medium"), Value::String("GPU".into())),
        (key("group_idx"), Value::from(0)),
    ])
}

fn cleared_map() -> Value {
    Value::Map(vec![(
        key("type"),
        Value::String("AllBlocksCleared".into()),
    )])
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_servicer::kv_wire::{Normalizer, WireBatch};
    use tokio::time::timeout;
    use zeromq::{DealerSocket, SubSocket};

    use super::*;
    use crate::engine::{EngineParams, GenEvent, NewRequest};

    fn stored(
        hashes: &[i64],
        parent: Option<i64>,
        tokens_per_block: &[&[u32]],
    ) -> common::KvCacheEvent {
        common::KvCacheEvent {
            event_id: 1,
            data: Some(common::kv_cache_event::Data::Stored(
                common::KvBlocksStored {
                    blocks: hashes
                        .iter()
                        .zip(tokens_per_block)
                        .map(|(hash, tokens)| common::KvBlock {
                            block_hash: *hash,
                            token_ids: tokens.to_vec(),
                            block_size: tokens.len() as i32,
                            ..Default::default()
                        })
                        .collect(),
                    parent_block_hash: parent,
                    ..Default::default()
                },
            )),
        }
    }

    fn removed(hashes: &[i64]) -> common::KvCacheEvent {
        common::KvCacheEvent {
            event_id: 2,
            data: Some(common::kv_cache_event::Data::Removed(
                common::KvBlocksRemoved {
                    block_hashes: hashes.to_vec(),
                    ..Default::default()
                },
            )),
        }
    }

    fn cleared() -> common::KvCacheEvent {
        common::KvCacheEvent {
            event_id: 3,
            data: Some(common::kv_cache_event::Data::Cleared(
                common::KvCacheCleared::default(),
            )),
        }
    }

    fn sample_batch() -> common::KvEventBatch {
        common::KvEventBatch {
            sequence_number: 9,
            timestamp: 0.0,
            events: vec![
                stored(&[11, -2], Some(10), &[&[1, 2, 3, 4], &[5, 6, 7, 8]]),
                removed(&[11]),
                cleared(),
            ],
            dp_rank: Some(0),
        }
    }

    fn keys(map: &Value) -> Vec<String> {
        match map {
            Value::Map(entries) => entries
                .iter()
                .map(|(k, _)| k.as_str().expect("string key").to_string())
                .collect(),
            other => panic!("not a map: {other:?}"),
        }
    }

    fn field<'a>(map: &'a Value, name: &str) -> &'a Value {
        match map {
            Value::Map(entries) => entries
                .iter()
                .find(|(k, _)| k.as_str() == Some(name))
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("no field {name}")),
            other => panic!("not a map: {other:?}"),
        }
    }

    #[test]
    fn encodes_vllm_event_batches_field_for_field() {
        let payload = encode_batch(&sample_batch(), 3);
        let value = rmpv::decode::read_value(&mut payload.as_slice()).expect("msgpack");
        let Value::Array(batch) = value else {
            panic!("batch is array-like");
        };
        assert_eq!(batch.len(), 3, "[ts, events, data_parallel_rank]");
        assert!(batch[0].as_f64().is_some_and(|ts| ts > 1.7e9));
        assert_eq!(batch[2].as_i64(), Some(3));
        let events = batch[1].as_array().expect("events");
        assert_eq!(events.len(), 3);

        assert_eq!(
            keys(&events[0]),
            [
                "type",
                "block_hashes",
                "parent_block_hash",
                "token_ids",
                "block_size",
                "lora_id",
                "medium",
                "lora_name",
                "group_idx",
                "kv_cache_spec_kind"
            ]
        );
        assert_eq!(field(&events[0], "type").as_str(), Some("BlockStored"));
        let hashes = field(&events[0], "block_hashes")
            .as_array()
            .expect("hashes");
        assert_eq!(hashes[0].as_u64(), Some(11));
        assert_eq!(
            hashes[1].as_u64(),
            Some(u64::MAX - 1),
            "a negative proto hash is the same bits as vLLM's unsigned int"
        );
        assert_eq!(field(&events[0], "parent_block_hash").as_u64(), Some(10));
        assert_eq!(
            field(&events[0], "token_ids").as_array().map(Vec::len),
            Some(8),
            "token ids of every block, in order"
        );
        assert_eq!(field(&events[0], "block_size").as_i64(), Some(4));
        assert!(field(&events[0], "lora_id").is_nil());
        assert_eq!(field(&events[0], "medium").as_str(), Some("GPU"));
        assert!(field(&events[0], "lora_name").is_nil());
        assert_eq!(field(&events[0], "group_idx").as_i64(), Some(0));
        assert_eq!(
            field(&events[0], "kv_cache_spec_kind").as_str(),
            Some("full_attention")
        );

        assert_eq!(
            keys(&events[1]),
            ["type", "block_hashes", "medium", "group_idx"]
        );
        assert_eq!(field(&events[1], "type").as_str(), Some("BlockRemoved"));
        assert_eq!(keys(&events[2]), ["type"]);
        assert_eq!(field(&events[2], "type").as_str(), Some("AllBlocksCleared"));
    }

    #[test]
    fn the_relay_normalizes_the_payload_back_into_the_proto() {
        let payload = encode_batch(&sample_batch(), 0);
        let wire: WireBatch = rmp_serde::from_slice(&payload).expect("the relay decodes it");
        let mut event_id = 0;
        let relayed = Normalizer::new().normalize_batch(wire, 7, &mut event_id);
        assert_eq!(relayed.sequence_number, 7);
        let kinds: Vec<&str> = relayed
            .events
            .iter()
            .map(|e| match &e.data {
                Some(common::kv_cache_event::Data::Stored(_)) => "stored",
                Some(common::kv_cache_event::Data::Removed(_)) => "removed",
                Some(common::kv_cache_event::Data::Cleared(_)) => "cleared",
                None => "none",
            })
            .collect();
        assert_eq!(kinds, ["stored", "removed", "cleared"]);
        let Some(common::kv_cache_event::Data::Stored(stored)) = &relayed.events[0].data else {
            unreachable!()
        };
        assert_eq!(
            stored
                .blocks
                .iter()
                .map(|b| b.block_hash)
                .collect::<Vec<_>>(),
            [11, -2],
            "hashes round-trip to the proto's signed identity"
        );
        assert_eq!(stored.parent_block_hash, Some(10));
        assert_eq!(stored.blocks[1].token_ids, [5, 6, 7, 8]);
        let Some(common::kv_cache_event::Data::Removed(removed)) = &relayed.events[1].data else {
            unreachable!()
        };
        assert_eq!(removed.block_hashes, [11]);
    }

    #[test]
    fn replay_answers_from_the_start_sequence_and_ends_with_the_marker() {
        let mut publisher = Publisher::new(b"kv".to_vec(), 3, 0);
        for _ in 0..5 {
            let message = publisher.publish(&sample_batch());
            assert_eq!(message.len(), 3);
            assert_eq!(message.get(0).map(|t| t.as_ref()), Some(&b"kv"[..]));
        }
        assert_eq!(publisher.next_seq(), 5);

        let mut request = ZmqMessage::from(b"peer".to_vec());
        request.push_back(Vec::new().into());
        request.push_back(1u64.to_be_bytes().to_vec().into());
        let replies = publisher.replay(&request);
        // The buffer keeps the last three (2, 3, 4); 1 is gone.
        let seqs: Vec<u64> = replies[..replies.len() - 1]
            .iter()
            .map(|m| {
                assert_eq!(m.len(), 5, "[peer, empty, topic, seq, payload]");
                assert_eq!(m.get(2).map(|t| t.as_ref()), Some(&b"kv"[..]));
                let seq = m.get(3).expect("seq");
                u64::from_be_bytes(seq.as_ref().try_into().expect("8 bytes"))
            })
            .collect();
        assert_eq!(seqs, [2, 3, 4]);
        let end = replies.last().expect("end marker");
        assert_eq!(end.len(), 5, "[peer, empty, empty, END, empty]");
        assert_eq!(end.get(0).map(|t| t.as_ref()), Some(&b"peer"[..]));
        assert!(end.get(2).is_some_and(|f| f.is_empty()));
        assert_eq!(end.get(3).map(|t| t.as_ref()), Some(&END_SEQ[..]));
        assert!(end.get(4).is_some_and(|f| f.is_empty()));

        let mut late = ZmqMessage::from(b"peer".to_vec());
        late.push_back(Vec::new().into());
        late.push_back(99u64.to_be_bytes().to_vec().into());
        assert_eq!(publisher.replay(&late).len(), 1, "only the end marker");

        let mut bad = ZmqMessage::from(b"peer".to_vec());
        bad.push_back(vec![1, 2, 3, 4].into());
        assert!(
            publisher.replay(&bad).is_empty(),
            "a malformed request is ignored"
        );

        publisher.restart();
        assert_eq!(publisher.next_seq(), 0);
        assert_eq!(
            publisher.replay(&request).len(),
            1,
            "the buffer is gone after a restart"
        );
    }

    #[tokio::test]
    async fn a_subscriber_gets_live_frames_and_can_replay_what_it_missed() {
        let engine = Engine::spawn(EngineParams::default());
        let mut pub_socket = PubSocket::new();
        let endpoint = pub_socket
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("pub binds")
            .to_string();
        let mut router = RouterSocket::new();
        let replay_endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("router binds")
            .to_string();
        let cfg = KvZmqConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            replay: true,
            topic: "kv".to_string(),
            buffer_steps: 100,
            dp_rank: 0,
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "the publisher ends with the engine when the test drops it"
        )]
        let _publisher = tokio::spawn(run(engine.clone(), cfg, pub_socket, Some(router)));

        let mut sub = SubSocket::new();
        sub.subscribe("kv").await.expect("subscribe");
        sub.connect(&endpoint).await.expect("connect");

        // The subscription reaches the publisher a moment after the connect;
        // keep the engine producing passes until a frame comes through.
        let mut receivers = Vec::new();
        let mut first = None;
        for i in 0..200 {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<GenEvent>();
            receivers.push(rx);
            engine.submit(NewRequest {
                request_id: format!("r{i}"),
                prompt_token_ids: (0..64).map(|t| t + i * 64).collect(),
                max_new: 1,
                events: tx,
            });
            if let Ok(Ok(message)) = timeout(Duration::from_millis(50), sub.recv()).await {
                first = Some(message);
                break;
            }
        }
        let first = first.expect("a live frame arrived");
        assert_eq!(first.len(), 3);
        assert_eq!(first.get(0).map(|t| t.as_ref()), Some(&b"kv"[..]));
        let live_seq = u64::from_be_bytes(
            first
                .get(1)
                .expect("seq")
                .as_ref()
                .try_into()
                .expect("8 bytes"),
        );
        let wire: WireBatch =
            rmp_serde::from_slice(first.get(2).expect("payload")).expect("decodes");
        assert!(!wire.events.is_empty());

        // Everything the subscriber missed is in the replay buffer, from 0.
        let mut dealer = DealerSocket::new();
        dealer
            .connect(&replay_endpoint)
            .await
            .expect("dealer connects");
        let mut request = ZmqMessage::from(Vec::new());
        request.push_back(0u64.to_be_bytes().to_vec().into());
        dealer.send(request).await.expect("replay request");
        let mut replayed = Vec::new();
        loop {
            let reply = timeout(Duration::from_secs(5), dealer.recv())
                .await
                .expect("replay answers")
                .expect("reply");
            assert_eq!(reply.len(), 4, "[empty, topic, seq, payload]");
            let seq = reply.get(2).expect("seq");
            if seq.as_ref() == END_SEQ {
                break;
            }
            replayed.push(u64::from_be_bytes(
                seq.as_ref().try_into().expect("8 bytes"),
            ));
            let _: WireBatch =
                rmp_serde::from_slice(reply.get(3).expect("payload")).expect("decodes");
        }
        assert_eq!(replayed[0], 0, "replay starts at the requested sequence");
        assert!(replayed.windows(2).all(|w| w[1] == w[0] + 1), "contiguous");
        assert!(
            replayed.contains(&live_seq),
            "the live frame is in the buffer too"
        );
        drop(receivers);
    }
}
