//! The relay's decoder and normalizer against engine-encoded batches, and
//! the round trip from those batches through the proto into the gateway's
//! positional index.
//!
//! `tests/fixtures/kv_events/manifest.json` lists fixture sets that
//! `generate.py` encodes with msgspec, the engines' own encoder, in the
//! current tagged-map layout and the legacy array layout, each with the
//! normalizer's expected output. Real captures drop in the same way: point
//! `SMG_KV_EVENT_CAPTURES` at a directory holding one publisher payload per
//! file and, optionally, a `manifest.json` of the same shape. With a manifest
//! they are checked like the generated sets; without one every `*.msgpack`
//! file in name order is one stream that must decode without losing a batch.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr
)]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use engine_servicer::kv_wire::{Counts, Normalizer, WireBatch};
use engine_zmq_client::codec::TrailingTolerant;
use kv_index::{
    compute_request_content_hashes,
    salt::{content_hash_with_seed, namespace_seed, namespaced_request_content_hashes},
    ApplyError, ContentHash, PositionalIndexer, SequenceHash, StoredBlock, WorkerBlockMap,
};
use serde::Deserialize;
use smg_grpc_client::common_proto::{
    kv_block_extra_key, kv_cache_event, KvBlocksStored, KvCacheEvent, KvCacheTier, KvEventBatch,
};

#[derive(Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    engine: String,
    layout: String,
    files: Vec<String>,
    #[serde(default)]
    expect: Option<Expect>,
}

#[derive(Deserialize)]
struct Expect {
    forwarded: Vec<Forwarded>,
    counts: ExpectedCounts,
}

#[derive(Deserialize)]
struct ExpectedCounts {
    forwarded_stored: u64,
    forwarded_removed: u64,
    forwarded_cleared: u64,
    duplicate_stores: u64,
    bigram_stores: u64,
    #[serde(default)]
    dropped: BTreeMap<String, u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Forwarded {
    Stored {
        dp_rank: Option<i32>,
        hashes: Vec<i64>,
        parent: Option<i64>,
        tier: String,
        cache_level: Option<i32>,
        tokens: Vec<Vec<u32>>,
        lora_name: Option<String>,
        cache_salt: Option<String>,
        group_idx: Option<u32>,
        #[serde(default)]
        session_id: Option<String>,
        #[serde(default)]
        extra_keys: Option<Vec<Vec<ExpectedKey>>>,
    },
    Removed {
        dp_rank: Option<i32>,
        hashes: Vec<i64>,
        tier: String,
        cache_level: Option<i32>,
    },
    Cleared {
        dp_rank: Option<i32>,
    },
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExpectedKey {
    Text(String),
    Number(i64),
    BlobLen(usize),
    Multimodal(String, i64),
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kv_events")
}

fn load_manifest(dir: &Path) -> Manifest {
    let text = fs::read_to_string(dir.join("manifest.json")).expect("manifest.json");
    serde_json::from_str(&text).expect("a well-formed manifest")
}

fn decode(bytes: &[u8]) -> Result<WireBatch, rmp_serde::decode::Error> {
    rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(bytes).map(|batch| batch.0)
}

/// Decode and normalize one stream of payload files in order.
fn run(dir: &Path, files: &[String]) -> (Vec<KvEventBatch>, Counts) {
    let mut normalizer = Normalizer::new();
    let mut event_id = 0;
    let batches = files
        .iter()
        .enumerate()
        .map(|(seq, file)| {
            let bytes = fs::read(dir.join(file)).unwrap_or_else(|error| panic!("{file}: {error}"));
            let batch = decode(&bytes).unwrap_or_else(|error| panic!("{file}: {error}"));
            normalizer.normalize_batch(batch, seq as u64, &mut event_id)
        })
        .collect();
    (batches, normalizer.counts().clone())
}

fn tier_named(name: &str) -> KvCacheTier {
    match name {
        "device" => KvCacheTier::Device,
        "host" => KvCacheTier::Host,
        "disk" => KvCacheTier::Disk,
        "external" => KvCacheTier::External,
        other => panic!("unknown tier {other:?} in the manifest"),
    }
}

fn check(fixture: &Fixture, expect: &Expect, batches: &[KvEventBatch], counts: &Counts) {
    let name = &fixture.name;
    let forwarded: Vec<(Option<i32>, &KvCacheEvent)> = batches
        .iter()
        .flat_map(|batch| batch.events.iter().map(move |event| (batch.dp_rank, event)))
        .collect();
    assert_eq!(
        forwarded.len(),
        expect.forwarded.len(),
        "{name}: forwarded event count; got {:#?}",
        forwarded.iter().map(|(_, event)| event).collect::<Vec<_>>()
    );
    for (index, ((dp_rank, event), expected)) in forwarded.iter().zip(&expect.forwarded).enumerate()
    {
        let at = format!("{name} forwarded event {index} (id {})", event.event_id);
        match (&event.data, expected) {
            (
                Some(kv_cache_event::Data::Stored(stored)),
                Forwarded::Stored {
                    dp_rank: want_rank,
                    hashes,
                    parent,
                    tier,
                    cache_level,
                    tokens,
                    lora_name,
                    cache_salt,
                    group_idx,
                    session_id,
                    extra_keys,
                },
            ) => {
                assert_eq!(dp_rank, want_rank, "{at}: dp_rank");
                let got_hashes: Vec<i64> = stored.blocks.iter().map(|b| b.block_hash).collect();
                assert_eq!(&got_hashes, hashes, "{at}: hashes");
                assert_eq!(&stored.parent_block_hash, parent, "{at}: parent");
                assert_eq!(stored.tier, Some(tier_named(tier) as i32), "{at}: tier");
                let got_tokens: Vec<Vec<u32>> =
                    stored.blocks.iter().map(|b| b.token_ids.clone()).collect();
                assert_eq!(&got_tokens, tokens, "{at}: tokens");
                for block in &stored.blocks {
                    assert_eq!(&block.cache_level, cache_level, "{at}: cache_level");
                    assert_eq!(
                        block.block_size as usize,
                        block.token_ids.len(),
                        "{at}: block_size"
                    );
                }
                assert_eq!(&stored.lora_name, lora_name, "{at}: lora_name");
                assert_eq!(&stored.cache_salt, cache_salt, "{at}: cache_salt");
                assert_eq!(&stored.group_idx, group_idx, "{at}: group_idx");
                assert_eq!(&stored.session_id, session_id, "{at}: session_id");
                if let Some(expected_keys) = extra_keys {
                    let got_keys: Vec<Vec<ExpectedKey>> = stored
                        .blocks
                        .iter()
                        .map(|block| {
                            block
                                .extra_keys
                                .iter()
                                .map(|key| match key.key.clone().expect("a key") {
                                    kv_block_extra_key::Key::Text(text) => ExpectedKey::Text(text),
                                    kv_block_extra_key::Key::Number(number) => {
                                        ExpectedKey::Number(number)
                                    }
                                    kv_block_extra_key::Key::Blob(blob) => {
                                        ExpectedKey::BlobLen(blob.len())
                                    }
                                    kv_block_extra_key::Key::Multimodal(mm) => {
                                        ExpectedKey::Multimodal(mm.identifier, mm.offset)
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    assert_eq!(&got_keys, expected_keys, "{at}: extra_keys");
                }
            }
            (
                Some(kv_cache_event::Data::Removed(removed)),
                Forwarded::Removed {
                    dp_rank: want_rank,
                    hashes,
                    tier,
                    cache_level,
                },
            ) => {
                assert_eq!(dp_rank, want_rank, "{at}: dp_rank");
                assert_eq!(&removed.block_hashes, hashes, "{at}: hashes");
                assert_eq!(removed.tier, Some(tier_named(tier) as i32), "{at}: tier");
                assert_eq!(&removed.cache_level, cache_level, "{at}: cache_level");
            }
            (Some(kv_cache_event::Data::Cleared(_)), Forwarded::Cleared { dp_rank: want_rank }) => {
                assert_eq!(dp_rank, want_rank, "{at}: dp_rank");
            }
            (got, want) => panic!("{at}: got {got:?}, wanted {want:?}"),
        }
    }

    let want = &expect.counts;
    assert_eq!(
        counts.forwarded_stored, want.forwarded_stored,
        "{name}: forwarded_stored"
    );
    assert_eq!(
        counts.forwarded_removed, want.forwarded_removed,
        "{name}: forwarded_removed"
    );
    assert_eq!(
        counts.forwarded_cleared, want.forwarded_cleared,
        "{name}: forwarded_cleared"
    );
    assert_eq!(
        counts.duplicate_stores, want.duplicate_stores,
        "{name}: duplicate_stores"
    );
    assert_eq!(
        counts.bigram_stores, want.bigram_stores,
        "{name}: bigram_stores"
    );
    let dropped: BTreeMap<String, u64> = counts
        .dropped
        .iter()
        .map(|(reason, count)| (reason.as_str().to_string(), *count))
        .collect();
    assert_eq!(dropped, want.dropped, "{name}: dropped");
}

#[test]
fn generated_fixtures_normalize_as_expected() {
    let dir = fixtures_dir();
    let manifest = load_manifest(&dir);
    assert!(
        manifest.fixtures.len() >= 4,
        "both engines in both layouts; run generate.py"
    );
    for fixture in &manifest.fixtures {
        let (batches, counts) = run(&dir, &fixture.files);
        let expect = fixture
            .expect
            .as_ref()
            .unwrap_or_else(|| panic!("{}: generated fixtures carry expectations", fixture.name));
        check(fixture, expect, &batches, &counts);
    }
}

/// Real captures, when present: `SMG_KV_EVENT_CAPTURES=<dir>`.
#[test]
fn recorded_captures_decode_and_normalize() {
    let Some(dir) = std::env::var_os("SMG_KV_EVENT_CAPTURES").map(PathBuf::from) else {
        eprintln!("SMG_KV_EVENT_CAPTURES unset; no recorded captures to check");
        return;
    };
    if dir.join("manifest.json").exists() {
        let manifest = load_manifest(&dir);
        for fixture in &manifest.fixtures {
            let (batches, counts) = run(&dir, &fixture.files);
            match &fixture.expect {
                Some(expect) => check(fixture, expect, &batches, &counts),
                None => eprintln!(
                    "{} ({} {}): {} batches, {} events forwarded, counts {counts:?}",
                    fixture.name,
                    fixture.engine,
                    fixture.layout,
                    batches.len(),
                    batches.iter().map(|b| b.events.len()).sum::<usize>()
                ),
            }
        }
        return;
    }
    let mut files: Vec<String> = fs::read_dir(&dir)
        .expect("a readable capture directory")
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.ends_with(".msgpack"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no *.msgpack payloads in {}",
        dir.display()
    );
    let (batches, counts) = run(&dir, &files);
    eprintln!(
        "{}: {} batches, {} events forwarded, counts {counts:?}",
        dir.display(),
        batches.len(),
        batches.iter().map(|b| b.events.len()).sum::<usize>()
    );
}

// ---------------------------------------------------------------------------
// Round trip: wire -> proto -> index
// ---------------------------------------------------------------------------

/// Apply normalized events to the index the way the gateway's monitor does
/// for the device tier: stores hashed under the event's namespace, removals
/// by engine hash, clears per worker. Host residency is the monitor's own
/// bookkeeping and has its tests there; here host events are skipped.
fn index(batches: &[KvEventBatch], indexer: &PositionalIndexer, name: &str) -> BTreeMap<i32, u32> {
    let mut workers: BTreeMap<i32, (u32, WorkerBlockMap)> = BTreeMap::new();
    for batch in batches {
        let rank = batch.dp_rank.unwrap_or(0);
        let (worker, blocks) = workers.entry(rank).or_insert_with(|| {
            let worker = indexer
                .intern_worker(&format!("{name}/rank{rank}"))
                .expect("worker ids");
            (worker, WorkerBlockMap::default())
        });
        for event in &batch.events {
            match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => {
                    apply_stored(indexer, *worker, blocks, stored);
                }
                Some(kv_cache_event::Data::Removed(removed)) => {
                    if device_tier(removed.tier) {
                        let hashes: Vec<SequenceHash> = removed
                            .block_hashes
                            .iter()
                            .map(|&h| SequenceHash::from(h))
                            .collect();
                        indexer.apply_removed(*worker, &hashes, blocks);
                    }
                }
                Some(kv_cache_event::Data::Cleared(_)) => indexer.apply_cleared(*worker, blocks),
                None => {}
            }
        }
    }
    workers
        .into_iter()
        .map(|(rank, (worker, _))| (rank, worker))
        .collect()
}

fn apply_stored(
    indexer: &PositionalIndexer,
    worker: u32,
    blocks: &mut WorkerBlockMap,
    stored: &KvBlocksStored,
) {
    if !device_tier(stored.tier) {
        return;
    }
    let seed = namespace_seed(stored.lora_name.as_deref(), stored.cache_salt.as_deref());
    let converted: Vec<StoredBlock> = stored
        .blocks
        .iter()
        .map(|block| StoredBlock {
            seq_hash: SequenceHash::from(block.block_hash),
            content_hash: content_hash_with_seed(&block.token_ids, seed),
        })
        .collect();
    let parent = stored.parent_block_hash.map(SequenceHash::from);
    if let Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound) =
        indexer.apply_stored(worker, &converted, parent, blocks)
    {
        indexer
            .apply_stored(worker, &converted, None, blocks)
            .expect("a parentless store applies");
    }
}

fn device_tier(tier: Option<i32>) -> bool {
    matches!(
        KvCacheTier::try_from(tier.unwrap_or_default()),
        Ok(KvCacheTier::Device | KvCacheTier::Unspecified)
    )
}

fn depth(indexer: &PositionalIndexer, worker: u32, hashes: &[ContentHash]) -> u32 {
    indexer
        .find_matches(hashes, false)
        .scores
        .get(&worker)
        .copied()
        .unwrap_or(0)
}

#[test]
fn fixtures_round_trip_into_the_index() {
    let dir = fixtures_dir();
    let manifest = load_manifest(&dir);
    let tokens: Vec<u32> = (1..=16).collect();
    for fixture in &manifest.fixtures {
        let (batches, _) = run(&dir, &fixture.files);
        let indexer = PositionalIndexer::new(64);
        let workers = index(&batches, &indexer, &fixture.name);
        let name = &fixture.name;
        let rank0 = workers[&0];
        let rank1 = workers[&1];
        let plain = |n: usize| compute_request_content_hashes(&tokens[..n], 4);
        match fixture.engine.as_str() {
            "vllm" => {
                // The chain of two device blocks survives the duplicate copy,
                // its per-copy removals and the reset (it is stored again).
                assert_eq!(depth(&indexer, rank0, &plain(8)), 2, "{name}: plain chain");
                // The digest-hashed continuation was cleared and not restored.
                assert_eq!(
                    depth(&indexer, rank0, &plain(16)),
                    2,
                    "{name}: cleared tail"
                );
                // The LoRA + salt chain matches only under its namespace, the
                // child included (it inherited the salt from block 0).
                let salted = namespaced_request_content_hashes(
                    &tokens[..8],
                    4,
                    Some("adapter"),
                    Some("salt-1"),
                );
                assert_eq!(depth(&indexer, rank0, &salted), 2, "{name}: salted chain");
                let lora_only =
                    namespaced_request_content_hashes(&tokens[..8], 4, Some("adapter"), None);
                assert_eq!(
                    depth(&indexer, rank0, &lora_only),
                    0,
                    "{name}: lora without salt"
                );
                let salt_only =
                    namespaced_request_content_hashes(&tokens[..8], 4, None, Some("salt-1"));
                assert_eq!(
                    depth(&indexer, rank0, &salt_only),
                    0,
                    "{name}: salt without lora"
                );
                assert_eq!(depth(&indexer, rank1, &plain(8)), 2, "{name}: rank 1");
            }
            "sglang" => {
                // Device chain of three pages; the HiCache demote and host
                // eviction left the load-back copy in place.
                assert_eq!(depth(&indexer, rank0, &plain(12)), 3, "{name}: plain chain");
                if fixture.layout == "map" {
                    let salted =
                        namespaced_request_content_hashes(&tokens[..8], 4, None, Some("tenant-a"));
                    assert_eq!(depth(&indexer, rank0, &salted), 2, "{name}: salted chain");
                    let other =
                        namespaced_request_content_hashes(&tokens[..8], 4, None, Some("tenant-b"));
                    assert_eq!(depth(&indexer, rank0, &other), 0, "{name}: other salt");
                }
                assert_eq!(depth(&indexer, rank1, &plain(4)), 1, "{name}: rank 1");
            }
            other => panic!("{name}: unknown engine {other}"),
        }
    }
}
