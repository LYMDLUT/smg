//! Replay a Mooncake-format trace through the gateway and score the routing.
//!
//! Each trace row is `{timestamp (ms), input_length, output_length, hash_ids}`,
//! where every `hash_id` stands for one 512-token block of prompt. The replayer
//! synthesizes a deterministic text block per `hash_id` (so rows sharing ids
//! share prompt prefixes exactly as the trace intends), sends each request as a
//! streaming chat completion at `timestamp / speedup` (open loop), and records
//! time to first token, inter-token latencies, the serving worker (the
//! gateway's `system_fingerprint`, set from the worker's `weight_version`
//! label) and the engine-reported `cached_tokens`. When the mock fleet's admin
//! API is given, every request is joined with the fleet's record of it, which
//! carries the arrival-time oracle: the most cached tokens any worker held.

// A command-line tool: the summary goes to stdout, progress to stderr.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{sync::Semaphore, task::JoinSet};

#[derive(Parser, Debug, Clone)]
#[command(about = "Replay a Mooncake trace through the gateway and score routing quality")]
struct Args {
    /// Mooncake JSONL trace.
    #[arg(long)]
    trace: PathBuf,
    /// Gateway base URL.
    #[arg(long, default_value = "http://127.0.0.1:31000")]
    gateway: String,
    /// Model id to request.
    #[arg(long, default_value = "mock-model")]
    model: String,
    /// Arrival speedup: trace time is divided by this.
    #[arg(long, default_value_t = 2.0)]
    speedup: f64,
    /// Rows to skip from the start of the trace.
    #[arg(long, default_value_t = 0)]
    skip: usize,
    /// Rows to replay after `skip` (0 = all).
    #[arg(long, default_value_t = 4000)]
    limit: usize,
    /// Words synthesized per 512-token trace block (about one token each).
    #[arg(long, default_value_t = 480)]
    words_per_block: usize,
    /// Cap on `max_tokens` per request (the trace's output_length otherwise).
    #[arg(long, default_value_t = 512)]
    max_output: u32,
    /// Safety cap on concurrently open requests.
    #[arg(long, default_value_t = 4096)]
    max_inflight: usize,
    /// Mock fleet admin API base URL (enables the oracle join).
    #[arg(long)]
    admin: Option<String>,
    /// First rows whose results are excluded from the statistics (warm-up).
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    /// TTFT SLO in ms for goodput.
    #[arg(long, default_value_t = 500.0)]
    slo_ttft_ms: f64,
    /// Per-request mean inter-token latency (TPOT) SLO in ms for goodput.
    #[arg(long, default_value_t = 50.0)]
    slo_itl_ms: f64,
    /// Output directory for `summary.json` and `requests.csv`.
    #[arg(long, default_value = "replay-out")]
    out: PathBuf,
    /// Label stored in the summary.
    #[arg(long, default_value = "run")]
    label: String,
    /// Seed mixed into the synthesized text.
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

#[derive(Deserialize, Debug, Clone)]
struct TraceRow {
    timestamp: u64,
    input_length: u32,
    output_length: u32,
    hash_ids: Vec<u64>,
}

#[derive(Serialize, Debug, Clone, Default)]
struct ReqResult {
    row: usize,
    trace_ts_ms: u64,
    trace_input_length: u32,
    sent_at_ms: f64,
    status: String,
    request_id: String,
    worker: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    oracle_tokens: Option<u32>,
    queued_ms: Option<f64>,
    ttft_ms: Option<f64>,
    latency_ms: f64,
    itl_mean_ms: Option<f64>,
    itl_p99_ms: Option<f64>,
    tokens_seen: u32,
}

const WORDS: &[&str] = &[
    "time",
    "year",
    "people",
    "way",
    "day",
    "man",
    "thing",
    "woman",
    "life",
    "child",
    "world",
    "school",
    "state",
    "family",
    "student",
    "group",
    "country",
    "problem",
    "hand",
    "part",
    "place",
    "case",
    "week",
    "company",
    "system",
    "program",
    "question",
    "work",
    "government",
    "number",
    "night",
    "point",
    "home",
    "water",
    "room",
    "mother",
    "area",
    "money",
    "story",
    "fact",
    "month",
    "lot",
    "right",
    "study",
    "book",
    "eye",
    "job",
    "word",
    "business",
    "issue",
    "side",
    "kind",
    "head",
    "house",
    "service",
    "friend",
    "father",
    "power",
    "hour",
    "game",
    "line",
    "end",
    "member",
    "law",
    "car",
    "city",
    "community",
    "name",
    "president",
    "team",
    "minute",
    "idea",
    "kid",
    "body",
    "information",
    "back",
    "parent",
    "face",
    "others",
    "level",
    "office",
    "door",
    "health",
    "person",
    "art",
    "war",
    "history",
    "party",
    "result",
    "change",
    "morning",
    "reason",
    "research",
    "girl",
    "guy",
    "moment",
    "air",
    "teacher",
    "force",
    "education",
    "foot",
    "boy",
    "age",
    "policy",
    "process",
    "music",
    "market",
    "sense",
    "nation",
    "plan",
    "college",
    "interest",
    "death",
    "experience",
    "effect",
    "use",
    "class",
    "control",
    "care",
    "field",
    "development",
    "role",
    "effort",
    "rate",
    "heart",
    "drug",
    "show",
    "leader",
    "light",
    "voice",
    "wife",
    "police",
    "mind",
    "price",
    "report",
    "decision",
    "son",
    "view",
    "relationship",
    "town",
    "road",
    "arm",
    "difference",
    "value",
    "building",
    "action",
    "model",
    "season",
    "society",
    "tax",
    "director",
    "position",
    "player",
    "record",
    "paper",
    "space",
    "ground",
    "form",
    "event",
    "official",
    "matter",
    "center",
    "couple",
    "site",
    "project",
    "activity",
    "star",
    "table",
    "need",
    "court",
    "oil",
    "situation",
    "cost",
    "industry",
    "figure",
    "street",
    "image",
    "phone",
    "data",
    "picture",
    "practice",
    "piece",
    "land",
    "product",
    "doctor",
    "wall",
    "patient",
    "worker",
    "news",
    "test",
    "movie",
    "north",
    "love",
    "support",
    "technology",
    "step",
    "baby",
    "computer",
    "type",
    "attention",
    "film",
    "tree",
    "source",
    "nothing",
    "network",
    "trade",
    "economy",
    "author",
    "window",
    "energy",
    "letter",
    "church",
    "cell",
    "ship",
    "island",
    "plant",
    "garden",
    "river",
    "bridge",
    "engine",
    "metal",
    "glass",
    "stone",
    "storm",
    "forest",
    "valley",
    "ocean",
    "desert",
    "signal",
    "memory",
    "logic",
];

/// Deterministic text for one trace block: `words_per_block` words drawn from
/// the vocabulary by a splitmix64 stream seeded with the block id.
fn block_text(hash_id: u64, seed: u64, words_per_block: usize) -> String {
    let mut x = hash_id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed ^ 0xD1B5_4A32_D192_ED03);
    let mut out = String::with_capacity(words_per_block * 7);
    for i in 0..words_per_block {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if i > 0 {
            out.push(' ');
        }
        out.push_str(WORDS[(z % WORDS.len() as u64) as usize]);
    }
    out
}

fn prompt_for(row: &TraceRow, seed: u64, words_per_block: usize) -> String {
    let mut prompt = String::new();
    for (i, id) in row.hash_ids.iter().enumerate() {
        if i > 0 {
            prompt.push('\n');
        }
        prompt.push_str(&block_text(*id, seed, words_per_block));
    }
    prompt
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

struct Job {
    client: reqwest::Client,
    url: String,
    model: String,
    row_index: usize,
    row: TraceRow,
    body_prompt: String,
    max_output: u32,
    sent_at_ms: f64,
}

async fn run_one(job: Job) -> ReqResult {
    let Job {
        client,
        url,
        model,
        row_index,
        row,
        body_prompt,
        max_output,
        sent_at_ms,
    } = job;
    let max_tokens = row.output_length.clamp(1, max_output);
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": body_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let mut result = ReqResult {
        row: row_index,
        trace_ts_ms: row.timestamp,
        trace_input_length: row.input_length,
        sent_at_ms,
        status: "ok".to_string(),
        ..Default::default()
    };
    let started = Instant::now();
    let response = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            result.status = format!("send-error: {e}");
            result.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
            return result;
        }
    };
    if !response.status().is_success() {
        result.status = format!("http-{}", response.status().as_u16());
        result.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        return result;
    }
    let mut stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut first_token: Option<Instant> = None;
    let mut last_token: Option<Instant> = None;
    let mut itls: Vec<f64> = Vec::new();
    let mut tokens_seen = 0u32;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                result.status = format!("stream-error: {e}");
                break;
            }
        };
        buf.extend_from_slice(&chunk);
        // SSE events end with a blank line.
        while let Some(pos) = find_double_newline(&buf) {
            let event: Vec<u8> = buf.drain(..pos + 2).collect();
            let text = String::from_utf8_lossy(&event);
            for line in text.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if result.request_id.is_empty() {
                    if let Some(id) = v.get("id").and_then(Value::as_str) {
                        result.request_id = id.to_string();
                    }
                }
                if result.worker.is_empty() {
                    if let Some(fp) = v.get("system_fingerprint").and_then(Value::as_str) {
                        result.worker = fp.to_string();
                    }
                }
                let has_content = v
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|c| c.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty());
                if has_content {
                    let now = Instant::now();
                    tokens_seen += 1;
                    match last_token {
                        None => first_token = Some(now),
                        Some(prev) => itls.push((now - prev).as_secs_f64() * 1000.0),
                    }
                    last_token = Some(now);
                }
                if let Some(usage) = v.get("usage").filter(|u| !u.is_null()) {
                    result.prompt_tokens = usage
                        .get("prompt_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32;
                    result.completion_tokens = usage
                        .get("completion_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32;
                    result.cached_tokens = usage
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .or_else(|| usage.get("cached_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32;
                }
            }
        }
    }
    result.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    result.ttft_ms = first_token.map(|t| (t - started).as_secs_f64() * 1000.0);
    result.tokens_seen = tokens_seen;
    if !itls.is_empty() {
        let mut sorted = itls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result.itl_mean_ms = Some(mean(&itls));
        result.itl_p99_ms = Some(percentile(&sorted, 0.99));
    }
    if result.ttft_ms.is_none() && result.status == "ok" {
        result.status = "no-tokens".to_string();
    }
    result
}

fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

#[derive(Deserialize, Debug)]
struct AdminRecords {
    records: Vec<AdminRecord>,
    next: u64,
}

#[derive(Deserialize, Debug, Clone)]
struct AdminRecord {
    request_id: String,
    worker: String,
    cached_tokens: u32,
    oracle_tokens: u32,
    queued_ms: f64,
}

async fn fetch_admin_records(client: &reqwest::Client, admin: &str) -> Result<Vec<AdminRecord>> {
    let mut since = 0u64;
    let mut all = Vec::new();
    loop {
        let page: AdminRecords = client
            .get(format!("{admin}/admin/requests?since={since}&limit=50000"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        if page.records.is_empty() {
            break;
        }
        since = page.next;
        all.extend(page.records);
    }
    Ok(all)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let text = fs::read_to_string(&args.trace)
        .with_context(|| format!("reading {}", args.trace.display()))?;
    let rows: Vec<TraceRow> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<TraceRow>(l).map_err(|e| anyhow!("{e}: {l}")))
        .collect::<Result<_>>()?;
    let end = if args.limit == 0 {
        rows.len()
    } else {
        (args.skip + args.limit).min(rows.len())
    };
    let rows: Vec<TraceRow> = rows[args.skip.min(rows.len())..end].to_vec();
    if rows.is_empty() {
        return Err(anyhow!("no rows selected"));
    }
    let t0 = rows[0].timestamp;
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(args.max_inflight)
        .timeout(Duration::from_secs(600))
        .build()?;
    let url = format!("{}/v1/chat/completions", args.gateway.trim_end_matches('/'));
    let inflight = Arc::new(Semaphore::new(args.max_inflight));
    let mut set: JoinSet<ReqResult> = JoinSet::new();
    let start = Instant::now();
    eprintln!(
        "replaying {} rows at {}x ({:.0} s of trace), {} words/block, max_output {}",
        rows.len(),
        args.speedup,
        (rows.last().map(|r| r.timestamp).unwrap_or(t0) - t0) as f64 / 1000.0,
        args.words_per_block,
        args.max_output
    );
    for (i, row) in rows.iter().enumerate() {
        let due = Duration::from_secs_f64((row.timestamp - t0) as f64 / 1000.0 / args.speedup);
        let elapsed = start.elapsed();
        if due > elapsed {
            tokio::time::sleep(due - elapsed).await;
        }
        let permit = inflight.clone().acquire_owned().await?;
        let prompt = prompt_for(row, args.seed, args.words_per_block);
        let job = Job {
            client: client.clone(),
            url: url.clone(),
            model: args.model.clone(),
            row_index: i,
            row: row.clone(),
            body_prompt: prompt,
            max_output: args.max_output,
            sent_at_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        set.spawn(async move {
            let _permit = permit;
            run_one(job).await
        });
        if (i + 1) % 500 == 0 {
            eprintln!(
                "  sent {} / {} ({:.0} s)",
                i + 1,
                rows.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    let mut results: Vec<ReqResult> = Vec::with_capacity(rows.len());
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("task failed: {e}"),
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    results.sort_by_key(|r| r.row);

    // Oracle join (optional).
    let mut admin_rows: HashMap<String, AdminRecord> = HashMap::new();
    if let Some(admin) = &args.admin {
        match fetch_admin_records(&client, admin.trim_end_matches('/')).await {
            Ok(recs) => {
                for r in recs {
                    admin_rows.insert(r.request_id.clone(), r);
                }
            }
            Err(e) => eprintln!("admin records unavailable: {e}"),
        }
        for r in &mut results {
            if let Some(a) = admin_rows.get(&r.request_id) {
                r.oracle_tokens = Some(a.oracle_tokens.max(a.cached_tokens));
                r.queued_ms = Some(a.queued_ms);
                if r.worker.is_empty() {
                    r.worker = a.worker.clone();
                }
                if r.cached_tokens == 0 && a.cached_tokens > 0 {
                    r.cached_tokens = a.cached_tokens;
                }
            }
        }
    }

    // Statistics over the non-warm-up rows.
    let scored: Vec<&ReqResult> = results.iter().filter(|r| r.row >= args.warmup).collect();
    let ok: Vec<&ReqResult> = scored
        .iter()
        .copied()
        .filter(|r| r.status == "ok")
        .collect();
    let mut ttft: Vec<f64> = ok.iter().filter_map(|r| r.ttft_ms).collect();
    ttft.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut itl_means: Vec<f64> = ok.iter().filter_map(|r| r.itl_mean_ms).collect();
    itl_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut itl_p99s: Vec<f64> = ok.iter().filter_map(|r| r.itl_p99_ms).collect();
    itl_p99s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut e2e: Vec<f64> = ok.iter().map(|r| r.latency_ms).collect();
    e2e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let prompt_total: u64 = ok.iter().map(|r| u64::from(r.prompt_tokens)).sum();
    let cached_total: u64 = ok.iter().map(|r| u64::from(r.cached_tokens)).sum();
    let oracle_total: u64 = ok
        .iter()
        .filter_map(|r| r.oracle_tokens)
        .map(u64::from)
        .sum();
    let oracle_known = ok.iter().filter(|r| r.oracle_tokens.is_some()).count();
    let within_slo = ok
        .iter()
        .filter(|r| {
            r.ttft_ms.is_some_and(|t| t <= args.slo_ttft_ms)
                && r.itl_mean_ms.is_none_or(|m| m <= args.slo_itl_ms)
        })
        .count();
    let within_slo_strict = ok
        .iter()
        .filter(|r| {
            r.ttft_ms.is_some_and(|t| t <= args.slo_ttft_ms)
                && r.itl_p99_ms.is_none_or(|p| p <= args.slo_itl_ms)
        })
        .count();
    let mut per_worker: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for r in &ok {
        let e = per_worker.entry(r.worker.clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 += u64::from(r.prompt_tokens.saturating_sub(r.cached_tokens));
    }
    let counts: Vec<f64> = per_worker.values().map(|v| v.0 as f64).collect();
    let balance_max_over_mean = if counts.is_empty() {
        f64::NAN
    } else {
        counts.iter().copied().fold(0.0, f64::max) / mean(&counts)
    };
    let summary = json!({
        "label": args.label,
        "rows": rows.len(),
        "scored": scored.len(),
        "ok": ok.len(),
        "errors": scored.len() - ok.len(),
        "speedup": args.speedup,
        "wall_s": wall_s,
        "req_per_s": ok.len() as f64 / wall_s,
        "ttft_ms": {"mean": mean(&ttft), "p50": percentile(&ttft, 0.5), "p90": percentile(&ttft, 0.9), "p99": percentile(&ttft, 0.99)},
        "itl_mean_ms": {"mean": mean(&itl_means), "p90": percentile(&itl_means, 0.9), "p99": percentile(&itl_means, 0.99)},
        "itl_p99_ms_per_request": {"p50": percentile(&itl_p99s, 0.5), "p90": percentile(&itl_p99s, 0.9)},
        "e2e_ms": {"p50": percentile(&e2e, 0.5), "p99": percentile(&e2e, 0.99)},
        "goodput_req_per_s": within_slo as f64 / wall_s,
        "within_slo_fraction": if ok.is_empty() { f64::NAN } else { within_slo as f64 / ok.len() as f64 },
        "within_slo_strict_fraction": if ok.is_empty() { f64::NAN } else { within_slo_strict as f64 / ok.len() as f64 },
        "prefix_reuse": if prompt_total == 0 { f64::NAN } else { cached_total as f64 / prompt_total as f64 },
        "oracle_prefix_reuse": if prompt_total == 0 || oracle_known == 0 { f64::NAN } else { oracle_total as f64 / prompt_total as f64 },
        "hit_over_oracle": if oracle_total == 0 { f64::NAN } else { cached_total as f64 / oracle_total as f64 },
        "oracle_known": oracle_known,
        "per_worker_requests": per_worker.iter().map(|(w, v)| json!({"worker": w, "requests": v.0, "uncached_prompt_tokens": v.1})).collect::<Vec<_>>(),
        "balance_max_over_mean": balance_max_over_mean,
        "slo": {"ttft_ms": args.slo_ttft_ms, "itl_ms": args.slo_itl_ms, "itl_metric": "per-request mean (strict variant: per-request p99)"},
    });
    fs::create_dir_all(&args.out)?;
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    let mut csv = String::from("row,trace_ts_ms,trace_input_length,sent_at_ms,status,request_id,worker,prompt_tokens,completion_tokens,cached_tokens,oracle_tokens,queued_ms,ttft_ms,latency_ms,itl_mean_ms,itl_p99_ms,tokens_seen\n");
    for r in &results {
        csv.push_str(&format!(
            "{},{},{},{:.1},{},{},{},{},{},{},{},{},{},{:.1},{},{},{}\n",
            r.row,
            r.trace_ts_ms,
            r.trace_input_length,
            r.sent_at_ms,
            r.status.replace(',', ";"),
            r.request_id,
            r.worker,
            r.prompt_tokens,
            r.completion_tokens,
            r.cached_tokens,
            r.oracle_tokens.map(|v| v.to_string()).unwrap_or_default(),
            r.queued_ms.map(|v| format!("{v:.1}")).unwrap_or_default(),
            r.ttft_ms.map(|v| format!("{v:.1}")).unwrap_or_default(),
            r.latency_ms,
            r.itl_mean_ms.map(|v| format!("{v:.2}")).unwrap_or_default(),
            r.itl_p99_ms.map(|v| format!("{v:.2}")).unwrap_or_default(),
            r.tokens_seen
        ));
    }
    fs::write(args.out.join("requests.csv"), csv)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_text_is_deterministic_per_hash_id() {
        assert_eq!(block_text(42, 7, 32), block_text(42, 7, 32));
        assert_ne!(block_text(42, 7, 32), block_text(43, 7, 32));
        assert_ne!(block_text(42, 7, 32), block_text(42, 8, 32));
        assert_eq!(block_text(1, 7, 32).split(' ').count(), 32);
    }

    #[test]
    fn prompts_share_prefixes_when_rows_share_hash_ids() {
        let a = TraceRow {
            timestamp: 0,
            input_length: 1024,
            output_length: 1,
            hash_ids: vec![0, 1],
        };
        let b = TraceRow {
            timestamp: 0,
            input_length: 1024,
            output_length: 1,
            hash_ids: vec![0, 2],
        };
        let pa = prompt_for(&a, 7, 16);
        let pb = prompt_for(&b, 7, 16);
        let shared = block_text(0, 7, 16);
        assert!(pa.starts_with(&shared) && pb.starts_with(&shared));
        assert_ne!(pa, pb);
    }

    #[test]
    fn percentile_and_mean() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 1.0), 5.0);
        assert_eq!(mean(&v), 3.0);
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn sse_event_boundary() {
        assert_eq!(find_double_newline(b"data: x\n\ndata: y"), Some(7));
        assert_eq!(find_double_newline(b"data: x\n"), None);
    }
}
