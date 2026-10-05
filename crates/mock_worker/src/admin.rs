//! Process-wide admin API for the simulated fleet: the ground truth a routing
//! benchmark needs that real engines do not expose.
//!
//! - `GET /admin/fleet`: every registered engine with its cache size and load.
//! - `GET /admin/requests?since=<seq>&limit=<n>`: admitted-request records
//!   (`request_id`, serving worker, prompt/cached/oracle tokens, queue wait),
//!   oldest first, `seq` strictly greater than `since`.
//! - `GET /admin/cache/{worker}`: the worker's cached block keys.
//! - `POST /admin/reset/{worker}` and `POST /admin/reset`: clear one or every
//!   cache and publish `AllBlocksCleared` (an engine restart, to the index).
//!
//! Worker names are `grpc:<port>` / `http:<port>`.

use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::{
    config::Config,
    engine::{self, Engine},
};

pub struct AdminState {
    cfg: Arc<Config>,
}

pub fn router(state: Arc<AdminState>) -> Router {
    Router::new()
        .route("/admin/health", get(health))
        .route("/admin/fleet", get(fleet))
        .route("/admin/requests", get(requests))
        .route("/admin/cache/{worker}", get(cache))
        .route("/admin/reset", post(reset_all))
        .route("/admin/reset/{worker}", post(reset_one))
        .with_state(state)
}

/// Serve the admin API on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("admin bind {host}:{port} failed: {e}");
            return;
        }
    };
    let state = Arc::new(AdminState { cfg });
    if let Err(e) = axum::serve(listener, router(state)).await {
        tracing::error!("admin server stopped: {e}");
    }
}

async fn health() -> &'static str {
    "ok"
}

fn find(name: &str) -> Option<Engine> {
    engine::fleet_engines()
        .into_iter()
        .find(|e| e.name() == name)
}

async fn fleet(State(state): State<Arc<AdminState>>) -> Json<Value> {
    let workers: Vec<Value> = engine::fleet_engines()
        .iter()
        .map(|e| {
            let load = e.load();
            json!({
                "worker": e.name(),
                "cache_blocks": e.cache_keys().len(),
                "block_size": state.cfg.engine.block_size,
                "num_running_reqs": load.num_running_reqs,
                "num_waiting_reqs": load.num_waiting_reqs,
                "num_waiting_uncached_tokens": load.num_waiting_uncached_tokens,
                "token_usage": load.token_usage,
                "cache_hit_rate": load.cache_hit_rate,
                "num_cached_blocks": load.num_cached_blocks,
                "num_preemptions": load.num_preemptions,
            })
        })
        .collect();
    Json(json!({ "workers": workers }))
}

async fn requests(Query(q): Query<HashMap<String, String>>) -> Json<Value> {
    let since = q.get("since").and_then(|v| v.parse().ok()).unwrap_or(0u64);
    let limit = q
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000usize);
    let records = engine::records_since(since, limit);
    let next = records.last().map(|r| r.seq).unwrap_or(since);
    let rows: Vec<Value> = records
        .iter()
        .map(|r| {
            json!({
                "seq": r.seq,
                "request_id": r.request_id,
                "worker": r.worker,
                "prompt_tokens": r.prompt_tokens,
                "cached_tokens": r.cached_tokens,
                "oracle_tokens": r.oracle_tokens,
                "queued_ms": r.queued_ms,
                "running_at_admit": r.running_at_admit,
                "waiting_at_admit": r.waiting_at_admit,
                "admitted_unix_ms": r.admitted_unix_ms,
            })
        })
        .collect();
    Json(json!({ "records": rows, "next": next }))
}

async fn cache(Path(worker): Path<String>) -> Response {
    match find(&worker) {
        Some(e) => {
            let mut keys = e.cache_keys();
            keys.sort_unstable();
            Json(json!({ "worker": worker, "blocks": keys })).into_response()
        }
        None => (StatusCode::NOT_FOUND, "unknown worker").into_response(),
    }
}

async fn reset_one(Path(worker): Path<String>) -> Response {
    match find(&worker) {
        Some(e) => {
            e.reset();
            Json(json!({ "reset": [worker] })).into_response()
        }
        None => (StatusCode::NOT_FOUND, "unknown worker").into_response(),
    }
}

async fn reset_all() -> Json<Value> {
    let names: Vec<String> = engine::fleet_engines()
        .iter()
        .map(|e| {
            e.reset();
            e.name().to_string()
        })
        .collect();
    Json(json!({ "reset": names }))
}
