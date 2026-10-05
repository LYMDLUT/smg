//! Render parity with the bellwether reference fixtures.
//!
//! bellwether (smg-project/bellwether) records, for each model at a pinned
//! Hugging Face revision, what the checkpoint's own chat template renders for
//! a corpus of chat requests: the prompt text and its token ids. This test
//! replays every render fixture through the tokenizer the way the gateway
//! does (`model_gateway/src/routers/grpc/utils/chat_utils.rs`: template
//! kwargs, the pop-and-prefix path for `continue_final_message`) and compares
//! text and ids byte for byte.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the `fixtures/` directory
//! of a bellwether checkout; without it the test prints a skip notice and
//! passes. Tokenizer files come from the Hugging Face cache snapshot at the
//! manifest's revision when it is there, else from a one-time download into
//! `.tokenizer_cache/bellwether/<slug>/<revision>/`.
//!
//! A difference is a finding, not something to hide: every known one is
//! listed in [`KNOWN_DIFFERENCES`] with its reason and where it is tracked,
//! the run fails on any other, and it fails again when a listed case starts
//! matching, so the list cannot rot.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    create_tokenizer,
    traits::{PromptEncoding, Tokenizer as TokenizerTrait},
};
use serde::Deserialize;
use serde_json::Value;

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const CACHE_DIR: &str = ".tokenizer_cache/bellwether";

/// Cases known to differ from the reference, by fixture id, each with the
/// reason and where it is tracked.
const KNOWN_DIFFERENCES: &[(&str, &str)] = &[
    (
        "qwen3-8b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and appending its \
         text after the generation header, which drops the empty think block the template writes \
         before a continued turn (smg-project/smg#2779)",
    ),
    (
        "deepseek-r1/render/continue-final-message",
        "the gateway's generation header adds `<think>\\n` before the continued text; the template \
         writes none for a continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-8b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always appended \
         (smg-project/smg#2780)",
    ),
    (
        "deepseek-r1/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always appended \
         (smg-project/smg#2780)",
    ),
];

/// The keys of a render request this test hands to the renderer the way the
/// gateway does. Any other key fails the case, so a corpus knob the
/// projection does not know is reported instead of rendering without it.
const PROJECTED_KEYS: [&str; 5] = [
    "messages",
    "tools",
    "chat_template_kwargs",
    "continue_final_message",
    "add_generation_prompt",
];

/// `fixtures/<slug>/manifest.toml`: the model and the revision its fixtures
/// were recorded at.
#[derive(Deserialize)]
struct Manifest {
    model: String,
    revision: String,
}

/// One line of `fixtures/<slug>/render/<set>.jsonl`, bellwether's
/// `case.schema.json`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    id: String,
    kind: String,
    model: String,
    request: Value,
    reference: Reference,
    #[serde(default)]
    witnesses: Option<Value>,
}

#[derive(Deserialize)]
struct Reference {
    source: String,
    input_ids: Vec<u32>,
    text: String,
    #[serde(default)]
    provenance: Value,
}

struct Rendered {
    text: String,
    ids: Vec<u32>,
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notice and the per-case report are test diagnostic output"
)]
fn render_fixtures_match_the_reference_byte_for_byte() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let manifests = read_manifests(&root).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !manifests.is_empty(),
        "no fixtures/<slug>/manifest.toml under {}",
        root.display()
    );

    let known: BTreeMap<&str, &str> = KNOWN_DIFFERENCES.iter().copied().collect();
    let mut seen = BTreeSet::new();
    let mut differences = BTreeMap::new();
    let mut matched = 0usize;
    let mut with_witnesses = 0usize;
    for (slug, manifest) in &manifests {
        let render_dir = root.join(slug).join("render");
        if !render_dir.is_dir() {
            continue;
        }
        let dir = tokenizer_dir(&manifest.model, &manifest.revision, slug)
            .unwrap_or_else(|e| panic!("{slug}: {e}"));
        let dir_str = dir
            .to_str()
            .unwrap_or_else(|| panic!("{slug}: tokenizer path {} is not UTF-8", dir.display()));
        let tok = create_tokenizer(dir_str)
            .unwrap_or_else(|e| panic!("{slug}: the tokenizer at {dir_str} should load: {e}"));
        let fixtures = read_fixtures(&render_dir).unwrap_or_else(|e| panic!("{slug}: {e}"));
        let recorded_by = fixtures
            .first()
            .map(|f| f.reference.provenance.to_string())
            .unwrap_or_default();
        println!(
            "{slug}: {} at {} from {dir_str}; {} cases recorded with {recorded_by}",
            manifest.model,
            manifest.revision,
            fixtures.len()
        );
        for fixture in fixtures {
            assert_eq!(fixture.kind, "render", "{}: not a render case", fixture.id);
            assert_eq!(
                fixture.model, manifest.model,
                "{}: model differs from the manifest",
                fixture.id
            );
            if fixture.witnesses.is_some() {
                with_witnesses += 1;
            }
            seen.insert(fixture.id.clone());
            let outcome = match render(tok.as_ref(), &fixture.request) {
                Err(e) => Err(e),
                Ok(got)
                    if got.text == fixture.reference.text
                        && got.ids == fixture.reference.input_ids =>
                {
                    Ok(())
                }
                Ok(got) => Err(describe(&got, &fixture.reference)),
            };
            match outcome {
                Ok(()) => {
                    matched += 1;
                    println!("  match   {}", fixture.id);
                }
                Err(why) => {
                    println!(
                        "  differs {} (reference {}): {why}",
                        fixture.id, fixture.reference.source
                    );
                    differences.insert(fixture.id, why);
                }
            }
        }
    }
    println!(
        "{matched} cases match, {} differ, {with_witnesses} carry engine witnesses",
        differences.len()
    );

    let unexpected: Vec<String> = differences
        .iter()
        .filter(|(id, _)| !known.contains_key(id.as_str()))
        .map(|(id, why)| format!("{id}: {why}"))
        .collect();
    assert!(
        unexpected.is_empty(),
        "differences not listed in KNOWN_DIFFERENCES:\n{}",
        unexpected.join("\n")
    );
    let healed: Vec<&str> = known
        .keys()
        .copied()
        .filter(|id| seen.contains(*id) && !differences.contains_key(*id))
        .collect();
    assert!(
        healed.is_empty(),
        "listed in KNOWN_DIFFERENCES but matching the reference now; remove: {}",
        healed.join(", ")
    );
}

/// Hand a corpus request to the renderer the way the gateway does: the
/// messages and tools as sent, `chat_template_kwargs` as template variables,
/// and `continue_final_message` on a trailing assistant message rendered
/// natively when the renderer can, else by popping the message and appending
/// its text after the generation header. SMG's chat request has no
/// `add_generation_prompt` field: the gateway appends the generation header
/// on every request that is not a native continuation, so a corpus case that
/// sets the field renders as if it had not.
fn render(tok: &dyn TokenizerTrait, request: &Value) -> Result<Rendered, String> {
    let object = request.as_object().ok_or("the request is not an object")?;
    if let Some(key) = object
        .keys()
        .find(|key| !PROJECTED_KEYS.contains(&key.as_str()))
    {
        return Err(format!("request key {key:?} is not projected by this test"));
    }
    let mut messages = object
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .ok_or("the request has no messages array")?;
    let tools = object.get("tools").and_then(Value::as_array).cloned();
    let kwargs: HashMap<String, Value> = object
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .map(|map| map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let continues_final_assistant = object
        .get("continue_final_message")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && messages
            .last()
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            == Some("assistant");
    let native_continuation =
        continues_final_assistant && tok.renderer_capabilities().native_assistant_continuation;
    let assistant_prefix = if continues_final_assistant && !native_continuation {
        messages
            .pop()
            .and_then(|message| message.get("content").and_then(prefill_text))
    } else {
        None
    };

    let params = ChatTemplateParams {
        add_generation_prompt: !native_continuation,
        tools: tools.as_deref(),
        template_kwargs: (!kwargs.is_empty()).then_some(&kwargs),
        ..Default::default()
    };
    let out = tok
        .apply_chat_template_with_encoding(&messages, params, assistant_prefix.as_deref())
        .map_err(|e| format!("render failed: {e}"))?;
    let ids = match out.encoding {
        PromptEncoding::FromText => tok
            .encode(&out.text, false)
            .map_err(|e| format!("encode failed: {e}"))?
            .token_ids()
            .to_vec(),
        PromptEncoding::Deferred(_) => {
            return Err(
                "the renderer deferred its encode; this test replays flat renderers only"
                    .to_string(),
            );
        }
    };
    Ok(Rendered {
        text: out.text,
        ids,
    })
}

/// The gateway's `prefill_text`: a string content, or the `text` parts of an
/// array.
fn prefill_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|part| part.get("text")?.as_str())
                .collect(),
        ),
        _ => None,
    }
}

/// Where the rendering and the reference part: the first differing token and
/// the text around the first differing byte.
fn describe(got: &Rendered, want: &Reference) -> String {
    let id_at = got
        .ids
        .iter()
        .zip(&want.input_ids)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.ids.len().min(want.input_ids.len()));
    let byte_at = got
        .text
        .bytes()
        .zip(want.text.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.text.len().min(want.text.len()));
    format!(
        "ids part at index {id_at} ({} rendered, {} in the reference); text parts at byte {byte_at}: rendered {:?}, reference {:?}",
        got.ids.len(),
        want.input_ids.len(),
        window(&got.text, byte_at),
        window(&want.text, byte_at)
    )
}

/// Up to 40 bytes before `at` and 60 after, cut on character boundaries.
fn window(text: &str, at: usize) -> &str {
    let mut start = at.saturating_sub(40);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (at + 60).min(text.len());
    while !text.is_char_boundary(end) {
        end += 1;
    }
    &text[start..end]
}

fn read_manifests(root: &Path) -> Result<Vec<(String, Manifest)>, String> {
    let entries = fs::read_dir(root).map_err(|e| format!("cannot read {}: {e}", root.display()))?;
    let mut manifests = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read {}: {e}", root.display()))?;
        let path = entry.path().join("manifest.toml");
        if !path.is_file() {
            continue;
        }
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let manifest: Manifest =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        manifests.push((entry.file_name().to_string_lossy().into_owned(), manifest));
    }
    manifests.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(manifests)
}

fn read_fixtures(dir: &Path) -> Result<Vec<Fixture>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "jsonl") {
            files.push(path);
        }
    }
    files.sort();
    let mut fixtures = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fixture: Fixture = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", file.display(), number + 1))?;
            fixtures.push(fixture);
        }
    }
    Ok(fixtures)
}

/// The checkpoint's tokenizer files at the manifest's revision: the Hugging
/// Face cache snapshot when it is there, else a one-time download.
fn tokenizer_dir(model: &str, revision: &str, slug: &str) -> Result<PathBuf, String> {
    if let Some(snapshot) = hf_cache_snapshot(model, revision) {
        return Ok(snapshot);
    }
    let dir = PathBuf::from(CACHE_DIR).join(slug).join(revision);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let client = reqwest::blocking::Client::new();
    // `config.json` only serves renderer detection, so a checkpoint without one is fine.
    for (file, min_bytes, required) in [
        ("tokenizer.json", 100_000usize, true),
        ("tokenizer_config.json", 100, true),
        ("config.json", 50, false),
    ] {
        let path = dir.join(file);
        if path.is_file() {
            continue;
        }
        let url = format!("https://huggingface.co/{model}/resolve/{revision}/{file}");
        let response = client
            .get(&url)
            .send()
            .map_err(|e| format!("GET {url}: {e}"))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND && !required {
            continue;
        }
        if !response.status().is_success() {
            return Err(format!("GET {url}: HTTP {}", response.status()));
        }
        let bytes = response.bytes().map_err(|e| format!("GET {url}: {e}"))?;
        if bytes.len() < min_bytes {
            return Err(format!(
                "{url}: {} bytes, expected at least {min_bytes}",
                bytes.len()
            ));
        }
        fs::write(&path, &bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(dir)
}

/// `<hub cache>/models--<org>--<name>/snapshots/<revision>`, the layout
/// `huggingface_hub` keeps, under `HF_HUB_CACHE`, `HF_HOME/hub` or the
/// default `~/.cache/huggingface/hub`.
fn hf_cache_snapshot(model: &str, revision: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
        })?;
    let dir = hub
        .join(format!("models--{}", model.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    (dir.join("tokenizer.json").is_file() && dir.join("tokenizer_config.json").is_file())
        .then_some(dir)
}
