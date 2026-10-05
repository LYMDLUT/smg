"""Exercise the actual Python program embedded in the cluster trigger manifest."""

import base64
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
import yaml

MANIFEST = Path(__file__).resolve().parents[1] / "k8s-runner-resources/ci-node-health-trigger.yaml"


@pytest.fixture
def trigger():
    cron = next(d for d in yaml.safe_load_all(MANIFEST.read_text()) if d["kind"] == "CronJob")
    code = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["args"][0]
    module = types.ModuleType("ci_node_health_trigger")
    exec(compile(code, str(MANIFEST), "exec"), module.__dict__)
    return module


def _runner(name, phase="Failed", repo="smg", pool="1-gpu-h100"):
    return {
        "metadata": {
            "name": name,
            "labels": {
                "actions.github.com/organization": "smg-project",
                "actions.github.com/repository": repo,
                "actions.github.com/scale-set-name": pool,
            },
        },
        "status": {
            "phase": phase,
            "reason": "InvalidPod",
            "message": "x" * 1000,
            "runnerJITConfig": "must-not-be-forwarded",
        },
        "spec": {"token": "must-not-be-forwarded"},
    }


def test_summary_counts_all_failures_but_bounds_examples_and_omits_credentials(trigger):
    items = [_runner(f"runner-{i}") for i in range(7)]
    items += [
        _runner("active", phase="Running"),
        _runner("foreign", repo="other"),
        _runner("other-pool", pool="unknown"),
    ]
    pools = trigger.summarize_runners(items)
    assert pools[0]["failed"] == 7
    assert [e["name"] for e in pools[0]["examples"]] == ["runner-0", "runner-1", "runner-2"]
    assert len(pools[0]["examples"][0]["message"]) == 500
    assert [p["failed"] for p in pools[1:]] == [0, 0, 0]
    assert "must-not-be-forwarded" not in json.dumps(pools)


def test_snapshot_reads_all_pages_before_reporting_clean_pools(trigger, monkeypatch):
    monkeypatch.setattr(trigger.Path, "read_text", lambda *a, **k: "dummy-token")
    monkeypatch.setattr(trigger.ssl, "create_default_context", lambda **k: object())
    requests = []

    def api(method, path, token, **kwargs):
        requests.append(path)
        assert method == "GET"
        if len(requests) == 1:
            return {"items": [], "metadata": {"continue": "next/page"}}
        return {"items": [_runner("failed-on-second-page")], "metadata": {}}

    monkeypatch.setattr(trigger, "api", api)
    snapshot = trigger.runner_snapshot()
    assert len(requests) == 2
    assert "continue=next%2Fpage" in requests[1]
    assert snapshot["pools"][0]["failed"] == 1
    assert "checked_at" in snapshot


def test_snapshot_failure_is_not_an_empty_healthy_result(trigger, monkeypatch):
    monkeypatch.setattr(trigger.Path, "read_text", lambda *a, **k: "dummy-token")

    def fail(**kwargs):
        raise OSError("token-bearing-details-must-not-leak")

    monkeypatch.setattr(trigger.ssl, "create_default_context", fail)
    snapshot = trigger.runner_snapshot()
    assert "error" in snapshot and "pools" not in snapshot
    assert "token-bearing-details" not in snapshot["error"]


@pytest.mark.skipif(shutil.which("openssl") is None, reason="OpenSSL is required by the runtime")
def test_app_jwt_signature_and_claims_are_valid(trigger, tmp_path, monkeypatch):
    key = tmp_path / "private-key.pem"
    (tmp_path / "app-id").write_text("12345")
    subprocess.run(["openssl", "genrsa", "-out", str(key), "2048"], check=True, capture_output=True)
    monkeypatch.setattr(trigger.time, "time", lambda: 1000)
    jwt = trigger.app_jwt(tmp_path)
    header, claims, signature = jwt.split(".")

    def decode(value):
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

    assert json.loads(decode(header)) == {"alg": "RS256", "typ": "JWT"}
    assert json.loads(decode(claims)) == {"iat": 940, "exp": 1540, "iss": "12345"}
    public = subprocess.run(
        ["openssl", "rsa", "-in", str(key), "-pubout"], check=True, capture_output=True
    ).stdout
    (tmp_path / "public.pem").write_bytes(public)
    (tmp_path / "signature").write_bytes(decode(signature))
    verified = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            str(tmp_path / "public.pem"),
            "-signature",
            str(tmp_path / "signature"),
        ],
        input=f"{header}.{claims}".encode(),
        capture_output=True,
    )
    assert verified.returncode == 0


def test_dispatch_failure_still_revokes_installation_token(trigger, tmp_path, monkeypatch):
    (tmp_path / "installation-id").write_text("123")
    monkeypatch.setattr(trigger, "Path", lambda *args: tmp_path)
    monkeypatch.setattr(trigger, "app_jwt", lambda *args: "dummy-jwt")
    monkeypatch.setattr(trigger, "runner_snapshot", lambda: {"error": "unavailable"})
    requests = []

    def api(method, path, token, body=None):
        requests.append((method, path, body))
        if path.endswith("/access_tokens"):
            assert body == {"repositories": ["smg"], "permissions": {"actions": "write"}}
            return {"token": "dummy-installation-token"}
        if path.endswith("/dispatches"):
            assert json.loads(body["inputs"]["runner_failures_json"]) == {"error": "unavailable"}
            raise RuntimeError("dispatch failed")
        return {}

    monkeypatch.setattr(trigger, "api", api)
    with pytest.raises(RuntimeError, match="dispatch failed"):
        trigger.main()
    assert requests[-1][:2] == ("DELETE", "/installation/token")


def test_snapshot_input_stays_bounded_with_unicode_in_every_example(trigger):
    items = [_runner(f"runner-{pool}-{i}", pool=pool) for pool in trigger.POOLS for i in range(3)]
    for item in items:
        item["status"]["message"] = "😀" * 500
    snapshot = {
        "checked_at": "2026-10-05T00:00:00+00:00",
        "pools": trigger.summarize_runners(items),
    }
    serialized = trigger.serialize_snapshot(snapshot)
    assert len(serialized.encode()) <= 48000
    assert json.loads(serialized) == snapshot


def test_oversize_snapshot_reports_blindness_without_blocking_dispatch(trigger):
    serialized = trigger.serialize_snapshot({"error": "x" * 48001})
    assert len(serialized.encode()) <= 48000
    assert json.loads(serialized) == {"error": "Runner snapshot exceeded dispatch size limit"}
