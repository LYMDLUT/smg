#!/usr/bin/env python3
"""Guardrail 5 runner: N fresh-process trials with an interleaved same-binary control pair.

For one named binary and configuration this runs `--trials` subject trials, each followed by a
control trial of `--control-command` (by default the very same command, so the pair measures the
noise floor an A/A comparison would show). Every trial holds the lock file, samples the
measurement cores for foreign load before and after, and is discarded (kept in the output, marked)
when a process above the threshold that is neither the trial nor allow-listed shows up.

Reported: medians with bootstrap 95% confidence intervals (percentile method) of achieved block
ops/s and lookup p50/p99, per series, plus the subject-minus-control difference of medians with
its own bootstrap interval. Finished trials are skipped on re-run, so an interrupted run resumes.

Command placeholders: `{json}` (result path) and `{trial}`.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import random
import re
import shlex
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hostload  # noqa: E402

METRICS = (
    ("achieved_m", "Achieved (M block ops/s)", 1.0),
    ("p50_us", "Lookup p50 (us)", 1.0),
    ("p99_us", "Lookup p99 (us)", 1.0),
)


def run_trial(args: argparse.Namespace, role: str, index: int, command_template: str) -> dict:
    out = pathlib.Path(args.out)
    record_path = out / "trials" / f"{role}-{index}.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    result = out / f"{role}-{index}.json"
    command = command_template.format(json=result, trial=index)
    cores = hostload.parse_cpu_list(args.cores)
    allow = re.compile(args.allow) if args.allow else None
    record = {"role": role, "index": index, "command": command, "started_at": time.time()}
    with open(args.lock, "w") as lock:
        if args.lock_scope == "trial":
            fcntl.flock(lock, fcntl.LOCK_EX)
        before = hostload.sample(cores, args.sample_seconds, {os.getpid()})
        started = time.time()
        with open(out / f"{role}-{index}.log", "w") as log:
            status = subprocess.run(shlex.split(command), stdout=log, stderr=subprocess.STDOUT)
        record["wall_s"] = time.time() - started
        after = hostload.sample(cores, args.sample_seconds, {os.getpid()})
        if args.lock_scope == "trial":
            fcntl.flock(lock, fcntl.LOCK_UN)
    foreign, background = [], []
    for rows in (before, after):
        f, b = hostload.classify(rows, args.threshold_pct, allow, args.discard_kernel_threads)
        foreign += f
        background += b
    record["foreign_load"] = foreign
    record["background_load"] = background
    record["exit_code"] = status.returncode
    if status.returncode != 0 or not result.exists():
        record["discarded"] = f"trial failed (exit {status.returncode})"
    else:
        data = json.loads(result.read_text())
        offered = data["offered_block_ops_per_sec"]
        achieved = data["achieved_block_ops_per_sec"]
        record.update(
            offered_m=offered / 1e6,
            achieved_m=achieved / 1e6,
            ratio=achieved / offered if offered else 0.0,
            generator_valid=bool(data["generator_valid"]),
            kept_up=bool(data["generator_valid"]) and achieved >= 0.99 * offered,
            p50_us=data["query_service"]["p50_ns"] / 1e3,
            p99_us=data["query_service"]["p99_ns"] / 1e3,
            p999_us=data["query_service"]["p999_ns"] / 1e3,
            e2e_p99_us=data["query_scheduled_to_finished"]["p99_ns"] / 1e3,
            drain_ms=data["drain_ns"] / 1e6,
            failure_reasons=data.get("failure_reasons", []),
        )
        if not data["generator_valid"]:
            record["discarded"] = (
                "generator invalid: " + ", ".join(data.get("failure_reasons", []))[:200]
            )
        elif foreign:
            record["discarded"] = "foreign load: " + ", ".join(
                f"{row['comm']}[{row['pid']}] {row['cpu_pct']:.0f}%" for row in foreign[:4]
            )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1))
    return record


def bootstrap_median(
    values: list[float], rng: random.Random, rounds: int
) -> tuple[float, float, float]:
    """(median, low, high) with a percentile-bootstrap 95% interval."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(rounds))
    return (
        statistics.median(values),
        medians[int(0.025 * rounds)],
        medians[min(rounds - 1, int(0.975 * rounds))],
    )


def bootstrap_difference(
    a: list[float], b: list[float], rng: random.Random, rounds: int
) -> tuple[float, float, float]:
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    diffs = sorted(
        statistics.median(rng.choices(a, k=len(a))) - statistics.median(rng.choices(b, k=len(b)))
        for _ in range(rounds)
    )
    return (
        statistics.median(a) - statistics.median(b),
        diffs[int(0.025 * rounds)],
        diffs[min(rounds - 1, int(0.975 * rounds))],
    )


def background_summary(records: list[dict]) -> list[str]:
    """Allow-listed and kernel background per process name: trials seen in, peak CPU share."""
    peak: dict[str, tuple[int, float]] = {}
    for record in records:
        seen: dict[str, float] = {}
        for row in record["background_load"]:
            name = "kernel threads" if row["kernel_thread"] else row["comm"]
            seen[name] = max(seen.get(name, 0.0), row["cpu_pct"])
        for name, pct in seen.items():
            count, top = peak.get(name, (0, 0.0))
            peak[name] = (count + 1, max(top, pct))
    return [
        f"{name}: in {count} of {len(records)} trials, peak {top:.0f}%"
        for name, (count, top) in sorted(peak.items(), key=lambda item: -item[1][1])
    ]


def summarize(args: argparse.Namespace, series: dict[str, list[dict]]) -> tuple[dict, str]:
    rng = random.Random(args.seed)
    summary: dict = {"trials_requested": args.trials, "series": {}}
    lines = [
        f"{args.name}: {args.trials} trials per series, fresh process each, lock held per trial, "
        f"foreign-load threshold {args.threshold_pct:.0f}% on cores {args.cores}.",
        "",
        "| Series | Used / discarded | Kept up | Achieved median [95% CI] (M block ops/s) "
        "| Lookup p50 [CI] (us) | Lookup p99 [CI] (us) | Drain median (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    kept: dict[str, dict[str, list[float]]] = {}
    for role, records in series.items():
        used = [r for r in records if "discarded" not in r]
        discarded = [r for r in records if "discarded" in r]
        kept[role] = {key: [r[key] for r in used] for key, _, _ in METRICS}
        stats = {}
        cells = []
        for key, _, _ in METRICS:
            med, lo, hi = bootstrap_median(kept[role][key], rng, args.bootstrap)
            stats[key] = {"median": med, "ci95": [lo, hi], "n": len(used)}
            cells.append(f"{med:.1f} [{lo:.1f}, {hi:.1f}]")
        drain = statistics.median([r["drain_ms"] for r in used]) if used else float("nan")
        summary["series"][role] = {
            "used": len(used),
            "discarded": [{"index": r["index"], "why": r["discarded"]} for r in discarded],
            "kept_up": sum(1 for r in used if r.get("kept_up")),
            "stats": stats,
            "drain_ms_median": drain,
            "background_load": background_summary(records),
        }
        lines.append(
            f"| {role} | {len(used)} / {len(discarded)} | {summary['series'][role]['kept_up']} of {len(used)} | "
            + " | ".join(cells)
            + f" | {drain:.0f} |"
        )
    roles = list(series)
    if len(roles) == 2:
        diffs = {}
        parts = []
        for key, label, _ in METRICS:
            d, lo, hi = bootstrap_difference(
                kept[roles[0]][key], kept[roles[1]][key], rng, args.bootstrap
            )
            diffs[key] = {"difference": d, "ci95": [lo, hi]}
            parts.append(f"{label}: {d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        summary["subject_minus_control"] = diffs
        lines += [
            "",
            f"Subject minus control (difference of medians, bootstrap 95% CI): {'; '.join(parts)}.",
        ]
    discarded_lines = [
        f"- {role} trial {d['index']}: {d['why']}"
        for role in roles
        for d in summary["series"][role]["discarded"]
    ]
    lines += ["", "Discarded trials:" + (" none" if not discarded_lines else "")] + discarded_lines
    background = background_summary([r for role in roles for r in series[role]])
    lines += ["", "Background load recorded (allow-listed daemons and kernel threads):"]
    lines += [f"- {entry}" for entry in background] or ["- none"]
    return summary, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="label of the subject")
    parser.add_argument("--command", required=True, help="subject trial command template")
    parser.add_argument(
        "--control-command",
        default="",
        help="control template; default: the subject's; 'none' disables",
    )
    parser.add_argument("--control-name", default="control (same binary)")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--lock-scope",
        choices=("trial", "run"),
        default="trial",
        help="hold the lock per trial (fair to other takers) or for the whole run (one queue wait)",
    )
    parser.add_argument("--lock", required=True)
    parser.add_argument("--cores", default="0-63")
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument(
        "--allow", default="", help="regex of background processes to record, not flag"
    )
    parser.add_argument("--discard-kernel-threads", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_lock = open(args.lock, "w")  # noqa: SIM115 - held until exit when the scope is "run"
    if args.lock_scope == "run":
        fcntl.flock(run_lock, fcntl.LOCK_EX)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    control = args.control_command or args.command
    series: dict[str, list[dict]] = {args.name: []}
    if control != "none":
        series[args.control_name] = []
    for index in range(args.trials):
        record = run_trial(args, "subject", index, args.command)
        series[args.name].append(record)
        print(f"subject {index}: " + describe(record), flush=True)
        if control != "none":
            record = run_trial(args, "control", index, control)
            series[args.control_name].append(record)
            print(f"control {index}: " + describe(record), flush=True)
        summary, text = summarize(args, series)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        (out / "summary.md").write_text(text)
    print(text)
    return 0


def describe(record: dict) -> str:
    if "achieved_m" not in record:
        return record.get("discarded", "failed")
    text = (
        f"{record['achieved_m']:.1f}M ({record['ratio'] * 100:.1f}% of offered), "
        f"p50 {record['p50_us']:.1f} us, p99 {record['p99_us']:.0f} us"
    )
    return text + (f" DISCARDED: {record['discarded']}" if "discarded" in record else "")


if __name__ == "__main__":
    sys.exit(main())
