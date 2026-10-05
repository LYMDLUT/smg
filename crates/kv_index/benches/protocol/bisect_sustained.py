#!/usr/bin/env python3
"""Bracket an indexer's sustained throughput: the highest offered rate at which trials keep up.

A trial keeps up when its generator was valid and it achieved at least `--keep-up-ratio` (0.99)
of the offered rate. A point passes when every one of its `--trials` fresh-process trials keeps
up. Starting from a rate known to pass (`--lo`) and one known to fail (`--hi`), the search moves
the geometric midpoint until hi / lo <= 1 + tolerance.

The command template runs one trial. Placeholders: `{rate}` (block ops per second, for a harness
with `--offered-block-ops-per-sec`), `{window_ms}` (for a harness driven by the window; needs
`--total-block-ops`), `{json}` (result path), `{point}` and `{trial}` (indices). Each trial runs
under the lock file (flock), with a foreign-load sample before and after it recorded in the
output.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hostload  # noqa: E402


def run_trial(
    args: argparse.Namespace, point: int, trial: int, rate: float, out: pathlib.Path
) -> dict:
    window_ms = (
        max(1, round(args.total_block_ops / rate * 1000.0)) if args.total_block_ops else None
    )
    result = out / f"point{point}-trial{trial}.json"
    command = args.command.format(
        rate=f"{rate:.0f}", window_ms=window_ms, json=result, point=point, trial=trial
    )
    cores = hostload.parse_cpu_list(args.cores)
    allow = re.compile(args.allow) if args.allow else None
    record = {
        "point": point,
        "trial": trial,
        "rate_requested": rate,
        "window_ms": window_ms,
        "command": command,
    }
    with open(args.lock, "w") as lock:
        if args.lock_scope == "trial":
            fcntl.flock(lock, fcntl.LOCK_EX)
        before = hostload.sample(cores, args.sample_seconds, {os.getpid()})
        started = time.time()
        with open(out / f"point{point}-trial{trial}.log", "w") as log:
            status = subprocess.run(shlex.split(command), stdout=log, stderr=subprocess.STDOUT)
        record["wall_s"] = time.time() - started
        after = hostload.sample(cores, args.sample_seconds, {os.getpid()})
        if args.lock_scope == "trial":
            fcntl.flock(lock, fcntl.LOCK_UN)
    record["exit_code"] = status.returncode
    foreign_b, background_b = hostload.classify(before, args.threshold_pct, allow, False)
    foreign_a, background_a = hostload.classify(after, args.threshold_pct, allow, False)
    record["foreign_load"] = foreign_b + foreign_a
    record["background_load"] = background_b + background_a
    if status.returncode != 0 or not result.exists():
        record.update(kept_up=False, error="trial failed")
        return record
    data = json.loads(result.read_text())
    offered = data["offered_block_ops_per_sec"]
    achieved = data["achieved_block_ops_per_sec"]
    record.update(
        offered=offered,
        achieved=achieved,
        ratio=achieved / offered if offered else 0.0,
        generator_valid=bool(data["generator_valid"]),
        kept_up=bool(data["generator_valid"]) and achieved >= args.keep_up_ratio * offered,
        lookup_p50_us=data["query_service"]["p50_ns"] / 1e3,
        lookup_p99_us=data["query_service"]["p99_ns"] / 1e3,
        drain_ms=data["drain_ns"] / 1e6,
        total_block_ops=data.get("total_block_ops"),
        failure_reasons=data.get("failure_reasons", []),
    )
    return record


def fmt_rate(rate: float) -> str:
    return f"{rate / 1e6:.1f}M"


def span(values: list[float], digits: int) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.{digits}f}" if lo == hi else f"{lo:.{digits}f}-{hi:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--command", required=True, help="trial command template (see module doc)")
    parser.add_argument(
        "--lo", type=float, required=True, help="rate known to keep up (block ops/s)"
    )
    parser.add_argument("--hi", type=float, required=True, help="rate known to fail (block ops/s)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--tolerance", type=float, default=0.10, help="stop when hi/lo <= 1 + tolerance"
    )
    parser.add_argument("--max-points", type=int, default=8)
    parser.add_argument("--keep-up-ratio", type=float, default=0.99)
    parser.add_argument(
        "--total-block-ops", type=int, default=0, help="needed when the template uses {window_ms}"
    )
    parser.add_argument("--verify-ends", action="store_true", help="run the endpoints first")
    parser.add_argument(
        "--lock-scope",
        choices=("trial", "run"),
        default="trial",
        help="hold the lock per trial (fair to other takers) or for the whole run (one queue wait)",
    )
    parser.add_argument(
        "--lock", required=True, help="lock file held for the duration of each trial"
    )
    parser.add_argument("--cores", default="0-63", help="cores to check for foreign load")
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument(
        "--allow", default="", help="regex of background processes to record, not flag"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_lock = open(args.lock, "w")  # noqa: SIM115 - held until exit when the scope is "run"
    if args.lock_scope == "run":
        fcntl.flock(run_lock, fcntl.LOCK_EX)
    if "{window_ms}" in args.command and not args.total_block_ops:
        parser.error("--total-block-ops is required with a {window_ms} template")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = args.lo, args.hi
    points: list[dict] = []

    def measure(rate: float) -> bool:
        index = len(points)
        trials = [run_trial(args, index, t, rate, out) for t in range(args.trials)]
        passed = all(t["kept_up"] for t in trials)
        points.append({"rate": rate, "passed": passed, "trials": trials})
        line = ", ".join(
            f"{t.get('ratio', 0) * 100:.1f}%{'' if t['kept_up'] else '!'}"
            + (" FL" if t["foreign_load"] else "")
            for t in trials
        )
        print(
            f"point {index}: {fmt_rate(rate)} offered -> {'pass' if passed else 'fail'} [{line}]",
            flush=True,
        )
        (out / "bracket.json").write_text(
            json.dumps({"lo": lo, "hi": hi, "points": points}, indent=1)
        )
        return passed

    if args.verify_ends:
        if not measure(lo):
            print(f"--lo {fmt_rate(lo)} does not keep up; lower it", file=sys.stderr)
            return 2
        if measure(hi):
            print(f"--hi {fmt_rate(hi)} keeps up; raise it", file=sys.stderr)
            return 2
    while hi / lo > 1.0 + args.tolerance and len(points) < args.max_points:
        mid = math.sqrt(lo * hi)
        if measure(mid):
            lo = mid
        else:
            hi = mid
    summary = {
        "lo_keeps_up": lo,
        "hi_fails": hi,
        "bracket_ratio": hi / lo,
        "within_tolerance": hi / lo <= 1.0 + args.tolerance,
        "trials_per_point": args.trials,
        "keep_up_ratio": args.keep_up_ratio,
        "points": points,
    }
    (out / "bracket.json").write_text(json.dumps(summary, indent=1))
    header = (
        "| Point | Offered | Verdict | Achieved / offered per trial | Lookup p50 (us) "
        "| Lookup p99 (us) | Foreign load |"
    )
    lines = [
        f"Sustained bracket: keeps up at {fmt_rate(lo)}, fails at {fmt_rate(hi)} "
        f"(ratio {hi / lo:.3f}, {args.trials} trials per point, keep-up ratio {args.keep_up_ratio}).",
        "",
        header,
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, p in enumerate(points):
        ok = [t for t in p["trials"] if "ratio" in t]
        ratios = ", ".join(f"{t['ratio'] * 100:.1f}%" for t in ok) or "failed"
        p50 = span([t["lookup_p50_us"] for t in ok], 1) if ok else "-"
        p99 = span([t["lookup_p99_us"] for t in ok], 0) if ok else "-"
        flagged = sum(1 for t in p["trials"] if t["foreign_load"])
        verdict = "pass" if p["passed"] else "fail"
        lines.append(
            f"| {i} | {fmt_rate(p['rate'])} | {verdict} | {ratios} | {p50} | {p99} | {flagged} |"
        )
    (out / "bracket.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["within_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main())
