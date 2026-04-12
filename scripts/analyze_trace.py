#!/usr/bin/env python3
"""Aggregate a Bevy Chrome-tracing JSON file into a top-span report.

Bevy's `trace_chrome` feature writes an array of duration events (phases
"B"/"E" pairs, or "X" complete events) via the tracing-chrome crate. This
script walks the file, groups events by span name, and prints totals
suitable for feeding to an LLM or a human.

Usage:
    python3 scripts/analyze_trace.py trace-2026-04-12_12-34-56.json [--top 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpanStats:
    count: int = 0
    total_us: float = 0.0
    min_us: float = float("inf")
    max_us: float = 0.0


@dataclass
class ThreadState:
    stack: list = field(default_factory=list)


def iter_events(path: Path):
    """Yield events from a tracing-chrome JSON file.

    tracing-chrome writes a single JSON array, one event per line between
    the brackets. We parse line-by-line so trace files larger than RAM
    still work.
    """
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def aggregate(path: Path) -> tuple[dict[str, SpanStats], float]:
    stats: dict[str, SpanStats] = defaultdict(SpanStats)
    threads: dict[int, ThreadState] = defaultdict(ThreadState)

    min_ts = float("inf")
    max_ts = 0.0

    for ev in iter_events(path):
        ph = ev.get("ph")
        name = ev.get("name") or ""
        ts = ev.get("ts")
        tid = ev.get("tid", 0)

        if ts is not None:
            if ts < min_ts:
                min_ts = ts
            if ts > max_ts:
                max_ts = ts

        if ph == "X":
            dur = float(ev.get("dur", 0))
            s = stats[name]
            s.count += 1
            s.total_us += dur
            if dur < s.min_us:
                s.min_us = dur
            if dur > s.max_us:
                s.max_us = dur
        elif ph == "B":
            threads[tid].stack.append((name, float(ts or 0)))
        elif ph == "E":
            st = threads[tid].stack
            if not st:
                continue
            open_name, open_ts = st.pop()
            if open_name != name:
                continue
            dur = float(ts or 0) - open_ts
            if dur < 0:
                continue
            s = stats[name]
            s.count += 1
            s.total_us += dur
            if dur < s.min_us:
                s.min_us = dur
            if dur > s.max_us:
                s.max_us = dur

    total_wall_us = max_ts - min_ts if max_ts > min_ts else 0.0
    return stats, total_wall_us


def fmt_us(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f} s"
    if us >= 1_000:
        return f"{us / 1_000:.2f} ms"
    return f"{us:.1f} us"


def print_report(stats: dict[str, SpanStats], wall_us: float, top: int) -> None:
    rows = [
        (
            name,
            s.count,
            s.total_us,
            s.total_us / s.count if s.count else 0.0,
            s.min_us if s.min_us != float("inf") else 0.0,
            s.max_us,
        )
        for name, s in stats.items()
    ]
    rows.sort(key=lambda r: r[2], reverse=True)
    rows = rows[:top]

    print(f"Wall clock covered: {fmt_us(wall_us)}")
    print(f"Unique span names:  {len(stats)}")
    print()
    header = f"{'rank':>4}  {'total':>10}  {'% wall':>7}  {'count':>8}  {'mean':>10}  {'min':>10}  {'max':>10}  name"
    print(header)
    print("-" * len(header))
    for i, (name, count, total, mean, mn, mx) in enumerate(rows, 1):
        pct = (total / wall_us * 100.0) if wall_us > 0 else 0.0
        print(
            f"{i:>4}  {fmt_us(total):>10}  {pct:>6.1f}%  {count:>8}  "
            f"{fmt_us(mean):>10}  {fmt_us(mn):>10}  {fmt_us(mx):>10}  {name}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="Path to a Bevy chrome trace JSON file")
    ap.add_argument("--top", type=int, default=40, help="Number of spans to show")
    args = ap.parse_args()

    if not args.trace.exists():
        print(f"error: {args.trace} not found", file=sys.stderr)
        return 1

    stats, wall_us = aggregate(args.trace)
    if not stats:
        print("error: no events parsed — is this a Bevy chrome trace?", file=sys.stderr)
        return 1

    print_report(stats, wall_us, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
