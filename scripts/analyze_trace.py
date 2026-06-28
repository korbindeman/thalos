#!/usr/bin/env python3
"""Aggregate a Bevy Chrome-tracing JSON file into a top-span report.

Bevy's `trace_chrome` feature writes an array of duration events (phases
"B"/"E" pairs, or "X" complete events) via the tracing-chrome crate. This
script walks the file, groups events by span name, and prints totals
suitable for feeding to an LLM or a human.

Usage:
    python3 scripts/analyze_trace.py trace-2026-04-12_12-34-56.json [--top 40]

Event windowing:
    Aggregate just the windows around a named span instead of the whole
    capture:

        python3 scripts/analyze_trace.py trace-…json \
            --around-name <span> --window-ms 200

    This finds every occurrence of `<span>` in the trace, builds a union of
    ±window/2 windows around them, and reports the top spans within that
    union. Use it to skip past the "boring" frames in a long capture and
    zoom in on a span of interest.
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


def collect_windows(path: Path, name: str, window_us: float) -> list[tuple[float, float]]:
    """First pass: find ts of every event matching `name` and build merged
    [start, end] windows of ±window_us/2 around each.

    Matches both `B` (begin span) and `X` (complete) phases — `slow_frame`
    is a zero-duration scope so either could be emitted depending on the
    tracing-chrome layer configuration.
    """
    half = window_us / 2.0
    ranges: list[tuple[float, float]] = []
    for ev in iter_events(path):
        if ev.get("name") != name:
            continue
        ph = ev.get("ph")
        ts = ev.get("ts")
        if ts is None or ph not in ("B", "X", "i"):
            continue
        ts = float(ts)
        ranges.append((ts - half, ts + half))

    if not ranges:
        return []

    ranges.sort()
    merged: list[tuple[float, float]] = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def ts_in_windows(ts: float, windows: list[tuple[float, float]]) -> bool:
    # Trace files are big but typical window counts (1..100) make a linear
    # scan fine. Binary-search if this ever turns into a hot path.
    for start, end in windows:
        if start <= ts <= end:
            return True
        if start > ts:
            return False
    return False


def aggregate(
    path: Path, windows: list[tuple[float, float]] | None = None
) -> tuple[dict[str, SpanStats], float]:
    stats: dict[str, SpanStats] = defaultdict(SpanStats)
    threads: dict[int, ThreadState] = defaultdict(ThreadState)

    min_ts = float("inf")
    max_ts = 0.0

    def in_scope(ts_val: float | None) -> bool:
        if windows is None:
            return True
        if ts_val is None:
            return False
        return ts_in_windows(ts_val, windows)

    for ev in iter_events(path):
        ph = ev.get("ph")
        name = ev.get("name") or ""
        ts = ev.get("ts")
        tid = ev.get("tid", 0)

        ts_f = float(ts) if ts is not None else None

        if ph == "X":
            if not in_scope(ts_f):
                continue
            dur = float(ev.get("dur", 0))
            s = stats[name]
            s.count += 1
            s.total_us += dur
            if dur < s.min_us:
                s.min_us = dur
            if dur > s.max_us:
                s.max_us = dur
            if ts_f is not None:
                if ts_f < min_ts:
                    min_ts = ts_f
                if ts_f > max_ts:
                    max_ts = ts_f
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
            # Filter on the span's BEGIN ts so an entire span is either in
            # or out of scope — splitting B/E across a window boundary would
            # give nonsensical durations.
            if not in_scope(open_ts):
                continue
            s = stats[name]
            s.count += 1
            s.total_us += dur
            if dur < s.min_us:
                s.min_us = dur
            if dur > s.max_us:
                s.max_us = dur
            if open_ts < min_ts:
                min_ts = open_ts
            if ts_f is not None and ts_f > max_ts:
                max_ts = ts_f

    if windows is not None:
        # Wall clock in this mode is the union of windows, not file extent.
        total_wall_us = sum(end - start for start, end in windows)
    else:
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
    ap.add_argument(
        "--around-name",
        type=str,
        default=None,
        help="Restrict aggregation to ±window_ms/2 windows around every "
        "event with this span name (e.g. 'Simulation::step').",
    )
    ap.add_argument(
        "--window-ms",
        type=float,
        default=200.0,
        help="Window width in ms around each --around-name event (default 200).",
    )
    args = ap.parse_args()

    if not args.trace.exists():
        print(f"error: {args.trace} not found", file=sys.stderr)
        return 1

    windows: list[tuple[float, float]] | None = None
    if args.around_name:
        window_us = max(args.window_ms, 0.0) * 1000.0
        windows = collect_windows(args.trace, args.around_name, window_us)
        if not windows:
            print(
                f"error: no '{args.around_name}' events found in {args.trace}",
                file=sys.stderr,
            )
            return 1
        coverage_us = sum(end - start for start, end in windows)
        print(
            f"Windowing on '{args.around_name}': {len(windows)} window(s), "
            f"{fmt_us(coverage_us)} covered"
        )
        print()

    stats, wall_us = aggregate(args.trace, windows)
    if not stats:
        print("error: no events parsed — is this a Bevy chrome trace?", file=sys.stderr)
        return 1

    print_report(stats, wall_us, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
