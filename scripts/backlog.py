#!/usr/bin/env python3
"""Backlog queue: one JSONL file, filter to read, rewrite one line to close.

docs/backlog.jsonl is the status authority. Analogous to runtime.jsonl:
append a record, `just queue` prints only what's pickable, closing strips the
note so the file cannot grow essays again.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs" / "backlog.jsonl"

LIVE = ("next", "wip", "blocked")
ITEM_STATUSES = LIVE + ("done", "later")
FORK_STATUSES = ("open", "resolved")
NOTE_MAX = 360
TITLE_MAX = 120

TRACK_ORDER = [
    "ntr",
    "atlas",
    "stab",
    "clean",
    "gfx",
    "bio",
    "giant",
    "audio",
    "sea",
    "nav",
    "meta",
    "dmg",
    "cine",
    "fork",
]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load() -> list[dict]:
    if not PATH.exists():
        return []
    rows = []
    for i, line in enumerate(PATH.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{PATH}:{i}: {e}") from e
        rec["_line"] = i
        rows.append(rec)
    return rows


def dump_record(rec: dict) -> str:
    out = {k: v for k, v in rec.items() if not k.startswith("_") and v not in (None, "")}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def replace(rec: dict) -> None:
    """Rewrite only the matching line so a close is a one-line git diff."""
    dumped = dump_record(rec)
    if not PATH.exists():
        PATH.write_text(dumped + "\n")
        return
    lines = PATH.read_text().splitlines()
    line_no = rec.get("_line")
    if isinstance(line_no, int) and 1 <= line_no <= len(lines):
        raw = lines[line_no - 1].strip()
        if raw:
            existing = json.loads(raw)
            if existing.get("id") == rec["id"]:
                lines[line_no - 1] = dumped
                PATH.write_text("\n".join(lines) + "\n")
                return
    ident = rec["id"]
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("id") == ident:
            lines[i] = dumped
            PATH.write_text("\n".join(lines) + "\n")
            return
    lines.append(dumped)
    PATH.write_text("\n".join(lines) + "\n")


def clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def cmd_queue(args: argparse.Namespace) -> int:
    if args.id == "--json":
        args.json = True
        args.id = None
    rows = load()
    if args.id:
        hits = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in rows
            if r.get("id") == args.id
        ]
        if not hits:
            print(f"no record {args.id}", file=sys.stderr)
            return 1
        payload = hits[0] if len(hits) == 1 else hits
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    items = [r for r in rows if r.get("kind", "item") == "item" and r.get("status") in LIVE]
    forks = [r for r in rows if r.get("kind") == "fork" and r.get("status") == "open"]
    long_notes = [r for r in items if len(r.get("note") or "") > NOTE_MAX]
    counts = Counter(r["status"] for r in items)

    if args.json:
        json.dump(
            {
                "next": counts["next"],
                "wip": counts["wip"],
                "blocked": counts["blocked"],
                "overlong_notes": [r["id"] for r in long_notes],
                "items": [
                    {k: v for k, v in r.items() if not k.startswith("_")}
                    for r in items
                ],
                "forks": [
                    {k: v for k, v in r.items() if not k.startswith("_")}
                    for r in forks
                ],
            },
            sys.stdout,
            indent=2,
            ensure_ascii=False,
        )
        sys.stdout.write("\n")
        return 0

    print(
        f"live queue · {counts['next']} next · {counts['wip']} wip · "
        f"{counts['blocked']} blocked · {PATH.relative_to(ROOT)}"
    )
    if long_notes:
        print(
            f"hygiene: {len(long_notes)} live note(s) over {NOTE_MAX} chars — "
            + ", ".join(r["id"] for r in long_notes[:8])
        )
    print("open one id with: just queue <id>   close with: just backlog done <id>")
    print()

    def track_key(r: dict) -> tuple:
        t = r.get("track") or ""
        try:
            ti = TRACK_ORDER.index(t)
        except ValueError:
            ti = 99
        return (ti, t, r.get("id") or "")

    current = None
    for r in sorted(items, key=track_key):
        track = r.get("track") or "(none)"
        if track != current:
            current = track
            print(f"## {track}")
        print(f"  {r['status']:<7} {r['id']:<48} {r.get('title', '')}")

    if forks:
        print()
        print("## forks")
        for r in forks:
            print(f"  {r['id']:<48} {clip(r.get('title', ''), 88)}")

    print()
    print("closed records stay in the jsonl as title-only lines · strategy → docs/roadmap/")
    return 0


def cmd_set_status(ident: str, status: str) -> int:
    rows = load()
    hits = [r for r in rows if r.get("id") == ident]
    if not hits:
        print(f"no record {ident}", file=sys.stderr)
        return 1
    stamp = now_stamp()
    for rec in hits:
        kind = rec.get("kind", "item")
        allowed = FORK_STATUSES if kind == "fork" else ITEM_STATUSES
        if status not in allowed:
            print(f"{ident} is {kind}; status must be one of {allowed}", file=sys.stderr)
            return 1
        rec["status"] = status
        if status in ("done", "later", "resolved"):
            rec.pop("note", None)
        if status == "done":
            rec.pop("est", None)
            rec.pop("deps", None)
        rec["updated"] = stamp
        replace(rec)
    extra = f" ({len(hits)} records)" if len(hits) > 1 else ""
    print(f"{ident} → {status}{extra}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    ident = args.id
    if not ident:
        slug = re.sub(r"[^a-z0-9]+", "-", args.title.lower()).strip("-")[:40]
        ident = "BL-" + now_stamp() + "-" + slug
    rec = {
        "id": ident,
        "kind": "item",
        "track": args.track,
        "status": args.status,
        "title": clip(args.title, TITLE_MAX),
        "note": clip(args.note or "", NOTE_MAX),
        "est": args.est or "",
        "deps": args.deps or "",
        "refs": args.refs or "",
        "updated": now_stamp(),
    }
    if any(r.get("id") == ident for r in load()):
        print(f"id already exists: {ident}", file=sys.stderr)
        return 1
    replace(rec)
    print(ident)
    return 0


def cmd_note(args: argparse.Namespace) -> int:
    rows = load()
    hits = [r for r in rows if r.get("id") == args.id]
    if not hits:
        print(f"no record {args.id}", file=sys.stderr)
        return 1
    if len(hits) > 1:
        print(f"{args.id} is not unique; refuse to edit a colliding frozen id", file=sys.stderr)
        return 1
    rec = hits[0]
    rec["note"] = clip(args.note, NOTE_MAX)
    rec["updated"] = now_stamp()
    replace(rec)
    print(f"{args.id} note {len(rec['note'])} chars")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("queue", help="print pickable work")
    q.add_argument("id", nargs="?", help="print one record as JSON")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=lambda a: cmd_queue(a))

    for name in ("next", "wip", "blocked", "done", "later"):
        s = sub.add_parser(name, help=f"set status to {name}")
        s.add_argument("id")
        s.set_defaults(func=lambda a, st=name: cmd_set_status(a.id, st))

    for name in ("open", "resolved"):
        s = sub.add_parser(name, help=f"set fork status to {name}")
        s.add_argument("id")
        s.set_defaults(func=lambda a, st=name: cmd_set_status(a.id, st))

    a = sub.add_parser("add", help="append a live item")
    a.add_argument("--track", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--id")
    a.add_argument("--status", default="next", choices=ITEM_STATUSES)
    a.add_argument("--note", default="")
    a.add_argument("--est", default="")
    a.add_argument("--deps", default="")
    a.add_argument("--refs", default="")
    a.set_defaults(func=cmd_add)

    n = sub.add_parser("note", help="set the live note (capped at 360)")
    n.add_argument("id")
    n.add_argument("note")
    n.set_defaults(func=cmd_note)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
