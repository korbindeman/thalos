#!/usr/bin/env python3
"""Inline local images into a presentation page as data URIs.

Reports live in `artifacts/reports/` and open in the user's browser, but the
captures they show must still be embedded: `artifacts/visual/latest/` and the
comparison dirs are overwritten on every rerun, so a page referencing them
live silently changes under the reader. Embedding freezes the evidence, and
keeps the page publishable as a claude.ai Artifact (whose CSP blocks every
external load) when it needs to travel — neither is something an agent should
be pasting by hand.

Workflow: write the page with `{{img:<path>}}` tokens where the captures go,
then run this script. It downscales each image, encodes it, and writes a
sibling `<name>.embedded.html` with the tokens replaced by `<img>` tags. The
token source stays editable, so the page can be regenerated after a recapture.

Usage:
    python3 scripts/present_embed.py page.html [-o out.html] [--width 1400]

Paths inside a token are resolved relative to the page, then relative to the
repository root, so `{{img:artifacts/visual/latest/spaceport-aerial.png}}`
works from anywhere.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

TOKEN = re.compile(r"\{\{img:([^}|]+?)(?:\|([^}]*))?\}\}")
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(raw: str, page: Path) -> Path:
    candidate = Path(raw.strip())
    for base in (page.parent, REPO_ROOT, Path.cwd()):
        probe = candidate if candidate.is_absolute() else base / candidate
        if probe.is_file():
            return probe
    raise SystemExit(f"present_embed: no such image: {raw.strip()}")


def encode(path: Path, max_width: int, quality: int) -> tuple[str, int]:
    """Return (data URI, encoded byte count) for a downscaled copy of `path`."""
    image = Image.open(path)
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.LANCZOS)

    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    buffer = io.BytesIO()
    if has_alpha:
        image.convert("RGBA").save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        image.convert("RGB").save(
            buffer, format="JPEG", quality=quality, optimize=True, progressive=True
        )
        mime = "image/jpeg"

    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}", len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path, help="HTML page containing {{img:…}} tokens")
    parser.add_argument("-o", "--out", type=Path, help="output path")
    parser.add_argument(
        "--width", type=int, default=1400, help="max image width in px (default 1400)"
    )
    parser.add_argument(
        "--quality", type=int, default=82, help="JPEG quality (default 82)"
    )
    args = parser.parse_args()

    page = args.page.resolve()
    if not page.is_file():
        raise SystemExit(f"present_embed: no such page: {args.page}")
    out = args.out or page.with_suffix(".embedded.html")

    source = page.read_text(encoding="utf-8")
    total = 0
    count = 0

    def substitute(match: re.Match[str]) -> str:
        nonlocal total, count
        path = resolve(match.group(1), page)
        alt = (match.group(2) or path.stem).strip()
        uri, size = encode(path, args.width, args.quality)
        total += size
        count += 1
        print(f"  {path.name}: {size / 1024:.0f} KiB", file=sys.stderr)
        return f'<img src="{uri}" alt="{alt}" loading="lazy">'

    embedded = TOKEN.sub(substitute, source)
    if count == 0:
        print("present_embed: no {{img:…}} tokens found", file=sys.stderr)

    out.write_text(embedded, encoding="utf-8")
    print(
        f"present_embed: {count} image(s), {total / 1024:.0f} KiB payload -> {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
