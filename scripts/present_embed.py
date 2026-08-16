#!/usr/bin/env python3
"""Inline local images into a presentation page as data URIs.

Report inputs live in `artifacts/reports/` as `<name>.html.in`. The `.in`
suffix is deliberate: the token-bearing input is not a browser page. This tool
publishes the one self-contained `<name>.html` that opens in the user's browser.
Captures must be embedded: `artifacts/visual/latest/` and the comparison dirs
are overwritten on every rerun, so a page referencing them live silently
changes under the reader. Embedding freezes the evidence and keeps the page
portable.

Workflow: write `<name>.html.in` with `{{img:<path>}}` tokens where the
captures go, then run this script. It encodes each image and writes the
canonical `<name>.html` with the tokens replaced by `<img>` tags. When Pillow
is available, large images are downscaled first. The token source stays
editable, so the page can be regenerated after a recapture.

Usage:
    python3 scripts/present_embed.py page.html.in [--width 1400]

Paths inside a token are resolved relative to the page, then relative to the
repository root, so `{{img:artifacts/visual/latest/spaceport-aerial.png}}`
works from anywhere.
"""

from __future__ import annotations

import argparse
import base64
import io
import mimetypes
import re
import sys
from html import escape
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

TOKEN = re.compile(r"\{\{img:([^}|]+?)(?:\|([^}]*))?\}\}")
TOKEN_HINT = re.compile(r"\{\{\s*img\b", re.IGNORECASE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
IMAGE_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
IMAGE_SRC = re.compile(r"\bsrc\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIX = ".html.in"
LEGACY_SOURCE_SUFFIX = ".source.html"
SOURCE_START = '<script type="text/x-thalos-report" id="thalos-report-source">\n'
SOURCE_END = "\n</script>\n"
UNPUBLISHED_MARKER = "Unpublished report input"


def resolve(raw: str, page: Path) -> Path:
    candidate = Path(raw.strip())
    for base in (page.parent, REPO_ROOT, Path.cwd()):
        probe = candidate if candidate.is_absolute() else base / candidate
        if probe.is_file():
            return probe
    raise SystemExit(f"present_embed: no such image: {raw.strip()}")


def encode(path: Path, max_width: int, quality: int) -> tuple[str, int]:
    """Return (data URI, encoded byte count) for a downscaled copy of `path`."""
    if Image is None:
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None or not mime.startswith("image/"):
            raise SystemExit(f"present_embed: unsupported image type: {path}")
        payload = path.read_bytes()
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime};base64,{encoded}", len(payload)

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


def default_output(page: Path) -> Path:
    if not page.name.endswith(SOURCE_SUFFIX):
        if page.name.endswith(LEGACY_SOURCE_SUFFIX):
            stem = page.name.removesuffix(LEGACY_SOURCE_SUFFIX)
            suggested = page.with_name(f"{stem}{SOURCE_SUFFIX}")
        elif page.name.endswith(".html"):
            suggested = page.with_name(f"{page.name}.in")
        else:
            suggested = page.with_name(f"{page.name}{SOURCE_SUFFIX}")
        raise SystemExit(
            "present_embed: report input must not itself be browser-renderable; "
            f"rename it to {suggested}"
        )
    return page.with_name(page.name.removesuffix(".in"))


def extract_report_source(document: str) -> str:
    """Extract the hidden report body from the visibly unpublished input page."""
    start = document.find(SOURCE_START)
    if start < 0 or document.find(SOURCE_START, start + len(SOURCE_START)) >= 0:
        raise SystemExit(
            "present_embed: missing required unpublished-input wrapper; "
            "start from scripts/present_template.html"
        )
    end = document.find(SOURCE_END, start + len(SOURCE_START))
    prefix = document[:start]
    trailing = "" if end < 0 else document[end + len(SOURCE_END) :]
    if end < 0 or trailing.strip() or UNPUBLISHED_MARKER not in prefix:
        raise SystemExit(
            "present_embed: invalid unpublished-input wrapper; "
            "start from scripts/present_template.html"
        )
    return document[start + len(SOURCE_START) : end]


def validate_published(html: str, count: int) -> None:
    """Reject a report that could display placeholders or load live images."""
    if count == 0:
        raise SystemExit(
            "present_embed: malformed or no image tokens; refusing to publish a "
            "report without embedded evidence"
        )
    if TOKEN_HINT.search(html):
        raise SystemExit(
            "present_embed: unresolved image token remains; refusing to publish"
        )

    visible_html = HTML_COMMENT.sub("", html)
    for tag in IMAGE_TAG.findall(visible_html):
        source = IMAGE_SRC.search(tag)
        if source is None or not source.group(2).startswith("data:image/"):
            raise SystemExit(
                "present_embed: non-embedded <img> remains; every report image "
                "must use a data URI"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path, help=".html.in input containing image tokens")
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
    out = default_output(page).resolve()

    source = extract_report_source(page.read_text(encoding="utf-8"))
    total = 0
    count = 0

    def substitute(match: re.Match[str]) -> str:
        nonlocal total, count
        path = resolve(match.group(1), page)
        alt = escape((match.group(2) or path.stem).strip(), quote=True)
        uri, size = encode(path, args.width, args.quality)
        total += size
        count += 1
        print(f"  {path.name}: {size / 1024:.0f} KiB", file=sys.stderr)
        return f'<img src="{uri}" alt="{alt}" loading="lazy">'

    embedded = TOKEN.sub(substitute, source)
    validate_published(embedded, count)

    out.write_text(embedded, encoding="utf-8")
    if Image is None:
        print(
            "present_embed: Pillow unavailable; embedded original images without resizing",
            file=sys.stderr,
        )
    print(
        f"present_embed: READY, {count} embedded image(s), "
        f"{total / 1024:.0f} KiB payload\nOPEN ONLY: {out.as_uri()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
