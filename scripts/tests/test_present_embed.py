from __future__ import annotations

import base64
import mimetypes
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "present_embed.py"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2nJcAAAAASUVORK5CYII="
)
SOURCE_START = '<script type="text/x-thalos-report" id="thalos-report-source">\n'
SOURCE_END = "\n</script>\n"


def wrapped_source(report: str) -> str:
    return (
        "<h1>Unpublished report input</h1>\n"
        "<p>Run just publish-report and open only the printed result.</p>\n"
        f"{SOURCE_START}{report}{SOURCE_END}"
    )


class PresentEmbedTest(unittest.TestCase):
    def run_embed(self, page: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(page)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_non_html_source_publishes_to_canonical_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence.png").write_bytes(PNG_1X1)
            source = root / "report.html.in"
            source.write_text(
                wrapped_source("<figure>{{img:evidence.png|evidence}}</figure>"),
                encoding="utf-8",
            )

            result = self.run_embed(source)

            self.assertEqual(result.returncode, 0, result.stderr)
            published = root / "report.html"
            self.assertTrue(published.is_file())
            self.assertTrue(source.is_file())
            self.assertNotEqual(mimetypes.guess_type(source.name)[0], "text/html")
            html = published.read_text(encoding="utf-8")
            self.assertIn('src="data:image/png;base64,', html)
            self.assertNotIn("{{img:", html)
            self.assertNotIn("Unpublished report input", html)
            self.assertNotIn("text/x-thalos-report", html)

    def test_browser_renderable_source_fails_with_rename_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "report.source.html"
            page.write_text("{{img:evidence.png}}", encoding="utf-8")

            result = self.run_embed(page)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("report.html.in", result.stderr)
            self.assertFalse((page.parent / "report.html").exists())

    def test_malformed_image_token_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.html.in"
            source.write_text(
                wrapped_source("<figure>{{img:evidence.png</figure>"),
                encoding="utf-8",
            )

            result = self.run_embed(source)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("malformed or no image tokens", result.stderr)
            self.assertFalse((source.parent / "report.html").exists())

    def test_non_embedded_image_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence.png").write_bytes(PNG_1X1)
            source = root / "report.html.in"
            source.write_text(
                wrapped_source(
                    '{{img:evidence.png|embedded}}'
                    '<img src="evidence.png" alt="live">'
                ),
                encoding="utf-8",
            )

            result = self.run_embed(source)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-embedded <img>", result.stderr)
            self.assertFalse((root / "report.html").exists())

    def test_unwrapped_input_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence.png").write_bytes(PNG_1X1)
            source = root / "report.html.in"
            source.write_text("{{img:evidence.png}}", encoding="utf-8")

            result = self.run_embed(source)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unpublished-input wrapper", result.stderr)
            self.assertFalse((root / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
