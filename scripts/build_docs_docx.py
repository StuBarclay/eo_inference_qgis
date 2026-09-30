#!/usr/bin/env python3
"""Combine the Markdown docs into a single Word document via pandoc.

Reads the docs in a deliberate reading order, concatenates them with a title
block and a page break between sections, and runs pandoc to produce
``docs/EO_Inference_Toolbox.docx`` with a table of contents.

Requires pandoc on PATH (``apt install pandoc`` / ``brew install pandoc``).

The combined Markdown is assembled in a temp directory; the final .docx is
written by pandoc to a temp path and then copied to its destination with
``shutil.copyfile`` (OneDrive-safe: overwrites in place without unlinking).

Usage
-----
    python scripts/build_docs_docx.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT = DOCS_DIR / "EO_Inference_Toolbox.docx"

# (source file, section heading) in reading order. README leads as the overview.
SECTIONS: list[tuple[Path, str]] = [
    (REPO_ROOT / "README.md", "Overview"),
    (DOCS_DIR / "ARCHITECTURE.md", "Architecture"),
    (DOCS_DIR / "METHODOLOGY.md", "Methodology"),
    (DOCS_DIR / "USER_GUIDE.md", "User guide"),
    (DOCS_DIR / "LIMITATIONS.md", "Limitations"),
    (DOCS_DIR / "ROADMAP.md", "Roadmap"),
    (DOCS_DIR / "RELEASE_NOTES.md", "Release notes"),
    (DOCS_DIR / "SMOKE_TEST.md", "Smoke test"),
]

TITLE_BLOCK = (
    "---\n"
    "title: EO Inference Toolbox\n"
    "subtitle: A QGIS Processing plugin for Clay, Prithvi-EO-2.0 and SatlasPretrain\n"
    "---\n\n"
)


def assemble_markdown() -> str:
    """Concatenate the docs into one Markdown string with page breaks."""
    parts: list[str] = [TITLE_BLOCK]
    for i, (path, _heading) in enumerate(SECTIONS):
        if not path.exists():
            raise SystemExit(f"missing doc: {path}")
        text = path.read_text(encoding="utf-8")
        if i > 0:
            # Raw OpenXML page break so each doc starts on a fresh page.
            parts.append('\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n')
        parts.append(text.rstrip() + "\n")
    return "\n".join(parts)


def build() -> Path:
    """Run pandoc over the combined Markdown to produce the .docx."""
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found on PATH; install it to build the docx.")

    combined = assemble_markdown()
    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / "combined.md"
        docx_path = Path(tmp) / "combined.docx"
        md_path.write_text(combined, encoding="utf-8")
        subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-o",
                str(docx_path),
                "--toc",
                "--toc-depth=2",
                "--standalone",
                "-f",
                "markdown+pipe_tables+raw_attribute",
            ],
            check=True,
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(docx_path, OUTPUT)  # OneDrive-safe overwrite
    return OUTPUT


def main() -> None:
    out = build()
    print(f"Built {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
