#!/usr/bin/env python3
"""Build a distributable QGIS plugin ZIP for the EO Inference Toolbox.

The archive contains a single top-level folder, ``eo_inference_toolbox/`` (the
name QGIS uses for the installed plugin directory), holding the package plus its
bundled ``LICENSE`` and ``NOTICE``. Nothing heavy is included: no weights, no
caches, no tests, no editor metadata.

Design notes
------------
* **stdlib only** — runs anywhere, no third-party build tooling.
* **Reproducible** — entries are added in sorted order with a fixed timestamp, so
  the same source tree always yields a byte-identical ZIP.
* **OneDrive-safe** — the ZIP is assembled in a temp directory and then copied to
  the destination with ``shutil.copyfile``. We never unlink the destination in
  place (unlink is blocked on the user's OneDrive-synced working tree), and
  ``copyfile`` truncates-and-overwrites an existing file rather than removing it.

Usage
-----
    python scripts/build_plugin_zip.py [--outdir DIR]

Writes ``eo_inference_toolbox-<version>.zip`` (version read from metadata.txt).
"""

from __future__ import annotations

import argparse
import configparser
import shutil
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "eo_inference_toolbox"
PACKAGE_DIR = REPO_ROOT / PACKAGE

# Files/dirs never shipped in the plugin ZIP.
EXCLUDE_DIRS = {"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".git", ".idea"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

# A fixed, deterministic timestamp for every entry (1980-01-01, the ZIP epoch).
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def read_version() -> str:
    """Read ``version=`` from the plugin's metadata.txt."""
    parser = configparser.ConfigParser()
    parser.read(PACKAGE_DIR / "metadata.txt", encoding="utf-8")
    return parser["general"]["version"].strip()


def iter_plugin_files() -> list[Path]:
    """Every file that belongs in the ZIP, as absolute paths, sorted."""
    files: list[Path] = []
    for path in PACKAGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(PACKAGE_DIR).parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def build_zip(destination: Path) -> Path:
    """Assemble the ZIP in a temp dir, then copy it to ``destination``."""
    files = iter_plugin_files()
    if not files:
        raise SystemExit(f"no files found under {PACKAGE_DIR}")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / destination.name
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                arcname = Path(PACKAGE) / path.relative_to(PACKAGE_DIR)
                info = zipfile.ZipInfo(arcname.as_posix(), date_time=ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, path.read_bytes())
        destination.parent.mkdir(parents=True, exist_ok=True)
        # copyfile overwrites in place without unlinking (OneDrive-safe).
        shutil.copyfile(staged, destination)

    return destination


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the EO Inference Toolbox plugin ZIP.")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "dist",
        help="Directory to write the ZIP into (default: ./dist).",
    )
    args = ap.parse_args()

    version = read_version()
    destination = args.outdir / f"{PACKAGE}-{version}.zip"
    out = build_zip(destination)

    with zipfile.ZipFile(out) as zf:
        n = len(zf.namelist())
    size = out.stat().st_size
    print(f"Built {out} ({n} files, {size} bytes)")


if __name__ == "__main__":
    main()
