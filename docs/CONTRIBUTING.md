# Contributing — developer install and checks

The maintainer-facing companion to the end-user docs. For installing the plugin
*inside QGIS* see [USER_GUIDE.md](USER_GUIDE.md); for the manual end-to-end
verification that needs a running QGIS see [SMOKE_TEST.md](SMOKE_TEST.md); for the
layering rules the code must obey see [ARCHITECTURE.md](ARCHITECTURE.md). This file
covers three things: setting up a development environment, the automated **checks**
("the four gates"), and **building and verifying** the release artifacts.

## Development environment

The pure-Python core (`eo_inference_toolbox/eocore/`) is deliberately free of GDAL,
QGIS, PyTorch and scikit-learn, and heavy libraries are imported lazily at call
time, so the whole automated suite runs in a plain Python virtual environment with
**no geospatial or deep-learning stack installed**. Python 3.10 or newer is required
(CI runs 3.10 and 3.12).

```bash
git clone https://github.com/StuBarclay/eo_inference_qgis
cd eo_inference_qgis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"  # ruff, mypy, pytest, numpy
```

Two optional extras improve coverage locally but are not required: `pip install
pyyaml` lets the Clay metadata-parsing test run instead of skip, and `pandoc` on
`PATH` is needed only to build the combined Word document (below).

The plugin's runtime tiers — `torch`, the backend packages (`claymodel`,
`terratorch`, `satlaspretrain-models`), `scikit-learn`, `huggingface_hub` and
`planetary-computer` — are intentionally **not** development dependencies. They are
opt-in tiers installed at run time from the wizard (see the user guide), the gates
do not need them, and the handful of tests that would need GDAL/QGIS are gated and
skip cleanly when those libraries are absent.

## The four gates (checking procedures)

Run all four from the repository root before every commit. They are exactly what
CI enforces, so a local pass means a green build:

```bash
ruff check .            # 1. lint
ruff format --check .   # 2. formatting (use `ruff format .` to apply fixes)
mypy                    # 3. types (config in pyproject.toml)
pytest                  # 4. tests
```

A clean tree gives `ruff check` "All checks passed!", `ruff format --check` "N files
already formatted", `mypy` "Success: no issues found in 69 source files", and
`pytest` "336 passed, 2 skipped" (the two skips are the GDAL/QGIS-gated tests).

`mypy` reads its configuration from `pyproject.toml`: strict everywhere, with a
relaxation for the QGIS-facing modules (`provider`, `plugin`, `algorithms.*`,
`gui.*`) that subclass untyped QGIS base classes, while `eocore` stays fully strict.
It type-checks both `eo_inference_toolbox` and `tests`, so new tests are checked too.

### GDAL/QGIS-gated tests

Two test files call `importorskip` on `osgeo` / `qgis` and therefore skip in a plain
virtual environment. CI runs them for real in the official `qgis/qgis:release-3_34`
container (the `gdal-qgis` job). To run them locally, use a Python that already has
PyQGIS and GDAL — the OSGeo4W Python on Windows, or the QGIS container.

### On a cloud-synced working tree (OneDrive / Dropbox / iCloud)

This repository typically lives in a synced folder, which has two consequences for
the checks and builds:

- `mypy`'s sqlite cache can raise a disk-I/O error on a synced path. Point it
  elsewhere: `mypy --cache-dir /tmp/mypy_cache` (any non-synced directory works).
- The build scripts below already write safely (they assemble in a temp directory
  and overwrite in place), but avoid running index-touching `git` operations from a
  secondary/sandboxed process while the sync client holds file locks — commit and
  push from your normal shell.

## Building and verifying the release artifacts

Two scripts produce the shippable outputs. Both are safe to re-run on a synced tree:
each assembles in a temporary directory and then overwrites the destination in place
with `shutil.copyfile`, never unlinking it (unlink is blocked on OneDrive).

```bash
python scripts/build_plugin_zip.py   # -> dist/eo_inference_toolbox-<version>.zip
python scripts/build_docs_docx.py    # -> docs/EO_Inference_Toolbox.docx (needs pandoc)
```

The ZIP version is read from `eo_inference_toolbox/metadata.txt`. The docx
concatenates the user-facing docs in reading order (README, then Architecture,
Methodology, User guide, Limitations, Roadmap, Release notes, Smoke test); the
developer-facing docs — this file, `ADDING_A_BACKEND.md` and `DEVELOPMENT_PLAN.md` —
are intentionally left out of it.

### What ships in the ZIP

The archive contains a single top-level `eo_inference_toolbox/` folder (the name QGIS
installs the plugin under), holding the package plus its bundled `LICENSE` and
`NOTICE`. The builder includes **every non-`.pyc` file under the package**, so
non-code assets such as `eocore/backends/clay_assets/metadata.yaml` are shipped. It
excludes `__pycache__`, the `.mypy_cache` / `.ruff_cache` / `.pytest_cache` / `.git`
/ `.idea` directories, and `.pyc` / `.pyo` files. The `docs/`, `tests/` and
`scripts/` trees, `pyproject.toml` and the root `README.md` are development files and
are **not** in the ZIP.

### Verifying the ZIP

A good ZIP has exactly one top-level folder, carries the vendored assets, and
contains none of: a `tests/` tree, `.pyc` files, editor directories, or the stray
root `main.py`:

```bash
python - <<'PY'
import glob, os, zipfile
z = sorted(glob.glob('dist/eo_inference_toolbox-*.zip'), key=os.path.getmtime)[-1]
names = zipfile.ZipFile(z).namelist()
print('ZIP:', os.path.basename(z), os.path.getsize(z), 'bytes;', len(names), 'files')
print('single top folder:', {n.split("/", 1)[0] for n in names} == {'eo_inference_toolbox'})
print('clay metadata.yaml present:', any(n.endswith('clay_assets/metadata.yaml') for n in names))
print('no tests:', not any('/tests/' in n or n.startswith('tests/') for n in names))
print('no stray main.py:', not any(n.endswith('/main.py') for n in names))
print('no .pyc:', not any(n.endswith('.pyc') for n in names))
PY
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`, with two
jobs. **lint-and-test** sets up a plain Python (3.10 and 3.12), installs the tooling,
and runs the four gates — this is the pure-core gate and mirrors the local commands
above. **gdal-qgis** runs inside the `qgis/qgis:release-3_34` container so the
GDAL/QGIS-gated tests execute for real rather than skipping. Keep both green.

## Before tagging a release

Work through [SMOKE_TEST.md](SMOKE_TEST.md) in a real QGIS (both a 3.34 LTR / PyQt5
build and a 4.x / PyQt6 build), record the run in its log block, and clear the
remaining standing release-blockers: a version bump with a changelog entry in both
`metadata.txt` and [RELEASE_NOTES.md](RELEASE_NOTES.md), and rebuilt ZIP + docx. (The
contact email in `metadata.txt` is set and the stray root `main.py` has been removed.) Then create the GitHub Release with the ZIP attached and submit
to plugins.qgis.org; clear the `experimental` flag in `metadata.txt` once it has been
verified in a real QGIS.
