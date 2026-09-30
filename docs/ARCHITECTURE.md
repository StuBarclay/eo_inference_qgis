# Architecture

This document describes how the EO Inference Toolbox is put together: the layers,
the dependency boundaries that keep it installable with zero third-party packages,
and the contracts that hold the pieces together.

## Design goals

The plugin has to satisfy three constraints that pull in different directions. It
must **install and load in a bare QGIS** with nothing but the standard library and
what QGIS already ships (PyQGIS, GDAL, NumPy). It must be able to **run large deep
learning models** (Clay, Prithvi, SatlasPretrain) that drag in PyTorch and hundreds
of megabytes of weights. And its numerical heart must be **unit-testable in a plain
Python sandbox** with no GDAL, no QGIS and no torch.

The way all three are met is a strict three-layer separation with a hard rule about
which layer may import what.

```
        gui/                 (Qt dialog; imports qgis + eocore facts only)
          |
   provider.py / plugin.py   (QGIS Processing provider + plugin lifecycle)
          |
     algorithms/             (thin GDAL/QGIS/sklearn wrappers; lazy heavy imports)
          |
       eocore/               (pure Python + NumPy; no GDAL, QGIS, torch, sklearn)
          |
    eocore/backends/         (torch-free registry + lazily-imported model classes)
```

The golden rule: **dependencies only ever point downward, and the heavy machinery
(torch, scikit-learn, GDAL, weight downloads, network I/O) is imported only at call
time, never at import time.** The QGIS-facing layers (`gui/`, `algorithms/`,
`provider.py`, `plugin.py`) do import `qgis` at module level, so they load inside QGIS
but not in a bare sandbox; `eocore/` imports nothing beyond NumPy, so it — and the
tests that cover it — run under `pytest` with no QGIS, GDAL or torch present.

## Layer 1 — `eocore/` (pure core)

Everything numerically interesting lives here and depends on nothing beyond NumPy.
There is no GDAL, no QGIS, no torch and no scikit-learn anywhere in `eocore/`, which
is what lets the whole layer run under `pytest` in a bare sandbox.

The modules are small and single-purpose. `descriptor.py` defines
`EmbeddingDescriptor` — the frozen record of what a backend produces (band count,
whether it is unit-normalised, native resolution, whether it is temporal) plus
`BackendCapabilities`. `normalize.py` is the correctness linchpin: `l2_normalize`,
`normalize_rows` and `ensure_unit` turn arbitrary embedding vectors into unit
vectors so that cosine similarity collapses to a dot product downstream.
`similarity.py` and `change.py` are the analytics: a per-pixel cosine similarity
map against a reference vector, and pairwise / trajectory / anomaly change between
dated embeddings. `ml.py` holds the sklearn-free plumbing (cube ↔ matrix reshape,
valid-pixel masking, label encoding, chunked prediction, scatter-back-to-grid) that
the classify/cluster algorithms build on. `rgb.py` reduces a high-dimensional
embedding to a 3-band preview by PCA or a chosen band triplet. `grid.py` and
`tiling.py` handle geospatial bookkeeping — a `GridSpec` with geotransform maths,
and a windowing iterator plus a `MosaicAccumulator` that averages overlapping tiles.
`stac.py` builds STAC query bodies and parses responses with an *injectable* fetch
function, so it is fully testable without a network. `model_io.py` saves and loads a
trained-model envelope that stamps the embedding band count and backend id so a model
can never be silently reused against an incompatible raster.

## Layer 2 — `eocore/backends/` (backend registry + models)

This sub-package is deliberately split in two. `base.py` is **import-safe
everywhere**: it holds the `EmbeddingBackend` protocol, the `TileMeta` dataclass,
and `BACKENDS` — a torch-free registry of *facts* about each backend (display name,
default bands, default STAC collection, nominal embedding dimension, licence, pip
packages, weight hints, import names). The GUI and the docs read these facts without
ever importing torch.

The concrete backend classes — `clay.py`, `prithvi.py`, `satlas.py` — import torch
and their model packages **lazily, inside methods**. They are only ever instantiated
through `load_backend(backend_id, **config)`, which is called at run time by the
embed algorithm. Backends are **pluggable**: each registers its `BackendInfo` plus a
lazy loader with a single `register_backend(info, loader)` call, and `load_backend`
consults that table — so adding a backend is one new module plus one registration,
with no edits to the dispatch. The loader defers all heavy imports to call time, so
registration keeps `base.py` torch-free. Importing `eocore.backends` therefore never
pulls in a heavy dependency. See [ADDING_A_BACKEND.md](ADDING_A_BACKEND.md) for the
step-by-step extension guide. Helper modules `_deps.py` (dependency detection),
`_weights.py` (weight resolution / caching / checksum) and `_pool.py` (multi-scale
feature pooling for Satlas) live alongside.

The uncertain external-API seams (the attribute paths and forward-pass outputs of
each upstream model) are guarded by `unsupported_model_error`, `require_model_attr`
and `select_feature` in `base.py`: if an installed checkpoint or library version has
drifted, the backend fails loud with an actionable "pin a known-good version" message
rather than an opaque `AttributeError` at a user's first embed run.

Each backend implements the same contract: `embed_tile(image[C_in, h, w], meta) ->
embedding[D, h, w]`. A backend is responsible for upsampling its own native
patch/token grid back to the tile's pixel resolution, so every backend presents the
identical per-pixel interface regardless of its internal architecture. None of the
three produces unit-normalised output (`is_unit_normalized = False` across the
board), which is exactly why the embed algorithm normalises before writing.

## Layer 3 — `algorithms/` (the QGIS Processing layer)

Each of the seven algorithms is a thin `QgsProcessingAlgorithm` subclass that
translates QGIS parameters into calls down into `eocore`, and writes results back
out through GDAL. Shared machinery lives in the underscore-prefixed helpers:
`_common.py` (the `EoInferenceAlgorithm` base with group / backend-enum plumbing),
`_raster.py` (GeoTIFF read/write, `/vsicurl` handling, metadata stamping such as
`MD_BACKEND` and `MD_UNIT`), `_features.py` (vector geometry helpers),
`_sklearn.py` (the opt-in classifier/estimator factory), `_stac_fetch.py` (the
concrete network fetch, including Planetary Computer signing detection) and
`_styling.py` (auto-applied layer styling).

Every heavy import in this layer happens **inside** `processAlgorithm`, never at
module import. GDAL, scikit-learn and the STAC network stack are all pulled in only
when the relevant algorithm actually executes, so `import
eo_inference_toolbox.algorithms` stays cheap and cannot fail because a library is
missing. When a needed tier is absent, the algorithm raises a clear
`QgsProcessingException` telling the user what to install rather than crashing on an
`ImportError`.

## Layer 4 — provider, plugin and GUI

`provider.py` defines `EoInferenceProvider` (id `eoinfer`) whose `loadAlgorithms`
calls `iter_algorithms()` to register the seven algorithms in workflow order.
`plugin.py` is the QGIS lifecycle object returned by `classFactory`: `initGui`
registers the provider and adds the toolbar action / menu entry, and `unload`
removes every GUI element and the provider again.

`gui/dialog.py` is the tabbed wizard. It imports only `qgis.core`, Qt widgets and
the torch-free backend *facts* from `eocore.backends`. The Backends tab shows a card
per backend with live PyTorch / package status badges and an Install button; the
install runs `pip install` on a `QgsTask` worker thread so the UI never blocks and
QGIS objects are never touched off the main thread. The Workflow tab launches each
algorithm's own dialog via `processing.execAlgorithmDialog`.

## Cross-cutting invariants

Three invariants recur throughout and are worth stating on their own.

First, **L2-normalisation is a correctness prerequisite, not an option.** Because no
backend emits unit vectors, the embed algorithm normalises every embedding raster it
writes; the similarity and change algorithms then rely on that to treat cosine
similarity as a plain dot product. Skipping normalisation would silently corrupt
every downstream comparison.

Second, **backend identity is stamped and checked.** The embed step records which
backend produced a raster (`MD_BACKEND`), and a trained model records the backend and
embedding band count it was fitted on. Reusing a model against a raster from a
different backend, or with a different band count, is rejected rather than producing
meaningless predictions.

Third, **heavy tiers are opt-in, detected and self-installing.** PyTorch, the model
packages and scikit-learn are never hard dependencies and never bundled in the ZIP.
They are detected at run time; when missing, the GUI offers a guided install and the
algorithms fail with an actionable message.

## Testing strategy

The pure core is covered by an ordinary NumPy-only `pytest` suite that runs anywhere.
The GDAL and QGIS bridges are exercised by two gated test files
(`test_gdal_gated.py`, `test_qgis_gated.py`) that call `pytest.importorskip` and so
**skip cleanly** when those libraries are absent (the lint sandbox) and **run** when
they are present (the GDAL/QGIS container job, or inside QGIS itself). The torch
backend code and the GDAL/QGIS modules are additionally `py_compile`-checked so a
syntax or obvious typing error is caught even where the libraries cannot be imported.
See [SMOKE_TEST.md](SMOKE_TEST.md) for the manual in-QGIS verification that closes the
loop the automated suite cannot reach.
