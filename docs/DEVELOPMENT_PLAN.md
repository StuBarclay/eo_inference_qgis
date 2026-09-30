# EO Inference Toolbox — Development Plan

**Status:** working document, drafted 2026-09-29 against the v0.1.0 source tree.
Read it alongside [ROADMAP.md](ROADMAP.md) (the product-direction sketch this
document expands and grounds), [LIMITATIONS.md](LIMITATIONS.md) (the honest inventory
of current gaps) and [METHODOLOGY.md](METHODOLOGY.md).

> **Implementation status (2026-09-29, working tree → 0.2.0).** The whole code-side
> of both backlogs below has been implemented in the working tree, each as an
> opt-in default-off change with a guard or unit test: the hardening items **H-4,
> H-6, H-7, H-8, H-9, H-10, H-11, H-12, H-13, H-14**, and the engineering items
> **E-1 through E-13** (H-5 was the diagnosis behind E-1). A subsequent deep review
> added one HIGH + eight MEDIUM + seven LOW fixes — opt-in low-memory streaming for
> the analytics algorithms, a nearest-neighbour Change align fix, fail-closed model
> loading with a documented pickle trust boundary, GUI robustness, and more tests
> (see [RELEASE_NOTES.md](RELEASE_NOTES.md)). A final follow-up pass then delivered the
> three near-term roadmap items: automatic low-memory selection by a pixel × band
> memory budget (on by default only where streaming is byte-identical), safer model
> serialization (a JSON-metadata zip container storing the estimator via `skops`
> through a trusted-types gate when installed, pickle otherwise), and byte-identical
> parity tests for the streamed *Classify*, *Cluster* and *RGB* apply paths. The
> pure-core and orchestration work is verified in the sandbox — four gates green,
> 315 tests — while the PyTorch/QGIS paths are lint/type-checked and await the manual
> [SMOKE_TEST.md](SMOKE_TEST.md).
> **H-1** (a real contact email) and **H-2** (the stray root `main.py`) are now
> resolved; the release-blockers still open on the user's machine (the sandbox cannot
> do them) are **H-3** (commit and push) and signing off the manual smoke test in a
> real QGIS, after which the `experimental` flag can clear. The item text below is
> preserved as the original design record.

This plan has three parts, in the order they should be worked:

1. **Hardening & code-review backlog** — bugs, correctness risks, tech debt and
   test/CI gaps found by reading the v0.1.0 code. These come first because they are
   cheap, they de-risk everything downstream, and a few are release-blockers.
2. **Buildable engineering backlog** — the concrete next features, each with a design
   sketch, effort estimate, risk, and acceptance criteria you can check off.
3. **Capability / product roadmap** — the higher-level direction (phased), mapping the
   features above onto releases and naming the larger product bets.

Every item is gated by the three **non-negotiable design constraints** restated at the
end: zero-third-party install/load, heavy tiers opt-in and self-installing, pure core
testable in a bare sandbox. Any item that would break one of these needs a different
design, not an exception.

## How to read the item fields

- **Effort** — rough size, not a commitment: **S** ≈ up to a day, **M** ≈ 2–4 days,
  **L** ≈ a week or more of focused work.
- **Risk** — likelihood the change introduces a regression or turns out harder than it
  looks: **Low / Med / High**.
- **Acceptance** — the observable conditions that mean the item is done. Where the pure
  core is involved these should land as sandbox-runnable unit tests; where torch/QGIS is
  involved, as a step added to [SMOKE_TEST.md](SMOKE_TEST.md).

---

## Part 1 — Hardening & code-review backlog

These are findings from reading the v0.1.0 tree, ordered release-blockers first, then
correctness/robustness, then tech debt, then test/CI. File and line references are to the
state at drafting; re-check before you touch them.

### Release-blockers (before any public / plugins.qgis.org submission)

**H-1 — Placeholder author email in `metadata.txt`.**
`email=stuart.barclay@example.com` is a placeholder. The QGIS plugin repository requires a
real, monitored contact address and will surface it publicly. Replace with the address you
want associated with the listing.
*Effort S · Risk Low.* Acceptance: `metadata.txt` carries a real email; ZIP rebuilt.

**H-2 — Stray root `main.py`.**
A 544-byte PyCharm run-configuration stub sits at the repo root. It is correctly excluded
from the plugin ZIP (the builder only walks `eo_inference_toolbox/`), so it is not a
shipping bug, but it is confusing in the repo and reads like an entry point that is not
one. Remove it (this has to be done on Windows — the sandbox cannot unlink files on the
OneDrive tree).
*Effort S · Risk Low.* Acceptance: file removed; `git status` clean; ZIP unaffected.

**H-3 — Commit the GDAL-lazy-import fix.**
The fix that keeps `osgeo` imports at call time (so the package imports in a bare
environment) is present in the working tree but, per the release notes, not yet committed.
Until it is committed and pushed it is not really shipped and can be lost.
*Effort S · Risk Low.* Acceptance: fix committed and pushed; a fresh clone imports
`eo_inference_toolbox` with no GDAL present.

### Correctness & robustness

**H-4 — `torch.set_grad_enabled(False)` is a process-wide side effect.**
All three backends call `torch.set_grad_enabled(False)` at model-load time
(`clay.py:139`, `prithvi.py:130`, `satlas.py:120`) *and* already wrap the actual forward
pass in a local `with torch.no_grad():` (`clay.py:148`, `prithvi.py:139`, `satlas.py:128`).
The global call is therefore redundant for our own inference, but it is not harmless: it
disables autograd for the entire QGIS Python process, which can silently break any other
torch-using plugin the user has loaded. Drop the global call and rely on the local
`no_grad()` context.
*Effort S · Risk Low.* Acceptance: no `set_grad_enabled` calls remain; smoke test still
produces identical embeddings; a note in the PR explains why the local context is
sufficient.

**H-5 — Whole-raster read into memory in `embed.py`.**
`embed.py:151` does `raster = _raster.read_raster(path)` and then tiles the already-loaded
`raster.cube` (`:171`, `:244`). The tiling and overlap-averaged mosaic machinery is sound,
but because the whole cube is materialised up front, peak memory scales with the *input*
raster, not the tile. For large AOIs this is the practical ceiling on what the plugin can
embed. This is the single most impactful robustness fix and is the enabling work for E-1
(windowed reads) and E-2 (batching). Recorded here as the *diagnosis*; the *fix* is E-1.
*Effort — see E-1 · Risk Med.*

**H-6 — Backend model-API access is flagged "VERIFY during smoke test" and untested.**
The Clay loader reaches into an uncertain external API surface
(`getattr(getattr(model, "model", model), "encoder", None)` and the datacube construction
in `_build_datacube`), and the code itself carries a "VERIFY during smoke test" note. The
Prithvi and Satlas loaders have the analogous exposure. None of these paths are exercised
by automated tests (by design — they need weights and torch). The risk is that an upstream
checkpoint or library version bump silently changes the attribute path and the failure only
shows up at a user's first embed run. Mitigations: (a) pin known-good versions in the
install hints, (b) add a defensive, explicit error message when the expected attribute is
absent (fail loud with a "this checkpoint/library version is not supported" message rather
than an opaque `AttributeError`), and (c) capture the verified version in
[SMOKE_TEST.md](SMOKE_TEST.md).
*Effort M · Risk High (external dependency).* Acceptance: each backend raises a clear,
actionable error if its expected model structure is missing; smoke test records the exact
library + checkpoint versions verified.

**H-7 — Hardcoded `reflectance_scale` assumption (Clay).**
The Clay datacube builder multiplies pixels by a fixed reflectance scale that assumes
Sentinel-2 L2A DN (÷10000). If a user feeds already-reflectance or 8-bit imagery the input
is silently mis-scaled and the embeddings are quietly wrong — exactly the "artefact of the
inputs" failure LIMITATIONS.md warns about, but with no signal to the analyst. Surface the
scale as a parameter (default preserves current behaviour) and/or auto-detect obviously
non-DN ranges and warn.
*Effort M · Risk Med.* Acceptance: scale is user-visible with a documented default; a
value out of the plausible DN range emits a feedback warning; unit test covers the scaling
math.
*Update (0.2.0):* the input-scale parameter and the mis-scale warning shipped as
described. A follow-up went further for Clay specifically: rather than a fixed ÷10000,
Clay now standardises each band to `(DN − mean) / std` from its vendored per-sensor
`metadata.yaml`, matching Clay's own embedding pipeline. The scale parameter is retained
as an optional pre-scale into DN units (default 1.0 = raw DN); unmapped custom bands pass
through unstandardised. Torch-free tests cover the stats, band-name mapping and maths.

**H-8 — No weight-file integrity check.**
`_weights.py` streams downloads to a `.part` file and atomically replaces on success, which
is good, but there is no checksum verification and no size/progress feedback. A truncated or
corrupted-but-complete download, or a changed upstream artefact, is accepted silently. Add
expected-hash verification (see E-5).
*Effort — see E-5 · Risk Med.*

**H-9 — Packaging metadata drift.**
`pyproject.toml` declares `stac = ["requests"]`, but there is no `import requests` anywhere
in the package — STAC search uses stdlib `urllib` (`eocore/stac.py`), and `pystac` /
`pystac_client` are only in the mypy override list, not used in code. Conversely,
`planetary_computer` *is* used (HLS href signing in `_stac_fetch.py`, install button in
`dialog.py`) but is declared in **no** optional-dependency tier. Net effect: the extras
advertise a dependency we don't use and omit one we do. Fix: drop `requests` (or replace
the extra's contents with what STAC signing actually needs), and add a tier that includes
`planetary-computer`.
*Effort S · Risk Low.* Acceptance: every name in `[project.optional-dependencies]` is
actually imported by the code path it claims to enable; the GUI install button and the
declared extra agree.

### Tech debt (DRY / consolidation)

**H-10 — Dependency-availability detection is duplicated four ways.**
`importlib.util.find_spec(...)`-based availability checks exist independently in
`eocore/backends/_deps.py:23`, `algorithms/_sklearn.py:24`, `algorithms/_stac_fetch.py:18`
and `gui/dialog.py:63`. They agree today but will drift. See E-9 for the consolidation.

**H-11 — pip-install logic is duplicated three ways.**
Opt-in install implementations live in `_deps.py`, `_sklearn.py` and (as the `QgsTask`
worker) `gui/dialog.py`, with the backends carrying their own install hints. One canonical
installer with a single timeout/policy, wrapped by the GUI task, would remove the drift risk
and make the CPU/CUDA torch selection (E-11) a one-place change. See E-9.

### Test & CI gaps

**H-12 — No end-to-end embed test with a fake backend.**
The tiling, mosaic and ML helpers are unit-tested, and the QGIS/GDAL bridges have gated
tests, but the *orchestration* in the embed algorithm (tile → embed → overlap-average →
L2-normalise → nodata → stamp descriptor) is only exercised by the manual smoke test,
because the real backends need torch and weights. A dummy `EmbeddingBackend` that returns a
deterministic array would let the whole orchestration path run in the bare sandbox. See E-13.

**H-13 — No GDAL/QGIS container CI job.**
The gated tests only run where GDAL/QGIS happen to exist, i.e. locally. A container job in CI
would run them on every push. This is the ROADMAP "near term" item; captured here as the
test-infrastructure gap it closes. See the roadmap Part 3.

**H-14 — Backend Protocol conformance not asserted.**
Nothing tests that each registered backend actually satisfies the `EmbeddingBackend`
Protocol and that its `BackendInfo` is internally consistent (band count, dimension, etc.).
A tiny registry-walk test would catch a malformed new backend at CI time rather than at load
time. See E-13.

---

## Part 2 — Buildable engineering backlog

The concrete next features. Each is self-contained; the suggested order is in Part 3. All of
these preserve the design constraints — none add a hard dependency or move heavy work into
import time.

### E-1 — Windowed / streaming raster reads in `embed`

**Problem.** Peak memory scales with the whole input raster (H-5), so large AOIs fail or
thrash.

**Design.** Read each tile window directly from GDAL at the point of use inside the tiling
loop, instead of reading the full cube once and slicing it. The `iter_windows` generator
already yields the window geometry; give `_raster` a `read_window(path, window)` that issues
a windowed `ReadAsArray`, and have the embed loop pull one window at a time. The
`MosaicAccumulator` already streams output, so only the read side changes. Keep the current
full-read path as a fallback for tiny rasters if it simplifies the code, but the default
should be windowed.

**Effort M · Risk Med** (GDAL windowed reads interact with block sizes and overviews; verify
on a real multi-band raster).

**Acceptance.**
- Embedding a raster larger than available RAM completes without loading the full cube.
- A unit test on a fake reader asserts that only window-sized arrays are requested.
- Smoke test embeds a large AOI and peak memory tracks tile size, not image size.

### E-2 — Batched tile inference

**Problem.** `backend.embed_tile` runs one tile per forward pass (batch dim 1). On GPU this
leaves most of the device idle; even on CPU there is per-call overhead.

**Design.** Add an optional `embed_tiles(batch, metas) -> list[array]` to the backend
Protocol with a default implementation that loops `embed_tile` (so existing backends keep
working). Where a backend can stack tiles into one tensor, override it. The embed algorithm
collects up to *N* windows (a "batch size" parameter, default small and CPU-safe), runs one
forward pass, then scatters results into the accumulator. Pairs naturally with E-1: the
windowed reader fills the batch buffer.

**Effort L · Risk Med.** Depends on E-1 landing first.

**Acceptance.**
- Backend Protocol gains `embed_tiles` with a default loop; existing per-tile backends
  unchanged in output.
- On a CUDA machine, throughput on a large AOI improves measurably vs. v0.1.0 (record the
  before/after in the smoke test).
- Batch size is a user-visible parameter with a conservative default.

### E-3 — Bilinear token-grid upsampling (option)

**Problem.** `_pool.upsample_nearest` maps the ViT patch grid to pixels with
nearest-neighbour, giving visibly blocky embeddings at patch boundaries.

**Design.** Add a bilinear upsample alongside the nearest one and expose an "upsampling"
choice (default nearest, to preserve current output and stay dependency-free — implement
bilinear in NumPy so the pure core stays torch-free). Document the trade-off: bilinear is
smoother but blends across patch edges, which can be undesirable for hard-boundary analyses.

**Effort S · Risk Low.**

**Acceptance.**
- NumPy bilinear upsampler with a unit test against a known small case.
- Algorithm parameter selects the method; nearest remains the default.

### E-4 — Overlap feathering in the mosaic

**Problem.** `MosaicAccumulator` uses a uniform weight of 1.0 per tile, so overlapping tiles
are averaged with a hard equal weight. With patch-quantised embeddings this can leave faint
seams at tile boundaries.

**Design.** Add an optional per-pixel weight window (cosine / linear taper toward tile edges)
so overlapping contributions blend smoothly. Keep uniform weighting as the default until the
seam improvement is confirmed. This is a pure-core change in `eocore/tiling.py`, fully
unit-testable.

**Effort M · Risk Low.**

**Acceptance.**
- Taper weighting available and unit-tested (a synthetic two-tile overlap shows no seam vs. a
  visible step under uniform weighting).
- Default behaviour unchanged unless the option is set.

### E-5 — Weight download: progress, size, and checksum

**Problem.** No integrity check (H-8), and the first-run download cost is invisible until it
happens (LIMITATIONS.md).

**Design.** Extend `_weights.py` to (a) verify an expected SHA-256 recorded per checkpoint in
the backend registry, failing loud on mismatch; (b) read `Content-Length` and report progress
through the QGIS feedback object so the progress bar moves during download; and (c) expose a
"pre-download weights" button per backend in the wizard (`gui/dialog.py`) plus a small cache
inspector (what is cached, how large). Downloads stay stdlib `urllib` for the default path and
`huggingface_hub` where already used.

**Effort M · Risk Low.**

**Acceptance.**
- A corrupted/altered weight file is rejected with a clear message.
- Download progress is visible in the algorithm feedback and the wizard.
- The wizard can pre-download a backend's weights and show cache size; smoke-test step added.

### E-6 — Accuracy reporting for classification

**Problem.** `classify.py` trains and predicts but never splits train/validation or reports
accuracy; map quality can only be eyeballed. `eocore/ml.py` has no metrics helpers.

**Design.** Add an opt-in (default off) holdout split and a metrics report: overall accuracy,
per-class precision/recall/F1, and a confusion matrix, written as a sidecar (JSON, plus an
optional HTML). Put the pure math (split, confusion matrix, per-class metrics from
predictions) in `eocore/ml.py` so it is sandbox-testable without sklearn; the algorithm layer
just wires it to the trained estimator. Mirrors the accuracy-report pattern already used in the
AlphaEarth toolbox, for consistency across the suite.

**Effort M · Risk Low.**

**Acceptance.**
- Opt-in split + metrics; default off preserves current behaviour.
- Confusion matrix and per-class metrics computed by pure-NumPy helpers with unit tests.
- Sidecar report written next to the classification output when enabled.

### E-7 — Clustering: auto-k and full-data option

**Problem.** `cluster.py` fits k-means at a fixed user-supplied k on a deterministic subsample,
with no guidance on choosing k.

**Design.** Add (a) an opt-in silhouette-based k selection over a small candidate range
(default off; report the chosen k and scores), and (b) an option to use `MiniBatchKMeans` for a
full-data fit on large rasters instead of subsampling. Keep the current fixed-k-on-subsample as
the default. The selection loop and scoring live behind the existing `_sklearn` guard.

**Effort M · Risk Low.**

**Acceptance.**
- Auto-k available, reports chosen k + per-k score; default off.
- MiniBatchKMeans path selectable for large inputs.
- Existing fixed-k behaviour unchanged by default.

### E-8 — STAC multi-scene mosaic / composite

**Problem.** STAC fetch takes the single least-cloudy scene; AOIs straddling scene boundaries
or lacking a clean acquisition can't be served (LIMITATIONS.md).

**Design.** Add an opt-in multi-scene mode: gather the top-*k* least-cloudy scenes in the
window and build a per-pixel median (or first-valid) composite over the AOI grid. Keep the
pure compositing math in `eocore/` (injectable fetch, testable without network); the algorithm
layer handles asset signing and reads. Single-scene remains the default for reproducibility.

**Effort L · Risk Med** (reprojection/alignment across scenes; cloud-mask handling).

**Acceptance.**
- Opt-in composite mode; single-scene default unchanged.
- Compositing math unit-tested with synthetic multi-scene stacks.
- Smoke test embeds an AOI that straddles two scenes.

### E-9 — Consolidate dependency detection & install

**Problem.** Availability checks (H-10) and pip-install logic (H-11) are duplicated across four
and three modules respectively.

**Design.** Make `eocore/backends/_deps.py` the single source of truth for `module_available`
and a canonical `pip_install` (policy + timeout in one place). Have `_sklearn.py`,
`_stac_fetch.py` and `gui/dialog.py` call it; the GUI keeps only the `QgsTask` wrapper that
runs the canonical installer off-thread. This also makes E-11 a one-line change.

**Effort M · Risk Med** (touches the GUI install path — verify the background task still
reports status correctly).

**Acceptance.**
- Exactly one `module_available` and one `pip_install` implementation; others delegate.
- No behaviour change to the wizard install buttons (smoke-test the sklearn install).

### E-10 — Formalise the pluggable-backend extension point

**Problem.** Adding a backend today means editing both the `BACKENDS` dict and the
`load_backend` if/elif chain. The Protocol exists but the wiring is not truly "drop-in".

**Design.** Replace the if/elif with a registration mechanism: backends register themselves
(decorator or explicit `register_backend(id, loader)` call) into a table that `load_backend`
consults, so a new backend is one new module plus one registration, with no edits to the
dispatch. Document the extension point (a short "adding a backend" guide). Keep the registry
torch-free and lazy (loaders imported only when their backend is chosen).

**Effort M · Risk Med.**

**Acceptance.**
- A new backend can be added without editing `load_backend`.
- Registry stays torch-free; conformance test (E-13) covers all registered backends.
- "Adding a backend" documented.

### E-11 — CPU-vs-CUDA torch install selection

**Problem.** The wizard installs torch with a plain `pip install torch`, which does not pick a
CPU-only or CUDA wheel; users on machines without CUDA can get an unnecessarily large/wrong
build, and CUDA users may not get GPU support.

**Design.** Once E-9 centralises install, detect the platform/GPU situation and offer the
appropriate `--index-url` (CPU vs. a CUDA build), presented as a choice in the wizard with a
sensible default. Never silently pick a multi-gigabyte CUDA wheel.

**Effort M · Risk Med** (index URLs change upstream; make them configurable, not hardcoded
forever).

**Acceptance.**
- Wizard offers CPU / CUDA torch install with a documented default.
- Index URLs are in one place and easy to update.

### E-12 — Optional resample-to-match in `change`

**Problem.** The change modes require inputs to share a pixel grid and CRS; misaligned inputs
are rejected rather than aligned (`change.py` docstring; LIMITATIONS.md).

**Design.** Add an opt-in "resample second+ rasters to the reference grid" step (nearest for
embeddings, to avoid blending distinct classes) before computing distances, with a loud note
that resampling embeddings is itself approximate. Default stays strict (reject misaligned).

**Effort M · Risk Med.**

**Acceptance.**
- Opt-in align step; strict default unchanged.
- Documented caveat that resampling embeddings is approximate.

### E-13 — Test scaffolding: fake backend + Protocol conformance

**Problem.** Orchestration and backend conformance are untested in the bare sandbox
(H-12, H-14).

**Design.** Add a deterministic dummy `EmbeddingBackend` to the test suite and use it to
exercise the embed orchestration (tile → embed → mosaic → normalise → stamp) without torch.
Add a registry-walk test asserting every registered backend satisfies the Protocol and has a
self-consistent `BackendInfo`. Both run in CI with no heavy deps.

**Effort M · Risk Low.**

**Acceptance.**
- Embed orchestration covered by a sandbox test using the dummy backend.
- A test fails if any registered backend violates the Protocol or has inconsistent metadata.

---

## Part 3 — Capability / product roadmap

This is the higher-level direction, phased and mapped to the items above. It refines
[ROADMAP.md](ROADMAP.md) rather than replacing it.

### Near term — "make v0.1 solid" (a v0.2 hardening release)

The theme is correctness, honesty and CI, not new capability. Ship the release-blockers
(H-1–H-3), the process-hygiene and metadata fixes (H-4, H-9), and stand up the test/CI
scaffolding: the GDAL/QGIS container job (H-13) and the fake-backend orchestration +
conformance tests (E-13). Fold in weight integrity + progress (E-5) since it is low-risk and
directly improves the first-run experience LIMITATIONS.md calls out. Outcome: the same
capability as v0.1.0, but self-verifying in CI, honest about its packaging, and safe to submit
to plugins.qgis.org.

### Medium term — "make it scale and report" (a v0.3 capability release)

The theme is large-AOI practicality and defensible outputs. Land windowed reads (E-1) and
batched inference (E-2) together — this is the biggest usability unlock, turning "works on a
demo tile" into "works on a real study area". Add accuracy reporting (E-6) so classifications
can be quoted, not eyeballed, and clustering auto-k (E-7) so non-experts get a reasonable k
without guessing. Improve embedding quality at the edges with bilinear upsampling (E-3) and
mosaic feathering (E-4). Consolidate the install path (E-9) and add CPU/CUDA torch selection
(E-11) as a rider on it. This is also where **task-specific profiles** (packaged WUI / built-up
/ canopy presets that pre-select backend, bands and defaults) become worthwhile, because by now
the underlying knobs are stable — profiles are a thin, high-value layer on top and match the
"non-expert gets a good result" goal in ROADMAP.md.

### Longer term / exploratory — "make it extensible and interactive"

Formalise the pluggable-backend extension point (E-10) and document "adding a backend", then
use it to add the first genuinely new backend beyond the current three. Build multi-scene STAC
compositing (E-8) so imagery acquisition stops being the weak link for awkward AOIs. Turn the
existing confidence raster into an **active-learning loop** (surface most-uncertain pixels →
analyst labels a few → retrain), which depends on accuracy reporting (E-6) being in place. Add
**hyperparameter tuning** for the classifier (opt-in, GUI-guided), mirroring the tuning parity
recently added to the AlphaEarth toolbox so the two plugins behave consistently. Optional
resample-to-match in change detection (E-12) rounds out robustness.

### Suggested release themes at a glance

- **v0.2 — Hardening & CI:** H-1..H-4, H-9, H-13, E-5, E-13.
- **v0.3 — Scale & report:** E-1, E-2, E-6, E-7, E-3, E-4, E-9, E-11; task profiles.
- **v1.0 — Extensible & interactive:** E-10 + a new backend, E-8, active learning, tuning,
  E-12; drop the `experimental` flag once the smoke test is routinely green in a container.

---

## Design constraints to preserve (non-negotiable)

Repeated because every item above is gated by them:

1. **Zero third-party dependencies to install or load.** The plugin must install and import
   with nothing but the standard library + what QGIS already ships. Heavy imports stay at call
   time, never at module import.
2. **Every heavy tier stays opt-in, detected, and self-installing — never bundled.** torch,
   scikit-learn, planetary-computer, huggingface_hub are discovered at run time and installed on
   request, not vendored and not hard dependencies.
3. **The pure core (`eocore/`) stays testable in a bare sandbox** with no GDAL, QGIS or torch.
   New logic belongs in the core (pure NumPy, injectable side effects) wherever possible, with
   the QGIS/GDAL layer kept thin.

A feature that appears to require breaking one of these needs a redesign, not an exception —
these properties are what make the plugin installable and maintainable in the first place.

---

## Provenance of the code-review findings

Part 1 was written from a direct read of the v0.1.0 tree and the specific claims were
re-verified against source before this document was finalised: no `import requests` in the
package (H-9); `planetary_computer` used but declared in no extra (H-9); `find_spec`
availability checks duplicated in `_deps.py`, `_sklearn.py`, `_stac_fetch.py`, `dialog.py`
(H-10); `torch.set_grad_enabled(False)` in all three backends alongside a local `no_grad()`
(H-4); full-raster read at `embed.py:151` before tiling (H-5); no checksum/progress logic in
`_weights.py` (H-8); placeholder email and `experimental=True` in `metadata.txt` (H-1); stray
root `main.py` present (H-2). Re-verify line numbers before acting — the tree will have moved.
