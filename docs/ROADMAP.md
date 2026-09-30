# Roadmap

Candidate future work, roughly in priority order. Nothing here is committed; it
records the intended direction so contributors and users know what is being
considered and why.

## Delivered in 0.2.0

Much of what this roadmap first sketched has since landed as opt-in, default-off
capability (see [RELEASE_NOTES.md](RELEASE_NOTES.md) and
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)): the GDAL/QGIS container CI job;
weight-management polish (pre-download, SHA-256 verify, cache inspector); optional
multi-scene STAC compositing; the formalised, documented pluggable-backend registry
([ADDING_A_BACKEND.md](ADDING_A_BACKEND.md)); classification accuracy reporting; and
GPU/batching ergonomics (windowed streaming reads, batched inference, CPU-vs-CUDA
torch install). A subsequent deep-review pass then added an opt-in low-memory
streaming mode across the analytics algorithms (similarity, change, classify,
cluster and RGB) so large high-dimensional embeddings are no longer bounded by RAM,
plus fail-closed model loading and GUI-robustness fixes. A final follow-up pass then
delivered the three near-term items that review had itself recorded: **automatic**
low-memory selection driven by a pixel × band memory budget (so the streamed path
engages without the user ticking a box, on by default only where it is byte-identical);
safer model serialization (a JSON-metadata zip container that stores the estimator via
`skops` through a trusted-types gate when it is installed, pickle otherwise); and
byte-identical parity tests for the streamed *Classify*, *Cluster* and *RGB* apply
paths. The items below are what remains.

## Near term

**Close the last of the manual-verification loop in CI.** The pure core, the
orchestration path (via the fake backend) and the gated GDAL/QGIS tests now run in
CI. The remaining gap is a small cached-weights fixture so at least one tiny
*real-model* end-to-end embed can run automatically rather than only in the manual
smoke test.

**Further serialization hardening.** The 2026-09-29 follow-up made a saved model a
JSON-metadata zip container and added optional `skops` trusted-types loading; the
residual gap is the pickle fallback used when `skops` is absent. Persisting the
parameters we fully control (cluster centroids, class labels) as JSON + `.npz` would
let more model kinds avoid pickle entirely, and auto-selecting the low-memory path
could grow a byte-budget estimate that also accounts for the transient prediction
buffers, not just the input cube.

## Medium term

**More inference backends.** Adding a backend is now a documented drop-in (one module
plus one `register_backend` call). The open work is shipping additional backends
(e.g. Clay variants, other open geospatial foundation models). Any archive-style
backend (as opposed to inference over supplied imagery) would still need a parallel
code path, since the current design assumes inference.

**Active-learning loop.** A guided cycle where the classifier surfaces its most
uncertain pixels, the analyst labels a few, and the model retrains — turning the
existing confidence raster and the new accuracy report into an interactive sampling
aid rather than static outputs.

**Task-specific profiles.** Packaged presets for common jobs (e.g. wildland-urban
interface delineation, built-up mapping, canopy/vegetation) that pre-select a
backend, bands, and sensible analysis defaults, so a non-expert can get a good result
without tuning every parameter.

## Longer term / exploratory

**Cloud-aware compositing.** The 0.2.0 composite is median / first-valid over the
least-cloudy scenes; a genuine per-pixel cloud-mask gap-fill would let more difficult
windows be embedded without hand-prepared imagery.

**Runtime estimation.** Clear feedback on expected runtime and download size before a
large-AOI job starts, building on the batching and weight-size work already in place.

## Design constraints to preserve

Whatever gets built, three properties are non-negotiable because they are what make
the plugin usable: it must still **install and load with zero third-party
dependencies**; every heavy tier must stay **opt-in, detected and self-installing,
never bundled**; and the **pure core must stay testable in a bare sandbox** with no
GDAL, QGIS or torch. New features that would break any of these need a different
design, not an exception.
