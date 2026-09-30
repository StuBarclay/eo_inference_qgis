# Release notes

## 0.2.0 — 2026-09-30 (working tree)

A hardening-and-capability release that works through the whole
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) backlog — the code-review findings
(H-4…H-14) and the buildable feature list (E-1…E-13). Every new capability is
**opt-in and default-off**, so the default behaviour is byte-for-byte 0.1.0; each
is backed by a guard or unit test. The pure-core and orchestration work is
verified in the sandbox; the PyTorch/QGIS paths are written and lint/type-checked
but await sign-off in the manual [SMOKE_TEST.md](SMOKE_TEST.md).

**Release-blockers still open** before any public / plugins.qgis.org submission:
committing and pushing the working tree (H-3) and signing off the manual
[SMOKE_TEST.md](SMOKE_TEST.md) in a real QGIS (a 3.34 LTR / PyQt5 build and a
4.x / PyQt6 build), after which the `experimental` flag can be cleared. The contact
email (H-1) and the stray root `main.py` (H-2) are now resolved. Commit/push is
handled on the user's machine, not in the sandbox.

### Larger AOIs and throughput

- **Windowed / streaming reads (E-1, H-5).** *Embed* now reads one tile window at
  a time straight off disk instead of materialising the whole cube, so peak
  memory tracks the tile size, not the scene — the practical ceiling on AOI size
  is lifted. A new pure `runner` module owns the streaming loop and is fully
  unit-tested with injected fakes.
- **Batched tile inference (E-2).** An optional `embed_tiles` batch capability
  (`BatchEmbeddingBackend`) runs several equal-shaped tiles in one forward pass; a
  user-visible batch-size parameter (default 1) keeps CPU behaviour unchanged.
  `embed_tile_batch` transparently falls back to a per-tile loop, so batching only
  ever changes throughput, never results.
- **CPU-vs-CUDA torch install (E-11).** The wizard offers a CPU-only or CUDA torch
  wheel via the appropriate index URL, with a documented default, instead of a
  blind `pip install torch`.

### Mosaic and upsampling quality

- **Bilinear token-grid upsampling (E-3).** A NumPy bilinear upsampler is
  selectable alongside the default nearest-neighbour, for smoother embeddings at
  the cost of blending across patch edges.
- **Overlap feathering (E-4).** An optional cosine taper feathers tile seams in
  the mosaic; uniform weighting stays the default.

### Analytics

- **Classification accuracy report (E-6).** An opt-in holdout split writes overall
  accuracy, per-class precision/recall/F1 and a confusion matrix as a JSON sidecar;
  the metrics maths lives in `eocore.ml` and is unit-tested without scikit-learn.
- **Clustering auto-k and full-data fit (E-7).** An opt-in silhouette sweep reports
  a chosen k over a candidate range, and a MiniBatchKMeans path fits large rasters
  without subsampling. Fixed-k-on-subsample stays the default.
- **STAC multi-scene composite (E-8).** An opt-in mode builds a per-pixel median /
  first-valid composite over the top-k least-cloudy scenes, for AOIs that straddle
  scene boundaries; single-scene stays the default. Compositing maths is
  network-free and unit-tested.
- **Resample-to-match in *Change* (E-12).** An opt-in nearest-neighbour align step
  brings a second raster onto the reference grid; the strict "reject misaligned"
  behaviour remains the default, with a documented caveat that resampling
  embeddings is approximate.

### Weights, robustness and correctness

- **Weight integrity + progress + cache (E-5, H-8).** Downloads verify an expected
  SHA-256 when one is recorded (failing loud on mismatch), report progress through
  the feedback object, and the wizard gains a *Download / verify weights* button
  plus a cache-size inspector.
- **Defensive model-API guards (H-6).** The uncertain external-model seams (Clay's
  encoder path, Prithvi's TerraTorch build, Satlas's pretrained-model entry point,
  and each forward-pass output) now fail with a single actionable "pin a known-good
  version" error instead of an opaque `AttributeError`. The torch-free guard
  helpers and Clay's encoder seam are unit-tested with fake models.
- **No process-wide autograd toggle (H-4).** The redundant, process-global
  `torch.set_grad_enabled(False)` is gone; inference still runs under a local
  `torch.no_grad()`, so a co-installed torch plugin is no longer affected.
- **Pluggable backend registry (E-10).** Backends now register a `BackendInfo`
  plus a lazy loader with one `register_backend` call; `load_backend` consults that
  table, so adding a backend needs no edit to the dispatch. The registry stays
  torch-free. See the new [ADDING_A_BACKEND.md](ADDING_A_BACKEND.md).
- **Consolidated dependency handling (E-9).** One `module_available` and one
  `pip_install` in `_deps.py`; the sklearn/STAC/GUI paths delegate to it.
- **Reflectance-scale parameter (H-7).** The embed input-scale override is exposed
  and a scale-sanity check warns when imagery looks mis-scaled for the backend.
- **Clay status-badge fix.** The Backends tab no longer reports Clay as
  `package: missing` after a correct install. Its import-name list required a
  non-existent top-level `clay` module in addition to the real `claymodel` package;
  because the badge (and `missing_import_names`) require *all* listed names to be
  importable, Clay stayed red even when installed. The list is now `("claymodel",)`,
  matching both the pip package and the runtime `from claymodel.module import …`.
  (The Embed step already worked regardless — it imports `claymodel` lazily and never
  consulted the badge.) A regression test pins each backend's import names.
- **Clay checkpoint path fix.** The Clay v1.5 weights live at `v1.5/clay-v1.5.ckpt`
  in the `made-with-clay/Clay` Hugging Face repo, not at the repo root. The code
  requested a bare `clay-v1.5.ckpt`, which returns 404 — so both the *Download /
  verify weights* action and the first Embed run (which uses the same constant to
  fetch weights lazily) would have failed. Fetching now uses the correct sub-path.
  The checkpoint is 5,158,764,629 bytes (~5.16 GB); its git-LFS SHA-256 was
  confirmed against Hugging Face's authoritative LFS pointer and is published in
  [SMOKE_TEST.md](SMOKE_TEST.md) for anyone who wants to verify a download by hand.
  It is deliberately **not** pinned in `weights_sha256`: the downloader re-hashes the
  file on every resolve (including cache hits), so pinning would re-read ~5 GB on each
  backend load for no added safety, since Hugging Face's LFS layer already
  content-addresses the download. A regression test pins the checkpoint path.
- **Clay embed metadata / masking fix.** The first *Embed* run with the Clay
  backend crashed after the weight download with
  `FileNotFoundError: 'configs/metadata.yaml'`. Clay's checkpoint saves a
  *relative* `metadata_path`, and Lightning re-opens it when re-instantiating the
  module — but the working directory inside QGIS is not the Clay source tree, so
  the open failed. The Clay v1.5 `metadata.yaml` is now vendored beside the
  backend and passed as an absolute path override. The same load now also pins
  `mask_ratio=0` and `shuffle=False` (the training checkpoint saved 0.75 / True),
  so the encoder returns every patch in spatial order instead of masking ~75% of
  them and scrambling the grid. A torch-free test asserts the vendored asset is
  present, absolute and parses with the sentinel-2-l2a band statistics.
- **Clay input standardisation.** Clay now feeds the encoder per-band
  `(DN − mean) / std` using the per-sensor statistics in the vendored
  `metadata.yaml`, exactly as Clay's own embedding pipeline does — replacing the
  earlier fixed ÷10000 reflectance scaling, which did not match the distribution
  the weights expect. The plugin's semantic band names are mapped to Clay's
  Sentinel-2 keys (`narrow_nir`→`nir08`, `swir1`→`swir16`, `swir2`→`swir22`); an
  unmapped custom band passes through unstandardised rather than erroring. The
  *Input scale* override is retained as an optional pre-scale into DN units
  (default 1.0, i.e. raw DN) for imagery in other units. Torch-free tests pin the
  per-band stats, the band-name mapping and the standardisation maths.
- **Clay wavelengths from metadata + patch-grid guard.** The per-band central
  wavelengths Clay consumes are now read from the vendored `metadata.yaml`
  (`wavelength` block, same semantic→S2 key mapping as the stats) instead of a
  separate hard-coded table — one source of truth, and a non-default band gets its
  true wavelength rather than a silent `0.0`. A new guard rejects an encoder token
  count that cannot correspond to the expected patch grid (e.g. a patch size that
  disagrees with the checkpoint) with an actionable error, instead of silently
  mis-slicing and scrambling the embedding grid. The Sentinel-2 L2A baseline-04.00
  `BOA_ADD_OFFSET` (−1000) is now documented as a known caveat in LIMITATIONS
  (matching Clay's raw-DN tutorial is deliberate; the plugin does not silently
  correct it). Torch-free tests pin the metadata-sourced wavelengths and the guard.
- **Satlas feature-map unpacking + batch-split guards.** The SatlasPretrain
  backend keeps **all** FPN levels: its forward output is normalised through a
  single `_as_feature_map_sequence` helper that treats any non-empty sequence as
  the full multi-scale set, instead of the previous `output[0] if tuple` handling
  that would have silently dropped every level but the finest had the library
  ever returned a tuple of maps (this backend never attaches a head, so the
  `(outputs, loss)` shape cannot occur). The batched path now guards that each
  feature map is batch-first with the expected batch size before splitting it per
  tile, so a map that is not batch-first fails loud rather than being mis-sliced
  along its channels. The default embedding-dim hint is corrected to 512 (four
  128-channel FPN levels concatenated) from a misleading 1024, and the Satlas
  input convention (true-colour 8-bit B4/B3/B2 ÷255; single-image RGB checkpoints)
  is documented in LIMITATIONS. Torch-free tests pin the default checkpoint id and
  RGB band order and exercise both guards.
- **Prithvi input standardisation + token/output guards.** Like Clay, the
  Prithvi-EO-2.0 backend now standardises each band as `(DN − mean) / std` using the
  per-band HLS statistics published in the model's `config.json`, exactly as
  Prithvi's own preprocessing does — replacing the earlier fixed ÷10000 reflectance
  scaling, which the pretrained backbone (it does not normalise internally) was never
  trained to see. Those statistics are HLS surface-reflectance values in raw
  digital-number units, keyed positionally to Prithvi's fixed six-band order
  (Blue, Green, Red, narrow-NIR, SWIR1, SWIR2); an unmapped band passes through
  unstandardised rather than erroring. The *Input scale* override is retained as an
  optional pre-scale into DN units (default 1.0 = raw HLS DN; set `1e4` for imagery
  already in reflectance `[0, 1]`). A new guard rejects a backbone token count — or a
  forward-pass output rank — that cannot correspond to the expected patch grid (e.g. a
  patch size that disagrees with the checkpoint) with an actionable error, instead of
  silently mis-slicing and scrambling the embedding grid. Prithvi's multi-temporal
  capability is not exposed: the plugin feeds a single acquisition (`time = 1`), and
  that (plus the fixed six-band 30 m HLS input) is documented in LIMITATIONS.
  Torch-free tests pin the model identifier, patch size, embedding width and band
  order, the published per-band stats and the standardisation maths (DN and
  pre-scaled reflectance), and the token-count guard's accept/reject behaviour.
- **STAC date-only datetime fix.** *Fetch imagery (STAC)* failed with
  `STAC search failed: HTTP Error 400: Bad Request` whenever a start/end date was
  given. Earth Search (and stac-fastapi generally) rejects a date-only `datetime`
  value — it must be RFC 3339, carrying a time and timezone. The query builder now
  promotes a bare `YYYY-MM-DD` start to `…T00:00:00Z` and an end to `…T23:59:59Z`
  (leaving already-timestamped and open-ended values untouched), so a whole-day
  range stays inclusive. A regression test pins the RFC 3339 output.

### Post-review hardening (second deep review, 2026-09-29)

A second pass over the whole plugin worked through one HIGH, eight MEDIUM and seven
LOW findings. As above, anything that changes behaviour is opt-in and default-off;
the pure-core additions are unit-tested.

- **Opt-in low-memory streaming for the analytics algorithms (HIGH).** *Similarity*,
  *Change*, *Classify*, *Cluster* and *Embedding → RGB* previously read the entire
  embedding cube into RAM — on a 768–1024-band embedding that is the practical ceiling
  on raster size. Each now offers a **Low-memory mode** checkbox (default off) that
  streams the raster in fixed tiles through the shared pure-core `eocore.streaming`
  helpers instead of loading it whole. The default whole-cube path is unchanged; the
  streamed *Similarity* path is byte-identical to it, *Cluster*'s streamed path fits
  **every** valid pixel via MiniBatch K-means (no subsampling), and the streaming core
  is unit-tested with NumPy fakes.
- **Change resample corrected to nearest-neighbour + nodata-aware (M1).** The opt-in
  *Change* align step is now genuinely nearest-neighbour (the code had drifted to a
  bilinear warp, which interpolates across embedding vectors and contradicted the
  documented behaviour) and propagates nodata through the warp; the aligned rasters are
  streamed window-by-window rather than materialised.
- **Model load hardened against corrupt/foreign files (M5).** `load_model` now fails
  closed with a clear `ValueError` on anything that is not one of our envelopes —
  unreadable pickle data, a non-dict, a dict missing the envelope keys, a non-integer
  version, or a newer envelope format than this build understands — instead of
  surfacing a raw `UnpicklingError`/`KeyError` later. The unavoidable pickle
  **trust boundary is now documented prominently**: unpickling can execute embedded
  code and cannot be sandboxed, so only load model files you trained or trust. Six
  regression tests cover the reject and accept paths.
- **`finalize_embedding` shared between algorithm and test (M7).** The embed tail
  (L2-normalise → stamp nodata → rewrite descriptor) lives in `eocore.runner` and the
  end-to-end orchestration test now drives that exact function, so the test can no
  longer drift from the code it is meant to guard.
- **GUI robustness (M3, M6, L2, L3, L4, L5).** Install/download buttons are disabled
  while a background task runs, so a double-click or parallel-backend launch can't fire
  concurrent `pip` tasks into the one shared interpreter (M3); worker-thread callbacks
  bail out via `sip.isdeleted(self)` if the dialog was closed mid-task (L3); the
  weights-cache size is measured in its own `QgsTask` so the recursive scan never
  stutters the UI (L2); the plugin disconnects the toolbar action and `deleteLater()`s
  both it and the dialog on unload (L4, M6); and a dead `iface` argument was removed
  (L5).
- **Optional-dependency helpers tested (M8).** `eocore.backends._deps` — the single
  `module_available` / `pip_install` the sklearn, STAC and GUI paths all delegate to —
  is covered by 27 torch-free tests exercising the exact `pip` command it builds and the
  optional-import diagnostics it raises, all without spawning pip.
- **CI runs the Clay metadata tests (M2); packaging version trued up (L6).** The
  pure-core CI job installs PyYAML so the vendored-metadata tests run in CI instead of
  skipping, and the `pyproject.toml` version/extras were aligned to 0.2.0.
- **Documented test gap (L7).** The automated suite stops at the orchestration
  boundary: algorithm glue that imports `qgis.core` cannot run in the bare sandbox and
  is covered by the manual [SMOKE_TEST.md](SMOKE_TEST.md); this boundary is now called
  out in [LIMITATIONS.md](LIMITATIONS.md) and CONTRIBUTING.

### Roadmap follow-ups (2026-09-29)

The three near-term items the second deep review recorded in [ROADMAP.md](ROADMAP.md)
are now implemented. As elsewhere, anything that would change analytic output is
opt-in and default-off; each addition is backed by pure-core tests.

- **Automatic low-memory selection.** The opt-in *Low-memory mode* checkbox is now
  backed by an *automatic* trigger. Each analytics algorithm gains two advanced
  parameters — *Auto low-memory* and an *Auto low-memory budget* in MB (default
  1024) — and estimates the whole-cube footprint (pixel count × band count × dtype,
  times the number of input rasters) before it runs. When that estimate exceeds the
  budget it switches to the streamed path on its own and logs the decision, so the
  RAM ceiling lifts without the user knowing to tick a box; a budget of 0 disables
  the auto branch. Auto-selection is **on by default only where streaming is
  byte-identical** to the whole-cube result — *Similarity*, *Change* and *Classify* —
  and **off by default** for *Cluster* and *Embedding → RGB*, whose unsupervised
  *fit* changes when streamed (a would-be out-of-memory run there still gets a
  valid-but-different result, but only when the user opts in). The estimate and the
  predicate live in the pure-core `eocore.streaming` module and are unit-tested.
- **Safer model serialization.** A trained *Classify* model is no longer a bare
  pickle. It is now a small **zip container** whose provenance — band count, backend
  id, kind, class list — is written as **plain JSON** (`envelope.json`), so reading
  and validating a model's metadata never executes code. When the optional `skops`
  package is installed the fitted estimator is stored with skops and reconstructed on
  load through a **trusted-types gate** that admits only `sklearn.` / `numpy.` /
  `scipy.` types and refuses anything else — shrinking the arbitrary-code trust
  boundary that the pickle path could only document. Without skops the estimator
  falls back to pickle, and models written by older builds (a raw pickle) still load
  unchanged; the envelope format version is bumped to 2 and a newer format is
  rejected with an actionable error. The wizard's *Install scikit-learn* button now
  pulls `skops` alongside, and `pyproject`'s `sklearn` extra lists it. Thirteen new
  tests cover the container, the skops dispatch and trusted-types gate (via an
  injected fake skops), and the legacy-pickle read path.
- **Parity tests for the streamed analytics.** The byte-identical guarantee that
  *Similarity* already had is now pinned for *Change*, *Classify*, *Cluster* and
  *Embedding → RGB* as well. A new torch-free `test_streaming_parity` suite runs the
  exact pure-core code paths whole-cube versus streamed and asserts exact equality,
  documenting the one subtlety it must respect: for *Cluster* and *RGB* only the
  *apply* stage is byte-identical (their streamed *fit* deliberately differs — a
  MiniBatch pass over every pixel, or a PCA/percentile fit on a sample), so those are
  pinned against a fixed estimator/transform.

### Clearer errors (2026-09-30)

- **Backend-aware band-mismatch message in *Embed*.** Picking the wrong backend for
  a raster used to fail deep in the model with a low-level message — for example
  imagery fetched with Satlas (3 RGB bands) then embedded with Prithvi produced
  "Prithvi needs exactly 6 bands ..., got 3" after the algorithm had already
  started. *Embed* now checks up front, for backends with a fixed band requirement
  (Prithvi, which standardises positionally against its six HLS bands, and Satlas,
  whose RGB checkpoint takes exactly three channels), that the imagery's bands match,
  and otherwise stops with a message that names the imagery's bands, the backend's
  requirement in order, and — when one matches exactly — the backend whose defaults
  fit the imagery (so the RGB-into-Prithvi case now says the bands match Satlas).
  Clay stays flexible (it consumes bands by name), so it is never blocked. The check
  is a pure-core, torch-free `band_requirement_error` helper with unit tests.
- **"Installed but failed to import" is told apart from "not installed".** When a
  backend's optional model package (TerraTorch for Prithvi, `claymodel` for Clay,
  `satlaspretrain_models` for Satlas) fails to import at *Embed* time, the message no
  longer always says "install it with `pip install …`". It now checks — via
  `importlib.util.find_spec`, which locates a package without importing it — whether the
  package is actually present, and distinguishes two cases: **genuinely absent**, where
  installing it is the fix; and **installed but failing to import**, which almost always
  means a broken or incompatible dependency in the QGIS Python, where reinstalling the
  top-level package changes nothing. Either way the underlying import error is now
  surfaced rather than swallowed, and the "installed but broken" path points at running
  the import in the QGIS Python Console to see the full traceback. This matters in
  practice because a broken deep import can run for a long time (a report of ~105 s)
  before failing, which the old flat "not installed" wording made baffling. The routing
  is a shared, torch-free `optional_import_error` helper in `eocore/backends/_deps.py`,
  with unit tests for both branches.
- **Native-library (DLL) clash hint on Windows.** When that installed-but-broken import
  carries a Windows native-DLL-load signature (`DLL load failed …`, "the specified
  procedure could not be found" = a wrong-version DLL missing a symbol, error 127; or
  "the specified module could not be found" = a dependent DLL absent, error 126), the
  message now adds a targeted hint: the likely cause is a compiled extension binding to a
  different build of a shared native library (HDF5, GDAL, …) than the one QGIS already
  loaded, because Windows permits only one DLL of a given base name per process — so no
  reinstall of the Python package fixes it; the offending package must be pinned to a
  build whose bundled native library matches the version QGIS ships. The detection is a
  small case-insensitive `_looks_like_native_library_clash` helper, unit-tested for the
  126/127 signatures and for *not* firing on ordinary Python `ImportError`s. A full worked
  example — Prithvi/TerraTorch failing on `h5py` because QGIS's HDF5 1.14.6 shadows the
  HDF5 2.0 that h5py ≥ 3.16 bundles, with the version-matched fix (`h5py==3.14.0`, any
  pre-3.16 build being HDF5 1.14.x) and how to read QGIS's HDF5 version with `ctypes` — is
  in the [User Guide](USER_GUIDE.md) troubleshooting section.

### Third deep review (2026-09-30)

A third whole-codebase pass (five parallel review agents over the pure core, the
backends, the algorithm layer, packaging/GUI and the tests/docs). The safe,
well-scoped fixes landed here; the larger items are recorded as recommendations for
a later release. As before, guardrail and error-text fixes each carry a unit test,
and no default behaviour changes silently.

Pure core (`eocore/`):

- **Similarity reference is nodata- and non-finite-aware.** `reference_from_points`
  now drops any reference point that lands on a nodata or non-finite pixel before
  averaging, and raises a clear error if none remain valid — so clicking a scene's
  black border no longer contaminates the reference vector. Wired at both the
  whole-raster and streamed Similarity call sites.
- **Embedding writes never leak NaN/Inf.** `finalize_embedding` folds a finiteness
  check into the coverage mask: a backend that emits a non-finite value for a pixel
  now writes nodata there and drops it from coverage, rather than persisting NaN/Inf
  into the embedding raster.
- **Cosine outputs are hard-clamped.** Similarity is clamped to `[-1, 1]` and cosine
  change distance to `[0, 2]`, so tiny floating-point overshoot can never reach a
  downstream fixed colour ramp, `acos`, or `sqrt`.

Backends (`eocore/backends/`):

- **Satlas local weights.** The initial safe fix rejected a local checkpoint with a
  clear message (Satlas fetched a pretrained checkpoint by id and had no local path);
  full local-checkpoint loading has since been implemented — see *Third-review
  backlog implemented* below.
- **Clay and Satlas model construction is guarded.** A drifted checkpoint or library
  now fails with the actionable "pin a known-good version" error rather than a raw
  `AttributeError`/`KeyError`/`TypeError`.
- **Prithvi validates a local checkpoint path up front** (a clear `FileNotFoundError`)
  before importing TerraTorch, mirroring Clay's behaviour.
- **Bilinear token-grid upsampling preserves a float32 input dtype** — it was
  silently promoting to float64 and doubling peak memory on the pooling path.

Algorithm layer (`algorithms/`):

- **Geographic-CRS pixel size is sane by default.** Fetch imagery now converts a
  backend's native metre resolution to a degree-equivalent when the target CRS is
  geographic (via a pure `metres_to_degrees` helper), instead of treating metres as
  degrees and building a degenerate ~1-degree-per-pixel grid.
- **No truncated rasters on failure.** `write_geotiff` and the warp-to-grid helper
  release the GDAL handle and remove a half-written GeoTIFF (and its `.aux.xml`
  sidecar) if the write or warp raises, before re-raising.

Packaging, docs and tests:

- **Version drift fixed.** `__version__` is now `0.2.0` (was a stale `0.1.0`), pinned
  by a new test that checks it against `metadata.txt` and `pyproject.toml`; mypy's
  checked set now also includes `scripts/`.
- **Documentation made faithful to the code.** The architecture "golden rule" is
  scoped so it no longer claims the QGIS-facing layers are bare-sandbox importable;
  the methodology note now states Prithvi standardises `(DN − mean)/std`; the Embed
  parameter table documents the four advanced parameters; stale test counts are
  corrected; a vacuous provider test now asserts the real `QgsProcessingProvider`
  subclass; and the streaming-parity docstring no longer overclaims end-to-end
  coverage.

### Third-review backlog implemented (2026-09-30)

The larger items the third review had recorded as recommendations are now
implemented, continuing the opt-in / guard-tested pattern; the working tree stays at
0.2.0 and unreleased.

- **Input nodata / off-scene void propagation (HIGH-1).** *Embed* now honours a
  declared input nodata **by default** — a correctness fix, not a toggle. Any pixel
  whose every band equals the raster's declared nodata (or is non-finite) is imputed
  to the valid band mean before the forward pass and given zero mosaic weight, so it
  lands as nodata in the embedding instead of being fed through the ViT as real
  ground (where its receptive field would smear the fill into valid neighbours). A
  separate **opt-in** heuristic (advanced *Detect off-scene void*, default off) also
  treats all-zero pixels as void, for the black corners of a rotated / reprojected
  scene that carry no declared nodata. On a clean scene both guards are byte-identical
  to the previous output, pinned by a dedicated test. Masking, imputation and
  reweighting live in the pure `runner` and are unit-tested with fakes.
- **Full Satlas local-checkpoint loading (HIGH-2).** The Satlas backend now loads a
  local `.pth` state dict instead of rejecting it: the architecture (channel count,
  multi-image flag, backbone) is looked up from the `SatlasPretrain_weights`
  catalogue for the selected checkpoint id, and the state dict is loaded through the
  same `satlaspretrain_models.Model(…, fpn=True, weights=…)` builder the download path
  uses — so a fine-tuned or air-gapped checkpoint works with the download path
  undisturbed. A fail-loud channel-count guard rejects a checkpoint whose expected
  input channels do not match the tile's band count. The catalogue lookup and
  channel-count logic are torch-free and unit-tested with a fake catalogue.
- **Forward-compat QGIS 4 / Qt6 CI job (MED).** A third CI job runs the GDAL/QGIS
  gated tests on the rolling `qgis/qgis:latest` (QGIS 4.x / Qt6) image. It is
  `continue-on-error` — a transient nightly breakage should not red the pipeline — and
  asserts the container really is a Qt6 build, so a genuine import/registration
  regression under Qt6 is still caught early. Justified because the plugin imports Qt
  only through the binding-agnostic `qgis.PyQt` shim.
- **Minor items closed.** `mean_embedding` gained an opt-in nodata-aware mode
  (default `None` = the unchanged plain mean); `fit_pca` now rejects non-finite input
  with a clear error instead of silently emitting an all-black preview; and
  `classify`'s saved-model load wraps `load_model` / `check_compatible` failures as a
  clean `QgsProcessingException` rather than a raw traceback. The partial-nodata rule,
  the `read_points` bounding-box locality, the tile-vs-patch-size edge behaviour, the
  English-only UI (no `tr()` catalogue) and the driver-from-extension output
  convention are now documented in LIMITATIONS. The Satlas channel-count guard shipped
  with HIGH-2 above.

### Satlas fetch: true-colour input out of the box

- **Fetch imagery now pulls Sentinel-2's 8-bit true-colour (TCI) asset for Satlas.**
  Previously *Fetch imagery* resolved one raw-reflectance COG per band for every
  backend, so imagery fetched for SatlasPretrain arrived as `uint16` surface
  reflectance (digital numbers up to ~10000). Fed to the default
  `Sentinel2_SwinB_SI_RGB` checkpoint at its `1/255` scale, that lands near ~36 — far
  outside the `[0, 1]` range the RGB model was trained on — so the default path only
  raised a scale-sanity warning and produced out-of-distribution embeddings. The fetch
  now retrieves Sentinel-2's pre-rendered `visual` asset (a single 3-band
  `red/green/blue` 8-bit COG, `0–255`) and warps that one multi-band asset in a single
  pass, so the fetched imagery is exactly the true-colour input the checkpoint expects
  and the Embed default lands in `~[0, 1]` with no manual *Input scale*. Clay and
  Prithvi are unchanged — they still fetch per-band reflectance and standardise it
  internally. A band-count guard rejects a fetched asset whose band count does not
  match the backend's convention, and torch-free tests pin the `visual` descriptor,
  the `href_for_asset` resolver and the single-asset warp helper.

### Tests & CI

- **GDAL/QGIS container job (H-13).** CI now runs the gated tests in the official
  QGIS 3.34 LTR container (PyQt5 / Qt5) on every push, alongside the pure-core gate
  job — plus a forward-compat job on the QGIS 4.x / Qt6 nightly (`continue-on-error`,
  with a Qt-major assertion) so a Qt6 regression surfaces early without blocking.
- **Fake-backend orchestration + conformance (E-13, H-12, H-14).** A deterministic
  dummy `EmbeddingBackend` exercises the full tile → embed → mosaic → normalise →
  stamp pipeline in the bare sandbox, and a registry-walk test asserts every
  registered backend satisfies the protocol and carries self-consistent metadata.
- The sandbox suite is now **336 pure-core/orchestration tests** (2 GDAL/QGIS-gated
  tests skip where those libraries are absent), all four gates green.

## 0.1.0 — 2026-09-28

Initial release. A QGIS Processing plugin that runs open Earth-observation foundation
models over your own imagery to produce per-pixel embedding rasters, with a
self-contained similarity / change / classification workbench.

### Processing provider

New provider **EO Inference** (id `eoinfer`) with seven algorithms, registered in
workflow order:

- **Fetch imagery (STAC)** (`fetchimagery`) — least-cloudy Sentinel-2 L2A (Earth
  Search, no auth) or HLS V2 (Planetary Computer, auto-signed) for an AOI and date
  range, read via `/vsicurl`.
- **Embed imagery** (`embed`) — run a backend over imagery to a per-pixel embedding
  GeoTIFF; tiled inference with overlap averaging; output L2-normalised and stamped
  with the backend id.
- **Embedding → RGB** (`torgb`) — PCA-to-3 or band-triplet 8-bit preview with a
  percentile stretch.
- **Similarity to reference** (`similarity`) — per-pixel cosine similarity to a mean
  reference vector built from points.
- **Change / trajectory** (`change`) — pairwise / trajectory / anomaly change across
  dated embeddings under a cosine or euclidean metric.
- **Classify embedding** (`classify`) — supervised random-forest / gradient-boosting
  classification, with optional confidence raster and a saveable model envelope that
  records backend id and band count.
- **Cluster embedding** (`cluster`) — unsupervised k-means over a sampled subset.

### Backends

Three inference backends behind one `EmbeddingBackend` protocol: **Clay v1.5**
(Apache-2.0), **Prithvi-EO-2.0** (IBM/NASA, Apache-2.0) and **SatlasPretrain** (Allen
Institute / AI2, ODC-BY). A torch-free registry exposes each backend's facts to the
GUI and docs without importing torch. Concrete backends import torch and their model
packages lazily, at run time.

### Dependencies

The plugin installs and loads with **no third-party dependencies**. PyTorch, the
model packages and scikit-learn are opt-in tiers, detected at run time and installed
on demand from the wizard; model weights are downloaded on first use and cached, never
bundled in the ZIP.

### GUI

A tabbed wizard: a Backends tab with per-backend cards, live PyTorch/package status
badges and background (off-GUI-thread) installs; a Workflow tab that launches each
algorithm's dialog; and an About tab. All long-running algorithm work executes off the
GUI thread and is cancellable.

### Quality

Four gates green in the sandbox — `ruff check`, `ruff format --check`, `mypy --strict`
(strict on `eocore`, relaxed on the QGIS/GDAL layers), and `pytest` (336 pure-core
tests pass; 2 GDAL/QGIS-gated tests skip when those libraries are absent and run where
present). GDAL/QGIS/torch modules are additionally `py_compile`-checked. Compatible
with QGIS 3.34 LTR and the 4.x series (PyQt5 and PyQt6).

### Licensing

Plugin code is GPL-2.0-or-later. Model weights and imagery are third-party and carry
their own licences (Clay Apache-2.0, Prithvi Apache-2.0, Satlas ODC-BY; Sentinel-2
Copernicus, HLS NASA) — see the root `NOTICE`.
