# User guide

A practical walkthrough of installing the plugin, setting up a backend, and running
each of the seven algorithms. For the reasoning behind the steps see
[METHODOLOGY.md](METHODOLOGY.md); for what the tool cannot do see
[LIMITATIONS.md](LIMITATIONS.md).

## Installation

The plugin targets QGIS 3.34 LTR and the QGIS 4.x series (PyQt5 and PyQt6 both
work). Install it either from a built ZIP via *Plugins → Manage and Install Plugins
→ Install from ZIP*, or by copying the `eo_inference_toolbox/` folder into your QGIS
plugin directory and enabling it. The plugin installs with **no third-party
dependencies** — the heavy tiers are added on demand later.

Once enabled you get a Processing provider called **EO Inference** (id `eoinfer`)
with seven algorithms under the *EO Inference* group, plus a toolbar button that
opens the wizard dialog.

## Setting up a backend

Open the wizard (toolbar button) and go to the **Backends** tab. Each of the three
backends — Clay v1.5, Prithvi-EO-2.0, SatlasPretrain — has a card showing its facts
and a live status badge for PyTorch and its model package. If a badge shows a package
missing, click **Install** on that card; the install runs in the background (you can
keep using QGIS) and the badge refreshes when it finishes. The extras card installs
scikit-learn (for classify/cluster) the same way.

Nothing is bundled: PyTorch, the model packages and the weight files are all fetched
on demand and cached. The first embed run with a given backend downloads its weights,
which can take a while; later runs reuse the cache. The weight files are large —
Clay's `v1.5/clay-v1.5.ckpt` checkpoint is about **5.2 GB**, so on a typical
connection expect roughly ten to thirty minutes for the first fetch (a few minutes on
a fast link). The download runs in the background and huggingface_hub renders its own
byte progress to the console/log rather than the wizard's small log pane, so a quiet
pane during a weight fetch is normal — watch the weights-cache size (the *Refresh
cache size* button) or the *EO Inference* tab of the Log Messages panel to confirm it
is progressing. You can trigger it ahead of time with *Download / verify weights*.

Which backend to pick: **Clay** is the recommended default — best documented, cleanest
multi-sensor behaviour, and it makes use of geolocation/time metadata. **Prithvi**
suits 30 m HLS multi-temporal work with its fixed six-band input. **Satlas** gives
multi-scale feature-map embeddings pooled to a per-pixel vector.

### Installing PyTorch and the model packages manually

The wizard's install buttons are the easy path, but PyTorch is a large, environment-
sensitive download and the wizard runs pip with its output *captured* (nothing streams
back until the install finishes) under a two-hour timeout. For PyTorch specifically —
and on any locked-down or read-only deployment where installing from inside QGIS
fails — installing from a terminal is more reliable, because you see live progress and
can pick the exact CPU or CUDA build. The steps below are for Windows; on Linux/macOS
use the QGIS Python the same way from a normal shell.

Do the installs **one at a time**, and let each finish before starting the next.
Running two installs into the same interpreter at once (for example clicking two
wizard buttons) can corrupt the environment through file-lock contention.

1. **Open the OSGeo4W Shell as Administrator.** QGIS's `site-packages` usually live
   under `C:\Program Files\...`, so administrator rights let pip write there. Find it
   in the Start menu under your QGIS folder (e.g. "QGIS 3.34"), right-click *OSGeo4W
   Shell* and choose *Run as administrator*. If you hit an "Access is denied" error
   later, this step was missed.

2. **Confirm the shell is driving the QGIS Python and that pip works:**

   ```bat
   python -c "import sys; print(sys.executable); print(sys.version)"
   python -m pip --version
   ```

   The first line should print a real `python.exe` inside your QGIS install and a
   3.x version. If it prints `qgis-bin.exe`/`qgis-ltr-bin.exe` instead, `python -m pip`
   will misbehave — run `py3_env` first, then retry; or check the path with
   *Plugins → Python Console* inside QGIS (`import sys; print(sys.executable)`).

3. **Install PyTorch.** Choose the build that matches your hardware and install it on
   its own before anything else. Adding `torchvision` on the **same** index keeps the
   two builds consistent (installing it separately later pulls a mismatched wheel).

   *CPU-only* (works everywhere, no GPU needed):

   ```bat
   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   ```

   *NVIDIA GPU (CUDA).* First check the driver — run `nvidia-smi` and read the "CUDA
   Version" shown top-right; it is the highest CUDA your driver supports and must be
   **12.1 or newer** for the wheels below. If it is lower, update your NVIDIA driver
   first. Then install the CUDA build (this matches the `cu121` index the wizard's
   *Install PyTorch (CUDA)* button uses; the download is roughly 2.5 GB):

   ```bat
   nvidia-smi
   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   ```

4. **Verify PyTorch — and, for the CUDA build, that it sees the GPU:**

   ```bat
   python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
   ```

   For a GPU install you want `CUDA available: True` and your GPU's name. `False` means
   a driver/toolkit mismatch (recheck step 3) — everything still works, just on CPU.

5. **Install the backend package(s)** you intend to use, one at a time. Clay is a
   build from a git repository, so it needs `git` on `PATH` (check with `git --version`;
   if missing, install Git for Windows and reopen the shell). This git build is the
   slowest step and the most likely to stall silently in the wizard — running it here
   lets you watch it:

   ```bat
   python -m pip install "git+https://github.com/Clay-foundation/model"
   python -m pip install terratorch
   python -m pip install satlaspretrain_models
   ```

   (`terratorch` is Prithvi; `satlaspretrain_models` is Satlas — install only the ones
   you need.)

6. **Optional tiers:** scikit-learn (for Classify/Cluster) and, for HLS imagery, the
   Planetary Computer signer:

   ```bat
   python -m pip install scikit-learn
   python -m pip install planetary-computer
   ```

7. **Back in QGIS**, reopen the wizard and click **Refresh status** — the relevant
   badges turn green. Do not use the wizard's combined *Install backend* button for a
   tier you are installing manually at the same time.

### Checking the install

Before a real analysis, a two-minute confidence check confirms the plugin and any
tier you installed are wired up. For a thorough end-to-end verification — every
backend, both Qt bindings, the guardrails — follow [SMOKE_TEST.md](SMOKE_TEST.md).

1. **The plugin loaded clean.** The **EO Inference** provider is in the Processing
   Toolbox with all seven algorithms, and the toolbar button opens the wizard. This
   holds even with no heavy tier installed — the plugin has no third-party imports at
   load time.
2. **The tier you need imports from the QGIS Python.** Open *Plugins → Python Console*
   and import the package for your backend — `import claymodel` (Clay),
   `import terratorch` (Prithvi) or `import satlaspretrain_models` (Satlas) — plus
   `import torch` for the Embed step and `import sklearn` for Classify/Cluster. An
   import that succeeds here is the real test; the Backends-tab badge is a cached view
   of the same check.
3. **Weights are present or will download.** Click *Download / verify weights* on the
   backend card, or simply start the first embed, and watch the weights-cache size
   grow (the *Refresh cache size* button) or the *EO Inference* Log Messages tab.
4. **A tiny run completes.** Fetch a small scene, embed it, and run Similarity with a
   couple of points — one full pass end to end. If the embed instead reports that "the
   vendored Clay metadata file is missing", the installed plugin copy is incomplete;
   reinstall it from the ZIP so the bundled model-metadata asset is present.

## The workflow, step by step

The **Workflow** tab lists the algorithms in run order; each button opens that
algorithm's standard Processing dialog. You can equally run them from the Processing
Toolbox under *EO Inference*.

### 1. Fetch imagery (STAC) — `eoinfer:fetchimagery`

Pulls a cloud-filtered scene for your AOI from a public catalogue.

| Parameter | Meaning |
|---|---|
| Area of interest (`EXTENT`) | The bounding box to fetch. |
| Output CRS (`CRS`) | CRS for the written raster (defaults to the project CRS). |
| Backend (`BACKEND`) | Chooses the default collection + bands (Clay/Satlas → Sentinel-2 L2A; Prithvi → HLS). For Satlas the pre-rendered 8-bit true-colour asset (Sentinel-2's `visual` / TCI, `0–255`) is fetched instead of raw reflectance, so it matches the RGB checkpoint out of the box. |
| Start / End date (`START`, `END`) | `YYYY-MM-DD` window; the least-cloudy scene in range is chosen. |
| Max cloud cover % (`MAX_CLOUD`) | Upper cloud limit for candidate scenes (default 20). |
| Pixel size (`PIXEL_SIZE`) | Output resolution in CRS units; 0 uses the sensor-native GSD. |
| Imagery (`OUTPUT`) | The resulting imagery GeoTIFF. |

You can skip this step entirely and supply your own local imagery to the embed step.

### 2. Embed imagery — `eoinfer:embed`

Runs a backend over imagery to produce the per-pixel embedding raster. **This is the
only step that needs PyTorch.**

| Parameter | Meaning |
|---|---|
| Imagery raster (`INPUT`) | Source imagery (from step 1 or your own). |
| Backend (`BACKEND`) | Which model to run. Must match the imagery's bands. |
| Bands (`BANDS`) | Optional band-order override; blank uses the backend default. |
| Tile size (`TILE`) | Window size in pixels for tiled inference (default 256). |
| Overlap (`OVERLAP`) | Tile overlap in pixels, averaged to avoid seams (default 32). |
| Ground sample distance (`GSD`) | Metres/pixel hint; 0 derives it from the raster. Clay uses this. |
| Input scale (`INPUT_SCALE`) *(advanced)* | Pre-scale into the units the weights expect; 0 = backend default. Clay/Prithvi standardise raw DN internally (leave 0 for DN imagery; set 1e4 only if yours is already reflectance in [0,1]); Satlas expects 8-bit true colour (default 1/255) — imagery fetched for Satlas in step 1 is already 8-bit true colour, so leave this at 0. |
| Upsampling (`UPSAMPLING`) *(advanced)* | Patch-grid upsampling method; bilinear is smoother but blends across patch edges (default nearest). |
| Batch size (`BATCH_SIZE`) *(advanced)* | Tiles per forward pass; 1 = one tile at a time, raise for GPU throughput. |
| Feather (`FEATHER`) *(advanced)* | Overlap feathering taper in pixels; 0 = uniform blend (off). |
| Detect off-scene void (`DETECT_VOID`) *(advanced)* | Treat all-zero pixels as off-scene fill (the black corners of a rotated / reprojected scene) and keep them out of the embedding. Default off. A declared raster nodata is always honoured regardless of this box. |
| Weights (`WEIGHTS`) | Optional explicit weight file/checkpoint id; blank uses the backend default. Also accepts a local checkpoint file — e.g. a fine-tuned Satlas `.pth`, loaded against the selected checkpoint id's architecture. |
| Cache dir (`CACHE_DIR`) | Where to cache downloaded weights. |
| Embedding (`OUTPUT`) | The L2-normalised embedding GeoTIFF (stamped with the backend id). |

The output is already unit-normalised, so it drops straight into the analysis steps.

Input nodata is handled automatically: a pixel whose every band equals the imagery's
declared nodata value (or is non-finite) is kept out of the forward pass and written
as nodata in the embedding, so off-scene fill never contaminates the valid pixels
around it through the model's receptive field. Rotated or reprojected scenes whose
off-scene corners are plain zeros — with no declared nodata — are only excluded when
*Detect off-scene void* is ticked, since an all-zero pixel is a heuristic rather than
a certain signal of fill.

### 3. Embedding → RGB — `eoinfer:torgb`

A viewable 3-band preview of an embedding raster.

| Parameter | Meaning |
|---|---|
| Embedding raster (`INPUT`) | The embedding to visualise. |
| Method (`METHOD`) | `pca` (fit a 3-component projection) or `bands` (pick a triplet). |
| Bands (`BANDS`) | For the `bands` method: which three embedding dimensions to map to R/G/B. |
| Low / High percentile (`LOW_PCT`, `HIGH_PCT`) | Stretch clip for the 8-bit output (defaults 2 / 98). |
| Seed (`SEED`) | RNG seed for the PCA sample (reproducibility). |
| RGB preview (`OUTPUT`) | The 8-bit preview raster. |

Preview only — never feed the RGB back into analysis.

### 4. Similarity to reference — `eoinfer:similarity`

Per-pixel cosine similarity to a reference built from points.

| Parameter | Meaning |
|---|---|
| Embedding raster (`INPUT`) | The embedding to search. |
| Reference points (`POINTS`) | Point layer sampling your target; their mean embedding is the reference. |
| Similarity (`OUTPUT`) | Similarity map, −1 … 1 (1 = identical to the reference). |

Use several points on the same target class for a more robust reference.

### 5. Change / trajectory — `eoinfer:change`

Change between dated embedding rasters.

| Parameter | Meaning |
|---|---|
| Embedding rasters (`INPUT`) | Two or more dated embeddings (same backend, same grid). |
| Mode (`MODE`) | `pairwise` (two dates), `trajectory` (summed consecutive movement), `anomaly` (distance from baseline mean). |
| Metric (`METRIC`) | `cosine` (1 − cosine similarity) or `euclidean`. |
| Change (`OUTPUT`) | The change/trajectory/anomaly raster. |

Cosine is the natural default for these normalised embeddings.

### 6. Classify embedding — `eoinfer:classify`

Supervised per-pixel classification. **Requires scikit-learn.**

| Parameter | Meaning |
|---|---|
| Embedding raster (`INPUT`) | The embedding to classify. |
| Training features (`TRAINING`) | Polygons/points labelled with a class field. |
| Class field (`CLASS_FIELD`) | Attribute holding the class label. |
| Classifier (`CLASSIFIER`) | `random_forest` or `gradient_boosting`. |
| N estimators (`N_ESTIMATORS`) | Number of trees (default 200). |
| Seed (`SEED`) | RNG seed. |
| Load model (`LOAD_MODEL`) | Optional saved model to apply instead of training (backend/bands checked). |
| Classification (`OUTPUT`) | Per-pixel class raster. |
| Output model (`OUTPUT_MODEL`) | *(advanced)* Save the trained model envelope. |
| Confidence (`CONFIDENCE`) | *(advanced)* Optional per-pixel confidence raster. |

A saved model records the backend id and embedding band count; loading it against a
mismatched raster is rejected rather than silently misapplied. The saved model is a
small zip file (keep the `.pkl` name): its metadata is plain JSON, and the fitted
classifier inside is stored with `skops` when that optional package is installed (it
rides along with the wizard's *Install scikit-learn* button) or with pickle otherwise.
Load only model files you trained or trust — a pickle-backed model can run code as it
loads, whereas the JSON metadata is always safe to read.

### 7. Cluster embedding — `eoinfer:cluster`

Unsupervised k-means segmentation. **Requires scikit-learn.**

| Parameter | Meaning |
|---|---|
| Embedding raster (`INPUT`) | The embedding to cluster. |
| Number of clusters (`N_CLUSTERS`) | *k* (default 8). |
| Seed (`SEED`) | RNG seed. |
| Max samples (`MAX_SAMPLES`) | Pixels sampled to fit k-means (default 100 000). |
| Clusters (`OUTPUT`) | Per-pixel cluster-id raster. |

## Large rasters and memory

The analytics steps (*Similarity*, *Change*, *Classify*, *Cluster*, *Embedding →
RGB*) read the whole embedding cube into memory by default — fastest, but it caps the
raster size a given machine can handle, and a 768–1024-band embedding is heavy. Two
parameters under each dialog's **Advanced** section lift that ceiling:

- **Low-memory mode** — tick it to force the step to stream the raster in fixed tiles
  instead of loading it whole. Slower, but peak memory tracks the tile, not the scene.
- **Auto low-memory** + **Auto low-memory budget (MB)** — with *Auto low-memory* left
  on, the step streams by itself whenever the estimated whole-cube size exceeds the
  budget (default 1024 MB), and notes the decision in the log; set the budget to 0 to
  turn the automatic behaviour off.

For *Similarity*, *Change* and *Classify* the streamed result is byte-for-byte the
same as the whole-cube result, so *Auto low-memory* is on by default there. *Cluster*
and *Embedding → RGB* fit on every pixel when streamed rather than on a sample, which
shifts the result slightly, so they stream only when you tick *Low-memory mode*
yourself.

## A typical session

For a "find everything like this" task: fetch a recent clear Sentinel-2 scene over
your AOI with Clay selected, embed it, then run Similarity with a handful of points
on your target. For "what changed here": fetch and embed two dates, then run Change
in pairwise/cosine mode. For a land-cover map: embed one scene, draw a few training
polygons per class, and run Classify with a random forest — save the model so you can
apply it to the next date without retraining.

## Troubleshooting

If an algorithm reports a missing package, open the wizard's Backends tab and install
the relevant tier (PyTorch for embed; scikit-learn for classify/cluster). If a STAC
fetch returns nothing, widen the date window or raise the max-cloud threshold. If a
change or classify run complains about band counts or backend mismatch, the inputs
were produced by different backends or checkpoints — re-embed them consistently. Long
runs execute off the GUI thread and can be cancelled from the Processing dialog.

### Embed stops with "This imagery has N band(s) … needs exactly M"

*Embed* checks the imagery's bands against the backend you picked before it starts, so
a wrong pairing stops immediately with a message that names the imagery's bands, the
backend's fixed requirement, and — when it can — the backend those bands actually
match. It almost always means the backend selected for *Embed* differs from the one the
imagery was fetched with: Prithvi needs six HLS bands (blue, green, red, narrow-NIR,
SWIR1, SWIR2) and the Satlas RGB checkpoint needs exactly three (red, green, blue), so a
3-band scene fetched for Satlas cannot be embedded with Prithvi, and vice versa. Either
switch the *Backend* to the one the message points at, or re-fetch the area with the
backend you want and embed that. Clay consumes whatever bands it is given (matching them
by name), so it is never blocked this way.

### A wizard install seems stuck, or has run for hours

The wizard captures pip's output rather than streaming it, so the log pane shows only
the "Installing… (this can take a while)…" line until the whole install finishes —
a quiet log is not by itself a sign of a hang. But each install has a two-hour
timeout, so an install still showing "Installing…" the next morning, with the status
badges still red and no "Install finished/failed" line, has stalled, been killed, or
already reported its result off the bottom of the small pane. Check it as follows:

1. Click **Refresh status** in the wizard. The badges re-test whether the package is
   actually importable, so a silent success will turn PyTorch green here.
2. Open **View → Panels → Log Messages** and select the **EO Inference** tab. Every
   wizard message is mirrored there *with timestamps*, so you can see how long ago the
   install started and whether a finished/failed line scrolled out of the dialog.
3. Look at the task/progress widget at the bottom of the QGIS window — is the
   "Installing…" task still active, or gone?

The usual causes are launching two installs at once, or Clay's git build stalling
(often because `git` is missing or prompting), or — on Windows — the QGIS
`sys.executable` pointing at `qgis-bin.exe` rather than a real `python.exe`. To
recover: close QGIS, end any lingering `pip.exe`/`git.exe`/`python.exe` processes in
Task Manager to release the file locks, then install from a terminal following
*Installing PyTorch and the model packages manually* above, which shows live progress.
Cancelling the task in QGIS only flags it; the underlying pip/git process may keep
running until it is killed or the machine is restarted.

### A backend still shows "package: missing"

Two things to know first. Each backend has its **own** package, and you only need the
one(s) you actually intend to run — a fresh install legitimately shows Clay, Prithvi
and Satlas all missing until you install one, and installing Clay does not turn
Prithvi or Satlas green. Second, the badge reflects whether the package is *importable
from the QGIS Python*, which is not always the interpreter a terminal `pip` used.

If a badge stays red for a backend you did install, confirm the package imports from
the very interpreter QGIS runs. Open *Plugins → Python Console* inside QGIS and run the
line for your backend:

```python
import claymodel  # Clay
import terratorch  # Prithvi
import satlaspretrain_models  # Satlas
```

If the import fails there but the same line works in your OSGeo4W Shell, the install
landed in a different Python — redo it in the OSGeo4W Shell that reports the QGIS
`python.exe` (see step 2 of the manual install above). If the import *succeeds* in the
QGIS console, the backend is ready to use even if a stale badge hasn't refreshed —
click **Refresh status**, and note that the Embed step imports the package directly
and will run regardless of the badge.

### Embed stops with "… is installed but failed to import"

This is the opposite of "package: missing". The backend's package (`terratorch` for
Prithvi, `claymodel` for Clay, `satlaspretrain_models` for Satlas) *is* present on the
QGIS Python's import path — so *Embed* does not tell you to install it — but importing
it raised part-way through, which almost always means a broken or incompatible
dependency underneath it (a version clash, a partial install, or a companion package
that itself fails to import). Reinstalling the top-level package usually changes
nothing, because the top-level package is not what is broken. Note that this failure can
take a while to surface: the import runs until it hits the broken dependency before it
raises, so a run that sits for a minute or two and *then* reports this message is
expected, not a second hang.

The message includes the underlying import error, but the fastest way to see the whole
picture is to reproduce it in QGIS's own interpreter. Open *Plugins → Python Console*
and run the import for your backend:

```python
import terratorch  # Prithvi
import claymodel  # Clay
import satlaspretrain_models  # Satlas
```

The full traceback names the dependency that actually failed. Fix that package
(commonly: install or pin the missing/clashing dependency it names, or reinstall it in
the same OSGeo4W Shell that reports the QGIS `python.exe`), then re-run the import until
it returns cleanly before trying *Embed* again.

#### A worked example: a native-library (DLL) version clash on Windows

One common shape of this failure on Windows is a *native-library version clash*, and it
is worth walking through because the fix is not a Python-package reinstall. The model
packages pull in compiled extensions (h5py, the GDAL bindings, PyTorch, …) that each ship
their own build of a shared native library. Windows loads only **one** DLL of a given
base name per process, so whichever copy loads first wins — and inside QGIS that is
almost always QGIS's own copy, loaded at start-up before your package gets a look in.

A concrete case seen with Prithvi: *Embed* reports that TerraTorch "is installed but
failed to import", and reproducing `import terratorch` in the QGIS Python Console shows
the real cause a few frames down, raised from `h5py` (which TerraTorch imports
transitively):

```
ImportError: DLL load failed while importing _errors:
The specified procedure could not be found.
```

"The specified procedure could not be found" (Windows error 127) means a DLL *did* load
but was the **wrong version** — it was missing a symbol h5py needed. What happened: QGIS
ships HDF5 and loads its `hdf5.dll` at start-up; the installed h5py bundles a *newer*
HDF5, but because QGIS's `hdf5.dll` is already resident, h5py binds to that older copy,
which does not export everything the newer h5py expects. (Contrast Windows error 126,
"The specified *module* could not be found", which means a dependent DLL is missing
entirely rather than being the wrong version.)

Two checks confirm it is a clash and not a broken install:

- The **same `import h5py` succeeds in the standalone OSGeo4W `python.exe`** (nothing has
  pre-loaded a conflicting `hdf5.dll` there) but **fails in the QGIS Python Console**. A
  genuinely broken install would fail in both.
- Read the HDF5 version QGIS actually ships. PowerShell is usually not on the OSGeo4W
  Shell `PATH`, so read it directly from the DLL with `ctypes` (adjust the path to your
  QGIS version):

  ```python
  import ctypes

  dll = ctypes.CDLL(r"C:\Program Files\QGIS 4.2.2\bin\hdf5.dll")
  maj, minor, rel = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
  dll.H5get_libversion(ctypes.byref(maj), ctypes.byref(minor), ctypes.byref(rel))
  print("QGIS HDF5", maj.value, minor.value, rel.value)  # e.g. 1 14 6
  ```

  Compare that with the HDF5 your h5py bundled, from the standalone shell:
  `python -c "import h5py; print(h5py.version.hdf5_version)"`.

The fix is to **match the versions** — pin the offending package to a build whose bundled
native library is **no newer than** the one QGIS ships, so QGIS's resident copy still
exports every symbol the package needs. For h5py the boundary is clean: **h5py 3.16 was
the first PyPI build to bundle HDF5 2.0**, so any h5py release *before* 3.16 bundles an
HDF5 1.14.x, all of which work with QGIS's 1.14.6 (1.14.6 is the highest 1.14 patch).
Install such a build into the QGIS user-site without disturbing anything else — `3.14.0`
sits safely below that 3.16 boundary, and `--no-deps` keeps it from dragging in the
NumPy 2.x that the newer wheels want:

```
python -m pip install --user --force-reinstall --no-cache-dir --no-deps --only-binary=:all: h5py==3.14.0
```

Then, **before restarting QGIS**, confirm the bundled HDF5 is now a 1.14.x:

```
python -c "import h5py; print(h5py.__version__, h5py.version.hdf5_version)"
```

expecting something like `3.14.0 1.14.6`. Restart QGIS fully, then re-run `import h5py`
and `import terratorch` in the Python Console; both should now return cleanly. If a pin
still clashes, step down another release (the newest pre-3.16 line, then 3.14.0 → 3.13.0
→ 3.12.1) until `h5py.version.hdf5_version` reports a 1.14.x; as a last resort, build
h5py from source against QGIS's own HDF5. The same principle applies to any native
library QGIS also loads — GDAL is the other common one: identify the shared library named
in the failing extension, read the version QGIS ships, and install a build of the Python
package that matches it.
