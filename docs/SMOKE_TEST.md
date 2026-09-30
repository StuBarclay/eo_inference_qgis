# Smoke test

The automated suite covers the pure core and (where GDAL/QGIS are present) the raster
and provider bridges. It cannot cover what needs a running QGIS, real weight
downloads and actual model inference. This checklist closes that gap: a manual
end-to-end pass to run before tagging a release or after any change to the backend or
algorithm layers.

Record each run in the template at the end so results are traceable across releases.

## Preconditions

- QGIS 3.34 LTR **or** a QGIS 4.x build, with the plugin installed and enabled.
- Network access (STAC fetch + weight downloads).
- Enough disk for weight caches (hundreds of MB per backend) and a GPU if you want
  inference to finish quickly (CPU works but is slow).

## Part A — Plugin loads clean (no heavy tiers)

Do this in a QGIS whose Python has **no** torch/sklearn installed, to prove the
zero-dependency install.

1. Enable the plugin. It should load with no error, no missing-import traceback.
2. The **EO Inference** provider appears in the Processing Toolbox with all seven
   algorithms under the *EO Inference* group.
3. The toolbar button opens the wizard. The **Backends** tab renders all three cards;
   PyTorch/package badges correctly show the tiers as *missing*.
4. Opening the *Embed* and *Classify* dialogs from the Workflow tab works, and running
   them without the needed tier fails with a **clear, actionable message** (not an
   `ImportError` crash).

## Part B — Tier install from the wizard

1. On a backend card, click **Install**. The UI stays responsive (install runs on a
   worker thread); the log pane shows progress.
2. When it finishes, the badge refreshes to *available* without restarting QGIS.
3. Repeat for the scikit-learn extras card.
4. **CPU-vs-CUDA torch (E-11).** Confirm the torch install offers a CPU-only vs. CUDA
   choice with a sensible default, and that it never silently pulls a multi-gigabyte
   CUDA wheel on a machine without a GPU.

### Part B2 — Weight pre-download, verify and cache (E-5)

1. On a backend card, click **Download / verify weights**. The button runs on a
   worker thread; the log pane announces the download and, when a `Content-Length`
   is known, shows periodic progress.
2. When it finishes, the log prints the file's **SHA-256** (and "verified" if a hash
   is recorded for that checkpoint in the registry). For the default **Clay v1.5**
   checkpoint the expected digest is
   `21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0`
   (5,158,764,629 bytes) — compare the printed value against it to confirm a clean
   download. This digest is deliberately **not** pinned in `weights_sha256` (the
   downloader would otherwise re-hash ~5 GB on every load, and Hugging Face's LFS
   layer already content-addresses the file — see RELEASE_NOTES), so the log will not
   say "verified" by default; the value is published here for manual comparison.
3. The **Weights cache** card shows the cache location and a size that grows after
   the download; **Refresh cache size** updates it.
4. *(Optional integrity check.)* Corrupt a cached weight file, set the backend's
   `weights_sha256` to the previously-recorded good digest, and re-run an embed —
   confirm it is **rejected** with a clear checksum-mismatch message.

## Part C — Per-backend end-to-end

Run the full chain for **each** backend. Use a small AOI (a few km²) to keep runtimes
and downloads modest.

### Clay v1.5

1. *Fetch imagery* with Backend = Clay v1.5, a small AOI, a ~1-month window, max cloud
   ~20 %. Confirm a Sentinel-2 (RGBN) GeoTIFF is written.
2. *Embed imagery* on that raster with Backend = Clay. First run downloads weights;
   confirm it caches and the output embedding raster is written, is float32, and is
   stamped with the backend id. Give it an accurate GSD.
   - **Encoder-output check (the one runtime-only seam).** The backend assumes the
     Clay encoder, loaded with `mask_ratio=0` and `shuffle=False`, returns
     `[CLS, patch_0, patch_1, …]` in **row-major** order, and that the patch count
     equals `grid_h × grid_w` (tile size ÷ patch size, e.g. 256 ÷ 8 = 32 → 1024
     patches). A mismatched patch count now raises an actionable error rather than
     scrambling the grid (`_check_token_count`); if it fires, pin a known-good
     `claymodel` version. To confirm **ordering** (not just count), embed a scene
     with an obvious spatial feature (coastline, field edge) and check the
     *Embedding → RGB* preview reproduces that geometry rather than a scrambled
     one. Record the confirmed `claymodel` version below.
3. *Embedding → RGB* (PCA) — confirm a sensible 3-band preview.
4. *Similarity to reference* with a few points on one target — confirm a −1…1 map that
   peaks near the points.
5. *Classify* with a handful of training polygons (install sklearn first) — confirm a
   class raster; save the model, then re-run with **Load model** and confirm the
   backend/band-count check passes.
6. *Cluster* (k=8) — confirm a cluster-id raster.
7. *Change* — embed a second date and run pairwise/cosine; confirm a change raster.

### Prithvi-EO-2.0

1. *Fetch imagery* with Backend = Prithvi. Confirm it targets **HLS** (six-band, 30 m)
   and that Planetary Computer signing works (a scene is returned).
2. *Embed* with Backend = Prithvi on the six-band input. Confirm weights download/cache
   and an embedding raster is written, is float32, and is stamped with the backend id.
   - **Input convention.** Prithvi standardises each band as `(DN − mean) / std` using
     the HLS statistics in the model's `config.json`, so it expects **raw HLS digital
     numbers** — leave *Input scale* at its default for HLS DN, and set it to `1e4`
     only if your imagery is already reflectance in `[0, 1]`. The scale-sanity check
     warns if the pixels land far outside the range the weights expect.
   - **Encoder-output check (the one runtime-only seam).** The backend assumes the
     TerraTorch backbone returns patch tokens (optionally preceded by a class/register
     token) in **row-major** order, with the patch count equal to `grid_h × grid_w`
     (tile size ÷ patch size 16, e.g. 224 ÷ 16 = 14 → 196 patches, +1 class token →
     197). A mismatched token count now raises an actionable error rather than
     scrambling the grid (`_check_token_count`), and a non-token output rank is caught
     too; if either fires, pin a known-good `terratorch` version. To confirm
     **ordering** (not just count), embed a scene with an obvious spatial feature and
     check the *Embedding → RGB* preview reproduces that geometry rather than a
     scrambled one. Record the confirmed `terratorch` version below.
3. Run *RGB*, *Similarity* and *Change* as above. Confirm results are sane at 30 m.

> **If *Embed* reports "TerraTorch is installed but failed to import"** and reproducing
> `import terratorch` in the QGIS Python Console fails inside `h5py` with `DLL load failed
> … The specified procedure could not be found`, this is a native-library (HDF5) clash on
> Windows, not a broken install: QGIS's own resident `hdf5.dll` shadows the newer HDF5 that
> h5py bundles. Pin h5py to a pre-3.16 release (all bundle HDF5 1.14.x, e.g. `h5py==3.14.0`)
> so it matches QGIS's HDF5, and confirm `h5py.version.hdf5_version` shows a 1.14.x before
> restarting QGIS. The full walk-through — including how to read QGIS's HDF5 version with
> `ctypes` — is in the User Guide, *Troubleshooting → A worked example: a native-library
> (DLL) version clash on Windows*.

### SatlasPretrain

1. *Fetch imagery* with Backend = Satlas (Sentinel-2). The fetch now pulls Sentinel-2's
   pre-rendered **8-bit true-colour** asset (the `visual` / TCI COG: B4/B3/B2, `0–255`),
   which is exactly what the RGB checkpoints expect. Confirm the written raster is
   **3-band** and 8-bit (values in `0–255`), and that the following *Embed* needs **no**
   *Input scale* (leave it `0` / default `1/255`) and raises no scale-sanity warning.
   Only if you instead supply your **own** raw L2A reflectance must you set *Input scale*
   so pixels land in `~[0, 1]`.
2. *Embed* with Backend = Satlas. Confirm the multi-scale features are pooled to a
   per-pixel vector and the embedding raster is written and stamped.
   - **Feature-map seam (runtime-only).** The backend keeps **all** FPN levels and
     concatenates them, so the embedding width is the sum of the level channels
     (expected ~512 for SwinB = four 128-channel levels; the *registry hint is 512*
     and the live descriptor reports the true value). Record the actual band count
     of the embedding raster below. If it comes back as 128 (a single level) the
     model's output was mis-unpacked — that is exactly the case
     `_as_feature_map_sequence` now guards against, so it should not happen on a
     known-good `satlaspretrain-models` version; if it does, pin that version.
3. **Local checkpoint (optional).** Point *Weights* at a local Satlas `.pth` state
   dict (e.g. a fine-tuned `Sentinel2_SwinB_SI_RGB`) and re-run *Embed*. Confirm it
   **loads and embeds** rather than being rejected: the architecture (channels /
   multi-image / backbone) is looked up from the SatlasPretrain catalogue for the
   checkpoint id and the state dict is loaded through the same `Model(…, weights=…)`
   builder as the download path. A checkpoint whose expected channel count does not
   match the imagery's band count must fail with a clear channel-count message.
4. Run *RGB*, *Similarity* and *Cluster* as above.

## Part D — Guardrails

1. **Cross-backend reuse is blocked.** Train/save a Classify model on a Clay embedding,
   then try to Load it against a Satlas embedding — confirm it is **rejected** with a
   backend-mismatch message, not silently applied.
2. **Band-count mismatch is blocked.** Try Change across two rasters with different
   band counts — confirm a clear error.
3. **Cancellation works.** Start a long embed and cancel from the Processing dialog —
   confirm it stops promptly and QGIS stays responsive.
4. **Off-scene fill is not embedded.** *Embed* an imagery raster that declares a
   nodata value (or one with all-zero off-scene corners, run with *Detect off-scene
   void* ticked). Confirm the corresponding embedding pixels come back as nodata — not
   a dense vector — and that valid pixels near the scene boundary match a
   fully-covered run (the fill is imputed out of the forward pass, so it cannot bleed
   into its neighbours).

## Part E — Both Qt bindings

Run at least Part A and one Part C chain under **both** QGIS 3.34 (PyQt5) and a QGIS
4.x build (PyQt6) to confirm the scoped-enum handling holds on both.

## Part F — Opt-in v0.2 features

Every item below is **default-off**; the pure-core maths is unit-tested, so this
part only confirms the QGIS wiring and that defaults preserve 0.1.0 behaviour.

1. **Large-AOI streaming + batching (E-1, E-2).** Embed an AOI larger than comfortable
   RAM and confirm it completes with peak memory tracking the tile size, not the
   scene. Raise the **batch size** above 1 and confirm the output is identical to the
   batch-size-1 run (only faster on a GPU).
2. **Upsampling + feathering (E-3, E-4).** Re-embed with **bilinear** upsampling and
   with a non-zero **overlap feathering** taper; confirm both produce sensible output
   and that the defaults (nearest, uniform) are unchanged.
3. **Classification accuracy report (E-6).** Enable the holdout split and confirm a
   JSON sidecar with overall accuracy, per-class precision/recall/F1 and a confusion
   matrix is written next to the class raster; default-off writes none.
4. **Clustering auto-k / full-data (E-7).** Enable the silhouette sweep and confirm the
   chosen k and per-k scores are reported; try the MiniBatchKMeans full-data path on a
   large raster.
5. **STAC multi-scene composite (E-8).** Enable multi-scene mode over an AOI that
   straddles two scenes and confirm a per-pixel composite is embedded end-to-end;
   single-scene stays the default.
6. **Change resample-to-match (E-12).** Feed *Change* two rasters on different grids
   with the align option **on** and confirm it resamples and runs; with it **off**
   confirm it still rejects the mismatch.
7. **Model-API guards (H-6).** These fail loud only against a drifted library version,
   so there is nothing to trigger on a good install — just confirm normal embeds do
   not hit the guard, and record the exact verified library + checkpoint versions
   below so a future mismatch is diagnosable.

---

## Recorded run log

Copy this block and fill it in for each release verification.

```
EO Inference Toolbox — smoke test record
Plugin version : 0.2.0
Date / tester  : __________ / __________
QGIS version   : __________ (Qt binding: PyQt5 / PyQt6)
OS             : __________
GPU present    : yes / no

Part A  plugin loads clean (no tiers)      : PASS / FAIL   notes: ____
Part B  tier install from wizard           : PASS / FAIL   notes: ____
Part B2 weight download / verify / cache   : PASS / FAIL   notes: ____
Part C  Clay end-to-end (fetch→...→change) : PASS / FAIL   notes: ____
Part C  Prithvi end-to-end                 : PASS / FAIL   notes: ____
Part C  Satlas end-to-end                  : PASS / FAIL   notes: ____
Part D  guardrails (reuse/band/cancel)     : PASS / FAIL   notes: ____
Part E  second Qt binding                  : PASS / FAIL   notes: ____
Part F  opt-in v0.2 features               : PASS / FAIL   notes: ____

Weights downloaded (backend → size / time / SHA-256):
  clay    : ____
  prithvi : ____
  satlas  : ____

Verified library + checkpoint versions (for H-6 diagnosability):
  torch                 : ____
  clay (claymodel)      : ____   checkpoint: ____
  prithvi (terratorch)  : ____   checkpoint: ____
  satlas (satlaspretrain-models) : ____   checkpoint: ____

Issues found / follow-ups:
  - ____
```
