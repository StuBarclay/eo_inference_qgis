# Limitations and known constraints

This is an honest inventory of what the toolbox does *not* do, where results should
be treated with caution, and the boundaries of the current release. Read it alongside
[METHODOLOGY.md](METHODOLOGY.md).

## Status

The plugin is marked **experimental**. The pure-Python core (`eocore/`) is
unit-tested with NumPy; the GDAL and QGIS bridges are covered by gated tests that run
where those libraries exist and by compile-checks everywhere else. The PyTorch
backend paths and the end-to-end in-QGIS behaviour are verified by a **manual smoke
test** ([SMOKE_TEST.md](SMOKE_TEST.md)), not by automated CI, because they need
GPU-class resources, large weight downloads and a running QGIS.

## Embeddings are only comparable within a backend and comparable imagery

Every similarity, change and classification result assumes the embeddings being
compared came from the **same backend and checkpoint**, over **comparable imagery**.
Mixing backends, or comparing a hazy scene against a clear one, or comparing different
sensors, will surface differences that are artefacts of the inputs rather than real
ground change. The tool stamps and checks the backend id to prevent the most obvious
mistake (reusing a model across backends), but it cannot police imagery quality for
you — that judgement stays with the analyst.

## The models themselves

None of the three backends ships a global pre-computed archive; you must run
inference, which needs PyTorch and, in practice, meaningful compute. CPU-only
inference works but is slow on large AOIs, though *Embed* now streams the scene
tile-by-tile (so memory is no longer the ceiling) and can run several tiles per
forward pass on a GPU. Weight files are large and downloaded on first use, so the
first run of each backend has a substantial one-off cost — the wizard can now
pre-download and verify weights and show the cache size, so that cost is visible
up front rather than a surprise on the first embed.
Backend-specific constraints apply: Prithvi requires its fixed six-band 30 m HLS
input and will not accept arbitrary band sets; Satlas emits multi-scale feature maps
that are pooled to a single per-pixel vector, which is a lossy summary of its native
output; Clay benefits from accurate geolocation/time/GSD metadata and degrades if
those are wrong. Nominal embedding dimensions and resolutions in the registry are
hints for the default checkpoint — the live model reports the true values once loaded.
The patch-based (ViT) backends divide each read tile into fixed-size patches
internally; when a tile's height or width is not an exact multiple of the backend's
patch size (commonly 16 px) the remainder is padded or cropped before the per-pixel
vectors are pooled back to the tile grid, so the output stays well-defined but the
trailing edge of such a tile carries marginally less context. Overlapping tiles with
feathering hide the resulting seams; if you want to avoid edge-patch padding
altogether, choose a tile size that is a whole multiple of the model's patch size.

Clay standardises each band as `(DN − mean) / std` using the per-sensor statistics
published with the model, exactly as Clay's own embedding pipeline does; the plugin
feeds raw Sentinel-2 L2A digital numbers straight in (the *Input scale* override is
for imagery in other units). One subtlety to be aware of: since ESA processing
baseline 04.00, Sentinel-2 L2A products carry a `BOA_ADD_OFFSET` of −1000, so the raw
DN are offset by +1000 relative to older products. The plugin does not remove this
offset, because doing so would diverge from the distribution Clay was trained and
demonstrated on — matching Clay's tutorial (raw DN through the published mean/std) is
the faithful choice. If Clay's published statistics predate the offset there may be a
small systematic shift, but it affects Clay's own pipeline identically and does not
change the internal consistency of embeddings used for similarity, change or
clustering. If you deliberately pre-correct the offset in your imagery, leave the
*Input scale* at its default so standardisation still sees DN-scale values.

Prithvi follows the same convention as Clay: it standardises each band as
`(DN − mean) / std` using the per-band statistics published in the model's
`config.json`, exactly as Prithvi's own preprocessing does — the pretrained
backbone does not rescale or normalise its input. Those statistics are HLS
surface-reflectance values in raw digital-number units (means in the thousands,
not near 0 or 1), shared across the 300M/600M variants, and the plugin keys them
positionally to Prithvi's fixed six-band order (Blue, Green, Red, narrow-NIR,
SWIR1, SWIR2). The plugin feeds raw HLS digital numbers straight in, so leave
*Input scale* at its default for HLS DN imagery; set it to `1e4` only if your
imagery is already reflectance in `[0, 1]` (it is then pre-scaled back into DN
units before standardisation). Prithvi is a *multi-temporal* model — its native
input is a short time series, shape `(batch, channels, time, height, width)` — but
this plugin feeds a single acquisition (`time = 1`). Embeddings therefore describe
one date and do not encode the temporal dynamics the model can otherwise capture;
exposing a multi-date input is future work (see the roadmap).

Satlas has its own input convention, and it differs from Clay's. The Sentinel-2
*RGB* checkpoints (the default, `Sentinel2_SwinB_SI_RGB`) expect a true-colour
image built from the B4/B3/B2 bands **in that order**, with 8-bit `0–255` values
divided by 255 — i.e. a TCI-style visual product, *not* raw reflectance. The
default *Input scale* of `1/255` matches that. The built-in *Fetch imagery* step now
supplies this automatically for Satlas — it pulls Sentinel-2's pre-rendered `visual`
(TCI) asset rather than reflectance bands — so the default path is correct out of the
box, and the caveat that follows applies mainly to imagery you supply yourself. If you
feed raw Sentinel-2 L2A
reflectance (digital numbers up to ~10000) the `1/255` default lands the pixels
far outside the `[0, 1]` range the weights saw, which produces meaningless
embeddings; set *Input scale* so your data maps into roughly `[0, 1]` (Satlas's
own recipe for non-8-bit sources scales and then clips to `[0, 1]`). The
scale-sanity check warns when the scaled maximum is implausible, but it cannot
prove the units are right — supply a true-colour 8-bit input, or scale to match
it. This backend also feeds a single image per tile, so it targets the
single-image (`_SI_`) checkpoints; the multi-image (`_MI_`) variants expect
several aligned images stacked on the channel axis and are outside this path.
Finally, Satlas emits several FPN feature maps that are resampled to the tile
grid and concatenated, so the embedding width is the sum of the FPN level
channels (about 512 for the SwinB default) rather than any single reported
figure; the registry value is a hint and the live descriptor reports the true
width once the model is loaded.

## Imagery acquisition

STAC fetch defaults to the least-cloudy single scene in the date window. An opt-in
multi-scene mode builds a per-pixel median / first-valid composite over the top few
least-cloudy scenes, which helps AOIs that straddle scene boundaries or lack one
clean acquisition; single-scene stays the default for reproducibility, and neither
mode does true cloud gap-filling, so for difficult windows you may still need to
supply your own prepared imagery. Sentinel-2 comes from Earth Search without
authentication; HLS comes from
Planetary Computer, which requires request signing (applied automatically) and is
subject to that service's availability and rate limits.

## Analytics

The RGB preview is a lossy visualisation and must never be used as analysis input.
By default every analytics algorithm (*Similarity*, *Change*, *Classify*, *Cluster*
and *Embedding → RGB*) reads the whole embedding cube into memory, which sets a
practical ceiling on raster size for high-dimensional embeddings; each offers a
**Low-memory mode** that streams the raster in tiles instead, at some speed cost.
That mode can be ticked explicitly, and it also engages **automatically** when the
estimated whole-cube footprint exceeds a memory budget (default 1 GB, an advanced
parameter that a value of 0 disables) — but auto-selection is on by default only for
the analytics whose streamed result is byte-identical to the whole-cube result
(*Similarity*, *Change*, *Classify*); *Cluster* and *Embedding → RGB* stream only
when you ask, because their fit changes when streamed (a would-be out-of-memory run
there still yields a valid but slightly different result). Clustering fits k-means on
a sampled subset of pixels by default, so cluster
assignments are approximate for very large rasters and depend on the seed and sample
size; an opt-in silhouette sweep can suggest k, and a MiniBatchKMeans path can fit
the full raster instead of subsampling. Classification quality is entirely bounded
by the quality and representativeness of your training features — a handful of
unrepresentative polygons will give a confident-looking but wrong map; an opt-in
holdout accuracy report (overall/per-class metrics and a confusion matrix) now lets
you quantify this rather than eyeball it, but it measures fit to *your* labels, not
ground truth. A saved *Classify* model is written as a small zip container whose
provenance (band count, backend id, class list) is plain JSON, so inspecting or
validating a model never runs code; the fitted estimator inside is stored with
`skops` — reconstructed on load through a trusted-types gate limited to
scikit-learn/numpy/scipy types — when that optional package is installed, or with
pickle otherwise. Loading a pickle-backed model (the fallback when `skops` is absent,
and any model written by an older build) can execute code embedded in the file, so
load only models you trained yourself or trust. The change modes assume the input
rasters share a grid and CRS; by
default misaligned inputs are rejected, and an opt-in step will nearest-neighbour
resample onto the reference grid (itself an approximation for embeddings).

## Platform and environment

The plugin is written for QGIS 3.34 LTR and the 4.x series and uses scoped Qt/QGIS
enums so it works under both PyQt5 and PyQt6, but it has been exercised most on recent
QGIS 4.x. Installing heavy tiers from inside QGIS relies on the QGIS Python
environment being pip-installable, which is normally true but can fail in locked-down
or read-only deployments; in those cases install PyTorch and scikit-learn into the
QGIS Python environment out-of-band. Weight downloads and STAC access both require
network connectivity. User-facing strings are English only: the UI text is not
wrapped for translation (there is no `tr()` / Qt Linguist catalogue), so the plugin
is not localised in this release. Output rasters are written with the GDAL driver
inferred from the output file extension (for example `.tif` → GeoTIFF); an
unrecognised or unusual extension may fail to write or fall back to a default, so
prefer a standard extension for the format you want.

## Not in this release

Adding a backend beyond the three listed is now a documented drop-in
([ADDING_A_BACKEND.md](ADDING_A_BACKEND.md)), but no additional backends ship in
this release. There is still no active-learning or sample-refinement loop and no
packaged task-specific profiles (e.g. WUI or canopy presets). The automated
end-to-end coverage stops at the orchestration boundary: the QGIS-facing algorithm
glue (anything that imports `qgis.core`) and the real PyTorch model runs are
verified by the GDAL/QGIS-gated tests and the manual smoke test, not by the
pure-core sandbox suite. These are candidate future work — see
[ROADMAP.md](ROADMAP.md).
