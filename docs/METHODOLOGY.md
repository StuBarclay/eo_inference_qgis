# Methodology

This document explains *what the toolbox computes and why* — the reasoning behind
each step, the choices that matter for correctness, and the assumptions a user
should be aware of when interpreting the outputs.

## Embeddings from inference, not from an archive

An Earth-observation *embedding* is a compact vector that summarises what a model has
learned about a patch of the Earth's surface — its land cover, texture, spectral
signature and context — in a form where geometric closeness means semantic
similarity. Some products (Google DeepMind's AlphaEarth is the best-known) ship these
as a **pre-computed global archive** you simply download.

The three models here work differently: they are **inference backends**. There is no
global archive to fetch. You supply source imagery — a local GeoTIFF, or a scene
pulled from a public STAC catalogue — and the model computes embeddings over *that
imagery* on the fly. This is more flexible (any AOI, any date, any supported sensor)
and it is the only option for these particular models, but it means the quality and
comparability of your embeddings depend on the imagery you feed in.

## The pipeline

The seven algorithms compose into a straightforward pipeline. You **fetch or supply
imagery**, **embed** it into a per-pixel embedding raster, and then explore that
raster with **RGB preview**, **similarity search**, **change/trajectory** analysis,
and supervised **classification** or unsupervised **clustering**. Only the embed step
needs PyTorch; everything downstream operates on the embedding raster with NumPy (and
scikit-learn for classify/cluster).

## Step 1 — Imagery acquisition

Imagery can come from disk or from public STAC. The *Fetch imagery* algorithm queries
one of two no-friction catalogues depending on the chosen backend's default
collection: **Sentinel-2 L2A** via Element 84's Earth Search
(`earth-search.aws.element84.com`, no authentication), or **Harmonized
Landsat-Sentinel (HLS) V2** via Microsoft's Planetary Computer
(`planetarycomputer.microsoft.com`), which requires request signing that the fetch
layer applies automatically. The algorithm selects the least-cloudy scene in the
date window (a simple, transparent policy), resolves the backend's required band
assets, and reads them through GDAL's `/vsicurl` so only the AOI is transferred
rather than whole scenes.

The band selection is backend-driven. Clay and Satlas default to Sentinel-2
Red/Green/Blue/NIR; Prithvi requires its fixed six-band set (Blue, Green, Red,
narrow-NIR, SWIR1, SWIR2) at 30 m, which is why its default collection is HLS rather
than S2. You can override the band list, but the backend's expectations still govern
what a run will actually accept. For Satlas the fetch does not pull raw reflectance
bands at all: it retrieves Sentinel-2's pre-rendered 8-bit true-colour asset (the
`visual` / TCI product, red/green/blue in `0–255`) in a single warp, because the
default `Sentinel2_SwinB_SI_RGB` checkpoint was trained on exactly that true-colour
input rather than on surface reflectance.

## Step 2 — Embedding

The *Embed* algorithm is the only step that runs a model. It reads the imagery cube,
tiles it into windows (default 256 px with 32 px overlap), and hands each tile to the
selected backend's `embed_tile`. The backend upsamples its native patch/token grid
back to pixel resolution internally, so the result is a dense per-pixel embedding at
the input grid. Overlapping tiles are averaged by the `MosaicAccumulator` to avoid
seams at tile boundaries.

Three methodological points matter here. **Metadata-awareness:** Clay consumes tile
centre lat/lon, acquisition time and ground sample distance alongside the pixels
(`TileMeta`), so supplying an accurate GSD and geolocation improves its output;
Prithvi and Satlas largely ignore these. **Input standardisation:** each backend
expects imagery in the units its weights were trained on. Clay standardises every
band to `(DN − mean) / std` using the per-sensor statistics published with the model
(vendored as `clay_assets/metadata.yaml`), exactly as Clay's own embedding pipeline
does; Prithvi likewise standardises each band to `(DN − mean) / std` using the HLS
statistics published in its `config.json`, and Satlas expects 8-bit true colour.
An *Input scale* override lets you pre-convert imagery that is not already in the
expected units (for Clay and Prithvi, a multiplier into raw DN before standardisation).
**Normalisation:** all three backends emit non–unit-normalised vectors, so the embed
step **L2-normalises every pixel vector before writing**. This is not cosmetic — it is what makes the cosine-similarity =
dot-product identity used downstream exactly true. The written raster is stamped with
the backend id (`MD_BACKEND`) and a unit-normalisation flag (`MD_UNIT`) so downstream
steps know its provenance.

## Step 3 — Visualising embeddings (RGB)

A high-dimensional embedding cannot be viewed directly. The *Embedding → RGB*
algorithm reduces it to three display bands by one of two methods: **PCA**, which
fits a 3-component principal-component projection on a sample of pixels and maps the
top components to R/G/B (best for revealing overall structure), or **bands**, which
simply picks a chosen triplet of embedding dimensions. Output is stretched to 8-bit
using a low/high percentile clip (defaults 2 % / 98 %) so a few outliers do not
wash out the image. This is a visualisation aid only; it discards most of the
embedding information and should never be used as the input to analysis.

## Step 4 — Similarity search

Given an embedding raster and one or more reference points, *Similarity to reference*
builds a reference vector (the L2-normalised mean of the embeddings at those points)
and computes a per-pixel **cosine similarity** map against it. Because the raster is
already unit-normalised, this is a single dot product per pixel: fast and exact.
Values run from −1 (opposite) through 0 (unrelated) to 1 (identical). Picking several
points that all sample the same target class gives a more robust reference than a
single pixel.

## Step 5 — Change and trajectory

*Change / trajectory* compares embeddings across a stack of dated rasters (or two
dates) under a chosen metric — **cosine** distance (1 − cosine similarity) or
**euclidean** distance. Three modes are offered. **Pairwise** measures the distance
between two dates directly, the natural "what changed between then and now" map.
**Trajectory** sums consecutive-step distances across a multi-date stack, capturing
total movement through embedding space even when the endpoints happen to look
similar. **Anomaly** measures each date's distance from a baseline mean, highlighting
departures from a norm. Cosine is usually the right default because it matches the
normalised geometry the embeddings live in; euclidean is offered for cases where
magnitude differences are meaningful.

## Step 6 — Classification and clustering

Both draw on the same reshape-mask-predict-scatter plumbing in `eocore.ml` and both
require the opt-in scikit-learn tier.

*Classify embedding* is supervised. It reads training polygons/points with a class
field, samples embedding vectors under them, fits either a **random forest** or a
**gradient boosting** classifier, and predicts a per-pixel class raster (optionally
with a confidence raster). The trained model can be saved to an envelope that records
the embedding band count and backend id; loading it later checks those match the
raster, so a model can never be silently applied to incompatible embeddings.

*Cluster embedding* is unsupervised k-means over a sampled subset of pixels (default
up to 100 000 samples for tractability), assigning every pixel to its nearest of *k*
clusters. It is the quickest way to see the natural groupings in an embedding raster
when you have no labels.

## Assumptions and their consequences

A handful of assumptions run through the whole method. Embeddings are only comparable
when produced by **the same backend** from **comparable imagery** — mixing backends,
or comparing a cloudy scene against a clear one, will show "change" that is really an
artefact of the inputs. Similarity and change treat the embedding space geometry as
meaningful, which is exactly why normalisation is enforced. And because everything
downstream of the embed step is deterministic NumPy, results are reproducible given
the same embedding raster; the embedding step itself is deterministic per backend and
weight file, but different checkpoints will give different (internally consistent)
embeddings. The known boundaries of all this are collected in
[LIMITATIONS.md](LIMITATIONS.md).
