"""Backend protocol, tile metadata, and a torch-free registry of backend facts.

Nothing in this module imports torch or a model package. The concrete backend
classes live in sibling modules and are only imported by their loaders (see
:func:`register_backend`), which run at model-load time (not import time). That
keeps ``eocore.backends`` safe to import in the test sandbox and in QGIS before
any weights exist.

Contract for a backend
----------------------
A backend turns an *imagery* tile into an *embedding* tile at the same pixel
grid:

    embed_tile(image[C_in, h, w], meta) -> embedding[D, h, w]

The caller (the ``embed`` algorithm) is responsible for windowing a large
raster with :mod:`eocore.tiling`, feeding tiles here, and mosaicking the
returned embedding tiles. A backend must therefore upsample its native
patch/token grid back to the tile's pixel resolution internally, so every
backend presents the same simple per-pixel interface.

Embeddings are **not** unit-normalised (``EmbeddingDescriptor.is_unit_normalized
= False`` for all three models); the ``embed`` algorithm L2-normalises before
writing, so downstream similarity/change can use the dot-product shortcut.

Extending the toolbox
---------------------
Backends are pluggable (E-10): a new backend is one new module (a
:class:`BackendInfo` plus a loader) and one :func:`register_backend` call — the
:func:`load_backend` dispatch never needs editing. See docs/ADDING_A_BACKEND.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..descriptor import EmbeddingDescriptor
from ..stac import HLS_S30, PRITHVI_BANDS, RGBN_BANDS, S2_L2A

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class TileMeta:
    """Per-tile context handed to a backend alongside the pixel data.

    ``bands`` are the semantic band names (see :mod:`eocore.stac`) in the same
    order as the tile's first axis. ``gsd`` is the ground sample distance in
    metres. ``lon``/``lat`` (tile centre) and ``datetime`` are optional; Clay
    uses them, the others may ignore them.
    """

    bands: tuple[str, ...]
    gsd: float
    lon: float | None = None
    lat: float | None = None
    datetime: str | None = None


@runtime_checkable
class EmbeddingBackend(Protocol):
    """The minimal interface every inference backend implements."""

    backend_id: str
    display_name: str

    def descriptor(self) -> EmbeddingDescriptor:
        """Describe the embeddings this backend produces."""
        ...

    def required_bands(self) -> tuple[str, ...]:
        """Semantic band names this backend expects, in order."""
        ...

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        """Embed one ``(C_in, h, w)`` tile to a ``(D, h, w)`` cube."""
        ...


@runtime_checkable
class BatchEmbeddingBackend(EmbeddingBackend, Protocol):
    """A backend that can embed several tiles in one forward pass (E-2).

    This is an *optional* capability layered on top of :class:`EmbeddingBackend`:
    a backend that can stack tiles into a single tensor implements
    ``embed_tiles`` so a GPU is kept busy. Backends without it are still valid
    ``EmbeddingBackend`` s — :func:`embed_tile_batch` falls back to a per-tile
    loop, so batching never changes results, only throughput.
    """

    def embed_tiles(self, images: Sequence[NDArray[Any]], meta: TileMeta) -> list[NDArray[Any]]:
        """Embed a batch of ``(C_in, h, w)`` tiles to a list of ``(D, h, w)`` cubes."""
        ...


def embed_tile_batch(
    backend: EmbeddingBackend, images: Sequence[NDArray[Any]], meta: TileMeta
) -> list[NDArray[Any]]:
    """Embed a batch of tiles, using the backend's batch path when it has one.

    A single-tile batch always takes the per-tile ``embed_tile`` route, so the
    default (batch size 1) is byte-for-byte the v0.1.0 behaviour. For larger
    batches, a backend that implements ``embed_tiles`` (see
    :class:`BatchEmbeddingBackend`) runs one stacked forward pass; any other
    backend transparently falls back to looping ``embed_tile`` (same output,
    no speed-up). All tiles in a batch share one :class:`TileMeta` (the embed
    algorithm uses a scene-centre meta for every tile).
    """
    tiles = list(images)
    if len(tiles) <= 1:
        return [backend.embed_tile(image, meta) for image in tiles]
    batch_fn = getattr(backend, "embed_tiles", None)
    if callable(batch_fn):
        results = list(batch_fn(tiles, meta))
        if len(results) != len(tiles):
            raise RuntimeError(
                f"backend {getattr(backend, 'backend_id', '?')!r} returned "
                f"{len(results)} embeddings for a batch of {len(tiles)} tiles"
            )
        return results
    return [backend.embed_tile(image, meta) for image in tiles]


@dataclass(frozen=True)
class BackendInfo:
    """Torch-free facts about a backend, for the GUI, docs and descriptors.

    ``embedding_dim`` and ``resolution_m`` are nominal hints for the default
    checkpoint; the live backend reports the true values via ``descriptor()``
    once a model is loaded.
    """

    backend_id: str
    display_name: str
    summary: str
    default_bands: tuple[str, ...]
    default_collection: str
    embedding_dim: int
    resolution_m: float
    is_unit_normalized: bool
    licence: str
    pip_packages: tuple[str, ...]
    weights_hint: str
    #: Whether this backend consumes a *fixed* band set in a fixed order. Prithvi
    #: standardises positionally against its six ``PRITHVI_BANDS`` (ignoring the
    #: tile's band names) and the Satlas RGB checkpoint takes exactly three
    #: true-colour channels, so feeding either a different band set fails deep in
    #: the model with an opaque error (or silently mis-standardises). The embed
    #: algorithm calls :func:`band_requirement_error` up front to turn that into a
    #: clear, backend-aware message. Clay is flexible — it consumes bands by
    #: semantic name with an identity fallback — and leaves this ``False``.
    strict_bands: bool = False
    #: Native pixel-scaling default, used two ways. For Satlas it is the real
    #: normaliser (8-bit true colour ÷255). For Clay/Prithvi it is only the *units
    #: heuristic* behind the embed "input scale" warning — both standardise raw
    #: DN internally with per-band (DN - mean) / std, so they do NOT rescale to
    #: [0, 1]; the ÷10000 value merely lets the warning tell DN imagery (scales
    #: near 1) from already-reflectance imagery (scales near 1e-4, flagged).
    default_input_scale: float = 1.0
    #: STAC asset key of a pre-rendered *multi-band* asset to fetch instead of
    #: resolving one single-band COG per semantic band. Satlas sets this to
    #: ``"visual"`` so the Fetch-imagery algorithm pulls Sentinel-2's 8-bit
    #: true-colour (TCI) COG — the exact input the RGB checkpoint was trained on
    #: (red/green/blue, 0-255), so the default embed lands in ~[0, 1] with the
    #: ÷255 ``default_input_scale`` and no manual "Input scale" step. ``None``
    #: (the default, used by Clay/Prithvi) keeps the per-band reflectance path.
    fetch_asset: str | None = None
    import_names: tuple[str, ...] = field(default_factory=tuple)
    #: Expected SHA-256 of the primary downloaded checkpoint, or ``None`` if not
    #: yet recorded. When set, the download is verified and a mismatch fails loud
    #: (E-5 / H-8, see :func:`_weights.verify_sha256`). Left ``None`` by default so
    #: verification is opt-in and never rejects a good download against an unknown
    #: hash; the wizard's "verify weights" action prints the computed digest so a
    #: user (or a future release) can lock it in here.
    weights_sha256: str | None = None


# -- pluggable backend registry (E-10) ---------------------------------------
#: A lazy backend loader: ``(**config) -> EmbeddingBackend``. A loader imports
#: torch and its model package only when *called*, so registration — and this
#: whole module — stays torch-free.
BackendLoader = Callable[..., EmbeddingBackend]

#: Torch-free registry of backend *facts*, populated by :func:`register_backend`.
BACKENDS: dict[str, BackendInfo] = {}
#: Parallel registry of lazy loaders, keyed by backend id.
_BACKEND_LOADERS: dict[str, BackendLoader] = {}


def register_backend(info: BackendInfo, loader: BackendLoader, *, replace: bool = False) -> None:
    """Register a backend's facts and lazy loader — the single extension point (E-10).

    Adding a backend is one new module (defining a :class:`BackendInfo` and a
    loader function that imports torch / the model package only when *called*)
    plus one ``register_backend`` call; :func:`load_backend` consults this table,
    so the dispatch never needs editing. Registration stays torch-free because
    the loader defers its heavy imports to call time. Raises :class:`ValueError`
    on a duplicate id unless ``replace=True``. See docs/ADDING_A_BACKEND.md.
    """
    backend_id = info.backend_id
    if backend_id in _BACKEND_LOADERS and not replace:
        raise ValueError(
            f"backend {backend_id!r} is already registered; pass replace=True to override it"
        )
    BACKENDS[backend_id] = info
    _BACKEND_LOADERS[backend_id] = loader


def registered_backends() -> tuple[str, ...]:
    """Ids of all currently registered backends, in registration order."""
    return tuple(_BACKEND_LOADERS)


def backend_info(backend_id: str) -> BackendInfo:
    """Return the :class:`BackendInfo` for ``backend_id`` (raises on unknown)."""
    try:
        return BACKENDS[backend_id]
    except KeyError:
        raise KeyError(
            f"unknown backend {backend_id!r}; expected one of {registered_backends()}"
        ) from None


def descriptor_from_info(info: BackendInfo, *, n_bands: int | None = None) -> EmbeddingDescriptor:
    """Build an :class:`EmbeddingDescriptor` from a backend's registry facts."""
    from ..descriptor import BackendCapabilities

    return EmbeddingDescriptor(
        backend_id=info.backend_id,
        n_bands=int(n_bands if n_bands is not None else info.embedding_dim),
        is_unit_normalized=info.is_unit_normalized,
        native_dtype="float32",
        resolution_m=info.resolution_m,
        temporal=info.backend_id == "prithvi",
        capabilities=BackendCapabilities(
            needs_gpu=False,
            needs_auth=info.default_collection.startswith("hls"),
            extra_deps=info.pip_packages,
        ),
    )


def load_backend(backend_id: str, **config: Any) -> EmbeddingBackend:
    """Instantiate a registered backend by id. Loaders import torch lazily, at call time."""
    try:
        loader = _BACKEND_LOADERS[backend_id]
    except KeyError:
        raise KeyError(
            f"unknown backend {backend_id!r}; expected one of {registered_backends()}"
        ) from None
    return loader(**config)


# -- defensive model-API guards (H-6) ----------------------------------------
def unsupported_model_error(backend_id: str, detail: str) -> RuntimeError:
    """Build a uniform, actionable error for an unexpected model API (H-6).

    Backends reach into external checkpoints / libraries whose structure can
    drift between versions. When an expected attribute or output is missing,
    raise this instead of letting an opaque ``AttributeError`` / ``IndexError``
    surface only at a user's first embed run.
    """
    return RuntimeError(
        f"The {backend_id} backend cannot use the loaded model: {detail}. This "
        "usually means the installed checkpoint or model-library version differs "
        "from the one the plugin expects. Pin a known-good version (see the "
        "backend's install hint) and record the verified library + checkpoint "
        "versions in docs/SMOKE_TEST.md."
    )


def require_model_attr(obj: Any, name: str, backend_id: str, detail: str) -> Any:
    """Return ``getattr(obj, name)``; raise :func:`unsupported_model_error` if absent/None (H-6)."""
    value = getattr(obj, name, None)
    if value is None:
        raise unsupported_model_error(backend_id, detail)
    return value


def select_feature(output: Any, backend_id: str, *, index: int) -> Any:
    """Pick a feature tensor from a backbone's output (H-6).

    A backbone may return a single tensor or a (non-empty) sequence of hidden
    states; in the sequence case take ``output[index]``. An empty sequence means
    the model produced nothing usable — fail loud rather than ``IndexError``.
    """
    if isinstance(output, (tuple, list)):
        if not output:
            raise unsupported_model_error(backend_id, "the model returned no feature tensors")
        return output[index]
    return output


# -- built-in backends -------------------------------------------------------
# Each loader imports its module (and torch) only when called, so these three
# registrations keep this module torch-free at import.
def _load_clay(**config: Any) -> EmbeddingBackend:
    from .clay import ClayBackend

    return ClayBackend(**config)


def _load_prithvi(**config: Any) -> EmbeddingBackend:
    from .prithvi import PrithviBackend

    return PrithviBackend(**config)


def _load_satlas(**config: Any) -> EmbeddingBackend:
    from .satlas import SatlasBackend

    return SatlasBackend(**config)


register_backend(
    BackendInfo(
        backend_id="clay",
        display_name="Clay v1.5",
        summary=(
            "Multi-sensor ViT masked-autoencoder (Clay Foundation). Cleanest, "
            "best-documented backend; a good default. Consumes metadata (lat/lon, "
            "time, GSD) alongside pixels."
        ),
        default_bands=RGBN_BANDS,
        default_collection=S2_L2A,
        embedding_dim=768,
        resolution_m=10.0,
        is_unit_normalized=False,
        licence="Apache-2.0 (code and weights)",
        pip_packages=("torch", "git+https://github.com/Clay-foundation/model"),
        weights_hint="Hugging Face made-with-clay/Clay -> v1.5/clay-v1.5.ckpt (~5.2 GB)",
        # Clay standardises pixels internally with per-band (DN - mean) / std from
        # its vendored metadata, so it does NOT rescale imagery to [0, 1]. This
        # value is kept only as the units heuristic for the embed "input scale"
        # warning: raw Sentinel-2 DN x 1e-4 lands near 1 (passes the check), while
        # already-reflectance imagery scales to ~1e-4 and is flagged as mis-scaled.
        default_input_scale=1.0 / 10000.0,
        # The Clay pip package installs a single top-level ``claymodel`` package
        # (its pyproject only finds ``claymodel*``); the runtime import used by
        # this backend is ``from claymodel.module import ClayMAEModule``. There is
        # no importable top-level ``clay`` module, so it must NOT be listed here —
        # the status badge (and missing_import_names) require *all* names to be
        # importable, and an extra ``clay`` would keep Clay red even when a correct
        # install is present.
        import_names=("claymodel",),
    ),
    _load_clay,
)
register_backend(
    BackendInfo(
        backend_id="prithvi",
        display_name="Prithvi-EO-2.0 (IBM + NASA)",
        summary=(
            "Multi-temporal ViT MAE trained on HLS. Fixed six-band input "
            "(Blue, Green, Red, narrow-NIR, SWIR1, SWIR2) at 30 m. 300M / 600M "
            "variants (plus -TL temporal-location variants). Uses TerraTorch."
        ),
        default_bands=PRITHVI_BANDS,
        default_collection=HLS_S30,
        # Prithvi standardises positionally against PRITHVI_BANDS and hard-checks
        # the band count, so it needs exactly these six bands in this order.
        strict_bands=True,
        embedding_dim=1024,
        resolution_m=30.0,
        is_unit_normalized=False,
        licence="Apache-2.0",
        pip_packages=("torch", "terratorch"),
        weights_hint="Hugging Face ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        # Like Clay, Prithvi standardises pixels internally with per-band
        # (DN - mean) / std from its published config.json stats, so it does NOT
        # rescale imagery to [0, 1]. This value is kept only as the units heuristic
        # for the embed "input scale" warning: raw HLS DN x 1e-4 lands near 1
        # (passes), while already-reflectance imagery scales to ~1e-4 and is flagged.
        default_input_scale=1.0e-4,
        import_names=("terratorch",),
    ),
    _load_prithvi,
)
register_backend(
    BackendInfo(
        backend_id="satlas",
        display_name="SatlasPretrain (Allen Institute / AI2)",
        summary=(
            "Swin-v2 (or ResNet) backbone + FPN pretrained on Sentinel-2/1 and "
            "NAIP. Emits multi-scale feature maps that this backend pools to a "
            "single per-pixel vector. Single- and multi-image variants."
        ),
        default_bands=(RGBN_BANDS[0], RGBN_BANDS[1], RGBN_BANDS[2]),
        default_collection=S2_L2A,
        # Fetch Sentinel-2's pre-rendered 8-bit true-colour COG (the "visual"/TCI
        # asset: red/green/blue, 0-255) rather than raw reflectance bands, so the
        # fetched imagery is exactly what the RGB checkpoint was trained on and the
        # embed default (x 1/255) lands in ~[0, 1] out of the box, with no manual
        # "Input scale". (Raw reflectance fed here scales to ~36 and only warns.)
        fetch_asset="visual",
        # The default Sentinel2_SwinB_SI_RGB checkpoint takes exactly three
        # true-colour channels; a different band set fails opaquely in torch.
        strict_bands=True,
        # Nominal hint only. With the default concat fusion over the SwinB FPN, the
        # per-pixel vector is the sum of the FPN level channels — four 128-channel
        # levels = 512 (see the satlaspretrain README shapes), not the backbone's
        # deepest 1024-channel stage. The live descriptor reports the true fused
        # width once a model is loaded (and 'sum' fusion would differ again).
        embedding_dim=512,
        resolution_m=10.0,
        is_unit_normalized=False,
        licence="ODC-BY (weights); code Apache-2.0",
        pip_packages=("torch", "satlaspretrain-models"),
        weights_hint="checkpoint id, e.g. Sentinel2_SwinB_SI_RGB (auto-downloaded)",
        default_input_scale=1.0 / 255.0,
        import_names=("satlaspretrain_models",),
    ),
    _load_satlas,
)

#: Snapshot of the built-in backend ids (registration order) — the fixed set the
#: bundled GUI and Processing enum are built from. Third-party backends that call
#: :func:`register_backend` at import time are still reachable via
#: :func:`load_backend` and :func:`registered_backends`.
BACKEND_IDS: tuple[str, ...] = registered_backends()


def resolve_bands(backend_id: str, requested: Sequence[str] | None) -> tuple[str, ...]:
    """Return ``requested`` bands if given, else the backend's default set."""
    if requested:
        return tuple(requested)
    return backend_info(backend_id).default_bands


def _suggest_backend_for_bands(bands: tuple[str, ...], *, exclude: str) -> str | None:
    """Display name of a registered backend whose default bands are exactly ``bands``.

    Excludes ``exclude`` (the backend that was actually selected) and returns
    ``None`` when nothing matches, so :func:`band_requirement_error` can offer a
    concrete "select this one instead" pointer only when it is certain.
    """
    for backend_id, info in BACKENDS.items():
        if backend_id != exclude and tuple(info.default_bands) == bands:
            return info.display_name
    return None


def band_requirement_error(backend_id: str, resolved_bands: Sequence[str]) -> str | None:
    """Return a friendly message if a *strict* backend's fixed band set is unmet.

    Strict backends (see :attr:`BackendInfo.strict_bands`) consume a fixed band set
    in a fixed order: Prithvi standardises positionally against
    :data:`~eocore.stac.PRITHVI_BANDS`, and the Satlas RGB checkpoint takes exactly
    three true-colour channels. When the bands resolved for the imagery — from a
    band-order override, the GeoTIFF's stamped band names, or the backend default —
    are not exactly the backend's set, embedding would otherwise fail deep in the
    model with an opaque error (e.g. "Prithvi needs exactly 6 bands ..., got 3") or,
    worse, silently mis-standardise. This returns an up-front message naming the
    imagery's bands, the backend's requirement (in order), and — when one matches
    exactly — the backend whose defaults fit the imagery. Returns ``None`` when the
    bands are compatible or the backend is flexible (Clay consumes bands by name).
    """
    info = backend_info(backend_id)
    resolved = tuple(resolved_bands)
    required = tuple(info.default_bands)
    if not info.strict_bands or resolved == required:
        return None
    imagery = ", ".join(resolved) if resolved else "no named bands"
    needed = ", ".join(required)
    message = (
        f"This imagery has {len(resolved)} band(s) ({imagery}), but the "
        f"{info.display_name} backend needs exactly {len(required)} band(s) "
        f"({needed}) in that order."
    )
    suggestion = _suggest_backend_for_bands(resolved, exclude=backend_id)
    if suggestion is not None:
        message += (
            f" Those bands match the {suggestion} backend — select it instead, or "
            f"re-fetch this area with {info.display_name} first."
        )
    else:
        message += (
            f" Set the band-order override to '{needed}', or re-fetch this area "
            f"with {info.display_name} first."
        )
    return message


#: A model input is expected to land roughly in ``[0, PLAUSIBLE_INPUT_MAX]`` after
#: scaling. Scaled maxima far outside this band signal wrong input units.
PLAUSIBLE_INPUT_MAX = 1.5
#: Below this scaled maximum the imagery is almost certainly mis-scaled.
MIN_SCALED_MAX = 0.05


def scale_sanity_warning(data_max: float, effective_scale: float) -> str | None:
    """Warn if scaling ``data_max`` lands far outside the model's input band.

    Backends assume a particular input encoding (surface-reflectance DN for
    Clay/Prithvi, 8-bit for Satlas) and multiply raw pixels by ``effective_scale``
    to reach roughly ``[0, 1]``. If the scaled maximum is far below or above that
    band, the imagery is probably in the wrong units (already-reflectance, 8-bit
    fed to a DN backend, etc.) — a silent-accuracy failure worth flagging. Returns
    the warning text, or ``None`` when the range looks plausible.
    """
    if not math.isfinite(data_max) or data_max <= 0 or effective_scale <= 0:
        return None
    scaled = data_max * effective_scale
    if scaled < MIN_SCALED_MAX:
        return (
            f"Input pixels scale to a maximum of {scaled:.3g} (raw max {data_max:.4g} "
            f"x scale {effective_scale:.3g}), far below the ~[0, 1] range the model "
            "expects. The imagery may already be reflectance or 8-bit; set the "
            "'Input scale' parameter to match your data."
        )
    if scaled > 10.0 * PLAUSIBLE_INPUT_MAX:
        return (
            f"Input pixels scale to a maximum of {scaled:.3g} (raw max {data_max:.4g} "
            f"x scale {effective_scale:.3g}), far above the ~[0, 1] range the model "
            "expects. Check the 'Input scale' parameter for your imagery."
        )
    return None
