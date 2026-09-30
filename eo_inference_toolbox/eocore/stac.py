"""Build STAC searches for source imagery and parse the results.

Pure stdlib: the HTTP call goes through an injectable ``fetch`` so the query
building and item parsing are fully unit-testable with no network. Downloading
the asset hrefs themselves is GDAL's job (see ``algorithms/_stac_fetch.py``).

Sentinel-2 L2A is served without authentication by Element 84's Earth Search.
HLS is searched on the Microsoft Planetary Computer; note its asset hrefs may
need signing / an Earthdata login to actually read — that is handled (and
documented) in the fetch layer, not here.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# --- Known public STAC endpoints -------------------------------------------
EARTH_SEARCH_V1 = "https://earth-search.aws.element84.com/v1"
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# --- Collection ids ---------------------------------------------------------
S2_L2A = "sentinel-2-l2a"
HLS_L30 = "hls2-l30"
HLS_S30 = "hls2-s30"

# --- Semantic band names ----------------------------------------------------
BLUE = "blue"
GREEN = "green"
RED = "red"
NIR = "nir"
NARROW_NIR = "narrow_nir"
SWIR1 = "swir1"
SWIR2 = "swir2"

# Semantic band -> asset key, per collection.
_S2_EARTHSEARCH = {
    BLUE: "blue",
    GREEN: "green",
    RED: "red",
    NIR: "nir",
    NARROW_NIR: "nir08",
    SWIR1: "swir16",
    SWIR2: "swir22",
}
_HLS_S30 = {
    BLUE: "B02",
    GREEN: "B03",
    RED: "B04",
    NIR: "B08",
    NARROW_NIR: "B8A",
    SWIR1: "B11",
    SWIR2: "B12",
}
_HLS_L30 = {
    BLUE: "B02",
    GREEN: "B03",
    RED: "B04",
    NIR: "B05",
    SWIR1: "B06",
    SWIR2: "B07",
}

BAND_PLANS: dict[str, dict[str, str]] = {
    S2_L2A: _S2_EARTHSEARCH,
    HLS_S30: _HLS_S30,
    HLS_L30: _HLS_L30,
}

# The six bands Prithvi-EO-2.0 expects, in its fixed pre-training order.
PRITHVI_BANDS: tuple[str, ...] = (BLUE, GREEN, RED, NARROW_NIR, SWIR1, SWIR2)
# A common RGB+NIR set for Clay / Satlas when nothing else is specified.
RGBN_BANDS: tuple[str, ...] = (RED, GREEN, BLUE, NIR)

FetchJson = Callable[[str, bytes | None, Mapping[str, str]], dict[str, Any]]


#: Sentinel for an open-ended side of a datetime interval.
_OPEN_END = ".."


def _to_rfc3339(value: str, *, end: bool) -> str:
    """Promote a bare ``YYYY-MM-DD`` date to a full RFC 3339 UTC timestamp.

    STAC APIs (Earth Search / stac-fastapi, the Planetary Computer) reject a
    date-only ``datetime`` value with an HTTP 400 ("does not match RFC3339
    format"): a date must carry a time and a timezone. A start date maps to the
    first instant of the day and an end date to the last, so a whole-day range
    stays inclusive of both endpoints. Values that already look like a timestamp
    (they contain ``T``) and the open-ended sentinel ``..`` pass through
    unchanged, so callers can still supply their own precise timestamps.
    """
    text = value.strip()
    if text in ("", _OPEN_END) or "T" in text:
        return text
    return f"{text}T23:59:59Z" if end else f"{text}T00:00:00Z"


def datetime_interval(start: str | None, end: str | None) -> str | None:
    """Compose an RFC 3339 interval string from optional endpoints.

    Bare ``YYYY-MM-DD`` endpoints are promoted to full RFC 3339 timestamps (see
    :func:`_to_rfc3339`) because STAC APIs reject date-only values with an HTTP
    400. Already-timestamped endpoints are left untouched.
    """
    lo = _to_rfc3339(start, end=False) if start else None
    hi = _to_rfc3339(end, end=True) if end else None
    if lo and hi:
        return f"{lo}/{hi}"
    if lo:
        return f"{lo}/{_OPEN_END}"
    if hi:
        return f"{_OPEN_END}/{hi}"
    return None


@dataclass(frozen=True)
class StacSearch:
    """An immutable STAC ItemSearch request."""

    endpoint: str
    collections: tuple[str, ...]
    bbox: tuple[float, ...]
    datetime: str | None = None
    limit: int = 20
    query: Mapping[str, Any] | None = None

    def url(self) -> str:
        return self.endpoint.rstrip("/") + "/search"

    def body(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "collections": list(self.collections),
            "bbox": list(self.bbox),
            "limit": int(self.limit),
        }
        if self.datetime:
            payload["datetime"] = self.datetime
        if self.query:
            payload["query"] = dict(self.query)
        return payload


@dataclass(frozen=True)
class StacItem:
    """A parsed STAC item: just what the toolbox needs to fetch bands."""

    id: str
    collection: str
    datetime: str | None
    bbox: tuple[float, ...] | None
    cloud_cover: float | None
    assets: dict[str, str]
    properties: Mapping[str, Any] = field(default_factory=dict)


def parse_items(feature_collection: Mapping[str, Any]) -> list[StacItem]:
    """Parse a STAC FeatureCollection dict into a list of :class:`StacItem`."""
    items: list[StacItem] = []
    for feat in feature_collection.get("features", []) or []:
        if not isinstance(feat, Mapping):
            continue
        props = feat.get("properties", {}) or {}
        assets_raw = feat.get("assets", {}) or {}
        assets = {
            key: str(val.get("href", ""))
            for key, val in assets_raw.items()
            if isinstance(val, Mapping)
        }
        cloud = props.get("eo:cloud_cover")
        raw_bbox = feat.get("bbox")
        bbox = tuple(float(v) for v in raw_bbox) if isinstance(raw_bbox, list) else None
        items.append(
            StacItem(
                id=str(feat.get("id", "")),
                collection=str(feat.get("collection", "") or props.get("collection", "")),
                datetime=props.get("datetime"),
                bbox=bbox,
                cloud_cover=float(cloud) if isinstance(cloud, (int, float)) else None,
                assets=assets,
                properties=props,
            )
        )
    return items


def least_cloudy(items: Sequence[StacItem]) -> StacItem | None:
    """Return the item with the lowest cloud cover (unknown cover sorts last)."""
    if not items:
        return None

    def key(item: StacItem) -> float:
        return item.cloud_cover if item.cloud_cover is not None else float("inf")

    return min(items, key=key)


def hrefs_for_bands(item: StacItem, bands: Sequence[str]) -> list[str]:
    """Resolve the asset hrefs for ``bands`` on ``item`` via its collection plan."""
    plan = BAND_PLANS.get(item.collection)
    if plan is None:
        raise KeyError(f"no band plan registered for collection {item.collection!r}")
    hrefs: list[str] = []
    for band in bands:
        asset_key = plan.get(band)
        if asset_key is None:
            raise KeyError(f"band {band!r} not mapped for {item.collection!r}")
        href = item.assets.get(asset_key)
        if not href:
            raise KeyError(f"item {item.id!r} has no asset {asset_key!r} for band {band!r}")
        hrefs.append(href)
    return hrefs


def href_for_asset(item: StacItem, asset_key: str) -> str:
    """Return the href of a single named asset on ``item`` (raises if absent).

    Unlike :func:`hrefs_for_bands` (which resolves one *single-band* COG per
    semantic band through the collection's band plan), this looks up one asset
    by its STAC key — used to fetch a pre-rendered *multi-band* asset such as
    Sentinel-2's 8-bit ``visual`` (TCI) true-colour COG for the Satlas RGB
    checkpoint (see :attr:`~eo_inference_toolbox.eocore.backends.base.BackendInfo.fetch_asset`).
    """
    href = item.assets.get(asset_key)
    if not href:
        raise KeyError(f"item {item.id!r} has no asset {asset_key!r}")
    return href


def _cloud_query(max_cloud: float | None) -> dict[str, Any] | None:
    if max_cloud is None:
        return None
    return {"eo:cloud_cover": {"lt": float(max_cloud)}}


def sentinel2_search(
    bbox: Sequence[float],
    *,
    start: str | None = None,
    end: str | None = None,
    max_cloud: float | None = None,
    limit: int = 20,
    endpoint: str = EARTH_SEARCH_V1,
) -> StacSearch:
    """Build a Sentinel-2 L2A search (Earth Search, no auth)."""
    return StacSearch(
        endpoint=endpoint,
        collections=(S2_L2A,),
        bbox=tuple(float(v) for v in bbox),
        datetime=datetime_interval(start, end),
        limit=limit,
        query=_cloud_query(max_cloud),
    )


def hls_search(
    bbox: Sequence[float],
    *,
    sensor: str = "S30",
    start: str | None = None,
    end: str | None = None,
    max_cloud: float | None = None,
    limit: int = 20,
    endpoint: str = MPC_STAC,
) -> StacSearch:
    """Build an HLS search (``sensor`` = ``"S30"`` Sentinel-derived or ``"L30"``)."""
    collection = HLS_S30 if sensor.upper() == "S30" else HLS_L30
    return StacSearch(
        endpoint=endpoint,
        collections=(collection,),
        bbox=tuple(float(v) for v in bbox),
        datetime=datetime_interval(start, end),
        limit=limit,
        query=_cloud_query(max_cloud),
    )


def default_fetch_json(url: str, data: bytes | None, headers: Mapping[str, str]) -> dict[str, Any]:
    """Minimal stdlib POST/GET returning parsed JSON (documented STAC endpoints)."""
    method = "POST" if data is not None else "GET"
    request = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8")
    parsed: dict[str, Any] = json.loads(raw)
    return parsed


def search(spec: StacSearch, *, fetch: FetchJson | None = None) -> list[StacItem]:
    """Run a STAC search and return parsed items. ``fetch`` is injectable."""
    fetch_fn = fetch if fetch is not None else default_fetch_json
    body = json.dumps(spec.body()).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/geo+json"}
    result = fetch_fn(spec.url(), body, headers)
    return parse_items(result)
