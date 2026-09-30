"""Tests for STAC query building and result parsing (fetch is injected)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from eo_inference_toolbox.eocore.stac import (
    HLS_S30,
    S2_L2A,
    StacItem,
    datetime_interval,
    hls_search,
    href_for_asset,
    hrefs_for_bands,
    least_cloudy,
    parse_items,
    search,
    sentinel2_search,
)


def test_datetime_interval_variants() -> None:
    # Bare dates are promoted to full RFC 3339 timestamps (start->00:00:00Z,
    # end->23:59:59Z) so Earth Search / stac-fastapi does not reject them (400).
    assert (
        datetime_interval("2020-01-01", "2020-02-01") == "2020-01-01T00:00:00Z/2020-02-01T23:59:59Z"
    )
    assert datetime_interval("2020-01-01", None) == "2020-01-01T00:00:00Z/.."
    assert datetime_interval(None, "2020-02-01") == "../2020-02-01T23:59:59Z"
    assert datetime_interval(None, None) is None


def test_datetime_interval_rejects_date_only_regression() -> None:
    """Regression: Earth Search returns HTTP 400 for a date-only datetime.

    The interval must be RFC 3339 (carry a time + timezone), and an endpoint
    that already includes a time must be passed through verbatim.
    """
    # No bare ``YYYY-MM-DD`` token survives into the interval.
    interval = datetime_interval("2025-01-01", "2025-03-31")
    assert interval == "2025-01-01T00:00:00Z/2025-03-31T23:59:59Z"
    assert "T" in interval.split("/", 1)[0]
    # Already-timestamped endpoints are left untouched.
    assert (
        datetime_interval("2025-01-01T06:30:00Z", "2025-03-31T12:00:00Z")
        == "2025-01-01T06:30:00Z/2025-03-31T12:00:00Z"
    )


def test_sentinel2_search_body_and_url() -> None:
    spec = sentinel2_search([150.0, -34.0, 151.0, -33.0], start="2020-01-01", max_cloud=20)
    assert spec.url().endswith("/search")
    body = spec.body()
    assert body["collections"] == [S2_L2A]
    assert body["bbox"] == [150.0, -34.0, 151.0, -33.0]
    assert body["datetime"] == "2020-01-01T00:00:00Z/.."
    assert body["query"] == {"eo:cloud_cover": {"lt": 20.0}}


def test_hls_search_selects_collection() -> None:
    spec = hls_search([0.0, 0.0, 1.0, 1.0], sensor="S30")
    assert spec.collections == (HLS_S30,)


def _feature_collection() -> dict[str, Any]:
    return {
        "features": [
            {
                "id": "scene-cloudy",
                "collection": S2_L2A,
                "bbox": [150.0, -34.0, 151.0, -33.0],
                "properties": {"datetime": "2020-01-05", "eo:cloud_cover": 42.0},
                "assets": {"red": {"href": "https://x/red.tif"}},
            },
            {
                "id": "scene-clear",
                "collection": S2_L2A,
                "properties": {"datetime": "2020-01-10", "eo:cloud_cover": 3.0},
                "assets": {
                    "red": {"href": "https://x/clear_red.tif"},
                    "green": {"href": "https://x/clear_green.tif"},
                    "blue": {"href": "https://x/clear_blue.tif"},
                    "nir": {"href": "https://x/clear_nir.tif"},
                    "visual": {"href": "https://x/clear_visual.tif"},
                },
            },
        ]
    }


def test_parse_items_reads_fields() -> None:
    items = parse_items(_feature_collection())
    assert [it.id for it in items] == ["scene-cloudy", "scene-clear"]
    assert items[0].cloud_cover == 42.0
    assert items[0].assets["red"] == "https://x/red.tif"


def test_least_cloudy_picks_lowest() -> None:
    items = parse_items(_feature_collection())
    best = least_cloudy(items)
    assert best is not None
    assert best.id == "scene-clear"


def test_least_cloudy_empty_is_none() -> None:
    assert least_cloudy([]) is None


def test_hrefs_for_bands_resolves_plan() -> None:
    items = parse_items(_feature_collection())
    hrefs = hrefs_for_bands(items[1], ["red", "green", "blue", "nir"])
    assert hrefs == [
        "https://x/clear_red.tif",
        "https://x/clear_green.tif",
        "https://x/clear_blue.tif",
        "https://x/clear_nir.tif",
    ]


def test_hrefs_for_bands_missing_asset_raises() -> None:
    items = parse_items(_feature_collection())
    with pytest.raises(KeyError):
        hrefs_for_bands(items[0], ["nir"])


def test_hrefs_for_bands_unknown_collection_raises() -> None:
    item = StacItem(
        id="x",
        collection="mystery",
        datetime=None,
        bbox=None,
        cloud_cover=None,
        assets={},
    )
    with pytest.raises(KeyError, match="band plan"):
        hrefs_for_bands(item, ["red"])


def test_href_for_asset_resolves_named_asset() -> None:
    # scene-clear carries a pre-rendered 'visual' (TCI) asset; the resolver looks
    # it up by STAC key (unlike hrefs_for_bands, which maps semantic band -> asset).
    items = parse_items(_feature_collection())
    assert href_for_asset(items[1], "visual") == "https://x/clear_visual.tif"


def test_href_for_asset_missing_asset_raises() -> None:
    # scene-cloudy only has a 'red' asset, so a 'visual' lookup must fail loudly.
    items = parse_items(_feature_collection())
    with pytest.raises(KeyError, match="visual"):
        href_for_asset(items[0], "visual")


def test_search_uses_injected_fetch() -> None:
    captured: dict[str, Any] = {}

    def fake_fetch(url: str, data: bytes | None, headers: Mapping[str, str]) -> dict[str, Any]:
        captured["url"] = url
        captured["data"] = data
        return _feature_collection()

    spec = sentinel2_search([150.0, -34.0, 151.0, -33.0])
    items = search(spec, fetch=fake_fetch)
    assert captured["url"].endswith("/search")
    assert captured["data"] is not None  # POST body was sent
    assert len(items) == 2
