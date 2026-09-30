"""Resolve STAC asset hrefs into readable URLs before GDAL warps them.

Earth Search (Sentinel-2 L2A) serves plain, anonymously-readable COG hrefs, so
nothing is needed there. The Microsoft Planetary Computer (used for HLS) hands
back hrefs that must be *signed* with a short-lived token before ``/vsicurl``
can read them; if the ``planetary_computer`` package is installed we sign, and
otherwise we return the href unchanged and let the caller warn.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..eocore.backends._deps import module_available


def _planetary_computer_available() -> bool:
    """True if the ``planetary_computer`` signer is importable (shared detector)."""
    return module_available("planetary_computer")


def is_planetary_computer(href: str) -> bool:
    """True if an href points at Microsoft Planetary Computer blob storage."""
    return "blob.core.windows.net" in href or "planetarycomputer" in href


def sign_href(href: str) -> str:
    """Sign a Planetary Computer href if possible; else return it unchanged."""
    if not is_planetary_computer(href) or not _planetary_computer_available():
        return href
    import planetary_computer  # type: ignore[import-not-found]

    signed: str = planetary_computer.sign(href)
    return signed


def sign_hrefs(hrefs: Sequence[str]) -> tuple[list[str], bool]:
    """Sign a list of hrefs. Returns ``(hrefs, needed_but_missing_signer)``."""
    needs_signing = any(is_planetary_computer(h) for h in hrefs)
    missing_signer = needs_signing and not _planetary_computer_available()
    return [sign_href(h) for h in hrefs], missing_signer
