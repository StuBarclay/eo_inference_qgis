"""Resolve and cache model weights on demand. Weights are never bundled.

Checkpoints are large and separately licensed, so they are downloaded the first
time a backend runs and cached under a user directory (override with the
``EO_INFER_WEIGHTS`` environment variable). Prithvi (TerraTorch) and Satlas
(satlaspretrain-models) fetch their own weights from Hugging Face; for those
this module only provides the cache directory. Clay's checkpoint is fetched
explicitly here.

Integrity and progress (E-5 / H-8)
----------------------------------
A download can be verified against an expected SHA-256 (:func:`verify_sha256`),
failing loud on mismatch. Verification is *opt-in*: it only runs when a digest is
actually supplied (the registry records ``None`` until a known-good hash is
locked in), so a good download is never rejected against an unknown hash. The
stdlib download path reads ``Content-Length`` and reports byte progress through
an injected callback so a UI progress bar can move; :func:`cache_size_bytes` and
friends let the wizard show what is already cached.

Everything here is stdlib-only (no torch, GDAL or QGIS), so it imports and is
unit-testable in the sandbox.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

ENV_CACHE = "EO_INFER_WEIGHTS"

#: Progress callback: ``(bytes_done, total_bytes_or_None) -> None``. ``total`` is
#: ``None`` when the server sends no ``Content-Length``.
ProgressFn = Callable[[int, "int | None"], None]

#: Injectable stream-download hook: ``(url, dest, progress) -> None``.
DownloadFn = Callable[[str, Path, "ProgressFn | None"], None]

_CHUNK = 1 << 20  # 1 MiB streaming/hashing block.


class ChecksumError(RuntimeError):
    """Raised when a downloaded weight file fails its SHA-256 check (H-8)."""


def default_cache_dir() -> Path:
    """Weights cache directory (``EO_INFER_WEIGHTS`` or a per-user default)."""
    override = os.environ.get(ENV_CACHE)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "eo_inference_toolbox" / "weights"


def ensure_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the cache dir, creating it if needed."""
    path = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


# -- integrity ---------------------------------------------------------------
def sha256_file(path: str | Path, *, chunk_size: int = _CHUNK) -> str:
    """Return the hex SHA-256 of a file, read in bounded chunks (never loads it whole)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str | None) -> None:
    """Verify ``path`` against ``expected`` SHA-256; raise :class:`ChecksumError` on mismatch.

    Opt-in and default-safe: when ``expected`` is falsy (``None``/empty) this is a
    no-op, so verification never rejects a good download against an unrecorded
    hash. The comparison is case-insensitive.
    """
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != str(expected).strip().lower():
        name = Path(path).name
        raise ChecksumError(
            f"checksum mismatch for {name}: expected SHA-256 {expected}, got {actual}. "
            "The download may be corrupt or the recorded checksum is stale — delete the "
            "file and retry, or correct the expected SHA-256."
        )


def _content_length(response: object) -> int | None:
    """Best-effort ``Content-Length`` (bytes) from an ``urlopen`` response, or ``None``."""
    headers = getattr(response, "headers", None)
    raw = headers.get("Content-Length") if headers is not None else None
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def default_download(url: str, dest: Path, progress: ProgressFn | None = None) -> None:
    """Stream a URL to ``dest`` via stdlib urllib, reporting byte progress.

    Reads ``Content-Length`` (when present) and calls ``progress(done, total)``
    for each block plus once up front — so a UI bar can move during the download
    (E-5). Writes to a ``.part`` sidecar and atomically renames on success.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "eo-inference-toolbox"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        total = _content_length(response)
        done = 0
        if progress is not None:
            progress(0, total)
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    tmp.replace(dest)


def fetch_url(
    url: str,
    filename: str,
    *,
    cache_dir: str | Path | None = None,
    download: DownloadFn | None = None,
    force: bool = False,
    expected_sha256: str | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Download ``url`` to ``<cache>/<filename>`` unless already present.

    A freshly downloaded file is verified against ``expected_sha256`` (when
    given) and a mismatch raises :class:`ChecksumError`. A cache hit is returned
    as-is without re-hashing (use :func:`verify_sha256` explicitly to re-check a
    cached file). ``progress`` is forwarded to the downloader.
    """
    cache = ensure_cache_dir(cache_dir)
    dest = cache / filename
    if dest.exists() and not force:
        return dest
    downloader = download if download is not None else default_download
    downloader(url, dest, progress)
    verify_sha256(dest, expected_sha256)
    return dest


def hf_download(
    repo_id: str,
    filename: str,
    *,
    cache_dir: str | Path | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Fetch a file from the Hugging Face Hub (lazy import; opt-in dependency).

    Verifies the resolved file against ``expected_sha256`` when one is recorded.
    huggingface_hub renders its own download progress to the console/log.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "huggingface_hub is required to fetch weights automatically; "
            "install it or pass an explicit local weights path"
        ) from exc
    cache = ensure_cache_dir(cache_dir)
    path: str = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache))
    resolved = Path(path)
    verify_sha256(resolved, expected_sha256)
    return resolved


def resolve_local(weights: str | Path | None) -> Path | None:
    """Validate an explicit local weights path, if one was provided."""
    if not weights:
        return None
    path = Path(weights).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path}")
    return path


# -- cache inspection (wizard) ----------------------------------------------
def cached_weight_files(cache_dir: str | Path | None = None) -> list[tuple[Path, int]]:
    """List ``(path, size_bytes)`` of cached weight files (recursive, ``.part`` skipped)."""
    cache = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    if not cache.exists():
        return []
    files = [
        (path, path.stat().st_size)
        for path in cache.rglob("*")
        if path.is_file() and path.suffix != ".part"
    ]
    return sorted(files, key=lambda item: str(item[0]))


def cache_size_bytes(cache_dir: str | Path | None = None) -> int:
    """Total size in bytes of all cached weight files."""
    return sum(size for _, size in cached_weight_files(cache_dir))


def human_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``512.0 MB``)."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"
