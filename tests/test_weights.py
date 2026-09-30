"""Tests for weight download integrity, progress and cache inspection (E-5, H-8).

Everything here is stdlib-only: the HTTP layer is faked by monkeypatching
``urllib.request.urlopen``, so no network, torch or GDAL is needed.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from eo_inference_toolbox.eocore.backends import _weights


class _FakeResponse:
    """Minimal stand-in for an ``http.client.HTTPResponse`` context manager."""

    def __init__(self, data: bytes, *, with_length: bool = True) -> None:
        self._buffer = io.BytesIO(data)
        self.headers: dict[str, str] = {"Content-Length": str(len(data))} if with_length else {}

    def read(self, n: int = -1) -> bytes:
        return self._buffer.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self._buffer.close()


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, data: bytes, *, with_length: bool = True
) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(data, with_length=with_length)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _recorder(sink: list[tuple[int, int | None]]) -> _weights.ProgressFn:
    """A two-argument progress callback that appends ``(done, total)`` to ``sink``."""

    def record(done: int, total: int | None) -> None:
        sink.append((done, total))

    return record


# -- checksums ---------------------------------------------------------------
def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"clay-weights-bytes" * 100
    path = tmp_path / "w.bin"
    path.write_bytes(payload)
    assert _weights.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_verify_sha256_none_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "w.bin"
    path.write_bytes(b"anything")
    # Opt-in: no expected digest recorded -> never rejects (default-safe).
    _weights.verify_sha256(path, None)
    _weights.verify_sha256(path, "")


def test_verify_sha256_accepts_correct_and_is_case_insensitive(tmp_path: Path) -> None:
    payload = b"good-bytes"
    path = tmp_path / "w.bin"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _weights.verify_sha256(path, digest)
    _weights.verify_sha256(path, digest.upper())  # case-insensitive


def test_verify_sha256_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "w.bin"
    path.write_bytes(b"corrupt")
    with pytest.raises(_weights.ChecksumError, match="checksum mismatch"):
        _weights.verify_sha256(path, "0" * 64)


# -- download + progress -----------------------------------------------------
def test_default_download_writes_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"model-checkpoint-payload"
    _patch_urlopen(monkeypatch, data)
    dest = tmp_path / "sub" / "ckpt.bin"
    calls: list[tuple[int, int | None]] = []

    _weights.default_download("https://example.test/ckpt.bin", dest, _recorder(calls))

    assert dest.read_bytes() == data
    assert not dest.with_suffix(dest.suffix + ".part").exists()  # atomic rename cleaned up
    assert calls[0] == (0, len(data))  # announced up front with the known total
    assert calls[-1] == (len(data), len(data))  # ends at 100%
    done_values = [done for done, _ in calls]
    assert done_values == sorted(done_values)  # monotonic


def test_default_download_without_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"no-length-header"
    _patch_urlopen(monkeypatch, data, with_length=False)
    dest = tmp_path / "ckpt.bin"
    calls: list[tuple[int, int | None]] = []

    _weights.default_download("https://example.test/ckpt.bin", dest, _recorder(calls))

    assert dest.read_bytes() == data
    assert all(total is None for _, total in calls)  # total unknown -> None throughout


# -- fetch_url orchestration -------------------------------------------------
def test_fetch_url_skips_download_on_cache_hit(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ckpt.bin").write_bytes(b"already-here")
    called = {"n": 0}

    def fake_download(url: str, dest: Path, progress: Any) -> None:
        called["n"] += 1

    result = _weights.fetch_url(
        "https://example.test/ckpt.bin", "ckpt.bin", cache_dir=cache, download=fake_download
    )
    assert result == cache / "ckpt.bin"
    assert called["n"] == 0  # cache hit -> downloader never invoked


def test_fetch_url_forces_redownload_and_verifies_checksum(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    payload = b"fresh-weights"
    digest = hashlib.sha256(payload).hexdigest()

    def fake_download(url: str, dest: Path, progress: Any) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)

    result = _weights.fetch_url(
        "https://example.test/ckpt.bin",
        "ckpt.bin",
        cache_dir=cache,
        download=fake_download,
        force=True,
        expected_sha256=digest,
    )
    assert result.read_bytes() == payload


def test_fetch_url_raises_on_checksum_mismatch(tmp_path: Path) -> None:
    def fake_download(url: str, dest: Path, progress: Any) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"tampered")

    with pytest.raises(_weights.ChecksumError):
        _weights.fetch_url(
            "https://example.test/ckpt.bin",
            "ckpt.bin",
            cache_dir=tmp_path / "cache",
            download=fake_download,
            expected_sha256="a" * 64,
        )


# -- cache inspection --------------------------------------------------------
def test_cache_inspection_lists_files_and_skips_part(tmp_path: Path) -> None:
    cache = tmp_path / "weights"
    (cache / "clay").mkdir(parents=True)
    (cache / "clay" / "clay-v1.5.ckpt").write_bytes(b"x" * 1000)
    (cache / "other.bin").write_bytes(b"y" * 24)
    (cache / "half.ckpt.part").write_bytes(b"z" * 999)  # in-progress temp, excluded

    files = _weights.cached_weight_files(cache)
    names = [p.name for p, _ in files]
    assert "clay-v1.5.ckpt" in names
    assert "other.bin" in names
    assert "half.ckpt.part" not in names
    paths = [str(p) for p, _ in files]
    assert paths == sorted(paths)  # returned in a stable, path-sorted order
    assert _weights.cache_size_bytes(cache) == 1024


def test_cache_size_zero_when_absent(tmp_path: Path) -> None:
    assert _weights.cache_size_bytes(tmp_path / "does-not-exist") == 0
    assert _weights.cached_weight_files(tmp_path / "nope") == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1536, "1.5 KB"),
        (5 * 1024 * 1024, "5.0 MB"),
        (2 * 1024**3, "2.0 GB"),
    ],
)
def test_human_bytes(value: int, expected: str) -> None:
    assert _weights.human_bytes(value) == expected
