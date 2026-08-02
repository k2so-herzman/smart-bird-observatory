"""Tests for the local-filesystem blob store.

Covers the happy path (write + read-back + key layout), the atomicity
contract (no partial file ever visible at the final path, no leftover
temp files), and the error paths (unwritable destination, missing blob,
key traversal) — all hermetic under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from banshee.events import ImageEvent
from banshee.localfs_store import LocalStorageConfig, LocalStore, LocalStoreError


def _make_image_event(
    station: str = "horus",
    ts: datetime | None = None,
    content_type: str = "image/jpeg",
    body: bytes = b"\xff\xd8\xff\xe0fake",
) -> ImageEvent:
    return ImageEvent(
        schema_version=1,
        station=station,
        captured_at=ts or datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(2328, 1748),
        content_type=content_type,
        image_bytes=body,
        size_bytes=len(body),
        changed_fraction=0.04,
        sha256=hashlib.sha256(body).hexdigest(),
    )


def _store(tmp_path: Path) -> LocalStore:
    return LocalStore(LocalStorageConfig(root=tmp_path / "media"))


# ---- ensure_ready ----------------------------------------------------------


def test_ensure_ready_creates_root(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert not store.cfg.root.exists()
    store.ensure_ready()
    assert store.cfg.root.is_dir()
    # Idempotent — second call is a no-op, not an error.
    store.ensure_ready()


def test_ensure_ready_fails_clearly_when_root_is_a_file(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.write_text("not a directory")
    store = LocalStore(LocalStorageConfig(root=root))
    with pytest.raises(LocalStoreError, match="cannot create storage root"):
        store.ensure_ready()


# ---- put_image: layout + content -------------------------------------------


def test_put_image_writes_blob_at_keyed_path(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    event = _make_image_event(ts=datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc))

    key = store.put_image(event, event_id="deadbeef")

    assert key == "horus/image/2026/04/17/deadbeef.jpg"
    blob = store.cfg.root / "horus" / "image" / "2026" / "04" / "17" / "deadbeef.jpg"
    assert blob.read_bytes() == event.image_bytes


def test_put_image_key_matches_minio_scheme_for_png(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    event = _make_image_event(content_type="image/png")
    key = store.put_image(event, event_id="abc123")
    assert key.endswith("abc123.png")
    assert (store.cfg.root / key).is_file()


def test_put_image_creates_intermediate_directories(tmp_path: Path) -> None:
    """No pre-seeding: a brand-new station/date path is created on demand."""
    store = _store(tmp_path)
    store.ensure_ready()
    key = store.put_image(_make_image_event(station="seth"), "id-1")
    assert key.startswith("seth/")
    assert (store.cfg.root / key).is_file()


def test_put_image_leaves_no_temp_files(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    store.put_image(_make_image_event(), "id-1")
    leftovers = [p for p in store.cfg.root.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


# ---- put_image: atomicity + error paths ------------------------------------


def test_failed_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the rename step fails (disk full at the wrong moment), the
    destination path must not exist and the temp file must be cleaned up
    — a partial image never looks ingested."""
    store = _store(tmp_path)
    store.ensure_ready()
    event = _make_image_event()

    def _boom(src: object, dst: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(LocalStoreError, match="No space left on device"):
        store.put_image(event, "id-1")

    dest = store.cfg.root / store.image_key(event, "id-1")
    assert not dest.exists()
    assert [p for p in store.cfg.root.rglob("*") if p.is_file()] == []


def test_put_image_surfaces_unwritable_destination(tmp_path: Path) -> None:
    """A blocked directory tree raises LocalStoreError with the OS detail
    instead of crashing the process — the pipeline drops just the event."""
    store = _store(tmp_path)
    store.ensure_ready()
    # Occupy the station path with a *file* so mkdir(parents=True) fails
    # regardless of the uid the tests run as.
    (store.cfg.root / "horus").write_text("in the way")

    with pytest.raises(LocalStoreError, match="failed to write"):
        store.put_image(_make_image_event(station="horus"), "id-1")


# ---- get_object_stream ------------------------------------------------------


def test_get_object_stream_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    body = b"jpegbytes" * 100
    event = _make_image_event(body=body)
    key = store.put_image(event, "id-1")

    stream, length = store.get_object_stream(key)
    assert length == len(body)
    assert b"".join(stream) == body


def test_get_object_stream_missing_key_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    with pytest.raises(LocalStoreError, match="failed to open"):
        store.get_object_stream("horus/image/2026/01/01/nope.jpg")


def test_key_escaping_root_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    with pytest.raises(LocalStoreError, match="escapes storage root"):
        store.get_object_stream("../secret.txt")


# ---- remove_object ----------------------------------------------------------


def test_remove_object_deletes_blob(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    key = store.put_image(_make_image_event(), "id-1")
    assert (store.cfg.root / key).exists()
    store.remove_object(key)
    assert not (store.cfg.root / key).exists()


def test_remove_object_missing_blob_is_silent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_ready()
    # Best-effort contract: never raises, even for nonsense keys.
    store.remove_object("horus/image/2026/01/01/ghost.jpg")
    store.remove_object("../outside-root.jpg")
