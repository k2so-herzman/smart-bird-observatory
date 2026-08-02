"""Tests for the backend factory + shared key scheme in banshee.blobstore."""

from __future__ import annotations

from pathlib import Path

import pytest

from banshee.blobstore import build_store, extension_for, image_key
from banshee.config import ThothStorageConfig
from banshee.localfs_store import LocalStorageConfig, LocalStore
from banshee.minio_store import MinioConfig, MinioStore

from .test_localfs_store import _make_image_event


def test_extension_for_jpeg_and_fallback() -> None:
    assert extension_for("image/jpeg") == ".jpg"
    assert extension_for("image/png") == ".png"
    assert extension_for("application/x-nonsense") == ".jpg"
    assert extension_for("") == ".jpg"


def test_image_key_is_backend_agnostic(tmp_path: Path) -> None:
    """Both backends must produce the identical media_key so DB rows stay
    portable between storage backends."""
    event = _make_image_event()
    local = LocalStore(LocalStorageConfig(root=tmp_path))
    minio = MinioStore(MinioConfig(endpoint="x:9000", access_key="a", secret_key="b"))
    assert (
        local.image_key(event, "id-1")
        == minio.image_key(event, "id-1")
        == image_key(event, "id-1")
    )


def test_build_store_selects_local_by_default(tmp_path: Path) -> None:
    cfg = ThothStorageConfig(
        db_path=tmp_path / "events.db",
        local=LocalStorageConfig(root=tmp_path / "media"),
    )
    store = build_store(cfg)
    assert isinstance(store, LocalStore)
    assert store.cfg.root == tmp_path / "media"


def test_build_store_selects_minio_when_configured(tmp_path: Path) -> None:
    cfg = ThothStorageConfig(
        db_path=tmp_path / "events.db",
        minio=MinioConfig(endpoint="x:9000", access_key="a", secret_key="b"),
    )
    assert isinstance(build_store(cfg), MinioStore)


def test_build_store_explicit_local_beats_minio_block(tmp_path: Path) -> None:
    cfg = ThothStorageConfig(
        db_path=tmp_path / "events.db",
        backend="local",
        minio=MinioConfig(endpoint="x:9000", access_key="a", secret_key="b"),
    )
    assert isinstance(build_store(cfg), LocalStore)


def test_build_store_minio_without_config_raises(tmp_path: Path) -> None:
    cfg = ThothStorageConfig(db_path=tmp_path / "events.db", backend="minio")
    with pytest.raises(ValueError, match="no MinIO config"):
        build_store(cfg)
