"""Shared blob-store surface for Thoth media backends.

Two backends implement this surface:

- :class:`banshee.localfs_store.LocalStore` — the default. Writes blobs
  to a directory on the ingest host's local disk (NVMe on thoth).
- :class:`banshee.minio_store.MinioStore` — legacy object storage. Kept
  for anyone who still runs a MinIO server; not required at runtime.

Both use the same key scheme, so the ``media_key`` column in SQLite is
backend-agnostic:

    {station}/image/{YYYY}/{MM}/{DD}/{event_id}.{ext}

Extension is derived from the event's content_type so the key stays
truthful when a station starts emitting PNG or HEIC.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

from .events import ImageEvent

if TYPE_CHECKING:
    from .config import ThothStorageConfig

# Fallback when mimetypes can't resolve (e.g. exotic content_type);
# stations only emit JPEG today so this keeps backward compatibility.
_DEFAULT_IMAGE_EXT = ".jpg"


def extension_for(content_type: str) -> str:
    """Map a MIME content type to a filename extension."""
    ext = mimetypes.guess_extension(content_type or "")
    if not ext:
        return _DEFAULT_IMAGE_EXT
    # mimetypes returns ".jpe" for image/jpeg on some platforms — normalize.
    if ext == ".jpe":
        return ".jpg"
    return ext


def image_key(event: ImageEvent, event_id: str) -> str:
    """Backend-agnostic storage key for an image event.

    This is the value recorded in the ``media_key`` column, so it must
    stay stable across backends — a MinIO object name and a path
    relative to the local storage root are the same string.
    """
    day = event.captured_at.strftime("%Y/%m/%d")
    ext = extension_for(event.content_type)
    return f"{event.station}/image/{day}/{event_id}{ext}"


class BlobStore(Protocol):
    """The media-storage surface the pipeline, API, and classifier use.

    Implementations must fail per-call (raise on the single operation)
    rather than crash the process — callers treat a raised exception as
    "this event failed", log it, and keep serving the MQTT loop.
    """

    def ensure_ready(self) -> None:
        """Prepare the backend for writes (create bucket / root dir).

        Called once at service startup. Idempotent.
        """
        ...

    def put_image(self, event: ImageEvent, event_id: str) -> str:
        """Persist the image blob and return its storage key."""
        ...

    def remove_object(self, key: str) -> None:
        """Best-effort delete of a blob (orphan cleanup). Never raises."""
        ...

    def get_object_stream(self, key: str) -> tuple[Iterator[bytes], int | None]:
        """Fetch a blob as a ``(byte_iterator, length)`` pair."""
        ...


def build_store(storage_cfg: ThothStorageConfig) -> BlobStore:
    """Construct the configured blob-store backend.

    Selection follows ``storage_cfg.effective_backend``: an explicit
    ``backend`` value wins; otherwise MinIO is used when (and only when)
    a MinIO config block is present, and local storage is the default.

    Imports are deferred so selecting the local backend never touches
    the ``minio`` client package.
    """
    backend = storage_cfg.effective_backend
    if backend == "minio":
        if storage_cfg.minio is None:
            raise ValueError("storage backend is 'minio' but no MinIO config is set")
        from .minio_store import MinioStore

        return MinioStore(storage_cfg.minio)
    if backend == "local":
        from .localfs_store import LocalStore

        return LocalStore(storage_cfg.local)
    raise ValueError(f"unknown storage backend: {backend!r}")
