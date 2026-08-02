"""Local-filesystem blob storage for Thoth.

The default media backend since MinIO was decommissioned — thoth now
stores blobs directly on its local NVMe. Images land at:

    {root}/{station}/image/{YYYY}/{MM}/{DD}/{event_id}.{ext}

The key recorded in SQLite is the path relative to ``root``, using the
same scheme MinIO object keys used, so existing readers (API, classifier)
resolve blobs identically regardless of backend.

Writes are atomic: bytes go to a temp file in the destination directory,
are fsync'd, then renamed into place. A partially written image can
never be observed at its final path, and a crash mid-write leaves only
a ``.tmp`` file that a later cleanup can sweep.

Failures (disk full, permissions) raise :class:`LocalStoreError` with
the underlying OS error spelled out. The pipeline treats that as
"this one event failed" — it logs and drops the event; the process
keeps running.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from . import blobstore
from .events import ImageEvent

log = logging.getLogger(__name__)

_READ_CHUNK_BYTES = 32 * 1024

DEFAULT_LOCAL_ROOT = Path("/var/lib/thoth/media")


class LocalStoreError(RuntimeError):
    """A single blob operation against the local filesystem failed.

    Wraps the underlying :class:`OSError` with the path and the OS
    error string (``ENOSPC``, ``EACCES``, ...) so journal lines say
    exactly what went wrong instead of a bare traceback.
    """


@dataclass(frozen=True)
class LocalStorageConfig:
    """Settings for the local-filesystem media backend."""

    root: Path = DEFAULT_LOCAL_ROOT
    """Directory that holds all media blobs (units: absolute path).

    Created automatically at service startup. On thoth this should
    point at the NVMe-backed data volume, e.g. ``/var/lib/thoth/media``.
    Set via ``THOTH_STORAGE_ROOT`` in production or ``storage.local.root``
    in YAML.
    """


def _oserror_detail(exc: OSError) -> str:
    """Human-readable OS error detail: 'No space left on device (errno 28)'."""
    if exc.strerror:
        return f"{exc.strerror} (errno {exc.errno})"
    return str(exc)


class LocalStore:
    """Filesystem-backed blob store with the same surface as MinioStore."""

    def __init__(self, cfg: LocalStorageConfig) -> None:
        self.cfg = cfg

    def ensure_ready(self) -> None:
        """Create the storage root if it doesn't exist. Idempotent.

        Raises :class:`LocalStoreError` when the root can't be created
        or isn't writable — better to fail startup with a clear message
        than to fail every event later.
        """
        root = self.cfg.root
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LocalStoreError(
                f"cannot create storage root {root}: {_oserror_detail(exc)}"
            ) from exc
        if not os.access(root, os.W_OK):
            raise LocalStoreError(f"storage root {root} is not writable")
        log.info("local blob store ready at %s", root)

    def image_key(self, event: ImageEvent, event_id: str) -> str:
        return blobstore.image_key(event, event_id)

    def _path_for(self, key: str) -> Path:
        """Resolve a storage key to an absolute path under the root.

        Rejects keys that escape the root (``..`` components, absolute
        paths) — keys come from the DB and MQTT-derived event fields,
        so a hostile station name must not become a path traversal.
        """
        root = self.cfg.root.resolve()
        candidate = (root / key).resolve()
        if candidate != root and root not in candidate.parents:
            raise LocalStoreError(f"key {key!r} escapes storage root {root}")
        return candidate

    def put_image(self, event: ImageEvent, event_id: str) -> str:
        """Atomically write the image and return its key.

        Bytes land in a temp file next to the destination, get fsync'd,
        then rename into place — readers either see the complete file
        or nothing.
        """
        key = self.image_key(event, event_id)
        dest = self._path_for(key)
        tmp_path: Path | None = None
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=dest.parent, prefix=f".{event_id}.", suffix=".tmp"
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as fh:
                fh.write(event.image_bytes)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, dest)
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise LocalStoreError(
                f"failed to write {dest}: {_oserror_detail(exc)}"
            ) from exc
        log.debug("wrote %s to %s (%d bytes)", event_id, dest, event.size_bytes)
        return key

    def remove_object(self, key: str) -> None:
        """Delete a blob. Used to clean up orphans when the downstream
        index insert fails after a successful write. Best-effort: if the
        delete itself fails we log and move on — the next GC pass will
        sweep it."""
        try:
            self._path_for(key).unlink(missing_ok=True)
            log.warning("removed orphan local blob %s", key)
        except Exception:
            log.exception("failed to remove orphan local blob %s", key)

    def get_object_stream(self, key: str) -> tuple[Iterator[bytes], int | None]:
        """Open a blob and return a ``(byte_iterator, length)`` pair.

        Mirrors :meth:`MinioStore.get_object_stream` so the API and
        classifier stream media without buffering whole images. The
        caller is responsible for consuming (or closing) the iterator —
        the file handle is closed when iteration finishes.
        """
        path = self._path_for(key)
        try:
            fh = open(path, "rb")
        except OSError as exc:
            raise LocalStoreError(
                f"failed to open {path}: {_oserror_detail(exc)}"
            ) from exc
        length = os.fstat(fh.fileno()).st_size

        def _iter() -> Iterator[bytes]:
            try:
                while chunk := fh.read(_READ_CHUNK_BYTES):
                    yield chunk
            finally:
                fh.close()

        return _iter(), length
