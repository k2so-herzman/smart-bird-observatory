"""MinIO blob storage for Thoth (legacy backend).

The local-filesystem backend (:mod:`banshee.localfs_store`) is the
default since the MinIO host was decommissioned; this module is kept
compiling and working for anyone who still configures object storage.

Images land at:

    {bucket}/{station}/image/{YYYY}/{MM}/{DD}/{event_id}.{ext}

Extension is derived from the event's content_type so the key stays
truthful when a station starts emitting PNG or HEIC. No thumbs are
generated at ingest — imgproxy handles resize on demand. Bucket
creation is idempotent and happens on first write, so a fresh MinIO
server auto-provisions without a separate bootstrap step.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from . import blobstore
from .events import ImageEvent

log = logging.getLogger(__name__)

# Key/extension logic moved to banshee.blobstore so both backends share
# one scheme; alias kept for existing imports.
_extension_for = blobstore.extension_for


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str  # "host:port" or full URL — we normalize
    access_key: str
    secret_key: str
    bucket: str = "thoth"
    secure: bool = False


def _split_endpoint(endpoint: str) -> tuple[str, bool]:
    """Accept either 'http://host:port' or 'host:port'.

    Returns (host:port, secure). The Minio() client wants the bare
    host:port form and a separate secure flag, but our env.example
    stores the full URL for readability. Normalize here.
    """
    if "://" in endpoint:
        parsed = urlparse(endpoint)
        host = parsed.netloc or parsed.path
        return host, parsed.scheme == "https"
    return endpoint, False


class _MinioLike(Protocol):
    """The subset of the ``minio.Minio`` surface this module uses.

    Exists so tests can substitute a fake without importing or
    monkey-patching the real client. Anything listed here must match
    ``minio.Minio`` at the call-site level.
    """

    def bucket_exists(self, bucket_name: str) -> bool: ...

    def make_bucket(self, bucket_name: str) -> None: ...

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: object,
        length: int,
        content_type: str,
    ) -> object: ...

    def remove_object(self, bucket_name: str, object_name: str) -> None: ...

    def get_object(self, bucket_name: str, object_name: str) -> object: ...


def _build_default_client(cfg: MinioConfig) -> Minio:
    """Factory for the real ``minio.Minio`` client from a MinioConfig.

    Pulled out of ``__init__`` so the real-client construction path is
    exercised in one place and tests can bypass it entirely by
    injecting their own client.
    """
    host, scheme_secure = _split_endpoint(cfg.endpoint)
    # Explicit `secure` config wins over a scheme sniff, so operators
    # can force TLS on a proxied setup that advertises http://.
    secure = cfg.secure or scheme_secure
    return Minio(
        host,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=secure,
    )


class MinioStore:
    """Thin Minio wrapper — bucket ensure + image put.

    The client is built eagerly so credential problems surface at
    startup rather than on first event. In tests, pass an in-memory
    fake via the ``client`` parameter to bypass the network entirely.
    """

    def __init__(
        self,
        cfg: MinioConfig,
        client: _MinioLike | None = None,
    ) -> None:
        self.cfg = cfg
        self._client = client if client is not None else _build_default_client(cfg)

    def ensure_ready(self) -> None:
        """BlobStore protocol entry point — delegates to :meth:`ensure_bucket`."""
        self.ensure_bucket()

    def ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist. Idempotent."""
        try:
            if not self._client.bucket_exists(self.cfg.bucket):
                self._client.make_bucket(self.cfg.bucket)
                log.info("created MinIO bucket %s", self.cfg.bucket)
        except S3Error as exc:
            # BucketAlreadyOwnedByYou is benign under a race
            if exc.code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                return
            raise

    def image_key(self, event: ImageEvent, event_id: str) -> str:
        return blobstore.image_key(event, event_id)

    def put_image(self, event: ImageEvent, event_id: str) -> str:
        """Upload the image and return its key."""
        from io import BytesIO

        key = self.image_key(event, event_id)
        body = BytesIO(event.image_bytes)
        self._client.put_object(
            self.cfg.bucket,
            key,
            body,
            length=event.size_bytes,
            content_type=event.content_type,
        )
        log.debug(
            "uploaded %s to s3://%s/%s (%d bytes)",
            event_id,
            self.cfg.bucket,
            key,
            event.size_bytes,
        )
        return key

    def remove_object(self, key: str) -> None:
        """Delete an object. Used to clean up orphans when the downstream
        index insert fails after a successful upload. Best-effort: if the
        delete itself fails we log and move on — the next GC pass will
        sweep it."""
        try:
            self._client.remove_object(self.cfg.bucket, key)
            log.warning("removed orphan MinIO object %s", key)
        except Exception:
            log.exception("failed to remove orphan MinIO object %s", key)

    def get_object_stream(self, key: str) -> tuple[Iterator[bytes], int | None]:
        """Fetch an object and return a ``(byte_iterator, length)`` pair.

        Used by the read API to stream media back to HTTP clients
        without buffering the whole image in memory. ``length`` is the
        object's reported size in bytes, or ``None`` if the underlying
        client can't determine it.

        The caller is responsible for consuming the iterator fully —
        the underlying ``urllib3`` response is closed when iteration
        completes.
        """
        response = self._client.get_object(self.cfg.bucket, key)

        length: int | None = None
        try:
            header_len = response.headers.get("Content-Length")
            if header_len is not None:
                length = int(header_len)
        except (AttributeError, ValueError, TypeError):
            length = None

        def _iter() -> Iterator[bytes]:
            try:
                yield from response.stream(amt=32 * 1024)
            finally:
                response.close()
                response.release_conn()

        return _iter(), length
