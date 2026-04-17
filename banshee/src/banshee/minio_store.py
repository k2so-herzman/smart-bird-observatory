"""MinIO blob storage for Thoth.

Images land at:

    {bucket}/{station}/image/{YYYY}/{MM}/{DD}/{event_id}.jpg

No thumbs are generated at ingest — imgproxy handles resize on demand.
Bucket creation is idempotent and happens on first write, so a fresh
MinIO server auto-provisions without a separate bootstrap step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from .events import ImageEvent

log = logging.getLogger(__name__)


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


class MinioStore:
    """Thin Minio wrapper — bucket ensure + image put.

    The client is instantiated eagerly so credential problems surface
    at startup rather than on first event.
    """

    def __init__(self, cfg: MinioConfig) -> None:
        host, scheme_secure = _split_endpoint(cfg.endpoint)
        # Explicit `secure` config wins over a scheme sniff, so operators
        # can force TLS on a proxied setup that advertises http://.
        secure = cfg.secure or scheme_secure
        self.cfg = cfg
        self._client = Minio(
            host,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=secure,
        )

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
        day = event.captured_at.strftime("%Y/%m/%d")
        return f"{event.station}/image/{day}/{event_id}.jpg"

    def put_image(self, event: ImageEvent, event_id: str) -> str:
        """Upload the JPEG and return its key."""
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
