"""Thoth / Banshee configuration.

Defines the frozen dataclasses that represent Banshee's runtime
configuration and the two loaders that populate them.

Two loaders live here:

- `from_env()` — reads ``/etc/thoth/env`` style environment variables,
  the production path. systemd's ``EnvironmentFile=/etc/thoth/env``
  populates these before the service starts.
- `load(path)` — reads a YAML file. Kept for local dev and tests.

Both produce the same :class:`BansheeConfig`.

Typical YAML layout::

    mqtt:
      host: broker.local
      port: 1883
      username: banshee
      password: secret
      topic_prefix: sbo
      station_filter: "+"
      client_id: banshee-prod
    storage:
      db_path: /var/lib/thoth/events.db
      minio:
        endpoint: "http://minio.local:9000"
        access_key: minioadmin
        secret_key: minioadmin
        bucket: thoth
        secure: false
    influx:
      url: "http://192.168.1.24:8086"
      token: ""
      org: herzman
      bucket: sbo
    notify:
      telegram_min_confidence: 0.75
      ha_enabled: true

Environment variables (production path)
-----------------------------------------
MQTT fields come from :class:`sbo_shared.MqttBaseConfig` plus banshee's
own overrides:

.. list-table::
   :widths: 30 15 55

   * - ``MQTT_HOST``
     - required
     - Broker hostname or IP.
   * - ``MQTT_PORT``
     - optional
     - Broker port (default 1883).
   * - ``MQTT_USERNAME``
     - optional
     - Broker username.
   * - ``MQTT_PASSWORD``
     - optional
     - Broker password.
   * - ``MQTT_TOPIC_PREFIX``
     - optional
     - Topic namespace prefix (default ``sbo``).
   * - ``MQTT_STATION_FILTER``
     - optional
     - MQTT single-level wildcard (default ``+``).
   * - ``MQTT_CLIENT_ID``
     - optional
     - Paho client ID (default auto-generated).
   * - ``THOTH_DB_PATH``
     - optional
     - SQLite file path (default ``/var/lib/thoth/events.db``).
   * - ``MINIO_ENDPOINT``
     - required
     - MinIO endpoint as ``host:port`` or full URL.
   * - ``MINIO_ACCESS_KEY``
     - required
     - MinIO access key.
   * - ``MINIO_SECRET_KEY``
     - required
     - MinIO secret key.
   * - ``MINIO_BUCKET``
     - optional
     - MinIO bucket name (default ``thoth``).
   * - ``MINIO_SECURE``
     - optional
     - Use TLS (default ``false``).
   * - ``INFLUX_URL``
     - optional
     - InfluxDB URL (default ``http://192.168.1.24:8086``).
   * - ``INFLUX_TOKEN``
     - optional
     - InfluxDB API token (default empty → writes disabled).
   * - ``INFLUX_ORG``
     - optional
     - InfluxDB organisation (default ``herzman``).
   * - ``INFLUX_BUCKET``
     - optional
     - InfluxDB bucket (default ``sbo``).
   * - ``TELEGRAM_MIN_CONFIDENCE``
     - optional
     - Minimum classifier confidence for Telegram alerts (default ``0.75``).
   * - ``HA_ENABLED``
     - optional
     - Enable Home Assistant notifications (default ``true``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from sbo_shared import MQTT_DEFAULT_PORT, MqttBaseConfig

from .minio_store import MinioConfig


class ConfigError(ValueError):
    """Raised when a required environment variable is missing or invalid.

    Subclasses :class:`ValueError` so existing call sites that catch
    ``ValueError`` keep working, while banshee-specific code can narrow
    to ``ConfigError``.
    """


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise ConfigError(f"required env var {name} is unset")
    return val


def _optional(name: str, default: str) -> str:
    return os.environ.get(name, default)


_TRUTHY = frozenset({"true", "yes", "on", "1"})
_FALSY = frozenset({"false", "no", "off", "0", ""})


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var accepting common truthy/falsy forms."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm in _TRUTHY:
        return True
    if norm in _FALSY:
        return False
    raise ConfigError(
        f"env var {name}={raw!r} is not a valid boolean "
        f"(expected one of: {sorted(_TRUTHY | _FALSY)})"
    )


@dataclass(frozen=True)
class MqttConfig(MqttBaseConfig):
    """Subscriber-side MQTT config.

    Extends :class:`sbo_shared.MqttBaseConfig` with the two fields only
    a subscriber needs:

    * ``station_filter`` — MQTT single-level wildcard pattern. ``"+"``
      means "every station"; a concrete name like ``"horus"`` narrows
      the subscription to one station.
    * ``client_id`` — paho ``client_id``. Useful to set explicitly when
      running multiple Banshee processes against the same broker.
    """

    station_filter: str = "+"
    """MQTT single-level wildcard that filters which stations Banshee
    subscribes to.

    ``"+"`` (default) means every station publishes to the shared topic
    prefix and Banshee ingests all of them.  Set to a concrete station
    name (e.g. ``"horus"``) to restrict a dedicated Banshee instance to
    one station.  The value is interpolated into the subscription topic as
    ``{topic_prefix}/{station_filter}/image``.
    """

    client_id: str | None = None
    """Paho MQTT client identifier sent to the broker at connect time.

    ``None`` (default) lets paho auto-generate a random ID, which is fine
    for a single Banshee process.  Set explicitly (e.g. ``"banshee-prod"``)
    when running multiple Banshee instances against the same broker so
    their sessions do not collide.
    """

    @classmethod
    def from_env(cls) -> "MqttConfig":
        """Build from process env (``/etc/thoth/env`` in production).

        Translates the shared base's :class:`ValueError` into banshee's
        :class:`ConfigError` so callers can catch a single, repo-local
        exception type regardless of which field triggered the error.
        """
        try:
            base = MqttBaseConfig.from_env()
        except ValueError as exc:
            # ConfigError subclasses ValueError — existing catchers
            # keep working; banshee-specific catchers still see
            # ConfigError.
            raise ConfigError(str(exc)) from exc
        return cls(
            host=base.host,
            port=base.port,
            username=base.username,
            password=base.password,
            topic_prefix=base.topic_prefix,
            station_filter=_optional("MQTT_STATION_FILTER", "+"),
            client_id=os.environ.get("MQTT_CLIENT_ID") or None,
        )


@dataclass(frozen=True)
class ThothStorageConfig:
    """SQLite + MinIO storage settings for the Thoth ingest service.

    Groups the two persistence back-ends used by Banshee: the SQLite
    event index (metadata + classifier results) and the MinIO blob store
    (raw image files).
    """

    db_path: Path
    """Filesystem path to the SQLite database file (units: absolute path).

    Parent directories are created automatically on first write via
    :meth:`banshee.eventstore.EventStore.init`.  The default in
    production is ``/var/lib/thoth/events.db`` (set via
    ``THOTH_DB_PATH``).  In tests, pass an in-memory placeholder and
    inject a ``:memory:`` connection factory instead.
    """

    minio: MinioConfig
    """Connection settings for the MinIO blob store.

    See :class:`banshee.minio_store.MinioConfig` for field-level docs.
    Populated from ``MINIO_*`` environment variables in production or from
    the ``storage.minio`` YAML block in dev/test.
    """

    @classmethod
    def from_env(cls) -> "ThothStorageConfig":
        """Build from process environment variables.

        Returns:
            A fully-populated :class:`ThothStorageConfig`.

        Raises:
            ConfigError: if ``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``, or
                ``MINIO_SECRET_KEY`` are unset or empty.
        """
        return cls(
            db_path=Path(_optional("THOTH_DB_PATH", "/var/lib/thoth/events.db")),
            minio=MinioConfig(
                endpoint=_require("MINIO_ENDPOINT"),
                access_key=_require("MINIO_ACCESS_KEY"),
                secret_key=_require("MINIO_SECRET_KEY"),
                bucket=_optional("MINIO_BUCKET", "thoth"),
                secure=_bool_env("MINIO_SECURE", False),
            ),
        )


# Homelab InfluxDB endpoint. Lives here rather than inline in default
# args so a grep for the IP finds exactly one hit and operators have
# a named symbol to override in tests.
DEFAULT_INFLUX_URL = "http://192.168.1.24:8086"


@dataclass(frozen=True)
class InfluxConfig:
    """InfluxDB connection settings.

    An empty ``token`` is treated by :class:`banshee.influx.InfluxWriter`
    as "writes disabled" — a convenience for local dev where you may
    not want to run a real Influx instance. In production, systemd's
    EnvironmentFile supplies ``INFLUX_TOKEN``.
    """

    url: str = DEFAULT_INFLUX_URL
    """HTTP(S) URL of the InfluxDB v2 instance (units: URL).

    Default ``http://192.168.1.24:8086`` targets the homelab InfluxDB
    server.  Override with ``INFLUX_URL`` in production or when running
    against a different host.  Must include scheme and port.
    """

    token: str = ""
    """InfluxDB API token used for write authentication.

    An empty string (default) is treated by
    :class:`banshee.influx.InfluxWriter` as a signal to disable writes
    entirely — useful for local dev without a live Influx instance.
    In production, set ``INFLUX_TOKEN`` in ``/etc/thoth/env``.
    """

    org: str = "herzman"
    """InfluxDB organisation name that owns the target bucket.

    Must match the organisation configured in the InfluxDB server.
    Default ``"herzman"`` is the homelab org; override with
    ``INFLUX_ORG`` if deploying to a different organisation.
    """

    bucket: str = "sbo"
    """InfluxDB bucket where bird-observation measurements are written.

    Default ``"sbo"`` (Smart Bird Observatory).  The bucket must already
    exist in the target organisation; Banshee does not auto-create it.
    Override with ``INFLUX_BUCKET``.
    """

    @classmethod
    def from_env(cls) -> "InfluxConfig":
        """Build from process environment variables.

        All fields are optional; missing variables fall back to the field
        defaults.

        Returns:
            A fully-populated :class:`InfluxConfig`.
        """
        return cls(
            url=_optional("INFLUX_URL", DEFAULT_INFLUX_URL),
            token=_optional("INFLUX_TOKEN", ""),
            org=_optional("INFLUX_ORG", "herzman"),
            bucket=_optional("INFLUX_BUCKET", "sbo"),
        )


@dataclass(frozen=True)
class NotifyConfig:
    """Settings that control outbound notifications.

    Covers the Telegram alert channel and the Home Assistant webhook.
    Both channels are gated on the classifier confidence score so that
    only high-confidence identifications generate notifications.
    """

    telegram_min_confidence: float = 0.75
    """Minimum classifier confidence required to send a Telegram alert
    (units: fraction, 0.0–1.0).

    Classification results below this threshold are recorded in the
    database but do not trigger a Telegram message.  Default 0.75 (75 %)
    keeps false-positive alerts rare without missing clear sightings.
    Override with ``TELEGRAM_MIN_CONFIDENCE``.
    """

    ha_enabled: bool = True
    """Whether to fire Home Assistant webhook notifications (units: bool).

    When ``True`` (default), Banshee calls the HA webhook for each
    classified event so automations can react (e.g. turn on a spotlight).
    Set to ``False`` (or ``HA_ENABLED=false``) to disable HA integration
    without touching Telegram.
    """

    @classmethod
    def from_env(cls) -> "NotifyConfig":
        """Build from process environment variables.

        All fields are optional; missing variables fall back to the field
        defaults.

        Returns:
            A fully-populated :class:`NotifyConfig`.

        Raises:
            ConfigError: if ``TELEGRAM_MIN_CONFIDENCE`` is set but not
                parseable as a float, or if ``HA_ENABLED`` is set to an
                unrecognised boolean string.
        """
        return cls(
            telegram_min_confidence=float(
                _optional("TELEGRAM_MIN_CONFIDENCE", "0.75")
            ),
            ha_enabled=_bool_env("HA_ENABLED", True),
        )


@dataclass(frozen=True)
class ClassifierConfig:
    """Settings for the Thoth classifier worker.

    Empty ``model_path`` is a signal to ``thoth-classify.service`` to
    fall back to the no-op :class:`banshee.classifier.model.DummyClassifier`,
    which lets the pipeline deploy end-to-end before a real model
    artifact is staged.

    Environment variables (all optional):

    * ``THOTH_MODEL_PATH`` — path to a ``.tflite`` model file. Empty
      or missing file triggers the dummy fallback.
    * ``THOTH_LABELS_PATH`` — path to a newline-separated labels file.
      Required when ``THOTH_MODEL_PATH`` is set; missing labels also
      trigger the dummy fallback (with a loud warning).
    * ``THOTH_CLASSIFY_POLL_INTERVAL`` — seconds between DB polls when
      the queue is empty. Default ``2.0``.
    * ``THOTH_CLASSIFY_BATCH_SIZE`` — max rows processed per tick.
      Default ``8``.
    """

    model_path: str = ""
    """Filesystem path to a ``.tflite`` model file.

    Empty (default) means no real model is configured and
    ``thoth-classify`` runs the :class:`DummyClassifier`. Must be an
    absolute path for production; relative paths work in dev but
    systemd's WorkingDirectory makes them surprising.
    """

    labels_path: str = ""
    """Filesystem path to a UTF-8 labels file (one label per line).

    Required whenever ``model_path`` is set. An empty value while
    ``model_path`` is populated is treated as a misconfiguration and
    the worker falls back to the dummy classifier (with a WARNING)
    rather than crash.
    """

    poll_interval_seconds: float = 2.0
    """Seconds to sleep between DB polls when the pending queue is empty.

    Short enough that new events classify in near-real-time; long
    enough that an idle service doesn't thrash SQLite. 2 s matches
    the sort of latency humans tolerate on a bird dashboard.
    """

    batch_size: int = 8
    """Maximum number of pending rows processed per worker tick.

    Bounded so shutdown latency stays low even if a backlog has
    accumulated — the worker finishes its current batch before
    checking the stop flag.
    """

    @classmethod
    def from_env(cls) -> "ClassifierConfig":
        """Build from process environment variables.

        All fields are optional.

        Raises:
            ConfigError: if ``THOTH_CLASSIFY_POLL_INTERVAL`` or
                ``THOTH_CLASSIFY_BATCH_SIZE`` are set but unparseable.
        """
        try:
            poll = float(_optional("THOTH_CLASSIFY_POLL_INTERVAL", "2.0"))
        except ValueError as exc:
            raise ConfigError(
                f"THOTH_CLASSIFY_POLL_INTERVAL is not a valid float: {exc}"
            ) from exc
        try:
            batch = int(_optional("THOTH_CLASSIFY_BATCH_SIZE", "8"))
        except ValueError as exc:
            raise ConfigError(
                f"THOTH_CLASSIFY_BATCH_SIZE is not a valid int: {exc}"
            ) from exc
        return cls(
            model_path=_optional("THOTH_MODEL_PATH", ""),
            labels_path=_optional("THOTH_LABELS_PATH", ""),
            poll_interval_seconds=poll,
            batch_size=batch,
        )


@dataclass(frozen=True)
class BansheeConfig:
    """Top-level configuration for the Banshee ingest service.

    Aggregates the subsystem configs into a single frozen object.
    Obtain one via :meth:`from_env` (production) or :func:`load`
    (dev / tests).
    """

    mqtt: MqttConfig
    """MQTT broker connection and subscription settings.

    Controls which broker Banshee connects to, what credentials it uses,
    and which station topics it subscribes to.  See :class:`MqttConfig`.
    """

    storage: ThothStorageConfig
    """SQLite event-index and MinIO blob-store settings.

    See :class:`ThothStorageConfig` for field-level docs.
    """

    influx: InfluxConfig
    """InfluxDB write settings for time-series metrics.

    An empty token disables writes; see :class:`InfluxConfig`.
    """

    notify: NotifyConfig = field(default_factory=NotifyConfig)
    """Outbound notification settings (Telegram + Home Assistant).

    Defaults to :class:`NotifyConfig` with all defaults, which enables
    HA notifications and sets the Telegram confidence threshold to 0.75.
    See :class:`NotifyConfig`.
    """

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    """Classifier worker settings.

    Defaults to :class:`ClassifierConfig` with an unset model path, so
    ``thoth-classify.service`` runs the :class:`DummyClassifier`
    fallback. See :class:`ClassifierConfig` for the env-var surface.
    """

    @classmethod
    def from_env(cls) -> "BansheeConfig":
        """Build a complete :class:`BansheeConfig` from process env.

        Delegates to each subsystem's ``from_env()`` classmethod.  All
        required variables (``MQTT_HOST``, ``MINIO_ENDPOINT``,
        ``MINIO_ACCESS_KEY``, ``MINIO_SECRET_KEY``) must be set; optional
        variables fall back to their field defaults.

        Returns:
            A fully-populated, frozen :class:`BansheeConfig`.

        Raises:
            ConfigError: if any required environment variable is unset or
                if any boolean variable holds an unrecognised value.
        """
        return cls(
            mqtt=MqttConfig.from_env(),
            storage=ThothStorageConfig.from_env(),
            influx=InfluxConfig.from_env(),
            notify=NotifyConfig.from_env(),
            classifier=ClassifierConfig.from_env(),
        )


def load(path: Path | str) -> BansheeConfig:
    """Load a :class:`BansheeConfig` from a YAML file (dev / test path).

    Reads the file at *path*, parses it with :func:`yaml.safe_load`, and
    constructs a :class:`BansheeConfig` from the resulting mapping.  The
    ``influx`` and ``notify`` top-level keys are optional; missing keys
    fall back to all field defaults.

    Args:
        path: Filesystem path to the YAML configuration file.  Accepts
            both :class:`str` and :class:`pathlib.Path`.

    Returns:
        A fully-populated, frozen :class:`BansheeConfig`.

    Raises:
        FileNotFoundError: if *path* does not exist.
        yaml.YAMLError: if the file contains invalid YAML.
        KeyError: if a required top-level key (``mqtt``, ``storage``) or
            a required nested key is absent from the YAML.
        TypeError: if a field value has the wrong Python type after
            deserialisation.
    """
    data = yaml.safe_load(Path(path).read_text())

    mqtt = MqttConfig(**data["mqtt"])

    storage_data = dict(data["storage"])
    storage_data["db_path"] = Path(storage_data["db_path"])
    minio_data = storage_data.pop("minio")
    storage = ThothStorageConfig(
        db_path=storage_data["db_path"],
        minio=MinioConfig(**minio_data),
    )

    influx = InfluxConfig(**data.get("influx", {}))
    notify = NotifyConfig(**data.get("notify", {}))
    classifier = ClassifierConfig(**data.get("classifier", {}))

    return BansheeConfig(
        mqtt=mqtt,
        storage=storage,
        influx=influx,
        notify=notify,
        classifier=classifier,
    )
