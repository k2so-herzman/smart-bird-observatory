"""Station configuration loader.

Defines the frozen dataclasses that represent a station's runtime
configuration, and the :func:`load` function that deserialises a YAML
file into a :class:`HorusConfig`.

Typical YAML layout::

    station: feeder-north
    camera: imx477
    mqtt:
      host: broker.local
      port: 1883
      topic_prefix: sbo/feeder-north
    capture:
      width: 2304
      height: 1296
      jpeg_quality: 90
      interval_s: 1.0
    motion:
      pixel_threshold: 25
      frame_fraction: 0.02
      cooldown_s: 5.0
    storage:
      local_dir: /var/lib/horus/captures
      max_local_mb: 512
    heartbeat_interval_s: 60

Environment / loading precedence
---------------------------------
Configuration is loaded exclusively from the YAML file whose path is
passed to :func:`load`.  No environment-variable override layer is
applied at this level — callers are responsible for resolving the path
(e.g. from ``$HORUS_CONFIG``).

The MQTT subset of a station's config is just :class:`MqttBaseConfig`
from :mod:`sbo_shared` — re-exported here as :class:`MqttConfig` so
existing YAML files and call sites don't have to change. Any field
added to the shared base (TLS, QoS overrides, etc.) flows into horus
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from sbo_shared import MqttBaseConfig

# Re-export — horus.config.MqttConfig is preserved as the canonical
# import path for every existing call site.
MqttConfig = MqttBaseConfig
__all__ = [
    "CaptureConfig",
    "ClassifierConfig",
    "DetectorConfig",
    "HorusConfig",
    "MotionConfig",
    "MqttConfig",
    "StorageConfig",
    "load",
]


@dataclass(frozen=True)
class CaptureConfig:
    """Parameters controlling how the camera captures still frames.

    All fields have defaults so the ``capture:`` YAML block is optional;
    omitting it entirely is equivalent to listing every default explicitly.
    """

    width: int = 2304
    """Horizontal resolution in pixels passed to ``rpicam-still --width``.

    Default 2304 is the full-width native resolution of the IMX477 sensor.
    Valid range: 1–sensor-max (sensor-dependent; IMX477 max is 4056).
    """

    height: int = 1296
    """Vertical resolution in pixels passed to ``rpicam-still --height``.

    Default 1296 is the 16:9 crop of the IMX477 sensor at full width.
    Valid range: 1–sensor-max (IMX477 max is 3040).
    """

    jpeg_quality: int = 90
    """JPEG compression quality passed to ``rpicam-still -q``.

    Range 0–100; higher values produce larger files with less compression
    artefacts.  Default 90 is a good balance for scientific imagery where
    fine feather detail matters.
    """

    interval_s: float = 1.0
    """Seconds between successive capture attempts (units: seconds).

    The motion-gate loop sleeps for this duration between frames.  Lower
    values increase CPU/IO load and reduce the minimum detectable inter-
    event gap; values below ~0.3 s are not useful with ``rpicam-still``
    because the subprocess overhead dominates.  Default 1.0 s.
    """

    rpicam_extra_args: tuple[str, ...] = ()
    """Additional CLI arguments forwarded verbatim to ``rpicam-still``.

    Useful for sensor-specific tuning (e.g. ``("--gain", "2.0")`` or
    ``("--awbgains", "1.5,1.2")``).  Arguments are appended after all
    standard flags, so they can override horus's own settings.
    Default is an empty tuple (no extra args).
    """

    # ------------------------------------------------------------------
    # Picamera2 spike — optional low-resolution preview stream used by
    # the persistent :class:`horus.camera.Camera` for motion detection.
    # All fields default to zero so existing YAML (and the rpicam-still
    # legacy path) continues to work unchanged.  When ``lores_width``
    # and ``lores_height`` are both positive, :class:`Camera` configures
    # picamera2 in video mode with a dual-stream pipeline (full-res
    # ``main`` + small ``lores``) and starts a background thread that
    # samples the lores stream at ``preview_fps`` for motion analysis.
    # ------------------------------------------------------------------

    lores_width: int = 0
    """Width of the low-resolution motion-detection stream in pixels.

    Zero (default) disables the preview stream entirely — :class:`Camera`
    then behaves as a still-only session (commit-1 semantics).  Typical
    enabled value is 320.  Must be paired with a positive ``lores_height``.
    """

    lores_height: int = 0
    """Height of the low-resolution motion-detection stream in pixels.

    Zero disables the preview stream.  Typical enabled value is 180
    (320×180 → 16:9 at negligible cost).  Must be paired with a positive
    ``lores_width``.
    """

    preview_fps: float = 15.0
    """Target sample rate for the motion-detection preview thread (Hz).

    The camera's ISP runs the lores stream at its own internal rate
    (typically 30-60 fps); this knob just controls how often horus
    *pulls* a fresh frame for the motion gate.  15 fps is enough to
    catch a bird landing on a feeder without burning CPU on redundant
    frames.  Has no effect when the lores stream is disabled.
    """

    lens_position: float = 0.0
    """Manual-focus lens position in diopters (AfMode=Manual).

    Zero (default) means "don't touch focus" — picamera2's default
    continuous-AF behavior stays in effect.  Any positive value locks
    the camera to AfMode.Manual and sets ``LensPosition`` to that
    diopter value via ``picam2.set_controls`` after ``start()``.

    Feeder-at-33cm benchmarks: PDAF settled at LP 3.0 yesterday on
    the rpicam path.  This knob exists so the picamera2 path can
    match that without re-implementing the full AF window/range
    plumbing (``rpicam_extra_args`` flags don't apply to picamera2).
    """


@dataclass(frozen=True)
class MotionConfig:
    """Thresholds for the frame-differencing motion detector.

    A frame is considered to contain motion when the fraction of pixels
    whose absolute difference from the previous frame exceeds
    ``pixel_threshold`` is at least ``frame_fraction``.
    """

    pixel_threshold: int = 25
    """Per-pixel brightness change (0–255) that counts as "different".

    Pixels whose absolute grey-level delta between consecutive frames is
    greater than this value are counted as changed.  Lower values make
    the detector more sensitive to subtle movement (e.g. a distant bird)
    but also more prone to false triggers from lighting flicker.
    Default 25 (roughly 10 % of full scale).
    """

    frame_fraction: float = 0.02
    """Fraction of total pixels that must exceed ``pixel_threshold``
    for the frame to be classified as motion (units: fraction, 0–1).

    Default 0.02 means at least 2 % of pixels must change.  Increase to
    suppress small-area disturbances (insects near the lens); decrease to
    catch distant or small subjects.  Valid range: 0.0–1.0.
    """

    cooldown_s: float = 5.0
    """Minimum time between consecutive motion-event publications (units: seconds).

    After a motion event is published to MQTT, further triggers are
    suppressed for this many seconds.  Prevents a single protracted
    movement from flooding the broker.  Default 5.0 s.
    """


@dataclass(frozen=True)
class ClassifierConfig:
    """On-device bird-gate classifier settings.

    When ``enabled`` is False (the default), the whole block is a no-op —
    horus publishes every motion event exactly as before. This keeps
    the feature fully opt-in and lets a station fall back to Thoth-side
    classification by just flipping one flag.

    When enabled, every motion-triggered capture is classified locally
    *after* crop. The top-1 confidence is attached to the MQTT payload
    as ``bird_score`` regardless of the outcome, and events with a score
    below ``min_confidence`` are dropped before publish.
    """

    enabled: bool = False
    """Master switch. False → classifier never loads, bird_score absent.

    Default False so a horus install without the model on disk still
    boots cleanly. Flip to True in YAML once ``model_path`` and
    ``labels_path`` exist on the station.
    """

    model_path: Path = Path("/opt/horus/models/inat_bird_quant.tflite")
    """Absolute filesystem path to the ``.tflite`` model file.

    Default matches the Thoth install layout so the two services run
    the same model — on-device scores stay comparable to post-ingest
    scores recorded by Thoth. Override per-station if you want to
    experiment with a different model.
    """

    labels_path: Path = Path("/opt/horus/models/inat_bird_labels.txt")
    """Absolute path to the labels file (UTF-8, one label per line).

    ``labels[output_index]`` is the human-readable species name for
    that class. Default mirrors ``model_path``.
    """

    min_confidence: float = 0.10
    """Drop any capture whose top-1 classifier score falls below this
    threshold. Applied unconditionally — including when the object
    detector is enabled — so the classifier floor catches non-bird
    crud (apples, leaves, shadows) that the detector occasionally
    passes through at low confidence.

    Range: ``0.0`` (publish everything, score only — "dry run" mode)
    to ``1.0`` (publish nothing). Calibrated from Thoth historical
    data: real birds on the feeder score 0.42–0.62 cropped; non-bird
    objects that leak past the detector (apples, bark, sky) collapse
    to 0.03–0.08 because the iNat classifier has no confident label
    for them. 0.10 sits below real-bird scores and above garbage with
    comfortable headroom.

    Set to 0.0 to disable the floor entirely — useful for
    observability-only runs (attach bird_score to every event without
    gating) or to restore pre-2026-04-22 behavior where the classifier
    was purely a label attachment when the detector was live.
    """

    gated_archive_dir: Path | None = None
    """Optional directory where gated (dropped) crops are archived for
    human review of false-negative rate.

    When set, every capture that the classifier drops for being below
    ``min_confidence`` is copied (not moved — the original still gets
    deleted on the capture-local ring buffer) into this directory with
    the score + label encoded in the filename. A human can then flip
    through the archive and flag actual birds the gate missed.

    Layout: ``<archive_dir>/YYYY-MM-DD/<timestamp>_score-<sss>_<label>.jpg``.
    Day-level partitioning makes pruning cheap (whole directory unlink).

    Defaults to None → archiving disabled, matching the pre-feature
    behavior. Point this at a dedicated path like
    ``/var/lib/horus/gated`` to turn it on.
    """

    gated_archive_max_age_days: int = 7
    """Drop gated-archive day-directories older than this. Rolling
    window so the archive cannot grow unbounded while still giving a
    human time to review recent near-threshold decisions. Default 7d
    balances storage with realistic review cadence."""


@dataclass(frozen=True)
class DetectorConfig:
    """On-device object detector (bird/no-bird gate) settings.

    The detector replaces the species classifier as the primary gate
    when ``enabled``. It's a COCO-trained object detection model that
    returns a clean zero when no bird is present, where the species
    classifier would have returned a "least-wrong" species label with
    moderate confidence on any textured scene.

    When both the detector and the classifier are enabled, the gating
    decision comes from the detector; the classifier still runs on the
    same bytes to attach a species label to the published event (so
    Thoth keeps getting species hypotheses). When the detector is
    disabled (default), horus falls back to the legacy classifier-gate
    behavior — fully backwards compatible.
    """

    enabled: bool = False
    """Master switch. False → detector never loads, gate falls through
    to the classifier (legacy behavior). Default False so stations
    without the detector model on disk still boot.
    """

    model_path: Path = Path("/opt/horus/models/efficientdet_lite0.tflite")
    """Absolute path to the object-detector ``.tflite`` file.

    Designed for EfficientDet-Lite0 (COCO-90, 320×320, ~5MB) but any
    TFLite model with the standard 4-output layout
    (boxes, classes, scores, num_detections) works. See
    :class:`~horus.detector.BirdDetector` for layout requirements.
    """

    labels_path: Path = Path("/opt/horus/models/coco_labels.txt")
    """Absolute path to the COCO labels file, one class per line.

    The label string for the bird class must be exactly ``"bird"``
    (case-insensitive). BirdDetector resolves the bird class index by
    name at startup — hardcoding an index would silently return the
    wrong class on a different COCO variant.
    """

    min_score: float = 0.30
    """Minimum per-detection confidence for a bird to count.

    Starting point for EfficientDet-Lite0; tune from the gated-archive
    review. Higher → fewer false positives, more false negatives.
    Lower → more events published, more noise for Thoth to filter.
    """


@dataclass(frozen=True)
class StorageConfig:
    """Parameters for the local on-disk frame ring-buffer.

    Captured frames are written to ``local_dir`` and pruned when the
    total size exceeds ``max_local_mb``.  This directory is *not* the
    long-term store — images are forwarded to Banshee over MQTT and the
    local copy exists only for short-term buffering and post-mortem
    debugging.
    """

    local_dir: Path = Path("/var/lib/horus/captures")
    """Absolute path to the local ring-buffer directory (units: filesystem path).

    Sub-directories are created automatically per day (``YYYY-MM-DD/``).
    The directory and its parents are created on first write if they do
    not exist.  Default ``/var/lib/horus/captures`` follows the FHS
    convention for application state data.
    """

    max_local_mb: int = 512
    """Maximum total size of ``local_dir`` before old files are pruned (units: mebibytes).

    Pruning deletes the oldest files (by mtime) until the directory is
    back under this limit.  Default 512 MB is enough for roughly 30
    minutes of 1-fps full-resolution JPEG captures at quality 90.
    Set to a larger value on stations with ample storage; set to 0 to
    disable local buffering entirely (not recommended — you lose the
    post-mortem debugging window).
    """


@dataclass(frozen=True)
class HorusConfig:
    """Top-level configuration for a single horus station instance.

    Aggregates all sub-configs.  ``station`` and ``camera`` are required
    (no defaults); all sub-configs have their own defaults so the
    corresponding YAML blocks are optional.
    """

    station: str
    """Unique human-readable identifier for this station (e.g. ``"feeder-north"``).

    Used as part of MQTT topic paths and in log messages.  Must be a
    valid MQTT topic component: no ``+``, ``#``, or ``/`` characters.
    Required — no default.
    """

    camera: str
    """Camera model/profile identifier (e.g. ``"imx477"``).

    Currently informational only — stored in published metadata so
    downstream consumers can apply sensor-specific corrections.
    Required — no default.
    """

    mqtt: MqttConfig
    """MQTT broker connection parameters.

    See :class:`MqttConfig` (alias for :class:`sbo_shared.MqttBaseConfig`)
    for field documentation.  The ``mqtt:`` YAML block is required.
    """

    capture: CaptureConfig
    """Camera capture settings.  See :class:`CaptureConfig`."""

    motion: MotionConfig
    """Motion-detection thresholds.  See :class:`MotionConfig`."""

    storage: StorageConfig
    """Local ring-buffer storage settings.  See :class:`StorageConfig`."""

    classifier: ClassifierConfig = ClassifierConfig()
    """On-device bird-gate classifier.

    Defaults to disabled — stations without the model or the
    tflite-runtime wheel continue to work exactly as before. See
    :class:`ClassifierConfig`.
    """

    detector: DetectorConfig = DetectorConfig()
    """On-device COCO bird/no-bird object detector.

    When enabled, replaces the classifier as the gate — the classifier
    still runs (if enabled) to attach a species label to the published
    event. Defaults to disabled so existing stations fall through to
    the legacy classifier-gate behavior. See :class:`DetectorConfig`.
    """

    heartbeat_interval_s: int = 60
    """Seconds between MQTT heartbeat publishes (units: seconds).

    The station publishes a lightweight status message on this cadence so
    Banshee can detect stalled stations.  Default 60 s.  Set to 0 to
    disable heartbeats (not recommended in production).
    """


def load(path: Path | str) -> HorusConfig:
    """Load a station YAML config file and return a :class:`HorusConfig`.

    Reads the file at *path*, parses it with ``yaml.safe_load``, and
    constructs the frozen dataclass hierarchy.  The ``rpicam_extra_args``
    YAML value (a sequence) is coerced to a ``tuple``; ``storage.local_dir``
    is coerced from ``str`` to :class:`~pathlib.Path`.

    Args:
        path: Filesystem path to the YAML configuration file.  Accepts
            both :class:`str` and :class:`~pathlib.Path`.

    Returns:
        A fully populated, immutable :class:`HorusConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        KeyError: If a required top-level key (``station``, ``camera``,
            or ``mqtt``) is missing from the YAML document.
        TypeError: If a field value has the wrong type and cannot be
            coerced (e.g. a non-sequence value for ``rpicam_extra_args``).
    """
    data = yaml.safe_load(Path(path).read_text())

    mqtt = MqttConfig(**data["mqtt"])
    capture = CaptureConfig(
        **{
            k: (tuple(v) if k == "rpicam_extra_args" else v)
            for k, v in data.get("capture", {}).items()
        }
    )
    motion = MotionConfig(**data.get("motion", {}))
    storage_data = dict(data.get("storage", {}))
    if "local_dir" in storage_data:
        storage_data["local_dir"] = Path(storage_data["local_dir"])
    storage = StorageConfig(**storage_data)

    classifier_data = dict(data.get("classifier", {}))
    for key in ("model_path", "labels_path"):
        if key in classifier_data:
            classifier_data[key] = Path(classifier_data[key])
    # gated_archive_dir is optional — coerce only when present so YAML
    # users can leave it unset and get the None default.
    if classifier_data.get("gated_archive_dir") is not None:
        classifier_data["gated_archive_dir"] = Path(classifier_data["gated_archive_dir"])
    classifier = ClassifierConfig(**classifier_data)

    detector_data = dict(data.get("detector", {}))
    for key in ("model_path", "labels_path"):
        if key in detector_data:
            detector_data[key] = Path(detector_data[key])
    detector = DetectorConfig(**detector_data)

    return HorusConfig(
        station=data["station"],
        camera=data["camera"],
        mqtt=mqtt,
        capture=capture,
        motion=motion,
        storage=storage,
        classifier=classifier,
        detector=detector,
        heartbeat_interval_s=data.get("heartbeat_interval_s", 60),
    )
