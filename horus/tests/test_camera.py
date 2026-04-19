"""Tests for horus.camera — rpicam-still subprocess wrapper and sidecar metadata."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from horus import camera
from horus.config import CaptureConfig


def _fake_run_succeeds(*, image_bytes: bytes = b"\xff\xd8\xff\xd9", metadata: dict | None = None):
    """Return a patched subprocess.run that writes the expected outputs.

    The real rpicam-still writes both the JPEG and the --metadata file.
    Tests swap this in so we can assert on the command invocation while
    the filesystem side effects still look realistic to the caller.
    """
    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        # Mirror rpicam-still: write image at -o <path> and metadata at --metadata <path>.
        out_path = Path(cmd[cmd.index("-o") + 1])
        md_path = Path(cmd[cmd.index("--metadata") + 1])
        out_path.write_bytes(image_bytes)
        md_path.write_text(json.dumps(metadata or {}))
        completed = subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        return completed
    return _run


def test_capture_invokes_rpicam_with_metadata_flag(tmp_path: Path) -> None:
    out = tmp_path / "cap.jpg"
    cfg = CaptureConfig()
    fake = _fake_run_succeeds(metadata={"LensPosition": 5.1, "AfState": 2, "FocusFoM": 1234})

    with patch("horus.camera.subprocess.run", side_effect=fake) as mock_run:
        camera.capture(out, cfg)

    cmd = mock_run.call_args.args[0]
    assert "--metadata" in cmd
    md_arg = cmd[cmd.index("--metadata") + 1]
    assert md_arg == str(out) + ".json"
    assert "--metadata-format" in cmd
    assert cmd[cmd.index("--metadata-format") + 1] == "json"


def test_capture_writes_sidecar_and_logs_af_fields(tmp_path: Path, caplog) -> None:
    out = tmp_path / "cap.jpg"
    md = {"LensPosition": 5.1, "AfState": 2, "FocusFoM": 1234, "ExposureTime": 500}
    fake = _fake_run_succeeds(metadata=md)

    caplog.set_level("INFO", logger="horus.camera")
    with patch("horus.camera.subprocess.run", side_effect=fake):
        camera.capture(out, CaptureConfig())

    sidecar = camera.metadata_path_for(out)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == md

    # Only the configured AF subset shows up in the log summary.
    records = [r.message for r in caplog.records if "af metadata" in r.message]
    assert records, "expected an AF summary log line"
    assert "LensPosition" in records[0]
    assert "AfState" in records[0]
    assert "FocusFoM" in records[0]
    assert "ExposureTime" not in records[0]  # not in _AF_LOG_FIELDS


def test_capture_tolerates_corrupt_metadata_at_debug(tmp_path: Path, caplog) -> None:
    """Corrupt sidecar (partial write, killed mid-flush) logs at DEBUG, not WARN —
    this is usually transient and we don't want to page ops for noise.
    capture() still succeeds."""
    out = tmp_path / "cap.jpg"

    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\xff\xd8\xff\xd9")
        Path(cmd[cmd.index("--metadata") + 1]).write_text("not-json{")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    caplog.set_level("DEBUG", logger="horus.camera")
    with patch("horus.camera.subprocess.run", side_effect=_run):
        result = camera.capture(out, CaptureConfig())

    assert result == out
    af_lines = [r for r in caplog.records if "af metadata" in r.message]
    assert not af_lines
    # Corruption path: logged at DEBUG only, never at WARNING.
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings, f"unexpected WARNING on corrupt metadata: {[r.message for r in warnings]}"
    debugs = [r for r in caplog.records if r.levelname == "DEBUG" and "not valid JSON" in r.message]
    assert debugs, "expected DEBUG log for corrupt metadata"


def test_capture_warns_when_sidecar_is_missing(tmp_path: Path, caplog) -> None:
    """If rpicam-still exits 0 but writes no metadata file at all, we log
    at WARNING — this is structurally unexpected and ops should see it.
    capture() still succeeds so the pipeline isn't blocked."""
    out = tmp_path / "cap.jpg"

    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        # Write only the JPEG, skip the metadata file entirely.
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\xff\xd8\xff\xd9")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    caplog.set_level("DEBUG", logger="horus.camera")
    with patch("horus.camera.subprocess.run", side_effect=_run):
        result = camera.capture(out, CaptureConfig())

    assert result == out
    assert not camera.metadata_path_for(out).exists()
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "did not write metadata" in r.message]
    assert warnings, "expected WARNING when sidecar is absent"


def test_capture_raises_on_rpicam_failure(tmp_path: Path) -> None:
    out = tmp_path / "cap.jpg"

    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

    with patch("horus.camera.subprocess.run", side_effect=_run):
        with pytest.raises(camera.CameraError, match="boom"):
            camera.capture(out, CaptureConfig())


def test_discard_removes_both_image_and_sidecar(tmp_path: Path) -> None:
    img = tmp_path / "cap.jpg"
    sidecar = camera.metadata_path_for(img)
    img.write_bytes(b"\xff\xd8\xff\xd9")
    sidecar.write_text("{}")

    camera.discard(img)

    assert not img.exists()
    assert not sidecar.exists()


def test_discard_is_idempotent_on_missing_files(tmp_path: Path) -> None:
    img = tmp_path / "never_existed.jpg"
    # Should not raise.
    camera.discard(img)


def test_discard_removes_image_when_sidecar_absent(tmp_path: Path) -> None:
    img = tmp_path / "cap.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    # No sidecar created — mimics capture path that failed before metadata write.
    camera.discard(img)
    assert not img.exists()


# ---------------------------------------------------------------------------
# read_af_fields — public helper used by main._tick to attach AF to payload
# ---------------------------------------------------------------------------


def test_read_af_fields_returns_subset_when_sidecar_valid(tmp_path: Path) -> None:
    """Only fields in _AF_LOG_FIELDS are surfaced; the rest of the (verbose)
    libcamera metadata stays out of the MQTT payload to keep it compact."""
    img = tmp_path / "cap.jpg"
    camera.metadata_path_for(img).write_text(
        json.dumps({
            "LensPosition": 3.02,
            "AfState": 2,
            "FocusFoM": 1234,
            "ExposureTime": 500,
            "AnalogueGain": 2.0,
        })
    )
    result = camera.read_af_fields(img)
    assert result == {"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234}


def test_read_af_fields_returns_empty_when_no_af_keys(tmp_path: Path) -> None:
    """Sidecar exists and parses but has none of the AF fields → empty dict.
    Callers distinguish this from None (missing) — see events.py contract."""
    img = tmp_path / "cap.jpg"
    camera.metadata_path_for(img).write_text(json.dumps({"ExposureTime": 500}))
    result = camera.read_af_fields(img)
    assert result == {}


def test_read_af_fields_returns_none_when_sidecar_missing(
    tmp_path: Path, caplog
) -> None:
    """Missing sidecar → WARNING (structurally unexpected) + None return."""
    img = tmp_path / "cap.jpg"  # no sidecar written
    caplog.set_level("DEBUG", logger="horus.camera")
    result = camera.read_af_fields(img)
    assert result is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings, "expected WARNING on missing sidecar"


def test_read_af_fields_returns_none_when_sidecar_corrupt(
    tmp_path: Path, caplog
) -> None:
    """Corrupt sidecar (partial write) → DEBUG (transient) + None return.
    Mirrors the log-level policy used by _log_af_summary."""
    img = tmp_path / "cap.jpg"
    camera.metadata_path_for(img).write_text("not-json{")
    caplog.set_level("DEBUG", logger="horus.camera")
    result = camera.read_af_fields(img)
    assert result is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings, "corrupt sidecar must log at DEBUG only, never WARN"
    debugs = [r for r in caplog.records if r.levelname == "DEBUG" and "not valid JSON" in r.message]
    assert debugs


def test_read_af_fields_returns_none_on_other_oserror(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    """Permissions / EIO / any non-ENOENT OSError is structurally wrong —
    log at WARNING (like missing sidecar) and return None. Covers the
    else-branch of the three exception handlers in read_af_fields."""
    img = tmp_path / "cap.jpg"
    camera.metadata_path_for(img).write_text("{}")  # exists, is valid

    original_open = Path.open

    def _boom_on_sidecar(self, *args, **kwargs):
        if self.suffix == ".json":
            raise PermissionError("EACCES: no read bit")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _boom_on_sidecar)
    caplog.set_level("DEBUG", logger="horus.camera")
    result = camera.read_af_fields(img)
    assert result is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "could not read metadata" in r.message]
    assert warnings, "unexpected OSError must log at WARNING"


# ---------------------------------------------------------------------------
# Camera (persistent picamera2 session) — uses FakePicamera2 so tests run
# without the ARM-only wheel.  The factory seam
# ``camera._picamera2_factory`` is monkeypatched to return FakePicamera2
# instead of the real class.
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for a picamera2 ``CompletedRequest``."""

    def __init__(self, *, metadata: dict, image_bytes: bytes = b"\xff\xd8\xff\xd9"):
        self._metadata = metadata
        self._image_bytes = image_bytes
        self.released = False
        self.saved_streams: list[str] = []

    def save(self, stream_name: str, path: str) -> None:
        self.saved_streams.append(stream_name)
        Path(path).write_bytes(self._image_bytes)

    def get_metadata(self) -> dict:
        return dict(self._metadata)

    def release(self) -> None:
        self.released = True


class FakePicamera2:
    """Stub that lets tests exercise :class:`horus.camera.Camera` without
    a real libcamera stack.

    Exposes the subset of the picamera2 API that :class:`Camera`
    touches: ``create_still_configuration``, ``configure``, ``start``,
    ``stop``, ``close``, ``options``, ``capture_request``.  Tests can
    inspect the recorded calls (``configure_calls``, ``started``,
    ``closed``, ``requests_issued``) to verify the session lifecycle.
    """

    default_metadata = {
        "LensPosition": 5.1,
        "AfState": 2,
        "FocusFoM": 1234,
        "ExposureTime": 500,
    }

    def __init__(self, *, metadata: dict | None = None):
        self.options: dict = {}
        self.started = False
        self.closed = False
        self.configure_calls: list[dict] = []
        self.requests_issued = 0
        self._metadata = metadata if metadata is not None else dict(self.default_metadata)
        # Failure injection knobs — tests flip these to exercise the
        # error paths in Camera.start / Camera.capture.
        self.fail_on_start: Exception | None = None
        self.fail_on_request: Exception | None = None
        self.fail_on_save: Exception | None = None
        self.fail_on_metadata: Exception | None = None

    def create_still_configuration(self, **kwargs) -> dict:
        return {"kind": "still", **kwargs}

    def configure(self, config: dict) -> None:
        self.configure_calls.append(config)

    def start(self) -> None:
        if self.fail_on_start is not None:
            raise self.fail_on_start
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def capture_request(self):
        if self.fail_on_request is not None:
            raise self.fail_on_request
        self.requests_issued += 1
        req = _FakeRequest(metadata=self._metadata)
        if self.fail_on_save is not None:
            err = self.fail_on_save

            def _boom(*_a, **_k):
                raise err

            req.save = _boom  # type: ignore[method-assign]
        if self.fail_on_metadata is not None:
            err_md = self.fail_on_metadata

            def _boom_md():
                raise err_md

            req.get_metadata = _boom_md  # type: ignore[method-assign]
        return req


def _install_fake_picamera2(monkeypatch, instance: FakePicamera2 | None = None) -> FakePicamera2:
    """Wire ``camera._picamera2_factory`` to return ``instance`` (or a
    fresh FakePicamera2) on next call.  The factory returns a *class*,
    so we synthesize a zero-arg constructor that hands back the
    pre-built instance — tests can then inspect that instance
    directly."""
    picam = instance or FakePicamera2()

    class _Handle:
        def __new__(cls):
            return picam

    monkeypatch.setattr(camera, "_picamera2_factory", lambda: _Handle)
    return picam


def test_camera_start_configures_and_starts_picamera2(monkeypatch) -> None:
    picam = _install_fake_picamera2(monkeypatch)
    cam = camera.Camera(CaptureConfig(width=1600, height=900, jpeg_quality=85))

    cam.start()

    assert picam.started is True
    assert picam.options["quality"] == 85
    assert len(picam.configure_calls) == 1
    cfg = picam.configure_calls[0]
    # The "main" stream carries the configured resolution into the session.
    assert cfg["main"]["size"] == (1600, 900)


def test_camera_start_is_idempotent(monkeypatch) -> None:
    picam = _install_fake_picamera2(monkeypatch)
    cam = camera.Camera(CaptureConfig())

    cam.start()
    cam.start()  # must not re-configure nor re-start

    assert len(picam.configure_calls) == 1


def test_camera_start_raises_camera_error_on_failure(monkeypatch) -> None:
    """A half-configured picamera2 must be closed before surfacing the
    error — otherwise the /dev/video* fds leak and the sensor wedges
    until reboot.  Camera.start() catches, closes, and re-raises as
    CameraError so the daemon can degrade."""
    picam = _install_fake_picamera2(monkeypatch)
    picam.fail_on_start = RuntimeError("camera busy")
    cam = camera.Camera(CaptureConfig())

    with pytest.raises(camera.CameraError, match="camera busy"):
        cam.start()

    assert picam.closed is True


def test_camera_stop_is_idempotent_and_swallows_errors(monkeypatch, caplog) -> None:
    picam = _install_fake_picamera2(monkeypatch)
    cam = camera.Camera(CaptureConfig())
    cam.start()

    cam.stop()
    cam.stop()  # must not raise on a stopped camera

    assert picam.started is False


def test_camera_capture_writes_image_and_sidecar(tmp_path: Path, monkeypatch, caplog) -> None:
    picam = _install_fake_picamera2(monkeypatch)
    cam = camera.Camera(CaptureConfig())
    cam.start()

    out = tmp_path / "cap.jpg"
    caplog.set_level("INFO", logger="horus.camera")
    result = cam.capture(out)

    assert result == out
    assert out.exists() and out.stat().st_size > 0
    sidecar = camera.metadata_path_for(out)
    assert sidecar.exists()
    md = json.loads(sidecar.read_text())
    assert md["LensPosition"] == 5.1
    # AF log line must fire so ops can grep for focus state.
    af_logs = [r for r in caplog.records if "af metadata" in r.message]
    assert af_logs, "expected AF summary log line after capture"


def test_camera_capture_before_start_raises(tmp_path: Path) -> None:
    cam = camera.Camera(CaptureConfig())
    with pytest.raises(camera.CameraError, match="before Camera.start"):
        cam.capture(tmp_path / "cap.jpg")


def test_camera_capture_releases_request_on_save_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """Even if save() explodes mid-flight, the picamera2 request MUST
    be released — otherwise the buffer pool drains and subsequent
    captures block forever.  Verifies the try/finally contract."""
    picam = _install_fake_picamera2(monkeypatch)
    picam.fail_on_save = RuntimeError("disk full")
    cam = camera.Camera(CaptureConfig())
    cam.start()

    with pytest.raises(camera.CameraError, match="disk full"):
        cam.capture(tmp_path / "cap.jpg")

    assert picam.requests_issued == 1


def test_camera_capture_tolerates_metadata_read_failure(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A transient metadata-read failure must not drop the frame —
    the JPEG is already on disk and useful.  Sidecar is skipped,
    WARNING logged, capture returns the image path."""
    picam = _install_fake_picamera2(monkeypatch)
    picam.fail_on_metadata = RuntimeError("metadata not ready")
    cam = camera.Camera(CaptureConfig())
    cam.start()

    out = tmp_path / "cap.jpg"
    caplog.set_level("WARNING", logger="horus.camera")
    result = cam.capture(out)

    assert result == out
    assert out.exists()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("get_metadata" in r.message for r in warnings)


def test_camera_context_manager_starts_and_stops(monkeypatch) -> None:
    picam = _install_fake_picamera2(monkeypatch)
    with camera.Camera(CaptureConfig()) as cam:
        assert picam.started is True
        assert cam is not None
    assert picam.started is False


def test_picamera2_factory_raises_camera_error_when_missing(monkeypatch) -> None:
    """On a dev machine without picamera2, the factory must raise
    CameraError (not ImportError) so callers get a single exception
    type to catch and degrade on."""
    import builtins
    original_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "picamera2":
            raise ImportError("No module named 'picamera2'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(camera.CameraError, match="picamera2 is not installed"):
        camera._picamera2_factory()


# ---------------------------------------------------------------------------
# Dual-stream (main + lores) preview — used by the motion path to avoid
# paying the subprocess / file-decode tax for every motion check.  We
# extend FakePicamera2 with a video-config entrypoint + capture_array so
# the preview thread has something to pull from.
# ---------------------------------------------------------------------------

import numpy as np


class _FakeDualPicamera2(FakePicamera2):
    """FakePicamera2 with video-configuration + capture_array support."""

    def __init__(self, *, lores_size: tuple[int, int] = (320, 180), **kwargs):
        super().__init__(**kwargs)
        self.video_config_calls: list[dict] = []
        self.lores_size = lores_size
        self.capture_array_calls = 0
        # Successive lores frames handed out by capture_array — tests can
        # pre-seed this with specific frames to exercise the preview loop.
        self.lores_frames: list[np.ndarray] = []

    def create_video_configuration(self, **kwargs) -> dict:
        self.video_config_calls.append(kwargs)
        return {"kind": "video", **kwargs}

    def capture_array(self, stream_name: str = "main") -> np.ndarray:
        self.capture_array_calls += 1
        if self.lores_frames:
            frame = self.lores_frames.pop(0)
            # YUV420 physical layout is (H*3/2, W). For the Y-plane
            # slicing logic in Camera._preview_loop to work we return a
            # 2D array of that stacked shape. Tests construct the frame
            # with the expected shape when they care about content.
            return frame
        lw, lh = self.lores_size
        return np.zeros((lh * 3 // 2, lw), dtype=np.uint8)


def _install_fake_dual(monkeypatch, instance: _FakeDualPicamera2 | None = None) -> _FakeDualPicamera2:
    """Like :func:`_install_fake_picamera2` but hands out the dual-stream
    fake so tests can inspect ``create_video_configuration`` calls and
    seed lores frames for the preview thread."""
    picam = instance or _FakeDualPicamera2()

    class _Handle:
        def __new__(cls):
            return picam

    monkeypatch.setattr(camera, "_picamera2_factory", lambda: _Handle)
    return picam


def test_camera_lores_enabled_uses_video_configuration(monkeypatch) -> None:
    """Both lores dims set → video_configuration with dual streams.
    create_still_configuration must not be called — it'd tear down the
    preview between captures, defeating the whole purpose."""
    picam = _install_fake_dual(monkeypatch)
    cfg = CaptureConfig(
        width=2304, height=1296,
        lores_width=320, lores_height=180,
        preview_fps=30.0,
    )
    cam = camera.Camera(cfg)

    cam.start()
    try:
        assert len(picam.video_config_calls) == 1
        vcfg = picam.video_config_calls[0]
        assert vcfg["main"]["size"] == (2304, 1296)
        assert vcfg["lores"]["size"] == (320, 180)
        assert vcfg["lores"]["format"] == "YUV420"
        # Still-config path must NOT be used when lores is enabled.
        assert picam.configure_calls[0]["kind"] == "video"
    finally:
        cam.stop()


def test_camera_lores_disabled_uses_still_configuration(monkeypatch) -> None:
    """Default CaptureConfig has lores_width=0 → still-only path, no
    preview thread, no video_configuration."""
    picam = _install_fake_dual(monkeypatch)
    cam = camera.Camera(CaptureConfig())  # lores_width=0 by default

    cam.start()
    try:
        assert picam.video_config_calls == []
        assert picam.configure_calls[0]["kind"] == "still"
        assert cam._preview_thread is None
    finally:
        cam.stop()


def test_camera_only_one_lores_dim_set_falls_back_to_still(monkeypatch) -> None:
    """Width set, height zero → treat as disabled, not crash. A half-
    configured lores stream is the kind of thing an ops typo produces
    and we'd rather run still-only than wedge the daemon."""
    picam = _install_fake_dual(monkeypatch)
    cam = camera.Camera(CaptureConfig(lores_width=320, lores_height=0))

    cam.start()
    try:
        assert picam.video_config_calls == []
        assert cam.lores_enabled is False
    finally:
        cam.stop()


def test_camera_latest_lores_returns_frame_from_preview_thread(monkeypatch) -> None:
    """End-to-end: Camera starts, preview thread pulls a seeded frame
    via capture_array, latest_lores() exposes it to callers."""
    picam = _install_fake_dual(monkeypatch, _FakeDualPicamera2(lores_size=(320, 180)))
    # Seed a distinctive frame: Y-plane gradient 0..LH-1 replicated row-wise.
    y_plane = np.arange(180, dtype=np.uint8).reshape(-1, 1) * np.ones((1, 320), dtype=np.uint8)
    # YUV420 stacked: Y at top (180 rows), UV at bottom (90 rows). Fill UV with 0.
    stacked = np.vstack([y_plane, np.zeros((90, 320), dtype=np.uint8)])
    # Seed multiple copies so the preview thread doesn't run out.
    picam.lores_frames = [stacked.copy() for _ in range(10)]

    cfg = CaptureConfig(
        lores_width=320, lores_height=180,
        preview_fps=200.0,  # aggressive sampling so the test doesn't wait long
    )
    cam = camera.Camera(cfg)
    cam.start()
    try:
        # Poll up to 1s for the preview thread to land a frame.
        deadline = time.monotonic() + 1.0
        latest = None
        while time.monotonic() < deadline:
            latest = cam.latest_lores()
            if latest is not None:
                break
            time.sleep(0.01)
        assert latest is not None, "preview thread failed to publish a frame"
        ts, frame = latest
        assert ts > 0.0
        assert frame.shape == (180, 320)
        # Confirm we got the Y-plane specifically, not the stacked YUV420
        # (row-index-gradient would only look right in the Y plane).
        assert frame[0, 0] == 0
        assert frame[179, 0] == 179
    finally:
        cam.stop()


def test_camera_latest_lores_returns_none_when_stream_disabled(monkeypatch) -> None:
    """Still-only Camera never populates latest_lores — callers must
    handle the None case cleanly."""
    _install_fake_dual(monkeypatch)
    cam = camera.Camera(CaptureConfig())
    cam.start()
    try:
        time.sleep(0.05)  # give any rogue thread a chance to misbehave
        assert cam.latest_lores() is None
    finally:
        cam.stop()


def test_camera_preview_loop_tolerates_capture_errors(monkeypatch, caplog) -> None:
    """A transient capture_array failure must not kill the preview
    stream — the loop logs and continues. Otherwise one bad frame
    takes out motion detection for the rest of the daemon's life."""
    import time as _time

    picam = _install_fake_dual(monkeypatch)
    failures_remaining = {"n": 3}
    original_capture = picam.capture_array

    def _flaky(stream_name: str = "main"):
        if failures_remaining["n"] > 0:
            failures_remaining["n"] -= 1
            raise RuntimeError("driver glitch")
        return original_capture(stream_name)

    picam.capture_array = _flaky  # type: ignore[method-assign]

    cam = camera.Camera(CaptureConfig(
        lores_width=320, lores_height=180, preview_fps=200.0,
    ))
    caplog.set_level("INFO", logger="horus.camera")
    cam.start()
    try:
        deadline = _time.monotonic() + 1.0
        latest = None
        while _time.monotonic() < deadline:
            latest = cam.latest_lores()
            if latest is not None:
                break
            _time.sleep(0.01)
        assert latest is not None, "preview thread died on first capture error"
        # The three failures should have been logged at EXCEPTION/ERROR level.
        error_lines = [r for r in caplog.records if "capture_array" in r.message]
        assert len(error_lines) >= 1
    finally:
        cam.stop()


def test_camera_stop_joins_preview_thread(monkeypatch) -> None:
    """stop() must signal and join the preview thread BEFORE closing
    picamera2 — otherwise the loop calls capture_array on a closed
    pipeline and spams the log on shutdown."""
    picam = _install_fake_dual(monkeypatch)
    cam = camera.Camera(CaptureConfig(
        lores_width=320, lores_height=180, preview_fps=100.0,
    ))
    cam.start()
    assert cam._preview_thread is not None and cam._preview_thread.is_alive()

    cam.stop()

    assert cam._preview_thread is None
    assert picam.started is False
    assert picam.closed is True


def test_camera_capture_still_works_in_dual_stream_mode(monkeypatch, tmp_path: Path) -> None:
    """Still captures via capture_request() keep working alongside the
    preview thread — this is the whole point of the dual-stream config."""
    picam = _install_fake_dual(monkeypatch)
    cam = camera.Camera(CaptureConfig(
        lores_width=320, lores_height=180, preview_fps=60.0,
    ))
    cam.start()
    try:
        out = tmp_path / "cap.jpg"
        result = cam.capture(out)
        assert result == out
        assert out.exists()
    finally:
        cam.stop()
