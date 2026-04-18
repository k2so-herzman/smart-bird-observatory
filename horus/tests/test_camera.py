"""Tests for horus.camera — rpicam-still subprocess wrapper and sidecar metadata."""

from __future__ import annotations

import json
import subprocess
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
