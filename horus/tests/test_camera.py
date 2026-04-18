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


def test_capture_tolerates_missing_or_corrupt_metadata(tmp_path: Path, caplog) -> None:
    """If rpicam-still exits 0 but the metadata file is malformed, we still
    return the image path — metadata logging is diagnostic, not required."""
    out = tmp_path / "cap.jpg"

    def _run(cmd, capture_output, text, timeout):  # noqa: ARG001
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\xff\xd8\xff\xd9")
        Path(cmd[cmd.index("--metadata") + 1]).write_text("not-json{")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    caplog.set_level("DEBUG", logger="horus.camera")
    with patch("horus.camera.subprocess.run", side_effect=_run):
        result = camera.capture(out, CaptureConfig())

    assert result == out
    # No AF summary logged because parsing failed — but we didn't raise.
    af_lines = [r for r in caplog.records if "af metadata" in r.message]
    assert not af_lines


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
