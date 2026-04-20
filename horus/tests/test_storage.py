"""Tests for horus.storage — gated-archive helpers.

The gated archive is how we detect classifier false negatives: every
capture the on-device gate drops gets copied here with score + label in
the filename so a human can flip through the day's misses and flag any
actual birds the gate rejected. Pruning is time-windowed so the archive
stays bounded without manual cleanup.

Covered here:
* ``save_gated_sample`` — filename encoding (score + sanitized label),
  day-partitioned layout, parent directory creation.
* ``prune_gated`` — deletes date-shaped dirs older than max_age_days
  and leaves everything else alone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from horus.storage import _sanitize_label, prune_gated, save_gated_sample


def _write_fake_jpeg(path: Path) -> None:
    """Minimal JPEG-ish bytes — enough for shutil.copy to round-trip.

    We don't care about the contents for archive tests — the archive
    treats images as opaque byte blobs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")


def test_save_gated_sample_writes_score_and_label_to_filename(tmp_path):
    archive = tmp_path / "gated"
    src = tmp_path / "capture.jpg"
    _write_fake_jpeg(src)

    ts = datetime(2026, 4, 18, 17, 2, 45, tzinfo=timezone.utc)
    dest = save_gated_sample(
        archive, src, score=0.287, label="Ardea alba (Great Egret)", timestamp=ts
    )

    assert dest.exists()
    assert dest.parent == archive / "2026-04-18"
    # Score encoded to 3 decimals; label sanitized to filename-safe.
    assert "score-0.287" in dest.name
    assert "Ardea-alba-Great-Egret" in dest.name
    assert dest.name.startswith("20260418T170245")
    assert dest.suffix == ".jpg"


def test_save_gated_sample_handles_missing_label(tmp_path):
    """A None label still has to produce a valid filename. The reviewer
    sees ``unknown`` so they know the model returned no top-1."""
    archive = tmp_path / "gated"
    src = tmp_path / "capture.jpg"
    _write_fake_jpeg(src)

    dest = save_gated_sample(archive, src, score=0.1, label=None)
    assert "unknown" in dest.name
    assert dest.exists()


def test_save_gated_sample_preserves_image_bytes(tmp_path):
    """The archive is a copy, not a move — original must stay intact and
    the copy must be byte-identical so reviewers aren't looking at a
    re-encoded / lossy version of what the classifier actually saw."""
    archive = tmp_path / "gated"
    src = tmp_path / "capture.jpg"
    payload = b"\xff\xd8\xff\xe0" + b"ORIGINAL-IMAGE-BYTES" + b"\xff\xd9"
    src.write_bytes(payload)

    dest = save_gated_sample(archive, src, score=0.2, label="junco")
    assert src.exists(), "source must still exist — archive is a copy"
    assert dest.read_bytes() == payload


def test_save_gated_sample_creates_nested_dirs(tmp_path):
    """Archive root doesn't need to exist in advance — the daemon may
    come up on a fresh host."""
    archive = tmp_path / "does" / "not" / "exist"
    src = tmp_path / "capture.jpg"
    _write_fake_jpeg(src)

    dest = save_gated_sample(archive, src, score=0.3, label="x")
    assert dest.exists()
    assert archive.exists() and archive.is_dir()


def test_sanitize_label_collapses_punctuation_runs():
    """Regression on filename tidiness: sanitization must collapse
    runs of dashes from stripped punctuation so we don't get filenames
    like ``----great-----egret----.jpg``."""
    assert _sanitize_label("Ardea alba (Great Egret)") == "Ardea-alba-Great-Egret"
    assert _sanitize_label("   spaced   out   ") == "spaced-out"
    assert _sanitize_label("!!!???") == "unknown"  # nothing survives sanitization
    assert _sanitize_label("") == "unknown"
    assert _sanitize_label(None) == "unknown"


def test_prune_gated_removes_old_day_dirs(tmp_path):
    archive = tmp_path / "gated"
    today = datetime.now(timezone.utc).date()
    # 10 days old — should be pruned with max_age_days=7.
    old_day = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    # 2 days old — should survive.
    recent_day = (today - timedelta(days=2)).strftime("%Y-%m-%d")

    (archive / old_day).mkdir(parents=True)
    (archive / old_day / "capture.jpg").write_bytes(b"x")
    (archive / recent_day).mkdir(parents=True)
    (archive / recent_day / "capture.jpg").write_bytes(b"y")

    removed = prune_gated(archive, max_age_days=7)
    assert removed == 1
    assert not (archive / old_day).exists()
    assert (archive / recent_day).exists()


def test_prune_gated_leaves_non_date_dirs_alone(tmp_path):
    """An operator may drop a README or review-notes dir into the
    archive root. We must not nuke those — only YYYY-MM-DD directories
    are eligible for pruning."""
    archive = tmp_path / "gated"
    archive.mkdir()
    old_day = (datetime.now(timezone.utc).date() - timedelta(days=99)).strftime("%Y-%m-%d")
    (archive / old_day).mkdir()
    (archive / "README.md").write_text("reviewer notes live here")
    (archive / "flagged-as-bird").mkdir()  # reviewer-organized subfolder

    removed = prune_gated(archive, max_age_days=7)
    assert removed == 1
    assert (archive / "README.md").exists()
    assert (archive / "flagged-as-bird").exists()


def test_prune_gated_on_missing_archive_is_noop(tmp_path):
    """A station that hasn't gated anything yet has no archive dir. The
    prune call must succeed silently rather than the daemon having to
    probe for existence itself."""
    assert prune_gated(tmp_path / "never-created", max_age_days=7) == 0
