"""Tests for Daemon.run() loop-level exception resilience.

The capture loop is the longest-lived thing in this process.  If a
single bad frame (corrupt JPEG, disk ENOSPC, transient MQTT type
error, numpy NaN from a flaky sensor read) escapes unhandled out of
``_tick`` / ``_heartbeat_maybe`` / ``storage.prune`` /
``_prune_gated_maybe``, the whole daemon exits — systemd restarts
us, we lose the warmup window, we miss birds.

These tests pin the contract:

1. An ``Exception`` from any loop-body step is caught, logged, and
   the loop continues.  The daemon doesn't exit, doesn't crash,
   doesn't drop the camera session.
2. ``KeyboardInterrupt`` / ``SystemExit`` still unblock run() —
   we catch ``Exception``, not ``BaseException``, on purpose.

Motivated by an incident where two ``save_gated_sample`` tracebacks
correlated with the Pi going unreachable; a belt-and-suspenders
guard here makes the daemon strictly more resilient and costs
nothing on the happy path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from horus.config import (
    CaptureConfig,
    HorusConfig,
    MotionConfig,
    MqttConfig,
    StorageConfig,
)
from horus.main import Daemon


@pytest.fixture
def cfg(tmp_path: Path) -> HorusConfig:
    # interval_s=0 so the loop doesn't add wall-clock delay between
    # ticks — we drive iteration count via _stop below.
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(interval_s=0.0),
        motion=MotionConfig(cooldown_s=10.0),
        storage=StorageConfig(local_dir=tmp_path),
        heartbeat_interval_s=999.0,  # suppress heartbeat during test
    )


def _make_daemon(cfg: HorusConfig) -> Daemon:
    """Build a Daemon with a mocked bus — avoids a real MQTT connection."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.dropped_publishes = 0
    # _maybe_start_camera would try to open picamera2 hardware; skip it.
    daemon._maybe_start_camera = MagicMock()  # type: ignore[method-assign]
    return daemon


def test_run_survives_tick_exception(cfg, caplog):
    """_tick() raising must NOT terminate the loop."""
    daemon = _make_daemon(cfg)

    iterations = {"n": 0}

    def ticking_bomb() -> None:
        iterations["n"] += 1
        if iterations["n"] >= 3:
            daemon._stop = True  # exit after a few iterations
        raise RuntimeError("simulated _tick failure")

    daemon._tick = ticking_bomb  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR, logger="horus"), \
         patch("horus.main.storage.prune"):
        rc = daemon.run()

    assert rc == 0, "run() should exit cleanly when _stop is set"
    assert iterations["n"] >= 3, "loop must have kept iterating after exceptions"
    # The guard logs "unhandled exception in capture loop; continuing" —
    # verify at least one record was captured with the traceback.
    messages = [r.getMessage() for r in caplog.records]
    assert any("unhandled exception in capture loop" in m for m in messages), (
        f"expected loop-guard log record; got: {messages}"
    )


def test_run_survives_storage_prune_exception(cfg, caplog):
    """A failure in storage.prune() also must not kill the daemon."""
    daemon = _make_daemon(cfg)
    daemon._tick = MagicMock()  # type: ignore[method-assign]

    iterations = {"n": 0}

    def pruning_bomb(*_args, **_kwargs) -> None:
        iterations["n"] += 1
        if iterations["n"] >= 3:
            daemon._stop = True
        raise OSError("simulated disk pressure")

    with caplog.at_level(logging.ERROR, logger="horus"), \
         patch("horus.main.storage.prune", side_effect=pruning_bomb):
        rc = daemon.run()

    assert rc == 0
    assert daemon._tick.call_count >= 3, "tick must still run each iteration"


def test_run_propagates_keyboard_interrupt(cfg):
    """Ctrl-C / systemd TERM must still unblock run().

    We catch ``Exception``, not ``BaseException`` — so KeyboardInterrupt
    and SystemExit still walk the stack and terminate the loop, which
    is what systemd's shutdown sequence relies on.
    """
    daemon = _make_daemon(cfg)
    daemon._tick = MagicMock(side_effect=KeyboardInterrupt)  # type: ignore[method-assign]

    with patch("horus.main.storage.prune"), \
         pytest.raises(KeyboardInterrupt):
        daemon.run()
