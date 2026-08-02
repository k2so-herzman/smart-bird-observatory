"""Thoth image classifier package.

The classifier is deployed as the ``thoth-classify.service`` systemd unit.
It polls the SQLite event store for image rows where ``classified_at IS NULL``,
fetches the media from the blob store, runs a pluggable :class:`Classifier`, and
writes ``species`` + ``confidence`` + ``classified_at`` back to the row.

Two design choices worth calling out:

* **DB polling instead of an MQTT queue.** The design doc mentions a
  "classify queue" but for Phase 1 a simple polling loop against the
  authoritative index is dramatically simpler: no new topic constants,
  no second subscriber wiring, and retry-on-crash is free (the next tick
  just picks up where the last one stopped). An MQTT queue can land later
  if poll latency becomes a real problem.

* **Pluggable model, Dummy default.** The :class:`~.model.Classifier`
  protocol separates the pipeline (blob fetch + DB write) from the
  model so the pipeline can be merged, deployed, and exercised end-to-end
  before a real bird-ID model is chosen. The default
  :class:`~.model.DummyClassifier` writes ``species="unclassified"``
  with ``confidence=0.0`` — enough to mark rows processed without
  claiming identifications we cannot back up.
"""

from .model import Classifier, ClassificationResult, DummyClassifier
from .worker import ClassifierWorker

__all__ = [
    "Classifier",
    "ClassificationResult",
    "ClassifierWorker",
    "DummyClassifier",
]
