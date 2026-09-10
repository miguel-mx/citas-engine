"""Live progress for a running analysis.

/analyze answers once, at the end, and the heavy part takes minutes — so until now
the caller had nothing to show but "solicitado". This keeps a small record of where
each run is, which SIAB polls on a separate request while it waits.

Deliberately in-process and deliberately not authoritative: it is a courtesy for
the person watching the page, never an input to the analysis. If it is missing,
stale, or lost to a restart, the run is unaffected and the UI simply shows less.
That is why nothing here raises — a progress bug must not be able to fail an
analysis that took twenty minutes to compute.

Scope: one uvicorn process. The service runs single-process on loopback (see the
systemd unit), so a poll always reaches the worker doing the job. Running multiple
workers would need this in Redis instead.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# Runs older than this are forgotten. Comfortably longer than any analysis: the
# HTTP client that drives one gives up at 900 s.
_TTL_SECONDS = 3600

_lock = threading.Lock()
_runs: dict[str, "Progress"] = {}


@dataclass
class Progress:
    """Where one analysis is right now."""

    phase: str = "starting"
    done: int = 0
    total: int = 0
    detail: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "done": self.done,
            "total": self.total,
            "detail": self.detail,
            "elapsed_seconds": round(time.time() - self.started_at, 1),
            # Only the fan-out over articles has a real denominator; everything else
            # reports its phase and no fraction, rather than inventing one.
            "percent": round(100 * self.done / self.total) if self.total > 0 else None,
        }


def start(job_id: str | None) -> None:
    if not job_id:
        return
    with _lock:
        _prune()
        _runs[job_id] = Progress()


def set_phase(
    job_id: str | None,
    phase: str,
    done: int = 0,
    total: int = 0,
    detail: str | None = None,
) -> None:
    if not job_id:
        return
    with _lock:
        run = _runs.get(job_id)
        if run is None:
            run = _runs[job_id] = Progress()
        run.phase = phase
        run.done = done
        run.total = total
        run.detail = detail
        run.updated_at = time.time()


def advance(job_id: str | None, done: int) -> None:
    """Move the counter without touching the phase — the per-article tick."""
    if not job_id:
        return
    with _lock:
        run = _runs.get(job_id)
        if run is not None:
            run.done = done
            run.updated_at = time.time()


def get(job_id: str) -> dict | None:
    with _lock:
        run = _runs.get(job_id)
        return run.as_dict() if run else None


def finish(job_id: str | None) -> None:
    """Drop the record; the caller now has the real answer."""
    if not job_id:
        return
    with _lock:
        _runs.pop(job_id, None)


def _prune() -> None:
    """Called under the lock, on the rare path (a run starting)."""
    cutoff = time.time() - _TTL_SECONDS
    for job_id in [k for k, v in _runs.items() if v.updated_at < cutoff]:
        del _runs[job_id]
