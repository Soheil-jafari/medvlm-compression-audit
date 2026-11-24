"""A training stage that stops cleanly before its session's time limit (2026-11-24).

E4's quantization-aware training takes about 7 to 14 h (an estimate), longer than the sessions of
about 6 h the team runs in parallel. With ``MEDCOMPRESS_STOP_AFTER_HOURS`` set, the training saves
its state after the update that crosses the limit and the stage raises :class:`SessionPaused`;
the stage exits with ``cli.PAUSED_EXIT_CODE``, the card stops there and exports the resume
state as a hand-off, and the next session attaches it and continues. The limit is read from the
environment, not the config, so it is not part of the run's resume fingerprint.
"""

from __future__ import annotations

import os

STOP_AFTER_ENV = "MEDCOMPRESS_STOP_AFTER_HOURS"


class SessionPaused(Exception):
    """The stage saved its state and stopped on the session's time limit."""


def stop_after_seconds() -> float | None:
    """The training time limit of this session in seconds (None: no limit)."""
    value = os.environ.get(STOP_AFTER_ENV, "").strip()
    if not value:
        return None
    hours = float(value)
    if hours <= 0:
        raise ValueError(f"{STOP_AFTER_ENV} must be positive, got {value!r}")
    return 3600.0 * hours
