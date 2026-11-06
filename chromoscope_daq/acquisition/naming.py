"""Output file naming per README: YYMMDD_HHMMSS_identifier.ext.

The timestamp is taken once at the start of an acquisition and reused for
every file written by that acquisition.
"""

from __future__ import annotations

import datetime as _dt


def acquisition_timestamp(now: _dt.datetime | None = None) -> str:
    """Return the 'YYMMDD_HHMMSS' stamp used to build file names."""
    return (now or _dt.datetime.now()).strftime("%y%m%d_%H%M%S")


def build_basename(timestamp: str, identifier: str) -> str:
    """Return 'YYMMDD_HHMMSS_identifier' (no extension)."""
    stamp = timestamp or acquisition_timestamp()
    identifier = (identifier or "").strip()
    if identifier:
        return f"{stamp}_{identifier}"
    return stamp


def build_filename(timestamp: str, identifier: str, extension: str) -> str:
    """Build 'YYMMDD_HHMMSS_identifier.ext'.

    ``extension`` may be given with or without a leading dot. ``identifier``
    may be empty, in which case the file is 'YYMMDD_HHMMSS.ext'.
    """
    ext = extension.lstrip(".")
    return f"{build_basename(timestamp, identifier)}.{ext}"
