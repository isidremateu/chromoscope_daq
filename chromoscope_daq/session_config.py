"""Settings persistence: session-to-session defaults + configuration files.

The full application state (camera configuration, perspective
correction, intensity calibration, profile cuts, display and
acquisition parameters) is stored as json. The default file lives under
the user's home directory and is written automatically when settings
are applied and when the application closes; configuration files can be
saved and loaded from the File menu.
"""

from __future__ import annotations

import json
import os

_FILE_TYPE = "chromoscope_settings"
_VERSION = 1

__all__ = ["SettingsError", "default_path", "save_settings", "load_settings"]


class SettingsError(ValueError):
    """Invalid or unreadable settings file."""


def default_path() -> str:
    """Path of the automatic (session-to-session) settings file."""
    return os.path.join(os.path.expanduser("~"), ".chromoscope_daq",
                        "settings.json")


def save_settings(path: str, state: dict) -> None:
    """Write the state atomically (write temp file, then replace)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {"type": _FILE_TYPE, "version": _VERSION, "state": state}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def load_settings(path: str) -> dict:
    """Read a settings file; returns the inner state dict."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SettingsError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("type") != _FILE_TYPE:
        raise SettingsError(f"not a Chromoscope DAQ settings file: {path}")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise SettingsError(f"missing state section in {path}")
    return state
