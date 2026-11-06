"""Modal tkinter dialogs for camera connection handling.

These dialogs are intentionally simple and modal: they are used only
during startup and reconnection, not during acquisition (the main
acquisition UI will have its own window).

All functions must be called from the main (tkinter) thread. The
camera-disconnect callback from the SDK runs on a native SDK thread and
must *not* open dialogs directly — it only signals an event, and the
main thread reacts (see :mod:`chromoscope_daq.camera.connection`).
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Sequence

_RETRY_THROTTLE_MS = 500


def _ensure_root() -> tk.Tk:
    """Return a hidden root window, creating it if necessary."""
    root = tk.Tk()
    root.withdraw()
    return root


def select_camera(serial_numbers: Sequence[str]) -> Optional[str]:
    """Ask the user to pick one camera from ``serial_numbers``.

    Returns the chosen serial number, or ``None`` if the user cancelled.
    """
    if len(serial_numbers) == 1:
        return serial_numbers[0]

    root = _ensure_root()
    choice: dict[str, Optional[str]] = {"serial": None}

    dialog = tk.Toplevel(root, title="Select camera")
    dialog.transient(root)
    dialog.grab_set()
    dialog.resizable(False, False)

    ttk.Label(
        dialog,
        text="Multiple cameras detected.\nSelect the camera to use:",
    ).pack(padx=16, pady=(16, 8))

    combo = ttk.Combobox(dialog, values=list(serial_numbers), state="readonly")
    combo.current(0)
    combo.pack(padx=16, pady=(0, 12))

    def confirm() -> None:
        choice["serial"] = combo.get()
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.pack(padx=16, pady=(0, 16))
    ttk.Button(buttons, text="OK", command=confirm).pack(side="left", padx=4)
    ttk.Button(
        buttons, text="Cancel", command=dialog.destroy
    ).pack(side="left", padx=4)

    root.wait_window(dialog)
    root.destroy()
    return choice["serial"]


def ask_connect_camera() -> bool:
    """Pop up a dialog when no camera is detected.

    Returns ``True`` if the user wants to retry (after connecting a
    camera), ``False`` to exit the application.
    """
    root = _ensure_root()
    retry = messagebox.askretrycancel(
        parent=root,
        title="No camera detected",
        message=(
            "No Chromoscope camera was detected.\n\n"
            "Connect the camera (USB) and make sure it is not in use by "
            "another application (e.g. ThorCam), then press Retry.\n"
            "Press Cancel to exit."
        ),
    )
    root.destroy()
    if retry:
        # Give the driver a moment before rescanning.
        time.sleep(_RETRY_THROTTLE_MS / 1000.0)
    return bool(retry)


def ask_connection_lost(serial_number: str) -> bool:
    """Pop up a dialog when the camera connection is lost mid-run.

    Returns ``True`` if the user wants to attempt reconnection,
    ``False`` to exit the application.
    """
    root = _ensure_root()
    retry = messagebox.askretrycancel(
        parent=root,
        title="Camera disconnected",
        message=(
            f"The connection to camera {serial_number} was lost.\n\n"
            "Reconnect the camera (USB), then press Retry to continue.\n"
            "Press Cancel to exit."
        ),
    )
    root.destroy()
    return bool(retry)
