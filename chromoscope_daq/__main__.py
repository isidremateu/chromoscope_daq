"""Entry point for ``python -m chromoscope_daq``.

Launches the tkinter main window by default; ``--cli`` runs the
console-driven loop instead. Both use the CameraService connect flow
and dispose of the camera and SDK before exiting.
"""

from __future__ import annotations

import sys


def main() -> int:
    if "--cli" in sys.argv:
        from .cli import main as cli_main
        return cli_main()
    from .ui.main_window import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
