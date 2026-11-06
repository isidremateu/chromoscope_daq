"""Entry point for ``python -m chromoscope_daq``.

Minimal bootstrap for now: connect to the camera (with the full
select / no-camera / connection-lost dialog flow), print its
properties, and dispose of everything cleanly.

Acquisition (Streaming / Time Series), processing, and saving are
added in subsequent modules.
"""

from __future__ import annotations

import sys

from . import __version__
from .camera import CameraService, UserAbortedError


def main() -> int:
    print(f"Chromoscope DAQ v{__version__}")
    service = CameraService()
    try:
        camera = service.connect()
    except UserAbortedError:
        print("No camera connected — exiting.")
        return 1
    except Exception as error:  # noqa: BLE001
        print(f"Fatal: {error}", file=sys.stderr)
        return 1

    try:
        print(f"Connected to camera '{camera.name}' "
              f"(serial {service.serial_number})")
        print(f"  Sensor: {camera.sensor_width_pixels} x "
              f"{camera.sensor_height_pixels} px @ {camera.bit_depth} bit")

        # --- placeholder: main application loop goes here -------------
        # Streaming mode  : free-run acquisition, auto-exposure loop.
        # Time Series mode: software-triggered acquisition with Δt.
        print("Camera connected successfully. "
              "Acquisition modules not implemented yet.")
        # ---------------------------------------------------------------
    finally:
        # SDK requirement: dispose camera and SDK before exiting.
        service.shutdown()
        print("Camera and SDK disposed. Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
