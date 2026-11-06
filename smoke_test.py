"""Hardware smoke test for Chromoscope DAQ — run this on Windows.

Exercises the full acquisition stack against the real CS165CU1/M:

1. Snapshot  -> one .tif and one .jpg
2. Streaming -> 3 s free-run at 30 fps with mp4 saving + auto-exposure
3. Time Series -> 5 software-triggered frames (dt 0.5 s), classic + OME tif

All files are written to a time-stamped folder next to this script so you
can inspect them afterwards.

Usage (from the repo root, with the camera attached)::

    .venv\\Scripts\\python smoke_test.py
"""

from __future__ import annotations

import os
import sys
import threading

from chromoscope_daq.camera import (CameraService, CameraServiceError,
                                    UserAbortedError)
from chromoscope_daq.acquisition import (AutoExposureController,
                                         SnapshotSession, StreamingSession,
                                         TimeSeriesSession,
                                         acquisition_timestamp,
                                         camera_fps_max, minimum_interval_s)


def _fmt_range(value) -> str:
    try:
        lo, hi = value
        return f"{lo} .. {hi}"
    except Exception:  # noqa: BLE001 - just reporting
        return repr(value)


def print_camera_info(camera) -> None:
    print(f"Camera            : {camera.name}")
    print(f"Sensor            : {camera.sensor_width_pixels} x "
          f"{camera.sensor_height_pixels} px, {camera.bit_depth} bit")
    print(f"Exposure range us : "
          f"{_fmt_range(camera.exposure_time_range_us)}")
    gain_lo, gain_hi = camera.gain_range
    print(f"Gain range index  : {_fmt_range(camera.gain_range)} "
          f"(= {camera.convert_gain_to_decibels(gain_lo):.1f} .. "
          f"{camera.convert_gain_to_decibels(gain_hi):.1f} dB)")
    print(f"Frame rate range  : "
          f"{_fmt_range(camera.frame_rate_control_value_range)}")
    fps_max = camera_fps_max(camera)
    print(f"camera_fps_max    : {fps_max}")


def smoke_snapshot(camera, ts: str, outdir: str) -> None:
    print("\n--- 1/3 Snapshot ------------------------------------------------")
    for fmt in ("tif", "jpg"):
        session = SnapshotSession(
            camera, ts, f"smoke_snap_{fmt}", save_dir=outdir,
            save_enabled=True, fmt=fmt)
        bundle = session.run()
        path = os.path.join(outdir,
                            f"{ts}_smoke_snap_{fmt}.{fmt}")
        size = os.path.getsize(path) if os.path.isfile(path) else -1
        print(f"[ok] {fmt.upper():3s} saved: {path} ({size} bytes) "
              f"mean luminance: {bundle.mean_luminance:.1f}")


def smoke_streaming(camera, ts: str, outdir: str) -> None:
    print("\n--- 2/3 Streaming (3 s @ 30 fps, mp4 + auto-exposure) -----------")
    exp_lo, exp_hi = camera.exposure_time_range_us
    gain_lo, gain_hi = camera.gain_range
    # The controller works in dB; the camera's gain_range is index units.
    auto = AutoExposureController(
        (exp_lo, exp_hi),
        (camera.convert_gain_to_decibels(gain_lo),
         camera.convert_gain_to_decibels(gain_hi)))
    stop = threading.Event()
    session = StreamingSession(
        camera, fps=30.0, timestamp=ts, identifier="smoke_stream",
        save_dir=outdir, save_enabled=True, duration_s=3.0,
        auto_exposure=auto)
    stats = session.run(stop)
    path = os.path.join(outdir, f"{ts}_smoke_stream.mp4")
    size = os.path.getsize(path) if os.path.isfile(path) else -1
    print(f"[ok] frames: {stats.frames}, dropped: {stats.dropped_mp4}, "
          f"exposure: {stats.exposure_us:.1f} us, "
          f"gain: {stats.gain_db:.1f} dB")
    print(f"[ok] mp4 saved: {path} ({size} bytes)")


def smoke_time_series(camera, ts: str, outdir: str) -> None:
    print("\n--- 3/3 Time Series (5 frames, dt >= 0.5 s) ---------------------")
    # Coerce dt up to the minimum allowed interval for the current exposure.
    dt_min = minimum_interval_s(camera.exposure_time_us, camera_fps_max(camera))
    interval = max(0.5, dt_min)
    print(f"     exposure {camera.exposure_time_us:.1f} us -> "
          f"dt_min {dt_min * 1000.0:.1f} ms, using dt = {interval:.3f} s")
    stop = threading.Event()
    session = TimeSeriesSession(
        camera, interval_s=interval, timestamp=ts,
        identifier="smoke_ts", save_dir=outdir, save_enabled=True,
        max_frames=5)
    stats = session.run(stop)
    classic = os.path.join(outdir, f"{ts}_smoke_ts.tif")
    ome = os.path.join(outdir, f"{ts}_smoke_ts_OME.tif")
    for path in (classic, ome):
        size = os.path.getsize(path) if os.path.isfile(path) else -1
        print(f"[ok] saved: {path} ({size} bytes)")
    print(f"[ok] frames: {stats.frames}")


def main() -> int:
    print("Chromoscope DAQ — hardware smoke test")
    outdir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"smoke_out_{acquisition_timestamp()}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output folder    : {outdir}")

    service = CameraService()
    try:
        camera = service.connect()
    except UserAbortedError:
        print("No camera connected — aborting smoke test.")
        return 1
    except CameraServiceError as error:
        print(f"SDK error: {error}", file=sys.stderr)
        return 1

    ts = acquisition_timestamp()
    try:
        print_camera_info(camera)
        # The camera persists its exposure across sessions (it was found
        # stuck at 10.3 s after a previous auto-exposure run); set a sane
        # starting point so the smoke test runs fast.
        camera.exposure_time_us = 1000
        print(f"Exposure set to   : {camera.exposure_time_us} us")
        smoke_snapshot(camera, ts, outdir)
        smoke_streaming(camera, ts, outdir)
        smoke_time_series(camera, ts, outdir)
        print("\nAll smoke tests completed successfully.")
        print(f"Inspect the files in: {outdir}")
        return 0
    finally:
        service.shutdown()
        print("Camera and SDK disposed.")


if __name__ == "__main__":
    sys.exit(main())
