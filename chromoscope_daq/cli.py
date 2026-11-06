"""Console-driven acquisition loop (``python -m chromoscope_daq --cli``).

Interactive fallback to the tkinter main window: connects via the
CameraService dialog flow and runs Snapshot / Streaming / Time Series
from the terminal, with the file-naming and saving rules from the README.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Callable, Optional

from . import __version__
from .acquisition import (AcquisitionError, AutoExposureController,
                          SnapshotSession, StreamingSession,
                          TimeSeriesSession, acquisition_timestamp,
                          build_filename, camera_fps_max, minimum_interval_s)
from .camera import CameraService, UserAbortedError

#: Default directory for saved files (created on demand).
DEFAULT_SAVE_DIR = "data"


# ----------------------------------------------------------------------
# Small input helpers
# ----------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = _ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            return int(float(raw))
        except ValueError:
            print("Please enter a whole number.")


def _ask_dir() -> str:
    path = _ask("Save directory", DEFAULT_SAVE_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _ask_identifier(default: str) -> str:
    """Ask for a file identifier; path separators are not allowed."""
    identifier = _ask("Identifier", default)
    return identifier.replace("/", "_").replace("\\", "_").strip()


def _gain_range_db(camera) -> tuple:
    """The camera's gain range converted to dB (SDK uses index units)."""
    gain_lo, gain_hi = camera.gain_range
    return (camera.convert_gain_to_decibels(gain_lo),
            camera.convert_gain_to_decibels(gain_hi))


def _print_camera_info(camera) -> None:
    fps_max = camera_fps_max(camera)
    exp_lo, exp_hi = camera.exposure_time_range_us
    print(f"Connected to camera '{camera.name}' "
          f"(sensor {camera.sensor_width_pixels} x "
          f"{camera.sensor_height_pixels} px @ {camera.bit_depth} bit)")
    print(f"  Exposure us : {exp_lo} .. {exp_hi} (current: "
          f"{camera.exposure_time_us})")
    print(f"  Gain dB     : "
          f"{_gain_range_db(camera)[0]:.1f} .. "
          f"{_gain_range_db(camera)[1]:.1f} (current: "
          f"{camera.convert_gain_to_decibels(camera.gain):.1f})")
    print(f"  Max fps     : {fps_max if fps_max is not None else 'unknown'}")


# ----------------------------------------------------------------------
# Acquisition runners
# ----------------------------------------------------------------------

def _run_acquisition(target: Callable[[], object],
                     stop_message: str,
                     manual_stop: bool) -> object:
    """Run ``target`` in a worker thread; optional Enter-to-stop."""
    stop_event = threading.Event()
    box: dict = {}

    def work() -> None:
        try:
            box["result"] = target()
        except Exception as error:  # noqa: BLE001 - re-raised below
            box["error"] = error

    thread = threading.Thread(target=work, name="acquisition")
    thread.start()
    if manual_stop:
        input(stop_message)
        stop_event.set()
        thread.join()
    else:
        try:
            thread.join()
        except KeyboardInterrupt:
            print("\nStopping...")
            stop_event.set()
            thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _run_snapshot(camera) -> None:
    identifier = _ask_identifier("snap")
    fmt = _ask("Format jpg/tif", "tif").lower()
    save_dir = _ask_dir()
    ts = acquisition_timestamp()

    def target():
        session = SnapshotSession(camera, ts, identifier, save_dir=save_dir,
                                  save_enabled=True, fmt=fmt)
        return session.run()

    bundle = _run_acquisition(target, "", manual_stop=False)
    ext = "jpg" if fmt.startswith("jpg") else "tif"
    path = os.path.join(save_dir, build_filename(ts, identifier, ext))
    print(f"[ok] saved {path} "
          f"(mean luminance {bundle.mean_luminance:.1f} / 1023)")


def _run_streaming(camera) -> None:
    fps_max = camera_fps_max(camera)
    default_fps = min(30.0, fps_max) if fps_max else 30.0
    fps = _ask_float(f"Frame rate fps (max {fps_max})", default_fps)
    if fps_max is not None and fps > fps_max:
        print(f"  Coerced fps {fps} -> {fps_max} (camera maximum).")
        fps = fps_max
    duration = _ask_float("Duration s (0 = run until Enter)", 0.0)
    identifier = _ask_identifier("stream")
    save_dir = _ask_dir()

    auto = AutoExposureController(camera.exposure_time_range_us,
                                  _gain_range_db(camera))
    ts = acquisition_timestamp()
    last = {}
    session = StreamingSession(camera, fps=fps, timestamp=ts,
                               identifier=identifier, save_dir=save_dir,
                               save_enabled=True,
                               duration_s=duration or None,
                               auto_exposure=auto,
                               on_frame=lambda b, i: last.update(
                                   lum=b.mean_luminance))

    def target():
        return session.run(threading.Event())

    if duration:
        print(f"Streaming for {duration:g} s ...")
        stats = _run_acquisition(target, "", manual_stop=False)
    else:
        stats = _run_acquisition(target, "Streaming — press Enter to stop...",
                                 manual_stop=True)
    path = os.path.join(save_dir, build_filename(ts, identifier, "mp4"))
    print(f"[ok] frames {stats.frames}, dropped {stats.dropped_mp4}, "
          f"exposure {stats.exposure_us:.1f} us, "
          f"gain {stats.gain_db:.1f} dB")
    if "lum" in last:
        print(f"     last frame mean luminance: "
              f"{last['lum']:.1f} / 1023")
    print(f"[ok] saved {path}")


def _run_time_series(camera) -> None:
    print(f"Current exposure {camera.exposure_time_us} us "
          f"(= {camera.exposure_time_us / 1e6:.3g} s), gain "
          f"{camera.convert_gain_to_decibels(camera.gain):.1f} dB "
          "(Time Series uses manual exposure).")
    exposure_raw = input(
        "Exposure us (1000000 = 1 s) [keep current]: ").strip()
    if exposure_raw:
        try:
            camera.exposure_time_us = int(round(float(exposure_raw)))
        except ValueError:
            print("  Not a number — keeping current exposure.")

    dt_min = minimum_interval_s(camera.exposure_time_us,
                                camera_fps_max(camera))
    interval = _ask_float(f"Interval dt s (min {dt_min:.3f})",
                          max(1.0, dt_min))
    if interval < dt_min:
        print(f"  Coerced dt {interval:g} -> {dt_min:.3f} s (minimum).")
        interval = dt_min
    max_frames = _ask_int("Number of images (0 = run until Enter)", 10)
    identifier = _ask_identifier("series")
    save_dir = _ask_dir()

    ts = acquisition_timestamp()
    last = {}
    session = TimeSeriesSession(camera, interval_s=interval, timestamp=ts,
                                identifier=identifier, save_dir=save_dir,
                                save_enabled=True,
                                max_frames=max_frames or None,
                                on_frame=lambda b, i: last.update(
                                    lum=b.mean_luminance))

    def target():
        return session.run(threading.Event())

    if max_frames:
        print(f"Acquiring {max_frames} frames at dt = {interval:g} s ...")
        stats = _run_acquisition(target, "", manual_stop=False)
    else:
        stats = _run_acquisition(target, "Acquiring — press Enter to stop...",
                                 manual_stop=True)
    classic = os.path.join(save_dir, build_filename(ts, identifier, "tif"))
    ome = os.path.join(save_dir, build_filename(ts, identifier + "_OME",
                                                "tif"))
    print(f"[ok] frames {stats.frames}")
    if "lum" in last:
        print(f"     last frame mean luminance: {last['lum']:.1f} / 1023")
    print(f"[ok] saved {classic}")
    print(f"[ok] saved {ome}")


_RUNNERS = {
    "1": ("Snapshot", _run_snapshot),
    "2": ("Streaming", _run_streaming),
    "3": ("Time Series", _run_time_series),
}


def _menu() -> str:
    print("\nMain menu")
    for key, (label, _) in sorted(_RUNNERS.items()):
        print(f"  [{key}] {label}")
    print("  [q] Quit")
    return input("Choice: ").strip().lower()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

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
        _print_camera_info(camera)
        while True:
            if service.connection_lost:
                print("Camera connection lost.")
                try:
                    camera = service.handle_connection_lost()
                except UserAbortedError:
                    break
                _print_camera_info(camera)

            choice = _menu()
            if choice in ("q", "quit", "exit"):
                break
            runner = _RUNNERS.get(choice)
            if runner is None:
                continue
            try:
                runner[1](camera)
            except UserAbortedError:
                break
            except AcquisitionError as error:
                print(f"Acquisition failed: {error}")
            except KeyboardInterrupt:
                print("\nInterrupted.")
            except Exception as error:  # noqa: BLE001 - keep the loop alive
                print(f"Error: {error}")
    finally:
        # SDK requirement: dispose camera and SDK before exiting.
        service.shutdown()
        print("Camera and SDK disposed. Goodbye.")
    return 0
