"""Camera discovery, connection, and disconnect handling.

Wraps the Thorlabs TSI SDK (``thorlabs_tsi_sdk``) behind a small
service object with the lifecycle required by the SDK:

- exactly one :class:`TLCameraSDK` instance per process,
- the camera and the SDK must be :meth:`~TLCameraSDK.dispose` d before
  the process exits (the native libraries crash otherwise).

Disconnect events are delivered by the SDK on a native thread. This
module only records them (an event + serial number); reacting — dialog,
reconnect or shutdown — is the caller's job, from the main thread.
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import typing
import zipfile
from typing import Optional


def _register_dll_dir(lib_dir: str) -> None:
    """Make ``lib_dir`` discoverable for the SDK's bare-name DLL loads."""
    os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        # Windows-only API; guarded for portability.
        os.add_dll_directory(lib_dir)
    except (OSError, AttributeError):
        pass


def _extract_dlls_from_zip(zip_path: str, package_dir: str) -> Optional[str]:
    """Extract the native SDK DLLs from the vendored zip into the
    installed package's ``lib`` folder.

    Newer setuptools versions used by pip's build isolation sometimes
    omit the DLLs when building the SDK wheel. Returns the DLL directory
    (or None when the zip does not contain them).
    """
    arch = "windows-x86_64" if sys.maxsize > 2 ** 32 else "windows-x86"
    marker = f"/lib/{arch}/"
    target = os.path.join(package_dir, "lib")
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if marker in name and name.endswith(".dll"):
                os.makedirs(target, exist_ok=True)
                out_path = os.path.join(target, os.path.basename(name))
                with archive.open(name) as source, open(out_path, "wb") as dst:
                    dst.write(source.read())
                extracted += 1
    return target if extracted else None


def _add_sdk_dll_directory() -> None:
    """Make the Thorlabs SDK's native DLLs loadable on Windows.

    The SDK loads ``thorlabs_tsi_camera_sdk.dll`` by bare name, which
    searches the executable directory, the system directories and PATH —
    but not the package's own ``lib`` folder. Additionally, pip builds
    with newer setuptools sometimes omit the DLLs entirely. This helper
    registers the DLL folder (extracting it from the vendored SDK zip
    when missing) before the SDK initializes.
    """
    if platform.system() != "Windows":
        return
    try:
        import thorlabs_tsi_sdk

        package_dir = os.path.dirname(os.path.abspath(thorlabs_tsi_sdk.__file__))
    except ImportError:
        return
    lib_dir = os.path.join(package_dir, "lib")
    if not os.path.isfile(os.path.join(lib_dir,
                                       "thorlabs_tsi_camera_sdk.dll")):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        zip_path = os.path.join(
            repo_root, "Scientific Camera Interfaces", "SDK",
            "Python Toolkit", "thorlabs_tsi_camera_python_sdk_package.zip")
        if os.path.isfile(zip_path):
            try:
                lib_dir = (_extract_dlls_from_zip(zip_path, package_dir)
                           or lib_dir)
            except (OSError, zipfile.BadZipFile):
                pass
    if os.path.isfile(os.path.join(lib_dir,
                                   "thorlabs_tsi_camera_sdk.dll")):
        _register_dll_dir(lib_dir)


_add_sdk_dll_directory()

from thorlabs_tsi_sdk.tl_camera import TLCamera, TLCameraSDK

from ..ui import dialogs

if typing.TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["CameraService", "CameraServiceError", "UserAbortedError"]


class CameraServiceError(RuntimeError):
    """Raised when the SDK fails to initialize or open a camera."""


class UserAbortedError(RuntimeError):
    """Raised when the user chose to exit instead of (re)connecting."""


class CameraService:
    """Owns the TSI SDK and at most one open camera.

    Usage::

        service = CameraService()
        try:
            camera = service.connect()
        except UserAbortedError:
            return  # user chose to exit; SDK already disposed
        ...
        service.shutdown()  # disposes camera + SDK (required!)
    """

    def __init__(self) -> None:
        self._sdk: typing.Optional[TLCameraSDK] = None
        self._camera: typing.Optional[TLCamera] = None
        self._serial_number: typing.Optional[str] = None
        self._disconnect_event = threading.Event()
        # Reentrant: connect() may call shutdown() while holding the lock.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def camera(self) -> typing.Optional[TLCamera]:
        """The currently open camera, or ``None``."""
        return self._camera

    @property
    def serial_number(self) -> typing.Optional[str]:
        """Serial number of the currently open camera, or ``None``."""
        return self._serial_number

    @property
    def connection_lost(self) -> bool:
        """``True`` once the camera reported disconnection."""
        return self._disconnect_event.is_set()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> TLCamera:
        """Initialize the SDK, discover cameras, and open one.

        Handles the interaction flow from the README spec:

        - more than one camera found → user selects one;
        - no camera found → pop-up offering connect-and-retry or exit;
        - repeats until a camera is opened or the user aborts.

        Returns the opened :class:`TLCamera`.
        Raises :class:`UserAbortedError` if the user chose to exit.
        """
        with self._lock:
            if self._camera is not None:
                return self._camera

            if self._sdk is None:
                self._sdk = self._create_sdk()
                self._sdk.set_camera_disconnect_callback(
                    self._on_sdk_disconnect
                )

            while True:
                serial = self._discover_and_select()
                if serial is None:
                    # User aborted the selection dialog.
                    self.shutdown()
                    raise UserAbortedError("Camera selection cancelled.")

                try:
                    self._camera = self._sdk.open_camera(serial)
                except Exception as error:  # noqa: BLE001 - SDK raises bare RuntimeError subclasses
                    if not dialogs.ask_connect_camera():
                        self.shutdown()
                        raise UserAbortedError(
                            "Camera could not be opened."
                        ) from error
                    continue

                self._serial_number = serial
                self._disconnect_event.clear()
                return self._camera

    def handle_connection_lost(self) -> typing.Optional[TLCamera]:
        """React to a mid-run disconnect; returns a camera if reconnected.

        Must be called from the main thread after
        :attr:`connection_lost` becomes ``True``. Disposes the stale
        camera object and asks the user to reconnect or exit. On
        retry, runs the full discovery/selection flow again.
        """
        if not self.connection_lost and self._camera is not None:
            return self._camera

        serial = self._serial_number or "unknown"
        self._dispose_camera()

        if not dialogs.ask_connection_lost(serial):
            self.shutdown()
            raise UserAbortedError("User chose to exit after disconnect.")

        return self.connect()

    def shutdown(self) -> None:
        """Dispose the camera and the SDK.

        Must be called before process exit (SDK requirement). Safe to
        call multiple times.
        """
        with self._lock:
            self._dispose_camera()
            if self._sdk is not None:
                try:
                    self._sdk.dispose()
                except Exception:  # noqa: BLE001 - best effort on shutdown
                    pass
                self._sdk = None
            self._serial_number = None
            self._disconnect_event.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _create_sdk() -> TLCameraSDK:
        try:
            return TLCameraSDK()
        except Exception as error:  # noqa: BLE001
            raise CameraServiceError(
                f"Could not initialize the Thorlabs TSI SDK: {error!r}. "
                "Make sure the Thorlabs camera drivers (ThorCam installer) "
                "are installed, the Microsoft Visual C++ Redistributable "
                "(x64) is present, and no other application is using the "
                "SDK."
            ) from error

    def _discover_and_select(self) -> typing.Optional[str]:
        """Discover cameras and pick one via dialogs; ``None`` = abort."""
        assert self._sdk is not None
        serial_numbers = list(self._sdk.discover_available_cameras())
        if not serial_numbers:
            if not dialogs.ask_connect_camera():
                return None
            return self._discover_and_select()
        if len(serial_numbers) == 1:
            # Exactly one camera: connect without asking.
            return serial_numbers[0]
        return dialogs.select_camera(serial_numbers)

    def _on_sdk_disconnect(self, serial_number: str, _context) -> None:
        """SDK-thread callback: only record the event, no UI here."""
        # NOTE: runs on a native SDK thread — must not touch tkinter.
        self._disconnect_event.set()

    def _dispose_camera(self) -> None:
        if self._camera is not None:
            try:
                self._camera.dispose()
            except Exception:  # noqa: BLE001 - camera may already be gone
                pass
            self._camera = None
