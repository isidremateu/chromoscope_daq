"""Diagnose Thorlabs TSI SDK DLL loading problems.

Run with the project's virtual environment::

    .venv\\Scripts\\python diagnose_sdk.py
"""

import ctypes
import os
import sys


def main() -> int:
    print(f"Python        : {sys.version}")
    print(f"64-bit Python : {sys.maxsize > 2 ** 32}")
    try:
        import thorlabs_tsi_sdk

        package_dir = os.path.dirname(os.path.abspath(thorlabs_tsi_sdk.__file__))
    except ImportError as error:
        print(f"SDK package   : NOT INSTALLED ({error})")
        return 1
    print(f"Package       : {package_dir}")

    lib = os.path.join(package_dir, "lib")
    print(f"lib folder    : {lib} (exists: {os.path.isdir(lib)})")
    if os.path.isdir(lib):
        print(f"lib contents  : {sorted(os.listdir(lib))[:12]}")
    dll = os.path.join(lib, "thorlabs_tsi_camera_sdk.dll")
    print(f"SDK DLL       : {os.path.isfile(dll)} ({dll})")
    if not os.path.isfile(dll):
        print("-> The SDK pip package did not install the native DLLs.")
        print("   Re-install from the zip (see README, step 3).")
        return 1

    print("\n1) Full-path load test...")
    try:
        ctypes.CDLL(dll)
        print("   OK")
    except OSError as error:
        print(f"   FAILED: {error}")
        print("   -> The DLL was found but a DEPENDENCY is missing.")
        print("      Install the Microsoft Visual C++ Redistributable (x64):")
        print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return 1

    print("\n2) Bare-name load test (after PATH registration)...")
    os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    try:
        ctypes.CDLL("thorlabs_tsi_camera_sdk.dll")
        print("   OK")
    except OSError as error:
        print(f"   FAILED: {error}")
        return 1

    print("\n3) TLCameraSDK() test...")
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

    sdk = TLCameraSDK()
    print("   OK — SDK initialized.")
    sdk.dispose()
    print("\nAll checks passed; the application should work on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
