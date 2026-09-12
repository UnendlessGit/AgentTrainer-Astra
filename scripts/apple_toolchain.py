"""Pair Apple's selected compiler with its own SDK without changing the Mac."""
import os
from pathlib import Path
import subprocess


def build_environment() -> dict[str, str]:
    environment = dict(os.environ)
    developer = environment.get("DEVELOPER_DIR") or subprocess.check_output(["xcode-select", "-p"], text=True).strip()
    environment["DEVELOPER_DIR"] = developer
    environment.pop("SDKROOT", None)
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], env=environment, text=True).strip()
    if not Path(developer).is_dir() or not Path(sdk).is_dir():
        raise RuntimeError("The selected Apple developer directory has no usable macOS SDK")
    environment["SDKROOT"] = sdk
    return environment
