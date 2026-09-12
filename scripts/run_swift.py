"""Run SwiftPM with a compiler/SDK pair from the same selected developer root."""
from pathlib import Path
import subprocess
import sys
from apple_toolchain import build_environment

if __name__ == "__main__":
    result = subprocess.run(["xcrun", "swift", *sys.argv[1:]], cwd=Path(__file__).resolve().parents[1], env=build_environment())
    raise SystemExit(result.returncode)
