#!/usr/bin/env python3
"""Build a native development bundle without touching an installed application."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", choices=["debug", "release"], default="debug")
    args = parser.parse_args()
    compute_dist = ROOT / "build" / "compute"
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(compute_dist),
        "--workpath", str(ROOT / ".local" / "compute-build"), str(ROOT / "packaging/astra-compute.spec"))
    compute_app = compute_dist / "AstraCompute.app"
    run(sys.executable, str(ROOT / "scripts/check_worker.py"),
        str(compute_app / "Contents/MacOS/AstraCompute"), "--offline")
    run("swift", "build", "-c", args.configuration, "--product", "AgentTrainerAstra")
    binary_dir = Path(subprocess.check_output(
        ["swift", "build", "-c", args.configuration, "--show-bin-path"], cwd=ROOT, text=True).strip())
    destination = ROOT / "build" / "AgentTrainer Astra.app"
    staging = ROOT / "build" / "AgentTrainer Astra.staging.app"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "Contents" / "MacOS").mkdir(parents=True)
    resources = staging / "Contents" / "Resources"
    resources.mkdir()
    shutil.copy2(binary_dir / "AgentTrainerAstra", staging / "Contents" / "MacOS" / "AgentTrainerAstra")
    shutil.copy2(ROOT / "Resources" / "Info.plist", staging / "Contents" / "Info.plist")
    helpers = staging / "Contents" / "Helpers"
    helpers.mkdir()
    shutil.copytree(compute_app, helpers / "AstraCompute.app", symlinks=True)
    # A deliberately blank placeholder; the final icon will be made in Icon Composer.
    from PIL import Image
    Image.new("RGBA", (1024, 1024), (225, 226, 229, 255)).save(resources / "AppIcon.icns")
    run("codesign", "--force", "--sign", "-", str(staging))
    run("codesign", "--verify", "--deep", "--strict", str(staging))
    run(sys.executable, str(ROOT / "scripts/check_worker.py"),
        str(helpers / "AstraCompute.app/Contents/MacOS/AstraCompute"), "--offline")
    if destination.exists():
        process = subprocess.run(["pgrep", "-f", str(destination / "Contents" / "MacOS" / "AgentTrainerAstra")], capture_output=True)
        if process.returncode == 0:
            raise RuntimeError("Quit the development app before replacing its bundle.")
        backup = destination.with_name("AgentTrainer Astra.previous.app")
        if backup.exists(): shutil.rmtree(backup)
        destination.rename(backup)
        try: staging.rename(destination)
        except BaseException:
            backup.rename(destination)
            raise
    else:
        staging.rename(destination)
    print(destination)

if __name__ == "__main__":
    main()
