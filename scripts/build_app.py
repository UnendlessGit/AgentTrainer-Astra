#!/usr/bin/env python3
"""Build a native development bundle without touching an installed application."""
from pathlib import Path
import argparse
import hashlib
import json
import plistlib
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
    weight_spec = json.loads((ROOT / "assets/weights.json").read_text())["convnextTiny"]
    weights = ROOT / "vendor/weights" / weight_spec["artifact"]
    if not weights.exists():
        run(sys.executable, str(ROOT / "scripts/prepare_weights.py"))
    with weights.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != weight_spec["artifactSHA256"]:
        raise RuntimeError("The pretrained backbone fails its pinned artifact digest; rebuild and qualify the conversion.")
    compute_dist = ROOT / "build" / "compute"
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(compute_dist),
        "--workpath", str(ROOT / ".local" / "compute-build"), str(ROOT / "packaging/astra-compute.spec"))
    compute_app = compute_dist / "AstraCompute.app"
    run(sys.executable, str(ROOT / "scripts/check_worker.py"),
        str(compute_app / "Contents/MacOS/AstraCompute"), "--offline")
    run("swift", "build", "-c", args.configuration, "--product", "AgentTrainerAstra")
    run("swift", "build", "-c", args.configuration, "--product", "AstraControl")
    binary_dir = Path(subprocess.check_output(
        ["swift", "build", "-c", args.configuration, "--show-bin-path"], cwd=ROOT, text=True).strip())
    destination = ROOT / "build" / "AgentTrainer Astra.app"
    staging = ROOT / "build" / "AgentTrainer Astra.staging.app"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "Contents" / "MacOS").mkdir(parents=True)
    resources = staging / "Contents" / "Resources"
    resources.mkdir()
    (resources / "Weights").mkdir()
    shutil.copy2(weights, resources / "Weights" / weight_spec["artifact"])
    shutil.copy2(ROOT / "assets/weights.json", resources / "Weights" / "manifest.json")
    shutil.copy2(ROOT / "assets/LICENSE.ConvNeXt", resources / "Weights" / "LICENSE.ConvNeXt")
    shutil.copy2(binary_dir / "AgentTrainerAstra", staging / "Contents" / "MacOS" / "AgentTrainerAstra")
    shutil.copy2(ROOT / "Resources" / "Info.plist", staging / "Contents" / "Info.plist")
    helpers = staging / "Contents" / "Helpers"
    helpers.mkdir()
    shutil.copytree(compute_app, helpers / "AstraCompute.app", symlinks=True)
    control = helpers / "AstraControl.app"
    (control / "Contents/MacOS").mkdir(parents=True)
    shutil.copy2(binary_dir / "AstraControl", control / "Contents/MacOS/AstraControl")
    parent_info = plistlib.loads((ROOT / "Resources/Info.plist").read_bytes())
    control_info = {
        "CFBundleIdentifier": "com.unendless.agenttrainer.astra.control",
        "CFBundleName": "AgentTrainer Astra Control",
        "CFBundleDisplayName": "AgentTrainer Astra Control",
        "CFBundleExecutable": "AstraControl", "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": parent_info["CFBundleShortVersionString"],
        "CFBundleVersion": parent_info["CFBundleVersion"],
        "LSMinimumSystemVersion": "15.0", "LSBackgroundOnly": True,
        "NSInputMonitoringUsageDescription": parent_info["NSInputMonitoringUsageDescription"],
        "NSAccessibilityUsageDescription": parent_info["NSAccessibilityUsageDescription"],
    }
    (control / "Contents/Info.plist").write_bytes(plistlib.dumps(control_info))
    run("codesign", "--force", "--sign", "-", str(control))
    # A deliberately blank placeholder; the final icon will be made in Icon Composer.
    from PIL import Image
    Image.new("RGBA", (1024, 1024), (225, 226, 229, 255)).save(resources / "AppIcon.icns")
    run("codesign", "--force", "--sign", "-", str(staging))
    run("codesign", "--verify", "--deep", "--strict", str(staging))
    run(sys.executable, str(ROOT / "scripts/check_worker.py"),
        str(helpers / "AstraCompute.app/Contents/MacOS/AstraCompute"), "--offline")
    run(sys.executable, str(ROOT / "scripts/check_inference.py"),
        str(helpers / "AstraCompute.app/Contents/MacOS/AstraCompute"), "--offline")
    run(sys.executable, str(ROOT / "scripts/check_control.py"),
        str(control / "Contents/MacOS/AstraControl"))
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
