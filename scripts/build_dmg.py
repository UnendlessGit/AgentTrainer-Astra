"""Create, mount, relocate and verify a personal-installation Astra disk image."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile

from check_bundle import validate_bundle

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> bytes:
    result = subprocess.run(arguments, capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed ({result.returncode}): " + result.stderr.decode(errors="replace"))
    return result.stdout


@contextmanager
def destination_lock(destination: Path):
    descriptor = os.open(destination.with_name("." + destination.name + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid() or details.st_nlink != 1:
            raise ValueError("The image publication lock is not a private regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def detach_owned_image(image: Path, mount: Path) -> None:
    """Find partial attaches as well as the ordinary successful mount."""
    def images():
        info = plistlib.loads(run("hdiutil", "info", "-plist"))
        return [item for item in info.get("images", []) if item.get("image-path") and Path(item["image-path"]).resolve() == image.resolve()]
    for item in images():
        entities = item.get("system-entities", [])
        target = next((entry["mount-point"] for entry in entities if entry.get("mount-point") == str(mount)), None)
        if target is None:
            target = next((entry["dev-entry"] for entry in entities if entry.get("dev-entry")), None)
        if target is None:
            raise RuntimeError("The owned image is attached but has no detachable device")
        run("hdiutil", "detach", target)
    if images() or os.path.ismount(mount):
        raise RuntimeError("The owned disk image did not detach completely")


def publish_pair(image: Path, report: dict, destination: Path) -> None:
    report_path = destination.with_suffix(".verification.json")
    staged_report = image.with_suffix(".verification.json")
    with staged_report.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True); file.write("\n")
        file.flush(); os.fsync(file.fileno())
    with image.open("rb") as file:
        os.fsync(file.fileno())
    # Publish the receipt first: the DMG becomes visible only after its complete
    # verification record exists. The caller holds this output's exclusive lock.
    os.link(staged_report, report_path)
    try:
        os.link(image, destination)
    except BaseException:
        if report_path.exists() and os.path.samefile(report_path, staged_report): report_path.unlink()
        raise
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def create_dmg(bundle: Path, destination: Path) -> dict:
    bundle = bundle.resolve(strict=True)
    initial = validate_bundle(bundle)
    if not initial["adHocSigned"]:
        raise ValueError("This packaging path is configured for the requested ad-hoc personal build")
    destination = destination.resolve()
    if destination.suffix != ".dmg" or destination.is_relative_to(bundle):
        raise ValueError("Choose a new .dmg output path outside the application bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination_lock(destination):
        if destination.exists() or destination.with_suffix(".verification.json").exists():
            raise FileExistsError("The disk image or its verification record already exists")
        return build_locked(bundle, destination, initial)


def build_locked(bundle: Path, destination: Path, initial: dict) -> dict:
    temporary = Path(tempfile.mkdtemp(prefix="astra-dmg-", dir=destination.parent))
    preserve = False
    mount = temporary / "Mounted"
    try:
        stage = temporary / "Contents"
        stage.mkdir()
        app_name = "AgentTrainer Astra.app"
        run("ditto", str(bundle), str(stage / app_name))
        os.symlink("/Applications", stage / "Applications")
        (stage / "Read Me.txt").write_text(
            f"AgentTrainer Astra {initial['version']}\n\n"
            "Drag AgentTrainer Astra to Applications, then open it there.\n"
            "Recording and computer control use the macOS privacy permissions shown in the app.\n"
            "Training and inference run locally. Recordings and checkpoints stay outside the app bundle.\n\n"
            "This personal build is ad-hoc signed and is not notarized.\n")
        packed = temporary / "AgentTrainer-Astra.dmg"
        run("hdiutil", "create", "-volname", "AgentTrainer Astra", "-srcfolder", str(stage), "-format", "UDZO", "-fs", "HFS+", str(packed))
        run("hdiutil", "verify", str(packed))
        mount.mkdir()
        attempted = False
        try:
            attempted = True
            # A failing/timed-out attach can still have attached a device. Keep
            # scratch storage after an uncertain result, even if cleanup finds
            # no current mount; a service may still be completing the request.
            try:
                response_bytes = run("hdiutil", "attach", "-readonly", "-nobrowse", "-noautoopen", "-mountpoint", str(mount), "-plist", str(packed))
            except BaseException:
                preserve = True
                raise
            response = plistlib.loads(response_bytes)
            if not any(item.get("mount-point") == str(mount) for item in response.get("system-entities", [])):
                raise RuntimeError("The disk image did not mount at its requested verification location")
            if not (mount / "Applications").is_symlink() or os.readlink(mount / "Applications") != "/Applications":
                raise RuntimeError("The disk image's Applications link is invalid")
            mounted = validate_bundle(mount / app_name)
            relocated = temporary / "Relocated" / app_name
            run("ditto", str(mount / app_name), str(relocated))
            copied = validate_bundle(relocated)
            if (mounted["version"], mounted["build"]) != (initial["version"], initial["build"]):
                raise RuntimeError("The packaged application changed version during copying")
            if (copied["version"], copied["build"]) != (initial["version"], initial["build"]):
                raise RuntimeError("The relocated application changed version during copying")
        finally:
            if attempted:
                try: detach_owned_image(packed, mount)
                except BaseException:
                    preserve = True
                    raise
        with packed.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        report = {"dmg": str(destination), "sha256": digest, "bytes": packed.stat().st_size,
                  "application": initial, "readOnlyMountVerified": True, "relocatedCopyVerified": True,
                  "uiLaunched": False, "privacyPermissionsRequested": False, "workflowQualification": "separate release gate"}
        publish_pair(packed, report, destination)
        return report
    finally:
        if preserve or os.path.ismount(mount):
            print(f"Packaging scratch preserved after an uncertain mount/cleanup: {temporary}", file=sys.stderr)
        else:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, default=ROOT / "build/AgentTrainer Astra.app")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(create_dmg(arguments.app, arguments.output), indent=2))
