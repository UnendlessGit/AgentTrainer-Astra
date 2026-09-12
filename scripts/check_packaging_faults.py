"""Exercise packaging failure boundaries without mounting or launching an app."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
from unittest.mock import patch

import build_dmg
from check_bundle import validate_bundle, validate_load_graph


def expect_error(operation, expected):
    try:
        operation()
    except expected:
        return
    raise AssertionError("The invalid packaging operation was accepted")


def check(bundle: Path) -> dict:
    bundle = bundle.resolve(strict=True)
    validate_bundle(bundle)
    passed = []
    with tempfile.TemporaryDirectory(prefix="astra-packaging-faults-") as folder:
        root = Path(folder)
        source = root / "source.dmg"; source.write_bytes(b"owned verified-image fixture")
        destination = root / "output.dmg"
        original_link = os.link
        def fail_image(source, target):
            if Path(target) == destination: raise OSError("injected image publication failure")
            original_link(source, target)
        with patch.object(build_dmg.os, "link", side_effect=fail_image):
            expect_error(lambda: build_dmg.publish_pair(source, {"fixture": True}, destination), OSError)
        assert source.read_bytes() == b"owned verified-image fixture"
        assert not destination.exists() and not destination.with_suffix(".verification.json").exists()
        passed.append("failed image publication rolls back only its own complete report")

        image, mount = root / "attached.dmg", root / "Mounted"
        records = [{"image-path": str(image), "system-entities": [{"dev-entry": "/dev/fixture-owned"}]},
                   {"image-path": str(root / "unrelated.dmg"), "system-entities": [{"dev-entry": "/dev/fixture-unrelated"}]}]
        detached = []
        def image_tools(*args):
            if args[:2] == ("hdiutil", "info"): return plistlib.dumps({"images": records})
            assert args == ("hdiutil", "detach", "/dev/fixture-owned")
            detached.append(args[-1]); records.pop(0); return b""
        with patch.object(build_dmg, "run", side_effect=image_tools):
            build_dmg.detach_owned_image(image, mount)
        assert detached == ["/dev/fixture-owned"] and len(records) == 1
        passed.append("partial attach discovery detaches only the owned image")

        cleanups = []
        def failed_attach(*args):
            if args[0] == "ditto": Path(args[-1]).mkdir(parents=True); return b""
            if args[:2] == ("hdiutil", "create"): Path(args[-1]).write_bytes(b"image"); return b""
            if args[:2] == ("hdiutil", "verify"): return b""
            if args[:2] == ("hdiutil", "attach"): raise subprocess.TimeoutExpired(args, 1)
            raise AssertionError(args)
        with patch.object(build_dmg, "run", side_effect=failed_attach), patch.object(build_dmg, "detach_owned_image", side_effect=lambda *args: cleanups.append(args)):
            expect_error(lambda: build_dmg.build_locked(root, root / "timeout.dmg", {"version": "0.1.0"}), subprocess.TimeoutExpired)
        assert len(cleanups) == 1 and cleanups[0][1].is_dir()
        assert not (root / "timeout.dmg").exists()
        passed.append("uncertain attach still cleans up and preserves scratch")

        busy_cleanup = []
        def successful_attach(*args):
            if args[:2] == ("hdiutil", "attach"):
                point = Path(args[args.index("-mountpoint") + 1])
                os.symlink("/Applications", point / "Applications")
                return plistlib.dumps({"system-entities": [{"mount-point": str(point)}]})
            return failed_attach(*args)
        def still_busy(image, point):
            busy_cleanup.append((image, point)); raise OSError("injected detach failure")
        initial = {"version": "0.1.0", "build": "1"}
        with patch.object(build_dmg, "run", side_effect=successful_attach), patch.object(build_dmg, "validate_bundle", return_value=initial), patch.object(build_dmg, "detach_owned_image", side_effect=still_busy):
            expect_error(lambda: build_dmg.build_locked(root, root / "busy.dmg", initial), OSError)
        assert len(busy_cleanup) == 1 and busy_cleanup[0][0].is_file() and busy_cleanup[0][1].is_dir()
        assert not (root / "busy.dmg").exists() and not (root / "busy.verification.json").exists()
        passed.append("detach failure preserves backing image and publishes nothing")

        stuck = {"images": [{"image-path": str(image), "system-entities": [{"dev-entry": "/dev/fixture-owned"}]}]}
        with patch.object(build_dmg, "run", side_effect=lambda *args: plistlib.dumps(stuck) if args[:2] == ("hdiutil", "info") else b""):
            expect_error(lambda: build_dmg.detach_owned_image(image, mount), RuntimeError)
        with patch.object(build_dmg, "run", side_effect=RuntimeError("injected image inventory failure")):
            expect_error(lambda: build_dmg.detach_owned_image(image, mount), RuntimeError)
        passed.append("remaining device and failed inventory cannot count as detached")

        executable = root / "Probe.app/Contents/MacOS/Probe"
        executable.parent.mkdir(parents=True)
        original = bundle / "Contents/Helpers/AstraControl.app/Contents/MacOS/AstraControl"
        shutil.copy2(original, executable)
        validate_load_graph(executable.parents[2], [executable], [executable])
        for dependency, label in [("@rpath/astra-nonexistent.dylib", "missing relative dependency rejected"),
                                  ("@loader_path/../../../../outside.dylib", "escaping loader path rejected")]:
            shutil.copy2(original, executable)
            subprocess.run(["install_name_tool", "-change", "/usr/lib/libsqlite3.dylib", dependency, str(executable)], check=True, capture_output=True)
            expect_error(lambda: validate_load_graph(executable.parents[2], [executable], [executable]), ValueError)
            passed.append(label)

        copied = root / "Mode.app"
        subprocess.run(["ditto", str(bundle), str(copied)], check=True)
        entry = copied / "Contents/MacOS/AgentTrainerAstra"
        entry.chmod(0o644)
        expect_error(lambda: validate_bundle(copied), ValueError)
        passed.append("non-executable entry point rejected")
    return {"passed": True, "checks": passed, "imagesMounted": False, "inputPosted": False, "gpuUsed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("bundle", type=Path)
    print(json.dumps(check(parser.parse_args().bundle), indent=2))
