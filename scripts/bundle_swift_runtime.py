"""Embed needed Swift compatibility libraries and remove build-host search paths."""
from pathlib import Path
import re
import subprocess
from apple_toolchain import build_environment


def bundle_swift_runtime(bundle: Path, executable_name: str, *, environment: dict | None = None) -> None:
    environment = build_environment() if environment is None else environment
    executable = bundle / "Contents/MacOS" / executable_name
    frameworks = bundle / "Contents/Frameworks"
    frameworks.mkdir(exist_ok=True)
    subprocess.run(["xcrun", "swift-stdlib-tool", "--copy", "--platform", "macosx", "--scan-executable", str(executable),
                    "--destination", str(frameworks), "--sign", "-"], check=True, env=environment)
    commands = subprocess.check_output(["otool", "-arch", "arm64", "-l", str(executable)], text=True, env=environment)
    paths = re.findall(r"cmd LC_RPATH\s+cmdsize [0-9]+\s+path (.*?) \(offset [0-9]+\)", commands)
    for path in paths:
        if path.startswith("/") and not path.startswith(("/usr/lib/", "/System/Library/")):
            subprocess.run(["install_name_tool", "-delete_rpath", path, str(executable)], check=True, env=environment)
    if "@executable_path/../Frameworks" not in paths:
        subprocess.run(["install_name_tool", "-add_rpath", "@executable_path/../Frameworks", str(executable)], check=True, env=environment)
