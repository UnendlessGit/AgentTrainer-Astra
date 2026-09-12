"""Collect the installed, locked runtime distributions' redistribution notices."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys


PACKAGES = ("mlx", "mlx-metal", "numpy", "safetensors", "pillow", "pyinstaller")


def bundle_notices(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in PACKAGES:
        distribution = importlib.metadata.distribution(name)
        selected = []
        for item in distribution.files or ():
            path = Path(str(item))
            if path.is_absolute() or ".." in path.parts:
                continue
            in_license_directory = "licenses" in path.parts and any(part.endswith(".dist-info") for part in path.parts)
            named_notice = re.fullmatch(r"(?:LICEN[CS]E|COPYING|NOTICE|COPYRIGHT)(?:\.[A-Z0-9_-]+)?", path.name.upper()) is not None
            if not (in_license_directory or named_notice):
                continue
            source = Path(distribution.locate_file(item))
            if not source.is_file():
                continue
            relative = Path(name) / path
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            selected.append(relative.as_posix())
        if not selected:
            raise RuntimeError(f"No installed redistribution notices found for {name}")
        entries.append({"name": name, "version": distribution.version, "files": sorted(selected)})
    python_notice = Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "LICENSE.txt"
    if not python_notice.is_file():
        raise RuntimeError("The embedded Python runtime's license could not be located")
    shutil.copyfile(python_notice, destination / "LICENSE.Python.txt")
    entries.append({"name": "CPython", "version": sys.version.split()[0], "files": ["LICENSE.Python.txt"]})
    manifest = {"formatVersion": 1, "packages": entries}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    index = ["AgentTrainer Astra — bundled runtime notices", "",
             "These files reproduce the notices distributed with the bundled dependencies.",
             "ConvNeXt weight and source attribution is in the adjacent Weights folder.", ""]
    for entry in entries:
        index.append(f"{entry['name']} {entry['version']}")
        index.extend(f"  {path}" for path in entry["files"])
        index.append("")
    (destination / "README.txt").write_text("\n".join(index))
    return manifest
