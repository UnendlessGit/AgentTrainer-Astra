"""Validate a relocatable Astra bundle without launching UI or requesting TCC."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess


def output(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT).strip()


def validate_load_graph(bundle: Path, code: list[Path], executables: list[Path]) -> None:
    """Resolve each private load edge in its executable/rpath context.

    macOS keeps many system libraries only in its dyld shared cache. The public
    cache query validates those paths without loading/initializing frameworks.
    """
    bundle = bundle.resolve(strict=True)
    code = [path.resolve(strict=True) for path in code]
    executables = [path.resolve(strict=True) for path in executables]
    contains_system = ctypes.CDLL(None)._dyld_shared_cache_contains_path
    contains_system.argtypes = [ctypes.c_char_p]; contains_system.restype = ctypes.c_bool
    metadata = {}
    for path in code:
        paths, dependencies = [], []
        for block in output("otool", "-arch", "arm64", "-l", str(path)).split("Load command ")[1:]:
            command = re.search(r"\bcmd (LC_[A-Z0-9_]+)", block)
            if not command:
                continue
            kind = command.group(1)
            if kind == "LC_RPATH":
                match = re.search(r"\bpath (.*?) \(offset [0-9]+\)", block)
                if not match: raise ValueError("Malformed runtime search path")
                paths.append(match.group(1))
            elif kind in {"LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB", "LC_REEXPORT_DYLIB", "LC_LOAD_UPWARD_DYLIB"}:
                match = re.search(r"\bname (.*?) \(offset [0-9]+\)", block)
                if not match: raise ValueError("Malformed dynamic dependency")
                dependencies.append((match.group(1), kind == "LC_LOAD_WEAK_DYLIB"))
        metadata[path] = paths, dependencies

    def system(path: Path) -> bool:
        return str(path).startswith(("/usr/lib/", "/System/Library/"))

    def expanded(value: str, loader: Path, executable: Path) -> Path:
        if value == "@loader_path" or value.startswith("@loader_path/"):
            path = loader.parent / value.removeprefix("@loader_path").lstrip("/")
        elif value == "@executable_path" or value.startswith("@executable_path/"):
            path = executable.parent / value.removeprefix("@executable_path").lstrip("/")
        elif value.startswith("/"): path = Path(value)
        else: raise ValueError(f"Unsupported loader-relative path: {value}")
        path = path.resolve()
        if not path.is_relative_to(bundle) and not system(path):
            raise ValueError(f"Dynamic load path escapes the bundle: {value}")
        return path

    def available(path: Path) -> bool:
        return path.is_file() or (system(path) and contains_system(os.fsencode(path)))

    visited = set()
    def walk(path: Path, executable: Path, inherited: tuple[Path, ...]) -> None:
        own, dependencies = metadata[path]
        search = tuple(dict.fromkeys([*(expanded(value, path, executable) for value in own), *inherited]))
        key = path, executable, search
        if key in visited: return
        visited.add(key)
        if len(visited) > 4096: raise ValueError("The bundle's dynamic-load graph exceeds its supported bound")
        for dependency, weak in dependencies:
            if dependency.startswith("@rpath/"):
                candidates = [expanded(str(parent / dependency.removeprefix("@rpath/")), path, executable) for parent in search]
                resolved = next((candidate for candidate in candidates if available(candidate)), None)
            else:
                candidate = expanded(dependency, path, executable)
                resolved = candidate if available(candidate) else None
                if resolved is None and weak and system(candidate): continue
            if resolved is None: raise ValueError(f"{path.relative_to(bundle)} cannot resolve {dependency}")
            if system(resolved): continue
            if resolved not in metadata: raise ValueError(f"A private dependency is not validated Mach-O code: {resolved}")
            walk(resolved, executable, search)

    for executable in executables: walk(executable, executable, ())
    # Python extensions are loaded by absolute filename, not necessarily listed
    # in a Mach-O LC_LOAD command. Validate them in their enclosing helper's
    # executable context as well as traversing ordinary static load edges.
    for path in code:
        owner = next((executable for executable in reversed(executables) if path.is_relative_to(executable.parent.parent)), executables[0])
        inherited = tuple(expanded(value, owner, owner) for value in metadata[owner][0])
        walk(path, owner, inherited)


def validate_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve(strict=True)
    output("codesign", "--verify", "--deep", "--strict", str(bundle))
    ad_hoc = "Signature=adhoc" in output("codesign", "-dv", "--verbose=4", str(bundle))
    info = plistlib.loads((bundle / "Contents/Info.plist").read_bytes())
    if info.get("CFBundleIdentifier") != "com.unendless.agenttrainer.astra" or info.get("CFBundleExecutable") != "AgentTrainerAstra":
        raise ValueError("The selected bundle is not AgentTrainer Astra")
    version, build = info["CFBundleShortVersionString"], info["CFBundleVersion"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("The application version must have three numeric components")
    helpers = {"AstraCompute": "compute", "AstraControl": "control"}
    for name, suffix in helpers.items():
        path = bundle / f"Contents/Helpers/{name}.app"
        metadata = plistlib.loads((path / "Contents/Info.plist").read_bytes())
        if metadata.get("CFBundleIdentifier") != f"com.unendless.agenttrainer.astra.{suffix}" or metadata.get("CFBundleExecutable") != name:
            raise ValueError(f"{name} has invalid bundle identity")
        if (metadata.get("CFBundleShortVersionString"), metadata.get("CFBundleVersion")) != (version, build):
            raise ValueError(f"{name} and the application have different versions")
    notices = bundle / "Contents/Resources/ThirdPartyNotices"
    manifest = json.loads((notices / "manifest.json").read_text())
    expected = {"mlx", "mlx-metal", "numpy", "safetensors", "pillow", "pyinstaller", "CPython"}
    if {item["name"] for item in manifest["packages"]} != expected:
        raise ValueError("Bundled runtime notices are incomplete")
    for item in manifest["packages"]:
        if not item["files"] or any(not (notices / file).resolve(strict=True).is_relative_to(notices) for file in item["files"]):
            raise ValueError("A runtime notice is missing or escapes the bundle")
    weights = bundle / "Contents/Resources/Weights"
    weight_info = json.loads((weights / "manifest.json").read_text())["convnextTiny"]
    with (weights / weight_info["artifact"]).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != weight_info["artifactSHA256"]:
            raise ValueError("The bundled pretrained visual weights changed")
    if not (weights / "LICENSE.ConvNeXt").is_file() or not (bundle / "Contents/Resources/AppIcon.icns").is_file():
        raise ValueError("The visual attribution or application icon is missing")
    if not list(bundle.rglob("mlx.metallib")):
        raise ValueError("The MLX Metal library is missing")
    macho_magics = {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"}
    code = []
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(bundle):
                raise ValueError(f"Bundle link escapes its installation: {path.relative_to(bundle)}")
            continue
        if not path.is_file():
            continue
        with path.open("rb") as file:
            magic = file.read(4)
        if magic not in macho_magics:
            continue
        architectures = output("lipo", "-archs", str(path)).split()
        if "arm64" not in architectures:
            raise ValueError(f"Bundled code cannot run on Apple Silicon: {path.relative_to(bundle)}")
        code.append(path.relative_to(bundle).as_posix())
    expected_binaries = ["Contents/MacOS/AgentTrainerAstra"] + [f"Contents/Helpers/{name}.app/Contents/MacOS/{name}" for name in helpers]
    if not set(expected_binaries).issubset(code):
        raise ValueError("A required executable is absent or not native Apple Silicon code")
    executables = [bundle / item for item in expected_binaries]
    if any(not os.access(path, os.X_OK) for path in executables):
        raise ValueError("A required application entry point is not executable")
    validate_load_graph(bundle, [bundle / path for path in code], executables)
    return {"bundle": str(bundle), "version": version, "build": build, "nativeCodeFiles": len(code),
            "nestedSignaturesVerified": True, "weightsVerified": True, "noticesVerified": True,
            "externalRuntimePaths": False, "privateLoadGraphVerified": True,
            "adHocSigned": ad_hoc, "notarizationChecked": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    print(json.dumps(validate_bundle(parser.parse_args().bundle), indent=2))
