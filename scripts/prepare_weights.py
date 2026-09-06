#!/usr/bin/env python3
"""Build-time conversion and independent numerical qualification of bundled weights.

Run with the repository's dev environment. PyTorch is never a runtime dependency.
The only accepted serialized input has a pinned SHA-256; weights_only loading adds
a second restriction. Downloads are atomic and bounded to the known byte count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(directory: Path, spec: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / Path(spec["sourceURL"]).name
    if destination.exists():
        if destination.stat().st_size != spec["sourceBytes"] or digest(destination) != spec["sourceSHA256"]:
            raise ValueError(f"Existing source weight file fails its pinned digest: {destination}")
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix="download-", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(spec["sourceURL"], timeout=60) as source:
                remaining = spec["sourceBytes"]
                while chunk := source.read(min(1024 * 1024, remaining + 1)):
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise ValueError("Weight download exceeds its declared size")
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != spec["sourceBytes"] or digest(temporary) != spec["sourceSHA256"]:
            raise ValueError("Downloaded weights fail the pinned size/digest")
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reference_state(state: dict) -> dict:
    """Map original FB parameters to torchvision's independently shipped layers."""
    result = {}
    for name, value in state.items():
        match = re.fullmatch(r"downsample_layers\.(\d)\.(\d)\.(weight|bias)", name)
        if match:
            stage, slot, suffix = match.groups()
            target = f"features.{2 * int(stage)}.{slot}.{suffix}"
        elif match := re.fullmatch(r"stages\.(\d)\.(\d+)\.(dwconv|norm|pwconv1|pwconv2)\.(weight|bias)", name):
            stage, block, component, suffix = match.groups()
            slot = {"dwconv": 0, "norm": 2, "pwconv1": 3, "pwconv2": 5}[component]
            target = f"features.{2 * int(stage) + 1}.{block}.block.{slot}.{suffix}"
        elif match := re.fullmatch(r"stages\.(\d)\.(\d+)\.gamma", name):
            target = f"features.{2 * int(match[1]) + 1}.{match[2]}.layer_scale"
            value = value[:, None, None]
        elif name.startswith("norm."):
            target = "classifier.0." + name.split(".")[-1]
        elif name.startswith("head."):
            target = "classifier.2." + name.split(".")[-1]
        else:
            raise ValueError(f"Unexpected reference parameter {name}")
        if target in result:
            raise ValueError(f"Duplicate reference parameter {target}")
        result[target] = value
    return result


def qualify(encoder, original: dict) -> list[dict]:
    import mlx.core as mx
    import numpy as np
    import torch
    from torchvision.models import convnext_tiny

    torch.set_num_threads(2)
    reference = convnext_tiny(weights=None).eval()
    reference.load_state_dict(reference_state(original), strict=True)
    reports = []
    rng = np.random.default_rng(718)
    # Non-square and production-resolution inputs expose padding/layout mistakes
    # that checking only a 224-pixel classification input would miss.
    for height, width in ((224, 224), (160, 256), (432, 768)):
        pixels = rng.normal(size=(1, height, width, 3)).astype(np.float32)
        outputs = encoder(mx.array(pixels))
        summary = encoder.summary(outputs)
        mx.eval(outputs, summary)
        with torch.no_grad():
            value = torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy())
            expected = []
            for index, layer in enumerate(reference.features):
                value = layer(value)
                if index % 2:
                    expected.append(value.permute(0, 2, 3, 1).numpy())
            expected_summary = reference.classifier[0](reference.avgpool(value)).flatten(1).numpy()
        differences = []
        for index, (actual, target) in enumerate(zip([*outputs, summary], [*expected, expected_summary])):
            actual = np.asarray(actual)
            # FP32 Metal convolution and CPU torch use different summation order.
            # Absolute tolerance protects near-zero entries; normalized RMS and
            # per-stage max error are retained as evidence, not just a pass flag.
            difference = np.abs(actual - target)
            error = float(difference.max())
            rms = float(np.sqrt(np.mean((actual - target) ** 2)))
            normalized = rms / max(float(np.sqrt(np.mean(target ** 2))), 1e-8)
            if not np.all(np.isfinite(actual)) or not np.allclose(actual, target, atol=1e-3, rtol=1e-3):
                raise AssertionError(f"ConvNeXt mismatch at {height}x{width} stage {index}: max={error}, relative RMS={normalized}")
            differences.append({"stage": index, "maximumAbsoluteError": error, "normalizedRMSError": normalized})
        reports.append({"shape": [1, height, width, 3], "comparisons": differences})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-reference", action="store_true", help="Convert only; not sufficient for release qualification")
    parser.add_argument("--directory", type=Path, default=ROOT / "vendor/weights")
    args = parser.parse_args()
    import mlx.core as mx
    import torch
    from astra.model.config import ModelConfig
    from astra.model.convnext import ConvNeXtEncoder, convert_facebook_weights

    spec = json.loads((ROOT / "assets/weights.json").read_text())["convnextTiny"]
    source = acquire(args.directory, spec)
    state = torch.load(source, map_location="cpu", weights_only=True)["model"]
    converted = convert_facebook_weights({name: tensor.numpy() for name, tensor in state.items()})
    encoder = ConvNeXtEncoder(ModelConfig())
    encoder.load_weights(list(converted.items()), strict=True)
    mx.eval(encoder.parameters())
    comparisons = [] if args.skip_reference else qualify(encoder, state)
    destination = args.directory / spec["artifact"]
    temporary = destination.with_suffix(".partial.safetensors")
    try:
        encoder.save_weights(str(temporary))
        # Strict read-back uses the same public path as a packaged app.
        ConvNeXtEncoder(ModelConfig()).load_pretrained(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"sourceSHA256": digest(source), "artifactSHA256": digest(destination),
              "parameterCount": sum(value.size for value in converted.values()),
              "mlxVersion": mx.__version__, "torchVersion": torch.__version__,
              "referenceQualified": not args.skip_reference, "comparisons": comparisons}
    (args.directory / "conversion-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
