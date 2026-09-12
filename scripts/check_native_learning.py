#!/usr/bin/env python3
"""Run the actual native coordinator against an assembled local compute bundle."""
import argparse
import json
from pathlib import Path
import subprocess
import uuid
from apple_toolchain import build_environment

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if not (bundle / "Contents/Helpers/AstraCompute.app/Contents/MacOS/AstraCompute").is_file():
        parser.error("An assembled Astra app bundle is required")
    output = (args.output or ROOT / ".local/verification" / ("native-learning-" + str(uuid.uuid4()))).resolve()
    if output.exists():
        parser.error("Use a new verification output directory to preserve earlier evidence")
    output.mkdir(parents=True)
    environment = dict(build_environment(), ASTRA_VERIFY_BUNDLE=str(bundle), ASTRA_NATIVE_VERIFY_ROOT=str(output))
    with (output / "swift-test.log").open("wb") as log:
        subprocess.run(["swift", "test", "--filter", "BundledLearningTests"], cwd=ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    report = json.loads((output / "native-learning-report.json").read_text())
    if not report["completed"] or report["privacyPermissionsUsed"]:
        raise RuntimeError("Native bundled verification did not establish the expected workflow")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
