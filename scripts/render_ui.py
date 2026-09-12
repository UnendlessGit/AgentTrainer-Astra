#!/usr/bin/env python3
"""Export actual SwiftUI/AppKit views using only windows owned by the test host.

No screen capture, accessibility reads, privacy grants, training, or input posting.
Images contain generated fixture metadata and are never release screenshots.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from apple_toolchain import build_environment


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / ".local/ui-renders")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="astra-render-", dir=output) as workspace:
        environment = {**build_environment(), "ASTRA_UI_RENDER_DIR": str(output), "ASTRA_WORKSPACE_ROOT": workspace}
        result = subprocess.run(["swift", "test", "--filter", "renderActualApplicationViews"], cwd=root,
                                env=environment, capture_output=True, text=True, timeout=180)
        (output / "render.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Native view export failed. See {output / 'render.log'}\n" + result.stderr[-5000:])
    manifest = json.loads((output / "manifest.json").read_text())
    print(json.dumps({"output": str(output), "renderedViews": len(manifest["images"]),
                      "screenCapture": False, "accessibilityReads": False, "fixtureDataOnly": True}, indent=2))


if __name__ == "__main__":
    main()
