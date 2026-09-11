#!/usr/bin/env python3
"""One project-local, graceful stop/build/bundle launch entry point."""
from pathlib import Path
import argparse
import os
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "build/AgentTrainer Astra.app"
BINARY = BUNDLE / "Contents/MacOS/AgentTrainerAstra"


def processes():
    result = subprocess.run(["pgrep", "-f", "^" + re.escape(str(BINARY)) + "$"], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("Unable to inspect the existing development process")
    return [int(value) for value in result.stdout.split()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="run", choices=("run", "debug", "logs", "telemetry", "verify"))
    args = parser.parse_args([argument.removeprefix("--") for argument in sys.argv[1:]])
    previous = processes()
    for process in previous:
        try:
            os.kill(process, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60
    while set(processes()) & set(previous):
        if time.monotonic() >= deadline:
            raise RuntimeError("Astra is still saving or releasing controls. Wait for it to finish before rebuilding.")
        time.sleep(0.1)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_app.py")], cwd=ROOT, check=True)
    if args.mode == "debug":
        subprocess.run(["lldb", "--", str(BINARY)], cwd=ROOT, check=True)
        return
    launch = ["/usr/bin/open", "-n"]
    if workspace := os.environ.get("ASTRA_WORKSPACE_ROOT"):
        if not Path(workspace).is_absolute():
            raise ValueError("An isolated Astra workspace must use an absolute path")
        launch.extend(["--env", "ASTRA_WORKSPACE_ROOT=" + workspace])
    launch.append(str(BUNDLE))
    subprocess.run(launch, check=True)
    if args.mode in ("logs", "telemetry"):
        predicate = 'process == "AgentTrainerAstra"' if args.mode == "logs" else 'subsystem == "com.unendless.agenttrainer.astra"'
        subprocess.run(["/usr/bin/log", "stream", "--info", "--style", "compact", "--predicate", predicate], check=True)
    elif args.mode == "verify":
        deadline = time.monotonic() + 10
        while not processes():
            if time.monotonic() >= deadline:
                raise RuntimeError("The application bundle did not remain running after launch")
            time.sleep(0.1)
        print(f"Running development bundle: {BUNDLE}")


if __name__ == "__main__":
    main()
