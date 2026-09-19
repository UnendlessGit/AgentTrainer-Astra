#!/usr/bin/env python3
"""Check the packaged CPU collector and separate PPO worker with native leases."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["ASTRA_COMPUTE_TEST_BINARY"] = str(args.binary.resolve(strict=True))
    environment["ASTRA_COMPUTE_TEST_OFFLINE"] = "1" if args.offline else "0"
    # Development Python creates fixtures; both target child roles receive
    # only a system PATH and, when requested, an enforced network denial.
    subprocess.run([sys.executable, "-W", "error", "-m", "pytest", "-q",
        "python/tests/test_collector.py::test_native_collector_cpu_lease_to_immutable_rollout_and_separate_learner_job[complete]",
        "python/tests/test_collector.py::test_native_collector_cpu_lease_to_immutable_rollout_and_separate_learner_job[cancel_at_boundary]"],
        cwd=root, env=environment, check=True, timeout=180)
    print("Packaged collector/learner lease, update, boundary and cancellation checks passed.")


if __name__ == "__main__":
    main()
