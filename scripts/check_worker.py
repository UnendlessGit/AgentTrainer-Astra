#!/usr/bin/env python3
"""Exercise a frozen compute executable outside the development environment."""
from pathlib import Path
import argparse
import json
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    command = [str(binary)]
    if args.offline:
        command = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", *command]
    requests = [
        {"version": 1, "kind": kind, "sequence": index,
         "requestID": f"00000000-0000-0000-0000-{index + 1:012d}", "payload": {}}
        for index, kind in enumerate(["ping", "diagnose", "shutdown"])
    ]
    result = subprocess.run(
        command, input="".join(json.dumps(value) + "\n" for value in requests), text=True,
        capture_output=True, env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}, cwd="/tmp", timeout=45,
    )
    if result.returncode:
        raise RuntimeError(f"Frozen worker failed ({result.returncode}): {result.stderr[-8192:]}")
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert [item["kind"] for item in responses] == ["hello", "ack", "ack", "ack"]
    assert [item["sequence"] for item in responses] == [0, 1, 2, 3]
    assert [item["requestID"] for item in responses[1:]] == [item["requestID"] for item in requests]
    diagnostic = responses[2]["payload"]
    assert diagnostic["metalAvailable"] is True
    assert diagnostic["matrixCheck"] == 1240.0
    assert diagnostic["recurrentShape"] == [1, 3, 8]
    assert diagnostic["recurrentFinite"] is True
    assert responses[-1]["payload"]["stopped"] is True
    print(json.dumps({"passed": True, "offline": args.offline, "diagnostic": diagnostic}, indent=2))


if __name__ == "__main__":
    main()
