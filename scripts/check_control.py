#!/usr/bin/env python3
"""Check packaged control IPC and joined unarmed cleanup, without TCC or input.

This deliberately never requests permissions, arms desktop control or posts an
input event. Real permission and control qualification is a separate release gate.
"""
from pathlib import Path
import argparse
import json
import select
import signal
import subprocess
import tempfile
import time


ENVIRONMENT = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}


def request(kind: str, index: int = 0, **fields: object) -> dict:
    return dict(version=1, kind=kind, sequence=index,
                requestID=f"00000000-0000-0000-0000-{index + 1:012d}", payload={}, **fields)


def framed(messages: list[dict]) -> str:
    return "".join(json.dumps(value) + "\n" for value in messages)


def check_stream(binary: Path, source: str) -> list[dict]:
    result = subprocess.run([str(binary)], input=source, capture_output=True, text=True,
                            env=ENVIRONMENT, cwd="/tmp", timeout=15)
    if result.returncode:
        raise RuntimeError(f"Control helper failed ({result.returncode}): {result.stderr[-4096:]}")
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert responses[0]["kind"] == "hello" and responses[0]["payload"]["role"] == "control"
    assert [value["sequence"] for value in responses] == list(range(len(responses)))
    assert not any(value["kind"] == "control.receipt" for value in responses), responses
    return responses


def check_signal(binary: Path) -> None:
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=ENVIRONMENT, cwd="/tmp", bufsize=0)
    try:
        assert process.stdin and process.stdout
        process.stdin.write(framed([request("ping")]).encode())
        # Seeing the ping proves the signal handlers and read loop are installed.
        buffer = b""
        deadline = time.monotonic() + 10
        found = False
        while not found:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and select.select([process.stdout], [], [], remaining)[0], "Helper handshake timed out"
            chunk = process.stdout.read(16_384)
            assert chunk, "Helper closed before handshake"
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                found |= json.loads(line)["kind"] == "ack"
        process.send_signal(signal.SIGTERM)
        # Keep stdin open: termination must come from the signal, not from the
        # EOF that communicate() would otherwise send on our behalf.
        process.wait(timeout=10)
        _, error = process.communicate(timeout=10)
        assert process.returncode == 0, error[-4096:]
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


def check_backpressure(binary: Path) -> None:
    # Deliberately never drain stdout while the process runs. A regular-file
    # command source avoids confusing an input writer stall with output recovery.
    with tempfile.TemporaryFile() as source:
        for index in range(50_000):
            source.write(framed([request("ping", index)]).encode())
        source.seek(0)
        process = subprocess.Popen([str(binary)], stdin=source, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=ENVIRONMENT, cwd="/tmp")
        try:
            process.wait(timeout=15)
            out, error = process.communicate(timeout=2)
            assert process.returncode == 0, error[-4096:]
            # The bounded output channel has failed closed; it cannot emit all
            # 50,000 replies into an unread pipe or retain them indefinitely.
            assert len(out) < 4 * 1_048_576
            assert b'"kind":"hello"' in out
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    kinds = ["ping", "state", "disarm", "shutdown"]
    requests = [request(kind, index) for index, kind in enumerate(kinds)]
    responses = check_stream(binary, framed(requests))
    assert all(operation in responses[0]["payload"]["capabilities"] for operation in kinds)
    replies = [value for value in responses if value["kind"] in {"ack", "error"}]
    assert [value["kind"] for value in replies] == ["ack"] * len(kinds), responses
    assert [value["requestID"].lower() for value in replies] == [value["requestID"] for value in requests]
    assert all(reply["payload"]["cleanupSettled"] is True for reply in replies[1:]), responses
    assert replies[-1]["payload"]["stopped"] is True
    assert check_stream(binary, "")[-1]["kind"] == "hello"
    for malformed in ['{"version":1', "x" * 1_048_577, "{}\n"]:
        assert check_stream(binary, malformed)[-1]["kind"] == "control.error"
    wrong_sequence = check_stream(binary, framed([request("ping", 1)]))
    assert wrong_sequence[-1]["kind"] == "error" and wrong_sequence[-1]["payload"]["recoverable"] is False
    inactive = check_stream(binary, framed([
        request("observation", 0, runID="00000000-0000-0000-0000-000000000099"),
        request("heartbeat", 1, runID="00000000-0000-0000-0000-000000000099"),
        request("shutdown", 2),
    ]))
    assert [reply["kind"] for reply in inactive[1:]] == ["error", "error", "ack"], inactive
    assert inactive[-1]["payload"]["cleanupSettled"] is True
    check_signal(binary)
    check_backpressure(binary)
    print(json.dumps({"passed": True, "permissionsRequested": False, "controlArmed": False,
                      "cleanupSettled": True, "processExited": True,
                      "checks": ["ordered IPC", "inactive observation/heartbeat", "graceful shutdown",
                                 "EOF", "malformed and oversized input", "SIGTERM", "output backpressure"]}, indent=2))


if __name__ == "__main__":
    main()
