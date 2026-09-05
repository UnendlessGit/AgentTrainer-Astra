"""Packaged worker entry point; capabilities list only implemented operations."""
from __future__ import annotations

import argparse
import json
import platform
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from astra import __version__
from astra.protocol import Message, ProtocolError, Sender, messages


def capabilities() -> dict:
    return {
        "role": "compute", "protocolVersion": 1, "runtimeVersion": __version__,
        "capabilities": ["capabilities", "ping", "shutdown", "diagnose"],
        "pythonVersion": platform.python_version(), "metalAvailable": mx.metal.is_available(),
        "device": mx.device_info(),
    }


def diagnose() -> dict:
    # Forces Metal library loading and materialization inside the frozen worker.
    left = mx.arange(128, dtype=mx.float32).reshape(8, 16)
    product = left @ left.T
    mx.eval(product)
    state = nn.GRU(4, 8)(mx.ones((1, 3, 4)))
    mx.eval(state)
    array = np.asarray(state)
    return {**capabilities(), "matrixCheck": float(product[0, 0].item()),
            "recurrentShape": list(array.shape), "recurrentFinite": bool(np.isfinite(array).all()),
            "activeBytes": mx.get_active_memory(), "peakBytes": mx.get_peak_memory()}


def serve() -> int:
    sender = Sender(sys.stdout.buffer)
    sender.send("hello", capabilities())
    try:
        for request in messages(sys.stdin.buffer):
            if request.kind == "shutdown":
                sender.send("ack", {"stopped": True}, request=request)
                return 0
            if request.kind == "ping":
                sender.send("ack", {"alive": True}, request=request)
            elif request.kind == "capabilities":
                sender.send("ack", capabilities(), request=request)
            elif request.kind == "diagnose":
                sender.send("ack", diagnose(), request=request)
            else:
                sender.send("error", {"code": "protocol.unsupportedOperation",
                                      "message": "This runtime does not support the requested operation.",
                                      "recoverable": True}, request=request)
    except ProtocolError as error:
        sender.send("error", {"code": error.code, "message": str(error), "recoverable": False})
        return 65
    except BrokenPipeError:
        return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentTrainer Astra local compute worker")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    if args.diagnose:
        print(json.dumps(diagnose(), allow_nan=False))
        return 0
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
