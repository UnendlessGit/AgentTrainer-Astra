"""Packaged worker entry point; capabilities list only implemented operations."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import platform
import select
import sys
import threading

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from astra import __version__
from astra.protocol import MAX_MESSAGE_BYTES, Message, ProtocolError
from astra.jobs import JOB_OPERATIONS, JobError, JobManager


class OutputOverflow(RuntimeError):
    pass


class BoundedSender:
    """One writer, bounded bytes, coalesced progress, reliable terminal replies.

    Only progress can be replaced/dropped. An undeliverable control/terminal
    reply fails the channel so the input loop cancels work and exits. Writes
    bypass Python's stdout buffer to avoid a blocked flush at process exit.
    """
    def __init__(self, stream, *, maximum_bytes=4 * MAX_MESSAGE_BYTES, maximum_messages=128):
        self.stream = stream
        self.maximum_bytes, self.maximum_messages = maximum_bytes, maximum_messages
        self.failed = threading.Event()
        self._condition = threading.Condition()
        self._queue = deque()
        self._bytes = 0
        self._closing = False
        self._thread = threading.Thread(target=self._write, name="Astra protocol writer", daemon=True)
        self._thread.start()

    def send(self, kind, payload, *, request=None):
        # Round-trip creates an immutable owned snapshot before the producer
        # changes its metrics. Reserve enough bytes for the eventual sequence.
        message = Message(kind, 0, payload, request_id=request.request_id if request else None,
                          run_id=request.run_id if request else None)
        try:
            encoded = message.encode()
        except ProtocolError as error:
            self.failed.set()
            raise OutputOverflow("Compute output could not encode a bounded protocol message") from error
        owned = Message.decode(encoded)
        size = len(encoded) + 32
        with self._condition:
            if self.failed.is_set() or self._closing:
                raise OutputOverflow("Compute output channel is unavailable")
            if kind == "job.progress":
                for index, (old, cost) in enumerate(self._queue):
                    if old.kind == kind and old.run_id == owned.run_id:
                        if self._bytes - cost + size <= self.maximum_bytes:
                            self._queue[index] = (owned, size); self._bytes += size - cost
                        return
            if len(self._queue) >= self.maximum_messages or self._bytes + size > self.maximum_bytes:
                if kind == "job.progress": return
                # Reclaim progress before rejecting a terminal/control message.
                retained = deque((item, cost) for item, cost in self._queue if item.kind != "job.progress")
                self._queue = retained; self._bytes = sum(cost for _, cost in retained)
                if len(retained) >= self.maximum_messages or self._bytes + size > self.maximum_bytes:
                    self.failed.set(); self._condition.notify_all()
                    raise OutputOverflow("Compute output backpressure exceeded its bounded queue")
            self._queue.append((owned, size)); self._bytes += size
            self._condition.notify()

    def _write(self):
        sequence = 0
        try:
            try:
                descriptor = self.stream.fileno()
            except (AttributeError, OSError):
                descriptor = None
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._queue or self._closing or self.failed.is_set())
                    if self.failed.is_set(): return
                    if not self._queue: return
                    message, cost = self._queue.popleft(); self._bytes -= cost
                encoded = Message(message.kind, sequence, message.payload, message.request_id, message.run_id).encode()
                remaining = memoryview(encoded)
                while remaining:
                    count = os.write(descriptor, remaining) if descriptor is not None else self.stream.write(remaining)
                    if count is None or count <= 0:
                        raise BrokenPipeError("Compute output closed during a message")
                    remaining = remaining[count:]
                if descriptor is None: self.stream.flush()
                sequence += 1
        except Exception:
            self.failed.set()
            with self._condition: self._condition.notify_all()

    def close(self, timeout=5.0):
        with self._condition:
            self._closing = True; self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive() and not self.failed.is_set()


def requests(stream, output_failed):
    """Bounded partial-line framing with an output-failure wakeup every 100 ms."""
    descriptor = stream.fileno()
    pending = bytearray()
    previous_sequence = -1
    while not output_failed.is_set():
        if not select.select([descriptor], [], [], .1)[0]: continue
        data = os.read(descriptor, 65536)
        if not data:
            if pending: raise ProtocolError("protocol.framing", "Input closed inside a protocol frame.")
            return
        pending.extend(data)
        while (newline := pending.find(b"\n")) >= 0:
            if newline + 1 > MAX_MESSAGE_BYTES:
                raise ProtocolError("protocol.tooLarge", "Message exceeds the transport limit.")
            request = Message.decode(bytes(pending[:newline + 1]))
            del pending[:newline + 1]
            if request.sequence <= previous_sequence:
                raise ProtocolError("protocol.sequence", "Incoming sender sequence must increase.")
            previous_sequence = request.sequence
            yield request
        if len(pending) >= MAX_MESSAGE_BYTES:
            raise ProtocolError("protocol.tooLarge", "Message exceeds the transport limit.")
    raise OutputOverflow("Compute output channel failed")


def capabilities(role="compute") -> dict:
    if role == "collector":
        from astra.collector import COLLECTOR_OPERATIONS
        return {"role":"collector","protocolVersion":1,"runtimeVersion":__version__,
            "capabilities":["capabilities","ping","shutdown",*COLLECTOR_OPERATIONS],
            "pythonVersion":platform.python_version(),"collectionVersion":1}
    if role == "actor":
        from astra.inference import INFERENCE_OPERATIONS
        # Liveness/control never enters the actor's MLX owner thread. Device
        # initialization and all tensors belong to inference operations there.
        return {"role": "actor", "protocolVersion": 1, "runtimeVersion": __version__,
                "capabilities": ["capabilities", "ping", "shutdown", *INFERENCE_OPERATIONS],
                "pythonVersion": platform.python_version()}
    return {
        "role": "compute", "protocolVersion": 1, "runtimeVersion": __version__,
        "capabilities": ["capabilities", "ping", "shutdown", "diagnose", *JOB_OPERATIONS, "cancel", "job.status", "job.externalBoundary"],
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


def serve(role="compute") -> int:
    sender = BoundedSender(sys.stdout.buffer)
    if role == "actor":
        from astra.inference import INFERENCE_OPERATIONS, InferenceManager
        manager = InferenceManager(sender.send)
    elif role == "collector":
        from astra.collector import COLLECTOR_OPERATIONS, CollectorManager
        manager = CollectorManager(sender.send)
    else:
        manager = JobManager(sender.send)
    exit_code = 0
    sender.send("hello", capabilities(role))
    try:
        for request in requests(sys.stdin.buffer, sender.failed):
            try:
                if request.kind == "shutdown":
                    if request.payload: raise JobError("job.invalidConfiguration", "Shutdown payload must be empty")
                    clean = manager.close()
                    sender.send("ack", {"stopped": clean}, request=request)
                    if not clean: exit_code = 70
                    break
                if request.kind in ("ping", "capabilities", "diagnose"):
                    if request.payload: raise JobError("job.invalidConfiguration", "This operation requires an empty payload")
                    if request.kind == "ping": sender.send("ack", {"alive": True, "busy": manager.busy}, request=request)
                    elif request.kind == "capabilities": sender.send("ack", capabilities(role), request=request)
                    elif role != "compute": raise JobError("protocol.unsupportedOperation", "Diagnostics are available in the compute role")
                    elif manager.busy: raise JobError("job.busy", "Diagnostics require an idle compute role")
                    else: sender.send("ack", diagnose(), request=request)
                elif role == "actor" and request.kind in INFERENCE_OPERATIONS:
                    manager.submit(request)
                elif role == "collector" and request.kind in COLLECTOR_OPERATIONS:
                    manager.submit(request)
                elif role == "compute" and request.kind == "job.externalBoundary":
                    manager.external_boundary(request)
                elif role == "compute" and request.kind in JOB_OPERATIONS:
                    manager.submit(request)
                elif role == "compute" and request.kind in ("cancel", "job.status"):
                    if set(request.payload) != {"jobID"} or not isinstance(request.payload["jobID"], str):
                        raise JobError("job.invalidConfiguration", "Job control requires a jobID")
                    result = (manager.cancel(request.payload["jobID"], run_id=request.run_id) if request.kind == "cancel"
                              else manager.status(request.payload["jobID"]))
                    sender.send("ack", result, request=request)
                else:
                    raise JobError("protocol.unsupportedOperation", "This runtime does not support the requested operation.")
            except (ValueError, TypeError, KeyError, OSError) as error:
                sender.send("error", {"code": getattr(error, "code", "job.invalidConfiguration"),
                                      "message": str(error)[:2048] or type(error).__name__, "recoverable": True}, request=request)
    except ProtocolError as error:
        sender.send("error", {"code": error.code, "message": str(error), "recoverable": False})
        exit_code = 65
    except (BrokenPipeError, OutputOverflow):
        exit_code = 74
    finally:
        if not manager.close(): exit_code = 70
        if not sender.close(): exit_code = 74
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentTrainer Astra local compute worker")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--role", choices=("compute", "actor", "collector"), default="compute")
    args = parser.parse_args()
    if args.diagnose:
        if args.role != "compute": parser.error("--diagnose requires the compute role")
        print(json.dumps(diagnose(), allow_nan=False))
        return 0
    return serve(args.role)


if __name__ == "__main__":
    raise SystemExit(main())
