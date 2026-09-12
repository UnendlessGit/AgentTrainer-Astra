"""Read-only mapped BGRA transport with an explicit owned MLX ingestion copy.

Native publication precedes delivery of a reference on the inherited control
pipe. Slots stay immutable until the exact returned acknowledgement is sent.
Neither a timeout nor closing this reader authorizes reuse of an unacknowledged
slot; the coordinator must join the consumer before retiring that whole ring.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import mmap
import os
from pathlib import Path
import stat
import struct
import threading
import uuid

import mlx.core as mx
import numpy as np

from .recordings import validate_surface

VERSION = 1
HEADER_BYTES = SLOT_HEADER_BYTES = 128
MAXIMUM_SLOTS = 16
MAXIMUM_SLOT_BYTES = 256 * 1024**2
MAXIMUM_FILE_BYTES = 1024**3


class FrameRingError(ValueError):
    pass


def _uint(value, maximum=2**64 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise FrameRingError("Invalid frame-ring integer")
    return value


def _uuid(value):
    if type(value) is not str or len(value) != 36:
        raise FrameRingError("Invalid frame-ring UUID")
    try:
        return uuid.UUID(value)
    except ValueError as error:
        raise FrameRingError("Invalid frame-ring UUID") from error


def _metadata_fingerprint(metadata):
    required = {"id", "eventNanos", "observedNanos", "surface", "byteCount", "pixelFormat", "codec"}
    if type(metadata) is not dict or set(metadata) != required:
        raise FrameRingError("Invalid frame-ring metadata fields")
    try:
        frame_id = _uuid(metadata["id"])
        source, observed = _uint(metadata["eventNanos"]), _uint(metadata["observedNanos"])
        surface = validate_surface(metadata["surface"])
        size = _uint(metadata["byteCount"], MAXIMUM_SLOT_BYTES)
        if size != surface["pixelWidth"] * surface["pixelHeight"] * 4 or metadata["pixelFormat"] != "bgra8-srgb" or metadata["codec"] != "raw":
            raise FrameRingError("The ring transports compact raw BGRA frames only")
        identity = surface["id"].encode()
        data = bytearray(b"ASTRAM01" + frame_id.bytes + struct.pack("<QQI", source, observed, len(identity)) + identity)
        def rectangle(value):
            # Swift JSON may encode negative zero as the integer -0. Geometry
            # treats both zeros identically, so fingerprint positive zero.
            return struct.pack("<4d", *(0.0 if value[key] == 0 else value[key] for key in ("x", "y", "width", "height")))
        data += rectangle(surface["globalBounds"])
        data += struct.pack("<II", surface["pixelWidth"], surface["pixelHeight"])
        data += rectangle(surface["contentBounds"])
        data += struct.pack("<QQ", surface["geometryRevision"], size)
        data += b"bgra8-srgb\0raw\0"
        return hashlib.sha256(data).digest()
    except (TypeError, ValueError, OverflowError, KeyError, struct.error) as error:
        if isinstance(error, FrameRingError):
            raise
        raise FrameRingError("Frame metadata cannot be represented by the ring contract") from error


@dataclass(frozen=True)
class OwnedFrame:
    pixels: mx.array
    metadata: dict
    _acknowledgement: dict

    @property
    def acknowledgement(self) -> dict:
        """Available only after the owned ingestion copy and header check finish."""
        return dict(self._acknowledgement)


@dataclass(frozen=True)
class OwnedCPUFrame(OwnedFrame):
    pixels: np.ndarray


class FrameRingReader:
    def __init__(self, path: Path, *, run_id: str, ring_id: str):
        self._lock = threading.RLock()
        self._descriptor = None
        self._mapping = None
        self._seen = {}
        self.run_id, self.ring_id = _uuid(run_id), _uuid(ring_id)
        try:
            # Nonblocking open prevents a substituted FIFO/device from hanging
            # before fstat can reject it as a non-regular transport artifact.
            self._descriptor = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
            info = os.fstat(self._descriptor)
            self._file_identity = (info.st_dev, info.st_ino)
            self._validate_file(info)
            header = os.pread(self._descriptor, HEADER_BYTES, 0)
            if len(header) != HEADER_BYTES:
                raise FrameRingError("Truncated frame-ring header")
            magic, version, header_bytes, ring, run, slots, slot_header, capacity, stride, size = struct.unpack("<8sII16s16sIIQQQ", header[:80])
            if (magic != b"ASTRAR01" or version != VERSION or header_bytes != HEADER_BYTES or slot_header != SLOT_HEADER_BYTES
                or ring != self.ring_id.bytes or run != self.run_id.bytes or any(header[80:])):
                raise FrameRingError("Frame-ring identity, version or header layout does not match")
            if (not 1 <= slots <= MAXIMUM_SLOTS or not 1 <= capacity <= MAXIMUM_SLOT_BYTES
                or stride != ((SLOT_HEADER_BYTES + capacity + 63) // 64) * 64
                or size != HEADER_BYTES + slots * stride or size != info.st_size):
                raise FrameRingError("Frame-ring dimensions exceed their bounds or disagree with the file")
            self.slot_count, self.slot_capacity, self.slot_stride, self.byte_count = slots, capacity, stride, size
            self._header = header
            self._mapping = mmap.mmap(self._descriptor, size, access=mmap.ACCESS_READ)
        except BaseException:
            self.close()
            raise

    def _validate_file(self, info):
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid()
            or info.st_nlink != 1 or not HEADER_BYTES <= info.st_size <= MAXIMUM_FILE_BYTES):
            raise FrameRingError("The frame ring must be a private, bounded regular file owned by this user")

    def _current_header(self):
        if self._mapping is None or self._descriptor is None:
            raise FrameRingError("The frame-ring reader is closed")
        info = os.fstat(self._descriptor)
        self._validate_file(info)
        if (info.st_dev, info.st_ino) != self._file_identity or info.st_size != self.byte_count or self._mapping[:HEADER_BYTES] != self._header:
            raise FrameRingError("The mapped frame ring changed identity or dimensions")

    def copy_frame(self, reference: dict) -> OwnedFrame:
        """Synchronously detach one BGRA frame from its mapped lease.

        The temporary NumPy view is read-only and never escapes this method.
        mx.array owns the single ingestion copy; mx.eval completes it before
        acknowledgements become available to the caller.
        """
        return self._copy_frame(reference, cpu=False)

    def copy_cpu_frame(self, reference: dict) -> OwnedCPUFrame:
        """Detach immutable CPU BGRA for lossless rollout spooling.

        This avoids a GPU upload/readback when the next consumer needs raw
        source bytes. It shares every lease/header check with MLX ingestion.
        """
        return self._copy_frame(reference, cpu=True)

    def _copy_frame(self, reference: dict, *, cpu: bool):
        with self._lock:
            self._current_header()
            fields = {"version", "runID", "ringID", "slot", "leaseID", "sequence", "offset", "size", "metadata"}
            if type(reference) is not dict or set(reference) != fields:
                raise FrameRingError("Invalid frame lease reference")
            if (_uint(reference["version"]) != VERSION or _uuid(reference["runID"]) != self.run_id or _uuid(reference["ringID"]) != self.ring_id):
                raise FrameRingError("Frame reference belongs to another ring or run")
            slot = _uint(reference["slot"], self.slot_count - 1)
            sequence = _uint(reference["sequence"])
            lease_id = _uuid(reference["leaseID"])
            size = _uint(reference["size"], self.slot_capacity)
            offset = _uint(reference["offset"], self.byte_count)
            if (sequence == 0 or sequence <= self._seen.get(slot, 0) or size == 0
                or offset != HEADER_BYTES + slot * self.slot_stride + SLOT_HEADER_BYTES):
                raise FrameRingError("Frame reference is stale or outside its declared slot")
            metadata = deepcopy(reference["metadata"])
            digest = _metadata_fingerprint(metadata)
            if metadata["byteCount"] != size:
                raise FrameRingError("Frame size disagrees with its lease metadata")
            frame_id = _uuid(metadata["id"])
            expected = struct.pack("<IIII16s16sQQ32s32x", 2, VERSION, slot, 0, lease_id.bytes, frame_id.bytes, sequence, size, digest)
            start = offset - SLOT_HEADER_BYTES
            if self._mapping[start:offset] != expected:
                raise FrameRingError("Frame lease is stale, incomplete or has different metadata")
            surface = metadata["surface"]
            view = np.ndarray((surface["pixelHeight"], surface["pixelWidth"], 4), dtype=np.uint8, buffer=self._mapping, offset=offset)
            view.setflags(write=False)
            try:
                if cpu:
                    owned = np.frombuffer(view.tobytes(), dtype=np.uint8).reshape(view.shape)
                else:
                    owned = mx.array(view)
                    mx.eval(owned)
            finally:
                del view
            self._current_header()
            if self._mapping[start:offset] != expected:
                raise FrameRingError("Frame ownership changed while its ingestion copy was in progress")
            self._seen[slot] = sequence
            # The caller owns its request dictionary. Build the acknowledgement
            # from the exact values checked above, even if that dictionary was
            # changed by another callback while MLX completed the copy.
            acknowledgement = {"version": VERSION, "runID": str(self.run_id), "ringID": str(self.ring_id),
                               "slot": slot, "leaseID": str(lease_id), "sequence": sequence}
            result_type = OwnedCPUFrame if cpu else OwnedFrame
            return result_type(owned, metadata, acknowledgement)

    def close(self):
        with self._lock:
            if self._mapping is not None:
                self._mapping.close(); self._mapping = None
            if self._descriptor is not None:
                os.close(self._descriptor); self._descriptor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except (AttributeError, OSError, BufferError):
            pass  # Explicit close reports errors; finalizers cannot do so.
