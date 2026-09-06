"""Read sealed native recordings without changing their files or SQLite state.

This is the shared source path for dataset creation and Python evaluation tools.
Frame/index agreement and checksums are checked before pixels reach a model.
"""
from __future__ import annotations

from collections import OrderedDict
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
from typing import Iterator
import uuid

import numpy as np

MAXIMUM_FRAME_BYTES = 256 * 1024**2
MAXIMUM_EVENT_BYTES = 256 * 1024


class RecordingError(ValueError):
    pass


def _integer(value, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise RecordingError("Invalid recording integer")
    return value


def _json(data: bytes, limit=262_144):
    if not data or len(data) > limit:
        raise RecordingError("Recording metadata exceeds its supported size")
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RecordingError("Recording JSON has duplicate fields")
            result[key] = value
        return result
    def constant(_):
        raise RecordingError("Nonfinite recording JSON")
    try:
        return json.loads(data, object_pairs_hook=object_pairs, parse_constant=constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise RecordingError("Recording metadata is malformed") from error


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RecordingError("A recording artifact must be a regular file inside its package")


def _rect(value: dict) -> tuple[float, float, float, float]:
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        raise RecordingError("Invalid surface rectangle")
    fields = tuple(value[name] for name in ("x", "y", "width", "height"))
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in fields):
        raise RecordingError("Surface geometry contains nonfinite coordinates")
    x, y, width, height = fields
    if width <= 0 or height <= 0 or not math.isfinite(x + width) or not math.isfinite(y + height):
        raise RecordingError("Surface geometry is empty or overflowed")
    return fields


def validate_surface(value: dict) -> dict:
    required = {"id", "globalBounds", "pixelWidth", "pixelHeight", "contentBounds", "geometryRevision"}
    if not isinstance(value, dict) or set(value) != required or not isinstance(value["id"], str) or not 1 <= len(value["id"].encode()) <= 256:
        raise RecordingError("Invalid surface descriptor")
    width, height = _integer(value["pixelWidth"], 1, 32768), _integer(value["pixelHeight"], 1, 32768)
    _integer(value["geometryRevision"], maximum=2**64 - 1)
    _rect(value["globalBounds"])
    x, y, w, h = _rect(value["contentBounds"])
    if x < 0 or y < 0 or x + w > width or y + h > height or width * height * 4 > MAXIMUM_FRAME_BYTES:
        raise RecordingError("Surface content is outside its native pixel buffer")
    return value


def validate_frame(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"id", "eventNanos", "observedNanos", "surface", "byteCount", "pixelFormat", "codec"}:
        raise RecordingError("Invalid frame metadata")
    uuid.UUID(value["id"])
    _integer(value["eventNanos"]); _integer(value["observedNanos"])
    surface = validate_surface(value["surface"])
    size = _integer(value["byteCount"], 1, MAXIMUM_FRAME_BYTES)
    if size != surface["pixelWidth"] * surface["pixelHeight"] * 4 or value["pixelFormat"] != "bgra8-srgb" or value["codec"] not in ("raw", "lzfse"):
        raise RecordingError("Unsupported frame pixel format or codec")
    return value


def validate_event(value: dict) -> dict:
    if not isinstance(value, dict):
        raise RecordingError("Invalid raw event")
    for field in ("sequence", "eventNanos", "observedNanos"):
        _integer(value.get(field))
    if value.get("origin") not in ("physical", "agent", "reconciliation", "boundary") or value.get("kind") not in (
        "keyDown", "keyUp", "keyRepeat", "buttonDown", "buttonUp", "pointer", "scroll", "flags", "gap"
    ):
        raise RecordingError("Unknown raw input provenance or operation")
    for field in ("x", "y", "dx", "dy", "scrollX", "scrollY"):
        item = value.get(field)
        if item is not None and (type(item) not in (int, float) or not math.isfinite(item)):
            raise RecordingError("Raw input contains nonfinite arguments")
    if "keyCode" in value:
        _integer(value["keyCode"], maximum=127)
    if "button" in value:
        _integer(value["button"], maximum=31)
    if "modifiers" in value:
        _integer(value["modifiers"], maximum=2**64 - 1)
    if "isDown" in value and type(value["isDown"]) is not bool:
        raise RecordingError("Invalid physical input state")
    required = {"keyDown": ("keyCode",), "keyUp": ("keyCode",), "keyRepeat": ("keyCode",),
                "buttonDown": ("button",), "buttonUp": ("button",), "pointer": ("x", "y"),
                "scroll": ("scrollX", "scrollY"), "flags": ("modifiers",), "gap": ()}
    if any(value.get(field) is None for field in required[value["kind"]]):
        raise RecordingError("Raw input is missing its operation arguments")
    return value


class FrameDecoder:
    """Bounded OS LZFSE decoder; no third-party codec or subprocess per frame."""
    def __init__(self):
        self._library = ctypes.CDLL("/usr/lib/libcompression.dylib")
        self._decode = self._library.compression_decode_buffer
        self._decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
        self._decode.restype = ctypes.c_size_t

    def read(self, descriptor: int, offset: int) -> tuple[dict, np.ndarray]:
        _integer(offset)
        prefix = os.pread(descriptor, 20, offset)
        if len(prefix) != 20 or prefix[:8] != b"ASTRAF01":
            raise RecordingError("Frame header is truncated or has the wrong version")
        metadata_length, payload_length = struct.unpack("<IQ", prefix[8:])
        if not 1 <= metadata_length <= 65536 or not 1 <= payload_length <= MAXIMUM_FRAME_BYTES:
            raise RecordingError("Frame declares an unsupported block length")
        length = 20 + metadata_length + payload_length + 32
        if offset + length > os.fstat(descriptor).st_size:
            raise RecordingError("Frame ends inside an incomplete archive block")
        raw = os.pread(descriptor, metadata_length + payload_length + 32, offset + 20)
        if len(raw) != metadata_length + payload_length + 32:
            raise RecordingError("Frame bytes changed or became unreadable")
        metadata = validate_frame(_json(raw[:metadata_length], limit=65536))
        expected = raw[-32:]
        digest = hashlib.sha256(prefix)
        digest.update(memoryview(raw)[:-32])
        if digest.digest() != expected:
            raise RecordingError("Frame checksum does not match; its pixels cannot be used")
        payload = raw[metadata_length:-32]
        size = metadata["byteCount"]
        if metadata["codec"] == "raw":
            if len(payload) != size:
                raise RecordingError("Raw frame length does not match its geometry")
            pixels = np.frombuffer(payload, dtype=np.uint8)
        else:
            pixels = np.empty(size, dtype=np.uint8)
            count = self._decode(pixels.ctypes.data, size, ctypes.c_char_p(payload), len(payload), None, 0x801)
            if count != size:
                raise RecordingError("LZFSE frame cannot be decoded to its declared size")
        pixels = pixels.reshape(metadata["surface"]["pixelHeight"], metadata["surface"]["pixelWidth"], 4)
        pixels.setflags(write=False)
        return {"offset": offset, "length": length, "metadata": metadata, "digest": expected.hex()}, pixels


class RecordingReader:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = None
        self._connection = None
        self._shards = OrderedDict()
        self._decoder = FrameDecoder()
        try:
            if self.directory.is_symlink() or not self.directory.is_dir():
                raise RecordingError("A recording package must be a local directory")
            self._lock = os.open(self.directory / ".writer.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                fcntl.flock(self._lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RecordingError("Recording still has an active writer or recovery operation") from error
            path = self.directory / "manifest.json"
            _regular(path)
            if path.stat().st_size > 262144:
                raise RecordingError("Recording manifest is too large")
            self.manifest = _json(path.read_bytes())
            manifest = self.manifest
            if manifest.get("schemaVersion") != 1 or manifest.get("status") not in ("complete", "interrupted", "failed"):
                raise RecordingError("Only sealed recordings of a supported version can be read")
            if self.directory.stem.lower() != str(uuid.UUID(manifest["id"])):
                raise RecordingError("Recording package identity does not match its manifest")
            for field in ("frameCount", "eventCount", "storedBytes", "stoppedNanos"):
                _integer(manifest.get(field))
            for field in ("firstObservedNanos", "firstInvalidObservedNanos"):
                if field in manifest:
                    _integer(manifest[field])
            parts = manifest.get("indexPath", "index.sqlite").split("/")
            if parts != ["index.sqlite"]:
                if len(parts) != 3 or parts[0] != "Recovery" or parts[2] != "index.sqlite":
                    raise RecordingError("Recording index path escapes its package")
                uuid.UUID(parts[1])
            index = self.directory
            for component in parts:
                index = index / component
                if index.is_symlink():
                    raise RecordingError("Recording index cannot resolve through a symbolic link")
            _regular(index)
            wal = Path(str(index) + "-wal")
            if wal.exists() and wal.stat().st_size:
                raise RecordingError("Recording index has an unsealed WAL and needs native recovery")
            self._connection = sqlite3.connect(index.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA query_only=ON")
            if self._connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RecordingError("Recording index failed its SQLite integrity check")
            counts = self._connection.execute("SELECT (SELECT COUNT(*) FROM frames), (SELECT COUNT(*) FROM events)").fetchone()
            if tuple(counts) != (manifest["frameCount"], manifest["eventCount"]):
                raise RecordingError("Recording index counts do not match the sealed manifest")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._connection is not None:
            self._connection.close(); self._connection = None
        for descriptor in self._shards.values():
            os.close(descriptor)
        self._shards.clear()
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN); os.close(self._lock); self._lock = None

    @property
    def connection(self):
        if self._connection is None:
            raise RecordingError("Recording reader is closed")
        return self._connection

    def _rows(self, sql, args=()) -> Iterator[sqlite3.Row]:
        cursor = self.connection.execute(sql, args)
        try:
            while batch := cursor.fetchmany(256):
                yield from batch
        finally:
            if self._connection is not None:
                cursor.close()

    def frames(self) -> Iterator[dict]:
        for row in self._rows("SELECT * FROM frames ORDER BY observed,id"):
            yield self._frame_row(row)

    def frame_at(self, observed_nanos: int, *, surface_id: str) -> dict | None:
        _integer(observed_nanos)
        row = self.connection.execute("SELECT * FROM frames WHERE observed<=? AND json_extract(CAST(block AS TEXT),'$.metadata.surface.id')=? ORDER BY observed DESC,id DESC LIMIT 1",
                                      (observed_nanos, surface_id)).fetchone()
        return self._frame_row(row) if row is not None else None

    @staticmethod
    def _frame_row(row) -> dict:
        block = _json(row["block"])
        metadata = validate_frame(block.get("metadata"))
        if row["id"] != metadata["id"] or row["observed"] != metadata["observedNanos"] or row["source_time"] != metadata["eventNanos"]:
            raise RecordingError("Frame index identity or timestamps disagree with its block")
        if row["offset"] != block.get("offset") or row["length"] != block.get("length"):
            raise RecordingError("Frame index byte range disagrees with its block")
        _integer(row["offset"]); _integer(row["length"], 1, MAXIMUM_FRAME_BYTES + 65536 + 52)
        if not isinstance(row["shard"], str) or re.fullmatch(r"frames-[0-9]{5,}\.astraframes", row["shard"]) is None:
            raise RecordingError("Frame index contains an invalid shard path")
        return {"shard": row["shard"], "block": block}

    def pixels(self, reference: dict) -> np.ndarray:
        self.connection  # Reject use after close before opening another file.
        shard = reference["shard"]
        if not isinstance(shard, str) or re.fullmatch(r"frames-[0-9]{5,}\.astraframes", shard) is None:
            raise RecordingError("Invalid recording shard")
        descriptor = self._shards.pop(shard, None)
        if descriptor is None:
            path = self.directory / shard
            _regular(path)
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        self._shards[shard] = descriptor
        while len(self._shards) > 4:
            _, old = self._shards.popitem(last=False)
            os.close(old)
        block, pixels = self._decoder.read(descriptor, reference["block"]["offset"])
        if block != reference["block"]:
            raise RecordingError("Decoded frame disagrees with its indexed identity, geometry or checksum")
        return pixels

    def events(self, start: int = 0, end: int = 2**63 - 1, *, time_axis: str = "observed") -> Iterator[dict]:
        _integer(start); _integer(end)
        if end < start or time_axis not in ("observed", "source_time"):
            raise RecordingError("Invalid input interval")
        for row in self._rows(f"SELECT * FROM events WHERE {time_axis}>=? AND {time_axis}<? ORDER BY {time_axis},sequence", (start, end)):
            event = validate_event(_json(row["event"], limit=MAXIMUM_EVENT_BYTES))
            if (row["sequence"], row["observed"], row["source_time"]) != (event["sequence"], event["observedNanos"], event["eventNanos"]):
                raise RecordingError("Input index disagrees with its raw event identity or time")
            yield event
