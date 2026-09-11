"""Lossless temporary rollout frames with bounded RAM and disk ownership.

This is an internal, ephemeral store, not an importable recording format. Frame
references carry their exact shape, byte range and checksum in immutable Python
metadata. No executable deserialization or sparse/mapped writable file is used.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

MINIMUM_FREE_DISK_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class StoredFrame:
    owner: FrameSpool
    offset: int
    nbytes: int
    shape: tuple[int, int, int]
    checksum: bytes

    def __post_init__(self):
        if (type(self.offset) is not int or self.offset < 0 or type(self.nbytes) is not int
                or not 0 < self.nbytes <= 256 * 1024**2 or type(self.shape) is not tuple
                or len(self.shape) != 3 or self.shape[2] != 4
                or any(type(value) is not int or not 0 < value <= 32768 for value in self.shape)
                or math.prod(self.shape) != self.nbytes or type(self.checksum) is not bytes
                or len(self.checksum) != 32):
            raise ValueError('Invalid stored rollout frame shape, extent or checksum')

    def read(self) -> np.ndarray:
        return self.owner.read(self)


class FrameSpool:
    """One private file, bounded raw-frame cache, explicit idempotent cleanup.

    The RAM budget counts cached BGRA bytes and retained serialized metadata;
    transient environment/preprocessing tensors are separately admitted by the
    learner. A frame larger than the budget is rejected before writing it.
    """
    def __init__(self, *, memory_bytes: int, disk_bytes: int, directory: Path | None = None):
        if type(memory_bytes) is not int or memory_bytes < 1 or type(disk_bytes) is not int or disk_bytes < 1:
            raise ValueError('Rollout storage budgets must be positive integers')
        self.memory_limit, self.disk_limit = memory_bytes, disk_bytes
        self._temporary = tempfile.TemporaryDirectory(prefix='.astra-rollout-', dir=directory)
        self.directory = Path(self._temporary.name)
        self.path = self.directory / 'frames.bgra'
        self._descriptor = None
        self._cache = OrderedDict()
        self.cache_bytes = self.metadata_bytes = self.disk_bytes = self.peak_memory_bytes = 0
        self.maximum_frame_bytes = 0
        try:
            self._check_disk(0)
            self._descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except BaseException as failure:
            try:
                self.close()
            except BaseException as cleanup:
                failure.add_note(f'Rollout cleanup also failed ({type(cleanup).__name__}).')
            raise

    @property
    def closed(self):
        return self._descriptor is None

    def _check_open(self):
        if self.closed:
            raise ValueError('This rollout has been retired; its temporary frames are closed')

    def _check_disk(self, incoming):
        if self.disk_bytes + incoming > self.disk_limit:
            raise OSError('Complete-episode rollout exceeds its configured disk budget')
        if shutil.disk_usage(self.directory).free - incoming < MINIMUM_FREE_DISK_BYTES:
            raise OSError('Rollout storage needs at least 10 GiB of free disk space after this write')

    def _evict(self, incoming=0):
        while self._cache and self.cache_bytes + self.metadata_bytes + incoming > self.memory_limit:
            _, frame = self._cache.popitem(last=False)
            self.cache_bytes -= frame.nbytes

    def reserve_metadata(self, count: int):
        self._check_open()
        if type(count) is not int or count < 0:
            raise ValueError('Invalid retained rollout metadata size')
        if self.metadata_bytes + count + self.maximum_frame_bytes > self.memory_limit:
            raise MemoryError('Rollout metadata exceeds its configured RAM budget')
        self._evict(count)
        self.metadata_bytes += count
        self.peak_memory_bytes = max(self.peak_memory_bytes, self.cache_bytes + self.metadata_bytes)

    def _remember(self, offset, pixels):
        self._evict(pixels.nbytes)
        if self.cache_bytes + self.metadata_bytes + pixels.nbytes <= self.memory_limit:
            self._cache[offset] = pixels
            self.cache_bytes += pixels.nbytes
            self.peak_memory_bytes = max(self.peak_memory_bytes, self.cache_bytes + self.metadata_bytes)

    def append(self, pixels: np.ndarray) -> StoredFrame:
        self._check_open()
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3
                or pixels.shape[2] != 4 or not pixels.flags.c_contiguous
                or not all(0 < value <= 32768 for value in pixels.shape)):
            raise ValueError('Rollout frames must be compact uint8 BGRA')
        if pixels.nbytes > min(self.memory_limit, 256 * 1024**2):
            raise MemoryError('One rollout frame exceeds its configured RAM budget')
        if self.metadata_bytes + pixels.nbytes > self.memory_limit:
            raise MemoryError('Rollout RAM budget cannot retain metadata and one working frame')
        self._evict(pixels.nbytes)
        # Immutable byte backing prevents a caller changing a cached frame after
        # its checksum was recorded. The write never retains mutable renderer data.
        owned = np.frombuffer(pixels.tobytes(), dtype=np.uint8).reshape(pixels.shape)
        data = memoryview(owned).cast('B')
        self._check_disk(len(data))
        offset = self.disk_bytes
        written = 0
        while written < len(data):
            try:
                count = os.pwrite(self._descriptor, data[written:], offset + written)
            except InterruptedError:
                continue
            if count <= 0:
                raise OSError('Rollout frame write made no progress')
            written += count
        self.disk_bytes += written
        self.maximum_frame_bytes = max(self.maximum_frame_bytes, written)
        self._remember(offset, owned)
        return StoredFrame(self, offset, written, tuple(pixels.shape), hashlib.sha256(data).digest())

    def read(self, frame: StoredFrame) -> np.ndarray:
        self._check_open()
        if frame.owner is not self or frame.offset + frame.nbytes > self.disk_bytes:
            raise ValueError('Invalid rollout frame reference')
        if frame.offset in self._cache:
            result = self._cache.pop(frame.offset)
            self._cache[frame.offset] = result
            if result.shape != frame.shape or hashlib.sha256(memoryview(result).cast('B')).digest() != frame.checksum:
                raise OSError('Rollout frame reference does not match cached bytes')
            return result
        self._evict(frame.nbytes)
        pieces, count = [], 0
        while count < frame.nbytes:
            try:
                data = os.pread(self._descriptor, frame.nbytes - count, frame.offset + count)
            except InterruptedError:
                continue
            if not data:
                raise OSError('Rollout frame is truncated')
            pieces.append(data)
            count += len(data)
        data = pieces[0] if len(pieces) == 1 else b''.join(pieces)
        if hashlib.sha256(data).digest() != frame.checksum:
            raise OSError('Rollout frame checksum does not match its recorded bytes')
        result = np.frombuffer(data, dtype=np.uint8).reshape(frame.shape)
        self._remember(frame.offset, result)
        return result

    def seal(self):
        self._check_open()
        os.fsync(self._descriptor)

    def close(self):
        descriptor, self._descriptor = self._descriptor, None
        self._cache.clear()
        self.cache_bytes = 0
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            self._temporary.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Explicit lifecycle calls report errors; finalizers cannot.
