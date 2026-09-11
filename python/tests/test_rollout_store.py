from dataclasses import replace
import os
from types import SimpleNamespace
import numpy as np
import pytest

from astra.learning.rollout_store import FrameSpool, MINIMUM_FREE_DISK_BYTES


def test_lossless_spool_roundtrip_evicts_cache_and_closes_without_mappings(tmp_path):
    pixels = np.arange(32 * 32 * 4, dtype=np.uint8).reshape(32, 32, 4)
    spool = FrameSpool(memory_bytes=pixels.nbytes, disk_bytes=10 * pixels.nbytes, directory=tmp_path)
    directory = spool.directory
    first = spool.append(pixels)
    second = spool.append(np.ascontiguousarray(255 - pixels))
    assert spool.cache_bytes == pixels.nbytes and spool.disk_bytes == 2 * pixels.nbytes
    np.testing.assert_array_equal(first.read(), pixels)
    np.testing.assert_array_equal(second.read(), 255 - pixels)
    assert not first.read().flags.writeable
    assert spool.peak_memory_bytes <= spool.memory_limit
    spool.seal(); spool.close(); spool.close()
    assert not directory.exists()
    with pytest.raises(ValueError, match='retired'):
        first.read()


def test_spool_detects_disk_corruption_and_changed_reference(tmp_path):
    pixels = np.full((32, 32, 4), 17, np.uint8)
    spool = FrameSpool(memory_bytes=pixels.nbytes, disk_bytes=3 * pixels.nbytes, directory=tmp_path)
    try:
        first = spool.append(pixels)
        spool.append(pixels)  # evict the first frame so its checksum covers disk bytes
        with spool.path.open('r+b') as file:
            file.write(b'\x00')
        with pytest.raises(OSError, match='checksum'):
            first.read()
        second = spool.append(pixels)
        with pytest.raises(OSError, match='reference'):
            replace(second, checksum=bytes(32)).read()
    finally:
        spool.close()


def test_spool_finishes_short_writes_and_reports_real_write_failure(tmp_path, monkeypatch):
    pixels = np.full((32, 32, 4), 41, np.uint8)
    spool = FrameSpool(memory_bytes=pixels.nbytes, disk_bytes=4 * pixels.nbytes, directory=tmp_path)
    original = os.pwrite
    def short_write(fd, data, offset):
        return original(fd, data[:71], offset)
    monkeypatch.setattr(os, 'pwrite', short_write)
    try:
        first = spool.append(pixels)
        spool.append(pixels)
        np.testing.assert_array_equal(first.read(), pixels)
        def failed_write(*_):
            raise OSError('Disk device disconnected')
        monkeypatch.setattr(os, 'pwrite', failed_write)
        with pytest.raises(OSError, match='disconnected'):
            spool.append(pixels)
    finally:
        spool.close()
    assert not list(tmp_path.iterdir())


def test_disk_free_reserve_and_capacity_are_checked_before_every_frame(tmp_path, monkeypatch):
    import astra.learning.rollout_store as storage
    pixels = np.zeros((32, 32, 4), np.uint8)
    free = MINIMUM_FREE_DISK_BYTES + 2 * pixels.nbytes
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda _: SimpleNamespace(free=free))
    spool = FrameSpool(memory_bytes=pixels.nbytes, disk_bytes=pixels.nbytes, directory=tmp_path)
    try:
        free = MINIMUM_FREE_DISK_BYTES
        with pytest.raises(OSError, match='10 GiB'):
            spool.append(pixels)
        free += 2 * pixels.nbytes
        spool.append(pixels)
        with pytest.raises(OSError, match='disk budget'):
            spool.append(pixels)
    finally:
        spool.close()
