import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import numpy as np
import pytest

from astra.recordings import RecordingError, RecordingReader


@pytest.fixture(scope="module")
def native_recording(tmp_path_factory):
    parent = tmp_path_factory.mktemp("native-recording")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["swift", "run", "AstraFixture", str(parent)], cwd=root, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_native_lzfse_and_raw_frames_decode_exactly_in_python(native_recording):
    with RecordingReader(Path(native_recording["directory"])) as reader:
        frames = list(reader.frames())
        assert len(frames) == 3
        assert {item["codec"] for item in native_recording["frames"]} == {"raw", "lzfse"}
        for reference, expected in zip(frames, native_recording["frames"]):
            pixels = reader.pixels(reference)
            assert pixels.dtype == np.uint8
            assert not pixels.flags.writeable
            assert reference["block"]["metadata"]["id"] == expected["id"]
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == expected["pixelSHA256"]
        assert reader.frame_at(1_001_999_999, surface_id="surface:0") is None
        assert reader.frame_at(1_002_000_000, surface_id="surface:0")["block"]["metadata"]["id"] == native_recording["frames"][0]["id"]
        assert reader.frame_at(1_036_000_000, surface_id="surface:0")["block"]["metadata"]["id"] == native_recording["frames"][1]["id"]
        assert reader.frame_at(1_100_000_000, surface_id="other") is None


def test_input_availability_and_original_late_source_time_remain_distinct(native_recording):
    with RecordingReader(Path(native_recording["directory"])) as reader:
        observed = list(reader.events())
        source = list(reader.events(time_axis="source_time"))
        assert len(observed) == native_recording["eventCount"]
        assert [event["sequence"] for event in observed] == [0, 1, 2, 3, 4, 5]
        assert [event["sequence"] for event in source] == [0, 1, 2, 4, 3, 5]
        assert [event["kind"] for event in reader.events(1_008_000_000, 1_012_000_000)] == ["keyDown"]
        assert observed[-1]["rawPlatformData"] == "AAEC/w=="
        assert observed[-1]["scrollY"] == -1.25


def test_corrupt_archive_and_mismatching_index_never_reach_a_model(native_recording, tmp_path):
    source = Path(native_recording["directory"])
    destination = tmp_path / source.name
    shutil.copytree(source, destination)
    with RecordingReader(destination) as reader:
        first = next(reader.frames())
        path = destination / first["shard"]
        with path.open("r+b") as stream:
            stream.seek(first["block"]["length"] - 1)
            byte = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([byte[0] ^ 1]))
        with pytest.raises(RecordingError, match="checksum"):
            reader.pixels(first)
    database = sqlite3.connect(destination / "index.sqlite")
    database.execute("UPDATE frames SET observed=observed+1")
    database.commit(); database.close()
    with RecordingReader(destination) as reader:
        with pytest.raises(RecordingError, match="timestamps"):
            list(reader.frames())


def test_readers_refuse_live_writers_and_cannot_follow_linked_shards(native_recording, tmp_path):
    source = Path(native_recording["directory"])
    destination = tmp_path / source.name
    shutil.copytree(source, destination)
    with (destination / ".writer.lock").open("r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RecordingError, match="active writer"):
            RecordingReader(destination)
        fcntl.flock(lock, fcntl.LOCK_UN)
    shard = next(destination.glob("*.astraframes"))
    shard.unlink()
    shard.symlink_to(source / shard.name)
    with RecordingReader(destination) as reader:
        with pytest.raises(RecordingError, match="regular file"):
            reader.pixels(next(reader.frames()))
    with pytest.raises(RecordingError, match="closed"):
        list(reader.events())
