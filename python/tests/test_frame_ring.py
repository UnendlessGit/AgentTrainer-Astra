from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import uuid

import mlx.core as mx
import numpy as np
import pytest

import astra.frame_ring as transport
from astra.frame_ring import FrameRingError, FrameRingReader


@pytest.fixture(scope="module")
def ring_fixture_executable():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["swift", "build", "--product", "AstraFixture"], cwd=root, check=True, capture_output=True, timeout=60)
    return root / ".build/debug/AstraFixture"


def _message(process):
    ready, _, _ = select.select([process.stdout], [], [], 10)
    if not ready:
        pytest.fail("Native frame fixture did not produce its next bounded reply")
    line = process.stdout.readline(4097)
    assert line.endswith(b"\n") and len(line) <= 4096
    return json.loads(line)


def _release(process, frame):
    process.stdin.write(json.dumps({"release": frame.acknowledgement}).encode() + b"\n")
    process.stdin.flush()


@pytest.fixture
def native_ring(ring_fixture_executable, tmp_path):
    process = subprocess.Popen([str(ring_fixture_executable), "--frame-ring", str(tmp_path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        first = _message(process)
        assert first["pendingCloseRefused"]
        yield process, first
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def _reader(report):
    reference = report["reference"]
    return FrameRingReader(Path(report["path"]), run_id=reference["runID"], ring_id=reference["ringID"])


def _digest(frame):
    return hashlib.sha256(np.asarray(frame.pixels).tobytes()).hexdigest()


def test_native_mapped_frame_is_owned_before_release_and_survives_slot_reuse(native_ring):
    process, first = native_ring
    with _reader(first) as reader:
        owned = reader.copy_frame(first["reference"])
        assert owned.pixels.dtype == mx.uint8 and owned.pixels.shape == (5, 7, 4)
        assert owned.metadata["codec"] == "raw"
        assert owned.metadata["observedNanos"] == 2**63 + 12
        assert owned.metadata["surface"]["id"] == "display:α"
        assert _digest(owned) == first["pixelSHA256"]
        with pytest.raises(FrameRingError, match="stale"):
            reader.copy_frame(first["reference"])
        _release(process, owned)
        second = _message(process)
        assert second["reference"]["slot"] == first["reference"]["slot"]
        assert second["reference"]["leaseID"] != first["reference"]["leaseID"]
        assert second["pixelSHA256"] != first["pixelSHA256"]
        assert _digest(owned) == first["pixelSHA256"]  # No mapped NumPy alias survives.
        with _reader(second) as fresh_reader:
            with pytest.raises(FrameRingError, match="lease"):
                fresh_reader.copy_frame(first["reference"])
        latest = reader.copy_frame(second["reference"])
        assert _digest(latest) == second["pixelSHA256"]
        _release(process, latest)
        assert _message(process) == {"closed": True, "pathExists": False}
        assert process.wait(timeout=10) == 0
        # Writer unmap/unlink does not truncate a still-live consumer mapping.
        offset, size = second["reference"]["offset"], second["reference"]["size"]
        assert hashlib.sha256(reader._mapping[offset:offset + size]).hexdigest() == second["pixelSHA256"]
    assert _digest(owned) == first["pixelSHA256"] and _digest(latest) == second["pixelSHA256"]
    with pytest.raises(FrameRingError, match="closed"):
        reader.copy_frame(second["reference"])


def test_reference_identity_geometry_and_ranges_are_verified_before_copy(native_ring):
    _, report = native_ring
    with _reader(report) as reader:
        original = report["reference"]
        invalid = []
        for key, value in (("runID", str(uuid.uuid4())), ("ringID", str(uuid.uuid4())), ("leaseID", str(uuid.uuid4())),
                           ("offset", original["offset"] + 1), ("size", 513), ("slot", 1), ("sequence", True)):
            changed = deepcopy(original); changed[key] = value; invalid.append(changed)
        for mutation in (lambda value: value["surface"]["globalBounds"].update(x=-1900),
                         lambda value: value["surface"].update(geometryRevision=18),
                         lambda value: value.update(observedNanos=value["observedNanos"] + 1),
                         lambda value: value.update(codec="lzfse")):
            changed = deepcopy(original); mutation(changed["metadata"]); invalid.append(changed)
        for reference in invalid:
            with pytest.raises(FrameRingError):
                reader.copy_frame(reference)
        assert _digest(reader.copy_frame(original)) == report["pixelSHA256"]


def test_header_change_during_ingestion_cannot_produce_an_acknowledgement(native_ring, monkeypatch):
    _, report = native_ring
    original_eval = transport.mx.eval
    def changed_during_copy(*arrays):
        original_eval(*arrays)
        with Path(report["path"]).open("r+b") as stream:
            stream.seek(report["reference"]["offset"] - transport.SLOT_HEADER_BYTES + 16)
            stream.write(uuid.uuid4().bytes)
    with _reader(report) as reader:
        monkeypatch.setattr(transport.mx, "eval", changed_during_copy)
        with pytest.raises(FrameRingError, match="ownership changed"):
            reader.copy_frame(report["reference"])
        assert not reader._seen


def test_acknowledgement_retains_the_validated_identity_if_the_request_changes(native_ring, monkeypatch):
    process, report = native_ring
    reference = deepcopy(report["reference"])
    original_eval = transport.mx.eval
    def request_changed_during_copy(*arrays):
        original_eval(*arrays)
        reference["leaseID"] = str(uuid.uuid4())
        reference["sequence"] += 100
        reference["metadata"]["surface"]["globalBounds"]["x"] = 42
    with _reader(report) as reader:
        with monkeypatch.context() as changes:
            changes.setattr(transport.mx, "eval", request_changed_during_copy)
            owned = reader.copy_frame(reference)
        assert owned.metadata == report["reference"]["metadata"]
        assert uuid.UUID(owned.acknowledgement["leaseID"]) == uuid.UUID(report["reference"]["leaseID"])
        assert owned.acknowledgement["sequence"] == report["reference"]["sequence"]
        _release(process, owned)
        latest = reader.copy_frame(_message(process)["reference"])
        _release(process, latest)
        assert _message(process) == {"closed": True, "pathExists": False}
        assert process.wait(timeout=10) == 0


def test_private_file_checks_precede_mapping_and_references_do_not_authorize_writes(native_ring, tmp_path):
    _, report = native_ring
    reference, path = report["reference"], Path(report["path"])
    wrong = dict(report); wrong["reference"] = {**reference, "ringID": str(uuid.uuid4())}
    with pytest.raises(FrameRingError, match="identity"):
        _reader(wrong)
    linked = tmp_path / "linked"
    linked.symlink_to(path)
    with pytest.raises(OSError):
        FrameRingReader(linked, run_id=reference["runID"], ring_id=reference["ringID"])
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(FrameRingError, match="regular file"):
        FrameRingReader(fifo, run_id=reference["runID"], ring_id=reference["ringID"])
    os.chmod(path, 0o644)
    with pytest.raises(FrameRingError, match="private"):
        _reader(report)
    os.chmod(path, 0o600)
    with _reader(report) as reader:
        with pytest.raises(TypeError):
            reader._mapping[reference["offset"]] = 1


def test_truncated_mapping_is_rejected_before_touching_missing_pages(native_ring):
    _, report = native_ring
    with _reader(report) as reader:
        with Path(report["path"]).open("r+b") as stream:
            stream.truncate(64)
        with pytest.raises(FrameRingError, match="bounded regular file"):
            reader.copy_frame(report["reference"])


def test_native_cpu_frame_is_immutable_and_survives_acknowledged_slot_reuse(native_ring):
    from astra.environments.external import FrameRingResolver
    process, first = native_ring
    with _reader(first) as reader:
        resolver = FrameRingResolver({str(reader.ring_id): reader})
        resolved = resolver(first['reference'], first['reference']['metadata'])
        assert resolved.pixels.dtype == np.uint8 and resolved.pixels.shape == (5, 7, 4)
        assert not resolved.pixels.flags.writeable
        with pytest.raises(ValueError): resolved.pixels.flags.writeable = True
        assert hashlib.sha256(resolved.pixels.tobytes()).hexdigest() == first['pixelSHA256']
        process.stdin.write(json.dumps({'release': resolved.acknowledgement}).encode() + b'\n')
        process.stdin.flush()
        second = _message(process)
        assert second['reference']['slot'] == first['reference']['slot']
        assert hashlib.sha256(resolved.pixels.tobytes()).hexdigest() == first['pixelSHA256']
        with pytest.raises(transport.FrameRingError, match='stale'):
            reader.copy_cpu_frame(first['reference'])
        latest = reader.copy_cpu_frame(second['reference'])
        assert _digest(latest) == second['pixelSHA256']
        _release(process, latest)
        assert _message(process) == {'closed': True, 'pathExists': False}
        assert process.wait(timeout=10) == 0
    assert hashlib.sha256(resolved.pixels.tobytes()).hexdigest() == first['pixelSHA256']


def test_cpu_copy_rejects_lease_mutation_before_acknowledgement(native_ring, monkeypatch):
    _, report = native_ring
    original = transport.np.frombuffer
    def replaced_during_copy(*args, **kwargs):
        result = original(*args, **kwargs)
        with Path(report['path']).open('r+b') as stream:
            stream.seek(report['reference']['offset'] - transport.SLOT_HEADER_BYTES + 16)
            stream.write(uuid.uuid4().bytes)
        return result
    with _reader(report) as reader:
        monkeypatch.setattr(transport.np, 'frombuffer', replaced_during_copy)
        with pytest.raises(FrameRingError, match='ownership changed'):
            reader.copy_cpu_frame(report['reference'])
        assert not reader._seen
