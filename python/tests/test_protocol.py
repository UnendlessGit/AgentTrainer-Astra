import io
import json
import pytest
from astra.protocol import MAX_MESSAGE_BYTES, MAX_UINT64, Message, ProtocolError, messages


def test_large_integer_timestamps_are_exact():
    message = Message("heartbeat", MAX_UINT64, {"clock": MAX_UINT64 - 17})
    assert Message.decode(message.encode()) == message


@pytest.mark.parametrize("sequence", [True, -1, 0.5, MAX_UINT64 + 1])
def test_sender_sequence_requires_uint64(sequence):
    with pytest.raises(ProtocolError):
        Message("execute", sequence, {}).encode()


def test_protocol_stops_at_corruption_without_interpreting_suffix():
    stream = io.BytesIO(b"broken\n" + Message("execute", 0, {}).encode())
    iterator = messages(stream)
    with pytest.raises(ProtocolError):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_nan_bool_version_and_oversized_frames_are_rejected():
    raw = {"version": 1, "kind": "execute", "sequence": 0, "payload": {"x": float("nan")}}
    with pytest.raises(ProtocolError):
        Message.decode(json.dumps(raw).encode() + b"\n")
    with pytest.raises(ProtocolError):
        Message("ping", 0, {}, version=True).encode()
    with pytest.raises(ProtocolError):
        next(messages(io.BytesIO(b"x" * (MAX_MESSAGE_BYTES + 1))))


def test_multiple_messages_and_truncated_final_message():
    first, second = Message("ping", 0, {}), Message("shutdown", 1, {})
    assert list(messages(io.BytesIO(first.encode() + second.encode()))) == [first, second]
    with pytest.raises(ProtocolError):
        list(messages(io.BytesIO(first.encode()[:-1])))
