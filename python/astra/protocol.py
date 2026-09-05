"""Bounded process transport shared with AstraCore's WireMessage."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, BinaryIO, Iterator
from uuid import UUID

VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576
MAX_UINT64 = (1 << 64) - 1


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _reject_constant(_: str) -> None:
    raise ProtocolError("protocol.nonFinite", "Nonfinite JSON numbers are not allowed.")


@dataclass(frozen=True)
class Message:
    kind: str
    sequence: int
    payload: dict[str, Any]
    request_id: str | None = None
    run_id: str | None = None
    version: int = VERSION

    def validate(self) -> Message:
        if type(self.version) is not int or self.version != VERSION:
            raise ProtocolError("protocol.version", "Incompatible protocol version.")
        if not isinstance(self.kind, str) or not re.fullmatch(r"[A-Za-z0-9.]{1,80}", self.kind):
            raise ProtocolError("protocol.envelope", "Invalid message kind.")
        if type(self.sequence) is not int or not 0 <= self.sequence <= MAX_UINT64:
            raise ProtocolError("protocol.sequence", "Invalid sender sequence.")
        if not isinstance(self.payload, dict):
            raise ProtocolError("protocol.envelope", "Payload must be an object.")
        for identifier in (self.request_id, self.run_id):
            if identifier is not None:
                if not isinstance(identifier, str):
                    raise ProtocolError("protocol.identifier", "Invalid message identifier.")
                try:
                    UUID(identifier)
                except ValueError as error:
                    raise ProtocolError("protocol.identifier", "Invalid message identifier.") from error
        return self

    def encode(self) -> bytes:
        self.validate()
        value = {"version": self.version, "kind": self.kind, "sequence": self.sequence, "payload": self.payload}
        if self.request_id is not None:
            value["requestID"] = self.request_id
        if self.run_id is not None:
            value["runID"] = self.run_id
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as error:
            raise ProtocolError("protocol.invalidJSON", "Payload cannot be encoded as finite JSON.") from error
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise ProtocolError("protocol.tooLarge", "Message exceeds the transport limit.")
        return encoded

    @classmethod
    def decode(cls, data: bytes) -> Message:
        if not data or len(data) > MAX_MESSAGE_BYTES or not data.endswith(b"\n"):
            raise ProtocolError("protocol.framing", "Invalid or incomplete protocol frame.")
        try:
            value = json.loads(data, parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise ProtocolError("protocol.envelope", "Message must be an object.")
            return cls(kind=value["kind"], sequence=value["sequence"], payload=value["payload"],
                       request_id=value.get("requestID"), run_id=value.get("runID"),
                       version=value["version"]).validate()
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError("protocol.invalidJSON", "Malformed protocol message.") from error


def messages(stream: BinaryIO) -> Iterator[Message]:
    while True:
        data = stream.readline(MAX_MESSAGE_BYTES + 1)
        if not data:
            return
        yield Message.decode(data)


class Sender:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.sequence = 0

    def send(self, kind: str, payload: dict[str, Any], *, request: Message | None = None) -> None:
        message = Message(kind, self.sequence, payload,
                          request_id=request.request_id if request else None,
                          run_id=request.run_id if request else None)
        encoded = message.encode()
        self.stream.write(encoded)
        self.stream.flush()
        self.sequence += 1
