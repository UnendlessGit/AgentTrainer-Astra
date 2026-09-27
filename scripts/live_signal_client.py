#!/usr/bin/env python3
"""Minimal local state telemetry client; Python standard library only.

Example (copy the port/session/signal IDs from Astra's local source panel):
    python3 scripts/live_signal_client.py --port 49152 --session SESSION_UUID --signal SIGNAL_UUID --number 42

The session token is prompted without echo. It is never written to a file or
printed. The CLI repeats one test value; a real adapter should call publish()
with a fresh measurement from its application. No external clocks are sent.
"""
from __future__ import annotations

import argparse
import getpass
import json
import math
import socket
import time
import uuid

MAXIMUM_MESSAGE = 256 * 1024


class LocalStateClient:
    def __init__(self, *, port: int, session_id: str, token: str):
        if not 1 <= port <= 65535 or len(token) != 64:
            raise ValueError('Use the current local source port and session token from Astra')
        self.port, self.session_id, self.token = port, str(uuid.UUID(session_id)), token
        self.connection = None
        self.input = bytearray()

    def close(self):
        if self.connection is not None:
            self.connection.close(); self.connection = None
        self.input.clear()

    def request(self, operation: str, **fields):
        if self.connection is None:
            self.connection = socket.create_connection(('127.0.0.1', self.port), timeout=3)
            self.connection.settimeout(3)
        data = json.dumps(dict(version=1, op=operation, token=self.token, sessionID=self.session_id, **fields),
                          separators=(',', ':'), allow_nan=False).encode() + b'\n'
        if len(data) > MAXIMUM_MESSAGE:
            raise ValueError('A state update exceeds the local message limit')
        try:
            self.connection.sendall(data)
            while b'\n' not in self.input:
                chunk = self.connection.recv(16384)
                if not chunk: raise ConnectionError('Astra closed the local source connection')
                self.input.extend(chunk)
                if len(self.input) > MAXIMUM_MESSAGE:
                    raise ConnectionError('Astra returned an oversized local response')
            line, _, rest = self.input.partition(b'\n'); self.input = bytearray(rest)
            response = json.loads(line)
            if response.get('ok') is not True:
                raise RuntimeError(response.get('message', 'Local source rejected the update'))
            return response
        except (OSError, ConnectionError, ValueError):
            self.close(); raise

    def publish(self, values: dict[str, float | str | bool | None], *, confidence: float = 1):
        """Read the current binding every time, including after reset/reconnect.

        Never retry an uncertain sequence blindly: the next call queries Astra's
        actual nextSequence and publishes a new current measurement instead.
        """
        state = self.request('binding.get')
        binding = state['binding']
        if binding is None: return False
        allowed = {str(uuid.UUID(signal['id'])) for signal in binding['signals']}
        updates = []
        for identifier, value in values.items():
            identifier = str(uuid.UUID(identifier))
            if identifier not in allowed:
                raise ValueError('A signal is not part of the currently announced reward definition')
            updates.append(dict(signalID=identifier, value=value, confidence=confidence))
        self.request('values.put', bindingID=binding['bindingID'], episodeID=binding['episodeID'],
                     sequence=state['nextSequence'], values=updates)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--signal', required=True)
    parser.add_argument('--interval', type=float, default=.1, help='Test sender interval in seconds; keep below the signal maximum age')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--number', type=float)
    group.add_argument('--text')
    group.add_argument('--flag', choices=['true', 'false'])
    group.add_argument('--unknown', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or not .01 <= args.interval <= 10:
        parser.error('Use a sending interval from .01 to 10 seconds')
    value = args.number if args.number is not None else args.text if args.text is not None else (args.flag == 'true') if args.flag else None
    if type(value) is float and not math.isfinite(value): parser.error('Numeric values must be finite')
    client = LocalStateClient(port=args.port, session_id=args.session, token=getpass.getpass('Astra session token (hidden): '))
    print('Local test telemetry started. This repeats the supplied value; Ctrl-C stops it.')
    try:
        while True:
            try:
                client.publish({args.signal: value})
            except (OSError, ConnectionError, RuntimeError) as error:
                print(type(error).__name__ + ': waiting for the current local binding/connection')
                client.close()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == '__main__':
    main()
