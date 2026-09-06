# Native compute transport

`ComputeProcess` owns a bounded inherited-pipe connection, validates the runtime handshake and monotonically ordered sender sequence, correlates requests by UUID, and separates job events from acknowledgements. Protocol state uses one serial queue; blocking pipe I/O runs outside the UI thread. Invalid frames, unknown replies, timeout and launch failures resolve outstanding requests and close the child.

Three native integration tests cover concurrent correlated requests, graceful shutdown, missing executables, wrong handshakes, malformed output and a nonresponsive child. They run Python transport fixtures with no capture or input access. All40 integrated Swift tests passed on2026-09-06.

An initial implementation used Foundation's `read(upToCount:)`, which held the short handshake pending on a live pipe. The test failed at startup. Reading available pipe bytes with POSIX `read` fixed the issue; the timeout regression remains enabled.

The client does not yet implement the shared-memory image ring, job catalog, process-independent input executor or full app quit/join coordination. Current development app builds remain ad-hoc signed; replacing a bundle can invalidate macOS privacy grants tied to the previous code identity. Live capture/control qualification is deferred until a stable installed build can be granted access by the user.
