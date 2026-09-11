# Native runtime transport

`ComputeProcess` owns one bounded inherited-pipe connection and one child lifetime. The caller selects the expected compute, actor or control role and permitted asynchronous events. Sender sequences and request/run identities are validated. An explicit `acceptingError` request returns the full rejection envelope, including leased-frame acknowledgements; transport faults always throw.

Shutdown is coalesced and waits for actual process exit, stdout/stderr drain and writer closure. Graceful acknowledgement alone does not prove termination. A grace timeout sends TERM, followed by KILL if the owned child remains alive. Nonblocking pipe I/O bounds cancellation latency even when the child never reads. Writer closure occurs on its owning queue, preventing concurrent close and descriptor reuse. A terminated child cannot leave the join waiting indefinitely on inherited pipe handles held by a descendant.

Eight native integration tests pass on 2026-09-08: concurrent correlation, wrong handshake/missing executable, malformed output and timeout, exact UInt64 rejection data, run mismatch, finalizer join and concurrent stops, ignored TERM escalation, a full input pipe during cancellation, and closed admission before launch. Fixtures use no capture or input access. The delayed-finalizer test proves shutdown waits beyond the acknowledgement; process IDs are absent after each lifecycle qualification.

A control helper must first confirm `cleanupSettled` before its owner calls generic forced shutdown. If owned key/button releases remain pending, the UI retains the cleanup owner and reports the condition. Process-exit proof is necessary for frame-ring retirement; it does not prove successful input cleanup.

The original Foundation `read(upToCount:)` held a short handshake pending on a live pipe. POSIX available-byte reading remains in use. Final installed capture/control and privacy attribution are separate release gates; these tests do not grant or validate TCC permissions.
