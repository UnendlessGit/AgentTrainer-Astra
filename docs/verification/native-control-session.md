# Shared native control session and reset binding

`NativeControlSession` owns one fresh protected helper for one physical policy or reset episode. It uses the existing `ComputeProcess` transport and paired guardian. The reusable component is ready for integration into native desktop learning and ordinary inference; those coordinator integrations and installed-app physical effects remain separate evidence gates.

## Public boundary

- `NativeControlConfiguration` fixes run identity, current scope, capabilities, packet capacity, recovery directory and the initial policy packet counter. The helper must advertise support for a nonzero origin and acknowledge the exact counter. Reset helpers start at zero; a collecting actor can retain its produced-packet counter across physical episodes.
- `start()` arms a fresh helper and validates its guardian/ledger handshake. `submit(packet)` waits only for admission, returning a `NativeControlSubmission` with its own awaitable terminal receipt. `execute(packet)` awaits both stages for reset steps. Normal submissions are ordered, bounded, unique and monotonic.
- `observation(afterSequence:)` returns the helper's atomic causal state/history snapshot. Consumers still enforce their observation freshness and coverage requirements.
- `requestStop()` closes local admission immediately. `disarm()` acknowledges settled disarm but keeps this helper alive. A produced packet known never forwarded can then use `rejectLatePacket` to obtain its actual helper rejection. Reserved work that observes closed admission before its transport attempt waits for acknowledged disarm. A concurrent transport attempt remains in flight and is forwarded exactly once. Already-forwarded uncertain packets are never resent.
- `shutdown()` drains arm/disarm, reserved submissions and in-flight heartbeat requests, joins transport and callbacks, then waits for cleanup and the matching guardian's exit. The shared `NativeControlOwner` blocks successors throughout this lifetime. Cancellation never tears down an uncancelled control request as a shortcut to completion.

Each callback includes an immutable session UUID as well as the run UUID. Raw receipt offers occur synchronously before completing terminal waiters; audit/collector backpressure becomes a real failure. Admission/terminal identities and outcomes must agree in either arrival order, and executed receipts require exact per-command scheduling/posting evidence. Unexpected ordinary non-admission stops further admission without rolling back the actor's produced sequence or erasing the rejected receipt. Missing receipts fail the session.

A terminal ledger is cleanup proof, but a known live guardian may still retain its shared desktop lease. The session waits for that matching process to exit before releasing the host owner gate. Partial armed snapshots are never terminal proof. Abandoning the session schedules independently owned joined cleanup rather than dropping its helper. An unconfirmed dead-owner result keeps the gate closed until an explicit human acknowledgement; acknowledgement does not rewrite the recorded result. No automatic timeout upgrades cleanup or kills the control helper.

## Reset binding

`NativeResetDriver` implements `ResetControlDriver` using this session. It receives an already-owned capture source, an owned observation closure, a prior-actor/control join witness and a scope verifier. Production scope verification checks original process launch, focus, window/display identity and geometry; the driver never starts capture or asks for privacy access. A serialized scope watcher keeps bounded health evidence, while observations verify the source before and after awaiting producer data. Returned coverage must match every bound surface and preserve the producer's original evidence.

Manual reset with empty action capabilities creates no helper. Every authored attempt creates a fresh session, even when retrying the same reset ID. Actions use the reset ID, distinct from the upcoming policy episode. Release joins delayed preparation, observation work, the scope watcher and the control session. Foreign releases cannot borrow an in-progress proof. Repeated release returns the original result, and a known scope/control failure cannot disappear merely because the source later looks unchanged. After successful release, the driver permits the final owned readiness observation; it fabricates neither readings nor readiness.

## Verification

Permission-free checks cover both in-memory fault injection and real subprocess transport:

- Admission-before-terminal and terminal-before-admission, persistent nonzero packet origins, wrong arm counters, unexpected rejection, inconsistent terminal evidence, missing terminal receipts and a throwing required terminal-audit callback.
- Stop during an unresolved arm; a reserved submission and caller cancellation racing disarm; exactly-once forwarding; heartbeat request draining; same-helper post-disarm rejection; successor gating and old callback isolation.
- An actual `ComputeProcess` Python child with delayed heartbeat, actual stdio rejection envelopes, joined process exit and no cancellation-induced SIGTERM. This child cannot post input.
- A real child holding a fixture flock, named by the mapped recovery ledger: even after the ledger settles and command runtime joins, the host gate remains closed until the matching child exits and releases the lease. The fixture does not impersonate permission or recovery posting.
- Manual reset without any helper; fresh helpers for repeated reset contexts; prior-owner rejection; stale/future/changed scope; foreign source coverage; cancellation during delayed preparation; joining an observation that ignores cancellation; foreign/repeated release semantics.

An independent peer review found the terminal-audit callback/promise lifetime defect; it was fixed and covered. The subsequent control and reset-driver read-only review found no further blocker.

The first expanded focus passed 19 reported tests (22 parameterized cases) on September 12. After the machine reset, `.local/native-control-reset-restored2.log` passed 52 reported tests (66 cases) on Xcode 27 / Swift 6.4, without warnings. This includes the formerly pending scope-failure, deadline, missing-receipt, abandonment and blocked-heartbeat regressions. `.local/inference-native-session-final.log` subsequently passed 20 coordinator tests (34 cases) after ordinary Run adopted the shared session. Further pre-arm intervention and closing-count refinements receive their own regression run before the next integrated build.

These tests do not inspect a user window, grant privacy access, change Secure Input, post a Core Graphics event, or establish installed-bundle input attribution. Existing guardian virtual-effect fault tests remain the evidence for independent crash cleanup. Simultaneous loss of executor and guardian or power loss cannot establish actual OS release through this design.
