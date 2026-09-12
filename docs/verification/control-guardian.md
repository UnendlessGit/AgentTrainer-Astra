# Paired guardian verification

The guardian uses the actual shared-memory C shim, inherited desktop flock, native process spawning, and kernel exit watch. A separate `AstraRecoveryFixture` executable supplies file-backed input effects and physical-state fixtures. It is a test target and is not copied into the distributable app. No Core Graphics event is posted and no privacy access is requested in this verification.

## Fault campaign

Targeted tests cover:

- Executor SIGKILL after reserving a down, after posting it, while posting remains in flight, and after an up's effect but before its completion bookkeeping.
- Guardian SIGKILL, causing the live executor to stop and settle locally.
- Failed release retries that retain the surviving guardian's lease, including preservation of an overlapping physical hold.
- Unavailable physical-state verification retains ownership instead of inventing a physical transfer or sending a blind release.
- Active Secure Input or stale keyboard-trust verification invalidates native physical-state attribution. Fresh proof is required before guardian cleanup; see [Secure Input verification](secure-input.md).
- Parent death before exit-watch registration and during readiness checks, without admitting any input.
- Exit-watch registration failure while the executor is still the guardian's parent. A validated empty, never-armed journal is stopped and settled before returning the startup failure; readiness and input admission remain prohibited.
- Guardian permission-check denial before arming.
- A live blocked executor post: the guardian remains passive rather than racing an active producer.
- Simultaneous peer death: possible holds remain unconfirmed instead of being erased.
- Descriptor replacement rejection, stale guardian health, and conservative shared reservations.
- 128 concurrent reservation/stop boundary races. An admitted reservation prevents settlement until completion; a losing reservation cannot post after the phase becomes terminal.

The host integration fixture separately verifies that a nonzero executor exit does not finish the UI run until the mapped guardian publishes cleanup proof. It exercises the actual ledger and coordinator with virtual process boundaries; kernel-death evidence comes from the separate real-process campaign above. Older control handshakes are rejected before arming.

The initial combined control, process-lifetime, guardian, and inference campaign passed 52 Swift tests, including parameterized fault modes. That guardian/ledger subset had 11 test functions covering 14 cases. Ordinary `scripts/check_control.py .build/debug/AstraControl` also passed ordered/unarmed IPC, malformed input, EOF, signal handling, backpressure, and rejection of an unprotected arm before permission checks; no guardian is launched for an unarmed helper. That milestone's evidence is retained in `.local/guardian-integration-tests.log`.

On 2026-09-12, expanded input/storage/recovery/inference testing exposed an intermittent failed exit-watch registration before reparenting was visible through `getppid`. The guardian previously exited with an empty preparing ledger and no terminal proof. Startup now first validates that readiness was never published and arming never occurred; every failure after that validation atomically cancels admission and settles that empty journal. The production kernel death requirement is unchanged once readiness is published. A deterministic injected ESRCH case exercises this boundary while the executor is still the actual parent. Fixture failures now include the guardian's exit status, error and ledger state rather than only a generic timeout.

After the correction, the broader integration passed 90 tests, and 30 consecutive runs of the 11-test guardian/host-proof subset passed. The original failure logs remain locally in `.local/secure-input-integration.log` and `.local/guardian-bootstrap-stress-traced.log`; corrected evidence is in `.local/secure-input-integration-fixed.log` and `.local/guardian-bootstrap-stress-fixed.log`. These repeated checks specifically investigate the observed intermittent failure, rather than extending ordinary passing checks without a new concern.

## Scope of the result

The campaign proves software ownership and process-failure handling with virtual input effects. It does not establish real target consumption, installed helper TCC attribution, physical HID timing under concurrent human input, or recovery after both peers or the OS die. Terminal mapped state is the proof boundary; active snapshots may reflect conservative intermediate reservations. No cleanup promise is inferred from an unlocked file or a process exit alone.
