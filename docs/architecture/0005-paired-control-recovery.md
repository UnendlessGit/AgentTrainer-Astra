# Paired control recovery

The executor's independent threads cannot recover after the entire process is killed. UI-only cleanup also fails when the UI exits. Per-event synchronized disk journaling would add filesystem latency to the input scheduler, while asynchronous logging could omit the last down event. Astra therefore pairs the executor with a cleanup-only guardian and a small shared atomic ownership ledger.

The guardian is a re-execution of the same `AstraControl` binary with a private role. It receives only an existing desktop-lease descriptor and recovery mapping descriptor; standard streams point to `/dev/null` and other descriptors are close-on-exec. It requests no permissions. Already-granted Accessibility, Input Monitoring, and event-posting access are checked before readiness; actual installed-bundle attribution remains a release qualification.

The host creates a private, preallocated 4 KiB ledger before arming. The descriptor includes run/ledger UUIDs and file identity. A native C11 shim uses aligned lock-free atomics for phase, possible-held key/button masks, in-flight posting, readiness, process identity, and completion. No Swift references or process-local locks live in the mapping. Initialization and terminal persistence may synchronize; the event hot path uses only atomic memory operations, with no disk write, synchronization, or pipe round trip.

## Ownership and proof

Only the executor accepts action packets. It reserves a possible hold before posting a down event; successful normal up events may clear that reservation after the API returns. During stop, conservative bits remain until the existing local driver and provisional cleanup paths have all joined. A reservation racing stop rechecks the phase after incrementing in-flight state and cannot reach posting after terminal settlement. Partial active snapshots are conservative diagnostics; they are not cleanup proof.

Before publishing readiness, the guardian validates its actual parent and registers a kernel `EVFILT_PROC/NOTE_EXIT` watch. If the parent dies before readiness, an empty ledger with no armed marker can terminate harmlessly without posting. Once armed, the guardian may claim release authority only after the watched executor has exited. It never treats a slow heartbeat or closed pipe alone as evidence that an in-flight post cannot return.

The desktop lease is shared through duplicated references to the same locked file description. Every owner closes only its own reference: explicit `LOCK_UN` would also unlock the surviving peer's duplicate. If the executor dies, the guardian's descriptor retains exclusion during recovery. If the guardian dies, the executor stops admission and performs local cleanup while retaining its own descriptor.

When the guardian sees terminal settlement while the executor is still alive, the guardian may close its reference after terminal synchronization. The executor's `ControlGuardianPair` still retains the parent reference until guardian exit is reaped. Thus the lease never disappears while either side can still post. A guardian stuck after proven terminal settlement may be killed by its actual parent; reaping and signalling share one lock to prevent PID reuse between ownership checking and that exceptional kill. This is never permitted for pending cleanup.

Cleanup submits only up events for possibly owned controls. A currently physical hold transfers to the person and is not released synthetically. The native backend rechecks HID state immediately before each up event, and requires physical-state verification and posting permission. Failed releases keep their bits and lease for retry. These establish native release submission and ownership settlement, not proof that every arbitrary target application consumed an event.

## Runtime integration

The control handshake advertises `recoveryVersion: 1`; the live host rejects older helpers before arming. `ArmRequest.recovery` names the host-created ledger, and the arm acknowledgement echoes its ledger UUID and guardian PID. The host verifies the mapped armed/readiness evidence. This prevents an older helper from silently ignoring the new descriptor and executing without protection.

After command-process exit, the host continues waiting for a matching live guardian if the ledger is not settled. Guardian identity includes process birth time so a reused PID cannot impersonate the peer. A sticky armed marker distinguishes failure before input authority from interrupted execution. Saved results retain `cleanupConfirmed` and `cleanupRecoveredByGuardian`; an executor crash remains an interrupted run, never an automatic continuation of the policy.

## Explicit limit

The mapping is a same-boot process-crash mechanism, not a power-loss journal. Simultaneous death of executor and guardian can leave unconfirmed possible holds; no terminal proof is fabricated. Existing manual-release warnings and startup interrupted-run evidence remain necessary. Installed physical input effects, revoked permissions, forced death of the real signed helper, and final DMG qualification are still open. See [guardian verification](../verification/control-guardian.md).
