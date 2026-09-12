# External environment boundary

`environments/interface.py` defines the environment-neutral observation, decision and transition types. `PracticeAdapter` preserves existing practice configuration/checkpoint identities. `ExternalEnvironment` accepts asynchronous messages through a bounded mailbox and can drive the same `ReinforcementTrainer`, including complete-episode spooling, exact old-policy replay, joint packet likelihoods, candidate admission and reset-only policy activation. Rewards, OCR, image matching, manual feedback, input posting and reset automation remain the peer's responsibilities.

**This is currently a synchronous trainer adapter.** A decision waits for its observation, admission and sealed reward/outcome before returning. Delayed execution receipts do not block the next observation, but delayed reward computation still does. It does not yet provide real-time actor continuity during reward computation or learner work. The separately tested [asynchronous assembler](async-rollout-assembly.md) consumes continuously sampled actor records and delayed evidence; native/job/UI orchestration remains separate work.

## Specification and ownership

`EnvironmentSpec.to_dict()` emits every versioned field; `from_dict()` requires that complete schema. It pins timing, capability vocabulary, source capacity, maximum episode length, freshness, discount scale, seed behavior and reward/reset program fingerprints. An external source requires both program fingerprints. Live run and monotonic-clock UUIDs are bound separately so a fresh process can resume compatible learning without pretending to restore a live world.

Construct `ExternalEnvironment(spec, run_id=..., clock_id=..., emit=..., resolve_frame=...)`. `emit(kind, payload, request_id, run_id)` must enqueue into a bounded nonblocking transport whose owner assigns global sender sequences. The owner feeds decoded `Message` objects to `receive`; it must not give the learning thread a competing reader for the same pipe. Sender sequence gaps for other message families are permitted, but repetition/backward movement is rejected. UUID comparisons are semantic, including Swift's uppercase encoding.

The mailbox holds at most 128 messages and a configured byte limit (8 MiB by default), with bounded cancellable waits. Overflow, malformed identities or disconnection wake the actor and fail admission. A stopped acknowledgement remains receivable after a fault, so cleanup can still be verified. Reusing a faulted adapter cannot silently reset or emit another action.

Frames never appear as base64 or tensor JSON. `FrameRingResolver` resolves only pre-bound `FrameRingReader` objects; incoming paths do not authorize file access. `copy_cpu_frame` shares the existing read-only file, metadata, generation, lease and before/after-copy header checks with `copy_frame`. It copies directly into immutable CPU BGRA for spooling, avoiding an MLX upload/readback. Each completed ownership transfer emits its exact acknowledgement, including transfers preceding a later observation validation failure. Unacquired/failed leases still require coordinated consumer retirement; a control-stop acknowledgement alone is not permission to recycle unacknowledged frame slots.

## Wire envelopes

All envelopes retain the existing `Message` version, sender sequence, UUID `requestID` and bound `runID`. Every payload includes the bound `clockID`. Requests and asynchronous evidence correlate by packet/reset UUID, never by arrival order.

| Kind | Payload beyond clockID | Meaning |
|---|---|---|
| `environment.reset` | environmentSignature, previousEpisodeID, seed or null | Request a fresh confirmed episode; an unsupported physical-world seed remains null |
| `environment.ready` | episodeID, environmentSignature, controlsReleased, pendingPackets, observation | Matching reset request; new episode, cleared controls/ledger, and initial observation |
| `environment.action` | episodeID, episodeStep, policyID, decisionNanos, packet | Exact sampled packet and the observation/behavior identity that produced it |
| `environment.observation` | episodeID, observation | The next decision snapshot; requestID identifies the preceding submitted policy packet |
| `environment.reward` | episodeID, startNanos, endNanos, value | Value is finite or explicitly null; it may be resolved/revised before sealing |
| `environment.outcome` | episodeID, endNanos, outcome, optional reason | continuing/succeeded/failed/terminated/truncated/aborted, or null/unknown |
| `environment.watermark` | episodeID, throughNanos | No further reward/outcome changes at or before this cutoff |
| `environment.receipt` | episodeID, receipt, optional cancellationCause | Admission/final execution evidence for the actual packet UUID and sequence |
| `environment.framesConsumed` | observationID, acknowledgements | Completed owned copies; a multi-surface snapshot may emit several acknowledgements |
| `environment.abort` | episodeID or null, reason | Stop queued actions and the episode; null is possible while reset readiness is pending |
| `environment.stopped` | episodeID or null, controlsReleased, pendingPackets | Matching abort request proves cleared owned controls and zero pending packets |
| `environment.fault` | reason | Explicit peer failure; no synthetic reward or readiness is inferred |

`packet` follows the semantic `ActionPacket` fields in CONTRACTS.md. Its execute time is `decisionNanos + lead`, with the fixed configured duration and original decoded commands. The external packet sequence is monotonic across the run; episodeStep starts at zero after reset. A native coordinator must retain an explicit mapping to its control-helper lease/run/sequence identities and validate real helper receipts before returning external receipts. Mapping identity is not permission to convert failed, foreign or late effects into success.

`observation` contains id, episodeID, cutoffNanos, geometryRevision, frames, controlState and events. Each frame contains metadata (the existing FrameMetadata), reference, coverageNanos and coverageKind. Snapshot UUID/cutoff are independent of any frame UUID/event/availability time. All source/availability/coverage times must satisfy `source <= available <= coverage <= cutoff`. Fresh-frame coverage uses the frame's actual availability time and checks source age. Explicit `unchanged` coverage permits verified idle pixels while retaining their original timestamps; the producer must establish that coverage. Stale coverage and future frames cannot masquerade as a fresh snapshot. Surface roles/order/geometry are fixed within an episode and change only through a new reset. Authoritative controls must be valid and fresh, and executed events must be causal and strictly ordered.

Physical/boundary input or a history gap invalidates active rollout input. Operator feedback is a separate producer channel; its keys, clicks, held state and visual feedback must not be laundered into policy inputs or disable takeover guards. The adapter never infers reward from a key code or from missing marker traffic. Native manual feedback requires its producer's explicit coverage proof before the coordinator seals the scalar reward.

## Timing and admission

Reward covers the decision interval, never the delayed execution interval. A continuing interval is exactly one configured period; a terminal/truncated interval may end earlier at an integer-millisecond cutoff. Observation cutoff must equal that independently sealed interval end. A watermark sealing missing/null reward or unknown outcome rejects the rollout. `succeeded` and `failed` both map to PPO termination while their label is retained as `CollectedDecision.outcome_detail`. Truncation retains the pre-reset bootstrap observation; abort never becomes a successful training row.

For lead greater than or equal to the period, a just-sampled packet may execute after its next observation. The adapter therefore assembles that transition after admission while retaining bounded outstanding receipts. It verifies all final receipts before sealing an episode or resetting. Command indices, counts, schedules, posting timestamps and statuses must agree with the original packet. A late/rejected/failed receipt rejects the rollout. Cancellation is valid only when explicitly caused by the confirmed episode boundary and every cancelled command was not yet due; actual posted effects are never rewritten as cancellations. Delayed terminal detection and audited post-terminal suffixes are handled by the separate asynchronous assembler; real-time native coordination remains an integration gate.

Cancellation is checked before emitting reset/action requests and during waits. Abort waits for a matching cleanup proof even after a protocol fault. Failure to receive that proof remains a cleanup error, rather than a false assertion that the peer released controls. The native independent watchdog remains necessary and is not replaced by this Python protocol.

## Evidence

`test_external_environment.py` uses a separate threaded peer backed by the real permission-free practice environment, not its oracle. It serializes control messages, transfers owned pixels separately, delays receipts past subsequent observations, and runs actual small-policy PPO with old-policy replay, checkpoint-compatible identities and complete episodes. It tests null-to-known rewards, sealed unknown rejection, shifted reward windows, independent stale/future frame/cutoff validation, verified idle coverage, multiple surfaces, cancellation/readiness cleanup, unsealed known labels, native UUID casing and no action submission after cancellation.

`test_frame_ring.py` additionally exercises native mapped-slot CPU ownership, immutable bytes after acknowledged slot reuse/unlink, and rejection when the lease changes during CPU ingestion. Existing MLX ingestion tests remain unchanged. These are transport/numerical integration proofs; native desktop rollout orchestration, production-model performance, reward-detector throughput and end-to-end UI learning remain separate gates.

The admission review also validates saved recurrent anchors used by burn-in and independently replays the terminal truncation bootstrap value. Payload observation/episode/cutoff identities are checked against scalar transition records. Exact replay segments follow episode boundaries, supporting different reset geometries without mixing incompatible surface-role counts.

A minimal padding reproduction combined 32×64 and 64×64 observations with a pointer target at (.875, .3125). The narrow standalone cell index was 11. Retaining index 11 in the padded eight-column grid decoded y=.18847656 instead of .31347656. PPO batching now maps that exact physical cell to index 19 by its original row/column; sampled operation, timing and within-cell categories remain unchanged. A regression verifies decoded physical commands, actor log probabilities, ratio one and finite PPO gradients on the mixed-width batch. BC already encodes commands against the final padded layout and correctly produces index 19; its behavior was preserved. Within-episode external geometry changes remain explicit discontinuities requiring abort/reset.

The combined latest source RL math, spool, native frame-ring, external adapter, recurrent trainer, asynchronous job and data-preparation suite passed **113 tests** on September 11, 2026. The mixed-width regression includes an actual PPO gradient and unchanged non-cell token fields; recurrent-anchor and truncation-value tampering are rejected before optimizer admission.
