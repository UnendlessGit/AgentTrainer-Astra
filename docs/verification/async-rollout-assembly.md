# Asynchronous actor evidence and rollout admission

`learning/rollout_assembler.py` separates live actor decisions from delayed reward and execution evidence. It consumes the existing actor's opt-in `collectionRecord`, retains the original sampled token fields and behavior outputs, and assembles complete learning episodes for `ReinforcementTrainer.admit_external_rollout`. It never samples or posts an action, evaluates a reward detector, or resets the environment. A source fixture can execute several actor decisions before sending any rewards.

This is a Python assembly/admission boundary, with an independently tested native actor-record producer. Native desktop run orchestration, continuously running an actor during learner work, reward/reset integration, and publication of the current actor RNG progress with a checkpoint remain integration work. The synchronous [external adapter](external-environment.md) remains available for bounded protocol tests; its exact-period restriction does not apply here.

## Binding and ownership

Construct one `AsyncRolloutAssembler` with immutable environment/model/training configurations, run and clock UUIDs, checkpoint UUID and policy signature, distinct bound actor and environment producer UUIDs, context IDs, a new audit path in an existing run directory, and optional spool directory/limits. The environment signature includes reward/reset program fingerprints. Producer IDs are coordinator-authorized bindings; accepting a JSON field alone is not producer authentication.

The ingress reserves aggregate bytes and item capacity **before** copying a mutable source. Reservations cover concurrent copies, queued records and the record currently being processed; copy or enqueue failure releases them. Generic arrays are detached into immutable bytes. Already owned, contiguous, read-only arrays backed by immutable bytes can be reused, including `FrameRingReader.copy_cpu_frame` results. Metadata and event dictionaries are isolated from later caller mutation. Callers must not concurrently mutate an object while submitting it.

Default ingress limits are 256 MiB and 128 items. Default evidence limits are 256 unsealed decisions and 30 seconds of wall-clock lag. Missing labels, missing coverage, unresolved receipts, and missing truncation bootstrap evidence all consume that lag budget. Overflow or protocol faults reject learning and invoke `on_fault`; the coordinator must then stop/control-audit the actor. Slow cold detector initialization should complete before arming the actor. Limits are configurable and bounded, not promises that arbitrary detector latency is harmless.

Images enter the existing lossless checked disk spool. Training configuration bounds complete collection decisions, RAM/cache metadata and disk. No frame is resized or silently dropped to meet a budget. The assembler retains terminal suffix evidence within those same collection limits. `seal` transfers a rollout only once; its consumer owns closing/discarding it. Cancellation or `close` joins the evidence worker and cleans an untransferred spool. It cannot itself prove native controls were released.

## Coordinator calls

| Method | Required evidence and meaning |
|---|---|
| `begin_episode` | New episode/reset UUIDs, physical readiness nanoseconds, bound environment source, `controls_released=True`, `pending_packets=0`. Readiness may precede the first actual actor cutoff. |
| `submit_actor` | `ActorRecord(packet, collection, observation, events)` from the bound actor source. It returns after bounded ownership transfer/enqueue, without waiting for rewards. |
| `submit_evidence` | A validated existing `Message` envelope from the bound environment source, described below. |
| `close_episode` | Joined stop time, last actor packet sequence, released controls and zero pending packets. Closure cannot omit decisions known to have been sampled/executed. |
| `submit_bootstrap` | Bound behavior actor's pre-reset observation and critic value, episode/policy/preceding-packet IDs, and causal executed events. Needed if a truncation cutoff has no already sampled actor row. |
| `finish_collection` | No more physical episodes will be collected. Delayed evidence for declared actors may still arrive. |
| `seal` | Bounded cancellable wait for all complete episodes/effects and immutable audit publication; returns `CollectedRollout`. |
| `close` | Join and dispose this collector, without asserting anything about control cleanup. |

Actor records and evidence may arrive independently. Evidence for a not-yet-ingested actor is held within explicit item/byte/lag bounds, retaining producer order. Sequence gaps belonging to other wire families are allowed; repeated/backward environment sequences and skipped/repeated actor packet sequences are rejected. UUID comparison is case-independent.

A collection must include at least `training.rollout_decisions` learning decisions and finish all selected episodes. This remains a minimum, not a rule that discards sparse terminal reward beyond a target count. Each episode starts recurrent state at zero and uses one immutable behavior checkpoint. Within-episode surface geometry/role changes require an explicit boundary and reset; different reset episodes may have different supported geometry.

## Exact actor record

The original actor response retains sibling `packet` and `collectionRecord` objects. See [actor producer evidence](actor-collection-records.md) for opt-in prepare/reset/warmup behavior. Collection schema version 1 contains:

- `checkpointID`, `policySignature`, `modelSignature`, `episodeID`, `episodeStep`, `observationID`, `cutoffNanos`, `geometryRevision`, ordered `frameIDs`, and `contextIDs`.
- `previousStateID`, `nextStateID`, `recurrentReset`, `elapsedSeconds`, and `stateBefore`: one FP32 numeric row per recurrent layer, representing the state **before processing this observation**. Batch size is one. UUIDs alone are not recurrent replay evidence.
- `packetFields`: original Int32 arrays for `operation`, `offset`, `key`, `button`, `surface`, `cell`, `within_x`, `within_y`, `dx`, `dy`; every array has packet capacity plus END length. These are not reconstructed from rounded wire coordinates.
- `logProbability`: the original exact joint packet log probability; `value`: the original critic output.
- `sampler`: `kind="categorical"`, `temperature=1`, `mixture="none"`, `version=1`, `rngStreamID`, monotonic `drawIndex`, and two-UInt32-word `stateBefore`, `sampleKey`, `stateAfter`.
- `environmentResets`: monotonic actor reset generation. This can include a successful scratch reset before warmup; it is not proof of a physical environment reset and need not begin at one.

The assembler verifies the declared CPU RNG key split, continuity and nonreuse. Greedy fallback, temperature changes and mixtures are rejected even when a selected action has the same recomputed soft-policy likelihood. RNG state/draw count persist across actor episode resets and are separate from learner/minibatch RNG. Warmup emits no collection record. Trusted producer semantics remain necessary: recomputing log probability alone cannot prove how an action was sampled.

The original packet must retain its UUID, run, monotonic sequence, observation UUID, geometry revision, fixed duration T, exact decoded commands and execution time `cutoff + L`. It is neither re-encoded into new categories nor resampled at ingestion. Only row-major pointer cell layout is remapped when learner spatial padding changes, preserving the exact sampled physical cell and all other categories.

`ActorRecord.from_wire` resolves a retained snapshot through the shared `SnapshotDecoder`. Snapshot fields are `id`, `episodeID`, `cutoffNanos`, `geometryRevision`, `frames`, `controlState`, `events`. Each frame has `metadata`, out-of-band `reference`, `coverageNanos`, `coverageKind`. Collector references require their own ownership path; they cannot reuse an already-acknowledged single-consumer actor lease. Native retained-frame storage/reference routing remains the coordinator's responsibility.

## Timing, labels and actual suffix effects

Actual observation cutoffs are authoritative UInt64 host nanoseconds. They are independently checked against frame event/availability/coverage clocks. The first elapsed-time feature is nominal T; subsequent features use `(actualCutoff - previousActualCutoff) / 1e9` without millisecond rounding. Continuing intervals must be at least T and at most T plus an explicit cadence-delay allowance (500 ms by default). Terminal/truncated intervals may be shorter. Packets still use fixed T/L. Reward and GAE durations follow actual next observation or semantic outcome cutoffs.

Every environment `Message` binds `runID`, `clockID`, `episodeID`; packet evidence uses the original packet UUID as `requestID`:

| Kind | Additional payload |
|---|---|
| `environment.reward` | `startNanos`, `endNanos`, finite `value` or explicit null |
| `environment.outcome` | `endNanos`, continuing/succeeded/failed/terminated/truncated/aborted or unknown/null, optional bounded `reason` |
| `environment.receipt` | Original `receipt` with packet/run/sequence/status/time/control state and per-command results; optional explicit `cancellationCause` |
| `environment.watermark` | `throughNanos`, `throughSequence`, `complete=true` |

`throughSequence` certifies the last applied environment-evidence sequence before this watermark (null only when none exists). It is stronger than an arrival-time assumption. The native producer must wait for detector and manual-feedback coverage barriers before emitting it. Sealed unknown reward/outcome rejects learning; an empty marker list without explicit manual coverage cannot become zero. Feedback remains separate from policy actions, held-control features, and input history. Physical/boundary input or gaps reject the actor history instead of bypassing takeover safeguards.

A delayed terminal label can refer to a cutoff before the actor stopped. PPO admits only the complete prefix whose decisions precede that boundary. It still requires final receipts for every observed suffix packet. The immutable audit contains `excludedDecisions` with their actual commands/final receipts and `postTerminalEffects`, including effects from prefix packets that posted after the semantic boundary. A real posted action is never rewritten as cancelled. Failed/late/rejected suffix evidence remains auditable; such failures in the admitted prefix reject PPO. Cancellation in an admitted prefix is only valid for commands not yet due when an explicit episode boundary cancelled them.

Truncation uses an exact pre-reset bootstrap observation/value, either an existing actor row at that cutoff or a separately supplied value observation. Terminal success/failure bootstraps zero; abort never becomes a learning transition. Old-policy validation replays complete recurrence, saved anchors, sampled likelihoods, decoded wire commands and terminal truncation values before any optimizer update.

## Policy handoff and resumption boundary

`PolicyActivationGate` is a coordinator-owned metadata guard. It admits one learning rollout, allows one pending policy identity, consumes each rollout UUID once (including cancelled/no-change updates), and activates a changed policy only after joined stop plus a fresh confirmed reset. Episodes begun while learning remain continuity/audit episodes even if learning completes before they end. The gate does not load models or authenticate a native stop/reset acknowledgement.

The audit and rollout carry `actorProgress` schema 1: `runID`, `rngStreamID`, final sampled `drawIndex`, `rngState` after that draw, and `actorResetGeneration`. This preserves sampling consumed by the excluded suffix. Native orchestration must publish the latest actor progress with checkpoints and restore it through an explicit new physical episode. The current assembler does not silently replace physical reset counters or claim that learner checkpoint state alone resumes a live desktop world.

## Verification — September 12, 2026

`python/tests/test_rollout_assembler.py`: **16 passed**. Tests use actual small-policy categorical sampling and permission-free practice execution, while delivering labels independently. They verify an actual PPO update after old-policy replay, retained posted terminal suffix effects, exact noninteger-nanosecond durations and truncation, labels preceding actor ingress, sealed unknown rejection, a real wall-clock lag fault, categorical mode/RNG contract rejection, changed wire command rejection before optimizer admission, and RNG continuity across confirmed resets. Concurrency tests block a pixel clone and prove a second producer is rejected before copying; failed copies release capacity. Policy gate tests cover continuity, reset-only activation and duplicate admission after cancelled/no-change updates.

This is not production-model throughput or desktop learning-quality evidence. End-to-end native capture/reference transport, detector/feedback barriers, continuously running actor/learner orchestration, checkpoint actor-progress handoff, and installed-app live RL remain required separate gates.
