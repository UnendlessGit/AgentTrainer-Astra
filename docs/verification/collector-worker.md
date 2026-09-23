# Collector process and external learner job — integration contract

The dedicated collector role, immutable process handoff, and external learner job are implemented and tested. This is not a release or installed desktop learning-workflow claim. The collector uses an inherited bounded NDJSON channel and never loads policy weights, samples actions, posts controls or performs learner GPU work. The separate compute process owns the PPO update.

## Native request fields

Start the helper with `--role collector`. Every request uses existing `Message` version 1, a fresh `requestID`, strictly increasing sender `sequence`, and the actor's bound `runID`. `prepare` must complete before arming control. Paths are absolute, normalized, locally authorized paths; each ring is owned by this collector consumer, never a retired actor-consumer lease.

| Operation | Exact payload |
|---|---|
| `collector.prepare` | `schemaVersion:1`, `clockID`, `environment` (full `EnvironmentSpec.to_dict()`), `model` (full model config), `training` (reinforcement config), `policyID` (checkpoint UUID), `policySignature`, `actorSourceID`, `environmentSourceID`, `contextIDs`, `destination` (new package directory ending in canonical UUID), `rings:[{path,ringID}]`, `purpose:"learning"\|"audit"`, optional `limits`, optional `previousActorProgress` |
| `collector.begin` | `sourceID`, `episodeID`, `readyNanos`, `resetID`, `controlsReleased`, `pendingPackets` |
| `collector.actor` | `sourceID`, `response` (unchanged opt-in actor response), `observation` (exact retained snapshot below) |
| `collector.evidence` | `sourceID`, `message` (full original environment wire envelope: version, kind, sequence, payload, requestID when applicable, runID) |
| `collector.bootstrap` | `sourceID`, `episodeID`, `policyID`, `precedingPacketID`, `value`, `observation` (snapshot with its causal events) |
| `collector.end` | `sourceID`, `episodeID`, `stoppedNanos`, `lastActorSequence`, `controlsReleased`, `pendingPackets` |
| `collector.abort` | `reason` (bounded text); stops learning admission, continues collecting actual late decision/receipt audit until finalization |
| `collector.finish` | `throughSequence` (the last preceding source-mutating collector request sequence; ping/status/capabilities do not advance this evidence watermark); caller has joined source producers and submitted all final control events |
| `collector.status` | empty |
| `ping`, `capabilities`, `shutdown` | empty; shutdown joins CPU ownership, preserves an aborted audit if unfinished, and never claims native control release |

The snapshot contains `id`, `episodeID`, `cutoffNanos`, `geometryRevision`, `frames:[{metadata,reference,coverageNanos,coverageKind}]`, `controlState`, `events`. References use the existing frame-ring lease schema; metadata is duplicated only for validation against that lease. Real cutoff nanoseconds are preserved. `previousActorProgress`, when supplied, binds the first new actor record to the preceding collection's RNG stream/draw/state/reset generation. Collection purpose `audit` permits continuity episodes without reward labels and cannot produce trainable experience. It validates CPU frame leases but retains only exact actor/receipt/RNG and observation metadata, avoiding retention of image tensors that no learner may consume. Each collector process owns one collection/package; a subsequent learning or continuity collection starts a new collector with separately owned rings.

Ingress counts queued and in-flight work, reserving before any raw-frame copy. The streaming journal has a separate `maximum_journal_bytes` limit (256 MiB by default, at most 1 GiB), because numeric GRU anchors can exceed a compact terminal report's allowance during a long learner update. `maximum_audit_bytes` remains 32 MiB by default. Source requests are flushed and fsynced before `collector.applied`; rejected overflow messages are not falsely acknowledged as durable. Their native producer must retain them in its independent audit.

Prepare acknowledgement returns `collectionID` (destination UUID), `status:"ready"`, `journalPath`, `destination`, `collectionVersion:1`. Other accepted operations acknowledge `{collectionID,status:"queued"}` before owner-thread work. An enqueue acknowledgement is not a durability or frame-release claim.

## Collector events

All reliable events retain the originating request UUID/run UUID where applicable:

| Event | Payload |
|---|---|
| `collector.framesConsumed` | `collectionID`, `observationID`, `acknowledgements` (exact completed CPU ownership transfers) |
| `collector.applied` | `collectionID`, `requestSequence`, `journalSequence`, `status`; confirms that this input's audit entry has been flushed and its application attempted |
| `collector.fault` | `collectionID`, `code`, `message`, `learningAborted:true`, `auditContinuable`, `journalPath`; true only for a healthy explicitly requested operator abort, false for invalid/overflow/storage faults |
| `collector.sealed` | `collectionID`, `path`, `manifest`, `actorProgress`, `learningEligible:true` |
| `collector.audited` | `collectionID`, `path`, `manifest`, `actorProgress` or null, `learningEligible:false`, `controlClosureKnown` |

No collector event alone proves physical cleanup. `collector.framesConsumed` permits releasing only the named leases. `collector.applied` means source audit data is durable, not that it passed PPO admission. Native must retain its independent control audit if collector transport/storage fails before that acknowledgement.

## External learning publication

Compute job: `train.reinforcement.external` with `{checkpointPath,rolloutPath,destination,resume}`, with optional `boundaryTimeoutSeconds` (1–600, default 60). Each compute job uses a fresh job `runID`; the package independently binds the live actor `runID`, allowing several serialized updates in one actor run. It verifies a sealed eligible package, checkpoint/environment/model/configuration identities, original categorical behavior likelihoods, recurrent anchors and truncation values, then performs exactly one PPO update. No environment reset or step occurs in the compute process.

Publication is a separate phase: the job emits `job.progress` with `phase:"waiting_for_actor_boundary"`. Native confirms a joined physical episode/control boundary and sends `job.externalBoundary` with `{jobID,auditPath}`, where auditPath is the latest immutable completed collector package. It may be the original rollout package if the actor remained stopped. The boundary package must continue the same actor stream and preserve every consumed RNG draw, including continuity and excluded suffix actions.

The required `waiting_for_actor_boundary` progress item is reliable protocol
control: output coalescing/reclamation can remove ordinary metrics but cannot
replace or drop this handshake. A saturated reliable queue fails the channel
explicitly. A clean stop before a learnable rollout uses
[`checkpoint.externalBoundary`](stopped-actor-checkpoint.md) to preserve the
latest real cursor with unchanged weights and optimizer state.

Cancellation during the update or while waiting for its publication boundary restores pre-update parameters, optimizer and counters; it still requires the latest boundary proof before publishing a resumable checkpoint. Timeout or shutdown without that proof reports `checkpointPublished=false`. The boundary request uses the compute job's `runID` and `jobID`; its package must match the rollout's actor run/clock/policy/source identities, environment/model/context configuration, and RNG stream. Older draw/reset generation, inconsistent unchanged-draw state, discontinuous continuation, reused rollout and reused boundary proof are rejected. No further actor sampling may occur after that boundary until publication and a fresh physical reset/activation. The external checkpoint state wraps the existing version-2 learner state plus current actor progress and requires a new confirmed physical reset; actor reset generations are never relabelled physical reset counts. Changed weights receive a new checkpoint identity and activate only at the next confirmed reset.

The completed job returns `checkpointPath`, `manifest`, `checkpointPublished`, `cancelled`, `resumable`, `requiresEnvironmentReset`, `sourceKind`, `provenance`, `actorProgress`, `boundaryCollectionID`, `rolloutID`, and update `metrics`. Missing-boundary cancellation returns no checkpoint path and a reason. External state is `{kind:"reinforcement_external",schemaVersion:1,learner:<reinforcement v2>,actorProgress,consumedRolloutIDs,requiresEnvironmentReset:true}`. The saved learner policy identity equals the new immutable checkpoint UUID. Actor reset generation remains distinct from learner physical reset counts.

Native may accept `auditContinuable=true` only after its own explicit abort request and must keep forwarding final actor/receipt evidence in that case. False latches failure and requires retaining unacknowledged source evidence in the independent native journal. A healthy explicit `collector.abort` switches to validated audit-only draining and can finish as `audited` after every actual receipt and joined source boundary arrives. Protocol/storage/overflow faults preserve an `aborted` journal that may include unvalidated evidence and cannot authorize checkpoint progress. An interrupted channel finalizes an aborted package when storage permits; otherwise the fsynced `.collector-<UUID>.partial/journal.ndjson` remains for recovery. Neither form invents control release.

## Immutable package

A package publishes by an exclusive atomic directory rename after syncing its files and parent. `manifest.json` binds collection/run/clock/policy/source/configuration identities, actor progress, explicit sampler declaration, source closure evidence, status, decision count and SHA-256/byte extents. `journal.ndjson` preserves input envelopes. `audit.json` preserves completion/failure and actual terminal suffix effects. Eligible learning packages additionally contain original lossless `frames.bgra` and bounded `decisions.ndjson` with scalar transitions, original token arrays, FP32 recurrent anchors and exact observation references. Numeric state never uses pickle or executable deserialization.

The compute reader verifies package hashes, structural/resource bounds and individual frame checksums, then uses a bounded read-only frame cache. It does not resample or replay OS actions. Learner admission still recomputes behavior probabilities, recurrent anchors, wire commands and truncation values before gradients. Rollout-consumption and boundary-use claims are fsynced sidecars adjacent to the immutable packages. A crashed consumer must collect fresh experience rather than silently admit the same rollout twice.

## Evidence

The focused collector suite currently passes ten tests. Real native mapped-slot fixtures feed a separate collector process, retire/reuse its leases, and verify source pixels after native ring deletion. A separate compute process performs real small-policy PPO and publishes only after a bound external boundary. Checks cover cancellation rollback, missing-boundary no-publication, wrong run, corrupted image rejection, explicit-abort and continuity audits, saved optimizer/RNG resumption at a new physical episode, a later continuity audit advancing checkpoint RNG beyond the learning rollout, and overflow preserving all accepted journal records without claiming control release. The final focused collector, assembler, jobs, synchronous external adapter and recurrent trainer run passed **94 tests with warnings treated as errors**. Full application/frozen-worker integration remains the root verification step.

These are permission-free transport/numerical correctness checks with small policies. They do not establish production-model contention, sustained storage throughput, learning quality, or native reward/reset/live desktop integration.

### Native disarm receipts and global packet counters

The non-shipping `AstraReceiptFixture` executes the real `InputExecutor` against a private virtual backend and clock. Its actual disarm-cancelled and post-disarm rejected receipts carry `resultingState.valid=false`; posted effects remain posted, and cleanup failure leaves unresolved held keys visible. No Core Graphics input, capture, privacy preflight or event tap is used by this fixture.

The asynchronous evidence reader preserves those inactive final states verbatim. Active admission and executed learning-prefix snapshots still require `valid=true`. Only cancellation explicitly attributed to the semantic episode boundary can enter the learning prefix with an inactive final state, and every cancelled command must have been scheduled at or after that boundary. Administrative cancellation, overdue cancellation, late/rejected packets and failed commands do not enter the learning prefix. Invalid finals remain auditable. A receipt is never a substitute for separately joined control cleanup; the native fixture's cleanup-failure case cannot authorize a reset/publication boundary.

Collection schema 1 requires `packet.sequence == collectionRecord.sampler.drawIndex`. Both continue across episode resets. A resumed collector's first packet sequence is the preceding actor progress draw index plus one, and `collector.end.lastActorSequence` uses that global packet watermark rather than an episode-local counter. The original sample key chain remains independently checked.

The focused receipt, collector and assembler run passed **33 tests with warnings treated as errors**. Five tests consume actual native receipt fixtures, and the collector resume test uses the restored global sequence. The native fixture also proves that a subsequent disarm does not retroactively invalidate an earlier executed receipt.
