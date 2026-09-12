# Collector process and external learner job — integration contract

This document specifies the next integration increment. Implementation and tests are in progress; it is not a release or live-workflow claim. The collector uses an inherited bounded NDJSON channel and never loads policy weights, samples actions, posts controls or performs learner GPU work. The separate compute process owns the PPO update.

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
| `collector.finish` | `throughSequence` (the last preceding collector request sequence); caller has joined source producers and submitted all final control events |
| `collector.status` | empty |
| `ping`, `capabilities`, `shutdown` | empty; shutdown joins CPU ownership, preserves an aborted audit if unfinished, and never claims native control release |

The snapshot contains `id`, `episodeID`, `cutoffNanos`, `geometryRevision`, `frames:[{metadata,reference,coverageNanos,coverageKind}]`, `controlState`, `events`. References use the existing frame-ring lease schema; metadata is duplicated only for validation against that lease. Real cutoff nanoseconds are preserved. `previousActorProgress`, when supplied, binds the first new actor record to the preceding collection's RNG stream/draw/state/reset generation. Collection purpose `audit` permits continuity episodes without reward labels and cannot produce trainable experience.

Prepare acknowledgement returns `collectionID` (destination UUID), `status:"ready"`, `journalPath`, `destination`, `collectionVersion:1`. Other accepted operations acknowledge `{collectionID,status:"queued"}` before owner-thread work. An enqueue acknowledgement is not a durability or frame-release claim.

## Collector events

All reliable events retain the originating request UUID/run UUID where applicable:

| Event | Payload |
|---|---|
| `collector.framesConsumed` | `collectionID`, `observationID`, `acknowledgements` (exact completed CPU ownership transfers) |
| `collector.applied` | `collectionID`, `requestSequence`, `journalSequence`, `status`; confirms that this input's audit entry has been flushed and its application attempted |
| `collector.fault` | `collectionID`, `code`, `message`, `learningAborted:true`, `journalPath`; stop native control while continuing reliable final evidence delivery |
| `collector.sealed` | `collectionID`, `path`, `manifest`, `actorProgress`, `learningEligible:true` |
| `collector.audited` | `collectionID`, `path`, `manifest`, `actorProgress` or null, `learningEligible:false`, `controlClosureKnown` |

No collector event alone proves physical cleanup. `collector.framesConsumed` permits releasing only the named leases. `collector.applied` means source audit data is durable, not that it passed PPO admission. Native must retain its independent control audit if collector transport/storage fails before that acknowledgement.

## External learning publication

Proposed compute job: `train.reinforcement.external` with `{checkpointPath,rolloutPath,destination,resume}`. It verifies a sealed eligible package, checkpoint/environment/model/configuration identities, original categorical behavior likelihoods, recurrent anchors and truncation values, then performs exactly one PPO update. No environment reset or step occurs in the compute process.

Publication is a separate phase: the job emits `job.progress` with `phase:"waiting_for_actor_boundary"`. Native confirms a joined physical episode/control boundary and sends `job.externalBoundary` with `{jobID,auditPath}`, where auditPath is the latest immutable completed collector package. It may be the original rollout package if the actor remained stopped. The boundary package must continue the same actor stream and preserve every consumed RNG draw, including continuity and excluded suffix actions.

A cancellation restores the pre-update learner/optimizer but still requires the latest boundary proof before publishing a resumable checkpoint. A shutdown without that proof reports no checkpoint. The external checkpoint state wraps the existing version-2 learner state plus current actor progress and requires a new confirmed physical reset; actor reset generations are never relabelled physical reset counts. Changed weights receive a new checkpoint identity and activate only at the next confirmed reset.

The exact boundary-publication details are undergoing parent integration review. Native should implement the collector field table now and wait for the learner publication implementation before assuming additional result fields.
