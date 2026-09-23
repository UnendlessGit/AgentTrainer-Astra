# Collected experience, retrospective review and PPO

Implemented September 23, 2026. This extends the existing interval-feedback contract into collector production and compute admission. Native workflow integration is tracked separately; these checks do not establish live macOS capture or control qualification.

## Collector contract

Legacy `collector.prepare` schema1 remains strict and supports only `learning` or `audit`. Retrospective collection uses schema2, `purpose: "retrospective"`, and adds:

- `retrospective: {programBase64, programSHA256, sourceSessionID}`: exact original program bytes, their SHA-256, and native authoring-session identity. The program hash must equal `environment.reward_signature`; runtime surface rebinding never changes these bytes.
- Optional `behaviorBatchID`: defaults to the collection UUID. Every fragment of a suspended batch preserves this ID.
- Optional `continuationSource: {sourcePath, manifestSHA256}`: the previously joined pending source. The collector authenticates it, requires the exact original policy/task/settings/batch and copies its original `actorProgress` into the new binding's `previousActorProgress`. It does not replace the cursor's original run UUID. If an explicit previous cursor is also supplied, it must equal that original cursor exactly.

The collector advertises `retrospectiveCollectionVersion: 1`. Existing categorical actor records, raw receipts, frame acknowledgements and joined episode closure remain mandatory. `environment.reward` adds ordered `automaticComponents: [{ruleID, value}]` and `deferredManualRuleIDs`; these must enumerate every automatic/manual rule in original program order. `value` is the automatic subtotal only. Unknown automatic components remain unknown and cannot seal. Rate integrals, rising-edge amounts, score-delta bounds and the subtotal are checked against the frozen definition. Manual rules are explicitly deferred; no live manual-marker coverage is manufactured.

Every eligible interval retains a real endpoint observation. The existing actor row at the exact endpoint is preferred. `collector.endpoint` is a fallback with `{sourceID, episodeID, policyID, precedingPacketID, observation}` and optional finite `value`. Continuing/truncated intervals require the original actor value and a same-episode pre-reset bootstrap. Termination retains endpoint pixels and history without bootstrapping. Endpoint retention does not sample an action or advance an actor draw.

A valid source publishes using the existing `collector.audited` event with `manifest.status: "awaiting_manual_review"`, `manifestSHA256`, `learningEligible: false` and known joined control closure. Its manifest schema is2. Original `frames.bgra`, `decisions.ndjson`, `journal.ndjson`, `audit.json` remain immutable. Additional authenticated members are:

- `source.json`: exact `FeedbackTrajectoryMetadata` consumed by native/Python `VerifiedFeedbackSource`.
- `program.json`: original reward-program bytes.
- `review-frames.ndjson`: one row per eligible interval, `{packetID, observation, endpoint}`. Each observation is `{id, cutoffNanos, images}`; images contain `{offset, bytes, shape, checksum, metadata}` referencing `frames.bgra`. Shapes are `[height,width,4]`, pixels BGRA8. Check the authenticated index/member hashes and each frame SHA-256 before rendering a retained range.

Pending decision rewards contain explicit `null`; `automaticReward`, ordered components, receipts, endpoint observation, packet/draw identity and all behavior tensors remain separately retained. Neither a reviewer nor a learner can interpret absence of annotation as zero.

An operator Stop preserves only previously closed complete semantic episodes. The unfinished current episode is excluded and its actual receipts remain in the audit. No valid prefix produces an ordinary audited source with no learning claim. Complete episodes may publish below the configured PPO minimum; that preserves work, but does not reduce the learning batch requirement.

## Compute jobs

These use the ordinary correlated asynchronous job protocol:

| Operation | Payload | Result |
| --- | --- | --- |
| `feedback.inspect` | `sourcePath`, externally retained `manifestSHA256` | Authenticated `sourceSHA256`, `metadataPath`, `programPath`, `framesPath`, `frameIndexPath`, artifact hashes, `behaviorBatchID`, actual `actorProgress`, `decisions`, `minimumDecisions`, `remainingDecisions` |
| `feedback.materialize` | `sourcePath`, `manifestSHA256`, `revisionDirectory`, ordered `revisionChain: [{id,sha256}]`, new UUID `destination` | Immutable sealed schema2 `rolloutPath`, manifest/hash, batch identity, minimum/remaining counts and actual `learningEligible` |
| `feedback.combine` | ordered materialized `fragments: [{path,manifestSHA256}]`, new UUID `destination` | Immutable sealed schema3 reference batch once the original rollout minimum is met |

Materialization independently validates every revision and parent digest, re-derives explicit review coverage and manual rewards, and requires every interval/rule pair reviewed. A reviewed pair without annotations is zero; an unreviewed pair is unknown. The only changed decision field is total reward = original automatic subtotal + resolved manual contribution. Before PPO admission the consumer independently re-derives that transformation, checks unchanged behavior files and claims the original `behaviorBatchID`/`rolloutID` beside the original source. Another reward revision or output directory cannot reuse it. Partial reviewed fragments cannot directly train; a continuation fragment must enter through its complete ordered batch.

Schema3 references 1–128 independently authenticated reviewed fragments. Each must continue the exact previous pending source and actual actor cursor, preserve the batch/checkpoint/task/settings, and begin after a new physical reset. A separate `CompleteEpisodeBatch` composes strict legacy `Rollout` values: original run IDs, clock IDs, episode/action/observation IDs, timestamps and bootstrap evidence stay intact. Each fragment contains complete episodes; recurrence and GAE never cross a fragment boundary. Same-clock chronology is checked; unrelated monotonic clocks are not compared. One shared bounded frame cache and RAM budget covers all source files. PPO claims the stable batch ID at every contributing original-source parent.

`train.reinforcement.external` accepts reviewed schema2 inputs or schema3 batches. A joined pending source from the final fragment is a valid `job.externalBoundary` witness. Updated checkpoint actor progress comes from that latest real boundary, not from the first fragment or a fabricated cursor. Optimizer/model behavior and production defaults are unchanged.

## Reopening a suspended source

`inference.prepare` accepts `resumeCollection: {sourcePath,manifestSHA256}` with categorical collection, mutually exclusive with `resumeActor` or `seed`. It authenticates the pending source and requires its exact original checkpoint ID/signature/model; a new checkpoint with identical weights is not an alias. Acknowledgement reports `resumedActor: true`, `resumedCollection: true`, original stream/counters and `needsReset: true`. The actor advertises `resumeCollectionVersion: 1`. Recurrent state is reset only after new physical readiness; original artifacts and clocks are unchanged. Offline review needs no actor process.

## Focused evidence

`python/tests/test_review_pipeline.py` exercises a native shared-ring collector, pending source/frame inspection, unknown review rejection, immutable materialization, real separate-process PPO/checkpoint publication and changed-revision reuse rejection. A second workflow pauses below minimum, authenticates the original actor cursor on a new run, retains a real truncation bootstrap, combines two independently reviewed fragments under different clocks, checks independent GAE and performs real PPO with the final actual actor cursor. A bounded assembler check verifies that Stop keeps a completed prior episode and excludes the interrupted current episode. Component checks reject omitted/deferred rule mismatches and invalid automatic totals.

Run the focused file with warnings as errors. This is implementation confidence, not exhaustive regression or live desktop qualification.
