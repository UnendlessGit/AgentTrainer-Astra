# Proposed queued-control wire contract v1

Status: independent source review, September 27; **not implemented**. This refines [ADR 0010](0010-queued-control-observations.md). No model, Python, helper or checkpoint behavior changes in this review. It is a GRU input proposal, independent of the unchanged-schema passive-cue diagnostic.

## Ownership and negotiation

`InputExecutor` is the sole authority. Neither `NativeControlSession.pending`, transport receipt arrival order, nor the scheduler's remaining heap can produce this channel. The host dictionary includes uncertain reservations. The executor removes a scheduled item before preflight/post, and original command receipts omit intermediate motion ticks. Those representations cannot certify remaining work.

Add helper hello `controlFeedbackVersions: [1]` and optional `ArmRequest.controlFeedbackVersion`. Absent means the existing path, without feedback admission limits or a feedback allocation. Version 1 must be explicitly supported before arming. A successful opt-in arm returns `controlFeedbackVersion: 1` and a fresh executor-generated `controlEpochID` UUID. That ID identifies one physical arm, independently of persistent actor `runID` and nonzero `initialPacketSequence`. Failed arm creates no usable epoch. The internal generation changed by Stop is not this public epoch ID.

Extend `control.observation` with optional `feedbackCursor: {version: 1, controlEpochID, afterSequence: UInt64?}` alongside the existing raw-event `afterSequence`. An opt-in arm requires this cursor object, initially with null `afterSequence`; unsupported, missing or foreign-epoch requests fail explicitly. Existing arms preserve their current request/response shape. State/permission/cleanup checks remain independent of feedback negotiation. The new model must never fall back silently to an older helper while controlling the desktop.

## Snapshot shape

An opt-in `ControlObservation` adds the following immutable `controlFeedback` object. The enclosing existing `controlState`, `executedEvents`, `intervalCovered`, `cutoffNanos`, `lastSequence` and `controlCoverageNanos` retain their meanings.

| Field | Meaning |
|---|---|
| `version` | Exactly 1. |
| `controlEpochID`, `runID`, `geometryRevision` | Exact admitted lease identity; not learned numeric features. Source roles are resolved through this immutable scope. |
| `cutoffNanos` | Exactly the enclosing atomic cutoff. |
| `coverageNanos` | Exactly the cutoff only when the snapshot is complete and actual control coverage is valid; otherwise null. |
| `unavailableReason` | Null when complete; otherwise `postInFlight`, `arming`, `stopping`, `untrustedState`, `historyGap` or `overflow`. This is not a known-empty queue. |
| `acknowledgedThrough`, `throughSequence` | Accepted feedback cursor and highest included lifecycle sequence, initially null. These are separate from raw input and wire message sequences. |
| `changes` | Contiguous unacknowledged admission/terminal transitions in sequence order. |
| `packets` | One deduplicated row for every outstanding packet and every packet referenced by `changes`, ordered by original packet sequence. |

A change is `{sequence, packetID, kind: admitted|terminal, availableNanos}`. A packet row contains the **complete original `ActionPacket`**, its `admissionSequence`/`admittedNanos`, the per-command progress below, and optional `{sequence, status, availableNanos}` terminal metadata. Terminal status is the existing `executed`, `cancelled` or `late`. A never-admitted rejection belongs in the existing receipt audit, not the admitted queue. Faulted command details remain in receipts; no free-form error strings enter this channel.

`changes` establishes what became known since the preceding accepted observation; the full outstanding rows establish what remains planned now. An acknowledged admission's packet remains present until it terminates. Its final row remains until the terminal change is acknowledged, even if its admission was consumed long ago. A packet admitted and completed between cuts appears once, with both changes. Do not concatenate two copies of its commands in the model encoder.

Zero-command packets are real admitted plans: retain them until their scheduled duration-end sentinel, then retain their terminal transition until acknowledged. They mean no new commands, not release-all. An outstanding empty packet has a packet-presence/duration representation; it must not be confused with the encoder's zero output for no packet evidence.

## Atomicity, progress and motion

Use the existing executor lock to linearize admission, progress/terminal commits and observation. Assign every availability timestamp inside that lock at the actual commit. Copy the bounded immutable snapshot under the lock, then encode it outside. Later receipts cannot revise that copy. A clock equal to another event's timestamp does not override this serialization order.

Each expanded scheduler sample gains an **original semantic command index**, distinct from the existing endpoint-only receipt index. Popping an item for preflight does not advance its ledger progress. After successful posting/no-op resolution, commit progress under the same lock that commits actual state/history. Keep the immutable original command; never replace a relative endpoint with the scheduler's remaining residual delta.

Each original command has exactly one progress entry:

| Field | Meaning |
|---|---|
| `commandIndex` | Index in the original packet, with exact coverage `0..<commands.count`. |
| `status` | `pending`, `partial`, `posted`, `noOp`, `cancelled` or `failed`. `posted` means the backend posting call succeeded, not that the target application acknowledged an effect. A failed post may have affected the OS; it invalidates the observation. |
| `completedSampleCount` | Number of that command's expanded samples whose successful posting/no-op result was committed, initially zero. A failed or cancelled sample does not increment it. |
| `lastCompletedOffsetMs`, `lastCompletedAvailableNanos` | Actual last successfully posted/no-op sample and its lock-committed availability, null before any completion. Scheduled time is derived from the original packet plus this offset; it is not proof of completion. |
| `lastPostedNanos` | Actual source/post timestamp of the most recent successfully posted sample, or null. |
| `emittedDx`, `emittedDy` | Integral cumulative successfully posted raw deltas for a relative-motion command, initially zero; absent for other operations. These are not inferred from the system cursor. |

For motion, retain the previous **original motion command** as the anchor (find its index within the retained full packet); intervening key/button commands do not become anchors. Define scheduler interpolation version 1 for this channel: subsequent same-surface absolute knots interpolate every millisecond; relative segments split the original integral total with nearest-even cumulative rounding and a final residual; surface changes have no interpolated path; the first motion knot and equal-offset knots are instantaneous. Progress identifies the last completed sample of that exact expansion. A later knot with partially posted interpolation is `partial` even though no endpoint `CommandResult` exists yet. Whole-packet storage keeps an already completed anchor available.

Every returned source/post/availability time must be at or before the cutoff. A due but unposted sample remains pending/partial at its actual committed progress; it can have a negative schedule offset relative to the cutoff. Never compute completion by `cutoff - executeAt`, reconstruct relative progress from a warped pointer, or expand millisecond samples into the wire/model input.

A post reserved before the cut but not yet committed makes the existing actual state invalid through `inFlightPacketID`; it also makes feedback unavailable. The caller may poll for a later valid cut, but must not feed that uncertain snapshot to the actor or bootstrap. A selected item still in preflight remains pending and can coexist with a valid old actual state: no OS post has yet been reserved. Stopping, health failure, cleanup, history loss or failed posting likewise cannot certify usable feedback. Do not clear uncertain rows to manufacture an empty queue.

## Cursors and bounded storage

Use one monotonic lifecycle sequence per epoch for admissions and terminals only. Do **not** log every millisecond progress update. Store progress in the bounded live packet row, so taking a snapshot costs at most the retained semantic command count.

Validate both the raw-event and feedback cursors completely **before pruning either history**. Feedback acknowledgements must be monotonic, belong to this epoch and not exceed previously delivered complete history. Equal acknowledgements are allowed. Null is allowed only before the first nonnull acknowledgement. Reject a future/regressed/foreign cursor; never restart from an implicit empty queue. A valid snapshot contains every change after the accepted cursor through `throughSequence`. Partial/ineligible diagnostic snapshots do not advance the feedback delivery watermark. The host advances its consumed cursor only with the validated observation it uses; polling without actor consumption does not discard changes.

Proposed additional opt-in v1 ceilings:

| Resource | Limit |
|---|---:|
| Outstanding admitted packets | Existing 32 |
| Original commands per packet | Existing negotiated 16/32/64 |
| Expanded scheduled items | Existing 8,192 |
| Retained packet rows, including terminal rows awaiting acknowledgement | 64 |
| Retained lifecycle changes | 128, at most admission + terminal per row |
| Encoded feedback object | 262,144 bytes |
| Complete IPC message including raw history/feedback/envelope | Existing strict `< 1,048,576` bytes |

Reserve each packet's complete lifetime capacity **before admission**, including its eventual terminal record and maximum progress representation. A concrete conservative accounting rule is `encodedOriginalPacketBytes + 2,048 + 512 * commandCount`, plus 4,096 bytes per feedback envelope; admit only if the sum fits 262,144 bytes. V1 uses only the bounded fields above, no error strings/duplicated resulting states, so implementation tests can prove that reservation dominates actual worst-width JSON. Reserve two lifecycle slots per retained packet, freeing its admission slot on acknowledgement and the row/remaining reservation only when its terminal is acknowledged. Use checked arithmetic and reserve sequence-number room for every outstanding terminal as well as the new admission/terminal pair; impending overflow closes admission before accepting an unfinishable packet.

The byte reservation and row cap are explicit additional backpressure, not silently reduced model capacity. Admission fails before changing the lease counter or scheduling any event if reservation cannot be satisfied. No post/cleanup path allocates or serializes JSON to reserve terminal capacity. Admission can calculate original-packet bytes outside the scheduler critical section, then revalidate and commit the reservation under the lock. The final actual encoder must also check the feedback and **combined** wire limit outside the lock; overflow stops the run with an explicit error, never truncates history. Combined raw-history limits still matter even when the feedback object fits. Measure normal and maximum supported cadence/lead configurations before making this channel the default.

## Stop, reset and cleanup

Stop closes admission immediately. Record real cancellation/late terminal metadata under the executor lock, preserving any committed partial motion and leaving in-flight outcomes uncertain until the posting path resolves. Keep the public feedback epoch unchanged for the old ledger's final audit. Clearing the scheduler is not proof that no action was posted, and packet termination is not owned-control cleanup proof.

No valid actor observation is issued after that control epoch stops. Existing terminal receipts and the helper/guardian joined-cleanup proof finish the episode; this increment need not add a post-disarm actor-observation API. Previously captured snapshots remain immutable. Diagnostic reads during stopping, if supported, remain unavailable. An unresolved submission is still a stop/audit failure; host receipt timing cannot convert it into invented admission evidence.

The next physical policy or authored-reset run uses a fresh helper/epoch only after the previous owner has joined. A persistent actor's reused run ID or continued packet sequence does not reuse the cursor. Authored reset packets belong to the reset epoch, outside policy episodes. The next actor episode starts from its verified reset boundary and empty *new* executor queue; it must not inherit reset commands as its own past policy samples. Its real initial physical state remains independently observed. Never infer a new empty queue merely from recurrence reset or manual Ready.

## Shared transport and archived replay

The same object must flow without reconstruction through `NativeControlSession.observation` → owned `PolicyActorSession` input/hook → Python `inference.step` → echoed collection record → `CollectorSession`/`EnvironmentObservation` → immutable `ObservationRecord`/rollout artifact. Update strict field allowlists, validators, byte quotas and actor/collector equality checks together. Include its original bytes/canonical fields in observation identity and artifact integrity. A mismatching epoch/cutoff/progress echo is an admission failure.

PPO, burn-in, full-prefix replay and bootstrap use that **recorded original** snapshot. They never replace it with current queued actions, final future receipts or resampled actions. Original sampled actions remain likelihood targets. Preserve the pre-cleanup queue at a time-limit bootstrap observation; subsequent cancellation must not rewrite it. Retrospective correction lead-up remains original provenance; human follow-up never gains a queue from shifted teacher labels. Any archive extension and learned model/observation schema are explicit new versions; old artifacts/readers keep their exact old path.

Practice needs the same semantic packet ledger. Today `_schedule` stores only expanded command items: empty packets have no lifecycle entry, and `_pending` is also the oracle's “wait for actual commands” predicate. Add an independent packet/end ledger and retain real DecisionContext packet/run IDs. Do not make empty/end sentinels keep that oracle predicate true forever. Snapshot after the simulator's defined half-open step execution, before the *current* oracle/policy packet and before terminal cleanup; a sample due exactly at the next cutoff may still be pending. Live and practice must agree on known completed progress, not assume all due samples already ran.

Human BC feedback is unavailable unless the full segment has explicit provenance. An optional future `RecordingSession` can acquire the existing global `DesktopControlLock` after prior cleanup joins, before it admits frames/events, and hold it through producer joins. Persist bounded lock identity, recording ID, monotonic covered start/end and clean producer-join proof. Select only ranges within that interval. A missing/incomplete end proof after recovery means unavailable. This excludes **Astra-owned** queues only, without changing raw origins or the existing 178 actual-control features. Legacy recordings and a UI-only mutual-exclusion guard cannot certify known-empty.

## Implementation and acceptance order

1. Implement only opt-in native types/ledger/helper/session plumbing and permission-free fixtures while Python/model defaults remain frozen. Existing non-opt-in behavior remains exact. Check admission just before/after a cut; selected-preflight and blocked-post races; no-op/terminal-between-cuts; interleaved motion, nearest-even relative residuals, surface discontinuities and zero-command lifetime; cursor gaps/retries/atomic rejection; reserved terminal space/combined bytes; Stop/rearm with repeated run ID and fresh epoch.
2. Add shared Python/practice/archive transport and exact live-versus-replay equality fixtures. Cover cutoff-bound progress, empty/unavailable distinctions, time-limit bootstrap before cleanup, repeated polling and future-label invariance. No learned input activation before this chain is complete.
3. Introduce the separate GRU residual/schema/warm-start path from ADR 0010. Add it after the old temporal input projection and before normalization, preserving old parameter shapes. Prove old-path and zero-residual equality, padding/reset behavior, PPO ratio-one/value replay and subsequent encoder gradients (a zero output projection initially blocks encoder gradients by design). Keep IDs out of neural inputs and the no-evidence residual exactly zero.
4. Measure real maximum-payload/warm latency under current deadlines and use a bounded delayed-action diagnostic with identical visible/actual controls but different previously admitted packets. This establishes queue observability, not broad learning quality or a solution to passive-cue chance performance.

Review basis: actual `InputExecutor` admission/observation/expansion/service/Stop paths, `AstraControl` routing, `NativeControlSession` admission/terminal separation, native actor/collector construction, Python strict transport/rollout replay, practice scheduling and the current recorder's ownership. This review ran no tests, GPU experiments, GUI actions or physical input.
