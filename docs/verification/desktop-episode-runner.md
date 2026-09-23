# Desktop episode runner and parent-loop integration

`DesktopEpisodeRunner` runs one physical policy episode over a prepared, reset and warmed `PolicyActorSession`. The parent owns the actor, capture stream, collector and learner. The runner owns its fresh `NativeControlSession`, immutable evidence generation and manual/reward producer join. It does not stop or recreate the parent owners.

## Admission and evidence

Startup requires a successful matching `ResetResult`, an available categorical actor binding with the exact checkpoint and zero episode step, known persistent packet/draw counters, joined previous control, matching scope and fresh source evidence. Reward preparation, source verification and the evidence queue's readiness barrier complete before control is armed. Installing the control session and checking for an earlier Stop share the runner lock. The `arming` progress callback can therefore stop before publication without allowing a late helper to start.

The actor's owned-observation callback offers immutable pixels/input to the evidence bridge before compute. Reward analysis can discover a terminal condition while that prediction remains pending. New cutoffs are scheduled from the previous actual cutoff plus the policy period. The runner requires the helper's explicit `controlCoverageNanos` at that atomic cutoff; it never changes `ControlState.observedNanos` to pretend old input was newly sampled. Scope/geometry, source freshness and the immutable execution lead are checked independently while a prediction is outstanding.

Stop closes native admission synchronously, then drains the original prediction. `submitPreservingStoppedPacket` makes one locked reservation, choosing live submission or a genuine rejection after acknowledged disarm. A Stop between the caller's earlier state check and that reservation cannot lose the produced packet. Duplicate, foreign, out-of-order and already-closed submissions still fail; uncertain transport is never resent. Produced, admitted and executed counts remain distinct.

The exact response, observation, packet identifiers, absolute timing and random-stream progress are retained. A time limit remains `truncated`. The actual actor row at the terminal cutoff supplies the existing assembler's exact bootstrap observation/value, even when its unchanged action packet becomes a rejected suffix. A missing/untrustworthy sampled result fails continuity; it cannot become a zero-valued bootstrap or an empty successful episode.

## Joining and recovery

The result includes the original `DesktopEpisodeJoin` supplied to the evidence bridge, its optional completion, the actor state, native cleanup completion and the primary issue. `actorJoined` refers to this episode's pending request; the persistent actor process stays alive. Helper/guardian exit and manual callbacks join before evidence closure, which then joins reward work. Missing manual coverage remains unknown. No result is invented merely to seal an empty collection.

A reset's `ResetCancellation` is created before asynchronous host admission. It is sticky, bound to one reset ID and consumed once. `ResetRunner.run(..., cancellation:)` attaches it only after installing its active state, applies any earlier Stop before preparation, and detaches after joined completion. A wrong/reused token cannot prepare controls; an old token cannot cancel another reset.

`NativeControlOwner.pendingManualCleanup` exposes immutable evidence from a joined but unconfirmed helper. A persistent `NativeResetDriver` retains its exact failed native completion, rejects a new attempt until explicit acknowledgement, and guards acknowledgement by both reset and native session ID. Neither acknowledgement path rewrites an earlier false release proof. An episode owner can use the native owner warning directly; a retained reset driver routes acknowledgement through its own guarded API.

## Verification

On 2026-09-19, this permission-free command passed **51 reported tests / 61 cases**, with no compiler warnings or test failures:

```sh
./script/swift.sh test --filter 'desktopEpisodeRunner|desktopLearningLoop|resetRunner|nativeReset|nativeControl'
```

Log: `.local/desktop-runner-loop-final.log`.

The new runner workflows exercise terminal detection during a pending sample, unchanged disarmed rejection, actor-deadline Stop, empty cancellation, unknown actor progress, unproved control coverage, pending cleanup, manual-producer joins/unknown reward, repeated physical episodes with persistent counters, stale callbacks and Stop immediately before helper publication.

Parent-loop tests use the real native owner/session/bridge classes with injected virtual process responses. They collect two episodes per update across two updates, hold the actor pause while learning, activate a new checkpoint only after the next successful reset, preserve the original joined audit when stopped before the rollout minimum, stop during collector finalization, cancel reset before its runner exists, reject unconfirmed reset cleanup, and preserve a verified boundary without erasing the primary learner failure. The fake collector publishes protocol manifests for lifecycle testing; these tests do not exercise PPO kernels or authenticate a real learner package.

Independent review found the pre-publication Stop race; the atomic installation regression covers its correction. Parent review found the dispatch-reservation Stop race; the native reservation regression covers its correction. The earlier fixture run exposed a missing control-coverage echo and noncanonical progress UUIDs; the fixtures were corrected to preserve the actual contracts rather than weakening validation.

Live source capture, OS event consumption/privacy attribution, installed-app interaction and model-learning quality remain separate release evidence. This runner currently uses the native collector's explicit single-source contract. No test here grants privacy access, inspects a personal window, posts a Core Graphics event or performs GPU training.
