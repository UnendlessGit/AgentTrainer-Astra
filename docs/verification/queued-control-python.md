# Queued-control transport and replay

September 27, 2026. Implements the opt-in Python/practice portion of [wire v1](../architecture/queued-control-wire-v1.md). Existing model schema 2 remains the default; no old checkpoint, optimizer state, observation, dataset or rollout is implicitly upgraded. This is computer-interaction plumbing and numerical correctness evidence, not a learning-quality result or a fix for passive cue learning.

## Preserved input and versions

`astra.control_feedback` validates the original bounded native object: exact fields and UUID identities, cutoff/coverage, contiguous lifecycle cursors, outstanding packet retention, original semantic command arguments, actual committed sample count and availability, original motion anchors, nearest-even relative progress, and terminal evidence. Absent/null optional fields retain their received representation. A supplied malformed snapshot fails; unavailable and genuinely empty evidence are distinct.

Model 3 actor requests require `controlFeedback`. Only pre-arm warmup may supply explicit null; warmup restores feedback state with recurrence/RNG/counters and emits no collection record. Actual observations require complete feedback. Collection record version 2 echoes the original object. The separately retained collector snapshot requires `observationSchemaVersion: 2`; equality is checked before acquiring frames and again on admission. Model 2 retains collection version 1 and rejects queue fields.

`ObservationRecord` owns canonical feedback bytes and an explicit observation schema. PPO likelihood/value replay, full prefixes, burn-in and bootstrap prepare tensors from those original bytes. They never read current executor state, final receipts or resampled actions. Feedback participates in metadata budgets and the checksummed decisions artifact. Dataset and rollout manifests add `observationSchemaVersion: 2` only for model 3; existing container versions retain their separate multi-source/review/batch meanings.

Retrospective source trajectories hash the complete original decisions, including queue evidence. Reward materialization changes only the reward field and preserves that evidence. Derived source identity also compares observation schema. Model 3 archive loading preserves and validates original review endpoints and retrospective metadata. Combined fragments carry the observation schema and independently authenticate every source; continuation requires the exact original checkpoint/model and actor cursor. No queue is reconstructed from receipts or review judgments.

## Causal practice and demonstrations

Practice has a separate bounded admitted-packet ledger and duration-end heap. Original `DecisionContext` packet/run/observation IDs survive admission. Semantic motion progress advances only after each actual simulator post/no-op, including intermediate interpolation; original relative endpoints remain unchanged. End sentinels share native ordering but stay outside the oracle's pending-command predicate. Empty packets remain visible until their true scheduled end, then until terminal acknowledgement.

Each half-open step processes work before its next cutoff. A sample due exactly at that cutoff can remain pending. The next observation is captured before current policy/oracle output and before terminal cleanup; time-limit bootstrap retains its original pending plans. Repeated snapshot reads do not consume history. Schema 3 stationary motion records actual simulator posts, matching native motion semantics; the legacy simulator keeps its original no-op path and unchanged configuration signature. Both practice BC and closed-loop evaluation opt in explicitly with model 3.

Human BC never derives queue evidence from lead-shifted labels. The reader validates an optional sealed recorder exclusion proof, including identity, paired end/join seal, Int64 clock bounds and first-frame/stop bounds. Only cutoffs within a complete, unrecovered recording's covered interval become known empty. Missing or recovered proof remains unavailable. No native control epoch is invented; final join metadata is an offline integrity gate, never a future neural feature.

## Focused checks

- Seventeen selected CPU checks passed for original packet IDs, empty duration ends, half-open boundaries, interleaved motion progress, no-op release, immutable pre-cleanup truncation input, dropped/forged queue history, snapshot acquisition ordering, exclusion proof, and legacy serialization omission.
- Two real small-model checks passed in 3.5 seconds: a three-decision practice rollout preserved original packet commands, archived/reloaded feedback and frame metadata matched byte-for-byte, original/imported recurrent likelihood and value replay matched (ratio one within tolerance), and one PPO update changed weights. A real native mapped-frame fixture passed the compiled model 3 actor, null warmup/restoration, first real observation and immutable collection echo.
- The generated native recording/dataset check changed a physical key-up inside the future lead-shifted label interval. Teacher commands changed while the current 178 actual-control features and every queue tensor stayed identical. Missing provenance was unavailable; complete exclusion was empty. A fixture initially chose episodes by random UUID order; selecting the earliest actual cutoff fixed that test's unstable assumption. The final focused proof check passes.

## Full generated desktop host

Command:

```sh
PYTHONPATH=python .venv/bin/python scripts/qualify_desktop_host.py \
  --queued-control --output .local/verification/desktop-host-queued-20260927-r1
```

The first run passed. It used model 3 `test_small`, 100 ms period and 200 ms lead, real source actor/collector/learner processes, generated pixels and the production `InputExecutor` with a virtual backend. The injected transport used actual negotiation, fresh executor epochs and feedback cursors.

The saved report records eight admitted decisions across two control epochs and two PPO optimizer updates with changed checkpoint weights. Original archive queue objects equal native snapshots; every retained packet equals the original submitted packet. There were ten outstanding packet-row observations, two rows with actual completed command progress and two terminal rows. Original frame IDs, metadata and timestamps were preserved. Actor and collector exited zero; both control owners, generated capture and cleanup joined.

Evidence stays private under `.local/verification/desktop-host-queued-20260927-r1`: `desktop-host-report.json`, `swift-test.log`, immutable `Library/DesktopRuns` collections and initial/learned checkpoints. No physical input, personal pixels or privacy permissions were used. This run does not qualify installed physical control, production-model deadline/memory capacity, maximum packet/lead combinations, or learning quality. Those remain separate gates.
