# Shared recording links and training intervals

A recording is one immutable source session, regardless of how many agents use it or how many intervals they select. The library now supports adding an existing recording to an agent, removing that link, and saving multiple non-overlapping training intervals per link. Duplication copies the agent's links and selections without copying source files. The recording's original creator remains provenance; startup recovery no longer recreates an intentionally removed creator link.

## Storage and compute contract

The native catalog migrates from schema 1 to 2 in one SQLite transaction. `agent_recordings.selection` is a nullable JSON blob: null selects the whole usable prefix; an explicit `RecordingTrainingSelection` contains schemaVersion 1 and ordered ranges with UI identity UUIDs plus absolute `startNanos`/`endNanos`. Each interval is half-open. Adjacent intervals may remain separate; empty, reversed, overlapping, unordered, non-finite or out-of-coverage edits are rejected. Source coverage ends at the earliest of stoppedNanos and firstInvalidObservedNanos. Invalid saved metadata stays on disk, appears as a catalog issue, and cannot silently become whole-recording supervision.

`dataset.prepare.selections` contains each source UUID exactly once. Its new form is:

```json
{
  "recording_id": "<source UUID>",
  "ranges": [
    {"start_nanos": 1000000000, "end_nanos": 2000000000},
    {"start_nanos": 3000000000, "end_nanos": 4000000000}
  ],
  "context_ids": []
}
```

The legacy form with optional start_nanos/end_nanos remains accepted. Mixing those keys with ranges is rejected. UI UUIDs do not enter the model or dataset contract. Context vocabulary semantics are unchanged.

New dataset revisions use container schema 2, independently of canonicalizer version 2. Each source keeps one session split and one canonical selection with resolved absolute ranges. `labelPartitions` contains one existing partition report per range. The episodes table adds `selection_index`, linking each episode to that range. Geometry changes can still create additional episodes within a range. The reader validates episode bounds, index identity and partition provenance and continues reading container schema 1 without rewriting it.

Sorted ranges share monotonic observation, label and frame cursors. Events in an excluded gap update causal physical held state and pointer state but do not become labels. At each range start, transient accumulators cover only the preceding decision interval, and a new recurrent episode starts at step zero. Thus gaps cannot carry model recurrence or leak labels; adjacent ranges also preserve the explicit episode boundary. Every range remains in its source session's split, preventing selection-level train/validation leakage.

Limits remain explicit: 256 intervals per source, 4,096 sources per native/IPC job, 100,000 selected intervals and at most 100,000 resulting episodes. The 1 MiB wire envelope, 8 MiB artifact metadata limit, and 64 KiB native selection limit remain enforced. The builder preserves bounded reader/cursor ownership and fails the entire staged revision when any source or action fit is invalid.

Behavioral training captures the chosen metadata before launching its task and persists sourceSelections in the run configuration. Later link or interval edits do not change that run. Exact resume uses its original immutable dataset/configuration, including after the source is unlinked from the agent.

Automatic control discovery uses the selected source-time intervals and refreshes when saved ranges change. Excluded keys and reconciliation snapshots do not expand the learned action vocabulary; selected clicks still enable the pointer capability needed for their position anchors. Interval membership uses binary search because source timestamps can arrive out of order relative to observation sequence.

## Product workflow

Use **Add from Library** in an agent's Demonstrations tab, or **Use with Agent** from the shared library. **Review Recording** in an agent opens its saved training selection. Enter relative seconds, use **Set Start/Set End** at the preview time, and add or remove intervals. **Save Selection** commits the draft. Closing an unsaved edit asks whether to discard it. Whole usable recording remains the default, and the list/training form displays the included duration. The editor does not modify raw frames, input events or recording manifests.

Closing/cancelling a sheet is disabled while its save is publishing, using state set synchronously when the action starts. Application termination also waits for active catalog saves, avoiding a misleading discard while the edit is committing.

## Evidence and remaining qualification

The focused native suite exercises independent agent selections, clone/unlink/recovery persistence, byte-preserved source files, atomic link/migration failures, visible corrupt selections, strict coverage, exact nanosecond preservation in untouched UI fields, immutable job snapshots and resume after unlink. Python tests exercise shared source splits, pre-range held state, transient reset, excluded gap labels, adjacent range resets, range-index tampering, no partial publication, real dataset.prepare-to-training jobs, and real interrupted-checkpoint optimizer/sampler resume against a schema-1 dataset.

On 2026-09-12, the combined canonicalization, data preparation, dataset, job and protocol campaign passed **86 Python tests with warnings treated as errors** in 24.80 seconds. The native selection/catalog/coordinator focus passed **26 tests** in 6.618 seconds, without compiler warnings. Logs are `.local/selection-data-final.log` and `.local/selection-native-final5.log`; ordinary native exports rebuild the actual app/test host. Earlier native reruns stopped on concurrently edited inference code and an invalid flags-event fixture; both were corrected and the complete focus was rerun.

Owned NSHostingView renders cover the actual interval editor, link sheet, agent demonstrations and shared library in light/dark at 1120×760 and 860×580. The first render exposed excess preview height at the minimum size; the preview now adapts to available height, keeping the two-range editor visible. The agent list prioritizes training selection over the redundant frame-count column. Artifacts remain in `.local/selection-ui-renders2`, with 16 base renders and four scrolled views. These are generated-fixture renders, not OS screenshots or keyboard/VoiceOver qualification. The existing offscreen selected-sidebar-row rendering artifact remains separate from actual app behavior.

Live recording, installed-bundle control, physical input, privacy lifecycle, context changes within recordings, correction recording and library relocation remain separate product work. This change does not claim the full application is complete.
