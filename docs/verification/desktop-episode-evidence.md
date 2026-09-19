# Native desktop episode evidence bridge

Verified September 19, 2026 with Xcode 27 / Swift 6.4. This increment connects the existing owned observation/actor collection events, `RewardAnalysisQueue`, and `CollectorSession`. It does not capture screens, execute controls, run a model, change `InferenceCoordinator`, or provide the complete desktop learning loop.

`DesktopEvidenceIdentity` binds actor run, clock, environment, and the two producer UUIDs. A shared `DesktopEnvironmentSequence` retains environment-message ordering across physical episodes, collector rotations and checkpoint activation. Each `DesktopEpisodeEvidence` captures one immutable episode/generation, policy ID/signature, scope and collector. Terminal and fault callbacks include their original generation; a callback offered to another generation cannot enter its audit or fault that new episode.

## Integration API

1. Prepare the collector and one shared environment-sequence owner. `DesktopEpisodeEvidence.prepare` warms the existing reward queue with the chosen program, scope and assets before controls start. Tests inject a CPU detector, rather than replacing the queue/evaluator.
2. Await `confirmReady(ResetResult)`. It requires the matching successful reset/context, cleanup proof, exact reward definition, ready snapshot, and post-cleanup coverage of every source.
3. Pass real `InferenceCollectionEvent`s to `offer(_:generation:)`. Observations and actor responses may arrive independently. The bridge pairs their original IDs/cutoffs/frames/state bindings and forwards the unchanged response. Schema-1 packet sequence must equal sampler draw index and continue between produced results. The bridge does not reconstruct policy categories or sample another action.
4. For programs with manual signals/markers, call `offerManualSeal` only when the input producer has completed its barrier for that snapshot. The baseline may carry manual state readings but has no policy interval, markers or marker-coverage window. Subsequent missing coverage stays unknown. Explicitly sealed unknown reward is encoded as null, and unknown outcome remains unknown; Python admission rejects it after its watermark.
5. `requestAuditAbort` switches a declared collection to its existing validated audit-only drain. An undeclared empty episode retains the request locally, so it cannot discard earlier valid episodes sharing that collector. Delayed manual seals outside learning after a terminal or abort do not create rewards or invalidate an otherwise complete audit.
6. After actor, control I/O and the manual producer have joined, call `finish(joined:)` with genuine cleanup and prediction-resolution proof, the exact last **produced** global packet sequence, physical stop time, and stop disposition. This awaits accepted reward work and closes only this episode. The parent owns the shared collector's eventual `finish` or `abandon` call.

The first actual actor cutoff initializes the reward baseline; reset readiness and warmup do not create a reward interval. Interval start identifies the original policy packet. Reward, outcome and watermark are emitted together under the shared environment-sequence lock, with an exact `throughSequence` for the preceding emitted outcome. Original nanosecond cutoffs are preserved.

Raw control evidence enters the collector's independent native audit before pairing/forwarding and has reserved headroom separate from actor-image pressure. Final execution evidence can arrive before its actor record. Inactive, rejected and actually posted suffix effects stay unchanged. Cancelled receipts are held until joined stop intent is known: `episodeBoundary` attribution requires the matching terminal cutoff and a native requested stop. Administrative cancellation is never promoted to a semantic terminal. Python retains final responsibility for per-command schedules, execution validity and PPO prefix admission.

Offers reserve bounded pending memory/items and do not wait for Vision, disk or learning. Owned images stay charged while awaiting actor pairing or manual coverage, and transfer to the existing collector/reward queues before their reservations are released. Episode metadata has a fixed decision capacity, and pending pairing/manual coverage has a wall-time bound. Retirement releases retained observation/response/feedback data and identity-history maps.

A valid zero-actor join returns `.empty`: no fabricated begin, end, decision or actor progress. Unpaired produced results, unresolved predictions, absent final receipts, or unconfirmed cleanup cannot become an empty or completed episode. A fault leaves whole-collector disposal to its owner so the independent native audit can drain.

## Verification

`./script/swift.sh test --filter desktop` built successfully and passed nine bridge tests plus the existing desktop control-lock test, without warnings. The bridge tests cover:

- original response identity and actual interval mapping while the detector is blocked;
- receipts arriving before actor pairing, inactive terminal suffixes and exact watermark prefixes;
- explicit manual gaps remaining null, and delayed ineligible manual seals after terminal/abort;
- produced counters including a rejected final packet;
- several physical episodes in one collector and global environment sequences across collector rotations;
- an empty intervening episode preserving earlier valid data;
- stale-generation callbacks leaving the current episode/collector untouched;
- final raw-control audit retention after observation backpressure;
- rejection of unconfirmed cleanup.

These are injected native protocol/lifecycle tests with the real collector session, frame-ring publication, reward queue and evaluator. They use neither a GPU nor OS input/capture. Installed target collection, real manual-feedback ownership, the persistent actor/reset loop, and end-to-end desktop PPO remain separate integration gates.
