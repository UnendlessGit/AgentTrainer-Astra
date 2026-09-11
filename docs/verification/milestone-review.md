# Current milestone engineering review — 2026-09-11

This is a bounded integration review before the next source milestone, not a final product/release audit. Original AgentTrainer and Git history were not changed. No privacy grants, real input posting, UI automation, model training, or frozen rebuild ran during this review.

## Scope

Read the current `RecordingReader`, `LibraryStore`, `LearningCoordinator`, `InferenceCoordinator`, `WorkspaceModel`, `ComputeProcess`, app quit wiring, associated native recording/learning/inference/process tests, `build_app.py`, `run_app.py`, `check_worker.py`, `check_inference.py`, `check_native_learning.py`, and the compute packaging specification. Inspected the Python job request/checkpoint-loading boundaries relevant to native resume and identity checks. This does not claim a new complete audit of all model, PPO, capture, storage-recovery, or experiment code.

## Defect found and corrected

**P1 — forced compute teardown also killed unsettled desktop control.** `InferenceDependencies.live` uses `ComputeProcess` for both actor and control runtimes. Its generic cancellation/timeout/shutdown path previously escalated from TERM to KILL after one second. The control helper deliberately remains alive, retaining its input ledger and desktop lock, when a post or release is still unsettled. Generic escalation could destroy that owner and allow a false clean-release presentation.

After coordination, the correction was implemented in the owning runtime files:

- `ComputeProcess` retains bounded TERM/KILL behavior for compute and actor roles. Control roles receive cooperative TERM only and are joined through actual process/I/O exit. The destructor likewise sends only TERM.
- Joined terminal status is exposed through `InferenceRuntime.shutdown`. A nonzero/missing control exit after an arm attempt records unconfirmed cleanup, preserves any original failure, and writes `cleanupConfirmed: false` into the immutable run result. Runs that never attempted control admission remain harmless.
- While a control helper is alive, the app remains responsive in “Waiting for owned controls to release…” rather than declaring the run finished. Early error messages no longer claim release before teardown is proved.
- If Quit encounters an unclean control exit, the first quit is declined and a manual-release warning is presented. A repeated quit may close the app without changing the unconfirmed-cleanup evidence. The warning is tracked per run to avoid trapping the user in an unclosable app.

This correction does **not** implement recovery of actual OS-held state after forced helper death, power loss, or an OS failure. Durable control-state journaling and installed physical-input qualification remain separate gates. Rebuilding an app whose quit was declined must continue to refuse replacement until it actually exits.

## Verification

Permission-free process fixtures simulate an unsettled owned hold with an exclusive file lock. Shutdown returning `control.cleanupPending`, request cancellation, and request timeout all leave the child alive and the lock held beyond the former kill deadline. Releasing an explicit test gate then proves cleanup, process exit, pipe join, and lock availability. Existing actor fixtures still prove bounded forced termination.

The inference fixture covers an unclean control exit during shutdown, false-release prevention in both UI state and saved results, and first-quit warning followed by a permitted repeat quit. It also preserves the existing real mapped-ring lifecycle checks, late-arm disarm, warmup qualification, identity rejection, receipts, and physical takeover behavior.

Targeted recording/catalog/resume checks exercise sealed frame identity and corruption, atomic immutable publication, copied-agent shared checkpoint ownership, exact saved behavioral resume configuration, read-only capability discovery, and library locking. These use protocol fixtures and native storage; they do not substitute for real model learning or final installed-bundle workflows.

Exact commands and results are retained locally in `.local/control-exit-proof-review.log` and `.local/milestone-storage-resume-review.log`. No test was disabled or weakened to accommodate the defect. The build performed by these targeted Swift tests produced no compiler warnings.

## Other reviewed boundaries

- Recording previews take a shared package lock, validate sealed manifest/index counts, compare frame identity and timestamps against the index, and verify decoded frame blocks before displaying pixels. Corrupt frames fail visibly rather than becoming training evidence.
- Shared checkpoints retain immutable creator/model identity; duplicating an agent adds references instead of overwriting weights or creator metadata. Resume reads the selected checkpoint's original run identity and canonical saved training configuration, preserving its dataset and target.
- Learning callbacks are scoped by process generation and job/request/run identity. Terminal publication joins outstanding metadata persistence; late closing-child callbacks cannot overwrite a new session's state.
- Inference retains published image leases until the actor acknowledges them or its process has exited. Warmup uses the actual observation/decoder path and rejects inadequate timing; packets are not retimed after a missed deadline.
- Build scripts stage a new ad-hoc-signed bundle, verify bundled resources, and refuse replacement while the development app is running. Frozen diagnostics and small actor IPC checks are scoped as packaging/integration evidence, not broad learning-quality evidence.

No additional concrete P1 defect was established in the reviewed storage/resume/packaging paths. A clean frozen build, real control and permission attribution, crash-state recovery, broader BC/RL learning evidence, performance experiments, and the final DMG remain open requirements.
