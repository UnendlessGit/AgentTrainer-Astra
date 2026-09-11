# Implementation plan and evidence gates

The user's full requested scope remains active. A passing component check or a partial vertical slice is not completion. Track evidence here as it is produced; do not replace unimplemented requirements with a smaller definition of success.

## Sequence

1. **Independent baseline:** commit ADR 0001 before opening original implementation or documentation.
2. **Original audit:** comprehensively read original docs/release history/tests/source, record coverage and product lessons, and reconcile the baseline using evidence. Leave the original untouched.
3. **Platform foundation:** build/install the native app and packaged MLX helper; prove capture, permissions, timing, shared-memory transport, native input, and cleanup.
4. **Data foundation:** complete recording/recovery/replay, shared libraries, selections, dataset revisions, and checkpoints.
5. **Learning workflows:** complete BC -> evaluation -> inference and fresh-policy RL through fixtures and real native capture/control. Exercise paths while implementing.
6. **Product completion:** agent-centered interface, rewards/reset rehearsal, correction recording, comparison, advanced configuration, recovery and accessibility.
7. **Experiments/performance:** run the ADR comparisons and select measured defaults; diagnose numerical, learning, scheduling and resource failures.
8. **Final review/release:** independent engineering/ML/RL/UX reviews, fix findings, clean build, installed-DMG workflow checks, public source push to main, final report.

## Acceptance evidence required

| Area | Required evidence | Status |
|---|---|---|
| Independence | Baseline Git commit precedes original audit; audit coverage and reconciliation | Baseline 505ec37; complete audit mapped in LEGACY_AUDIT.md; ADR 0002 |
| Original integrity | Original source/Git state unchanged by Astra work | 57 tracked hashes match, HEAD unchanged and Git clean after audit; recheck at release |
| Build | Clean native and Python build; reproducible locked dependencies | Locked development build; 250 Python checks with warnings treated as errors, 109 active Swift checks plus two separately exercised opt-in workflows passed on 2026-09-11. Final clean release pending |
| Packaging | Installed app has all MLX/native/Metal resources; no developer-machine runtime dependencies | Current frozen compute/actor and native control checks pass offline in the signed development bundle. Native coordinator completed production-model training/checkpoint/evaluation through bundled helper. Installed full workflows pending |
| Capture/control | Window/app/display/desktop sources; geometry; monotonic timing; actual input effects | Native capture/input/scope components implemented; geometry, routing, timing and pixel-copy checks pass. Live-source and actual-control qualification pending |
| Permissions | Denied/granted/revoked behavior and installed-bundle helper attribution | Pending |
| Cleanup | UI/helper/actor/learner failures, emergency stop, takeover, sleep; no remaining owned holds or stale commands | Virtual backend and child-process fault campaigns pass. Control shutdown retains unsettled ownership; joined exit status and persisted warnings prevent false release claims. Reciprocal forced-helper-death recovery and actual OS effects pending |
| Storage | Exact frame/input round-trip; partial-write recovery; corruption; low disk; bounded queues | Native recovery/storage checks and byte-exact Swift-to-Python raw/LZFSE fixtures pass; see verification/recording-storage.md. Live recording and storage-fault campaigns pending |
| Datasets | Causal canonicalization, high-rate pointer handling, session splits, immutable selections/configuration | Version-2 causal canonicalization, immutable dataset revisions, session splits, contiguous sampling and native/Python readers tested. High-rate motion qualification, multiple surfaces, selection/context editor pending |
| Model | Correct converted weights, shapes, finite gradients, step/sequence/reset/padding checks | Default 34.64M policy and conversion checks pass; finite production-shape backward verified. See verification/visual-policy.md; learning and integration remain separate gates |
| BC | Tiny-data overfit plus representative held-out and closed-loop learning | Production pointing learned 6/6 limited closed-loop trials. Native training/checkpoint/evaluation and resume verified; memory task achieved only 17/32 held-out successes. Broad learning gate remains open |
| PPO | Ratio-one, joint likelihood, masking, GAE timing/outcomes, recurrent state, real parameter improvement | Exact-math/recurrent replay tests, staged gradients, complete-episode disk spool and KL-admitted optimizer updates pass. Production admission/cancel/resume passed before latest collector change; fresh production collector qualification and learning improvement pending |
| RL workflows | Fresh-policy and BC-initialized runs; manual and validated automatic reset; live target continuity | Native practice PPO configuration, jobs, metrics and checkpoints integrated. External desktop adapter, authored reward/reset workflows and asynchronous live continuity pending |
| Checkpoints | Atomic write/integrity; reload/resume; runtime policy version and recurrent isolation | Immutable publication/integrity, optimizer/freeze/RNG resume, shared catalog links, saved-configuration resume and actor identity/reset tests pass. Actual desktop policy handoff remains part of live qualification |
| Evaluation | Frozen protocols/checkpoints; held-out seeds/layouts; action faults and interventions | Native checkpoint NLL evaluation and compiled-policy virtual closed-loop evaluation implemented; production pointing/memory evidence recorded. Broader task protocols and comparison UI pending |
| Experiments | Three-seed detail/temporal/cadence/hybrid comparisons and evidence-based decisions | Padding and staged-backward fixes measured; compiled actor comparison measured. Baseline delayed memory below target; matched recurrent-initialization comparison in progress. Required ablations remain open |
| UX | Both first-success paths, reusable library ownership, clear stops/takeover, errors/empty states | Pending |
| Accessibility | Keyboard, VoiceOver, contrast, chart tables, light/dark, narrow windows/long names | 56 actual native base-view renders plus 20 scrolled views in both themes/sizes inspected; contrast, Run spacing and recording-inspector clipping fixed. Actual keyboard/VoiceOver interaction pending |
| Performance | Warm end-to-end latency, GPU contention, aggregate memory, capture/storage bottlenecks | Production compiled actor median 58.51 ms / p95 59.58 ms; exact B2T64 backward peak 8.10 GB. These exclude full native capture/IPC and aggregate contention; full performance gate pending |
| Full workflows | Installed bundle recording -> training -> evaluation -> local inference and RL | Pending |
| Final review | Full engineering/product reviews; findings fixed; relevant checks passing | Pending |
| GitHub | New public UnendlessGit/AgentTrainer-Astra history, completed source pushed to main | Independent history and recording/model milestone d2f249f pushed; completed source/release pending |
| DMG | Ad-hoc signatures, blank icon, Applications alias, mounted/copied/installed clean-build verification | Pending |

## Scope and defaults

- Native macOS/Apple Silicon application. Deployment target macOS 15; report actual tested versions accurately.
- Fresh Astra formats and models; no legacy recording/model import requirement.
- Local-only core workflows. Build-time pretrained downloads are pinned and bundled for offline operation.
- Public source repository; private recordings, real user data, model artifacts, and audit copies stay out of Git.
- Personal installation with ad-hoc signing, no notarization requirement. Document real Gatekeeper/TCC limitations.
- Final DMG: `dist/AgentTrainer-Astra-1.0.0-arm64.dmg`.
- No automatic update service. Signed app contents remain read-only.

## Current state — 2026-09-11

The independent baseline and complete original audit are committed/pushed (505ec37, cdd0f4c); do not restart them. Original integrity was rechecked: all 57 source hashes, HEAD and clean Git state remain unchanged. The next integration milestone adds real native behavioral/PPO jobs, checkpoint evaluation/resume, compiled local inference, the scoped control helper, leased frame transport, recording inspection and reproducible build/run checks. [ADR 0003](architecture/0003-qualification-decisions.md) records measured design refinements; [milestone review](verification/milestone-review.md) distinguishes corrected defects from unqualified platform behavior.

The principal remaining implementation work is general desktop RL, reward/reset authoring and rehearsal, reciprocal control crash recovery, multiple-surface capture/data, context/selection/correction workflows, evaluation comparison and reusable library management. The principal evidence gaps are long-delay learning, broader three-seed ablations, sustained storage/performance, real capture/control/privacy lifecycle, keyboard/VoiceOver and installed-DMG workflows. See each verification document for exact evidence and reproduction commands.

Permission-free practice learning and owned native view rendering advance real implementation checks without reading personal screen content or posting OS input. They do not replace final installed macOS qualification. The user will handle necessary privacy grants for a stable build. No full recording/learning/control release gate or DMG is complete.
