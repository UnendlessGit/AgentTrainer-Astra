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
| Build | Clean native and Python build; reproducible locked dependencies | Sep12 integrated source passed305 Python checks with warnings as errors and154 active native checks (two opt-in skips); an additional early-catalog migration regression and existing migration checks passed separately. Fresh frozen build and final release pending |
| Packaging | Installed app has all MLX/native/Metal resources; no developer-machine runtime dependencies | Current frozen compute/actor and native control checks pass offline in the signed development bundle. Native coordinator completed production-model training/checkpoint/evaluation through bundled helper. Installed full workflows pending |
| Capture/control | Window/app/display/desktop sources; geometry; monotonic timing; actual input effects | Native capture/input/scope components implemented; geometry, routing, timing and pixel-copy checks pass. Live-source and actual-control qualification pending |
| Permissions | Denied/granted/revoked behavior and installed-bundle helper attribution | Pending |
| Cleanup | UI/helper/actor/learner failures, emergency stop, takeover, sleep; no remaining owned holds or stale commands | Paired guardian, atomic shared ledger, inherited desktop lease, startup-race fixes, Secure Input trust and joined host proof implemented. Real-process/virtual-input fault campaigns and integrated native suite pass. Actual installed OS effects/privacy lifecycle remain pending |
| Storage | Exact frame/input round-trip; partial-write recovery; corruption; low disk; bounded queues | Native recovery/storage checks and byte-exact Swift-to-Python raw/LZFSE fixtures pass; see verification/recording-storage.md. Live recording and storage-fault campaigns pending |
| Datasets | Causal canonicalization, high-rate pointer handling, session splits, immutable selections/configuration | Causal canonicalizer2 and dataset schema2 tested; shared recording links, multiple per-agent ranges, one session split per recording, selection-aware control discovery and schema1 exact resume implemented. Multiple-surface datasets, context authoring and high-rate live motion qualification remain open |
| Model | Correct converted weights, shapes, finite gradients, step/sequence/reset/padding checks | Default 34.64M policy and conversion checks pass; finite production-shape backward verified. See verification/visual-policy.md; learning and integration remain separate gates |
| BC | Tiny-data overfit plus representative held-out and closed-loop learning | Production pointing learned 6/6 limited closed-loop trials. Native training/checkpoint/evaluation and resume verified; memory task achieved only 17/32 held-out successes. Broad learning gate remains open |
| PPO | Ratio-one, joint likelihood, masking, GAE timing/outcomes, recurrent state, real parameter improvement | Exact-math/recurrent replay tests, staged gradients, complete-episode disk spool and KL-admitted optimizer updates pass. Production admission/cancel/resume passed before latest collector change; fresh production collector qualification and learning improvement pending |
| RL workflows | Fresh-policy and BC-initialized runs; manual and validated automatic reset; live target continuity | Practice PPO, native reward authoring/rehearsal, external adapter, asynchronous evidence assembly, exact actor collection records and native nonblocking hooks implemented/tested. Native worker/producer wiring, live actor/learner scheduling, checkpoint actor-progress handoff and reset actions pending |
| Checkpoints | Atomic write/integrity; reload/resume; runtime policy version and recurrent isolation | Immutable publication/integrity, optimizer/freeze/RNG resume, shared catalog links, saved-configuration resume and actor identity/reset tests pass. Actual desktop policy handoff remains part of live qualification |
| Evaluation | Frozen protocols/checkpoints; held-out seeds/layouts; action faults and interventions | Native checkpoint NLL evaluation and compiled-policy virtual closed-loop evaluation implemented; production pointing/memory evidence recorded. Broader task protocols and comparison UI pending |
| Experiments | Three-seed detail/temporal/cadence/hybrid comparisons and evidence-based decisions | Matched initialization and192-update paired GRU diagnostics remain at chance; tiny immediate-cue readout can overfit. Experimental4×384 causal transformer passed22 checks and full-resolution frozen-feature pilot; no campaign/default change. Broad learning and required multi-seed ablations remain open |
| UX | Both first-success paths, reusable library ownership, clear stops/takeover, errors/empty states | Pending |
| Accessibility | Keyboard, VoiceOver, contrast, chart tables, light/dark, narrow windows/long names | Owned native renders cover both themes/sizes:72 base+30 scrolled overall views, plus16 base+4 scrolled selection views. Layout/contrast fixes inspected. Actual keyboard/VoiceOver and installed interaction remain pending |
| Performance | Warm end-to-end latency, GPU contention, aggregate memory, capture/storage bottlenecks | Production compiled actor median 58.51 ms / p95 59.58 ms; exact B2T64 backward peak 8.10 GB. These exclude full native capture/IPC and aggregate contention; full performance gate pending |
| Full workflows | Installed bundle recording -> training -> evaluation -> local inference and RL | Pending |
| Final review | Full engineering/product reviews; findings fixed; relevant checks passing | Pending |
| GitHub | New public UnendlessGit/AgentTrainer-Astra history, completed source pushed to main | Independent history and native learning/PPO/inference milestone8eac37a pushed; completed source/release pending |
| DMG | Ad-hoc signatures, blank icon, Applications alias, mounted/copied/installed clean-build verification | Hardened packaging tools and8 fault checks pass; a private old-build trial mounted/relocated/validated/detached. This is not the release artifact. Fresh-source installed runtime and final DMG pending |

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
