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
| Independence | Baseline Git commit precedes original audit; audit coverage and reconciliation | Pending |
| Original integrity | Original source/Git state unchanged by Astra work | Pending |
| Build | Clean native and Python build; reproducible locked dependencies | Pending |
| Packaging | Installed app has all MLX/native/Metal resources; no developer-machine runtime dependencies | Pending |
| Capture/control | Window/app/display/desktop sources; geometry; monotonic timing; actual input effects | Pending |
| Permissions | Denied/granted/revoked behavior and installed-bundle helper attribution | Pending |
| Cleanup | UI/helper/actor/learner failures, emergency stop, takeover, sleep; no remaining owned holds or stale commands | Pending |
| Storage | Exact frame/input round-trip; partial-write recovery; corruption; low disk; bounded queues | Pending |
| Datasets | Causal canonicalization, high-rate pointer handling, session splits, immutable selections/configuration | Pending |
| Model | Correct converted weights, shapes, finite gradients, step/sequence/reset/padding checks | Pending |
| BC | Tiny-data overfit plus representative held-out and closed-loop learning | Pending |
| PPO | Ratio-one, joint likelihood, masking, GAE timing/outcomes, recurrent state, real parameter improvement | Pending |
| RL workflows | Fresh-policy and BC-initialized runs; manual and validated automatic reset; live target continuity | Pending |
| Checkpoints | Atomic write/integrity; reload/resume; runtime policy version and recurrent isolation | Pending |
| Evaluation | Frozen protocols/checkpoints; held-out seeds/layouts; action faults and interventions | Pending |
| Experiments | Three-seed detail/temporal/cadence/hybrid comparisons and evidence-based decisions | Pending |
| UX | Both first-success paths, reusable library ownership, clear stops/takeover, errors/empty states | Pending |
| Accessibility | Keyboard, VoiceOver, contrast, chart tables, light/dark, narrow windows/long names | Pending |
| Performance | Warm end-to-end latency, GPU contention, aggregate memory, capture/storage bottlenecks | Pending |
| Full workflows | Installed bundle recording -> training -> evaluation -> local inference and RL | Pending |
| Final review | Full engineering/product reviews; findings fixed; relevant checks passing | Pending |
| GitHub | New public UnendlessGit/AgentTrainer-Astra history, completed source pushed to main | Pending |
| DMG | Ad-hoc signatures, blank icon, Applications alias, mounted/copied/installed clean-build verification | Pending |

## Scope and defaults

- Native macOS/Apple Silicon application. Deployment target macOS 15; report actual tested versions accurately.
- Fresh Astra formats and models; no legacy recording/model import requirement.
- Local-only core workflows. Build-time pretrained downloads are pinned and bundled for offline operation.
- Public source repository; private recordings, real user data, model artifacts, and audit copies stay out of Git.
- Personal installation with ad-hoc signing, no notarization requirement. Document real Gatekeeper/TCC limitations.
- Final DMG: `dist/AgentTrainer-Astra-1.0.0-arm64.dmg`.
- No automatic update service. Signed app contents remain read-only.

## Current state

The independent baseline has been written. Its first commit and the original audit are the next work. No application or learning workflow has been implemented or verified yet.
