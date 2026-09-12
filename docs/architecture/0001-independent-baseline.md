# ADR 0001: Independent architecture baseline

Status: accepted implementation baseline, subject to recorded experimental evidence.

## Provenance and sequencing

This record is written before reading the original AgentTrainer documentation, release notes, tests, UI, model, training, or inference implementation. The planning investigation inspected only the empty Astra workspace, host tooling/hardware, GitHub access, and the original directory's existence. Independent model, RL, and native macOS reviewers did not inspect the original project.

First commit this record into Astra's new Git history. Only then perform the comprehensive original-project audit. Preserve this original decision record; later changes receive explicit reasoning and evidence rather than rewriting its provenance.

The user selected a fresh start without legacy recording/checkpoint imports, a public `UnendlessGit/AgentTrainer-Astra` repository, and an ad-hoc-signed personal-installation DMG. The original `/Users/endless/Documents/AgentTrainer` must not be modified.

## Product and hardware

The product is a general computer-interaction learning framework. Games, browsers, native applications, and the whole desktop are environments, not architectural special cases. Recording, imitation learning, reinforcement learning, evaluation, and trained local execution are all complete product workflows.

Observed development host: M3 Max, 14 CPU cores (10 performance, 4 efficiency), 30 GPU cores, 36 GB unified RAM; macOS 27.0; Xcode 26.6 and Swift 6.3.3. Approximately 243 GiB disk space was available during planning. No valid Developer ID signing identity was installed. Reinspect dynamic resource availability at runtime.

Target Apple Silicon and macOS 15+. Qualify actual installed-app behavior on available hardware; do not claim unperformed compatibility tests. Core workflows work offline without system Python, Homebrew, Xcode, or remote inference.

## Process architecture

1. Native SwiftUI application: domain coordination, ScreenCaptureKit, Metal preprocessing/preview, recording, SQLite persistence, permissions, and UI. AppKit is used where native platform behavior needs it. Background work stays off the main actor.
2. Native control helper: child process with an event tap, ordered execution scheduler, held-input ledger, deadline checking, physical takeover, and independent emergency cleanup.
3. Packaged Python MLX compute executable: actor and learner roles may run as separate child processes. Both use the same policy, action distribution, dataset semantics, and checkpoint code. Evaluation uses that same implementation.

Python is chosen for ML experimentation and reference validation, not a claim that Swift MLX lacks training support or GPU quality. Native code owns OS integration. A PyInstaller onedir macOS helper bundle is the initial packaging choice; explicitly collect MLX native libraries and Metal resources. All signed bundles are read-only. Runtime data lives outside the app and repository.

Use bounded, versioned messages over inherited local connections, with request/run identity, sequence numbers, acknowledgements, deadlines, and cancellation. Diagnostics are separate from protocol output. Do not send pixels as JSON or expose a local HTTP server for ordinary process coordination.

## Core contracts

- Environment adapter: observations, action capabilities, readiness, reward/outcome signals, episode reset, and closure. A real desktop and deterministic fixtures implement the same behavioral contracts. Only genuinely isolated adapters may be vectorized.
- Observation: timestamped surfaces with geometry and validity, cursor and actual held controls, executed-action history, elapsed time, and preprocessing identity.
- Action packet: capability-constrained timed commands and motion trajectories, immutable sampling identity, and a separate execution receipt.
- Episode/transition: demonstrations or rollouts with monotonically ordered timing, outcomes, interventions, policy identity, and masks/probabilities where applicable.
- Run specification: immutable environment, dataset, model, training, evaluation, and resource settings.
- Checkpoint manifest: parameter format, optimizer/RNG/sampler state, model/preprocessing/action versions, provenance, and integrity checks.

Freeze a shared protocol definition before delegating coupled subsystems. Reject unsupported versions and invalid capabilities explicitly.

## Capture, control, and ownership

Use ScreenCaptureKit with explicit SDR/sRGB pixel semantics, source dimensions, cursor metadata, and an initial queue depth of four. Release capture-owned surfaces promptly. Record display/window/process identity and launch generation, bounds, content rectangle, scale, and all source/global transforms. Do not assume Retina means 2x.

Convert capture Mach ticks and input-event timestamps to one native monotonic-nanosecond clock. Callback arrival time is not observation time. An idle/unchanged frame remains valid; blank, suspended, or failed capture is a different state.

Metal preprocesses IOSurfaces into a bounded shared-memory tensor ring. Publish only after GPU completion. Python takes a NumPy view and one explicit MLX ingestion copy, then acknowledges ownership before reuse. Do not claim cross-process zero-copy. Latest-first inference queues are bounded, while recording preserves gaps and input ordering explicitly.

Only one actor owns real desktop input. Native input is globally coupled through focus and pointer state. A selected-window environment must verify the target before posting events; whole-desktop operation is explicit. Tag synthetic events and distinguish physical input. Background synthetic control and protected content are not universally supported by macOS/apps.

The helper releases owned keys/buttons and cancels queued actions on parent EOF, expired lease, emergency stop, lost target, permission failure, and session/sleep interruption. The host performs reciprocal cleanup if the helper dies. Initial heartbeat 100 ms; maximum actuation lease 500 ms, verified under load. Validate capture/listen/post permissions from the installed native bundle rather than relying on Terminal/Xcode attribution.

## Recording, datasets, and checkpoints

Store immutable, independently addressable native-resolution lossless LZFSE frame blocks in bounded shards. SQLite indexes frames, events, episodes, annotations, and provenance. Preserve complete raw keyboard/mouse input; unchanged content can reference an existing frame with new timing. Strip alpha only when opaque. Lossy video may be an export/proxy but is not the sole default learning source.

Default capture is 30 fps, with explicit lower/higher-rate profiles. Measure compression throughput and disk rate during preflight. Display recording cost and remaining duration; reserve at least 10 GiB for finalization. Do not silently downscale, discard events, or hide overload. Safely finalize or pause when storage/backpressure requires it.

Append and flush frame data before committing index references. Integrity checking and recovery handle partial writes, missing/corrupt blocks, and interrupted recording. Exports use consistent database snapshots. Shared recordings are not duplicated when linked to agents. Trimming creates non-destructive agent-specific selections.

Derived datasets are immutable revisions keyed by source selections, preprocessing, action canonicalizer, timing, and schemas. Split by complete recording sessions/task instances, not neighboring frames. Default 80/10/10 where independent sessions permit it; expose insufficient validation data honestly. Cache only versioned deterministic features; changing trainable weights invalidates encoded-feature caches.

Checkpoint weights use safetensors and versioned metadata, with atomic publication and integrity checks. Include optimizer/master weights, RNG, sampler position, config, dependency versions, source hashes, and evaluation provenance. Live recurrent state belongs to a specific session and policy version. Training resume does not imply the desktop world can resume from a saved state.

## Policy model

Initial model: approximately 35M parameters, one implementation for BC/PPO/inference.

- ConvNeXt-Tiny pretrained global branch, full field at a maximum 768-pixel long edge.
- Compact residual detail branch, channels 32/64/128 and stride eight, full field at a maximum 1536-pixel long edge; additional 384x384 native-resolution crop around the observed cursor.
- Dense spatial feature fusion for pointing; 32 position-aware learned attention queries summarize features for the temporal core. Preserve surface metadata and valid-pixel masks.
- Two 512-unit GRU layers, normalized inputs/outputs, with elapsed time, actual cursor/held controls, executed-action history, and observation validity/age.
- Separate actor and value heads. No privileged reward-detector input to the actor.

These choices combine transferable features, spatial precision, and bounded persistent state. A globally pooled thumbnail discards small controls; cursor detail alone cannot locate distant targets. ConvNeXt is not assumed superior merely from image-classification scores. Verify the official pretrained conversion numerically, record source/license/hash, bundle needed weights, and retain a functional random-initialization path.

Disable dropout/stochastic depth by default. Keep losses, log probabilities, optimizer/master state, and reference checks FP32. Validate BF16 activations before enabling them. Budget approximately 24 GB maximum aggregate learning memory initially, reduced for system pressure and the target. Reduce microbatch size, accumulate gradients, and checkpoint activations before reducing sensory/temporal quality.

## Canonical actions and causality

An autoregressive packet represents one control interval. Initial cadence 10 Hz; qualify 20 Hz for tasks needing faster visual feedback. Packets contain ordered key/button down/up, repeat, absolute/relative motion knots, scroll, and END. Millisecond offsets are nondecreasing; equal-time order is stable. Key/button down/up are idempotent commands applied to the authoritative ledger at execution time. Repeats on unheld keys are logged no-ops. Masks depend on capability and syntax, never predicted future held state.

Pointing selects surface, dense spatial cell, and categorical within-cell coordinates. Relative and scroll arguments use exactly scored bounded categorical integer components. The distribution exposes sample, log_prob, entropy diagnostics, and teacher-forced scoring. Sum all active conditional log probabilities, including END, time, and arguments. A forced END at the grammar limit has probability one. Score sampled commands, not the executor's reduced physical-event list.

Raw high-polling mouse events remain intact. Canonicalize motion using time-aware piecewise-linear knots, preserving packet boundaries and all key/button/scroll anchors. Bound position error at the same time after quantization: initially one logical point absolute or one raw input count cumulative relative; millisecond rounding error <=0.5 ms. Relative interpolation uses remainder compensation. The native scheduler follows a 1 ms interpolation grid plus exact knot/control deadlines; actual OS jitter is measured separately, not guaranteed away.

Select the smallest budget in 16/32/64 that fits every canonical window at the chosen tolerance/cadence, then requalify latency. If needed, test a higher supported cadence and regenerate the entire dataset. If none fits, retain the source and identify incompatible intervals; never silently omit windows or loosen tolerances. Version all canonicalization/interpolation settings with datasets and models.

For observation cutoff t and period T, execute the packet in [t+L,t+L+T). Measure warmed end-to-end p99 under representative load; initial L = ceil_ms(p99 + max(5 ms, 0.2*p99)). Freeze T/L per run, condition the model on them, and use identical BC alignment. Commit complete packets before deadlines; never execute a late shifted/partial packet.

Actor input/history timestamps must be <=t. Do not select crops using future action targets or feed a prior shifted teacher packet containing future human input. Reset packet-decoder working state per packet so teacher forcing cannot contaminate persistent temporal memory. PPO transitions/rewards remain indexed on [t,t+T), not the delayed execution window. Delayed consequences are handled by trajectory returns.

## Behavioral training

Start with two episode lanes and 64-step truncated BPTT. Shuffle episodes, stream lanes contiguously, and carry detached recurrent state between chunks. Reset at true episode starts or explicit discontinuities, not arbitrary chunk boundaries. Teacher-force joint packet NLL averaged over valid intervals; mask padding, post-END fields, and unavailable data.

AdamW with bias correction explicitly enabled; new-layer LR 3e-4, pretrained LR 3e-5, gradient norm 1.0. Freeze pretrained vision for one epoch before fine-tuning. Avoid default inverse-frequency weights that distort action calibration. Monitor per-operation recall, false presses, releases, pointing and timing, alongside NLL. Support correction recordings and immutable dataset revisions.

## Reinforcement learning

Recurrent PPO is the initial auditable shared-policy baseline. It handles the mixed action distribution and behavioral initialization without requiring an unvalidated visual world model. Demonstration NLL is a separate configurable regularizer, never fabricated on-policy data.

Initial calibration: 512 valid decisions; 32 burn-in plus 64 loss steps; four epochs; effective batch 256 valid loss steps through accumulation; AdamW LR 1e-4 (1e-5 tuned pretrained vision), clip 0.2, value weight 0.5, gradient norm 0.5, entropy coefficient 0.01, sampled-KL guard 0.02. Use duration-aware discount with 30-second half-life and lambda .95 per 100 ms. These are experiment starting settings, not unmeasured claims of optimality.

Use exact joint ratios; never mean log probabilities or per-token PPO clipping. Normalize/weight the entropy regularizer separately to avoid rewarding long busy packets. Log packet length and ratio tails. Recomputed old-policy log probabilities must agree before the first optimizer update; no fresh random augmentation or batch-dependent normalization may alter that path.

Keep actor policy v immutable while learner updates a sealed v rollout. Actor continues controlling an unpaused live target. Additional v data is audit data, not the next PPO batch. Maintain at most one pending update; activate v+1 only at confirmed environmental reset with fresh recurrence. Pause learner work on repeated actor deadline misses. Separate processes do not guarantee GPU preemption.

Recurrent chunks begin from actual episode zero or saved behavior state plus burn-in. Burn-in is an approximation after parameter changes; compare against full-prefix replay. Keep terminated, truncated, and aborted distinct. Bootstrap collection limits from valid pre-reset state; never from a reset frame. Cancel pending commands at boundaries. Default episode watchdog 120 seconds is a collection truncation, not invented failure. Uninterrupted online policy replacement during unbounded inference is not promised.

Reward authoring: local OCR text/numbers, image matches, score deltas, rising-edge events, elapsed time, manual markers, and bounded all/any predicates. Missing/low-confidence OCR is unknown, not zero. Rehearse against recorded frames. Freeze definitions per run; reset signal baselines only after readiness.

Default reset is Manual Ready. Optional routines combine recorded actions with bounded condition waits, timeouts/retry caps, and confirmed starting state. Reset actions are outside policy episodes. RL works with fresh policies and no demonstrations.

## Product hierarchy

Sidebar: Agents, Library, Activity. Agent header: environment, selected checkpoint, readiness, contextual next action. Tabs: Demonstrations, Training, Evaluation, Run. An optional inspector provides detail without permanent clutter.

Recordings/environments are reusable; active sessions retain immutable snapshots. Training offers equally prominent Imitation/Reinforcement choices. Validation during training is distinct from explicit frozen-checkpoint evaluation. Run means live execution, and new checkpoints never silently replace its selected model.

Use native semantic colors/fonts/controls, light/dark modes, keyboard access, VoiceOver, accessible chart tables, meaningful empty/error states, and coalesced metric updates. Keep effective resolution/rate, disk cost, checkpoint and control scope visible. Put schema/architecture details and uncommon knobs in Advanced.

Stop agent immediately cancels control; Stop recording finalizes data; Stop training orderly checkpoints the selected experiment. Unrelated offline work continues when live control stops. Physical takeover displays You have control with explicit Resume/Record correction. Include a practice environment that uses the real recording/training/control path and supports direct RL.

## Experiments and evidence

Use three fixed seeds, identical episode splits and sample budgets, and separate tuning/held-out tests.

- Detail ablation: keep detail unless a smaller variant loses <=2 percentage points on every task and materially improves resources.
- Temporal learning: evaluate GRU credit horizon, initialization and optimization against held-out delayed-memory success and equal deadline/resource qualification. The original alternative-architecture proposal is superseded by the [2026-09-12 scope decision](../verification/temporal-scope.md).
- Cadence: 10 versus 20 Hz on tracking, taps/chords, small-target pointing, and dragging. Select by closed-loop benefit subject to measured deadlines, not nominal throughput.
- Hybrid learning: PPO with and without decaying demonstration regularization; verify improvement over suboptimal demonstrations without more execution faults.

Memory fixture: delayed visual cue with unseen trials; target >=90% success against 50% memoryless baseline. Include 2/8/30-second delays, randomized layouts, off-cursor small controls, native-window and browser interactions, and visual tracking. Numerical tests cover causal leakage, sequence/step equivalence, padding, reset isolation, sampled/scored probability equality, PPO ratio-one, GAE boundaries/timing, tiny-data overfit, checkpoint resume, and finite gradients.

Qualify p50/p95/p99 complete observation-to-dispatch latency, memory pressure, storage/backpressure, warm compilation, and actor/learner contention. Initial 10 Hz target is warmed inference p95 <80 ms; deadline compliance is also measured at the complete packet boundary. Test actual installed-bundle TCC and crash cleanup. A short success establishes limited learning/integration, not universal generalization.

## References

- [MLX interoperability](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
- [MLX AdamW defaults](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html)
- [ConvNeXt paper](https://arxiv.org/abs/2201.03545) and [official implementation](https://github.com/facebookresearch/ConvNeXt)
- [PPO](https://arxiv.org/abs/1707.06347), [recurrent state staleness](https://openreview.net/pdf?id=r1lyTjAqYX), [time-limit handling](https://proceedings.mlr.press/v80/pardo18a.html)
- [ScreenCaptureKit queue limits](https://developer.apple.com/documentation/screencapturekit/scstreamconfiguration/queuedepth)
- [TCC responsible code](https://developer.apple.com/forums/thread/678819)
- [PyInstaller macOS signing](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-code-signing)
