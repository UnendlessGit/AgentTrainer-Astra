# Recurrent PPO practice workflow

`python/astra/learning/reinforcement.py` connects the actual shared `AgentPolicy`, timed packet decoder, causal observation preparation, duration-aware GAE, PPO loss and grouped AdamW optimizer to `PracticeEnvironment`. It samples agent actions and evaluates their actual consequences. It never calls the privileged practice oracle or feeds rewards/answers into actor observations. Both fresh policies and compatible behavior-trained checkpoints can initialize the learner.

This is a synchronous **virtual-time practice** coordinator. It does not implement concurrent wall-clock desktop capture/control, actor/learner GPU arbitration, an external reward bridge, reset authoring, or live native policy execution. Those product and release gates remain open. Numerical-model tests and short gradient probes below do not prove closed-loop learning quality.

## API and ownership

Construct `ReinforcementTrainer(policy, environment, ReinforcementConfig(), policy_id=..., context_ids=..., restored_state=...)`. `policy` is the learner and checkpoint model; the trainer creates a private actor copy. The caller must not mutate the learner concurrently. Period, lead, vocabulary, context and shaping/return discount compatibility are checked.

- `collect(cancelled=..., on_decision=...)` returns `CollectedRollout` with immutable transition metadata, raw BGRA observations, causal control/event metadata, exact sampled packet tokens, behavior recurrent states and pre-reset bootstrap observations. One sealed rollout may be outstanding.
- `verify_behavior(rollout)` replays from episode zero and checks old joint likelihoods and behavior values before any optimizer update. It returns maximum likelihood disagreement and ratio bounds.
- `update(rollout, cancelled=..., on_update=...)` returns `ReinforcementMetrics`. It applies real recurrent PPO gradients to the learner, keeps the actor unchanged, and queues at most one pending policy snapshot.
- `collect()` treats the configured decision count as a minimum and finishes the running episode before sealing. No automatic old-actor tail is excluded from training. `activate_pending_at_reset()` resets the world, verifies a new episode identity and cleared controls, then activates pending weights with fresh recurrence. It rejects mid-episode activation.
- A sealed learning rollout begins at confirmed episode zero and ends at a true terminal/truncation boundary. The separate `advance_to_reset()` method is an explicit administrative/audit operation, not part of normal collection. It must not be used to omit late rewards from a learning run.
- `run_iteration(...)` composes collection/update and returns `IterationResult(metrics, checkpoint_state)`. Save the learner with the existing `save_checkpoint(..., kind="reinforcement", training_state=result.checkpoint_state)`. Restore optimizer, counters, explicit actor RNG key and next reset seed using `restored_state`. Resumption always starts a fresh world and episode; it does not claim to restore desktop state.
- `discard_rollout` permits a deliberate discard. `stop` aborts any still-running practice world without inventing a zero-duration training row. During collection/update, use the cancellation callback. Cancellation during update restores the pre-update learner parameters and optimizer; manual `update` leaves its sealed rollout available for retry, while `run_iteration` stops and discards on cancellation.

The collector stores lossless raw BGRA in a checksummed temporary disk spool rather than retaining every normalized global/detail/cursor tensor. Recurrent snapshots and metadata remain bounded in RAM, and a bounded raw-frame cache accelerates reuse. At 1280×720, 512 raw frames require approximately 1.9 GB before bootstrap frames and metadata. Defaults are 4 GiB for cache/metadata, 32 GiB disk, 65,536 hard decision limit and a 10 GiB free-disk reserve. Ordinary bounded writes fail cleanly; there is no sparse writable mapping. Resource limits report a failed collection rather than inventing termination. Completed updates, discards and stops retire owned spool data; a failed manual update retains it for retry. This proves a local practice store; live desktop rollout journaling remains separate work.

## Update behavior

Current configuration/state schema 2 defaults are a minimum 512 decisions followed by episode completion, four epochs, 64 loss steps, 32 burn-in steps, effective batch 256 valid decisions, new/pretrained learning rates `1e-4/1e-5`, gradient norm `.5`, and the `.2/.5/.01/.02` clipping/value/entropy/KL settings documented in [RL math](rl-math.md). Frozen parameter groups do not advance their optimizer clocks. Parameter/probability/optimizer arithmetic stays FP32.

Each recurrent chunk remains within an episode. Its default prefix begins from a saved behavior-state snapshot and replays up to 32 observations using current learner weights, then detaches that state before the 64-step gradient segment. This is an explicit approximation after updates. `state_replay="full_prefix"` replays from episode zero with current weights instead; `replay_state(..., mode=...)` supports comparison. Tests confirm equality before updates and measure the expected divergence afterward. Advantages are normalized once across valid rollout decisions. Joint old packet likelihoods and return targets remain fixed across epochs.

Chunk gradients are summed and divided by accumulated valid decision count. The final partial effective batch is retained. The staged chain-rule backward preserves the full recurrent objective with bounded visual/action working sets. Nonfinite losses, gradients, parameters or optimizer state reject the update. Each proposed AdamW step then receives a complete sampled-rollout check using exact recurrence; bounded backtracking scales the parameter delta while keeping one once-computed optimizer proposal. Moments advance only if a candidate is admitted. A failed proposal restores the previous parameters/state and never becomes pending actor weights. Accepted and proposed sampled KL, backtracks, step scale and rejected proposals are distinct diagnostics. This is a bound on the sampled rollout, not an exact population trust-region guarantee.

## Padding-gradient investigation and model version 2

The first real fresh-policy PPO run exposed a large finite gradient that ordinary loss/ratio tests did not reveal. Zero-filled letterbox cells passed through successive ConvNeXt LayerNorm sites before downstream spatial masking. Zero-initialized stem biases therefore acquired extreme derivatives. Ratios remained approximately one, and value errors were small, so clipping the loss or changing advantages would have hidden the cause.

The fix adds optional content rectangles to the ConvNeXt and detail encoders. Inputs, intermediate stages and block outputs are masked; normalization outputs are also masked before stride convolutions can mix them into valid cells. Cursor detail receives its actual intersection with the observed surface. This preserves real partial-edge features and stops padded pixels or learned normalization biases from introducing artificial padding content. No learned parameters were added or changed. The unmasked ConvNeXt API remains the original reference path.

Feature alignment also clamps interpolation to the first/last cells intersecting real source content, rather than the padded canvas. Otherwise, adding a taller or wider observation to a batch introduces extra zero cells and attenuates edge features compared with single-observation inference. Regression tests compare dense valid cells, bounds and pooled summaries for wide, tall and partially letterboxed observations alone and in mixed-size batches.

These altered padded-image semantics are explicitly `ModelConfig.schema_version=2`. Development model schema1 is rejected. Checkpoint container schema1 and upstream pretrained tensor artifacts remain unchanged.

Two-decision probes on the available M3 Max used the real policy, sampled joint packets, PPO gradients and all unfrozen layers. All ratios were within approximately `8e-6` of one. The production probes used 1280×720 source frames and the default 768/1536/384 visual limits.

| Probe | Stem bias gradient norm before | After masking | Largest leaf after masking |
|---|---:|---:|---|
| Small random, 96×64 source with global letterbox | 100,461,626 | 0.614 | Temporal input projection: 32.65 |
| Default random, 1280×720 source | 13,233,851 | 0.377 | Temporal input projection: 52.31 |
| Default pretrained, 1280×720 source | Stable before the fix | 0.616 | Temporal input projection: 73.66 |

A small square input without letterboxing had a largest leaf norm of 34.19 before the fix, independently isolating the padding path. Production gradient probes peaked at approximately 4.68 GB of MLX memory after the fix. These are short numerical probes, not throughput or learning-quality benchmarks.

The unmasked converted pretrained backbone was requalified against independently shipped torchvision layers at 224×224, 160×256 and 432×768. All stage and final-summary comparisons passed the existing combined `atol=1e-3, rtol=1e-3` criterion; maximum normalized RMS error was `1.35e-5`. The maximum absolute stage error was `0.00635` on large-magnitude activations, within that combined tolerance. The pinned upstream source SHA-256 was checked before `weights_only=True` loading. No runtime Torch dependency was introduced.

## Executed tests and remaining gates

`python/tests/test_reinforcement.py` exercises real collection/replay/update/reset activation, irregular truncation bootstraps, burn-in/full-prefix differences, frozen optimizer clocks, cancellation rollback, tampered behavior likelihood rejection, checkpoint load/resume, raw-memory admission, audit-tail exclusion, and cleanup after an environment has already ended. `python/tests/test_visual_padding.py` covers unmasked/full-content equivalence, padding-value and input-gradient invariance in both encoders, finite empty masks, regression of the amplified stem-bias gradient, and explicit old-model rejection.

The initial numerical campaign passed 11 recurrent PPO tests and nine visual-padding tests on September 6, 2026. Later admission and complete-episode storage qualification passed 77 combined RL checks, including a sparse terminal reward after the rollout minimum that enters GAE and an actual policy update. A full-model 128-decision source-job admission/cancel/resume campaign passed before the complete-episode collector change; see [reinforcement jobs](reinforcement-jobs.md) for exact measurements and limits. The new collector still requires a fresh production-scale campaign. Multi-seed held-out improvement, delayed-memory success, asynchronous native actor continuity, reward/reset workflows, evaluation and installed-app testing remain release gates.
