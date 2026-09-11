# Qualification decisions after implementation

The independent baseline remains the historical design record. These refinements follow measured implementation behavior, numerical comparisons and workflow failures. They do not inherit constraints from Original AgentTrainer.

## Visual semantics and identity

Model configuration version 2 masks padding inside every visual encoder stage, including normalization outputs before convolutions. The original downstream-only mask allowed zero padding to amplify random stem-bias gradients by seven to eight orders of magnitude. Per-stage masks reduced the pathological gradient to ordinary finite values without changing model capacity, resolution or pretrained tensors. Mixed-size feature fusion clamps interpolation to content-intersecting cells so batching does not alter single-observation edge features. Unmasked ConvNeXt still matches the independent torchvision reference.

Global and detail branches retain independent padding transforms rather than forcing source pixels into matching canvases. Native BGRA training preparation and the Metal actor path share alpha composition, Lanczos filtering, rounding, normalized padding and cursor geometry. Runtime clocks retain UInt64 precision; recording indexes retain their explicit signed SQLite bound. The event-availability cutoff, rather than its original source timestamp alone, controls what an actor may know.

Canonicalizer version 2 partitions demonstrations on one integer-millisecond execution grid. Boundary events rounding to the next interval are carried exactly once. Raw timestamps are never rewritten. Dataset manifests expose initial-lead/final-tail exclusions and refuse control sequences that cannot meet capacity or spatial reconstruction bounds. Absolute actions use half-open normalized coordinates `[0,1)`; sampled subcell centers are already interior, so no post-sampling clipping is introduced.

## Exact gradients with bounded memory

The initial whole-vision checkpoint graph reached 16.29 GB at eight decisions and could not meet the intended 64-decision BPTT window. The implemented backward pass applies the chain rule in stages: objective cotangents, bounded packet-decoder VJPs, one full recurrent-sequence VJP, then bounded visual VJPs. Dense pointing and pooled visual paths both contribute to the final visual gradient. Small reference tests compare every trainable tensor, recurrence, masks and cancellation against monolithic differentiation.

Measured two-lane, 64-decision production backward now peaks at 8.10 GB including a real optimizer update, actor/rollback references and another gradient accumulator. It retains FP32 parameters/probabilities/optimizer state, full imagery and the complete recurrent sequence. This is a memory implementation change, not a smaller model or shorter temporal objective.

## PPO admission and complete experience

The first-update KL failure was empirical: a nominal AdamW proposal could move the joint packet distribution far outside the `.02` sampled-KL target even when entropy regularization was disabled. Checking only before the next step retained that overshoot. The learner now checks candidate weights over the sealed behavior rollout with exact recurrent replay and bounded backtracking. Rejected proposals never become pending actor weights; accepted optimizer moments advance once. Candidate and accepted KL, step scale and rejected proposals are reported separately. This constrains the sampled rollout, not every possible state or the exact population KL.

Rollout length is now a minimum followed by the current episode's completion. Discarding an old actor's late tail excluded sparse terminal rewards beyond the initial rollout length. The collector instead stores lossless raw observations in a bounded disk spool, retains finite cache/metadata budgets, and trains complete fixed-behavior episodes. It never fabricates termination when disk or decision limits are reached; those failures cancel the invalid collection. A new actor snapshot activates only after a confirmed reset. Asynchronous wall-clock desktop collection and authored rewards/reset routines remain further implementation work.

## Actor execution and lifecycle

The actor compiles the same array policy graph, with a bounded shape cache and new owner on checkpoint replacement. Benchmarks retain full packet budget and FP32 output, matching sampled commands across varying images while reducing warm computation from approximately 92 ms to 59 ms. Native warmup consumes real frame leases but restores actor recurrence, RNG, packet sequence and observation cursors; it runs before arming controls and checks the measured timing margin. Runtime missed deadlines stop control rather than retiming learned actions.

The desktop executor has an independent watchdog and a per-user physical-input lease. Possibly posted holds are reserved before leaving the executor lock; disarm can invalidate admission while a backend post is in flight, and a stale returning post triggers another owned-release attempt. Permission and target proofs are polled independently with bounded freshness, so synchronous TCC queries do not sit on the millisecond posting path. Helper shutdown, actor shutdown and ring retirement have separate ownership barriers. Virtual tests prove these paths; actual OS input/permission attribution still needs installed-build qualification.

## Shared artifacts and review

Copying an agent links its existing immutable recordings/checkpoints instead of duplicating weights or leaving a foreign invisible selection. Resumed behavioral runs read the saved dataset and complete optimizer configuration, ignoring hidden draft settings. Recording review is read-only and checksum-verified; UI scrubbing coalesces disk work. Native component renders cover both themes and minimum sizes, while actual keyboard/VoiceOver checks remain distinct.

See the verification directory and implementation gates for exact commands, measurements, scopes and remaining limitations. None of these refinements establishes the still-open architectural ablations, broad learning quality, complete desktop RL or release readiness.
