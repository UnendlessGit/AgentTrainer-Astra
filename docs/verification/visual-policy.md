# Visual policy and checkpoint verification

Verified on 2026-09-06 using the M3 Max development Mac (14 CPU cores, 36 GiB unified memory, macOS 27.0 build 26A5425a), MLX 0.32.2 and the locked Python 3.12 environment. This qualifies numerical components and a bounded computation benchmark. It does not establish behavioral learning, PPO improvement, real-time control, or an installed-app workflow.

## Model and pretrained conversion

The default policy has **34,638,639 FP32 parameters**: a pretrained ConvNeXt-Tiny global encoder, shared detail/cursor CNN, explicit spatial transforms and dense pointing cells, 32 learned spatial queries, two 512-wide persistent GRU layers, a scalar value head, and a separate autoregressive packet decoder. Native observations and executed control history enter the persistent state; packet teacher forcing does not.

The upstream source weights are pinned in `assets/weights.json`. Their classification head is excluded. The converted 27,820,128-parameter backbone has SHA-256 `b55d15aebe3c2830f8baaf65ed0cf1578e2d1babad488361bec917cab7dc31dd`. The review rechecked the source digest, strictly loaded the converted artifact, and ran `scripts.prepare_weights.qualify` against independently shipped torchvision layers with the original weights. All four stage outputs and the pooled normalized output matched at 224×224, 160×256, and 432×768. The maximum normalized RMS error across these comparisons was 1.35×10⁻⁵; comparison uses `atol=1e-3, rtol=1e-3` to accommodate FP32 CPU/Metal summation differences. Maximum absolute error was 0.00635 in a large-magnitude intermediate stage. PyTorch/torchvision are build-time verification dependencies.

The integrated Python suite passed **48 tests** at this review; `swift test` passed **37 tests**. Relevant model checks establish:

- Sequence/step equivalence, reset isolation, and unchanged recurrent state through padding, including resets on padded samples.
- Finite gradients reaching the pretrained stem, final global stage, detail branch, cursor-position branch, query pooling, both GRU layers and value head. Visual microbatching/checkpointing preserves outputs and parameter gradients within numerical tolerances.
- Dense cells retain per-surface validity with unequal resolutions; unavailable images keep attention finite. Preprocessing preserves BGRA/RGB order, cursor coordinates, content padding, and differentiable alignment between independently padded feature grids.
- Sampled packets and teacher scoring agree on the exact joint probability. Analytical zero-logit checks count END, ordered time and active arguments. Capacity-forced END has probability one. Disabled capabilities/unavailable surfaces, pointer-mode grammar, and inactive arguments behave consistently. This is necessary evidence for PPO, not a completed PPO trainer.
- Native raw/LZFSE recording fixtures decode byte-exactly in Python. Late source time remains separate from availability time; source corruption, active writers and linked shards are rejected. These fixtures require no privacy permission.

## Checkpoint recovery

Model/state tensors use safetensors and bounded JSON trees. Publication refuses an existing destination. Tests cover artifact corruption, configuration signatures, invalid artifact paths, exact policy/vocabulary round-trip and AdamW continuation. Seven additional review regressions in `python/tests/test_checkpoint_review.py` verify:

- Frozen leaf paths survive reload without freezing unrelated leaves with the same name; freezing and then unfreezing the backbone preserve resumed parameter updates. Frozen values remain byte-identical through updates.
- Nonfinite optimizer moments are rejected before publication and on training-state load, including an artifact whose checksum was recomputed after inserting infinity. Failed publication leaves no visible checkpoint or owned staging directory.
- Oversized model metadata is rejected by the integer-only parameter budget before the model constructor can allocate any parameter arrays.
- An advanced saved MLX random key reproduces later normal, uniform and integer draws exactly. Successful loading and an injected constructor failure both preserve the caller's generator sequence.

MLX requires `optimizer.init(model.trainable_parameters())` when an unfreeze expands the trainable tree. This adds missing moment slots while preserving existing moments; calling only `unfreeze()` and then `update()` raises a missing-state error. The regression performs the explicit transition. The training workflow must do the same.

## Bounded production-shape benchmark

Command: `.venv/bin/python scripts/benchmark_policy.py --backward --output .local/verification/policy-benchmark-review.json`. Configuration signature: `97851f3eae16ad4636519d6b6febfb105865e65da1d09e6e6bb6f87de8ff9a6e`.

One surface, one observation: synthetic normalized RGB at 432×768 globally, 864×1536 detail and 384×384 cursor; default 16-command capacity. Two warmups preceded five measured inference repetitions. Timers synchronize the produced MLX arrays.

| Measurement | Result |
|---|---:|
| Median visual/recurrent encoding | 44.117 ms |
| Median autoregressive packet sampling | 56.354 ms |
| Median total inference | 100.446 ms |
| Total inference range | 99.412–102.812 ms |
| MLX peak allocation during inference | 936,508,393 bytes |
| One forward/backward, visual checkpointing enabled | 283.964 ms |
| MLX peak allocation during backward | 3,200,291,796 bytes (2.9805 GiB) |
| Packet NLL plus value penalty | 46.44533, finite |
| Pretrained stem gradient norm | 1.40241, nonzero |

All backward gradient tensors were finite. Memory values are MLX's allocation high-water readings, not total process RSS or aggregate memory across the app and workers. The backward measurement is one pass, not a warm training-throughput estimate. Inference excludes capture, preprocessing, IPC, input scheduling and host validation; it resets observation state for each repetition. The packet sampler still evaluates the bounded decoder after early END, so command count does not directly determine these timings.

The measured inference time leaves no end-to-end margin for a 100 ms decision period. Profile and optimize the decoder, then measure scheduling and contention with native capture/control before choosing a supported cadence. Reducing visual quality merely to satisfy this timing measurement is not a qualified decision.

## Remaining release evidence

BC overfit/held-out/closed-loop learning; recurrent PPO and live rollout correctness; long-sequence training memory and throughput; real observation distributions; checkpoint UI/job integration and recurrent policy-version isolation; three-seed architectural comparisons; frozen evaluation protocols; actual native capture/control; installed-bundle training/inference; and final release qualification remain open. The compute helper still advertises diagnostics only. None of the component checks substitutes for those gates.
