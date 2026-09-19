# Learner command-buffer granularity

2026-09-12. The learner-only `10/10` override did **not** improve actor deadlines. On matched actual-cutoff pacing, it produced 276 lead misses versus 8 at the Max device's `50/50` setting, and slowed completed learner segments slightly. No production setting changed and no wider sweep was started.

## Verified semantics and numerical gate

The September 12 test used installed `mlx` and `mlx-metal` version 0.32.2. The environment overrides were unset before testing; this is historical test provenance, not a claim about dependencies restored after the Mac reset. The pinned [environment readers](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/utils.h#L169) cache their values on first use, so each learner process received its overrides before importing MLX. The [Max-device defaults](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L574) are 50/50.

These are submission heuristics, not GPU priority or hard allocation limits. The [commit predicate](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L477) checks dispatch count or a shifted tracked data-size total. Despite the variable's MB name, the accumulated [`array.data_size()` is in elements, not bytes](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.h#L316). Thresholds trigger only after being exceeded; an operation is not preempted. No fast-synch, architecture override, precision, model, image or objective change was used.

Two isolated full-production GRU B1×T4 fixtures compared all gradients, one clipped AdamW update and optimizer tensors using identical checkpoint weights and generated 1280×720 pixels. This shorter sequence is only a tractable numerical check; measured contention retains B1×T64. Initial parameter bytes, scalar loss and gradient norm matched exactly.

The first strict elementwise gate (gradient rtol 2e-5 / atol 2e-6) failed on small FP32 depthwise-gradient differences. That failure is preserved. A repeated `50/50` control showed greater aggregate variation than changing to `10/10`:

| Difference from first 50/50 run | Same-setting repeat | 10/10 |
|---|---:|---:|
| Largest gradient absolute difference | 1.18e-5 | 7.63e-6 |
| Global gradient relative L2 difference | 4.60e-7 | 4.48e-7 |
| Largest updated-weight absolute difference | 7.75e-7 | 3.37e-7 |
| Update difference relative to the actual step | 6.19e-6 | 5.32e-6 |

On that evidence, the bounded timing experiment used the existing staged-backward tolerance (gradient rtol 2e-4 / atol 2e-5) and a stricter 1e-6 absolute / 1e-5 relative check for other non-initial tensors. Every leaf passed; all 1,460 stored tensors were compared. Aggregate cross-profile differences are below the observed same-profile variation. This is numerical agreement within measured FP32 variability, not bitwise gradient identity or automatic permission to change production settings.

## Matched timed windows

Both profiles used the same private immutable frozen actor helper/checkpoint and three generated full-size scenes as the [first contention screen](gpu-contention.md). The actor environment was unchanged. Only the learner received `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`, explicitly set to 50/50 or 10/10. Both actor loops used `next = lastActualCutoff + 100 ms`. OCR was absent.

Timing retains the fixture boundaries described in that screen: `latencyMS` includes publication of already-owned synthetic pixels into the benchmark ring and ends at response arrival, before packet validation and lease-release checks. The lead cutoff is assigned after ring publication, so the lead-miss check excludes publication, subsequent validation and control dispatch. Actual native pixel extraction and control-observation traffic are absent. Matching production pacing does not make these complete native-loop deadline measurements.

The model remains the full 34,638,639-parameter GRU, FP32, with the original global/detail/cursor paths and packet capacity 16. Learner work remains the same clearly labeled synthetic ratio-one PPO-objective exact staged backward, B1×T64, vision microbatch 1 and accumulation toward 256 decisions. It is not a full PPO iteration or learning-quality comparison.

| Learner profile | Actor requests | Median | P95 | P99 | Maximum | 100 ms lead misses |
|---|---:|---:|---:|---:|---:|---:|
| 50/50 | 580 | 92.20 ms | 96.92 ms | 101.05 ms | 106.51 ms | 8 (1.4%) |
| 10/10 | 556 | 100.15 ms | 116.37 ms | 118.72 ms | 127.09 ms | 276 (49.6%) |

Each profile had a configured 60-second window. Actual window plus in-flight drain totaled 60.12 and 60.02 seconds (120.14 seconds combined). Each completed three 64-decision backwards, then discarded an incomplete fourth; neither reached the 256-decision optimizer boundary, so **zero optimizer updates occurred in the timed windows**. Completed segments took 16.61–16.64 seconds at 50/50 and 17.29–17.30 seconds at 10/10.

Learner peak MLX allocation was 5.668 GB versus 4.945 GB. Lower allocation at 10/10 did not compensate for worse actor latency. Learner peak RSS was 349.4/348.7 MB; joined actor-child peak RSS was 356.8/355.0 MB. These are separate measurements, not an invented summed GPU/host memory total.

The coordinator reported holding builds, tests and other model work during both measured windows. Each start was explicitly gated after the actor/learner warmups finished. The retained logs do not establish a CPU-build overlap for this pair; they are not a system-wide resource trace. Actor setup/warmup took 0.61/0.62 seconds; learner setup/warmup took 14.55/13.17 seconds. The two numerical fixtures took about 1.55/1.52 seconds, with a separate same-setting repeat. Setup, coordination waits and teardown are not counted as contention time.

All sampled packet/observation IDs were unique, packet sequences contiguous, original packet timing unchanged, and actual cutoffs at least one period apart. The fixture checked sequence/draw equality, RNG chaining and exact ring acknowledgements. Both helpers joined and every frame lease was released. No screen, OS controls or TCC access was used.

## Interpretation and artifacts

This single ordered comparison is sufficient to reject this particular override as an improvement for the tested workload. It does not explain the precise driver/CPU scheduling cause or rule out every possible execution strategy. Both profiles still missed strict lead deadlines. Keep serial learning at released-control boundaries until another separately proposed scheduling approach is qualified.

The September 19 read-only audit confirmed the retained raw timing results and the numerical summaries. Raw reports, packets, logs and all three 1,460-tensor parity files remain under `.local/command-buffer-benchmark/`, including the original strict-gate failure and repeat-control evidence. The temporary frozen helper, generated images and checkpoint referenced by the old source report no longer exist after the Mac reset.

The source runner is `scripts/benchmark_command_buffers.py`, reusing `scripts/benchmark_contention.py` and `Tests/ContentionFixture/main.swift`. A future authorized reproduction must generate a fresh verified fixture/source report first, pass that report with `--source-report`, and use a new output directory. The per-profile gates then coordinate the measured windows. `--reuse-parity` only avoids recomputing numerical fixtures; it does not recreate the missing helper/images/checkpoint, and it is not a way to resume the retained September 12 run. Preserve the old evidence rather than overwriting it, and regenerate parity for any new source identity.

No benchmark process remains running. Production scheduling, model configuration and dependency defaults remain unchanged.
