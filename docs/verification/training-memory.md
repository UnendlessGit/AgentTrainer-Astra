# Exact backward with bounded memory

The original whole-vision activation-checkpoint graph did not meet Astra's default temporal requirement on the 36 GiB M3 Max. With the full production model and repeated native 1280×720 practice observations, four decisions peaked at 8.75 GB of MLX active allocation and eight at 16.29 GB. The guarded runner refused larger projected-unsafe graphs rather than exhausting unified memory. A naïve linear projection for 64 decisions was approximately 130 GB; that graph was never launched.

`learning/backward.py` now evaluates independent visual observations and per-packet decoder work in bounded slices, while differentiating the entire recurrent sequence. It first retains evaluated visual summaries and dense cells, evaluates the complete temporal core, and obtains the objective's cotangents for packet log probability, value and conditional entropy. It then replays decoder VJPs in small decision groups, one full-sequence temporal VJP, and sequential visual VJPs. Both dense pointing and summary paths contribute to visual gradients. Every accumulation is evaluated before the next slice so its working graph can be released.

This is the chain rule applied in stages. It does not shorten TBPTT, detach vision from the final gradient, lower image resolution, change parameter precision, or alter the loss. Nine independent regressions compare all trainable leaves and next recurrent state against a monolithic reference for BC and clipped PPO, frozen/unfrozen vision, mixed surfaces, contexts, nonzero incoming state, resets and padding. Auxiliary diagnostics remain auxiliary; zero cotangents preserve the original masking of invalid infinite log probabilities. Cancellation restores module parameter bindings and leaves the caller's prior complete chunk available for resume.

Command:

```sh
.venv/bin/python scripts/benchmark_training_memory.py --mode staged --lengths 4 8 16 32 64 --output .local/training-memory-staged.json
```

| Decisions in one lane | First backward | Peak MLX active allocation |
|---:|---:|---:|
| 4 | 0.90 s | 4.13 GB |
| 8 | 1.76 s | 4.26 GB |
| 16 | 3.50 s | 4.52 GB |
| 32 | 6.91 s | 5.03 GB |
| 64 | 14.49 s | 6.05 GB |

Each isolated process retained actor/rollback references and optimizer state, applied an actual clipped optimizer update, and computed a second backward while retaining the first gradient accumulator. This matters because `copy.deepcopy` initially shares immutable MLX buffers; merely allocating a named actor/rollback copy would understate an established update's memory. All parameter gradients remained finite and nonzero in the expected visual, recurrent and action branches.

The intended two-lane, 64-decision configuration also passed: first backward 32.41 seconds, peak MLX active allocation 8,097,969,376 bytes, with finite gradients through every branch. It retained the same actor, optimizer, rollback and first-gradient buffers during a real update and second backward. Report: `.local/training-memory-staged-two-lanes.json`.

The runner uses a low MLX allocation limit, bounded cache, a sampled watchdog, and sequential child processes; MLX's limit is a guideline, not a hard cap. The reported memory is MLX active allocation, not total machine memory. CPU RSS is recorded separately and does not account for all Metal allocations. Inputs repeat a real rendered frame to isolate memory scaling, so these figures do not establish learning quality, storage throughput or target/GPU contention. Multi-surface and full RL workload qualification remain separate measurements. Common all-padding trailing columns are trimmed from BC batches, preserving every real decision and avoiding full-resolution work for empty tails.
