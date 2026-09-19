# Full-GRU contention screening

2026-09-12. Naïve simultaneous GPU learning is **not qualified for live control** on this Mac at the unchanged 100 ms execution lead. The actor exceeded that lead in 103/330 requests with learner work and 94/325 with learner work plus OCR. A separate actor-only recovery using the production relative-cutoff cadence had zero misses in 145 requests. This supports keeping learning at joined, unarmed boundaries while evaluating scheduling improvements; it does not change the model, precision, cadence or action defaults.

## Scope and unchanged work

The benchmark uses the full 34,638,639-parameter production GRU policy, FP32, pinned pretrained ConvNeXt backbone, default global/detail/cursor paths and packet capacity 16. All four cases share one immutable frozen actor helper and one checkpoint, with keyboard, button, absolute/relative pointer and scroll capabilities. Three owned generated 1280×720 BGRA scenes contain practice imagery and a readable score. No screen capture, OS input, privacy request or real application target was used.

`Tests/ContentionFixture/main.swift` uses production `ComputeProcess`, `SharedFrameRing`, packet validation and `VisualRewardDetector`. The September 12 run built an isolated source snapshot and privately copied the frozen actor helper before execution. That helper's recorded executable SHA-256 is `d13830a650d3ef00517320b88ce10796a5a1ed061c754900e0e62ca211129e05`. The temporary snapshot is no longer present after the Mac reset; the retained report records which helper was used.

Reported `latencyMS` starts before the fixture publishes already-owned synthetic pixels into its ring and ends when the actor response arrives. It includes that fixture ring write, helper IPC, owned frame ingestion, Metal preparation, the full policy and sampling, materialization and wire return. Packet decoding/validation, counter checks and lease release happen after the timing endpoint. Real ScreenCaptureKit delivery, native pixel extraction/copying and control-helper observation/dispatch are not measured.

The lead-miss predicate has a narrower boundary: the fixture assigns its cutoff **after** ring publication and compares response arrival with that cutoff plus 100 ms. The production native loop obtains its control cutoff **before** pixel copying and ring publication, then validates and dispatches the result. Thus the reported misses exclude publication from the lead budget and exclude post-response validation/dispatch; they do not qualify the complete native deadline.

Every packet retains its original ID, commands, sequence and times. Outside the timed interval, the fixture checks exact frame lease acknowledgements, sequence/draw equality, RNG state chaining and immutable policy identity. Post-run checks also confirm unique packet/observation IDs, contiguous sequences, exact 100 ms packet lead/duration and at least 100 ms between actual cutoffs. Every ring was retired only after its actor joined.

OCR uses the real production accurate revision-3 Vision request and verifies that it reads `12345` from the generated full-frame pixels. Vision's default compute-device selection is unchanged; this is not a CPU-only claim.

## Timing results

| Case | Measured time | Actor requests | Median | P95 | P99 | Max | Lead misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| Actor alone | 20.01 s | 161 | 64.36 ms | 75.95 ms | 108.12 ms | 127.61 ms | 3 |
| Actor + OCR | 20.10 s | 168 | 62.71 ms | 66.36 ms | 68.42 ms | 94.53 ms | 0 |
| Actor + learner segments | 60.02 s | 330 | 95.91 ms | 105.55 ms | 106.51 ms | 107.00 ms | 103 |
| Actor + OCR + learner segments | 60.19 s | 325 | 97.03 ms | 111.28 ms | 117.31 ms | 120.07 ms | 94 |
| Actor-only recovery, relative cadence | 15.05 s | 145 | 65.09 ms | 66.67 ms | 67.36 ms | 89.31 ms | 0 |

Completed measured windows total **175.37 seconds**. The first four use a nominal 100 ms grid, skip missed slots, and also enforce the actor's minimum period between actual cutoffs. That yields lower achieved cadence than the production loop and is a specific limitation: they are not native 10 Hz qualification. The recovery instead schedules `next = lastActualCutoff + period`, matching the production coordinator, and is reported separately rather than pooled with the grid cases. Even at the grid protocol's approximately 5.5 requests/s under learner load, strict lead deadlines fail frequently.

The coordinator reported that no new builds or model jobs ran during these windows. An earlier native test helper was reported idle awaiting fixture exit and was terminated, with no reported GPU load. Retained parallel compilation logs lack start timestamps, so that coordination report does not independently exclude CPU-build overlap. Such overlap is not established and must not be asserted as the cause of the three early actor-only misses. Those misses remain in the results; the recovery does not erase them. These short samples do not establish robust tail latency across applications, thermal states or geometry changes.

## Learner work and its explicit limit

The learner uses the production exact `policy_gradients` implementation on **B1×T64**, vision microbatch 1, accumulating toward the default 256-decision optimizer boundary. Every visual parameter is trainable. Prepared tensor shapes are global `[1,64,448,768,3]`, detail `[1,64,720,1280,3]`, cursor `[1,64,384,384,3]`; no visual feature cache substitutes for training the encoder.

This is explicitly **synthetic ratio-one PPO-objective backward stress**: teacher-forced timed packets, synthetic advantages/returns and the production PPO scalar loss. It does not perform on-policy rollout generation, GAE, behavior-policy replay or full-policy KL admission and must not be called a PPO iteration or learning-quality result. The same graph and objective run in both contention cases.

Each 60-second learner window completed three exact 64-decision backwards and retained their accumulated gradients (192 decisions). An incomplete fourth backward was discarded at the deadline. **No optimizer update occurred**, because updating before the required 256 decisions would change the workload. Completed chunks took 15.92–15.99 seconds with the actor and 16.01–16.03 seconds with actor plus OCR. Gradients stayed finite, including nonzero backbone, detail, recurrent and pointing branches. This bounded result establishes contention during real full-quality backward segments, not the cost of a complete optimizer/validation cycle.

Learner peak MLX active allocation was 5.668 GB in each contention case; learner peak RSS was about 0.350 GB. These measures are distinct and are not added as though they were disjoint unified memory. OCR peak RSS was 97.6 MB alone with actor and 86.1 MB in the combined case. The recovery fixture records 40.2 MB native-fixture peak RSS and 355.1 MB joined actor-child peak RSS. The frozen actor protocol does not expose its MLX peak allocation, so this report does not invent a measured aggregate GPU peak. The learner has a sampled 11 GiB stop guard and a 12 GiB MLX allocation guideline.

## Setup, failures and reproduction

The isolated fixture build took 9.73 seconds; source snapshot, private helper copy and generated model/data preparation brought initial setup to 10.20 seconds. Actor setup/warmup took 1.78 seconds for the first completed case and 0.61–0.64 seconds for later cases. First real OCR setup/warmup took 32.60 seconds; the later OCR process took 0.20 seconds. Learner setup/warmup took 14.54/15.18 seconds, including a full warm backward of 14.21/14.79 seconds. Recovery preparation took 1.82 seconds. Setup and teardown are separate from measured contention.

An earlier fixture attempt completed three actor requests, then the actor correctly rejected a too-short actual cutoff interval caused by nominal-grid wakeup jitter. The fixture was repaired to respect the last actual cutoff; that aborted attempt remains in `.local/contention-benchmark/` and is excluded from the latency tables. A recovery coordination wait also expired before any measurement window began; its log and zero-measurement status remain recorded. No failing validation was disabled and no production code changed to accommodate these fixtures.

The September 19 read-only audit rechecked the retained raw samples and reports: the tabulated values, packet timing/counters and measured durations agree. Reports, per-role logs and numerical evidence remain under `.local/contention-benchmark-qualified/` and `.local/command-buffer-benchmark/`. The old `/tmp/AstraContention-…` directory, including generated pixels, checkpoint and frozen helper, is gone. Its recorded paths are historical provenance, not usable current artifacts.

A future authorized reproduction must first restore dependencies and build a verified frozen helper, then run `scripts/benchmark_contention.py` into a **new output directory** to generate a fresh private fixture and report. This executes GPU work; the audit did not run it. `--recovery-of` works only with a report whose referenced fixture still exists, so it cannot resume the retained September 12 report directly. Coordinate an exclusive benchmark window before invocation and record the new environment/helper identity; regenerated artifacts are a new run, not the missing original snapshot.

No benchmark process remains running. The next useful scheduling comparison can change only learner command-buffer submission granularity after numerical parity is checked; it must retain the full model and truthful deadline accounting.
