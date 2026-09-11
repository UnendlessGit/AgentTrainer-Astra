# Compiled actor execution

2026-09-11, Apple M3 Max, macOS 27.0 (26A428), MLX 0.32.2.

The actor now compiles the existing visual/recurrent policy and complete autoregressive packet decoder as one array-tree function. This changes execution, not model configuration: default ConvNeXt/detail/cursor inputs, FP32 arithmetic, two persistent GRUs, command capacity 16, capability masks, RNG splitting, and the sum of all active categorical log probabilities remain unchanged. Finite checks and all packet fields materialize together before wire conversion or recurrent state is committed.

`_PolicyExecution` captures the policy's parameter state as MLX inputs. The shape/dtype cache holds at most four callables. It stores no image values, cursor positions or episode IDs. A checkpoint replacement creates a new execution owner; closing the actor drops its execution cache. Initial `None` state becomes the temporal core's explicit zeros, giving warmup, reset and subsequent recurrent steps the same graph signature. Context/reset values remain tensor inputs. There is no sampling shortcut or reduced command budget after END.

## Measured comparison

`scripts/benchmark_actor.py` exercises owned BGRA ingestion, the qualified Metal preparation, the full policy/decoder, finite checks, command validation and JSON conversion. It cycles through three synthetic 1280×720 practice layouts, carries recurrent state and RNG, excludes the first three calls, and reports twelve subsequent samples. The default pretrained global backbone and all keyboard/button/absolute/relative/scroll capabilities are enabled. These are initialization/performance examples, not claims of learned behavior quality.

| Action selection | Execution | Median | P95 | Maximum |
|---|---|---:|---:|---:|
| Stochastic | Eager | 91.74 ms | 93.55 ms | 93.55 ms |
| Stochastic | Compiled | 58.51 ms | 59.58 ms | 60.01 ms |
| Greedy | Eager | 87.54 ms | 89.05 ms | 89.99 ms |
| Greedy | Compiled | 54.69 ms | 56.07 ms | 56.10 ms |

Compiled peak MLX allocation was 968,012,689 bytes; the largest observed process peak RSS was 306,102,272 bytes. The benchmark sets a 2,500 MiB MLX allocation limit and 64 MiB cache limit. This is one source at the default configuration; it does not establish an arbitrary multi-surface memory bound.

All fifteen packets, including the initial three, were identical between eager and compiled runs in both action-selection modes. The stochastic run included empty through full sixteen-command packets; every greedy packet used all sixteen commands. Maximum absolute differences were 1.526e-5 for joint log probability, 8.94e-8 for value and 5.37e-7 for persistent state. Small floating-point differences are expected from kernel fusion; the implementation preserves FP32 and the shared probability computation, not bitwise floating-point identity.

The first observed compilation in an earlier same-configuration process took 30.075 seconds. Later same-host process first calls took 0.172–0.214 seconds. Preparation must continue to budget for cold compilation. The native coordinator's warmup occurs before control arming and does not count its first sample toward runtime timing admission.

This benchmark excludes ScreenCaptureKit delivery, native frame publication, helper/control-observation traffic and process-pipe latency. Its margin is encouraging for the 100 ms default cadence; it does not prove full native 10 Hz operation or scheduling behavior under GPU contention. Native warmup admission and every runtime execution deadline remain enforced.

Reports retained outside source control:

- `.local/verification/actor-compiled-varied-stochastic.json`
- `.local/verification/actor-compiled-varied-greedy.json`
- `.local/verification/actor-compile-stochastic.json` (initial single-layout comparison and first compilation)

Reproduce with:

```sh
.venv/bin/python scripts/benchmark_actor.py --report /tmp/astra-actor-stochastic.json
.venv/bin/python scripts/benchmark_actor.py --greedy --report /tmp/astra-actor-greedy.json
.venv/bin/python -m pytest python/tests/test_inference.py -q
```

## Correctness checks

The actor test suite exercises native ring ingestion/acknowledgements, training-equivalent preparation, eager and compiled packet/RNG/state agreement across image, context and reset changes, bounded cache eviction, current parameter values after graph compilation, checkpoint activation after compiling the old policy, and nonfinite output failure before actor-state commit. All 23 inference test cases passed. Existing warmup state rollback, causal history, malformed input, owner-thread and worker-protocol checks continue to pass. These numerical/component checks complement, and do not replace, installed native capture/control qualification.
