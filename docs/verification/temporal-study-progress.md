# Frozen-visual temporal study

2026-09-12. The first Stage 1 GRU tranche and its paired-data follow-up are complete; the 90% memory target is not met. No production model, schema or training default changed. Further temporal work is limited to GRU diagnostics under the [current scope decision](temporal-scope.md).

September 19 follow-up: a [CPU exposure audit and bounded matched GRU pilot](gru-exposure-diagnosis.md)
found a learned cue-conditioned ranking at all three delays in one explicitly
experimental per-choice/clipping condition. Other matched controls remain at
chance; independent-seed replication and autonomous behavior remain open.

The experimental implementation lives in `experiments/temporal_study.py`, with `scripts/temporal_study.py` for bounded execution and `scripts/inspect_temporal.py` for representation/fit diagnostics. The cache contains exact production-resolution visual features from the common native initial checkpoint. Source pixels remain 1280×720 with the default global/detail/cursor processing. This freezes all visual parameters and therefore is not production end-to-end learning evidence.

## Verification

Nine small numerical tests pass in `experiments/test_temporal_study.py`:

- Cached temporal/action gradients agree with an ordinary frozen-vision policy for both objectives.
- T64 and T512 preserve forward state and loss while applying the intended gradient cut.
- Future observations/labels cannot change prior temporal state; reset and padding are exercised.
- Cancellation restores parameter bindings without updating them.
- Cache outputs equal uncached visual encoding; corrupt bytes and wrong visual weights are rejected.
- Cached oracle samples agree with `PracticeDemonstrations` controls, commands and visual phases.
- Complete-episode checkpoint resume matches uninterrupted parameter updates.
- Paired counterfactual source episodes differ only in visible cue inputs, retain equal cue-free control/features, and require opposite choice labels.

The objectives both normalize by **all valid decisions in the same two complete episodes**. Choice-only masks waiting losses but does not multiply the surviving gradients by episode length. Choice-packet NLL is reported separately using the selected-packet count.

## Longest-update pilots

Each pilot used two complete 307-decision episodes at the 30-second delay, actual clipped AdamW updates, and rollback/optimizer buffers. Every gradient and update was finite.

| Path | First update | Second update | Peak MLX allocation |
|---|---:|---:|---:|
| A: T64, natural loss | 3.530 s | 3.499 s | 1.030 GB |
| B: T512, natural loss | 3.497 s | 3.370 s | 3.003 GB |
| C: T512, choice-only | 0.459 s | 0.417 s | 2.899 GB |

These are frozen-feature diagnostic timings. They are not application training-throughput claims. Every process is bounded to six minutes and saves only a complete optimizer boundary. Paused arms resume with exact optimizer/RNG state and an unchanged episode-batch schedule; incomplete work is not counted as an update.

## Original single-cue-per-layout dataset

The unchanged first revision has 96 training episodes: 32 independent layouts for each 2/8/30-second delay. Preparing its full-resolution feature cache took 102.50 seconds and stored 2,198,705,760 tensor bytes. The dataset manifest hash is `bf06fff9fc25cac213d27f23384daa91ff745ddc8611b5e30e6a317b2b875656`.

All four arms completed 192 updates: A (geometric T64 natural), B (geometric T512 natural), C (geometric T512 choice-only) and D (native T512 choice-only). Every arm remains at 50% held-out paired-cue accuracy at all three delays, with zero cue-flipped choices. The exact 192 episode-pair sequence agrees across all arms; each consumed 53,888 valid decisions. Choice-only selected 384 of those decisions for its loss. A resumed after its six-minute boundary; its optimizer and data cursor were preserved. The initial production-model seed is 834.

C's ordinary training choice ranking is 84.4% at 2 seconds, 96.9% at 8 seconds and 75.0% at 30 seconds. Its held-out paired score remains 50%. A fixed ridge linear readout of the common frozen cue summaries classifies all 64 held-out counterfactual cue colors correctly after fitting 96 source cue frames. This is only a cue-information diagnostic: it neither remembers a delay nor chooses or executes an action. The common features contain accessible cue information.

The training/generalization gap suggests a dataset shortcut: every repeated training layout has one fixed cue/answer. Layout memorization can reduce loss without retaining the cue. Consequently, the unpaired dataset cannot by itself identify the limiting temporal mechanism.

## Approved paired-data follow-up

A/B finished on the unchanged revision. A separate paired revision now contains both visible cues for every training layout. The two complete mates form each optimizer batch. Their cue-free pixels, controls, timing and geometry are equal, while their supervised choices are opposite. This removes the layout-to-answer shortcut without appending privileged cue features or changing the action vocabulary.

The paired revision contains 192 episodes, passed the cue-free input equality checks, and has manifest hash `56ba52099b45a9a16849bdffb83fa4403a83a3d0266de566ef0bb1583759958f`. Preparation took 93.87 seconds, reusing verified original-cue episodes and the immutable visual cache. Natural T512 and choice-only T512 use identical initial weights, paired source batches, optimizer/example counts and held-out paired layouts; both paired arms have reached 192 matched updates and 53,888 valid decisions with an identical episode-pair sequence. Paired C completed its final validation and remains at 50% for every delay. Paired B reached its update target before the tranche boundary. A separately authorized forward-only pass completed its final validation in 10.66 seconds, with zero optimizer updates and an unchanged checkpoint hash. It also remains at 50% at every delay, with zero cue-flipped choices. The original bounded-run report retains its truthful `validationPending` status; the completed endpoint evidence is `.local/temporal-paired-study/validation-B-192-final.json`. The existing source and checkpoints remain immutable. No conclusion about GRU incapacity follows from the unpaired result.

Private reports are under `.local/temporal-study/` and progress logs under `.local/verification/temporal-*.log`. The final unseen seeds 2000–2127 remain unused.

The measured training/preparation/diagnostic tranche totals 1,796.82 seconds of the approved 1,800 seconds. The final B192 forward-only validation is recorded separately. Its first report-writing attempt failed after evaluation because of a duplicate `scope` key; the corrected bounded pass completed and preserved the checkpoint bytes. No optimizer updates occurred in either follow-up. No temporal process remains running.


## Completed paired comparison

| Paired arm, model seed 834 | Updates | Valid decisions | 2 s | 8 s | 30 s |
|---|---:|---:|---:|---:|---:|
| B: T512, natural packet loss | 192 | 53,888 | 50% | 50% | 50% |
| C: T512, choice-only diagnostic loss | 192 | 53,888 | 50% | 50% | 50% |

Both arms used the same 192 counterfactual episode-pair sequence and geometric GRU initialization. These are forced-choice validation results, not freely acting success. The paired source removes layout-to-answer memorization as a sufficient solution, yet neither objective learned useful cue-conditioned choice in this budget. This does not establish GRU incapacity: training duration, weak delayed gradients, frozen visual/readout compatibility and precise packet-target optimization remain possible limitations.

The subsequent [immediate-cue readout diagnostic](immediate-cue-diagnostic.md) uses an actual rendered cue replay. Its matched 192-update multi-layout arm also remains at 50%, while 512 extra repetitions of one opposite-cue pair fit both choices and generate successful greedy packets when admitted at readiness. The evaluator agrees with directly rendered input and the temporal-to-action gradient is connected. Weak initial cue conditioning, exposure and optimization remain important confounds. Further GRU studies need sufficient exposure; none of these 192-update chance results establishes architectural incapacity.

Final paired checkpoints:

- B: `.local/temporal-paired-study/checkpoints/fbbcf384-f5d4-4c4a-bc6e-1eb783b99f3b`.
- C: `.local/temporal-paired-study/checkpoints/af0b35f2-eff3-4484-aa6d-4d4c423cd884`.

The reserved unseen seeds 2000–2127 and three-model-seed production replication remain untouched/uncompleted. Production defaults remain unchanged.
