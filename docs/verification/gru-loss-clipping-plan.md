# Proposed three-seed GRU loss/clipping comparison

Status: implemented; matched campaign running as recorded below. No production defaults changed. This is the next bounded optimization diagnosis after [the exposure pilot](gru-exposure-diagnosis.md), not a repeat of the shallow 192-update screen and not a queued-control experiment.

## Question

The delayed per-choice pilot learned 100% paired training ranking and 81.25% development ranking after 512 additional paired updates, while the all-valid-denominator arm stayed at 50%. Their informative gradients differ by a constant 27 at the fixed 2-second/27-decision training length, but the original clip threshold 1 made 85.74% of per-choice updates clip versus 0% of all-valid updates. That confound is the most useful next intervention. The immediate-cue arm's failure also means one favorable endpoint is not a robust default.

Keep the GRU, full production visual/action dimensions, source images and frozen common visual map. Use the same delayed four-layout paired source, same batch of both cues, same 512 additional updates and exact data order. Three **independent temporal/action initializations** are required: 834, 835, 836. For each, reproduce the same 192-update preparatory history that produced the pilot's immediate-study C192 starting point, then reset AdamW moments exactly as in the pilot. Reuse the saved 834 preparation only after confirming its identity/configuration. Merely changing sampling RNG while loading identical weights is not a seed replication. A common frozen visual map makes this a three-head-seed diagnostic, not the three-full-model-seed production gate.

## Three informative arms per seed

All arms mask waiting losses for this diagnostic; they differ only as specified below. Keep LR 3e-4, weight decay .01, AdamW epsilon 1e-8, bias correction, T512, visual weights and model/data initialization fixed within each seed.

| Arm | Gradient denominator | Clip norm | Purpose |
| --- | --- | ---: | --- |
| A | 54 valid decisions | 1 | Reproduce the failed delayed baseline |
| B | 2 supervised packets | 1 | Reproduce the successful delayed pilot |
| C | 2 supervised packets | 27 | Change clipping alone relative to B; match A's effective threshold in unscaled gradient units |

B versus C isolates the clip threshold with the same gradient units, Adam epsilon and loss. A versus C checks the near-scale-equivalent path without conflating a 27× gradient change with a 27× stronger effective clipping rule. This three-arm design is more informative than only repeating A/B, and avoids spending a fourth arm on another near-equivalence condition before these outcomes are known.

Do **not** call A/C mathematically identical. The installed MLX clip implementation uses `min(max_norm/(norm+1e-6),1)`, and AdamW uses epsilon 1e-8. Scaling gradients and max_norm by 27 changes those additive constants relative to the gradients; weak recurrent directions may matter even though most previously measured active moments were not epsilon-dominated. Record these constants and per-group norms. If A/C diverge while B/C do not explain the effect, investigate scale-aware epsilon/stabilizer controls next; do not relabel that result as proof of a denominator-only mechanism.

Each arm gets 512 updates, 1,024 choice packets and 27,648 valid decisions. The nine-arm comparison totals 4,608 additional updates and 9,216 supervised choice packets, excluding the explicitly matched preparatory histories. Persist each endpoint and its exact exposure/order counters. A wall cutoff saves complete boundaries and leaves an arm incomplete; it never substitutes an unmatched endpoint. Use existing verified caches and coordinate GPU ownership with application work; no new execution budget is authorized by this document.

## Readouts and decisions

At fixed checkpoints, report training and already-used development paired margins/accuracy, complete choice-packet NLL, pre/post-clip norms and clipping fraction, cue-sensitive temporal gradients and the action-factor contributions. Evaluate original passive histories at 2/8/30 seconds; changing only evaluation delay tests retained information without introducing new training data. Preserve source/cache/initial-weight fingerprints and exact update/example counts.

At each final endpoint also decode the actual greedy packet, and a fixed small set of sampled packets, at the known choice window after the same passive history. Report END, wrong/invalid targets and command/timing failures. Correct-versus-wrong likelihood ranking can improve while both complete packets remain unlikely. This readiness-gated readout is still not freely acting success; do not hide premature-action/idle failure by promoting it to that claim. Reserved seeds 2000–2127 remain untouched.

- If B reliably succeeds while C/A fail, stronger effective clipping is the leading mechanism within this frozen, choice-only diagnostic.
- If B/C succeed but A fails, raw scale/epsilon/precision effects need isolation before changing a production loss.
- If seed outcomes are inconsistent or training fit fails, retain the uncertainty and inspect representation/optimizer sensitivity. Do not conclude GRU incapacity from these outcomes.
- If training fits but development layouts do not, broaden paired layout diversity and test trainable non-pretrained visual/readout layers next.

No choice-only arm is a deployable BC objective: waiting calibration remains untrained. A production candidate must retain idle/action supervision and be checked with end-to-end encoder learning and reloaded closed-loop behavior. Queue feedback is independently needed for delayed computer control, but cannot rescue a passive history that never contains issued actions. Keep the two investigations separate.

## Reproducible runner — September 27

`scripts/gru_loss_clipping.py` implements separate `audit`, `initials`, `train` and `evaluate` commands. The audit uses CPU only, pins the runner/model/optimizer source hashes and dependency versions, checks every retained episode and frozen visual file, and authenticates the saved seed 834 C192 preparation against the exact 192-update order before reuse. The other two head seeds start independently and use the same order seeded 834. Initial preparation performs zero optimizer updates. All arms discard preparatory optimizer moments and start with the same fresh AdamW state and RNG seed.

Each `train` invocation selects one seed/phase and is bounded to at most 360 seconds. It resumes only its exact plan/phase/seed checkpoint, snapshots every 64 updates and at the final/paused boundary, and rolls back a failed optimizer application before saving. `--stop-after` selects a pause boundary without redefining the 192/512-update research targets. A campaign lock prevents concurrent GPU owners. Evaluation is a separate process and does not advance its optimizer.

Example commands (choose a new campaign directory for the initial audit):

```sh
.venv/bin/python scripts/gru_loss_clipping.py audit --root .local/my-gru-study
.venv/bin/python scripts/gru_loss_clipping.py initials --root .local/my-gru-study
.venv/bin/python scripts/gru_loss_clipping.py train --root .local/my-gru-study --seed 835 --phase preparation --seconds 360
.venv/bin/python scripts/gru_loss_clipping.py train --root .local/my-gru-study --seed 835 --phase B --seconds 360
.venv/bin/python scripts/gru_loss_clipping.py evaluate --root .local/my-gru-study --seed 835 --phase B --seconds 120
```

Default inputs refer to the retained private generated studies; explicit CLI paths support another authenticated copy. The first two commands do not train. Seed834 reuses its verified existing preparation; 835/836 must reach 192 before their arms can start. Run each A/B/C arm to 512 for all three seeds before comparing endpoints.

The runner records per-update loss, exact pair identities/exposures, pre/post-clip and module gradient norms, clip multipliers and cue-summary gradients. Evaluations retain full-packet correct/wrong factors and separate actual greedy/two-sampled packet outcomes. These packets run through the real isolated practice scheduler **after a passive readiness history**, followed only by empty packets; they do not represent autonomous waiting behavior. Unused return rasters are cached during physics stepping, and the true readiness raster, control state and oracle label are checked against the ranked source before execution. No clock/state jump or future policy input is substituted.

The active campaign is `.local/gru-loss-clipping-2026-09-27`. Its CPU audit verified 384 episodes and 384 visual entries (2,931,607,680 bytes), including 53,888 preparatory decisions / 384 supervised packets. Independent initial checkpoints and a real one-update 835 save/resume/readout check were completed before sustained training. The matched campaign was then explicitly authorized; results are pending until complete endpoints have been evaluated. Reserved test layouts remain untouched and production defaults remain unchanged.
