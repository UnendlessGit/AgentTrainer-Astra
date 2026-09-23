# GRU exposure and optimizer diagnosis

September 19, 2026. Production defaults are unchanged. This is a mechanism
diagnosis using the existing GRU and full production visual/action dimensions,
not qualification of freely acting memory behavior.

## CPU audit

The read-only audit verified the three saved dataset manifests and every episode
checksum. Original/paired/immediate-replay datasets contain 96/192/192 episodes.
Every episode has exactly one nonempty oracle packet among 27, 87 or 307 valid
decisions (2/8/30-second delay). The 192-update unpaired study visits each episode
four times; each paired study visits each opposite cue only **twice**. All runs
consume 53,888 valid decisions and the choice-only runs select 384 packets.
Paired controls, geometry and times match exactly; visual keys differ only for
the intended visible cue observations, while choice labels select opposite
targets. Saved CPU evidence is
`.local/verification/gru-cpu-audit-2026-09-19.json`.

For paired C's final 16 updates at each delay, median gradient norms are
0.552/0.170/0.0482. Normalizing those exact choice-only gradients by supervised
packet count instead of episode length would give 14.89/14.75/14.81. All would
hit the existing norm clip of 1; only 3.125%/1.5625%/0% of the actual updates hit
it. Thus a denominator change also changes global clipping and the relative
weight of mixed-delay examples in Adam's moments. It is not simply an equivalent
learning-rate change.

The installed bias-corrected AdamW uses epsilon 1e-8. Saved paired C moments have
median `sqrt(v_hat)` around 1e-5–1e-4 in the input projection and action routes;
essentially none of those active parameters are epsilon-dominated. About 1.33%
of first-layer recurrent-weight entries are below epsilon. This does not support
epsilon as a broad explanation, though weak individual recurrent directions can
still be suppressed. Fixed global gradient scaling approximately cancels in
Adam away from epsilon; varying batch scales and nonlinear clipping do not.

Frozen cue summaries have 4,096 dimensions and a median opposite-cue RMS
difference of 0.0458 (range 0.0413–0.0547), compared with overall feature standard
deviation RMS 0.0317. The previously saved held-out linear color readout is 100%.
Cue information is visible in these frozen features; that alone does not mean
the recurrent policy has learned to retain and use it for spatial actions.

Code review found no new reset, label alignment or gradient-disconnection bug.
T64 cuts direct cue-to-choice credit for 8/30 seconds as expected; carried hidden
state is detached between chunks. Episode resets/padding are explicit, and the
staged backward engine keeps the complete configured temporal VJP. The existing
T512 and immediate-cue controls already rule out truncation as the sole cause of
the earlier chance results. Gate-convention/initial-retention experiments were
already completed; they were not repeated here. Production BC also trains
random visual layers, whereas the frozen study deliberately freezes them all.

## Matched bounded pilot

`scripts/gru_exposure_pilot.py` starts each arm from the same saved geometric-GRU
immediate-study C192 checkpoint, keeps its entire visual map frozen, and trains
the existing temporal/action modules. Every arm uses fresh identical AdamW
moments, MLX seed 834, learning rate 3e-4, weight decay .01, clip norm 1, a full
T512 horizon and the same deterministic layout order. No optimizer from the
starting checkpoint is resumed. Each arm reloads the identical starting weights;
no preceding arm's updates are carried forward.

Four training layouts (seeds 0–3) each have both original rendered cues and a
2-second delay. Every arm adds 512 paired updates: 1,024 supervised packets,
27,648 valid decisions and 128 additional exposures per cue. The replay arm adds
only the actual rendered cue in the final waiting observation. Labels, controls,
source times, episode lengths and all other inputs remain equal. Validation uses
those four training layouts and eight previously used development layouts
(1000–1007), both cues, original/replayed inputs and 2/8/30-second delays. Reserved
test seeds 2000–2127 remain unused.

The pilot completed all three arms in 225.15 seconds, peaking at 981,571,700 MLX
bytes. Each endpoint was saved before interpretation. The report and checkpoints
are under `.local/gru-exposure-pilot-2026-09-19/`; the log is
`.local/verification/gru-exposure-pilot-2026-09-19.log`.

| Arm | Loss denominator | Updates clipped | Training paired accuracy | Development paired accuracy |
|---|---|---:|---:|---:|
| Original delayed cue | 54 valid decisions | 0% | 50% | 50% |
| Original delayed cue | 2 supervised packets | 85.74% | 100% | 81.25% |
| Immediate cue replay | 2 supervised packets | 87.50% | 50% | 50% |

Each accuracy is the correct-versus-wrong **complete packet likelihood ranking**
with an original passive observation history. These are not sampled or greedy
packet executions and not autonomous success. Each row has the same accuracy at
all three delays and with/without the evaluation cue replay. The successful arm
ranks both cues correctly for all four training layouts and five of eight
development layouts. It was trained at 2 seconds only.

## Consequences and next discriminating comparison

The successful arm proves that the existing geometric GRU and frozen full visual
representation can learn a cue-conditioned action ranking that survives a
30-second delay. It rules out an inherently disconnected decoder or unavailable
cue information as sufficient explanations of all prior chance results.
Increased exposure alone did not fix the current normalization in this pilot.
Conversely, the immediate/per-choice arm failed, so one successful run does not
establish a robust normalization remedy or show that replay is harmful. The
result exposes substantial optimizer/representation interaction and sensitivity.

Next, hold effective clipping constant while testing denominator scaling, using
matched initial model seeds and layout orders. At the fixed 27-decision training
length, per-choice gradients are exactly 27 times the all-valid gradients before
clipping. Pair clip thresholds accordingly and report Adam epsilon and the
clip implementation's norm stabilizer; otherwise the ablation changes multiple
optimizer behaviors. Replicate the informative pilot with independently seeded
temporal/action initializations 835/836 and matched preparatory training histories
(changing only the RNG seed while loading the same weights is not a model-seed
replication),
then broaden paired layouts and compare explicitly trainable non-pretrained
visual/readout layers against the frozen control. Use training-fit curves,
cue-direction/context gradients, full-packet and spatial-cell factors, and
reloaded autoregressive execution alongside held-out rankings.

A choice-only objective leaves waiting behavior unqualified and must not replace
production imitation loss on this evidence. A production proposal must retain
calibrated idle/action learning, adequate long-horizon credit, independent-session
evaluation and ordinary closed-loop tests. Neither the requested three-seed
quality gate nor the 90% held-out memory target is complete.
