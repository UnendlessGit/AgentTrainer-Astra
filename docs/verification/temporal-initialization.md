# Experimental GRU initialization

2026-09-11. **Keep production defaults unchanged.** A matched 27-update production-model comparison did not improve held-out memory success. The experimental initialization improves an untrained retention mechanism; that result does not establish learned behavior quality.

## Derivation and independent review

The installed MLX 0.32.2 GRU uses gate order `r,z,n` and `h_next = (1-z) * candidate + z * h`. Its native input and candidate-recurrent biases are uniform near-zero draws, not exact zeros. Tallec and Ollivier derive gate initialization from approximate retention times under centered inputs and states. In MLX's complementary carry-gate convention, `b_z = log(H-1)` gives nominal `z = 1-1/H`. Their initialization samples timescales uniformly; deterministic geometric spacing is this experiment's multi-timescale variant. These values initialize trainable biases, not fixed retention guarantees. [Primary paper, section 2](https://arxiv.org/abs/1804.11188).

The experiment spaces 512 unit horizons geometrically from 1.01 to 512 decisions. The finite lower endpoint avoids `log(0)`, while covering fast responses and longer context. Only `temporal.layers.*.b[H:2H]` changes. Reset/candidate biases, recurrent weights, the packet decoder's separate GRU, and every visual/action parameter remain unchanged. No production initializer, `ModelConfig`, checkpoint schema, backward engine or trainer was modified.

An independent read-only review confirmed the gate sign/slice, finite FP32 endpoints and experimental isolation. It required exact initial-parameter and completed-update parity before a learning comparison. An analytic zero-weight GRU check verified both the state and VJP against `z^305`.

## Three-seed numerical mechanism check

`scripts/qualify_retention.py` used numerical-test-small models with seeds 834/835/836 and actual 64-pixel cue/wait encoder features. It measured a fixed random projection's derivative with respect to the cue summary after the complete recurrent sequence. This is an untruncated sensitivity diagnostic, not TBPTT training or learned success.

| Delay | Native bias: median cue-gradient norm | Exact zero update bias | Geometric horizons |
|---|---:|---:|---:|
| 2 s | 1.569e-3 | 1.867e-3 | 3.400e-2 |
| 8 s | 1.134e-9 | 3.861e-9 | 1.317e-2 |
| 30 s | 0 | 0 | 3.793e-3 |

The 27 cases plus analytic gate check completed in 3.29 seconds. They justified a bounded production-model experiment, not changing the default.

## Matched production-model comparison

Both arms used the original full visual configuration and native 1280×720 frames, ordinary pointer/button/arrow vocabulary, model seed 834, training seeds 0–31, held-out seeds 1000–1031, two lanes, T64, identical AdamW settings, and the default one-epoch pretrained freeze. The geometric initial checkpoint was checked against the saved native initial checkpoint: all 290 parameter leaves were bitwise equal except the two intended update-bias slices. Model/action manifests and policy signatures were also equal.

The native arm had 27 updates/1,458 decisions: 16 updates with the backbone frozen and 11 unfrozen. The geometric arm's first wall-limited attempt stopped at 26 updates. Its sampler, optimizer, RNG and recurrent state were resumed without reinitialization for exactly one more update. The final checkpoint reports `updateTargetReached=true`, 27 updates and 1,458 decisions. Serialized training configuration, dataset identity, epoch, sampler state and pending accumulation counts match the native arm. The first attempt took 550.86 seconds including its intermediate evaluation; the bounded resume and final evaluation took 103.01 seconds. The intermediate 26-update score is not the matched result.

| Final 27-update policy | Greedy successes | Timeouts | Mean decisions |
|---|---:|---:|---:|
| Native initialization | 17/32 | 0 | 41 |
| Geometric horizons | 17/32 | 0 | 31 |

Both remain well below the 90% target. This single-seed, short-training comparison supports no claim of improved accuracy, even though action timing differed.

## Paired-cue results before and after training

Each checkpoint was evaluated on 32 matched held-out layouts with both possible cues. Cue-free pixels and control features were identical within a pair. Only oracle evaluation labels differed; labels never entered the actor's observations. All four checkpoints preferred the same choice when the cue was flipped: balanced forced-choice accuracy stayed **50% at every delay**.

Mean RMS difference between the paired recurrent states:

| Policy | 2 s | 8 s | 30 s |
|---|---:|---:|---:|
| Native initial | 1.506e-4 | 3.115e-7 | 3.330e-8 |
| Geometric initial | 1.642e-3 | 4.996e-4 | 1.277e-4 |
| Native, 27 updates | 7.677e-5 | 1.302e-5 | 3.626e-7 |
| Geometric, 27 updates | 6.541e-5 | 1.912e-5 | 5.252e-6 |

Geometric initialization produced much greater initial long-delay sensitivity. That sensitivity decreased during training, and its remaining advantage at 30 seconds was not decoded into cue-dependent choices. Initialization alone did not solve this experiment. The result does not identify whether inadequate credit, visual/temporal representation learning, packet-loss imbalance or training duration is the dominant cause.

A lossless private phase cache now reuses exact cue/wait/choice frames across checkpoints. Its key includes environment and renderer-source fingerprints; creating each entry checks every waiting frame's pixel equality. Cached and uncached native-checkpoint results were identical across all three delays for the checked layout. Caching reduced the 32-layout probe from 96 seconds to roughly 11–14 seconds. This cache is diagnostic data, not a production observation shortcut.

## Artifacts

- Scripts: `scripts/qualify_memory.py`, `scripts/qualify_retention.py`.
- `.local/verification/gru-retention-numerical.json`.
- `.local/verification/memory-geometric-27.json` (26-update intermediate; explicitly unmatched).
- `.local/verification/memory-geometric-27-resumed.json` (matched final result).
- `.local/verification/memory-native-initial-probe.json`.
- `.local/verification/memory-geometric-27-initial-probe.json` and `memory-geometric-27-final-probe.json`.
- Final geometric checkpoint: `.local/verification/16004863-8c3d-441d-b20d-a53261cb1e48`.

Only experimental scripts and evidence changed. The next study must separate gradient horizon, loss weighting and temporal architecture before any production choice.
