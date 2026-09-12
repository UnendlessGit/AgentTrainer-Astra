# Proposed temporal credit and loss study

Status: **Stage 1 GRU tranche, paired follow-up and short-credit diagnostic completed. Further temporal work is GRU-only.** See [current GRU evidence](temporal-study-progress.md), [readout diagnosis](immediate-cue-diagnostic.md) and the [current scope decision](temporal-scope.md). Production training remains a separate future gate. The preceding matched initialization result is in [temporal-initialization.md](temporal-initialization.md).

## Question and interpretation

The present full-model checkpoint learns to execute choices but scores 50% on matched opposite-cue trials. Geometric gates increase initial cue retention without improving 27-update success. The next study must distinguish (a) inadequate gradient horizon, (b) the rarity of informative action labels, and (c) temporal representation. An additional near-chance short full-model run would not resolve those alternatives.

At 100 ms cadence, the ordinary oracle supplies one non-END packet per 27, 87 or 307 decisions at 2/8/30-second delays: 3.70%, 1.15% and 0.326%, respectively. Waiting labels dominate a natural per-decision loss. T64 directly connects cue and choice only for the 2-second task. Carrying a detached hidden state is not equivalent to keeping its gradient history.

## Stage 1: frozen-visual temporal diagnostic

Retain the full production sensory configuration and source pixels: native 1280×720, default global/detail/cursor limits, the pinned pretrained backbone, and unchanged visual pooling/dense pointing representations. Use the visual weights from the saved native initial checkpoint `.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171` as the single common, immutable visual checkpoint across every arm. Cache its exact visual outputs, keyed by source frame, transforms and visual-weight digest. Freeze **all** visual parameters for this diagnostic. This isolates temporal/action learning; it cannot establish production quality with a trainable encoder.

The temporal and autoregressive action modules still consume the same causal control features and produce the complete timed packet likelihood. No cue/answer metadata, manually extracted color features, direct classifier, reduced pointer vocabulary, or shortened packet budget may enter the policy. Cached demonstration visuals must not be used to impersonate freely acting observations, whose cursor and controls can change.

Use 32 independent training seeds per delay: 0–31, 100–131 and 200–231. The current development layouts 1000–1031 remain validation data, with counterfactual cue pairs at all three delays. Reserve 2000–2127 for final unseen-layout testing; do not use it to tune the models. Every arm receives identical episode order, augmentations, labels and model-seed assignments.

Proposed contrasts:

| Arm | Temporal representation | Gradient horizon | Loss |
|---|---|---:|---|
| A | Geometric-initialized GRU, 2×512 | 64 | Natural per-decision packet NLL |
| B | Same GRU | 512 | Same natural loss |
| C | Same GRU | 512 | Choice-packet-only diagnostic loss |
| D | Native-initialized GRU, 2×512 | 512 | Choice-packet-only diagnostic loss |

A–B isolates horizon, B–C isolates label imbalance, and C–D revisits initialization with informative credit. Both objectives divide by the same total valid decisions in the two-episode optimizer batch. Choice-only does not introduce an episode-length gradient multiplier; its separately reported choice NLL divides by selected packets. The choice-only loss masks waiting labels **only in the diagnostic objective**; it does not alter observations, episodes, action labels or the decoder. A policy trained only this way is not a deployable imitation model because its early-action/idle behavior is unqualified.

## Fair optimizer comparison

A T64 run has more chunks than a T512 run. Comparing equal chunk/update counts would therefore change the number of examples and optimizer batch size. Accumulate over the same two complete episodes before each update in every arm. T64 detaches at its cuts; T512 keeps the entire episode connected. The 307-decision maximum fits T512. Freeze parameter updates until the common episode batch ends so forward-state comparisons are meaningful.

Use identical optimizer hyperparameters, clipping, seeds and 192 completed updates initially: four complete passes over the 96-episode training source. Save checkpoints at updates 32/64/128/192. A wall-limited partial run is resumable, not a matched final result. Compare only common completed update/example counts. Start with model seed 834 for mechanism screening, then replicate the informative comparisons at 835/836 before selecting an architecture.

The subsequent tiny immediate-cue control needed hundreds of repeated pair exposures. Thus 192 updates were an initial screening budget, not enough evidence to conclude architectural incapacity. Further paired GRU studies must specify adequate example exposure and inspect training fit; repeating the same shallow budget is not a decisive comparison.

## Required checks before learning claims

- Check exact visual-weight/cache provenance and equality with uncached production encoding on representative cue/wait/choice frames and geometry.
- For the same GRU weights and episode, T64 accumulation and T512 must have identical forward state, packet scores and loss before updating; only the intended gradient cut may differ.
- Verify source/input causality, per-episode state reset, padding, cache eviction, no future-key attention and no demonstration-label state leakage.
- Compare cached and ordinary frozen-vision gradients on a tractable sequence. Check nonzero cue gradients where they should exist, finite gradients and actual parameter updates.
- Report action-time NLL separately from waiting NLL, balanced opposite-cue choice accuracy, cue-flip response, state/gradient sensitivity, and train-versus-validation curves. Overall packet NLL alone is insufficient.

## Budget and stopping decisions

Before a learning campaign, measure one complete update for each execution path at the longest delay and record peak MLX allocation/RSS. Keep the initial allocation guard at 10 GiB and bound cached visual residency; report disk-cache size separately. Do not infer long-horizon memory cost from short-sequence tests.

The first proposed compute tranche is at most 30 minutes for the four GRU diagnostic arms, including cache construction and validation. Each process gets at most 6 minutes before saving at a complete boundary; resume as needed in a later tranche to reach the same 192 updates. This is a budget proposal, not a throughput promise. If the measured pilot cannot fit the tranche, revise the allocation before launching the campaign rather than silently reducing quality or examples.

Success for advancing a candidate requires a clear learning curve and at least 90% balanced validation choice accuracy at all three delays. If even choice-only T512 cannot overfit the training episodes, inspect the feature/gradient/decoder path before adding longer runs. If it overfits training but fails unseen paired layouts, increase demonstration diversity and investigate generalization. If natural loss fails while choice-only succeeds, test an explicitly specified curriculum/calibration strategy before attributing the failure to temporal architecture.

Replicate informative GRU contrasts with three model seeds. No default change follows from one favorable seed. Further runs require a coordinated GPU allocation; prior comparison-campaign approval does not authorize an automatic new tranche.

## Stage 2: production qualification

Only after the diagnostic resolves a useful mechanism, train the selected candidates through the actual full-resolution production BC path with normal encoder training, immutable checkpoints and an explicitly approved loss/curriculum. Hold data, optimizer work and evaluation fixed. Evaluate reloaded policies in genuine closed-loop practice at 2/8/30 seconds, with both greedy and sampled action protocols, bounded answer windows, wrong choices, premature actions, timeouts and action faults reported.

A final memory claim requires at least 90% success on the reserved unseen layouts at every delay and reproducibility across model seeds, with uncertainty reported. Keep the forced-choice diagnostic separate from that claim. Additional native capture/control and broader computer-use evaluation remain independent release gates.
