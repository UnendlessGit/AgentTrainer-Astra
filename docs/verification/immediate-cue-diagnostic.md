# Immediate-cue readout diagnostic

2026-09-12. The bounded immediate-cue arm and its tiny readout control are complete. The 192-update multi-layout arm remains at 50% paired held-out accuracy. Repeated training on one opposite-cue pair learns both choices, and the saved policy's greedy packets successfully complete both virtual worlds when admitted at readiness. This distinguishes a working but initially weak cue-conditioned learning path from a disconnected decoder. It does not qualify held-out memory or freely acting behavior. Production defaults are unchanged.

## Actual rendered intervention and parity

The ordinary cue remains visible during the first 500 ms. `ImmediateCueEnvironment` repeats that same rendered cue for the final 100 ms waiting frame immediately before the choices appear. Its render-only clock view selects the ordinary cue raster with the current cursor. Logical time, input scheduling, rewards, oracle labels, episode lengths and decision counts remain unchanged. The model receives coherent BGRA images and causal controls, never a hidden cue category or answer label.

The paired training source preserves both opposite visible cues for each layout. Inputs differ only during the original cue and the one replay frame; all other cue-free pixels, controls and geometry remain equal. A full 192-episode comparison against the ordinary paired source confirmed unchanged commands, controls, geometry, observation times and lengths, with only the intended replay-frame visual keys changed. Visual-cache reuse is allowed because the actual rendered replay pixels match an already encoded cue image in these passive oracle episodes. No incompatible summary/dense/cursor features are combined.

Eleven numerical tests pass in `experiments/test_temporal_study.py`, including cached/full gradients, truncation causality, reset/padding, cancellation, resume, paired-source identity, moved-cursor replay rendering and a finite-difference check of the choice-likelihood context gradient. Validation branches each delay before its final waiting frame, so the 8/30-second cases cannot inherit earlier replay interventions.

A separate direct-rendered validation of held-out layout 1000 and both cues matched the cached evaluator's correct-minus-wrong packet log likelihood to a maximum absolute error of 1.91e-6. This checks that the evaluator used the intended visual intervention; it does not exhaust every possible layout.

## Matched 192-update arm

The arm retains full 1280×720 input, common frozen visual weights, geometric-initialized 2×512 GRU, complete T512 gradient horizon, ordinary packet decoder/capabilities, model seed 834, paired episode schedule and total-valid-decision loss normalization. Choice-only masks waiting losses exactly as in the preceding paired C arm. Both consume the same 53,888 valid decisions and 384 supervised choice packets; only the visible replay frame changes.

The source contains 192 episodes: opposite cues for 96 layouts across 2/8/30-second delays. The manifest hash is `761b630521f35c248efe96cc4403b6c9da96756b3518494059bb657a53aba27e`. Preparation took 176.44 seconds and training plus final validation 105.13 seconds, totaling 281.57 seconds of the separately approved ten-minute control. All 192 updates completed. All three held-out paired scores remain 50%, with zero cue-flipped choices across 32 layouts (seeds 1000–1031). These are forced-choice packet likelihood scores, not freely acting success.

The final checkpoint is `.local/temporal-immediate-study/checkpoints/2531737d-4b0c-44f0-b058-66f87378d335`. The full report is `train-C-834.json` in that directory's parent.

## Connection and tiny-pair learning

`scripts/readout_diagnostic.py` starts from that checkpoint and renders both opposite cues on training layout 0 with a 2-second delay. It keeps both complete 27-decision episodes, the frozen full-quality visual map and ordinary autoregressive packet objective. It trains the existing temporal/action modules with fresh AdamW (learning rate 3e-4, weight decay .01, norm clip 1). The two selected choice losses are divided by all 54 valid decisions, as in the multi-layout arm. Alternative packets are used only to evaluate correct-minus-wrong likelihood; neither answer labels nor alternatives become model inputs.

Before this extra training, the cue-pair temporal contexts differ by only 9.98e-5 RMS. The norm of the choice-likelihood difference gradient with respect to those contexts is nonzero (0.143), while its directional derivative along the actual cue difference is only −3.97e-6. Gradients reach both the original and repeated cue summaries (norms 1.36e-6 and 1.25e-6). Thus the path is connected, but its initial cue sensitivity is weak.

| Measurement on training pair only | Before | After 512 additional updates |
|---|---:|---:|
| Mean choice-packet NLL | 11.751 | 1.332 |
| Correct-minus-wrong log likelihood, cue 1 | 0.679 | 5.991 |
| Correct-minus-wrong log likelihood, cue 2 | −0.679 | 5.819 |
| Temporal context pair RMS difference | 0.000100 | 0.685 |
| Cue-direction likelihood derivative | −0.000004 | 4.948 |

The 512-update repeated-pair run took 34.18 seconds. Most learned choice discrimination is in the spatial cell factor; operation/timing/within-cell factors also fit the target packets. An earlier independent 256-update run took 18.94 seconds and reached margins 0.223/0.001. Both runs began with identical measured scores, but small early FP32 differences grew into different escape trajectories; the second run's 256-update margins were 2.076/1.451. Bitwise training reproducibility and stable convergence rates are not established by these runs.

The saved tiny checkpoint is `.local/temporal-immediate-study/cd427845-130e-4841-a51a-9e37138484d5`. Reloading reproduces its final forced-choice margins exactly. Ordinary greedy packet sampling and `decode_commands` then generated different pointer targets for the two cues. Each packet completed the corresponding virtual training world with reward +1. This check forces empty actions before readiness and afterward; it does not test freely acting waiting behavior, new layouts or retained information at longer delays. It posted no OS input.

## Interpretation and next comparison

The checked source and evaluator wiring agree; the temporal-to-action derivative is nonzero; and the unchanged recurrent/action path can fit this tiny immediate-cue mapping. The evidence therefore does not support a disconnected head or an inherent inability to express these choices. It does expose weak initial cue conditioning and a substantial exposure/optimization requirement. The multi-layout arm visits each cue only twice, whereas the tiny arm repeats each cue 512 times with a fresh optimizer. These budgets are not interchangeable, and successful tiny overfit does not establish generalization.

A future GRU/causal-transformer comparison must give both sufficient matched example exposure, preserve the full visual map and action interface, and report training fit as well as held-out choice and freely acting behavior. The frozen random visual-detail/query layers, optimizer sensitivity, loss normalization and multi-layout readout transfer remain possible limitations. No architectural default change follows from these diagnostics. The reserved unseen seeds 2000–2127 remain unused.

## Reproduction and private artifacts

The matched control uses `scripts/temporal_study.py --paired --cue-replay --arm C --updates 192`; its root is `.local/temporal-immediate-study`, common checkpoint `.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171` and shared visual root `.local/temporal-study/visual`. Reports and logs stay out of Git.

```sh
.venv/bin/python scripts/readout_diagnostic.py \
  --checkpoint .local/temporal-immediate-study/checkpoints/2531737d-4b0c-44f0-b058-66f87378d335 \
  --output .local/temporal-immediate-study/readout-repeat.json \
  --overfit-updates 512 --seconds 120 --greedy-ready
```

Key evidence files under `.local/temporal-immediate-study/` are `direct-render-validation.json`, `readout-connection.json`, `readout-tiny-overfit.json`, `readout-tiny-overfit-512.json`, `readout-tiny-reloaded.json`, `tiny-greedy-ready.json` and the reproducible combined read-only `readout-tiny-reloaded-greedy.json`. All are explicitly experimental.
