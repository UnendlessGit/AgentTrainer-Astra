# Matched additional training at a two-second wait

September 27, 2026. One fixed continuation addresses the additional-exposure confound in the [zero-wait control](gru-zero-wait-control.md). Production source, architecture, defaults and prior experimental artifacts remain unchanged. Both continuations start the original 836-B checkpoint `ccf81288-1c00-4f16-8fdc-4880e946c5c3`, whose policy SHA256 is `62c0579d9a05ad59fdef0dc9bc09c18045aae3e8b89af713ff7de588f986b251`. Neither starts the other's endpoint.

The matched settings are fresh AdamW moments/RNG seed 834, learning rate 0.0003, weight decay 0.01, epsilon 1e-8, per-choice normalization, clip norm 1, 512 updates and 1,024 supervised packets. All 512 paired layout/cue orders match the zero-wait arm. The original four training layouts retain their exact two-second wait and 27 observations per episode: 27,648 valid decisions versus 6,144 in zero-wait. Multiplying the existing valid-decision loss gradient by 27 rather than 6 keeps the same per-choice objective. Actual recurrent gradient paths and input sequences necessarily differ; matching chosen-label exposure does not equal matching recurrent computation.

CPU preparation verified all eight original training episodes against genuine rendered observations, physical control features, labels, source times, source checksums and original frozen visual keys. It also verified 48 genuine passive evaluation prefixes: both cues on training layouts 0–3 and development layouts 1000–1007 at 2 and 30 seconds. All 48 required visual cache tensors retain their original checksums and visual digest. Repeated unchanged rasters are cached only after the actual renderer produces that phase; clock, scheduler, controls and oracle behavior still advance. Reserved layouts 2000–2127 are not read.

At the fixed endpoint, each case executes one actual greedy packet and two independently keyed categorical packets after its passive prefix, followed only by empty packets. Sample keys match the corresponding original three-seed-study 2/30-second readouts. Semantic target success, both-cues-correct layouts and target changes with cue are primary; exact-center likelihood and factor margins are secondary. There is no autonomous waiting claim and no extra training/readout selection after seeing results.

Interpretation frozen before launch:

- Fitting training at both 2 and 30 seconds supports additional optimization as sufficient for this selected checkpoint; it does not justify an initializer change.
- Fitting at 2 seconds but failing at 30 points to duration extrapolation, since the continuation trains only the 2-second delay.
- Still failing the 2-second training cases means optimization through history remains problematic under this fixed budget. The next minimal readout would use the already saved zero-wait endpoint at 2/30 seconds before any architectural intervention; that is not another training arm and is not run automatically here.

Frozen files:

- Runner `scripts/gru_extended_wait.py`: SHA256 `471b925b7d2b6bbec929adcd035eb0f4693b1804e634a42635e5c78e3b8cce40`.
- Plan `.local/gru-extended-wait-2026-09-27/plan.json`: SHA256 `f73b3c041853758e561bdee583d97ff83dce8e92a57494353b84c073052ab249`.
- Zero-wait comparator result SHA256: `b465e3d1b0a22626acc4a56890f6b2b453e5dcc7711b935211ca3f7a05b9be09`.

## Fixed endpoint

The single run completed all 512 updates, 1,024 choice packets and 27,648 valid decisions. The negative endpoint is retained without retuning or restarting:

| Split | Delay | Greedy success | Sampled success | Both cues correct layouts | Center ranking |
| --- | --- | --- | --- | --- | --- |
| Training |2 seconds|4/8|8/16|0/4|4/8|
| Training |30 seconds|4/8|7/16|0/4|4/8|
| Development |2 seconds|8/16|16/32|0/8|8/16|
| Development |30 seconds|8/16|17/32|0/8|8/16|

No tested layout changes its greedy selected target when the cue changes. Every greedy and sampled packet begins with absolute pointer; all failures choose the wrong target. There are no END-first, invalid-packet or no-choice outcomes. Dense-cell-only ranking is also 50% for both splits/delays. At two seconds, paired mean signed cell margins are nearly zero (0.00023 training, 0.00026 development), while mean absolute margins are 3.046 and 4.130 nats: the endpoint has a strong target preference with little cue-dependent difference. This measures outputs, not absence of information inside the hidden state.

The original starting checkpoint's preserved study readout achieved 6/8 training and 11/16 development greedy successes at both delays. The additional wait-preserving optimization did not improve that result. First-16 per-choice NLL averaged 2.653; last-16 averaged 4.381. The optimizer clipped 197/512 updates, and the maximum pre-clip norm was 328.46. Complete per-update traces are retained, including lower mid-run losses; no earlier endpoint was selected after seeing the final regression.

By comparison, the matched zero-wait continuation fit all 8 training cue cases and achieved 14/16 development greedy successes at zero wait. Both continuations started the same weights and used the same additional selected-label exposure, order, optimizer settings and clipping. The favorable short-sequence result therefore cannot be attributed merely to receiving 512 extra updates under this tested setup. However, the recurrent inputs, valid-decision counts and gradient paths differ, and this is one selected checkpoint: it does not establish weak GRU capacity or a universal failure of the optimizer/visual model.

The predeclared third case applies: optimization through the two-second history still fails to fit even training under this budget. The next discriminating step is a readout of the already saved zero-wait endpoint at 2/30 seconds, with no retraining, before any architecture or learning-rate intervention. It is separately prepared; this run contains no such readout and authorizes no automatic training retry.

Evidence is private under `.local/gru-extended-wait-2026-09-27`: frozen `plan.json`, `started.json`, complete `progress.json`/`result.json`, `endpoint.json`, CPU-derived `analysis.json`, `run.log`, and checkpoint `checkpoints/d334b658-51f9-44fd-a7e3-1702ea8d8c36`. Result SHA256: `a2557676801652810d9f63aa275923c21b7518d22d61db7a7d9dd08a46ddde1a`. Final policy SHA256: `fd3beaf421b0bca34295e5b2eae092e1108bf0fc76a74970e815703773d3bec7`.

The process exited zero after 65.92 seconds including endpoint execution, reporting 706,864,404 bytes peak MLX allocation. Native CPU regression work ran concurrently; no intentional competing MLX job ran. This is diagnostic bookkeeping, not a throughput/deadline qualification. The GPU window was released before Python regression began. No production file was changed.

Initial preparation encountered an absent legacy `cueReplay` optional field; reading its documented default of false resolved the parser issue before plan freezing or GPU work. No source episode was changed.

## Prepared saved-endpoint readout

The next readout is prepared separately with **zero optimizer updates**. It loads the existing zero-wait endpoint `a94e2bfc-4010-46ae-9392-b6f2035f0458` and imports the frozen extended-wait runner's exact 2/30-second evaluation function, templates, actual packet execution and categorical keys. It does not change either training runner or checkpoint. The plan checks the common starting checkpoint and both completed-result identities, and publication rechecks that the selected checkpoint's bytes are unchanged.

- Launcher `scripts/gru_saved_endpoint_readout.py`: SHA256 `f282e7d510a295f3be1fe74f739945263897b4d239aa942c867e87a21699676f`.
- Plan `.local/gru-zero-endpoint-delayed-readout-2026-09-27/plan.json`: SHA256 `1257ec5d46c6bc0804dfdf42506db31fcc53ca01f8e74237003146ae490ca0c4`.

The single readout completed with zero optimizer updates, and the original checkpoint hashes still match. Results:

| Split | Delay | Greedy success | Sampled success | Both cues correct layouts | Center ranking |
| --- | --- | --- | --- | --- | --- |
| Training |2 seconds|8/8|16/16|4/4|8/8|
| Training |30 seconds|8/8|15/16|4/4|8/8|
| Development |2 seconds|13/16|29/32|5/8|11/16|
| Development |30 seconds|10/16|21/32|2/8|10/16|

All greedy packets begin with absolute pointer. At both delays, every training layout changes its selected target correctly with the cue. The same zero-wait checkpoint previously achieved 14/16 development greedy successes at zero delay; the readout falls to 13/16 at 2 seconds and 10/16 at 30 seconds. Sampled outcomes are small correlated readouts, not a separate independent success estimate.

This is the discriminating result: the unchanged GRU and frozen features can carry the short-trained mapping across the full 30-second prefix on all tested training layouts. A hard inability of this representation to retain that mapping is therefore not a sufficient explanation for the wait-trained endpoint's training failure. The matched continuations instead expose a history-dependent optimization path: equal additional choice exposure succeeds through the short sequence and collapses through the longer training sequence under these settings. This does not prove every optimizer, initializer or long-history objective would behave the same way, nor that visual generalization is fully solved.

Development robustness still degrades with duration. That is a separate unresolved limitation from fitting the training layouts, and it falls well below the broad memory-quality goal at 30 seconds. A controlled curriculum/optimization investigation is better motivated than claiming weak GRU capacity or changing architecture from this evidence. No new training, initializer, learning-rate or production-default decision was made here.

Readout evidence is under `.local/gru-zero-endpoint-delayed-readout-2026-09-27`: `plan.json`, `started.json`, `endpoint.json`, `analysis.json`, `result.json` and `run.log`. Result SHA256 is `3be4be53e7b1cf1987211192a66bdf8c0bb9da8e5205bfe4284d10277f9b9a7a`. The readout exited zero after 12.37 seconds, reporting 495,460,795 bytes peak MLX allocation. It ran after Python regression released the GPU and finished before the development-bundle/privacy workflow resumed. These times are not performance qualification.
