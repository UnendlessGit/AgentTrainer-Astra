# Zero-wait cue-to-choice mechanism probe

September 27, 2026. Approved as one bounded follow-up to [the completed clipping study](gru-loss-clipping-results.md). This uses existing model schema 2 and its GRU only. It is independent of the new opt-in queued-control channel and changes no production defaults.

The selected starting point is the existing 836-B endpoint (704 total prior head updates), with fresh AdamW moments/RNG. The sole run has 512 updates and 1,024 supervised choice packets, learning rate 0.0003, weight decay 0.01, per-choice normalization and clip norm 1. Four paired training layouts (0–3) retain the prior fixed order; eight previously used development layouts are 1000–1007. Reserved layouts 2000–2127 are not used. This is a selected-checkpoint mechanism probe, not an independent seed-quality estimate.

A script-local `ZeroWaitConfig` permits a zero-duration waiting phase while the production configuration still requires a positive delay. The actual practice renderer, readiness gate, scheduling and rewards use this same zero-delay config. Each new prefix contains the original five visible-cue observations at0–400ms followed by choice at500ms, with no waiting or synthetic combined image. These are newly rendered prefixes with their own times; old episodes are not retimestamped.

CPU preparation passed for all24 paired prefixes. Pixels, physical controls, surface geometry and oracle target labels match the original phase fixtures. The four training pairs additionally match their exact old cached control/features/label entries. The36 visual-cache keys select the original, checksum-verified tensors under the same frozen visual digest. No fresh visual fit or substituted features are allowed. The runner stops if a rendered/source/cache input differs.

The baseline and fixed endpoint evaluate semantic target-region execution first: one actual greedy packet and two categorical packets are each executed separately after the passive zero-delay prefix, followed only by empty packets. Report training/development successes, both-cues-correct layouts, target changes with the cue, complete packet factors and exact-center likelihood ranking. This does not test autonomous waiting. No endpoint selection or second positive-seeking run is allowed.

Frozen inputs, before GPU launch:

- Runner `scripts/gru_zero_wait.py`: SHA256 `2de471583e077bd8b673476f7694cad7fb8be697b12269eb21348d8164669f24`.
- Plan `.local/gru-zero-wait-2026-09-27/plan.json`: SHA256 `321dc52bf498882fb5f1fc654e7173c81f022e91fdf7e23fc2d3de600221e480`.
- Starting policy: `.local/gru-loss-clipping-2026-09-27/checkpoints/ccf81288-1c00-4f16-8fdc-4880e946c5c3`; policy SHA256 `62c0579d9a05ad59fdef0dc9bc09c18045aae3e8b89af713ff7de588f986b251`.

If the short sequence fits with identical frozen features, absence of a usable fixed-feature/action mapping is falsified as a sufficient explanation. If even training does not fit, long-history transport alone is insufficient; that would still not prove visual information is absent because head optimization can also fail. Results will remain separate from broad model-quality gates.

## Fixed endpoint

The single run completed all512updates,1,024supervised packets and6,144valid decisions. Baseline/endpoint actual execution:

| Split | Greedy before → after | Sampled before → after | Both cues correct layouts before → after | Endpoint center ranking |
| --- | --- | --- | --- | --- |
| Training |6/8 →8/8|8/16 →16/16|2/4 →4/4|8/8|
| Development |12/16 →14/16|22/32 →27/32|4/8 →6/8|12/16|

The endpoint changes its selected target correctly with the cue on all four training layouts and six of eight development layouts. Every greedy and sampled packet starts with absolute pointer; there are no END-first, invalid or no-choice failures. The two development greedy failures choose the wrong target on cue0 of layouts1002 and1005. Greedy development success is87.5%, below the90% memory target, and these small, previously used development cases are not a reserved-test result.

Exact-center likelihood again differs from semantic execution. Development dense-cell-only ranking is14/16, while adding within-cell coordinates reduces full oracle-center ranking to12/16. Mean signed cell margin is+6.841nats (mean absolute7.609); within-x/y mean absolute margins are2.841/3.630. Training center ranking is8/8, with mean signed cell margin+11.616 and within-coordinate absolute margins below0.0004. No action-template change explains the gain.

Mean per-choice NLL falls from2.600 over the first16updates to1.777 over the last16. The optimizer clipped179/512updates; maximum pre-clip norm was90.306. The report retains every original update/order/gradient measurement and the final fresh-optimizer state. There was no endpoint retuning or second run.

This establishes a usable frozen-feature/cue-to-action mapping on every training case and most development cases. Absence of such a mapping is therefore not a sufficient explanation of the earlier long-delay training failure. It supports investigating recurrent transport/optimization before unfreezing the visual map merely to explain that failure. It does **not** isolate shortening from512additional updates: no equally extended long-delay arm was trained, and this short-trained endpoint was not evaluated at long delays. It does not establish autonomous waiting or production generalization. No further GPU run or default change follows automatically.

Evidence is in `.local/gru-zero-wait-2026-09-27`: `plan.json`, `started.json`, complete `progress.json`/`result.json`, `baseline.json`, `endpoint.json`, CPU-derived `analysis.json`, `run.log`, and immutable checkpoint `checkpoints/a94e2bfc-4010-46ae-9392-b6f2035f0458`. The final result SHA256 is `b465e3d1b0a22626acc4a56890f6b2b453e5dcc7711b935211ca3f7a05b9be09`; final policy SHA256 is `7af7eea0b7d95f85e7216797f6ece94b782ee7b9330c7f996c68a314fca44f4a`.

The run took37.41seconds including baseline and endpoint execution, with641,324,324bytes reported peak MLX allocation. No intentional competing MLX job ran; root performed Swift/CPU work. This is diagnostic bookkeeping, not deadline/throughput qualification. The process exited zero and released the GPU timing window.
