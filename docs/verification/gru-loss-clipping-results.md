# Three-head-seed GRU clipping results

September 27, 2026. All nine planned endpoints completed. **No production loss, initializer or model change is supported by these results.** The 90% memory target remains unmet. This is a frozen-visual optimization diagnosis, not proof of weak GRU capacity or autonomous memory behavior.

## Matched work

The [planned comparison](gru-loss-clipping-plan.md) used independent temporal/action initializations 834/835/836 with one common immutable production-size visual map. The existing 834 C192 preparation was authenticated against its original checkpoint, dataset and complete 192-update order. Seeds 835/836 each completed that same 192-update preparation order: 53,888 valid decisions and 384 supervised packets. Each A/B/C arm then used fresh identical AdamW moments/RNG and 512 further paired updates: 27,648 decisions and 1,024 supervised packets. Every arm's exact episode-ID sequence matches the plan. The nine arms total 4,608 updates and 9,216 choice packets; no partial endpoint is substituted.

A divides choice-only gradients by 54 valid decisions and clips at 1; B divides by 2 choices and clips at 1; C divides by 2 choices and clips at 27. All use the same four paired training layouts. Development uses eight previously used layouts, both cues and all three delays. Reserved layouts 2000–2127 were not read. These are three head seeds under a fixed visual map, not three independently trained full visual models.

## Training fit and exact-packet ranking

Entries show paired correct-versus-wrong **oracle-center packet likelihood ranking**, in percent at 2/8/30 seconds. Training has 8 cue/layout cases per delay; development has 16. Last 16 NLL is the common per-choice reporting denominator, measured during the final 16 training updates.

| Seed | Arm | Updates clipped | Train rank 2/8/30 | Development rank 2/8/30 | Last 16 choice NLL |
| ---: | :---: | ---: | --- | --- | ---: |
| 834 |A|0%|50 /50 /50|50 /50 /50|3.199|
| 834 |B|86.72%|62.5 /50 /50|50 /50 /50|3.110|
| 834 |C|0%|75 /75 /62.5|56.25 /56.25 /56.25|3.153|
| 835 |A|0%|62.5 /62.5 /50|56.25 /56.25 /56.25|3.220|
| 835 |B|88.09%|50 /50 /50|50 /50 /50|3.067|
| 835 |C|0%|50 /50 /50|50 /50 /50|3.267|
| 836 |A|0%|50 /50 /50|50 /50 /50|3.197|
| 836 |B|83.40%|62.5 /62.5 /62.5|50 /50 /50|2.982|
| 836 |C|0%|62.5 /62.5 /62.5|56.25 /56.25 /50|3.189|

The models do not consistently fit even these four paired training layouts. This is not merely a train-versus-development gap. Clipping substantially changed the updates but did not produce a reliable improvement across seeds.

## Actual packet execution is a different measurement

After a passive cue/wait history, the actual greedy packet and two categorical samples were each executed through the real isolated practice scheduler at readiness. Only empty packets followed; no autonomous waiting decisions were made. The readiness pixels, controls and oracle labels were checked against the ranked source before each execution. The table reports 30-second outcomes; complete per-delay counts are retained in `summary.json`.

| Seed | Arm | Train greedy | Development greedy | Development sampled |
| ---: | :---: | ---: | ---: | ---: |
| 834 |A|50%|50%|59.38%|
| 834 |B|62.5%|50%|53.13%|
| 834 |C|62.5%|50%|59.38%|
| 835 |A|50%|50%|62.5%|
| 835 |B|50%|50%|50%|
| 835 |C|50%|50%|53.13%|
| 836 |A|50%|50%|59.38%|
| 836 |B|75%|68.75%|71.88%|
| 836 |C|50%|50%|56.25%|

Every endpoint readout began with an absolute-pointer operation: 648 greedy packets and 1,296 sampled packets across all splits/delays. There were **no END-first, invalid-packet or no-choice outcomes**; every unsuccessful packet chose the wrong target. The remaining problem here is target selection, not failure to emit an action template. These correlated, small readouts are not independent evidence of production success; do not pool repeated delays into a larger claimed test set.

Seed 836 B genuinely retains useful cue information for some layouts: at 30 seconds, both cues cause correct opposite-target clicks on development layouts 1003/1006/1007. Its greedy result is 11/16 despite its exact full-packet ranking being 8/16. Ranking two prescribed center packets therefore misses some semantically correct off-center clicks.

The saved action-factor margins explain part of that mismatch. At30seconds, development mean absolute margins across arms are approximately 1.15–1.85 nats for the dense-cell factor, but 1.41–2.41 nats for each within-cell coordinate factor. On training layouts, within-cell absolute margins are at most 0.015 nats. Operation/time/button contributions to the correct-versus-wrong difference are negligible. For 836 B, cell-only center ranking is 11/16, while adding coordinate factors reduces complete-packet ranking to 8/16; its mean signed cell margin is +0.969 nats. These are post-hoc decompositions of existing scores, not extra training. Target-region execution/cue flips should remain primary alongside NLL, not be replaced by exact-center ranking.

## The earlier favorable 834 pilot did not reproduce

The prior delayed/per-choice pilot achieved 100% training and 81.25% development ranking. The new 834 B does not reproduce that endpoint. `pilot-reconciliation.json` confirms identical initial policy SHA, visual digest, source manifest hashes, layout order and valid-exposure counts. First losses/norms agree exactly. Through update 256, maximum choice-NLL difference is 3.62e-6; its difference first exceeds 0.001 at update 367 and reaches 0.796 by update 512. The corresponding A trace stays within 3.22e-6 through 256 and first exceeds 0.001 at 478.

Small early numerical differences precede large late trajectory divergence. This establishes sensitivity of the positive result, but does not establish the source of those differences; process boundaries, added serialization/diagnostics and execution scheduling are not identical to the old runner. Neither a fresh positive retry nor endpoint retuning was performed. The old positive checkpoint/report and the new negative replication are both retained.

## Next minimal diagnostic

Before another long-delay campaign, use one **zero-wait cue-to-choice control with identical frozen features**. Retain the original five visible-cue observations followed immediately by the original choice observation, with no intervening waiting frames. Use an actually rendered zero-delay practice source and verify its cue/choice pixels, controls and cached features against the existing source. Keep the full action vocabulary and original target locations; do not add a privileged cue scalar, a future teacher action, or a new combined cue/choice image whose nonlinear visual features would confound the comparison.

Start from the informative 836-B checkpoint with fresh optimizer state, the same four paired layouts and a fixed 512 updates / 1,024 choice packets; keep per-choice normalization and clip 1 so shortening the sequence does not silently rescale gradients. Judge semantic target-region execution, both-cues-correct layouts and all packet factors on train/development layouts, not only exact-center ranking. This is a selected-checkpoint mechanism probe, not another seed-quality claim. This control was subsequently authorized and completed as one frozen run; see [zero-wait results](gru-zero-wait-control.md). The proposal below records the interpretation chosen before that run.

If the short control fits training and development with the same feature values, absence of a usable frozen-feature/action mapping is falsified as a sufficient explanation; long-history transport or its optimization becomes the next intervention. If it cannot fit even training, long-delay transport alone is insufficient, and a matched unfreeze of the non-pretrained visual/spatial map with the backbone frozen becomes useful. Failure would still not prove information is absent: head optimization can fail too. This removes the long prefix entirely, unlike the previous final-wait cue replay that retained an already evolved recurrent state before the cue/choice transition.

## Evidence and resource context

- Runner: `scripts/gru_loss_clipping.py`; CPU-only endpoint checker: `scripts/summarize_gru_loss_clipping.py`.
- Immutable campaign input hashes/order: `.local/gru-loss-clipping-2026-09-27/plan.json`.
- Initial/checkpoint identities: `initials.json`, `run-<seed>-<phase>.json` and `checkpoints/` in that directory.
- Per-update loss/clip/module/cue gradients and all completed exposures are in each run report and its saved optimizer state. Endpoint factor margins, decoded commands and actual execution outcomes are in `evaluations/`.
- `campaign-progress.json`, `summary.json`, `factor-analysis.json` and `pilot-reconciliation.json` bind the interpretation to the actual saved checkpoints and traces. Training reports retain their at-publication `validationPending` flag; subsequent immutable evaluations and `summary.json` provide completed endpoint evidence.

The nine-arm driver plus three preparation baselines took 645.37 seconds (502.47 training / 142.88 evaluation); new preparatory invocations took approximately 131 seconds in total, with a separate one-update/readout check. Maximum reported MLX allocation was 3.247 GB during preparation and 1.134 GB during the short-delay arms. Native CPU-only Swift build/checks ran during part of the campaign; there was no competing intentional MLX/model workload. Times are diagnostic bookkeeping, not inference-deadline or throughput qualification. The campaign exited successfully; no GPU study process remains active.
