# Experimental causal observation transformer

2026-09-12. The experimental implementation and longest-sequence numerical pilot pass. No transformer learning campaign has run, no production model/configuration/format changed, and no held-out memory improvement is established. The [short-credit readout evidence](immediate-cue-diagnostic.md) shows why adequate matched exposure is necessary before comparing architectures.

## Architecture and independent rationale

`experiments/causal_transformer.py` implements four pre-LayerNorm attention/MLP blocks, width 384, six 64-wide heads, and a 4× GELU MLP. A learned projection and LayerNorm expose the existing 512-wide interface. The original common checkpoint's packet decoder and value MLP are shared exactly. The same full-resolution visual summaries, observed controls, elapsed-time features and configured context embeddings enter the temporal module; no demonstrated action, cue category or answer enters attention.

Pre-LayerNorm is a conservative initialization choice with documented gradient benefits, rather than a promise that learning-rate tuning is unnecessary for this task. [Xiong et al., 2020](https://arxiv.org/abs/2002.04745). Rotary query/key positions represent relative observation order without a learned maximum-position table. [Su et al., 2021](https://arxiv.org/abs/2104.09864). Positions count valid observations and restart on episode reset; the unchanged explicit elapsed-time features still represent irregular real intervals. All parameters, visual features, attention scores and cache values remain FP32. These choices are experimental and require task-specific learning evidence.

Each layer has a fixed 512-slot key/value cache. Queries attend to themselves and at most 511 preceding valid keys from the same episode. At a call boundary, the latest 512 valid keys/values are retained, chronologically right aligned. Fixed allocation gives stable incremental shapes; validity and positions identify unoccupied slots. Padding produces zero output and preserves every cache tensor, including when its reset flag is set. An active reset excludes all earlier episode keys and restarts positions before processing that observation.

The sequence mask combines validity, call-local episode identity, nonnegative relative key age and age below 512. It applies at every layer. An all-padding query has exactly zero attention weights and finite gradients. Sequence and incremental calls use the same attention computation and masks. Attached cache tensors carry gradients across chunk boundaries; the caller must explicitly detach them at a TBPTT boundary. Cache state contains no action labels.

A 512-slot per-layer cache is a direct-key/storage limit. Higher-layer cached representations already summarize prior inputs. With four such layers, the maximal raw-input receptive field can reach 2,045 observations. Reconstructing state from only the last 512 raw observations would therefore be inexact.

## Numerical checks

Eleven tests in `experiments/test_causal_transformer.py` pass alongside eleven existing frozen-visual tests (22 total, 1.72 seconds):

- Full sequence versus single-step contexts, values and retained state through eviction, active resets and irregular padding.
- All-padding chunks, including padded reset flags, preserve every state tensor exactly and output zero.
- Active reset matches a fresh episode and cuts all previous-episode gradients.
- Future visual/control changes cannot change earlier outputs; future-key gradients are exactly zero.
- A one-layer control enforces its precise direct-key horizon.
- Attached chunk caches match full input gradients; explicit detachment cuts earlier credit without changing forward output.
- Irregular attached chunks, including an all-padding chunk and cache eviction, match full parameter gradients.
- An independent finite-difference input derivative agrees with autodiff.
- Staged decoder/temporal VJPs agree with a monolithic packet objective for natural and choice-only losses.
- The shared decoder/value parameters are initially identical, and mismatched cache shapes or reduced-precision input are rejected.

The cached visual tests separately cover equality with actual full encoding, data causality, corrupted/foreign caches, cancellation, exact episode labels and complete-boundary resume. The transformer remains outside the production checkpoint loader; no compatibility claim follows from using the shared output interface.

## Longest-update pilot

`scripts/pilot_transformer.py` uses two complete opposite-cue ordinary episodes on layout seed 200 at the 30-second delay: B2×T307, 614 valid decisions. Input pixels remain 1280×720 and all global/detail/cursor settings are unchanged. Frozen visual weights come from `.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171`; the visual digest is `d84ebc08774ae15d636653fc2c62892fa6d5d09e44ea3a93c33c417d70e68883`. Shared visual, initial decoder and initial value weights are checked bitwise. Model seed is 834.

The pilot performs two actual clipped AdamW updates for each objective from the same fresh initialization. Learning rate is 3e-4, weight decay .01, clipping norm 1. Both losses divide by all 614 valid decisions. Choice-only selects the two informative packets; natural loss selects all 614. The finite, nonzero gradients reach original cue summaries and change temporal parameters. No optimizer updates happen between decoder microbatches.

| Objective | First complete update | Second complete update | Peak MLX allocation |
|---|---:|---:|---:|
| Choice-only | 0.314 s | 0.075 s | 1.256 GB |
| Natural packet loss | 3.261 s | 3.357 s | 1.272 GB |

Choice-packet NLL changes from 38.252 to 30.763 between the first two forward passes. Natural mean NLL changes from 2.408 to 0.165; this is dominated by waiting labels and is not memory-learning evidence. The initial natural gradient norm is 77.980 before clipping; the finite parameter update is therefore specifically checked after clipping. These two-update results establish execution and numerical sanity only.

The final pilot process completed in 12.23 seconds with the 10 GiB MLX allocation guard. Process peak RSS was 0.557 GB; that OS measure is reported separately from MLX's GPU allocation accounting. An earlier pilot without the incremental timing portion also completed successfully; both reports are retained. These timings use frozen cached visual features and cannot be called application training throughput.

An occupied 512-slot actor cache was then exercised for 20 measured single-step calls after two warm calls. **Temporal-only** latency was median 1.436 ms, p95 3.006 ms, maximum 3.196 ms. This excludes visual encoding, packet sampling, capture, IPC and synchronization with other workers; it is not an end-to-end actor deadline qualification.

## State and parameter footprint; production promotion gate

| Item | Raw tensor bytes |
|---|---:|
| Original GRU temporal parameters | 21,897,220 |
| Experimental transformer temporal parameters | 36,285,956 |
| Unchanged action decoder parameters | 4,122,680 |
| Original GRU actor state, B1 | 4,096 |
| Transformer actor state, B1 | 6,294,020 |
| Transformer training cache, B2 | 12,588,040 |
| Pilot optimizer tensors | 80,817,320 |

The transformer has 9,071,489 temporal parameters versus the GRU's 5,474,305 (+65.7%); capacity is not matched. Counts come from actual parameter tensors, not the GRU-specific `ModelConfig` estimator. Its KV tensors alone occupy 6 MiB per actor, 1,536× the GRU state, plus position/validity metadata. Saving all 512 per-step state anchors would exceed 3 GiB raw before serialization overhead. That is unacceptable as a mechanical replacement for current JSON GRU anchors.

Promotion requires a distinct generic state contract and schema: actor-owned caches, bounded binary/sparse rollout anchors, exact recurrent replay and cache reconstruction/burn-in semantics, integrity/ownership checks, and reset/policy-version isolation. PPO must reconstruct contexts using current policy weights: cached behavior-policy keys/values cannot silently stand in for updated-policy context. The current GRU collection format rejects the cache's dimensions and integer/Boolean metadata, and its 1 MiB IPC envelope cannot carry this state. Do not widen the message limit and start emitting multi-megabyte per-step JSON. Training/checkpoint resume must preserve experimental architecture, cache policy, optimizer/RNG identity and complete update boundaries. Long-lived valid-position counters and FP32 rotary-position precision also need explicit bounds/rebasing before supporting arbitrarily long unreset episodes. Those integrations are not implemented by this prototype.

An independent read-only review of the implementation, all eleven attention tests and the pilot found no blocker for this bounded experiment. It confirmed the reset/window/gradient coverage and per-objective checkpoint reloads, and identified the cache-transport/PPO/position promotion gates above. This review does not substitute for a learning campaign or a production integration review.

## Reproduction and next gate

```sh
.venv/bin/python -m pytest experiments/test_causal_transformer.py experiments/test_temporal_study.py -q
.venv/bin/python scripts/pilot_transformer.py \
  --output .local/temporal-transformer-study/pilot-final.json --seconds 180
```

Private evidence lives in `.local/temporal-transformer-study/pilot.json`, `pilot-final.json` and `.local/verification/transformer-pilot.log`. No experimental checkpoint is registered in the app catalog.

The next learning comparison needs sufficient and equal episode exposure for GRU/transformer, balanced opposite-cue training layouts, identical full frozen visuals and initial packet decoder, unchanged loss normalization and optimizer settings, and training-fit plus held-out evaluation. A favorable single seed or two-update loss decrease cannot choose a production architecture. The final reserved layouts and three-seed qualification remain untouched.
