# Independent review: exact staged backward

Reviewed on 2026-09-08 against the current visual, temporal, packet decoder, behavioral and recurrent PPO implementations. This review owns no learning implementation changes. The root agent implemented the production execution path in `python/astra/learning/backward.py`; this review now includes regression coverage of that helper. Production-size qualification remains separate.

## Finding and decision

The current `checkpoint_vision=True` path does not bound the full recurrent gradient graph: the guarded production-shape probe measured 8,746,550,737 bytes at four decisions and 16,285,749,279 bytes at eight. Its projected 64-decision peak is approximately 130 GB. These measurements support changing graph execution, while retaining the full visual representation and contiguous recurrent loss horizon.

An explicit chain-rule split is mathematically appropriate. The visual boundary has **two differentiable outputs**, `summary` and `cells`. The temporal path uses the summary, and pointing likelihoods use dense cells directly. Differentiating only the recurrent summary would lose direct pointing gradients and change the trained model.

The current action decoder avoids a `B*T,K,N,D` gathered tensor, but still builds `B*T,K,S,N` logits and their softmax intermediates. A second split across independent decision packets bounds this graph without shortening either the persistent GRU sequence or the autoregressive sequence inside a packet.

## Exact execution schedule

1. Materialize each visual microbatch outside differentiation, evaluating summary, dense cells and immutable geometry/masks together. Retain detached feature outputs, not unevaluated visual graphs. Preserve row-major `B,T -> B*T` order for all image fields, packets and cotangents.
2. Run the complete temporal sequence, preserving all resets, valid-step masks, contexts, elapsed times and the incoming detached state. Materialize context, values and next state. Score decision packets in bounded microbatches and materialize whole-packet log probabilities and conditional entropy.
3. Differentiate the original scalar loss with respect to the small complete vectors of log probability, value and entropy. This keeps PPO clipping, normalization, masking and auxiliary diagnostics identical to the ordinary loss. Metrics, validity counts and next state are auxiliary outputs, not additional loss terms.
4. Recompute each packet microbatch with explicit action parameters, context and dense cells as differentiable inputs. Dot log probability and entropy with their detached cotangents. Accumulate action-parameter gradients and retain context/cell cotangents in the same decision order.
5. Recompute the **complete** temporal sequence with explicit temporal parameters and visual summary as differentiable inputs. Dot context and value with their detached cotangents. This gives temporal-parameter gradients and the full recurrent summary cotangent. Do not detach between decisions inside this stage.
6. Recompute each visual microbatch with explicit trainable visual parameters. Dot **both** summary and dense cells with their detached cotangents. Evaluate and accumulate the resulting parameter gradients before advancing to the next microbatch.
7. Assemble exactly the model's trainable gradient tree, perform finite checks and global clipping after complete accumulation, and apply one optimizer update at the existing effective-batch boundary. This schedule changes execution, not loss weighting or optimizer cadence.

Staged scalar VJPs must preserve loss masking before arithmetic: a rejected or padded decision can have `log_probability=-inf` with zero cotangent, and raw `0 * -inf` is NaN. Apply the original validity mask to log probabilities before dot products; add an explicit masked-invalid-likelihood regression. Do not clamp valid likelihoods or ratios to hide numerical faults.

For BC, the original objective is a **sum** over valid decisions; existing effective-batch accumulation divides once by the accumulated valid count. For PPO, preserve the current `result.total * result.valid_count` gradient construction and later count division. Dividing independently by each visual/action microbatch would change the weighting of incomplete or padded batches.

## MLX API evidence

The installed MLX API supports `mx.value_and_grad(function, argnums=(0, 1, 2))` over a parameter pytree and feature arrays. A scalar `sum(output * stop_gradient(cotangent))` implements the required VJP while retaining the parameter pytree. `mx.vjp` documents flat lists of array primals and cotangents, so it would require explicit stable parameter flattening/unflattening.

A module wrapper must pass `module.trainable_parameters()` explicitly, install that tree while tracing, and restore the previous full `module.parameters()` bindings in `finally`. Capturing module weights without passing them as differentiable inputs is insufficient. Frozen backbone leaves must stay absent from the trainable tree; frozen forward weights remain available through the module. Current visual layers share parameters between the dense and cursor detail paths, and between projection paths. Their contributions must accumulate into the single corresponding leaf.

Evaluate each stage and running gradient sum before advancing. Merely appending lazy VJP expressions to a list retains the graphs the staging is intended to release. Caller-owned policy mutation and optimizer updates must be excluded until every stage has completed. Current modules are deterministic during scoring; future dropout, stochastic depth, mutable statistics or random augmentation would require identical recomputation randomness/state or a rejected execution configuration.

## Independent numerical probe

The standalone local probe `.local/probe_staged_backward_review.py` uses `ModelConfig.test_small()` solely for numerical validation. It compares the original monolithic backward with the complete schedule above using the actual policy and PPO loss. The fixture has two lanes, three steps, two unequal-size surfaces, distinct packet labels, absolute and relative motion, key/button/scroll factors, nonzero incoming recurrent state, an active reset, padded reset, varying contexts, nonzero value loss, entropy and PPO clipping. It runs with both trainable and frozen pretrained backbones.

| Case | Trainable leaves checked | Original loss | Staged loss | Maximum absolute gradient difference |
|---|---:|---:|---:|---:|
| Trainable backbone | 165 | -2.6906995773 | -2.6906988621 | 1.728535e-6 |
| Frozen backbone | 111 | -2.6906995773 | -2.6906988621 | 1.728535e-6 |

Every gradient tensor passed `atol=2e-5, rtol=2e-4`; the process peak active MLX allocation was 17,874,545 bytes. The loss difference is below 7.2e-7. These are finite-precision equivalence results for this fixture, not production memory or learning-quality qualification.

The staged interface must explicitly document its differentiable boundaries. The decoder currently depends on temporal context and dense cells, not directly on `visual.summary`; a future new path through summary, bounds or persistent final state must expand the boundary rather than silently detach that dependency. A monolithic gradient-equivalence regression should cover both BC and PPO, frozen/unfrozen parameters, multiple lanes/surfaces, resets/padding and all active action factors.

## Integrated helper regression evidence

`policy_gradients(policy, observation, packets, objective, state, action_microbatch=..., cancelled=...)` now implements the staged schedule and is used by both training engines. Its objective accepts complete flattened log-probability, value and conditional-entropy vectors and returns `(loss, auxiliary)`. The helper returns the original loss, auxiliary diagnostics, next recurrent state and the complete assembled parameter-gradient tree. `_cotangent_dot` masks zero-cotangent outputs before multiplication.

`python/tests/test_staged_backward.py` compares this production helper against ordinary `nn.value_and_grad` through the complete policy. On 2026-09-08, `.venv/bin/python -m pytest python/tests/test_staged_backward.py -q` passed **9 tests in 1.25 seconds**:

- Behavioral and clipped PPO objectives with both frozen and unfrozen pretrained backbones; every gradient leaf, auxiliary result and next recurrent state matches the reference within the documented FP32 tolerances.
- Two lanes of three decisions with two unequal visual surfaces, different lane-major packet labels, all action families, nonzero recurrent carry, varying context IDs/timing, an active reset and a padded reset. Assertions ensure nonzero gradients through dense pointing, cursor/pooling vision, context embeddings and both recurrent layers.
- An actual forbidden-key packet producing `-inf` likelihood on an invalid padded row, under both behavioral and PPO objectives. The masked row does not contaminate the loss or any gradient.
- Cancellation at three computation boundaries, including after action or visual VJP work, preserves every original parameter object binding, the frozen/trainable mask and the incoming recurrent state.

No production-helper defect remained from this regression pass. Small-model equivalence is evidence for the chain rule and control boundaries; it does not prove production memory admission, optimizer peak allocation or learning quality.

## Memory-accounting corrections and remaining qualification

At 1280×720 with the production 128-wide stride-eight dense grid, one decision has 14,400 dense cells, or 7,372,800 bytes. One surface over 64 decisions therefore needs approximately 472 MB each for stored dense features and dense cotangents. The normalized image tensors add approximately 1.09 GB before loader copies. Default BC's two lanes double those amounts. More surfaces scale these caches and can exceed the budget even when each visual microbatch contains only one decision: that decision still contains all its surfaces. Admission must account for actual shapes, surfaces, lanes, raw rollout ownership, live features/cotangents, optimizer state, actor and rollback snapshots.

A separate installed-MLX probe verified that `copy.deepcopy(mx.array)` added **zero** active bytes. The current memory benchmark creates actor and rollback copies, initializes optimizer slots and performs only a backward pass. Those snapshots initially alias immutable buffers. Consequently its `includesActorOptimizerAndRollback` flag does not prove the peak after learner parameters and optimizer moments diverge from the retained snapshots. Qualification should include a real optimizer update while the actor and rollback references remain alive, then a subsequent gradient pass with an existing accumulation tree, and the supported two-lane BC shape. Report backward and complete-update peaks separately.

Production acceptance still requires guarded 64-step runs, exact full trainable-tree admission, real parameter/optimizer updates, finite gradients, cancellation/resume at coherent boundaries, and representative BC/PPO learning verification. Preserve full representation and recurrence; do not count a lower-resolution or shorter-horizon run as that evidence.
