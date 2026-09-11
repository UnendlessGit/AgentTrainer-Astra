"""Exact chain-rule backward with bounded visual and action working sets.

The temporal VJP spans the complete B,T sequence. Only independent visual
observations and per-packet decoder work are replayed in small slices; neither
temporal gradients nor either visual path are detached from the final gradient.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable
import mlx.core as mx
from mlx.utils import tree_map

from astra.model.actions import PacketBatch, flatten_visual
from astra.model.observation import ObservationBatch
from astra.model.vision import VisualFeatures


@dataclass(frozen=True)
class PolicyGradient:
    loss: mx.array
    auxiliary: Any
    next_state: tuple[mx.array, ...]
    gradients: dict


def _bound(module, parameters, function):
    previous = module.parameters()
    try:
        module.update(parameters)
        return function()
    finally:
        module.update(previous)


def _sum_gradients(left, right):
    result = tree_map(lambda value: value.astype(mx.float32), right) if left is None else tree_map(
        lambda before, after: before + after.astype(mx.float32), left, right)
    # Evaluate each accumulation before the next replay, releasing its tape.
    mx.eval(result)
    return result


def _cotangent_dot(value, cotangent):
    cotangent = mx.stop_gradient(cotangent)
    # The original objective may mask an invalid -inf log probability. Preserve
    # that zero derivative without creating 0 * -inf in the replay surrogate.
    return mx.sum(mx.where(cotangent != 0, value, 0) * cotangent)


def policy_gradients(policy, observation: ObservationBatch, packets: PacketBatch,
                     objective: Callable, state=None, *, action_microbatch: int = 4,
                     cancelled: Callable[[], bool] = lambda: False) -> PolicyGradient:
    """Differentiate objective(logp, value, conditional_entropy)->(loss, aux).

    The objective is evaluated over all flattened decisions simultaneously, so
    valid-count normalization, clipping, KL diagnostics and any future coupled
    scalar objective retain their exact meaning. No optimizer mutates weights
    until every stage completes and the caller admits the assembled gradient.
    """
    observation.validate_shapes(control_width=policy.config.control_width, maximum_surfaces=policy.config.maximum_surfaces,
                                contexts=len(policy.config.context_sizes))
    if type(action_microbatch) is not int or not 1 <= action_microbatch <= 64:
        raise ValueError("Invalid action backward microbatch")
    batch, time = observation.shape
    count = batch * time
    if packets.operation.shape != (count, policy.config.packet_capacity + 1):
        raise ValueError("Packet labels do not align with the B,T observation sequence")
    def check_cancel():
        if cancelled():
            raise InterruptedError("Learning cancelled during bounded gradient computation")
    flattened = tree_map(lambda value: value.reshape(count, 1, *value.shape[2:]), observation.as_tensors())
    def slice_observation(first, last):
        return ObservationBatch.from_tensors(tree_map(lambda value: value[first:last], flattened))
    visual_microbatch = policy._vision_microbatch
    chunks = []
    for first in range(0, count, visual_microbatch):
        check_cancel()
        output = policy.vision(slice_observation(first, first + visual_microbatch))
        values = tuple(mx.stop_gradient(getattr(output, name)) for name in VisualFeatures.__dataclass_fields__)
        mx.eval(values)
        chunks.append(values)
    combined = [mx.concatenate([chunk[index] for chunk in chunks]) for index in range(len(chunks[0]))]
    features = VisualFeatures(*(value.reshape(batch, time, *value.shape[2:]) for value in combined))
    mx.eval(tuple(vars(features).values()))
    del chunks, combined, output, values

    temporal = policy.temporal(features.summary, observation, state)
    mx.eval(temporal.context, temporal.value, temporal.state)
    contexts = mx.stop_gradient(temporal.context.reshape(count, -1))
    values = mx.stop_gradient(temporal.value.reshape(count))
    next_state = tuple(mx.stop_gradient(value) for value in temporal.state)
    flat_visual = flatten_visual(features)
    def action_slice(first, last):
        visual = VisualFeatures(**{name: getattr(flat_visual, name)[first:last] for name in VisualFeatures.__dataclass_fields__})
        labels = PacketBatch(**{name: getattr(packets, name)[first:last] for name in PacketBatch.__dataclass_fields__})
        return visual, labels
    log_probabilities, entropies = [], []
    for first in range(0, count, action_microbatch):
        check_cancel()
        last = min(first + action_microbatch, count)
        local_visual, labels = action_slice(first, last)
        score = policy.actions.log_prob(contexts[first:last], local_visual, labels)
        mx.eval(score.log_probability, score.conditional_entropy)
        log_probabilities.append(mx.stop_gradient(score.log_probability))
        entropies.append(mx.stop_gradient(score.conditional_entropy))
    log_probabilities = mx.concatenate(log_probabilities)
    entropies = mx.concatenate(entropies)
    (loss, auxiliary), (dlogp, dvalue, dentropy) = mx.value_and_grad(objective, argnums=(0, 1, 2))(log_probabilities, values, entropies)
    mx.eval(loss, auxiliary, dlogp, dvalue, dentropy, next_state)

    action_gradients = None
    context_cotangents, cell_cotangents = [], []
    for first in range(0, count, action_microbatch):
        check_cancel()
        last = min(first + action_microbatch, count)
        local_visual, labels = action_slice(first, last)
        def action_vjp(parameters, context, cells):
            def evaluate():
                score = policy.actions.log_prob(context, replace(local_visual, cells=cells), labels)
                return _cotangent_dot(score.log_probability, dlogp[first:last]) + _cotangent_dot(
                    score.conditional_entropy, dentropy[first:last])
            return _bound(policy.actions, parameters, evaluate)
        _, (gradient, dcontext, dcells) = mx.value_and_grad(action_vjp, argnums=(0, 1, 2))(
            policy.actions.trainable_parameters(), contexts[first:last], local_visual.cells)
        mx.eval(gradient, dcontext, dcells)
        action_gradients = _sum_gradients(action_gradients, gradient)
        context_cotangents.append(mx.stop_gradient(dcontext))
        cell_cotangents.append(mx.stop_gradient(dcells))
    dcontext = mx.concatenate(context_cotangents).reshape(temporal.context.shape)
    dcells = mx.concatenate(cell_cotangents).reshape(features.cells.shape)
    mx.eval(dcontext, dcells)
    del context_cotangents, cell_cotangents, score

    check_cancel()
    def temporal_vjp(parameters, summary):
        def evaluate():
            result = policy.temporal(summary, observation, state)
            return _cotangent_dot(result.context, dcontext) + _cotangent_dot(result.value.reshape(-1), dvalue)
        return _bound(policy.temporal, parameters, evaluate)
    _, (temporal_gradients, dsummary) = mx.value_and_grad(temporal_vjp, argnums=(0, 1))(
        policy.temporal.trainable_parameters(), features.summary)
    mx.eval(temporal_gradients, dsummary)
    dsummary = dsummary.reshape(count, 1, -1)
    dcells = dcells.reshape(count, 1, *dcells.shape[2:])
    vision_gradients = None
    for first in range(0, count, visual_microbatch):
        check_cancel()
        last = min(first + visual_microbatch, count)
        local_observation = slice_observation(first, last)
        def visual_vjp(parameters):
            def evaluate():
                result = policy.vision(local_observation)
                return _cotangent_dot(result.summary, dsummary[first:last]) + _cotangent_dot(result.cells, dcells[first:last])
            return _bound(policy.vision, parameters, evaluate)
        _, gradient = mx.value_and_grad(visual_vjp)(policy.vision.trainable_parameters())
        mx.eval(gradient)
        vision_gradients = _sum_gradients(vision_gradients, gradient)
    check_cancel()
    gradients = {"vision": vision_gradients, "temporal": temporal_gradients, "actions": action_gradients}
    return PolicyGradient(loss, auxiliary, next_state, gradients)
