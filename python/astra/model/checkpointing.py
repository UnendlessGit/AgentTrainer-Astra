"""Activation rematerialization with explicit trainable-parameter inputs.

Passing only an image to mx.checkpoint would not differentiate captured module
parameters. This wrapper passes their full trainable pytree and restores the
outer module bindings after tracing, including when the forward raises.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def checkpoint_call(module: nn.Module, function, *args):
    def call(parameters, *values):
        previous = module.parameters()
        try:
            module.update(parameters)
            return function(module, *values)
        finally:
            module.update(previous)
    return mx.checkpoint(call)(module.trainable_parameters(), *args)
