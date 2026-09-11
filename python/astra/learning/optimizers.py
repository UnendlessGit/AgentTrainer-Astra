"""Explicit bias-corrected AdamW groups without advancing frozen optimizers."""
from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


class GroupedAdamW:
    def __init__(self, *, learning_rate: float, pretrained_learning_rate: float, weight_decay: float = 0.01):
        self.optimizers = {
            "new_decay": optim.AdamW(learning_rate, weight_decay=weight_decay, bias_correction=True),
            "new_no_decay": optim.AdamW(learning_rate, weight_decay=0, bias_correction=True),
            "pretrained_decay": optim.AdamW(pretrained_learning_rate, weight_decay=weight_decay, bias_correction=True),
            "pretrained_no_decay": optim.AdamW(pretrained_learning_rate, weight_decay=0, bias_correction=True),
        }

    @staticmethod
    def group(name, value):
        prefix = "pretrained" if name.startswith("vision.backbone.") else "new"
        decay = value.ndim >= 2 and not name.endswith(".bias") and "embedding" not in name
        return prefix + ("_decay" if decay else "_no_decay")

    def update(self, model, gradients) -> None:
        flat = dict(tree_flatten(model.trainable_parameters()))
        grad = dict(tree_flatten(gradients))
        if flat.keys() != grad.keys():
            raise ValueError("Optimizer gradient tree does not match trainable model parameters")
        updated = []
        for group, optimizer in self.optimizers.items():
            selected = {name: value for name, value in flat.items() if self.group(name, value) == group}
            if not selected:
                continue
            # init is idempotent for existing slots and initializes newly
            # unfrozen tensors without discarding their peers' moments.
            optimizer.init(selected)
            values = optimizer.apply_gradients({name: grad[name] for name in selected}, selected)
            updated.extend(values.items())
        model.update(tree_unflatten(updated))

    @property
    def state(self):
        return {name: optimizer.state for name, optimizer in self.optimizers.items()}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict) or value.keys() != self.optimizers.keys():
            raise ValueError("Optimizer checkpoint has incompatible parameter groups")
        for name, optimizer in self.optimizers.items():
            optimizer.state = value[name]


def finite_gradients(gradients) -> bool:
    leaves = tree_flatten(gradients)
    return bool(mx.all(mx.stack([mx.all(mx.isfinite(value)) for _, value in leaves])).item()) if leaves else False
