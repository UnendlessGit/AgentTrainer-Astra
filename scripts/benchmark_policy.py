#!/usr/bin/env python3
"""Bounded production-shape policy benchmark on the actual local MLX device."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be between 1 and 100")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    import numpy as np
    from astra.model.actions import ActionVocabulary, PacketBatch, flatten_visual
    from astra.model.config import ModelConfig
    from astra.model.observation import ObservationBatch, SurfaceBatch
    from astra.model.policy import AgentPolicy

    config = ModelConfig().validate()
    vocabulary = ActionVocabulary((0, 1, 2, 4, 13, 49, 53, 123, 124, 125, 126), (0, 1, 2), True, True, True)
    mx.random.seed(491)
    model = AgentPolicy(config, vocabulary)
    model.vision.backbone.load_pretrained(ROOT / "vendor/weights/convnext_tiny.safetensors")
    # One synthetic surface at the full default visual sizes. This benchmark
    # measures computation, not learning quality, native capture or action I/O.
    surface = SurfaceBatch(mx.random.normal((1, 1, 432, 768, 3)), mx.random.normal((1, 1, 864, 1536, 3)),
                           mx.random.normal((1, 1, 384, 384, 3)), mx.array([[[0.0, 0.0, 1.0, 1.0]]]), mx.array([[[0.0, 0.0, 1.0, 1.0]]]),
                           mx.array([[[0.4, 0.35, 0.125, 0.222222]]]), mx.array([[[0.0, 0.0, 1728.0, 972.0]]]),
                           mx.ones((1, 1), dtype=mx.bool_))
    observation = ObservationBatch((surface,), mx.zeros((1, 1, config.control_width)), mx.full((1, 1), 0.1),
                                    mx.zeros((1, 1, 0), dtype=mx.int32), mx.zeros((1, 1), dtype=mx.bool_), mx.ones((1, 1), dtype=mx.bool_))
    mx.eval(model.parameters(), observation.as_tensors())
    mx.reset_peak_memory()
    timings = []
    for iteration in range(2 + args.repeats):
        start = time.perf_counter()
        encoding = model(observation)
        mx.eval(encoding.visual.cells, encoding.temporal.context, encoding.temporal.value, encoding.temporal.state)
        encoded = time.perf_counter()
        output = model.sample(encoding, key=mx.random.key(iteration))
        mx.eval(output.packets.operation, output.log_probability)
        sampled = time.perf_counter()
        output.packets.validate(config, vocabulary, flatten_visual(encoding.visual))
        if iteration >= 2:
            timings.append({"encodingMs": (encoded - start) * 1000, "packetMs": (sampled - encoded) * 1000,
                            "totalMs": (sampled - start) * 1000,
                            "commands": int(np.count_nonzero(np.asarray(output.packets.operation)))})
    report = {"device": mx.device_info(), "configSignature": config.signature,
              "parameterCount": sum(value.size for _, value in tree_flatten(model.parameters())),
              "precision": "FP32", "source": "Synthetic RGB at 768/1536/384; one surface, one observation",
              "warmupIterations": 2, "iterations": timings,
              "medianMs": {key: float(np.median([item[key] for item in timings])) for key in ("encodingMs", "packetMs", "totalMs")},
              "peakMemoryBytes": mx.get_peak_memory()}
    if args.backward:
        model.configure_execution(vision_microbatch=1, checkpoint_vision=True)
        # Meaningful multimodal packet with finite end-to-end NLL gradients.
        fields = {name: np.zeros((1, config.packet_capacity + 1), dtype=np.int32) for name in PacketBatch.__dataclass_fields__}
        fields["operation"][0, :3] = [1, 6, 5]
        fields["offset"][0, :3] = [0, 30, 80]
        fields["key"][0, 0] = 13
        fields["cell"][0, 1] = 8000
        fields["within_x"][0, 1] = 31
        fields["within_y"][0, 1] = 20
        packet = PacketBatch(**{name: mx.array(value) for name, value in fields.items()})
        packet.validate(config, vocabulary, flatten_visual(encoding.visual))
        def loss_fn(policy):
            encoded = policy(observation)
            return -mx.mean(policy.score(encoded, packet).log_probability) + 0.5 * mx.mean(encoded.temporal.value ** 2)
        mx.reset_peak_memory()
        start = time.perf_counter()
        loss, gradients = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, gradients)
        elapsed = time.perf_counter() - start
        finite = all(bool(mx.all(mx.isfinite(value)).item()) for _, value in tree_flatten(gradients))
        if not finite or not np.isfinite(float(loss)):
            raise RuntimeError("Production policy produced nonfinite loss/gradients")
        report["backward"] = {"milliseconds": elapsed * 1000, "loss": float(loss), "finite": finite,
                               "peakMemoryBytes": mx.get_peak_memory(), "checkpointVision": True,
                               "pretrainedStemGradientNorm": float(mx.linalg.norm(gradients["vision"]["backbone"]["downsamples"][0]["conv"]["weight"]))}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
