# Initial MLX feasibility probe

This is an environment capability check, not validation of Astra's final model or workflows.

- Isolated Astra virtual environment: CPython 3.12.13, MLX/MLX Metal 0.32.2, NumPy 2.4.3.
- MLX reports Metal available on Apple M3 Max.
- Reported physical memory: 38,654,705,664 bytes; recommended working set: 30,150,672,384 bytes.
- A fresh Conv2d -> pooled sequence -> GRU -> categorical head completed four real AdamW updates with bias correction enabled.
- Losses: 1.1903753, 1.1149228, 1.0381660, 0.9559093. All finite.
- Peak MLX allocation for this small probe: 2,280,488 bytes.

The probe ran on the GPU with shape `(2, 8, 24, 24, 3)` and did not read the original model. Full policy conversion, gradient, numerical, performance, packaging, and learning gates remain pending.
