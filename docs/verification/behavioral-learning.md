# Behavioral learning integration

The full default model, including the version 2 padding fixes, completed an actual local demonstration-training/checkpoint/closed-loop qualification on the M3 Max on 2026-09-06. Command:

```sh
.venv/bin/python scripts/qualify_behavioral.py --epochs 8 --output .local/verification/behavioral-default-v2.json
```

The run used the production ConvNeXt/detail/GRU/packet model with pinned pretrained visual weights, native 1280×720 synthetic pointing observations, causal control state, 100 ms decision period and 100 ms execution lead. Data provenance was explicitly `practice_oracle`: three training seeds 7/8/9, validation 70, independent test 700. The environment adapter scheduled real validated virtual commands; it did not post OS input or require privacy permissions.

Eight epochs produced 24 AdamW updates. Training packet NLL fell from 18.93291 to 1.63207. Validation packet NLL was 6.73902 on two decisions. After immutable checkpoint publication and reload, greedy closed-loop control succeeded on seeds 7/8/9 and 700/701/702, each in two decisions. These six trials demonstrate a working path and transfer on a simple rendered task. They are far too few, and too narrow, to establish general computer-use quality or robust held-out success rates.

The measured train/evaluate/save/reload/trial section took 17.33 seconds; MLX reported 4,952,534,926 peak allocated bytes. This is an integration run, not a training-throughput or aggregate application-memory benchmark. The default sensory/temporal architecture was retained; the short test used one lane and two-step chunks because these pointing demonstrations contain two decisions.

The engine supports contiguous episode lanes, explicit reset and padding masks, detached state across TBPTT chunks, exact joint packet NLL, decision-weighted gradient accumulation, finite-gradient admission, global norm clipping, bias-corrected AdamW, distinct pretrained/new parameter rates, and staged visual unfreezing. Frozen optimizer groups do not advance their bias-correction step. Cancellation retains sampler positions, recurrent state, accumulated gradients and optimizer/RNG state so no completed chunk is repeated or skipped on resume. Tests compare resumed and uninterrupted parameters, not merely checkpoint file existence.

Remaining evidence: larger session-held-out recordings, delayed-memory learning at 2/8/30 seconds, robust closed-loop evaluation and failure metrics, long-sequence training memory, corrections and context conditioning, repeated experiments, native UI/jobs integration, and actual macOS interaction. The oracle generator regenerates requested chunks instead of holding a long trial's normalized images in GPU memory; deterministic fixture identities allow checkpoint resume across process restarts.
