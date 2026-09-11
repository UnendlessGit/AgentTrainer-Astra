# Delayed-memory learning qualification

2026-09-11, Apple M3 Max, macOS 27.0. This bounded experiment did **not** meet the 90% delayed-memory success target. It establishes a real learning/checkpoint/evaluation result and motivates controlled temporal comparisons; it does not prove that further GRU training cannot solve the task.

## Training and held-out behavior

The run used the full default visual/recurrent/packet model, pinned pretrained ConvNeXt weights, native 1280×720 pixels, 100 ms cadence/lead, two lanes and 64-decision TBPTT. AdamW rates and one-epoch pretrained freezing remained at their defaults. Three epochs were requested with a 360-second training budget. The worker completed one epoch and part of the second: 27 updates and 1,458 training decisions, approximately 1.69 passes through the 864-decision source. It saved a resumable checkpoint at a completed update boundary. The full train/save/reload/evaluation section took 483.21 seconds.

`PracticeDemonstrations` generated 32 training episodes with seeds 0–31. Evaluation used independent seeds 1000–1031. The ordinary memory vocabulary was preserved: absolute pointer, button 0 and left/right keys. The oracle supplied three pointer/button commands at readiness and END elsewhere; it never supplied privileged cue features to the policy. The cue lasted 500 ms, followed by a 2,000 ms blank delay. Training and evaluation both had a 4,500 ms virtual episode limit, leaving two seconds to answer after readiness. This explicit limit bounded failed evaluation cost.

The first 54-decision update had mean packet NLL 3.26087. The complete first epoch's running mean was 1.07378; the last reported partial-second-epoch running mean was 0.61863 at update 26. These are training losses measured while weights were changing, not fixed-checkpoint held-out NLL. Only one of each 27 demonstration decisions contains non-END commands, so falling average loss alone is weak memory evidence.

A freshly loaded checkpoint, reset recurrent state per episode and the actual compiled policy/decoder drove the ordinary virtual environment. No oracle was called by this acting policy. Greedy action selection succeeded on **17/32 episodes (53.1%)**; all 32 terminated with a choice and none timed out. Mean episode length was 41 decisions. An answer-independent always-left baseline succeeded on 14/32 of the same held-out seeds. The trained success rate is consistent with chance on this small sample; its Wilson 95% interval is 36.4–69.1%. A stochastic-policy success comparison was not run in this initial budget.

Peak MLX active allocation was 6,073,859,866 bytes and process peak RSS was 2,149,171,200 bytes. MLX allocation/cache limits were 10 GiB/64 MiB. These are separate allocation measures, not a total application memory measurement.

## Counterfactual cue retention

The same trained checkpoint was tested on 32 held-out layouts, each paired with a counterfactual world whose only change was the visible cue color. Layout, pointer and control state were held equal. Assertions checked exact pixel equality throughout every cue-free waiting frame and at readiness, plus equality of the model's control features. Oracle calls supplied correct/wrong packet labels only to the evaluator.

The probe passively waited and evaluated the likelihood of the two possible choice packets. Identical waiting images reused their exact visual features; every recurrent decision still executed. This isolates internal cue retention and does not count as freely acting success. The checkpoint was trained only at 2 seconds; the 8/30-second rows measure delay transfer, not separate long-delay training runs.

| Blank delay | Balanced correct choices | Mean correct-minus-wrong joint log probability | Mean recurrent RMS difference between opposite cues |
|---|---:|---:|---:|
| 2 s | 32/64 (50%) | 8.106e-6 | 7.677e-5 |
| 8 s | 32/64 (50%) | 9.537e-7 | 1.302e-5 |
| 30 s | 32/64 (50%) | −1.043e-7 | 3.626e-7 |

Flipping the cue changed the preferred choice on **zero of 32 layouts at every delay**. Thus this checkpoint's choice preference carried no useful cue discrimination in the controlled probe. Its recurrent states remained slightly different, with a much smaller difference after long waits. The probe took 96.23 seconds; combined training/evaluation/probe compute stayed below ten minutes.

## Training-horizon consequence

At 100 ms cadence the first choice label occurs at steps 25, 85 and 305 for the 2/8/30-second tasks. With 64-step TBPTT, those labels belong to chunks beginning at 0, 64 and 256. Only the 2-second label shares a gradient-connected chunk with the cue at steps 0–4. The learner carries recurrent values across chunks but detaches their gradients. This is an explicit training-horizon limitation for direct long-delay credit, not proof that shared recurrent dynamics can never learn longer retention.

The next comparisons should hold visual inputs, action vocabulary, seeds, optimizer updates and evaluation fixed: ordinary versus measured multi-timescale GRU retention, and the independently proposed bounded causal-attention alternative. Longer training and an informative-action loss diagnostic should separate optimization/data imbalance from representational limits. No production architecture, model default, backward implementation or trainer was changed by this experiment.

## Artifacts and reproduction

Source: `scripts/qualify_memory.py`. Private artifacts:

- `.local/verification/memory-default-2s.json` and its progress log.
- `.local/verification/memory-default-paired-probe.json`.
- Initial checkpoint `.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171`.
- Resumable trained checkpoint `.local/verification/cf9f5009-529a-4dde-a926-54ee0cb15b28`.

```sh
.venv/bin/python scripts/qualify_memory.py --epochs 3 --training-budget-seconds 360 --budget-seconds 600 --skip-baseline-policy --output .local/verification/memory-default-2s.json
.venv/bin/python scripts/qualify_memory.py --probe-checkpoint .local/verification/cf9f5009-529a-4dde-a926-54ee0cb15b28 --budget-seconds 110 --output .local/verification/memory-default-paired-probe.json
```

No macOS capture, input posting, TCC access, original-project edits or production ML changes were used. The scripts' syntax and CLI help were also checked; no new implementation-mirroring unit tests were added for this experiment.
