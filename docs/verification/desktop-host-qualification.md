# Permission-free desktop host qualification

September 19, 2026, restored Xcode 27 / Swift 6.4 and locked Python 3.12.13.
Run `.venv/bin/python scripts/qualify_desktop_host.py`; optionally pass
`--bundle /absolute/AgentTrainer\ Astra.app` to use its frozen compute helper.
The test is opt-in and does not run during the ordinary Swift suite.

This runs the real `DesktopLearningHost`, configuration/source binding,
`NativeResetDriver`/`ResetRunner`, warmup, persistent actor, episode runner,
reward queue, evidence bridge, collector, learner boundary handshake and catalog
publication. Actor, collector and learner use actual separate Python processes.
`NativeControlSession` uses the real `InputExecutor` scheduler, deadlines,
admission, history, command receipts and cleanup with an in-memory input backend.
An owned capture timer generates all BGRA pixels and frame timestamps. The
fixture explicitly confirms Ready; the real reset path verifies fresh generated
source evidence and joined cleanup. Elapsed episode time produces automatic
reward and semantic termination. There is no screen capture, privacy preflight,
event tap, Core Graphics posting or personal pixel access.

The first two real-process runs exposed production integration defects hidden
by protocol fixtures: re-encoding native action capabilities discarded the
required scroll quantization, and the host failed to create its owned collection
parent directory. Both were fixed in product code; validation and package
preconditions remain strict.

The completed source-worker run
`.local/verification/desktop-host-fb2e42ab-2b84-4232-9e97-c0c0cc2f4340`
passed its native workflow in 7.08 seconds. Two complete episodes admitted eight
decisions into one PPO iteration with two real optimizer updates. Weights changed
and the new checkpoint was published into the actual library catalog. A second
host/actor run authenticated the saved stream, sampled the next packet and
stopped before its rollout minimum. Its separate boundary-only checkpoint kept
policy bytes and nonempty optimizer tensors exactly equal while advancing the
saved draw index from 9 to 10, including excluded terminal/suffix actions. Three
native virtual-control owners joined with no owned holds; two actor and two
collector processes exited with status zero. The generated source joined too.

Each run retains its report, Swift log, library, immutable collector packages,
native control journals and checkpoints in a new private output directory. The
script independently reloads checkpoints and checks weights, optimizer tensors,
RNG continuation and published session/package counts. It preserves failed-run
evidence instead of overwriting a prior result.

The model is explicitly `ModelConfig.test_small()` with the normal 100 ms cadence
and 100 ms lead. Sequence/batch lengths are small numeric fixtures; production
architecture/defaults are unchanged. Time-based reward verifies plumbing and
finite learning updates, not task-solving improvement. This evidence does not
qualify production-model timing, GPU contention, real capture/input, automatic
authored reset commands, privacy revocation or installed-DMG workflows. The first
passing result used source workers; frozen-bundle execution is a separate step.
