# Permission-free practice environments

`python/astra/environments/practice.py` implements two isolated local visual worlds. They produce real BGRA pixel arrays and the same frame, surface, control and raw-event contracts used by native recording. They never capture a screen, request TCC access, post OS input, sleep, access the network or modify source recordings. Independent instances can run concurrently because their controls do not affect the physical desktop.

These are general environment adapters for learning-path verification. They do not establish that installed-bundle capture/control, TCC attribution or real application interaction works; those remain separate release gates.

## API and immutable configuration

```python
from astra.environments import PracticeConfig, PracticeEnvironment

config = PracticeConfig(task="delayed_memory", seed=42, delay_ms=8000)
environment = PracticeEnvironment(config)
observation = environment.reset(seed=1001)
while True:
    # Privileged fixture demonstration. A policy instead receives observation
    # pixels, geometry and causal control state and supplies its own commands.
    transition = environment.step(
        environment.oracle_commands(),
        episode_id=observation.episode_id,
        provenance="oracle",
    )
    observation = transition.observation
    if transition.outcome != "continuing":
        break
```

`PracticeObservation` owns an immutable `pixels` array, `metadata` (also exposed as `frame_metadata`), `control_state`, and `episode_id`. Metadata contains exactly native FrameMetadata fields. No task answer, target coordinates or seed is inserted into model observations. Geometry explicitly maps global logical bounds—including negative origins and unequal horizontal/vertical pixel scales—to native source pixels. Frames are opaque SDR sRGB BGRA.

`PracticeTransition` contains observation, scalar reward, actual `duration_ms`, outcome, `raw_events`, `command_results`, submitted packet provenance, separate `cleanup_events`, and an optional administrative reason. Receipts identify packet sequence, command index, episode and provenance because execution can occur several decisions after admission. Raw events use origin `agent`, with detail `practice:agent` or `practice:oracle`; generated demonstrations are never falsely labeled physical human input. Fixture dataset construction must explicitly retain oracle provenance. If a caller mixes provenance within an episode, each receipt/event still identifies its actual source; the transition-level field describes the newly submitted packet.

`PracticeConfig` is frozen, validated, versioned, JSON serializable through `to_dict`/`from_dict`, and has a SHA-256 `signature`. Defaults are 1280×720 native pixels, matching logical bounds, 100 ms decisions, 100 ms execution lead, a 500 ms cue, a 2-second delay, a 120-second collection limit, and zero shaping. Default reset seeds progress reproducibly from the configured seed. Explicit `reset(seed=...)` exactly reproduces the world while creating a new episode identity. `current_seed` exists for run/evaluation manifests and is not an actor observation.

The default action vocabulary is absolute pointer and button 0. Delayed memory additionally permits left/right arrow key codes 123/124. `relative_pointer=True` is a separate configuration option for count/interpolation tests; ordinary pointing learning does not explore redundant equivalent pointer modes. Scroll and other keys/buttons are rejected. An entire packet is validated before clock, state, sequence or queue mutation, including argument relevance, capabilities, monotonic timing, finite geometry, mode exclusivity and the 64-command adapter maximum. A model's smaller immutable packet budget remains its own admission constraint.

## Tasks, rewards and held-out trials

Pointing randomizes the cursor, target center and small target radius across the field. A fresh button-down inside the visible teal target terminates with +1. An off-target click costs 0.05 and continues. Idempotent repeated downs do not fabricate additional clicks. Empty packets preserve held state.

Delayed memory independently randomizes a binary teal/orange cue, two later choice locations and which color occupies each location. The cue is visible only in `[0,cue_ms)`. The entire cue image is erased during the delay; two choice targets appear at `cue_ms + delay_ms`. Before readiness, clicks/keys provide zero reward and cannot reveal the answer. A choice terminates with +1 for the remembered color or −1 for the other color. A click outside either choice costs 0.05 and continues. Left/right keys select the spatially left/right choice. Oracle access is explicitly privileged; production policies must never call it.

Use disjoint **episode seed sets**, not neighboring frames, for training/tuning/held-out comparisons. Evaluate the required 2-, 8- and 30-second delays separately. The adapter accepts shorter timing configurations for focused numerical tests, but these do not replace long-memory experiments.

Optional pointing shaping is `scale × (gamma(dt) × Phi(next) − Phi(current))`, where `Phi` is negative pointer-to-target distance divided by the logical field diagonal and `gamma(dt) = 2 ** (−dt / discount_half_life_ms)`. The default half-life is 30 seconds. The learner must use the identical configured discount. True terminal potential is zero; collection truncation retains the actual pre-cleanup potential for value bootstrap. Memory shaping is rejected because target-distance feedback would disclose the answer. No shaping is enabled by default.

## Timing, state and boundaries

At observation cutoff `t`, `step` admits one complete packet for `t + lead_ms` and advances the reward interval `[t,t+period_ms)`. It does not shift rewards to the future execution window. A command at exactly the next cutoff belongs to the next transition, after that cutoff's observation. Event/source and availability times share a virtual monotonic nanosecond clock, and every event in a transition precedes its final observation. The clock survives resets; stale explicit episode identities are rejected.

The first motion knot executes at its own offset. Later absolute knots interpolate on a 1 ms grid; relative knots apply incremental integer counts with nearest-integer cumulative remainder compensation. Intermediate interpolated positions precede same-time controls. Equal-time explicit command order and old-packet/new-packet order are stable. A final motion knot can lie at the decision-period endpoint. There is no implicit motion before the first knot. Relative raw counts map one count to one logical point in this simulator and the cursor is bounded by the surface; nonzero raw deltas are still emitted at a boundary even if the pointer cannot move.

Key/button holds are idempotent, repeats on unheld keys are no-ops, and every reset clears held state and queued old-episode work. Terminal/truncated observations are captured **before** boundary cleanup; `cleanup_events` are outside the policy interval. This preserves the valid final world/control observation for truncation bootstrap. Cancellation receipts cover unexecuted commands. No new policy packet is admitted until reset after a boundary.

`abort(reason)` returns an administrative aborted boundary with zero elapsed time. This is excluded from PPO; it is not an invented zero-duration training transition, task failure or bootstrap target.

## Evidence on the development Mac

`uv run pytest python/tests/test_practice.py -q`: **35 passed** on 2026-09-06. Checks include deterministic unseen layouts, native BGRA and nonuniform pixel/logical geometry, paired counterfactual cue-erasure leakage, exact lead/reward boundaries, final knots, interpolation before controls, signed relative remainder, atomic malformed-packet rejection with an existing queue, held/repeat semantics, stale episode rejection, reset/abort cancellation, pre-cleanup truncation bootstrap, and duration-discounted shaping telescoping.

Additional oracle exercise used the production 1280×720 native raster, 100 ms decisions and a nonintegral 150 ms lead. Each row used unseen seeds 1001, 2003 and 3007:

| Task | Cue-to-choice delay | Successes | Decisions per episode |
|---|---:|---:|---:|
| Pointing | — | 3/3 | 2 |
| Delayed memory | 2 s | 3/3 | 27 |
| Delayed memory | 8 s | 3/3 | 87 |
| Delayed memory | 30 s | 3/3 | 307 |

An answer-independent always-left policy succeeded in 126/256 trials (49.2%) on separate seeds 10000–10255 at the 2-second delay, consistent with the binary chance baseline. This baseline exercise used 160×90 raster images to reduce unused rendering work; logical geometry and temporal task behavior were unchanged.

Oracle correctness and chance behavior are task validity evidence, **not learned-policy performance**. Required BC overfit/held-out runs, recurrent 90% memory success, PPO improvement, native practice UI integration, browser/window exercises and installed-app workflows remain to be produced by their corresponding learning/platform work.
