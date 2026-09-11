# Native inference coordination

2026-09-11 — implemented and verified the native coordinator and Run view through injected capture/process boundaries. This evidence does not establish installed-app capture, macOS posting effects, trained-agent success, or final timing qualification.

`InferenceCoordinator` owns one run's capture, actor, control helper, two-slot mapped frame ring, heartbeat, packet identities and joined shutdown. Native capture keeps one latest frame; frame publication uses the shared ring, and only the exact actor acknowledgement releases each slot. Checkpoint admission verifies the selected agent's link and immutable checkpoint identity. It deliberately excludes display names and serialized date precision from identity comparison.

Preparation loads the policy once, runs three actual actor warmups without starting the control helper, then confirms a fresh episode and original seed before arming. `inference.warmup` uses the real mapped-image/preprocessing/policy/decoder path while restoring actor recurrence, counters, observation history and RNG. Its response still releases the newly consumed image lease. The native coordinator checks every warmup's policy, observation, state and packet identity and requires the explicit warmup marker. The first sample pays compilation cost; each of the final two samples must fit `min(period, lead) - max(5 ms, 10% of that minimum)`. These two samples are a startup admission check, not a latency percentile benchmark or a guarantee across future images/actions. Runtime deadline checks remain mandatory. Astra never retimes actions to conceal a missed checkpoint lead.

Once armed, a separate heartbeat runs while inference awaits the actor. Each observation combines an image available before the helper's atomic executed-input cutoff with complete causal executed input history. Geometry is checked again after prediction so a window change during GPU work prevents admission. Unexpected capture termination, missing input coverage, invalid actor identities, wrong leases, late actions and inconsistent receipts stop the run. Terminal receipts must contain exactly one valid result for every original command; interpolation does not create extra receipt indices. Physical takeover and emergency stop are recorded as interventions rather than actor failures.

Stop cancels future decisions, closes the frame inbox, urgently disarms control, stops capture and shuts down the actor to interrupt a blocked request. Repeated Stop calls join the same work. A late arm response is disarmed again. Final cleanup joins the control process, capture and actor before retiring the ring; an actor shutdown acknowledgement alone is insufficient. The local run folder records immutable configuration and a terminal summary with prediction latency, all warmup timings, decision/execution counts and any failure or intervention.

The Run view uses native controls and semantic colors for system appearance, with checkpoint/environment selection, explicit environment refresh, deterministic or sampled actions, seed/context settings, timing feedback, Stop and export of the run summary. Workspace/menu/quit wiring is maintained in `WorkspaceModel` and `AstraApp`. Semantic names for context vocabularies, multi-surface/application/desktop inference, visual/accessibility review and installed-bundle control qualification remain open product/release gates.

## Checks

Command: `swift test --filter InferenceCoordinatorTests`.

Eight test functions cover eighteen cases using the production coordinator, `LibraryStore` and actual memory-mapped frame ring, with fixture capture and runtime endpoints:

- Warmup occurs before arm, does not reload the model, uses unique leased frames, and permits a normal run, stop and restart.
- Incompatible runtime identity, checkpoint signature and inadequate warm timing never arm.
- Wrong packet run, recurrent state, frame acknowledgement, control-observation run, interrupted input coverage, changed geometry, stopped capture and late actions never reach execution.
- Incorrect or incomplete terminal receipts never count as successful execution.
- Stop while arm is pending disarms again after its late response.
- Concurrent Stop calls interrupt a pending prediction and preserve its still-published unacknowledged slot until the actor's exit barrier completes.
- A worker crash interrupts its pending prediction and joins capture/control/actor cleanup.
- Physical takeover ends normally with an intervention reason.

The harness does not import MLX, request privacy permissions or post any OS input. Actual actor math, real child transport, ring ingestion and native executor behavior have separate verification documents and tests. End-to-end installed application evidence remains required.
