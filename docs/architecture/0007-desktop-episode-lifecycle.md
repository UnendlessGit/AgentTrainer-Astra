# Proposed continuous desktop episode lifecycle

2026-09-12. **Proposed implementation increment; not implemented or qualified by this note.** Extend the existing `InferenceCoordinator` and its shared prediction/receipt paths. Do not copy the coordinator into another RL loop. The GRU remains the temporal model. Actor, control, reset, collector and learner lifetimes must be distinct without changing an action's identity, commands or trained timing.

## Ownership and state boundaries

One desktop learning session owns a persistent actor process, actor run ID, actor frame ring, model/action configuration and RNG stream. Each physical episode owns a fresh episode ID, scoped actor-control child/guardian/recovery ledger, reward-analysis queue and control callback generation. ResetRunner uses a separate reset child and `resetID` as that child's control run ID; reset packets are never actor demonstrations or on-policy decisions.

| State | Across a confirmed physical reset |
|---|---|
| Actor `_sequence`, `_draw_index`, RNG key and `_rng_stream_id` | Preserve |
| Actor `_last_cutoff` and observation replay cache | Preserve; host-clock continuity and anti-replay still apply |
| Actor `_environment_resets` | Advance once for an accepted reset; it is a reset-generation counter, not fabricated physical evidence |
| GRU state, state ID and `_episode_step` | Clear state, allocate a new state ID, start episode step at zero |
| `_last_input_nanos`, `_last_event_sequence` | Clear; new control child's event history belongs to this episode |
| `_geometry_revision`, `_surfaces` | Clear and bind the first actual post-reset observation to the newly verified scope |
| Native global packet/draw progress | Preserve |
| Native episode step, input-history acknowledgement cursor, pending receipts and control generation | New episode-local state |

The source `inference.reset` now preserves packet/RNG counters and clears recurrence, last-input time, episode step, raw event cursor and surface/geometry binding. New control children can continue emitting their original event sequence starting at zero. Do not renumber raw input events. Require the first actor observation after reset to contain no pre-control executed events, consistent with the assembler's existing first-row check. Native orchestration remains pending.

The first elapsed-time feature remains the policy period; subsequent elapsed time is the actual within-episode cutoff difference. Preserve the actor's global monotonic cutoff guard. A new episode must follow verified readiness and the previous global cutoff, but does not inherit the prior episode's cadence interval after all its controls have joined. The continuing-episode cadence check now reflects that distinction. Do not reset or manufacture source clocks.

Capture may remain alive across episodes when its source binding stays valid. Rebind/restart it when necessary through the existing target-identity checks, with a fresh inbox/generation. A changed window/process identity is not permission to guess another target. Adopt the newly verified actual geometry at reset, without offsetting or rewriting surface revisions.

## Packet and RNG invariant

For collection schema 1, enforce:

`ActionPacket.sequence == collectionRecord.sampler.drawIndex`

Both counters count **real produced policy results**, including a result that cannot ultimately be admitted by control. They start together, advance once together, are both restored by warmup and both survive reset. Therefore `actorProgress.drawIndex` is the unambiguous last produced packet watermark; its successor is the next packet/draw index. Check the equality in the actor's collection path, native result validation, asynchronous assembler and artifact validation. No separate progress-schema version is needed for this agreed invariant.

The official `InferenceSession` producer and multi-episode assembler fixture satisfy this invariant. The resumed-package generator in `python/tests/test_collector.py` has been corrected to generate packet/receipt/end sequences as `first_draw + step`. Old inconsistent synthetic artifacts were not rewritten and cannot satisfy the strengthened contract.

Keep separate native counters for produced, admitted and executed packets, plus episode step. The present `decisions` counter advances only after control admission and cannot be the persistent actor's expected sequence. Advance produced/draw expectations when a fully validated actor result arrives; keep the original result even if Stop wins admission. Administrative reset/warmup acknowledgements do not create new `actorProgress`: progress is derived from validated real results and joined source publication.

## Control admission origin

Add `ArmRequest.initialPacketSequence`, defaulting to zero for existing single-run/reset clients, and initialize `ControlLease.nextSequence` from it. Reject an exhausted origin. The control helper must advertise support and acknowledge the exact origin; a nonzero origin must not silently fall back to an older helper's zero. Use backward-aware decoding rather than assuming a Swift initializer default changes synthesized `Decodable` behavior.

For an actor episode, use the actor's exact next global packet sequence. Keep `packet.runID` equal to the persistent actor run ID and pass every original packet unmodified. Reset helpers use `context.resetID` and origin zero. Wire-message sequence numbers remain local to each process connection; they are distinct from policy packet sequences.

`InputExecutor` does not need a global raw-event counter. Its event sequence, delivered/acknowledged cursor and coverage can reset normally for each newly armed episode. Its existing pending-map, scheduling order and absolute packet-time checks already work with a nonzero packet origin. Use a distinct recovery ledger ID/path per child even when successive actor-control children share the actor run ID.

## Startup, reset and checkpoint activation

The coordinator should reuse its current private prediction, scope, heartbeat, receipt and cleanup methods behind a session loop and an episode loop:

1. Prepare the persistent actor and frame ring once. Validate fixed model/action/timing identity.
2. Before reset actions, stop the prior episode and join every possible control post, control I/O offer and cleanup owner. A collector ack is not cleanup proof.
3. Run ResetRunner with a separate reset child. Accept readiness only after that child is released/joined and the actual ready observation/signals are verified.
4. Select the immutable current or pending checkpoint through `PolicyActivationGate`. Load/activate it only on `inference.reset`; keep the existing equal-policy-signature requirement. A failed load leaves the actor reset-required; do not silently act with old recurrent state or old weights.
5. Warm the actual policy path with no armed controls. Warmup restores recurrence, counters, RNG and replay history. Validate warmup packets against the global next sequence, not the current hard-coded zero. One real actor reset can precede warmup; another fabricated environment reset is unnecessary afterward.
6. Create and arm the episode's control child with the global packet origin, validate guardian/ledger and current scope, then create the first actual actor cutoff. Begin reward intervals from that cutoff, not warmup or pre-actor time.

Reset acknowledgement should expose the next packet/draw index and reset generation so native admission checks an authoritative actor value. Those administrative fields are distinct from the last-real-result `actorProgress`.

For process/checkpoint resume, extend actor preparation with an explicit resume mode that loads embedded progress from an integrity-validated `reinforcement_external` checkpoint. Do not accept an arbitrary caller-provided RNG dictionary as authenticated progress. Restore the checkpoint's run/stream binding, saved RNG state, reset generation and both next counters to `drawIndex + 1`; reject exhausted counters, incompatible identity, and a simultaneous fresh seed. Start with no recurrence and `needsReset = true` until a newly confirmed physical reset. A fresh run from the same model remains a separate explicit choice with a new stream and counters zero.

Preserve the checkpoint/publication evidence that authenticates the saved progress. Existing external learner publication requires a verified joined sealed/audited boundary with matching run, clock, policy and source bindings. Stream resume must use that evidence, not raw aborted journal fields. A new native execution directory/control generation is needed even when the authenticated actor run ID persists; do not overwrite an earlier run directory. Clock-domain/reboot handling remains an explicit resume gate because progress schema 1 itself does not carry a last cutoff or clock identity.

## Episode boundary and in-flight results

Add a graceful episode-boundary path distinct from the existing whole-session `requestStop`. The latter cancels `work`, closes capture and shuts down the actor; keep it for user stop and unrecoverable faults. A normal episode boundary stops new predictions and promptly disarms control, while leaving the actor and old control child alive long enough to resolve the outstanding actor request.

If that result arrives after acknowledged disarm, retain its original packet/observation/collection record and send the unchanged packet to the still-disarmed old helper. `InputExecutor.execute` already produces a genuine `.rejected` receipt when its lease is inactive. Do this only after positive disarm acknowledgement, never rearm that helper, and route the real receipt before joining it. Do not fabricate an executed/cancelled receipt, subtract an episode offset, change the run/packet ID, or retime the packet. If the actor cannot deliver a trustworthy result/progress boundary, abort continuity rather than guessing how many RNG draws occurred.

The Python receipt validator now retains truthful inactive final states for audited/boundary outcomes, with separate joined cleanup proof. It never relabels them valid. Admission and continuing executed-prefix states remain active; boundary-cancelled prefix commands must satisfy the existing exact timing rule. Late/rejected packets remain excluded from the learning prefix; invalid identity or impossible timestamps remain faults. Native receipt handling must likewise distinguish expected closing-phase cancellation/rejection from whole-session failure, preserving each raw receipt and cancellation intent.

Attach an episode/control-generation token to native callbacks in addition to the persistent run ID. Joined process I/O may already have queued MainActor work; a stale `control.stopped` or receipt must not stop or mutate the next episode. Route the old raw evidence to its old collector before generation-filtered UI delivery. Per-episode pending dictionaries and cleanup fields prevent accidental cross-episode reuse.

Only after actor-result drain, receipt reconciliation, reward-queue drain and confirmed helper/guardian cleanup may the coordinator emit the exact `collector.end` watermark and start reset. Keep produced/admitted/executed distinctions in that audit. An empty or invalid episode must not manufacture a decision, reward or progress advancement merely to satisfy sealing.

## Collectors, learning and continuity

A collector package binds one immutable policy. It may contain multiple complete physical episodes until the rollout minimum is reached. At a joined boundary, finish its publication and give the sealed learning package to the learner. Episodes run while learning or awaiting activation are continuity/audit experience under the still-active actor policy. They continue the same packet/RNG stream and are never admitted to a newer policy's PPO batch.

Use verified `actorProgress` from the latest sealed/audited boundary for collector rotation and external learner publication. Enforcing sequence/draw equality makes the existing progress watermark sufficient for both counters. Never activate a pending checkpoint mid-episode. If no continuity episode runs, the already joined sealed learning package can provide the unchanged actor boundary; do not generate an empty fake audit episode.

Truncation requires an exact pre-reset bootstrap observation/value. Prefer the already recorded next actor row at the exact boundary. If absent, a non-mutating value-only actor operation is a separate necessary addition: consume/acknowledge fresh real frame leases, use the correct preceding recurrent state and actual elapsed time, and preserve packet/RNG counters and actor state. Do not use a normal sampled step and discard its draw, reuse initial-only warmup, or evaluate a reset screen as the old episode's bootstrap.

## Integration gates

- Two and more episodes with one actor/ring, local control-event sequence restarting at zero, and global packet/RNG counters continuing exactly.
- Reset-time geometry changes accepted only at the verified new boundary; first-row events empty and elapsed feature correct.
- Actor result racing terminal/Stop: preserved original packet, real disarmed rejection receipt, truthful inactive state, complete lease release and no next-episode contamination.
- Delayed old helper callbacks ignored by the new episode while retained in the old audit.
- Cancel during arm/reset/load/warmup; guardian recovery and failed cleanup never become readiness.
- Learning package followed by continuity package, joined external learner boundary, checkpoint publication and activation at the next reset.
- Authenticated prepare resume restores both counters/RNG; fresh mode and corrupted/stale progress are distinguished; synthetic fixture sequence generation is corrected.
- Terminal versus truncated value/bootstrap semantics and zero-decision cancellation remain truthful.
- Actor/learner GPU contention is independently measured before enabling simultaneous live execution and updates. Serial learning at an unarmed joined boundary is the safe initial scheduling mode.

No source edits or GPU execution were performed for this review.

## Proposed bounded GPU contention benchmark

No benchmark has run for this increment. The available measurements are a roughly 58.5 ms compiled stochastic actor median without native ring/pipe/SCK timing, and an 8.10 GB peak for a full B2×T64 backward/update workload. Neither qualifies simultaneous acting and learning. The latter's recorded first backward was 32.41 seconds; do not infer spare GPU capacity from actor-only throughput or from the frozen-visual diagnostic timings.

Three scheduling options are worth distinguishing:

1. **Serialize learning at a joined, unarmed boundary.** Keep actor weights/process resident, but perform no policy decisions or physical control during GPU updates. This is the initial safe mode and requires no quality change. The already joined learning rollout can supply the unchanged external learner boundary when no continuity episode runs.
2. **Independent concurrent actor and learner processes.** This fits the existing isolation/ownership model, but offers no demonstrated deadline priority. Installed MLX 0.32.2 exposes device/stream creation and synchronization without a stream-priority parameter. Separate streams or CPU QoS/nice settings must not be described as guaranteed GPU preemption. Measure this as an experimental case before allowing live controls.
3. **Cooperative learner admission between bounded, fully evaluated stages.** Existing staged backward already evaluates independent visual/action slices and one full temporal VJP. A future gate could pause between those safe boundaries, with fixed weights until the complete optimizer boundary. It cannot preempt a submitted GPU kernel. Measure each stage's longest occupancy first; retain the full recurrent gradient horizon. Smaller runtime microbatches are a later exact-execution comparison, not permission to reduce resolution, precision, packet capacity or learning quality.

Reward analysis is a further resource consumer. `RewardAnalysisQueue` moves work off the UI thread, but `VisualRewardDetector` currently configures no CPU-only/compute-device restriction on `VNRecognizeTextRequest`. Thus accelerated Vision work must be included rather than assuming it cannot contend with MLX. Verify supported routing separately if choosing CPU or another accelerator; do not lower OCR recognition quality to improve a benchmark number.

The first proposed campaign has four separately bounded cases, each at most three minutes and at most six minutes including cold setup/teardown: actor alone; actor plus actual configured OCR; actor plus a full production learner update; actor plus both. Start with 300 warm measured decisions at the actual 100 ms cadence, plus marked cold/warm preparation samples. Use full 1280×720 owned synthetic practice pixels, exact native ring publication and a frozen actor helper process. Post no OS input. Retain ordinary immutable action packets and acknowledge every frame lease. The existing direct actor benchmark is useful as a microbenchmark but omits the IPC/ring boundaries needed here.

The learner case must exercise actual current-weight recurrent replay, full-quality staged backward, accumulation/clipping, an optimizer boundary and its applicable PPO validation; record each phase. Current PPO uses B1×T64 chunks and a 256-decision effective batch, so four full chunks precede a default optimizer attempt; the historical B2×T64 memory probe is a separate stress shape, not the PPO batch layout. Use the production model and full visual path, not a small model or frozen visual cache. Do not imply that a bounded case completed the default 512-decision/four-epoch iteration. Record exact chunks, decisions, optimizer attempts/admissions and KL backtracks, and truthfully report a bound-limited partial case. Begin with a bounded optimizer-boundary workload and settled buffers; do not run an unbounded learning campaign. Separate cold learner setup from steady overlap. Follow a failed high-contention case with actor-only recovery measurement; avoid launching all three if a resource guard or gross stall already stops the simpler pair.

Record actor end-to-end latency p50/p95/p99/max, strict lead misses, missed cadence slots, actor queue depth, frame/receipt lease status, and learner-stage/update times. Actual cutoffs and elapsed features remain actual; a slow call must not be relabeled as an on-time nominal tick. An offline benchmark may count would-stop events without physical control, but production keeps its existing immediate deadline failure behavior. Report process RSS and MLX allocations separately with timestamps; do not double-count unified memory or call MLX's allocation guideline a hard machine-wide cap. Include memory pressure, collector/detector backlog and cold geometry/shape changes.

A promising first run requires zero strict lead misses and the existing p95-under-80-ms default target, not just a favorable median. Three hundred decisions are a screening sample, not a tail-latency guarantee. Repeat across warm starts and representative geometry/input variants before enabling concurrent live execution; installed capture/control contention remains a separate final qualification. If concurrency fails, keep serial boundary learning while investigating measured stage occupancy. No packet retiming, altered categorical sampling or model-quality reduction is an admissible scheduling workaround.
