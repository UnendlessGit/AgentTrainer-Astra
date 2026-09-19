# Persistent policy actor — September 19, 2026

`PolicyActorSession` is a reusable native compute/transport owner. It is not yet adopted by `InferenceCoordinator` or a desktop RL loop. It posts no controls and owns no capture stream, reset helper, reward detector or learner. The existing production model, full visual paths, FP32 computation, categorical distribution, packet capacity and checkpoint timing are unchanged.

## Ownership and API

The session is a Swift actor with one runtime, one mapped frame ring and at most one operation in flight. `prepare` accepts an immutable checkpoint directory/document and either a fresh seed or explicit resume. Resume sends only `resumeActor: true`, never arbitrary caller RNG. The Python worker authenticates the embedded external-training cursor through checkpoint artifact integrity. An optional expected cursor is compared against the preparation counters/stream and the first real sampler's preceding RNG words.

`reset(confirmedEpisodeID:contextIDs:activating:)` requires a new episode identity. The caller is responsible for proving physical readiness and joined prior control cleanup before calling it. Replacement checkpoints must have the same policy/action signature. The ACK now reports `nextPacketSequence`, `nextDrawIndex` and `actorResetGeneration`; these are administrative counters, not `actorProgress`.

Global packet/draw counters and the last actual cutoff survive reset. Episode step, recurrent state, control-event cursor, surface binding and the cached drain ticket reset. The last real progress watermark remains unchanged until another real result. Schema 1 requires the original packet sequence to equal its sampler draw index. Each validated real result advances both exactly once. Reset increments only the administrative reset generation. Warmup uses fresh frame leases and the actual compute path but restores recurrent state, RNG, counters, cutoff and replay history. It is allowed only before an episode's first real decision, with no controls armed by the caller. Warmup produces no collection row or progress.

`beginPrediction` returns an owned `PolicyActorPrediction` ticket. Cancelling its waiter does not cancel the worker request. The caller must retain the ticket and reconcile its original packet; `drainPrediction` also retains access to the latest ticket. `requestStop` closes new work without dropping a sampled result. Normal `shutdown` drains before joining. `shutdown(interruptPending: true)` can escalate an ongoing graceful stop, joins a single shared process-exit task, and leaves unresolved sampled progress explicitly unknown. It never retires mapped leases on a timeout or shutdown acknowledgement alone. A forced stop winning during preparation or pixel publication cannot subsequently restart/send work to the joined worker.

Snapshots have an array-based owned-frame representation and a single-source native constructor. Current admission deliberately requires one surface, matching the existing native capture/collector API. Source time/availability/explicit unchanged coverage, settled input state, causal input-history cursor, immutable cadence and fixed per-episode geometry are validated before publication. Collection additionally requires an empty first-row event cursor and rejects physical/boundary input and gaps. Published raw metadata and pixel bytes are returned together; collection cannot accidentally describe raw pixels as LZFSE. Success requires all exact frame acknowledgements. Errors may acknowledge only the exact consumed prefix. Missing/foreign/repeated acknowledgements prevent verified progress and leave ownership for joined retirement.

Each result returns the original response and packet unchanged, plus its owned observation. It can be handed to the collector using `result.observation.singleSource()`. The session does not retime missed deadlines or decide whether control should admit the packet. The caller must enforce deadline/scope/current-geometry checks immediately before control submission, retain genuine rejection receipts for late drained packets, and distinguish produced/admitted/executed counters. Raw actor progress is evidence of a sampled result, not sealed collector/resume eligibility or proof of physical cleanup.

## Verification

The injected native suite exercises two episodes through one actor/ring, isolated warmup, reset-time same-signature checkpoint activation, exact packet/pixel preservation, global counter continuity, resumed first-RNG validation, stale source and changed geometry rejection, bad acknowledgements, counter disagreement, worker failure, cancelled waiters, draining after Stop, graceful-to-forced Stop escalation and ring retirement only after actual child join. A blocked-start fixture covers Stop winning during actor startup. Further regression cases cover a reset/warmup followed by Stop with no new packet, ineligible input history before sampling, and the last legal UInt64 packet index without overflow. No native fixture uses MLX, TCC, screen capture or OS input.

The Python actor suite additionally runs the actual small policy and native ring producer. New assertions cover administrative counter ACKs and a real result followed by a second confirmed episode's isolated warmup in both categorical-collection and ordinary inference modes. Existing checkpoint activation, authenticated resume, compiled/uncompiled, preprocessing, recurrence and frame-release tests remain active.

Commands:

```sh
swift test --filter PolicyActorSessionTests
.venv/bin/python -m pytest python/tests/test_inference.py -q -W error
```

Results: the Python suite passed 31 tests in 7.95 seconds. The final native focus passed 11 tests / 17 cases; the log is `.local/policy-actor-session-native.log`. These checks establish the reusable boundary, not live desktop continuity, production learning quality, installed permission behavior or a release gate. Coordinator adoption, complete episode/collector orchestration and installed physical qualification remain pending.

Independent read-only review identified and verified fixes for the previous-episode drain ticket, pre-control collection history, and last-legal-counter boundary. No unresolved actor ownership blocker was reported. Collector integration must still reconcile its freshness rule for control timestamps with the control helper’s intentional preservation of observed timestamps during silence; this increment never fabricates a new observation time.
