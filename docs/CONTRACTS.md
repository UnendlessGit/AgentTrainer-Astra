# Astra contracts, version 1

This document fixes subsystem integration. Swift and Python use the same JSON fixtures; native and learning code must not invent different timing or action meanings. ADR 0001/0002 govern the reasoning and release gates.

## Process protocol

Transport is UTF-8 newline-delimited JSON on inherited stdin/stdout. One complete message is at most 1 MiB including newline. Diagnostic text goes to stderr. No pixels, tensor data, passwords, or arbitrary recorded key content appear in diagnostics. Both sides reject malformed, oversized, unsupported-version and unknown-kind requests without executing an action.

Envelope fields (camelCase): `version` integer1, `kind` string, `sequence` nonnegative integer, `requestID` optional UUID string, `runID` optional UUID string, `payload` object. JSON integers remain integers; do not convert nanosecond clocks to Double. Requests and replies preserve requestID; asynchronous events preserve runID and monotonically increasing sender sequence. A new process begins with `hello` containing `role`, `protocolVersion`, `capabilities`, and runtime information. Replies are `ack` or `error` (code, message, recoverable). Every role accepts `ping` and `shutdown`.

Native control role supports `permissions`, `listen`, `stopListening`, `arm`, `heartbeat`, `execute`, `disarm`, `state`. Only an armed matching run with a live lease can execute. `input` events carry batches of RawInputEvent; `receipt` carries ExecutionReceipt; `fault` means action admission stopped and cleanup ran. Responses never imply the target consumed an OS-posted event.

Compute role supports `capabilities`, then production `infer`, `train`, `evaluate`, `cancel`, and checkpoint operations as implemented. Advertise supported operations truthfully. A compute process owns its MLX arrays/optimizer/RNG. Separate roles use one Python policy package. The coordinator owns job state and protocol validation.

## Geometry and clocks

`Point2D`: x,y finite Doubles. `Rect2D`: x,y,width,height finite Doubles, positive dimensions. Global coordinates are macOS logical display points in the Quartz top-left desktop space, including negative origins. Pixels have explicit dimensions/stride and an explicit source transform. Never assume a factor of2.

`SurfaceDescriptor`: id string (run-local stable role), globalBounds Rect2D, pixelWidth, pixelHeight, contentBounds Rect2D in source pixels, geometryRevision nonnegative integer. Source selection is separate from surface role: display/window IDs and process launch identity are runtime bindings, not learned semantic categories.

Clocks are UInt64 native monotonic nanoseconds, derived from mach_absolute_time using mach_timebase_info. Preserve source time and observation/receipt time separately. Sleep, permissions and target discontinuities create explicit boundaries. Python must use the host's supplied clock/timestamps for native deadlines, not a different epoch.

## Input and actions

`RawInputEvent`: sequence, eventNanos, observedNanos, origin (`physical`, `agent`, `reconciliation`, `boundary`), kind (`keyDown`, `keyUp`, `keyRepeat`, `buttonDown`, `buttonUp`, `pointer`, `scroll`, `flags`, `gap`); optional keyCode, button, x,y,dx,dy,scrollX,scrollY,modifiers,isDown,detail,rawPlatformData. Pointer x/y are global logical points; relative dx/dy are raw event counts; scroll records its declared unit in detail. The optional base64 rawPlatformData retains the native event serialization (including phase/precision fields); it is never directly executed instead of validated semantic commands. Keep raw repeat/flags evidence and provenance. Never rewrite source time to fix delivery ordering. Reconciliation snapshots use their actual availability time and never become expert press labels.

`ControlState`: keys integer array, buttons integer array, modifiers UInt64, pointer Point2D, observedNanos, revision, valid. Canonical arrays are sorted/unique. Separate physical state from synthetic owned state.

`ActionCapabilities`: keyCodes (0...127), mouseButtons (0...31), absolutePointer, relativePointer, scroll. Default key/button sets are empty. A selected capability can always be disabled at execution, never silently enabled beyond a checkpoint's immutable vocabulary.

`TimedCommand`: offsetMs nonnegative integer; operation (`keyDown`, `keyUp`, `keyRepeat`, `buttonDown`, `buttonUp`, `pointerAbsolute`, `pointerRelative`, `scroll`); relevant arguments keyCode, button, surfaceID, x,y,dx,dy. Absolute x/y are normalized within surface globalBounds; relative dx/dy are incremental knot counts; scroll dx/dy use configured point units. Irrelevant fields are omitted. Key/button commands are idempotent; repeats on unheld keys are no-ops. Explicit keyCode flags preserve modifier sides. Absolute trajectories and relative trajectories cannot be mixed inside one packet.

`ActionPacket`: id UUID, runID UUID, sequence, observationID UUID, geometryRevision, executeAtNanos, durationMs, commands. Controls use offsets in `[0,durationMs)`; a final motion knot may use durationMs as an interpolation endpoint. Equal-time order is array order. An empty command list means no new command, not release-all. New packets do not repeat old transient commands. Capacity is an immutable configured 16/32/64 maximum; END is a neural encoding detail, not an extra wire command.

The scheduler validates the entire packet before admission, checks run/sequence/geometry/deadline and capabilities, and acknowledges admission separately from execution. Trajectory interpolation follows the versioned canonicalizer, with cumulative relative remainder. Cancellation invalidates every queued command. Disarm/revocation releases all possibly posted owned inputs, including in-flight ones.

`ExecutionReceipt`: packetID, runID, sequence, status (`admitted`, `executed`, `cancelled`, `rejected`, `late`), observedNanos, commandResults array and resultingState. A command result contains commandIndex, scheduledNanos, postedNanos (optional), status (`posted`,`noOp`,`cancelled`,`failed`), message (optional). Failed/late receipts make rollout validity explicit.

## Model interface (Python)

`ModelConfig` is immutable/versioned and serialized by the learning package. Default global/detail/cursor sizes768/1536/384, detail channels32/64/128, query count32, GRU layers2/width512, ConvNeXt-Tiny pretrained global branch. Context IDs/vocabulary, capabilities, packet budget, decision period, execution lead and canonicalization are part of checkpoint identity.

`ObservationBatch` supplies normalized global/detail/cursor images, valid spatial masks/transforms, causal executed-control/context features, elapsed time and optional recurrent state. It must support `(batch,time,...)` and equivalent single-step inference. The policy produces an explicit next recurrent state plus action-distribution context and scalar value. The packet decoder's working state is separate and reset each packet.

The learning package owns neural tokenization of TimedCommand/ActionPacket. Its `sample`, `log_prob`, teacher-forced NLL, and entropy diagnostics share factors/masks. PPO scores the exact sampled command packet including END/timing/active arguments, not the physical event reduction. It must not hide action clipping or average component log probabilities.

The implemented v1 decoder uses integer millisecond offsets (0...period), 64 categorical subcell bins per pointing axis, and signed relative components in -4095...4095. Signed components use an exactly scored radix-64 coarse/fine factorization; the last coarse group's unavailable fine values are masked. Scroll uses eight integer units per logical point by default. Operation, time, active arguments, END, and every digit contribute to joint likelihood. Conditional entropy is a sum at the visited prefixes, explicitly a surrogate diagnostic/regularizer rather than exact autoregressive joint entropy. Forced END at capacity has probability one.

Global and detail images have separate normalized content rectangles. Feature fusion maps between them through surface coordinates; padding is not stretched into content. The dense pointing grid is the detail encoder's stride-eight grid with clipped valid cell bounds. Cursor crops use the observed pointer, and unknown pointer state masks the entire cursor branch.

Observed-control features have width178: key codes0...127; buttons0...31 at128...159; modifier bits16...23 at160...167; normalized pointer within the observed surfaces' union at168...169; tanh recent raw dx/dy divided by2048 at170...171; tanh point-scroll dx/dy divided by128 at172...173; bounded log1p seconds since input at174; pointer-in-surface at175; authoritative state validity at176; interval coverage at177. No future demonstration command is a feature. Observed source and receipt times must both satisfy the cutoff before actor-side executed events enter this representation.

## Storage and concurrency

Native LibraryStore is the sole metadata writer. Source recording files are append-only until sealed, then immutable. Agents link recordings; exclusions/context selections belong to dataset revisions. Compute reads sealed data and writes artifacts into allocated per-job staging paths; the coordinator atomically publishes validated manifests. No learner writes the live catalog directly.

Frame blocks use native LZFSE with explicit uncompressed byte size and checksum. A shared-memory frame reference is leased until the receiving process owns its MLX copy. Bounded queues have an explicit overload outcome; neither memory ownership nor dropped intervals may be implicit.

Cross-language fixtures, geometry mapping, malformed messages, causal input state, action admission/cancellation, exact raw round-trip, actual model gradients and production-entry workflows are required checks.

### Recording packages and recovery

A source package is `Recordings/<UUID>.astrarecord`. `manifest.json` uses schema version1 and milliseconds-since-1970 dates. Its optional `indexPath` is absent for the original `index.sqlite`; after recovery it is `Recovery/<UUID>/index.sqlite`. Swift readers use `RecordingManifest.indexURL(in:)`. Python readers must apply the same relative-path and symbolic-link checks, honor the manifest pointer, and open the index read-only. They must never assume the root index remains authoritative after recovery. Completed and recovered indexes are sealed as standalone SQLite databases without required WAL sidecars.

The index contains `frames` (UUID, observed time, source time, shard, byte offset, block length, encoded FrameBlock), `events` (sequence, observed time, source time, encoded RawInputEvent), and `health` (observed time, status, optional message). Indexed integers must fit signed64-bit storage; the JSON contract remains unsigned monotonic nanoseconds. Source event times may arrive late or out of order and are never rewritten. Frame UUIDs are unique; accepted frame observation times do not decrease. Raw event batches are validated completely before admission.

The writer synchronizes frame bytes before committing their index references. A successful seal drains pending input/frame indexes, checkpoints and closes SQLite, and then atomically publishes the terminal manifest. Storage failures stop admission; failed finalization throws consistently and keeps source data available for recovery. A per-package advisory lock is held until the writer and its database are closed. This lock does not replace the separate application-wide library/control leases.

`LibraryStore.recoverInterruptedRecordings()` runs explicitly at workspace startup. It skips active writers and rebuilds abandoned recording indexes from validated durable events and checksummed frame prefixes, including complete frame blocks written before an index transaction. Recovery preserves original frame shards, index and WAL files, copies source metadata for diagnosis, and publishes a separate index through the manifest pointer. Partial or corrupt source tails remain untouched. Failed derived work is cleaned up when possible; a process crash may leave an unpublished recovery directory. Sealed packages are reconciled with the catalog by reading their manifests, without a full integrity scan on every snapshot.

`firstInvalidObservedNanos` is a conservative availability-time boundary. Dataset builders must exclude supervision or temporal/action windows that cross it; keeping a raw record does not establish training validity. After an unclean shutdown, frames beyond the last durable input event are excluded conservatively because the current source format has no explicit producer watermark proving input coverage during silence. Health entries are diagnostic evidence, not such a watermark. See [recording storage verification](verification/recording-storage.md) for exercised guarantees and remaining gates.
