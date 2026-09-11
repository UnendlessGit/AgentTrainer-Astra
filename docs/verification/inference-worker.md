# Local inference actor

The packaged worker entry point now accepts `--role actor`. It loads Astra checkpoints, consumes native shared-frame references, runs the same `AgentPolicy` used by BC/PPO, preserves observation recurrence, and emits native `ActionPacket` commands. It does not post macOS input. Native capture/control coordination, installed-bundle qualification and live end-to-end latency remain separate release gates.

## Process and ownership

The default `compute` role retains learning jobs and diagnostics. The `actor` role advertises only `capabilities`, `ping`, `shutdown`, `inference.prepare`, `inference.reset`, `inference.step` and `inference.close`. A dedicated actor thread owns checkpoint construction/loading, preprocessing, MLX arrays, RNG, recurrent state and ring-reader cleanup. The input thread performs bounded protocol framing and remains responsive to liveness/shutdown. One inference operation may be in flight; overlapping requests are rejected instead of queued into a growing backlog. Each operation produces one correlated terminal `ack` or `error`, with no preliminary queued acknowledgement.

`shutdown` closes admission, joins the actor and reports whether it stopped. A computation already in progress may finish, but its result becomes a cancellation error instead of an actionable packet if shutdown intervened. The native coordinator must stop input immediately, ignore obsolete request/run identities, and join the actual worker process before retiring any unacknowledged ring leases. Closing the Python reader by itself never releases a native slot.

## Protocol payloads

All inference operations require the envelope's run UUID. JSON integers remain Python integers through timestamp/identity validation and packet emission; execute time is never calculated through floating point. Payload schemas reject unknown fields.

`inference.prepare`:

```json
{
  "checkpointPath": "/absolute/library/checkpoints/<checkpoint-uuid>",
  "ring": {"path": "/absolute/private/run/frames.astraring", "ringID": "<ring-uuid>"},
  "seed": 72,
  "deterministic": true
}
```

Seed and deterministic mode are optional, defaulting to `0` and `true`. Preparation loads and verifies the immutable checkpoint locally, opens the exact run/ring identity read-only, and returns model/action configuration, checkpoint ID/signature and `needsReset: true`. A second stream requires closing the current run first. Model parameters come from the checkpoint; there is no runtime download, PyTorch use or fallback random initialization.

`inference.reset`:

```json
{
  "confirmed": true,
  "episodeID": "<new-episode-uuid>",
  "contextIDs": [],
  "checkpointPath": "/optional/new/checkpoint/<checkpoint-uuid>",
  "seed": 72
}
```

The native coordinator supplies `confirmed: true` only after the environment reset has actually completed. A reset clears persistent observation state and remembered input age, establishes a new state UUID, and may activate a new checkpoint of the same model/action signature. Different configuration/capabilities require a new actor run. Omitted seed preserves the actor's stochastic stream; an explicit seed starts the requested reproducible stream. Reset failure after a confirmed environment reset keeps old weights but blocks their old recurrent state until a valid reset succeeds.

`inference.step`:

```json
{
  "observationID": "<fresh-observation-uuid>",
  "episodeID": "<current-episode-uuid>",
  "previousStateID": "<state-uuid-returned-by-reset-or-previous-step>",
  "cutoffNanos": 1000000000,
  "geometryRevision": 7,
  "frames": ["<SharedFrameReference object per observed surface>"],
  "controlState": "<ControlState object>",
  "executedEvents": [],
  "intervalCovered": true,
  "contextIDs": []
}
```

The quoted frame/control placeholders above stand for the structured objects in [the shared contracts](../CONTRACTS.md) and [ring layout](shared-frame-ring.md), not literal strings. A step requires complete valid control coverage, the current episode/state identity, strictly advancing observation time and nonoverlapping policy intervals. Context choices must match the current episode. Recent duplicate observation IDs, consumed ring leases, earlier geometry revisions and a changed surface descriptor without a new aggregate geometry revision are rejected. Recurrent state identities, monotonic cutoffs and per-slot ring sequences prevent replay; the bounded observation-ID history retains the latest 1,024 successful IDs.

Frame source/availability times and executed input must be at or before the cutoff; a source cannot be available before its source timestamp. Executed-event sequences must advance, with each event's availability in the newly covered interval. Only observed state and executed history enter control features; future teacher actions or predicted held state do not. The last real input's availability time persists through idle steps, matching recorded-dataset input-age features; reset marks it unknown until actual event evidence is provided. The initial elapsed feature uses one configured period, matching training; subsequent steps use the actual cutoff difference.

Successful output includes:

- `packet`: native `ActionPacket` with fresh packet ID, run ID, continuous run sequence beginning at zero, observation ID, aggregate geometry revision, `executeAtNanos = cutoffNanos + lead_ms * 1_000_000`, immutable `durationMs`, and decoded `TimedCommand` objects.
- `logProbability`, `value`, `conditionalEntropy`: finite FP32-model results, with exact joint packet probability including active conditional factors and END.
- `checkpointID`, `policySignature`, `episodeID`, `stateID`, `needsReset`: current actor identity, with a fresh state UUID for the next request. Persistent GRU state remains internal; the packet decoder resets for each observation.
- `surfaces`: the exact validated geometry used to decode commands.
- `releasedFrames`: exact ring acknowledgements for the frames successfully copied into owned MLX arrays.

The native coordinator releases each `releasedFrames` entry once before considering the packet or error. Post-ingestion errors also carry this array and `needsReset`; a copy that never completed supplies no acknowledgement. This releases valid owned copies even if later preprocessing or policy execution fails, without authorizing reuse of an invalid or uncertain lease. The actor commits new recurrence/RNG only after finite outputs and decoded command validation succeed. `inference.close` accepts an empty payload and retires this actor's model/reader state; the worker remains available for a new run.

## Visual equivalence and resource behavior

Raw BGRA has one owned MLX ingestion copy from the ring. Alpha composition, RGB conversion, resizing, normalization and cursor extraction then remain on Metal. No image is read back to CPU or routed through Pillow in the actor. CPU work prepares bounded geometry-only filter tables; the eight-entry coefficient cache is bounded, and the separable kernel avoids allocating an image-by-filter-support gather tensor.

The resize contract matches the [Pillow 12.3.0 RGB Lanczos implementation](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c): fractional crop bounds, scale-dependent Lanczos3 support, normalized signed 22-bit coefficients, and uint8 rounding/clipping after each separable pass. Native premultiplied BGRA is composited onto the ImageNet mean, rounded to RGB uint8 before resize, normalized once, and padded with normalized zero. Global/detail alignment and cursor crop geometry use the same training conventions. Recording validation retains signed SQLite timestamp defaults; the actor explicitly selects the native UInt64 bound through shared frame/event/observation/preprocessing validators.

## Verification

On 2026-09-08, macOS 27.0, MLX 0.32.2 and CPython 3.12.13:

```sh
.venv/bin/python -W error -m pytest python/tests/test_inference.py python/tests/test_data_preparation.py python/tests/test_recordings.py python/tests/test_protocol.py python/tests/test_frame_ring.py -q
.venv/bin/python -W error scripts/check_inference.py --qualify-local --report .local/inference-native-size.json
```

The component suite includes 16 inference cases: nine visual/cursor comparisons; deterministic and stochastic real-checkpoint/native-ring policy equivalence; idle input-age continuity; error acknowledgements; reset-only checkpoint activation; owner-thread admission/close; and the actual worker actor protocol. Together with the related observation/recording/protocol/ring checks, 42 tests passed with warnings treated as errors. The real Swift fixture exercises timestamps above `Int64.max`, exact geometry, ring-slot reuse and local checkpoint loading. Actor outputs and carried recurrence match independent CPU training preparation. Processing failure returns the completed-copy lease, wrong-ring references return none, and replacement-model signature failure cannot activate incompatible weights.

The synthetic 1280×720 qualification used the full default 34,638,639-parameter policy with pinned pretrained ConvNeXt weights. Every prepared visual/geometry field matched the CPU reference exactly, as did decoded commands, recurrent state, value and joint log probability. MLX peak active memory was 778,423,552 bytes. One raw ingestion plus preparation took 65.50 ms in that invocation; this is a single observation including setup, not a warm latency distribution or a 10 Hz scheduling qualification.

For the packaged actor, the verifier is:

```sh
.venv/bin/python -W error scripts/check_inference.py '/absolute/path/to/astra-compute' --offline
```

This runs a real native-ring/checkpoint/error/recovery workflow against the specified executable with only system PATH and denied network access. Its generator/test process uses development dependencies; the target actor receives no venv or PYTHONPATH. A frozen invocation has not yet been claimed by this document. Remaining gates include packaged execution, native coordinator/control integration, worker-crash cleanup, live capture geometry/permissions, aggregate actor/learner resource admission and measured scheduling under contention.
