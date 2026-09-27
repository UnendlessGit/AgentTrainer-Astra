# Local live state values

The optional local state source supplies existing `RewardSignal.kind.manual` values from another process on the same Mac. It supports finite numbers, text, Booleans and explicit unknown values. It does not supply action labels, input events, model features, reward markers or marker coverage. Retrospective reward feedback remains separate.

`LiveSignalStore` owns a session and the frozen manual signal definitions. Before observing each reset, the host calls `bind(episodeID:resetID:)`. The store publishes a fresh binding UUID and clears prior values, seals and update sequence. This source generation is distinct from the later episode runner's callback generation. Reset observations use `readings(binding:cutoffNanos:)`; actor observations use `seal(binding:observationID:cutoffNanos:)` exactly once. Both select the latest received value at or before the actual cutoff. Neither retimestamps an old value. The existing reward evaluator applies each signal's maximum age, minimum confidence and post-cleanup evidence barrier; missing/stale/unknown state is never converted to zero.

Admission assigns `eventNanos` and `observedNanos` from the injected host monotonic clock under the store lock, after validating the whole update. Sender clocks are forbidden. An update must be later than the most recent sealed cutoff, and seals must advance with distinct observation IDs. Thus packets arriving after an observation cannot change it. History is bounded to 4096 retained readings and a 4 MiB scalar payload budget; losing an older reading leaves an older cutoff unknown. Whole batches admit atomically. A forward sequence gap latches current values to unknown without advancing the expected sequence. Newer updates cannot conceal the missing history between reward cutoffs: this binding remains unknown until a fresh reset binding is announced, so required rewards stop the episode. Replayed/older sequences are rejected without poisoning an otherwise valid binding. Ordinary reconnect queries the expected sequence and can continue without a gap. Binding replacement discards the old episode completely.

`LoopbackLiveSignalServer.start(store:)` binds an ephemeral TCP port specifically on IPv4 `127.0.0.1`; there is no LAN listener. Startup exposes its endpoint only after bind/listen succeeds. A random 256-bit in-memory token authenticates every request together with the session UUID. The token is not logged, included in training artifacts or persisted by the server. Status reports connected only after a successfully authenticated request. Endpoint debug descriptions redact the token. One connection is serviced at a time; reconnecting retains the current binding's next sequence. A dedicated socket owner thread uses a bounded poll loop and closes its own descriptors. `stopAndJoin()` waits for that owner, then the store is closed. The host retains the server across episodes and joins it when the learning session ends.

Messages are newline-delimited JSON, at most 256 KiB including the newline. Requests are processed sequentially with bounded receive/response storage. An unauthenticated connection has a fixed two-second authentication deadline; authenticated idle connections expire after 30 seconds. An ordinary disconnect does not invent new values: existing receive timestamps expire normally under the configured maximum age.

## Wire protocol

Every request contains `version: 1`, `op`, `token`, and `sessionID`. The example token below is a placeholder, never a real credential.

```json
{"version":1,"op":"binding.get","token":"<session-token>","sessionID":"<session-uuid>"}
```

The reply is `{ok:true,sessionID,binding,nextSequence}`. `binding` is null while no episode is announced, otherwise it contains `sessionID`, `bindingID`, `episodeID`, `resetID`, `publishedAtNanos`, and `signals`. Each signal describes `id`, `name`, `maximumAgeMS` and `minimumConfidence`.

```json
{"version":1,"op":"values.put","token":"<session-token>","sessionID":"<session-uuid>","bindingID":"<current-binding-uuid>","episodeID":"<current-episode-uuid>","sequence":0,"values":[{"signalID":"<signal-uuid>","value":42,"confidence":1}]}
```

The reply is `{ok:true,receipt:{bindingID,sequence,nextSequence,receivedAtNanos}}`. Values must target distinct announced signal IDs. Text is limited to 4096 UTF-8 bytes. Numeric values must be finite; integers that cannot be represented exactly as the signal's Double value are rejected. Null publishes explicit unknown. Optional confidence defaults to one. Extra fields, including sender timestamps, are rejected. Failed requests return `{ok:false,code,message}`; authentication failure closes the connection without echoing secrets.

A binding may change between reading it and publishing. Query it again and send a fresh measurement. After an uncertain reply or reconnect, query `nextSequence`; do not replay an old sequence blindly. `binding.get` is not a value heartbeat and never refreshes freshness.

`scripts/live_signal_client.py` uses only Python's standard library. Its CLI prompts for the token without echo and repeats an explicitly supplied test value. Real adapters can reuse `LocalStateClient.publish({...})` with measurements from their application. No OS input is posted by this source.

## Focused evidence

The native implementation builds. One socket/causality check exercises the actual loopback listener, receive-side timestamps, lookup before a value exists, rejection at a sealed cutoff, future-value exclusion, latched sequence-gap unknowns, reconnect with preserved expected sequence, stale-value rejection by `RewardEvaluator`, reset generation rejection and joined closure. Logs: `.local/live-source-native-build.log` and `.local/live-source-socket-test.log`. The native Host/UI integration subsequently passed a generated capture / virtual-input run with the actual socket client and MLX actor/collector/learner. Two reset bindings received36 authenticated updates; eight admitted reward intervals exactly matched receive-time values at their cutoffs plus the rate integral. PPO changed weights with two optimizer updates. Capture, client, listener, control and compute workers joined. Evidence: `.local/verification/desktop-host-live-values-20260927-1/desktop-host-report.json`. This uses a small numerical-test policy and does not qualify physical input, privacy grants or production learning quality.


The desktop UI explicitly enables the source, shows connection state/current readings and copies connection settings only on operator action. It waits for current values before running the reset/control path. One per-episode producer seals values before sampling and joins its callbacks on close; this retires its binding without closing the run's listener. Stop, startup failure and normal completion join the listener. Session tokens and ports are absent from persisted learning metadata; the source kind and session identity remain available for provenance. The distribution includes an offline client guide and the optional standard-library example.
