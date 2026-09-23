# Silent control-state coverage

September 19, 2026. A control state retains the timestamp of its last actual
observation/change. Silence does not update that timestamp or its model features.

`ControlObservation.controlCoverageNanos` optionally certifies the helper's
current atomic cutoff. The executor emits it only for a valid, settled tracked
state with complete event history, an active unexpired lease, and healthy cached
backend/guardian state. The executor observation lock binds cutoff, lease and
trust checks; a detected health fault permanently invalidates this state and
requests generation-bound disarm. A later healthy cache read cannot revive it.
No blocking operating-system health query runs on this observation path.

Native actors, the native collector, Python actor and external observation
decoder validate coverage before acquiring pixels. A supplied proof must equal
the observation cutoff, follow the unchanged state timestamp, and accompany valid
state and complete history. Missing proof retains the existing freshness limit
(250 ms for native actors; the configured environment age for external
observations). Future, backward, noninteger or mismatched proof is rejected.

The actor echoes `controlCoverageNanos` in its immutable collection record. The
collector binds that echo to the owned observation and preserves it through
spooling and package reload. The proof is audit metadata; it introduces no model
input, precision or configuration-signature change.

The focused native run passed six tests / ten cases, including an actual virtual
`InputExecutor` remaining idle beyond 250 ms while receiving lease heartbeats,
unchanged state timestamps, sticky health loss, lease expiry/history gaps and
pre-copy native actor/collector rejection. The focused Python coverage, actor,
inference, external environment, assembler, reinforcement and collector run
passed 115 checks with warnings as errors. Real native mapped-slot fixtures
verify pre-copy rejection, actor echo and exact immutable artifact preservation.
These checks use no capture, privacy access or real input posting; live helper
and installed application qualification remain separate gates.
