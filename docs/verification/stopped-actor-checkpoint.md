# Saving a stopped actor without a learning update

`checkpoint.externalBoundary` is a compute-role job for a joined stop before a
trainable rollout exists. The request is:

```json
{
  "checkpointPath": "/absolute/source-checkpoint-uuid",
  "auditPath": "/absolute/collector-package-uuid",
  "destination": "/absolute/new-checkpoint-uuid",
  "resume": false
}
```

All three paths are required; `resume` defaults to false. The package must be
`audited` or `sealed`, have validated actor progress and known collector closure,
and match the checkpoint's policy/model/action identities. An aborted package,
unknown cursor or empty actor episode cannot invent resumable progress. Native
must independently hold the actor at the joined boundary, verify physical
cleanup, and prevent further draws until checkpoint publication and a confirmed
physical reset. The package is not an OS cleanup proof.

For `resume:true`, the source must contain external reinforcement state. The
package's `previousActorProgress` must exactly continue the saved stream/key/draw
and reset generation; only actor run ID may be rebound for a newly launched host
run. Current progress must contain new real draws after a confirmed actor reset.
Task, training configuration and contexts remain exact. The operation preserves
all optimizer tensor leaves, learner counters, learner RNG and consumed-rollout
history, changing only the saved actor cursor and new learner checkpoint identity.
It performs no forward pass, gradient, optimizer update, environment reset or
action. Fresh/BC initialization creates zero-update external PPO state with
unchanged numeric policy weights; it does not relabel the BC optimizer as PPO.

A successful response contains `checkpointPath`, `manifest`, `parameterCount`,
`checkpointPublished:true`, `actorProgress`, `boundaryCollectionID`,
`resumable:true`, `requiresEnvironmentReset:true`, `sourceKind:"external_rollout"`,
`provenance:"external_rollout"`, `boundaryOnly:true`, and `cancelled`. There is no
invented `rolloutID`, reward or PPO admission claim.

A private fsynced claim next to the package binds its manifest digest, source
checkpoint manifest digest, destination and operation. A process lock serializes
same-destination retries. A failed publication can retry only that exact request;
a completed retry validates and returns the same immutable checkpoint. Another
source, destination or PPO operation cannot reuse the boundary proof. Existing
destinations without the matching publication claim are rejected.

Cancellation before commit publishes nothing. Cancellation observed after the
atomic save reports the actual checkpoint with `cancelled:true`; cancellation
after the completed result is fixed cannot relabel committed weights as rolled
back. Job terminal kind derives from that result, rather than a later cancel flag.

Permission-free tests use real native frame leases and an independent collector
to create audited experience. They check unchanged policy bytes, a later run's
exact cursor, preservation of a nonempty real PPO optimizer, saved-task/cursor
mismatch rejection, exclusive proof consumption, idempotent retry, injected write
failure, and cancellation before/after commit and after the result. A model
forward trap proves the cursor-only operation does not execute the policy.
The final boundary-only, general jobs and real collector run passed **46 tests
with warnings as errors**, including seven boundary-publication cases.
These are small-policy correctness checks, not live desktop or learning-quality
qualification.
