# ADR 0010: Causal queued-control observations

Status: implemented as explicit model schema3, with native ledger, shared Python transport, practice/BC/PPO replay and checkpoint warm-start support. Existing schema2 remains the default and retains its exact path. Focused numerical and generated native-host integration checks pass; broad learning and installed physical qualification remain open. GRU only. This decision concerns action feedback under execution delay, not the separate passive-cue learning results.

## Evidence and consequence

`TemporalCore` currently receives visual features, 178 actual-control/history features, elapsed time and contexts. The packet decoder resets its own GRU every decision. Both `InferenceSession` and `ReinforcementTrainer` retain the observation GRU state, without folding the sampled packet into it. Native `InputExecutor` owns admitted pending packets, but `observation()` exposes only actual control state and executed events. Both continuing native loops already await admission before taking their next observation.

Two identical observation histories can therefore have identical persistent state despite different previous categorical samples still waiting to execute. Longer GRU credit alone cannot reconstruct an unobserved random draw. With positive execution lead, this can encourage repeated clicks/presses, excessive scrolling or reversals before prior actions take effect; value predictions also omit part of the delayed environment state. This is a representation deficiency, not evidence that these faults occur in every task. Idempotent input scheduling limits some physical duplicates but does not restore missing policy information.

The existing opposite-cue likelihood probes use passive observation histories and issue no previous actions. An empty queue is correct there. This gap cannot explain their chance rankings, and queue work must not be presented as the remedy for that separate optimization result.

## Observation contract

Add a separate versioned control-feedback channel, produced by the control owner at the **same atomic cutoff** as actual state/history. It contains:

- All admitted outstanding packets in original order, retaining immutable command identities, original semantic arguments and execution times.
- Admissions since the previously acknowledged control-feedback cursor, including packets that completed before this observation. This retains knowledge of attempted/no-op actions even when current physical state looks unchanged.
- Terminal transitions since that cursor, including packets admitted in an earlier observation. Otherwise a previously queued packet that finishes or becomes a no-op between observations silently disappears.
- Authoritative progress known at the cutoff and explicit coverage/availability. Scheduled time alone never proves execution. A due-but-unposted command remains pending, with its overdue timing represented.

Admission and progress evidence must have become available by the cutoff. Current-decision output is excluded. Since the present native loops join admission before predicting again, an unresolved submission remains a stop/audit condition, not guessed input. No action receipt arriving after a cutoff can retroactively modify its observation.

For ongoing motion, preserve the original trajectory and interpolation progress, including its preceding anchor. A remaining endpoint alone loses the already scheduled path. Do not expand the input into thousands of synthetic millisecond events. Keep the helper's existing packet/command/byte bounds; missing history or overflow is explicit, never truncation into a plausible empty queue. Run/episode/geometry/packet IDs authenticate evidence but are not learned numeric features.

Actual held controls, cursor, executed history and their timestamps remain unchanged. Do not substitute a predicted future held-state vector, retimestamp state, or restrict policy masks according to planned holds. Rewards/detector values remain outside actor observations.

The [native/wire contract](queued-control-wire-v1.md) defines opt-in negotiation, packet/progress records, cursor and memory bounds, in-flight invalidation, Stop/reset behavior and immutable rollout transport. Schema3 application policies negotiate this channel; schema2 policies omit it. Collection record2 and observation schema2 carry the exact original snapshot, while checkpoint and rollout container versions keep their existing independent meanings. [Native verification](../verification/queued-control-native.md) records the explicit capacity limits and current evidence.

## Shared GRU representation

Encode ordered semantic commands with a small command GRU and reduce ordered packet summaries into one control-feedback embedding. Commands carry operation/key/button, source role and actual normalized pointing coordinates or relative/scroll arguments, relative scheduling/admission times, and known progress. Past dense-cell indices cannot be reinterpreted against the current image grid. Batch independent packet encodings together with a fixed64-row shape and the checkpoint’s fixed16/32/64 command capacity, avoiding shape compilation at first real action admission. A command GRU64 and packet GRU128 feed a bias-free128→512 residual in the production configuration, adding208,360 parameters. Feature index constants are versioned in `model/queue_layout.py`; scheduling cost remains an empirical gate.

Add the embedding through a residual projection into the existing temporal input before its normalization, then use the existing persistent two-layer GRU and unchanged action/value heads. The residual's output projection starts at zero; the command encoder starts with ordinary finite weights so it can receive gradients once that projection learns. The inactive/empty branch has an exact zero output with no trainable bias. This preserves initial behavior instead of allowing an untrained queue branch to perturb a BC policy.

PPO stores the exact cutoff-bound channel in immutable observations and uses it for current-state replay, likelihoods, burn-in/full-prefix replay and bootstrap values. Replaying **recorded past behavior actions** is valid conditioning; resampling them under the updated policy would change the trajectory. The native actor, practice adapter, BC loader and PPO loader must produce one common representation.

## BC causality and coverage

Never construct this channel from earlier **shifted human targets**. At lead 200 ms and period 100 ms, the preceding teacher packet can contain a human action occurring 250 ms after its own observation; feeding it at the next 100 ms observation reveals future recorded behavior. Its position in the teacher sequence does not make it an actually issued historical command. ADR 0001's prohibition remains intact.

A new human-only recording can certify an empty Astra-owned queue only with continuous control-exclusion authority across the recorded interval, beginning after prior control has joined. The recorder now holds the existing cross-process desktop-control lock before admitting any producer and until capture/input/compression join and terminal publication. It persists optional bounded start/end/join proof. A joined start boundary or UI guard alone is insufficient; interrupted/recovered recordings without complete proof remain unavailable. This proves absence of Astra-owned queued controls, not absence of other OS automation. Older/unproven sources retain explicit unavailable coverage. Both known-empty and unavailable input keep the new residual inactive. Their BC loss never consumes future teacher packets through the persistent GRU.

A zero-initialized residual prevents regression, but it does **not** teach queue-aware behavior from human-only demonstrations. Train the branch using genuinely causal expert/practice trajectories, logged on-policy experience or future expert feedback collected with that real queue. Practice must snapshot its actual admitted scheduler state before the current oracle command. Do not fabricate pending packets from human labels, randomly attach unrealized plans to recorded pixels, or claim queued-control competence from a dormant branch. Any behavior-policy activation still occurs at a fresh, joined episode boundary.

## Compatibility and minimal acceptance

Introduce a new model/observation schema for this learned input. Keep schema 2 loading, serialization, policy signatures and inference behavior exact on the old path. The checkpoint container may remain version 1 if its integrity semantics are unchanged. Old artifacts do not acquire invented queue evidence.

The explicit `checkpoint.addQueuedControl` warm start writes a **new checkpoint ID/signature**, copies all old parameters, and adds the zero-output residual branch. It records source checkpoint/signature/policy hash, uses kind `initial`/step0 and retains no optimizer, sampler, recurrent state or dataset binding. The native checkpoint manager exposes this as a deliberate copy operation. It is not exact optimizer/sampler resume across architectures. Do not silently load missing new weights, reinterpret old recurrent anchors, or reuse old PPO likelihood/state evidence. A new immutable dataset revision can retain original recording bytes while declaring the correct empty/unavailable feedback provenance. Ordinary same-schema checkpoint resume remains exact.

Acceptance before making the new path the default: prove same-cutoff admission causality, partially executed trajectory fidelity, reset/empty/unavailable behavior, old-path parity and zero-residual parity; then require real collected queue snapshots to replay PPO ratios/values and one shared BC/practice/inference path. Measure added warm latency under the existing deadlines without reducing model quality. A small delayed-action task should exercise distinct queued actions with otherwise equal visible/actual-control state. Broad production qualification remains separate.

The live-value workflow and this shared queue implementation are complete at the focused integration level. General delayed-control learning, warm deadline qualification and physical release gates remain separate. The unchanged-schema [GRU loss/clipping results](../verification/gru-loss-clipping-results.md) did not establish reliable long-delay learning; they neither validate nor invalidate this causal observation channel.


## Current activation and limits

The Advanced option applies only to fresh models. Saved checkpoint selection inherits its architecture; the checkpoint manager’s separate copy operation is required for a schema2 warm start. Copies keep the original timing/capabilities/contexts and do not auto-select themselves or reuse historical PPO likelihoods. Human demonstrations can still train schema3 visual/temporal/action weights with an inactive queue branch; causal practice or reinforcement experience is needed to learn the new branch.

Native schema3 preflight rejects a nominal cadence requiring more than32 outstanding packets (`ceil(lead/period)+1`). Byte capacity is charged from actual original packets and reserved future progress, so the key-up fixture counts are not universal guarantees. The UI explicitly describes long-lead/full-packet backpressure. Variable scheduling delay, raw event history and actual packet mix still use the bounded runtime admission checks; runs stop on insufficient history capacity. No packet, trajectory or observation is shortened to fit.

Correction pre-roll preserves and validates schema3 feedback, including a retained suffix that begins after earlier cursor acknowledgements. It stays review-only. The buffer reserves metadata as well as pixels and evicts whole oldest observation groups before the sidecar budget is exceeded. This prevents a valid queue-aware inference run from producing an oversized or unreadable correction sidecar.
