# Opt-in native queued-control evidence

September 27. Implements the native portion of [wire v1](../architecture/queued-control-wire-v1.md): Core types/validation, executor-owned original packets and measured semantic progress, bounded admission/terminal retention, atomic dual-cursor validation, helper negotiation/transport checks and session identity authentication. These initial checks preceded the shared model3 integration; schema2 continues to opt out. Later schema3 actor/collector/PPO checks are separate evidence and do not expand this native fixture’s claims. No action mask was changed.

The independent review prompted three additional checks: every unacknowledged admission must be uniquely represented; sample counts/offsets/relative cumulative deltas must match the actual expansion algorithm; completed packet originals must remain authenticated after terminal receipts disappear from the session's pending dictionary. The host's bounded original map is independent of executor progress and expires only after terminal feedback acknowledgement. Corrupt negotiated evidence stops admission.

Focused command: `./script/swift.sh test --filter 'executor|nativeControl|feedbackReservation|feedbackEvidence|controlLease'`. The regression run reported **55 platform tests and 3 Core tests passing**, with no compiler warnings. These include 16 new feedback test functions (29 parameter cases), covering:

- Legacy omission and negotiation failures; real initial epoch and independent cursor payloads.
- Admission/terminal-between-cuts, no-ops and zero-command duration sentinels; immutable old snapshots.
- Exact partially completed relative trajectories with interleaved keys; absolute surface discontinuities.
- Selected preflight work remaining pending; blocked posts becoming unavailable; stale posts resolving after Stop without inventing cleanup proof.
- Both cursor failures leaving both histories intact, stale epochs, reserved terminal retention and rejection before sequence consumption.
- Worst-width progress encoding, malformed/borrowed admissions, progress corruption, full raw-history plus feedback exceeding the combined IPC limit.
- Original packet authentication after terminal receipts, 100 successive terminal acknowledgements releasing host identity retention, and the existing helper/guardian lifecycle regressions.

The built `AstraControl` executable separately passed hello capability negotiation, unsupported-version rejection before preparation, supported-version arm rejection without recovery protection, and joined unarmed shutdown. Those checks requested no permissions and posted no input. These are source-native checks, not frozen/installed-app qualification.

## Capacity measurements

The 256 KiB ceiling reserves `original JSON bytes + 2,048 + 512 × commands` per retained packet, plus a 4,096-byte envelope. This deliberately reserves future terminal/progress space. With full packets of key-up commands and no prior acknowledgements:

| Commands per packet | Admitted packets | Reserved bytes | Encoded final feedback bytes |
|---:|---:|---:|---:|
| 16 | 22 | 251,146 | 75,048 |
| 32 | 12 | 246,174 | 74,628 |
| 64 | 6 | 232,462 | 71,020 |

All admitted packets completed, their final rows remained available, and the next rejected admission did not consume its packet sequence. These are fixture-specific payload sizes, not universal maxima: operation arguments and source IDs affect the original JSON charge.

With 64-command packets, a 100 ms period/duration, one observation per decision and acknowledgement on the following observation, virtual streams completed 25 decisions at 0, 200 and 400 ms execution leads. Leads of 500, 1,000 and 2,000 ms backpressured after six admissions. Terminal rows still need the next observation's acknowledgement; counting only outstanding execution packets understates retention. No extra acknowledgement call was inserted to hide that limit. A separate empty-packet fixture filled all 64 rows/128 lifecycle changes and reclaimed them only after acknowledgement.

These results must **not** become a silent default reduction of supported delay/command capacity. Before activation, choose the negotiated budget/configuration policy explicitly and qualify the actual packet mix. Runtime warm actor latency was not measured while the independent GRU study was running. Subsequent integration implements shared Python/actor/collector/archive evidence, causal practice data, recording exclusion proof and the GRU input. Installed physical control, broad learning and warm production deadline qualification remain separate gates.
