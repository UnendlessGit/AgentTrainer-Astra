# Reset runner and authoring verification

The reset subsystem uses the [versioned reset/source-coverage contract](../architecture/0006-reset-and-source-coverage.md). Manual Ready remains the default. Authored plans contain timed control packets, pauses and typed condition waits on the same environment; their actions use a separate reset UUID and do not become policy samples. The editor saves immutable reward-definition revisions and never executes a reset.

## Runtime guarantees

`ResetRunner` binds the actual driver through `ResetControlDriver`. Preparation must prove joined prior owners and the same current environment/process/geometry. The driver handles the existing control helper, paired guardian, receipt stream and observation producer; the runner does not allocate an alternative posting backend. Empty capabilities require no controller arm.

Each submitted packet must return a complete terminal execution receipt with matching reset, packet, sequence, command indices and timestamps. An admission acknowledgement cannot advance the plan. Actions are validated against the bound scope before driver preparation. A dedicated deadline/health task can stop admission and cancel operation waits while an async submission is blocked. The runner nevertheless joins that operation before releasing or reporting completion.

Cancellation, timeout and scope/action failure wait for actual owned cleanup. Release is awaited independently of caller cancellation. Invalid/future release proofs remain unconfirmed, and cleanup is not repeatedly submitted after an attempt has already joined it. Only a configured condition timeout can retry, within the overall limit and only after confirmed cleanup. Manual confirmation is scoped to the currently waiting reset ID; stale clicks cannot start another reset or episode.

Final readiness requires a source observation after the release barrier, or explicit matching unchanged coverage through that barrier. A new wall-clock timestamp does not refresh an old image. Missing, stale and low-confidence readings remain unknown. The returned `ResolvedRewardSnapshot` preserves the original readings and supplies the exact validated values for baseline initialization. Native orchestration must also obtain source evidence at or after Ready for its first actual actor cutoff.

## Permission-free evidence

On 2026-09-12, `swift test --filter 'reset|Reset|reward|Reward'` passed **31 reported tests**, including seven parameterized unproven/foreign evidence cases, without compiler warnings. The new reset-specific focus passed 14 tests before the final integration. Exact logs are `.local/reset-focused-tests2.log` and `.local/reset-final-integration.log`.

Coverage includes:

- Old definitions without reset plans, bounded schema validation, unsupported keys, ambiguous fields, interval ordering and fractional raw motion rejection.
- Shared readiness/baseline snapshots, unchanged source coverage, preserved timestamps, source/surface identity, confidence, false freshness and conflicting definitions.
- Manual confirmation identity, empty-capability operation, authored timeout retries and reset-versus-policy UUID separation.
- Incomplete/admission-only receipts, changed scope, foreign episode readings, unjoined previous ownership and invalid cleanup proof.
- Cancellation during delayed preparation, an independent deadline while a post ignores cancellation, and retained ownership while a release remains blocked.
- A pause-boundary regression where health work crosses its deadline, preventing unsigned timestamp underflow.
- Key-chord release order, overflow-resistant authoring and pointer scope/half-open coordinate bounds.

The fixtures emit only in-memory virtual effects. No test grants privacy access, posts a Core Graphics event or changes an OS input mode.

## Native views

The actual Episode editor and reset controls were rendered in owned NSHostingViews at 1120×760 and 860×580, with light/dark appearances. Twelve base and six scrolled images are retained in `.local/reset-ui-renders2`. The review led to readable second-based pause/timeout fields, an earlier pointer-reference chooser, and guarded bindings when an advanced command row is removed. Key press/chord, click, absolute/relative pointer, scroll, pause and condition wait steps are editable, reorderable and removable. Individual timing/commands remain available in a disclosure section.

These renders do not establish keyboard or VoiceOver interaction. The reusable `NativeControlSession` and `NativeResetDriver` now provide the real-helper binding; see [lifecycle verification](native-control-session.md). Native desktop orchestration still owns activation and episode handoff. Live reset rehearsal, actual target consumption, installed helper attribution, physical takeover and denied/revoked access still require the separate native workflow qualification. No full application completion is claimed here.
