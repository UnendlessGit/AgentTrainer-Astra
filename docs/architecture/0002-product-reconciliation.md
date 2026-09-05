# ADR 0002: Product additions after the original audit

Status: accepted. ADR 0001 remains the independent architecture baseline. This record follows the complete source/documentation/test audit and isolated UI observation.

## Decisions

1. Add optional generic categorical context conditioning to agents. Fields/values have stable IDs, dataset/model vocabularies are immutable, blank values are explicit, and display renames do not alter learned semantics. Use learned field/value embeddings in the temporal input, with no predefined domain labels or reward-detector information. A change during control requires an explicit boundary.
2. Reusable environment/recording profiles snapshot source selection intent, geometry, capture settings, controls and context. Rebind disappeared targets deliberately; do not silently capture/control a replacement source.
3. Support explicit correction recording with bounded causal pre-roll and source checkpoint identity. Preserve actual raw observations/events and execution receipts; expert supervision starts at the trigger. Physical takeover alone does not record. Resume remains explicit.
4. Provide Astra-native import/export and independently relocatable recording/model storage. Use content checksums, durable staging/migration journals, reference-aware source ownership and visible recovery. No original recording/model compatibility is required.
5. Distinguish execution checkpoint selection, exact learner resume, and actor-only warm start. Preserve named/pinned/active checkpoints during bounded retention. Forking a learning project is explicit and cannot mutate the source.
6. Add an optional loopback external reward adapter using the same reward/outcome/readiness contract as local OCR, image, manual and fixture signals. Messages include protocol/run/episode/sequence identity and timing; reject stale/duplicate/wrong-session messages and require a readiness acknowledgement. Keep payloads/queues bounded and signal liveness separate from reward magnitude. No network configuration is needed for local authored RL or inference.
7. Raw event records keep source and observation timestamps, original order, repeats, smooth-scroll details, and reconciliation/boundary provenance. Initial state must be observed causally. Derived shortcut/exclusion labels never erase raw timing evidence. User-visible recording is scoped to the selected environment and explicit whole-desktop mode.
8. Preserve recoverable frames/events on writer, manifest, disk, or attachment failure. Quarantine invalid learning intervals with explanations; do not delete a recording because publication failed. Imported corruption and absent data are different UI states.
9. Optional observation/attention/action inspection is bounded and cannot silently delay the primary control path. Diagnostic exports omit raw user content by default and offer a preview. Charts preserve true step/time x coordinates and missing samples.
10. Include augmentation as an explicit training experiment, with temporally coherent per-sequence photometric transforms and jointly transformed spatial labels when spatial variants are evaluated. Keep PPO's scored observations deterministic. Auxiliary predictive objectives are not enabled merely because the original used them; add them only with independent held-out/closed-loop benefit and strict split membership.

## Rejected inheritances

Do not copy fixed 146-value actions, short-window recurrence, in-process learner/control coupling, shortcut-only RL, implicit reset sleeps, global mutable learning identities, heuristic input/parameter capacity labels, named-brain exceptions, custom palette tuning, updater service, or original signing/path rules. Similar tools (SwiftUI, MLX, SQLite) remain appropriate where independently justified.

The full evidence and remaining release gates are mapped in `docs/LEGACY_AUDIT.md` and `docs/IMPLEMENTATION_PLAN.md`.
