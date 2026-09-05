# Original domain, documentation, and history audit

Source: original HEAD `86387f1` (2.4.5), read only after Astra baseline commit `505ec37`. This report describes evidence and decisions, not inherited maintenance requirements. No original application/test was run during these reads.

## Coverage

- Full current README (258 lines), DEVELOPMENT_GUIDE (323), and all five release notes (2.2.0 through 2.4.5).
- Full `Sources/AgentTrainer/Core/Domain.swift` (1–1955).
- Full `Tests/AgentTrainerTests/DomainTests.swift` lines 1–4000, including rereading the output-truncated region around lines 1920–2025. The RL audit covers 4001–5657. The root completed the learning auditor's additional test assignment after that task's resumption was unavailable.
- All available release commit messages and tag names, spanning v1.3 through v2.4.5.
- Historical, removed documents at `f7f36e5`: `WindowsRecorder/README.md`, `RECORDING_FORMAT.md`, `TRAINING_AUDIT.md`, `IN_PLACE_UPDATE_GUIDE.md` read in full. These explain portability, past learning decisions, and signing lessons; they are not current Astra instructions.
- Current icon JSON/SVG and `.gitignore` were read in full; binary icons were inventoried and the actual application icon was observed in the isolated UI audit. Third-party build caches and Git objects are not application source.

## Complete product shape from docs and domain

The original is a local macOS recorder, library, model editor, BC trainer, live executor, correction recorder, and later-added PPO trainer. Its shell splits Home, Record, Library, AI Models, Training, Run, Diagnostics, and Settings (`Domain.swift:4–24`). The recording flow supports display/window/screen or window regions, source selection, FPS/cursor, resolution/aspect, reduced color detail/compression, key exclusions, folder destination, trims, reusable presets, and user-defined categorical context. The Library handles multi-selection, inspection, reenactment, bulk metadata, import/export, folders, and independently movable data/model roots.

Model profiles select recordings/folders, vision/action/network/training settings, restrictions, versions and progress. Versions snapshot runtime interpretation and capabilities, with separate current optimizer state. BC can pause/resume and publish runnable snapshots. Corrections explicitly suspend Run, retain causal pre-roll, supervise human actions only from the trigger, and return control after stale predictions/history are cleared. RL can start fresh or warm-start BC and publishes an actor usable without rewards.

Release 2.3 introduced generic context fields/values, not domain-specific task presets. Release 2.3.5 repaired a concrete physical-input problem by reading HID state rather than mutable foreground-session accumulated state, and revision-guarding reconciliation. Release 2.4 added PPO and external rewards; 2.4.5 corrected runtime sampling, hosting direction, per-category exploration, lifecycle ordering, short rollouts, and startup cancellation. The historical Windows recorder used shared artifacts and translated key semantics; its removal does not erase the lesson that file-format integrity and producer neutrality matter.

## Decisions and lessons

### Retain the useful capability, give it an Astra-native representation

1. **Generic context conditioning.** Stable field/value IDs with immutable trained vocabulary separate semantics from display names (`Domain.swift:182–399`, tests 26–149). Astra should support optional user-defined context in recording/run configuration and the shared policy. Renames must not alter learned meaning; live changes require an explicit episode/control boundary. This is an addition to ADR 0001, justified by a useful general-computer-use feature, not its old vector layout.
2. **Reusable environments/presets.** Old presets cover the complete recording configuration and tolerate missing source/destination IDs (`Domain.swift:115–180`, tests 934–997). Astra's reusable environment plus recording profile should preserve this discoverability without coupling settings to a transient window ID.
3. **Portable Astra exports/imports.** The historical companion and current tests 1176–1376 demonstrate why self-describing, byte-preserving recordings, strict path validation, and transactional batch publication are valuable. Fresh start removes old-format compatibility, not the ability to move Astra data between libraries/Macs.
4. **Corrections with causal pre-roll.** Preserve the observation before the mistake, distinguish actor context from expert supervision, and evaluate against ordinary behavior as well as correction fit (README DAgger section; `Domain.swift:884–920`, `RecordingManifest` supervision fields). Do not call mere physical takeover a correction recording or include actor mistakes as expert labels.
5. **Independent storage locations and immutable snapshots.** Tests 1430–1581 cover copy/verify/switch/cleanup and refusing implicit merges. Large lossless recordings make relocatable storage and clear disk accounting particularly important for Astra. Shared source deletion must be reference-aware.
6. **Checkpoint selection and continuation are different.** Tests 1870–1937 distinguish exact optimizer restore from selecting weights only. Replacing the selected execution checkpoint must invalidate an unrelated continuation state. Test with actual model/optimizer tensors, not only dummy file bytes.
7. **Metric semantics.** Separate optimizer time, unique recorded duration, repeated experience consumed, and live environment time. Old `TrainingDurationSummary` and validation reports expose these distinctions. Show false presses and sparse-control support; loss alone is inadequate.
8. **Optional model inspection.** The original supports bounded activation/channel/saliency views (`Domain.swift:751–790`). Astra should provide an optional observation/attention/action inspection view after core workflows are reliable, with diagnostic rate/memory caps and no effect on policy output.

### Preserve the engineering evidence, not arbitrary legacy policy

- The original stores late input timestamps as a clamped nondecreasing sequence (test 894–904). Astra must retain event source time, arrival/observation time, and sequence identity rather than silently changing time. Causal reconstruction should explicitly handle late delivery.
- Keep passive physical tracking separate from delayed shortcut filtering. Tests 1980–2400 cover preserving unrelated keys/modifiers, side-button shortcuts, immediate HUD state, balanced shutdown, HID reconciliation, and stale-poll rejection. Astra must distinguish raw physical evidence, derived supervision exclusions, and synthetic execution.
- Do not use the old first-frame callback's later physical snapshot as earlier truth. Initial state must have its real sampling time and validity.
- Preserve raw key repeats and high-rate motion. The old interchange explicitly removed repeats and normalized/clipped continuous motion; Astra's versioned packet canonicalizer has explicit measured tolerances instead.
- Schema, parameter-shape, training-objective, dataset, and execution-contract identity are related but different. The original has many optional compatibility fields and exact-layout promises. Astra can start clean, but must make compatibility failures explicit and retain recoverable artifacts.
- The old fixed input-to-parameter ratio labels (`Domain.swift:1846–1864`, test 621–633) do not establish learning capacity. Use measured memory, timing, and held-out performance instead.
- Name-specific model deletion protections (`Domain.swift:1367–1413`, tests 1752–1868) are historical user-data accommodations. Astra should use explicit user-owned protection/pinning and reference-aware retention, not embedded names.
- Detailed custom palette/animation tuning and a solid custom app shell are not requirements for Astra's native light/dark interface.
- An automatic updater and legacy signing identity are outside the selected personal-installation release. Do not adopt the historical updater's certificate, path, or permission-preservation prescriptions. Build Astra with its own stable bundle identifier and the authorized ad-hoc distribution mode.

## Test and historical evidence limits

The original's reported release test/benchmark counts are historical claims, not checks run during this audit. Many domain tests correctly check corruption, transactions and lifecycle properties; others assert schema constants, exact heuristic bands or dummy checkpoint bytes. They do not prove closed-loop task learning, long memory, or installed-bundle input effects. Permission-sensitive tests can skip. Astra's acceptance gates therefore require real learning and installed workflow evidence in addition to numerical/unit tests.

The historical training roadmap preferred offline learning before generic PPO. That recommendation predates current RL support and does not supersede the user's request for first-class RL. ADR 0001 keeps PPO and bounded authored episodes, while evaluating hybrid demonstration use experimentally.

Additional tests 2401–4000 cover pointer initialization, pulse aggregation, input allowlists, immutable capabilities, relative-camera event types and transient consumption, serialized release/re-enable, modifier channels, native preview sizing, shared input buffers, visual diagnostics, optimizer/compiled equivalence, RNG isolation, packed temporal batching, context rows, feature reuse/locality, generated video-cache construction, correction supervision, static cadence and validation coverage. They support testing these invariants, not copying the old action vector or fixed-window architecture. In particular the pointer-initialization test accepts a future event as initial position and the short-tap test establishes presence rather than precise event order. Astra must use causal seeds and timed commands.
