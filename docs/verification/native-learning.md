# Native behavioral learning coordination

On 2026-09-08, `swift test` built the native application and passed all 60 Swift tests. This includes 12 native learning coordinator tests (one has three identity cases), using a local protocol fixture that does not import MLX, capture the screen, or post input. These tests establish coordinator behavior, not model learning quality.

The native Training tab selects linked recordings or explicitly labeled generated practice demonstrations, configures timing, controls, starting checkpoint and training settings, then submits dataset preparation, optional visual-weight initialization and behavioral training to the bundled compute worker. It publishes immutable configuration/results files and validated checkpoint identities into the workspace catalog. Progress includes loss, updates, decisions, throughput and memory, with a chart and readable metric table. Evaluation uses the checkpoint's saved dataset and reports demonstration likelihood separately from closed-loop success.

The exercised native boundaries include:

- Full initialization/training/publication/selection/evaluation sequencing, including terminal delivery before the acknowledgement callback resumes.
- Cancellation that waits for a saved terminal checkpoint before shutdown; cancellation before training or during checkpoint inspection stays a cancellation and does not claim an unavailable checkpoint.
- Rejection of wrong request, job, acknowledgement-run and checkpoint identities; dataset failure stops before model creation.
- Per-child generation guards so closing-child failures cannot poison the next job. Stop waits for the work it originally targeted. Quit closes admission, requests recording/learning stop together and waits for both owners.
- Atomic no-overwrite run artifact publication under twelve competing writers; bounded metadata reads through one regular-file descriptor, without following a final symbolic link.
- Read-only recorded-control discovery with exact source/observed timestamps and sequence agreement against the SQLite index.
- Agent copies reuse explicit links to immutable checkpoints. A copied agent can evaluate the source checkpoint and train a new child checkpoint without changing the original agent's selection or duplicating its weights. Unlinked checkpoints are rejected.

The native UI uses system SwiftUI surfaces and colors with no forced appearance. Actual light/dark rendering, keyboard navigation, VoiceOver, narrow-window layouts, installed-bundle behavior, interactive quit, and the complete native recording → training → evaluation workflow remain unverified. No frozen rebuild or UI automation was used in this review.

Selecting a starting checkpoint currently begins a new training run from its policy weights. Exact optimizer/sampler resume is implemented in the compute backend but is not yet exposed as a native Resume workflow. Native reinforcement training now has a practice-environment workflow and optimizer/iteration resume; see [Reinforcement jobs](reinforcement-jobs.md). Live native inference remains a separate integration. This evidence does not close those product or release gates.

The September 8 reinforcement review passed 17 native coordinator tests. Progress events now require matching request, run and job identities before updating counters; a bounded latest sample preserves valid progress arriving before the acknowledgement continuation. Regressions cover both mismatched progress identities and progress/terminal events arriving before acknowledgement. This remains a protocol fixture qualification, not an interactive UI or production-model test.
