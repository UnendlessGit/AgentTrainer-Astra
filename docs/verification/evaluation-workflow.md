# Saved evaluation and checkpoint comparison

The agent's Evaluation page retains single-checkpoint scoring and adds an explicit shared-dataset comparison. The operator chooses one saved behavioral dataset, one split, and up to 64 linked checkpoints. Each candidate uses that same immutable scoring protocol through sequential `evaluate.behavioral` jobs; no concurrent GPU jobs are introduced. Reinforcement checkpoints can be scored when their model/actions/timing match the chosen demonstration dataset.

`EvaluationProtocol` records the source run/checkpoint, the full dataset request, provenance, split, sequence length, verification mode and policy signature. Its fingerprint covers the recorded dataset manifest (including source and index hashes) or the saved generated-demonstration specification plus full model/action configuration. Native source resolution runs once and is checked again before each candidate. The compute worker independently validates dataset contents and checkpoint integrity. Its actual checkpoint ID, dataset ID and provenance must match before metrics are accepted.

`EvaluationDocument` starts as a durable running attempt in the SQLite library. A result can finalize once; metrics and provenance cannot be overwritten. Failed, stopped and interrupted attempts publish no partial metrics. Unavailable splits retain the worker's explanation and dataset identity without inventing zero decisions or loss. Startup interruption recovery runs under the existing exclusive library lease. Finished attempts also export `Jobs/<evaluation-id>/evaluation.json`; successful worker responses retain `results.json` and the exact `configuration.json` request.

The history table shows checkpoint, mean NLL, decisions, split, status and date. Multiple selected results produce an ordered comparison only when dataset fingerprint, actual dataset ID, provenance, policy signature, scoring protocol and decision count match. Otherwise the UI explains the mismatch. Policy signature includes model configuration, action vocabulary, canonicalizer version and timing. The view clarifies that a split from one shared dataset does not prove that another candidate never saw those demonstrations during training. NLL is demonstration likelihood, not closed-loop success.

## Focused evidence — September 23, 2026

- Eight targeted Core/coordinator checks pass (`.local/evaluation-workflow-tests.log`): durable finalization/recovery, comparison admission, an explicit shared dataset across checkpoints trained from different source configurations, incompatible-policy failures, unavailable results, and preserved single/copied-agent workflows.
- The native comparison view passes the existing owned-view renderer as `evaluation-comparison`: four base views and four scrolled views across light/dark mode at 1120×760 and 860×580 (`.local/evaluation-ui-renders`). The narrow light setup and wide dark history/detail images were visually inspected. This is generated fixture data and never reads the screen or posts input.
- Existing behavioral evaluator implementation is unchanged; these checks target orchestration and persistence, not new learning-quality claims.

Remaining product scope: task-success/closed-loop protocol history and broader evaluation suites. This workflow does not claim generalization, learning improvement, or final installed-app qualification.
