# Original AgentTrainer audit and reconciliation

Independent baseline commit: `505ec37`. Original inspected HEAD: `86387f1a143eaad84c2b24d47cae095f2677da31` (2.4.5).

The baseline was committed and pushed before any original internals were inspected. All 57 tracked original files were inventoried; original source/documentation/test text was read completely across the coverage owners below. Binary icons were inventoried and observed in the release UI. Historical release notes, commit messages and tags were reviewed, plus removed recording-interchange, Windows companion, training-audit and update documents. This does not claim to have reread every source revision in Git history or run historical release tests.

| Coverage | Report |
|---|---|
| Model, preprocessing, datasets, BC, runtime, DAgger, memory, visualization, performance tests | [Learning audit](audit/learning.md) |
| RL configuration, policy/update mathematics, rollouts, rewards, lifecycle, checkpoints and RL tests | [RL audit](audit/reinforcement-learning.md) |
| Native app, input/capture, persistence, all remaining UI, packaging and updater | [Platform/product audit](audit/platform-product.md) |
| Complete domain, current/historical docs, release history, DomainTests 1–4000 | [Domain/product audit](audit/domain-product.md) |
| DomainTests 4001–5657, including numerical/learning/correction fixtures | Additional section in [RL audit](audit/reinforcement-learning.md) |
| Read-only original release UI observation | [UI observation](audit/ui-observation.md) |

Original integrity was verified after the audit: SHA-256 of all 57 tracked files matched the pre-read inventory, HEAD was unchanged, and Git remained clean. The inventory is local-only; no original source or private data is copied into Astra's Git history. The runtime UI copy was isolated from the original source tree and was not used for learning/capture.

## Integrated result

The evidence supports the independent architecture. The original loses within-tick input order, restarts a short recurrent history window, backdates some initial/control evidence, and pauses input during PPO updates without proving the world paused. Static review also found inconsistent probability formulas, incompatible production batching branches, incomplete checkpoint fallback, and data-deletion-on-finalization-error risks. Reports identify exact source locations and distinguish static findings from reproduced failures.

Preserve the product lessons through Astra's own interfaces: context conditioning, reusable configuration, explicit corrections with causal pre-roll, portable native data, safe relocation, protected/manual checkpoints, optional external rewards, input reconciliation/shortcut handling, diagnostics, and faithful operation state. The [follow-up decision](architecture/0002-product-reconciliation.md) records these additions without rewriting the independent baseline.

Historical test counts and tiny synthetic throughput runs do not satisfy Astra's release gates. Real production-entry training, persistent-memory learning, exact action scoring, crash recovery, and installed local workflows remain required.
