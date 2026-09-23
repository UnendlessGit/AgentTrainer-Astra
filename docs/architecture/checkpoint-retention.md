# Checkpoint presentation and explicit retention

Checkpoint names and optional `pinned` state live in the catalog; old documents without `pinned` decode as unpinned. Updating a name pins the checkpoint automatically. The operator can then explicitly unpin it without renaming. These changes are shared across linked agents and never rewrite immutable checkpoint manifests, policy tensors, hashes or optimizer state. Idempotent checkpoint publication preserves the catalog name, pin and original creation date.

Manage Checkpoints offers a concrete preview for one agent: keep the newest zero to one hundred unpinned checkpoints and show at most 256 removals per action. The preview identifies link-only removals, remaining owning agents, local model folders to delete and measured bytes. Cleanup occurs only after the operator selects Remove in this preview; there is no scheduled or implicit sweep.

Protection includes:

- Pinned checkpoints and every agent's selected checkpoint, including archived agents.
- Unfinished saved feedback, including failed/reviewable batches.
- Active learning inputs/outputs, active evaluation candidates and their source protocols, and runtime IDs supplied by the caller.
- The stopped inference policy while its correction history remains retained in memory; this temporary reference ends on discard, a new run or quitting.
- Source policies referenced by correction recordings. Their bounded, hash-bound prelude metadata is read without source pixels; unavailable provenance blocks removal.
- The requested newest-checkpoint allowance.

Completed ordinary learning/evaluation history remains historical metadata; it does not require retaining every former model's weights. Scores and provenance survive removal, while unavailable model-selection actions are labelled accordingly. Source recordings, datasets, inference audit history and other checkpoints are never swept by this operation.

All owning links are explicit. A one-time migration materializes creator ownership from early catalogs; reopening never reconstructs an explicitly removed creator link. Removing a shared checkpoint only unlinks this agent. The model folder is eligible for deletion only when the last owning link is removed and no protected reference exists.

The library actor re-evaluates the entire preview immediately before mutation. A changed pin, selection, name, reference, link or measured size invalidates the old preview. The UI blocks new recording/learning/control work and catalog selection during application. The caller must hold the exclusive library lease.

Catalog unlink/archive and a deletion journal commit together before filesystem deletion. An interrupted or failed filesystem operation leaves a durable pending entry tied to the requesting agent, surfaced as a catalog issue and explicit retry candidate. Reopening does not delete anything automatically. Historical checkpoint metadata remains archived; stale publications cannot resurrect a deleted model. Paths are derived only from checkpoint UUIDs under the regular local Models directory, and linked/nonregular files are rejected.

Focused verification covers rename auto-pinning and identity preservation, stale-preview refusal, pins/selections/active learning/unfinished feedback/correction-source protection, shared ownership, final-owner deletion, reopening without restored links, and legacy pin decoding. Native build and the three relevant checks pass in `.local/checkpoint-retention-tests.log`; this is implementation evidence, not release qualification.
