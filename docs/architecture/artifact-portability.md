# Artifact locations and portable archives

The catalog, jobs, datasets, rewards and experience journals remain in the workspace. Recordings and models have independently selectable roots in `ArtifactStorageLayout`. Each coordinator captures the immutable layout for its lifetime; switching locations requires joined idle workflows and reopening the store/coordinators while retaining the same library lease. Runtime paths use recording/checkpoint UUIDs. Saved dataset configurations are resolved by their original dataset UUID and current recording root, without rewriting immutable configuration, recording, dataset or checkpoint bytes.

Each root has a small identity marker bound to its artifact kind. An unavailable or replaced external volume produces a storage issue. The application must not recreate missing external storage, change to a different source, or fall back to a default folder silently. The catalog remains readable so the operator can reconnect storage and reload routing.

## Copy, verify, then switch

Migration previews identify the exact source/destination, ordered file inventory, byte sizes and SHA-256 hashes. A durable catalog journal records the approved plan. Copying uses bounded 4 MiB buffers, no executable serialization, exclusive temporary files and cancellation points. Input files, staged output and directory entries are synchronized and verified before an exclusive rename publishes the destination. Only then does a catalog transaction switch routing.

Original storage is retained. A cancelled or interrupted copy cannot delete source data. Retry uses the same journal identity and verifies existing staged files. Only regular, singly linked `.astra-copy-<UUID>` partial files not present in the expected inventory can be removed from the exact journal-owned staging directory. Unexpected files, changed hashes or links stop the transfer. A published destination can be verified and adopted after a crash before the routing transaction. Reopening does not automatically resume a copy.

These operations support local directories on connected macOS volumes. There is no cloud synchronization or background eviction. The old copy is deliberately retained after switching; reclaiming that copy is an explicit operator action, not part of migration.

## `.astraarchive` packages

An Astra archive is a directory package with bounded `archive.json` metadata and `payload/<kind>/<identity>/` items. Types are recording, checkpoint, dataset, run configuration, desktop configuration and reward image. Every regular payload file has a relative path, byte count and SHA-256 digest. Paths cannot escape the package; symlinks, special files, duplicate/case-colliding paths and resource-limit violations are rejected. Export is staged, fully verified and published without overwriting an existing destination.

Export also has a durable retry journal. The approved header hash is stable after decoding a recovery plan: UUID-keyed selection/context maps have a canonical order and catalog-only dates use millisecond precision. A retry preserves the same archive identity, verifies any existing header and payload, and refuses unexpected bytes. A previously published destination completes idempotently only if its complete inventory and archive header hash match. Raw recording, model, dataset and run-configuration files are copied without normalization.

The preview includes the actual dependency closure:

- A model retains its exact tensor, optimizer, pending-gradient, sampler and RNG bytes.
- Trained models include their original job configuration and catalog run metadata.
- Recorded BC includes its frozen dataset and all referenced recording packages.
- Corrections include the referenced source policy, so the failed behavior can be reproduced.
- Desktop checkpoints include their original desktop task configuration, immutable reward definition and content-addressed reward images.
- Context vocabulary, source-agent metadata, owning links and per-agent recording selections are preserved.

Export refuses an incomplete required resume dependency instead of presenting it as a portable resumable checkpoint. Parent model weights are not needed when a complete checkpoint already owns its full model and optimizer state. New initial policies have no optimizer state to resume. Historical training logs, unfinished feedback, raw desktop rollouts, claim files and inference audit logs are excluded explicitly in the preview; those workflows remain in the original workspace. Independent model/recording root migration leaves their experience/claim paths unchanged.

Import first verifies the full package and dependency identities. An existing artifact UUID is reused only when its bytes match; it is never overwritten or reassigned a new UUID to bypass a collision. Existing agent/checkpoint presentation is retained. Incompatible immutable catalog metadata or saved recording selections block import. Linking to an additional agent preserves the creator identity and the original agents' selections; the additional agent chooses its own training selections.

All new immutable files publish before one transaction publishes catalog ownership. An interrupted import can leave verified unlinked copies, tracked by its durable journal; retry reuses them and finishes the transaction. It never removes pre-existing files. Importing does not grant macOS permissions, execute recorded input or start a model.

## Evidence

`ArtifactTransferTests` uses generated pixels and temporary storage to exercise archive export/import/dedup, independent root switching/reopening, cancellation recovery and crashed temporary-file recovery. A second focused handoff uses a genuine small paused BC model, native AstraFixture recording, dataset and original run configuration prepared by `scripts/prepare_artifact_resume_fixture.py`. Native archive round-trip and re-export preserve every artifact hash. The Python worker resumed its pending gradient chunk to four decisions and two optimizer updates: counters/RNG matched the untouched-source continuation exactly, and maximum floating-point difference was 1.49e-8. This verifies format and state continuity, not production learning quality or physical-volume disconnect qualification.
