# Recording storage integrity

This review covers `RecordingWriter`, `RecordingManifest`, catalog recovery, and SQLite publication. It does not qualify native capture, event-tap coverage, live scope selection, replay, datasets, or the full recording workflow.

## Implemented guarantees

- Invalid manifests are rejected before creating a writer. Names, environment settings, schema version, clocks, counts, lifecycle state, and recovery paths are validated. Symbolic links and escaping index paths are rejected.
- Input batches either pass validation entirely or leave counters and sequence state unchanged. Source timestamps and native event payloads survive storage unchanged. Duplicate frame identities and decreasing frame observation times are rejected.
- Frame bytes are synchronized before index references commit. A terminal manifest is published after the archive and standalone SQLite index close. Finalization failures close admission, preserve source data, and throw again on repeated finish calls instead of returning a false success.
- Recovery uses a per-package exclusive advisory lock and skips live writers. It reconstructs a separate index from checksummed frame prefixes and committed raw events, restores orphan catalog entries and agent links, and leaves original shards/index/WAL unchanged. The atomic manifest pointer selects the new index. Recovery is idempotent once published; snapshot does not scan recording files.
- Backup writes use an exclusively created sibling staging file and a non-replacing hard link for publication. Failure cleanup only removes owned staging paths. Competing destinations and existing unrelated files remain intact. Backups and sealed recording indexes use rollback-journal mode for standalone read-only use. Busy checkpoints and operations after explicit close report errors.

## Verification

On the M3 Max development Mac, `swift test` passed all 34 integrated Swift tests on 2026-09-06. Relevant fixtures cover:

- exact raw frame/input round-trip with late source timestamps;
- invalid batch rollback, invalid lifecycle state, traversal and symbolic-link rejection;
- a blocked manifest destination during sealing, repeated failure, rejected later writes, and recovery of the preserved prefix;
- a copied crash instant containing committed WAL input, a complete but unindexed frame, and a truncated subsequent frame block; recovery indexes both complete frames, restores agent links, retains raw file bytes exactly, and bounds supervision at the last durable input event;
- recovery refusing a live writer and preserving the same recovery pointer on a repeated pass;
- concurrent backups to one destination, preexisting-file preservation, missing-destination failure, active-reader checkpoint failure, and explicit database close.

The read-only backup fixture initially failed when a backup inherited a WAL header associated with its temporary filename. Converting the completed backup to standalone rollback-journal mode fixed the failure; the fixture remains enabled.

## Remaining gates

- The package lock protects recording writers and recovery. A process-exclusive library lease still needs integration before assuming one coordinator across multiple app instances.
- Recovery conservatively excludes frames after the last durable input event. Long silent intervals may therefore be excluded unnecessarily. An explicit producer coverage watermark is needed before preserving that tail as expert supervision.
- Recovery preserves valid prefixes; it cannot reconstruct undelivered input, overwritten storage, or data whose identity/environment metadata is unavailable from both manifest and catalog. Corrupt source records and unknown versions remain visible as issues.
- Sealed-package startup reconciliation does not rehash every frame or verify every SQLite page. Dataset/replay readers still must perform integrity checks before using source data. Unpublished recovery folders left by process termination are retained; normal thrown failures clean up their derived work without deleting source artifacts.
- Real process-kill/power-loss campaigns, full-volume and device-disconnection faults, prolonged high-rate recording, external-library relocation, and installed-app capture/control workflows remain release gates. The WAL crash fixture reproduces an on-disk crash state; it is not a power-failure test.
