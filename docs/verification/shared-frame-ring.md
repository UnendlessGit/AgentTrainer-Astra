# Shared-frame ring: contract and verification

The native `SharedFrameRing` and Python `FrameRingReader` provide bounded mapped transport for compact raw BGRA observations. This component is ready for coordinator integration; it does not yet establish live native inference, capture-to-policy latency, or packaged-worker use of the ring.

## Ownership and integration

1. Create a fresh `SharedFrameRing(url:runID:slotCount:slotCapacity:)` for one run and one consumer. The caller chooses a private temporary directory and accounts for the ring in the run's aggregate memory/storage budget. Defaults are four slots; limits are 16 slots, 256 MiB per slot and 1 GiB total file size. Invalid configurations fail before file creation.
2. Creation uses an exclusive, no-follow, close-on-exec file descriptor and mode `0600`. The inode must be a regular file owned by this user, with one hard link. Before sizing or mapping, macOS `F_PREALLOCATE` with `F_ALLOCATEALL` and `F_PEOFPOSMODE` must reserve the complete file. Errors, unsupported filesystems and incomplete reservations fail closed; there is no sparse fallback. The ring is temporary IPC storage, not a durable archive.
3. Send the path, `runID` and `ringID` over the inherited control channel and open `FrameRingReader(path, run_id=..., ring_id=...)`. Python opens read-only, refuses symlinks and non-private/non-regular files, validates identities/layout/bounds before mapping, and retains that opened inode. No payload bytes belong in JSON messages.
4. Native `publish(pixels:metadata:)`, or its synchronous `compactPixels` buffer variant, copies exactly `pixelWidth * pixelHeight * 4` bytes into an available slot. The caller must remove native row padding. Metadata is validated and normalized to codec `raw`, pixel format `bgra8-srgb`. The native method serializes publishers and returns `SharedFrameReference` only after the slot payload/header and publication fences are complete. Deliver that reference after return.
5. The consumer calls `copy_frame(reference)`. It validates run/ring/slot/lease/sequence, frame size/offset and a metadata fingerprint. It compares the complete ring and slot headers before and after ingestion. A temporary read-only NumPy view feeds one `mx.array` ingestion copy; `mx.eval` finishes that copy before an `OwnedFrame` is returned. The NumPy view never escapes. `pixels` has MLX `uint8` shape `[height, width, 4]`; channel order is BGRA. Preprocessing must explicitly apply the policy's color conversion and geometry.
6. Send `OwnedFrame.acknowledgement` to native `release(_:)`. This acknowledgement contains the exact validated identity, independent of later changes to the input dictionary or returned metadata. Once it is available, the owned MLX frame survives slot reuse, reader close and writer unlink. Failed copies return no acknowledgement. Each reader rejects a repeated/older publication for a previously read slot. Retrying delivery of an existing acknowledgement does not require copying the frame again.
7. Native release accepts only the exact currently pending acknowledgement. Wrong versions, run/ring IDs, slot, lease UUID or sequence fail without reclaiming a slot. A repeated acknowledgement is an error. The coordinator must decide how to handle a full ring (`frameRing.full`), for example by applying bounded backpressure or explicitly rejecting that observation with diagnostics. It must never overwrite an outstanding lease or silently treat a stale observation as current.
8. Normal shutdown drains acknowledgements, closes the reader, then calls native `close()`. `close()` refuses pending leases. If the consumer fails, cancel and join its actual process before `closeAfterConsumerExit()`. A timeout, closed control pipe or reader-close intent alone is not proof of process exit. Forced cleanup retires the entire inode and disables further publications; it never recycles its pending slots. Cleanup unmaps/closes and unlinks only the original inode at the original path, without truncating or changing retained slot bytes. A new run needs a new ring identity. The destructor also retires the inode rather than reusing slots, so surviving external mappings cannot reference repurposed data.

These rules rely on cooperating Astra processes and pipe publication/acknowledgement ordering. The metadata digest binds reference data to its slot; it is not a payload checksum or protection against a malicious same-user process rewriting/truncating an open inode. The reader checks current file dimensions before touching mapped pages, but cannot make hostile concurrent truncation safe. Durable recordings use `FrameArchive` instead.

## Version 1 wire layout

All fixed-width integers are unsigned little-endian. UUIDs occupy their 16 RFC-order bytes, not the bytes of their text representation. Each ring header is 128 bytes. Slot stride is `alignUp(128 + capacity, 64)` and file size is `128 + slotCount * stride`. All reserved bytes must be zero.

| Ring-header offset | Bytes | Field |
|---|---:|---|
| 0 | 8 | ASCII `ASTRAR01` |
| 8 | 4 | Version `1` |
| 12 | 4 | Header length `128` |
| 16 | 16 | Ring UUID |
| 32 | 16 | Run UUID |
| 48 | 4 | Slot count |
| 52 | 4 | Slot-header length `128` |
| 56 | 8 | Payload capacity per slot |
| 64 | 8 | Aligned slot stride |
| 72 | 8 | Complete file byte count |
| 80 | 48 | Reserved |

Slot `s` starts at `128 + s * stride`; its payload starts 128 bytes later.

| Slot-header offset | Bytes | Field |
|---|---:|---|
| 0 | 4 | State: `0` available, `1` writing, `2` published |
| 4 | 4 | Version `1` |
| 8 | 4 | Slot index |
| 12 | 4 | Reserved |
| 16 | 16 | Lease UUID, fresh on each publication |
| 32 | 16 | Frame UUID |
| 48 | 8 | Monotonic ring publication sequence, beginning at `1` |
| 56 | 8 | Used payload bytes |
| 64 | 32 | SHA-256 metadata fingerprint |
| 96 | 32 | Reserved |

`SharedFrameReference` JSON has exactly `version`, `runID`, `ringID`, `slot`, `leaseID`, `sequence`, `offset`, `size`, `metadata`. The acknowledgement has exactly the first six identity fields through `sequence`, omitting the frame offset, size and metadata. Timestamps and sequences remain integers, including values above signed 64-bit range; do not pass them through binary64 JSON-number intermediaries.

The fingerprint hashes this concatenation:

- ASCII `ASTRAM01`; frame UUID; `eventNanos` and `observedNanos` as uint64.
- Surface-ID UTF-8 byte length as uint32, followed by its bytes.
- Global rectangle `x, y, width, height` as little-endian IEEE-754 binary64; pixel width and height as uint32.
- Content rectangle in the same binary64 order; geometry revision and frame byte count as uint64.
- Literal bytes `bgra8-srgb\0raw\0`, containing NUL separators.

Both languages normalize signed zero to positive zero before hashing geometry, since JSON encoders may normalize it. The same frame ID, timestamps, logical/pixel/content geometry, geometry revision and payload dimensions must match in the pipe reference and mapped header. Full-frame bytes remain compact native-resolution BGRA; the ring itself does not resize, compress or normalize observations.

## Evidence

On 2026-09-08, macOS 27.0 build 26A5425a, Apple Silicon, Swift 6.3.3, CPython 3.12.13 and MLX 0.32.2:

```sh
swift test -Xswiftc -warnings-as-errors --filter 'frameRing|concurrentFramePublishers'
.venv/bin/python -W error -m pytest python/tests/test_frame_ring.py -q
```

Five native tests and six Python tests passed. Native tests cover full-ring refusal, exact/stale/repeated lease checks, live-lease close refusal, concurrent publication, invalid dimensions, exclusive/symlink-safe creation, private permissions, actual allocated storage before publication, and retirement preserving an old inode while a new ring occupies the same path. Cleanup preserves unrelated replacement files.

The Python suite builds and launches the real `AstraFixture --frame-ring` producer over inherited pipes. It checks byte-exact MLX ingestion of two different frames through the same slot, unchanged first-frame pixels after acknowledgement/reuse/reader close/writer unlink, Unicode identity, fractional/negative geometry, signed zero, and timestamps above `Int64.max`. It also rejects stale/wrong identities and ranges, modified metadata, publication-header changes during ingestion, unsafe file types/permissions, and a preexisting truncation before touching missing mapped pages. A request-mutation regression verifies that an in-flight dictionary change cannot produce an acknowledgement for another lease.

Tests allocate only small kilobyte rings and copy 140-byte/256-byte fixture frames. They use no TCC permissions, screen content or input posting. They do not fill a filesystem or claim observed disk-full recovery; the preallocation failure path is fail-closed code, while actual successful reservation is checked through allocated blocks. Remaining qualification belongs to the live coordinator: capture integration, authenticated worker routing, consumer-crash/process-join cleanup, bounded aggregate memory, packaged/offline transport and representative end-to-end inference latency.
