# Fixed capture groups

`CaptureSource.bindings` freezes an ordered application-window or desktop-display set. Discovery offers application groups from current normal windows and a whole-desktop group from current displays. Groups contain one to sixteen leaves; window/display selections remain single leaves. Membership does not follow new windows, recycled windows, replaced processes or displays silently. New membership requires selecting the source again.

`ScreenCaptureGroup` owns one ScreenCaptureKit stream per leaf. A shared delivery queue preserves the native availability order without rewriting frame IDs, source timestamps, arrival timestamps, geometry revisions or pixels. Frames remain independent observations, not a stitched image or an invented simultaneous snapshot. Health events carry the source ID; Stop joins startup, all streams, their callbacks and the independent topology watcher. A geometry or membership change is an explicit unavailable boundary. The host owns the response to that boundary.

Surface descriptors optionally carry `nativeWindowID` or `nativeDisplayID`. Old files without either remain readable. Mapped frame fingerprints authenticate these optional identities using the `ASTRAN01` suffix, preserving the exact old fingerprint bytes when neither exists. `ControlScope.geometryRevision` is the immutable group epoch; individual source geometry revisions need not match it. Native absolute-pointer validation checks the requested surface's actual window recipient, so an overlapping sibling cannot receive a command intended for another window. Other pointer and keyboard actions remain restricted to the observed set.

Recording manifests persist `surfaceIDs` in capture-role order. Every original frame still has its own time and surface identity; the earliest frame remains the manifest's first observed time. Dataset assembly may begin later when every required role first exists. Physical input carries optional `surfaceID` only when recipient annotation or unique spatial ownership resolves it. Overlapping window inputs without a recipient stay unresolved. Adjacent displays use half-open bounds.

Explicit unchanged-source evidence is retained in the optional SQLite `coverage` table:

```sql
CREATE TABLE coverage (
  observed INTEGER NOT NULL, surface_id TEXT NOT NULL,
  frame_id TEXT NOT NULL, proof BLOB NOT NULL
);
```

`observed` is `verifiedAtNanos`; `proof` is Codable `CaptureFrameCoverage` JSON with stream/frame IDs, the original surface and source/availability times, `throughNanos`, `verifiedAtNanos`, and `kind: "unchanged"`. A proof commits only after its matching immutable frame is durable. It never duplicates or retimestamps that frame. Replay admits a proof only after its verification arrival and uses the proved source horizon for freshness. Silence and an idle status without timestamp/geometry provide no evidence. Recovery copies only proofs matching recovered frames and conservatively excludes visual evidence beyond durable input continuity.

Focused native verification: source-recipient disambiguation, half-open display ownership, changed membership rejection, independent source revisions, and delayed coverage publication pass in `MultipleSurfaceTests` and `RecordingCoverageTests`. This is synthetic contract evidence; live capture, display changes, permissions and installed control remain release qualification.

Dataset and actor integration use the same ordered roles and immutable source metadata; see [multiple-source data verification](../verification/multi-surface-data.md) for schema3, causal static-source coverage, routing, mixed-count batches and per-source mapped actor rings.
