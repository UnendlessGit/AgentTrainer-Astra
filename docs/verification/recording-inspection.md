# Recording inspection

The native recording reader takes a shared package lease, opens the sealed index read-only, checks manifest identity/counts, selects the latest frame available at the requested time, and verifies its archive checksum and metadata before returning pixels. Input rows retain separate source and observed timestamps. Nearby input lookup is bounded to 256 displayed events with an explicit overflow indicator.

The recording inspector displays the decoded native BGRA frame, a time scrubber, capture-interval navigation and nearby controls with provenance. Scrubbing coalesces requests while allowing at most one disk decode in flight. Closing the inspector cancels UI publication and releases its reader after the in-flight decode finishes. No recorded control event is executed by inspection.

A native fixture test verifies active-writer exclusion, causal frame selection, byte-exact pixels, late source input preservation, and corruption rejection. The real SwiftUI inspector is included in both-theme, normal/minimum-size render qualification. This does not yet cover video export, editing source selections, context annotation, correction pre-roll, or long-session performance; those remain product gates.
