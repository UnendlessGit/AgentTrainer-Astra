import Foundation
import AstraCore
import AstraPlatform

/// An owned copy of exactly what the actor observed. The actor's single-consumer
/// ring references are deliberately excluded: collectors publish their own
/// leases or persist these bytes before releasing their queue reservation.
struct InferenceCollectedObservation: Sendable {
    let runID: UUID
    let actorInput: JSONValue
    let frames: [PolicyActorOwnedObservation.Frame]
    var pixelByteCount: Int { frames.reduce(0) { $0 + $1.pixels.count } }
    init(runID: UUID, actorInput: JSONValue, frames: [PolicyActorOwnedObservation.Frame]) {
        self.runID = runID; self.actorInput = actorInput; self.frames = frames
    }
    init(runID: UUID, actorInput: JSONValue, frame: FrameMetadata, pixels: Data, coverage: CaptureFrameCoverage? = nil) {
        self.init(runID: runID, actorInput: actorInput, frames: [.init(metadata: frame, pixels: pixels, coverage: coverage)])
    }
    // Compatibility for explicitly single-source callers; never discard a source.
    private var single: PolicyActorOwnedObservation.Frame { precondition(frames.count == 1); return frames[0] }
    var frame: FrameMetadata { single.metadata }
    var pixels: Data { single.pixels }
    var coverage: CaptureFrameCoverage? { single.coverage }
}

enum InferenceCollectionEvent: Sendable {
    case prepared(runID: UUID, actor: JSONValue)
    case observation(InferenceCollectedObservation)
    case decision(JSONValue)
    /// Raw, transport-validated control evidence. The collector validates it
    /// against its original packet before treating it as execution proof.
    case control(WireMessage)
}

struct InferenceCollectionSink: Sendable {
    /// Must make a bounded nonblocking queue admission, never await disk, GPU or
    /// reward work. Throwing stops control; no source observation is dropped.
    let offer: @Sendable (InferenceCollectionEvent) throws -> Void
    /// Called after capture/actor/control ownership has joined and every control
    /// event has been offered. A collector can now finalize its own queued work.
    let finish: @Sendable (JSONValue) async throws -> Void
}

/// I/O offers can fail during shutdown after the UI has already requested stop.
/// Latch that failure synchronously so joined process exit cannot race its UI
/// delivery and accidentally publish a successful collection summary.
final class InferenceCollectionFault: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: String?
    var message: String? { lock.withLock { stored } }
    func record(_ error: any Error) { lock.withLock { if stored == nil { stored = error.localizedDescription } } }
}
