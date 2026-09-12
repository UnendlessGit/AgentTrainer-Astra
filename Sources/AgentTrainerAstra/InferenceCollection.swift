import Foundation
import AstraCore
import AstraPlatform

/// An owned copy of exactly what the actor observed. The actor's single-consumer
/// ring references are deliberately excluded: collectors publish their own
/// leases or persist these bytes before releasing their queue reservation.
struct InferenceCollectedObservation: Sendable {
    let runID: UUID
    let actorInput: JSONValue
    let frame: FrameMetadata
    let pixels: Data
    var coverage: CaptureFrameCoverage? = nil
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
