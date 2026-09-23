import Foundation
import AstraCore
import AstraPlatform

struct InferenceCorrection: Sendable {
    let agentID: UUID
    let checkpoint: CheckpointDocument
    let source: CaptureSource
    let seed: CorrectionRecordingSeed
    let contextValues: [UUID: UUID]
}

/// Holds the same immutable CPU copies already admitted by the actor; eviction
/// removes complete observation groups, never individual surfaces. Nothing is
/// persisted until the operator explicitly requests a correction recording.
final class InferenceCorrectionBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var observations: [PolicyActorOwnedObservation] = []
    private var bytes = 0
    func append(_ observation: PolicyActorOwnedObservation) {
        lock.withLock {
            guard let cutoff = observation.actorInput.fields?["cutoffNanos"]?.uint64 else { return }
            let cost = observation.frames.reduce(0) { $0 + $1.pixels.count }
            guard cost <= CorrectionRecordingSeed.maximumBytes else { observations.removeAll(); bytes = 0; return }
            observations.append(observation); bytes += cost
            while observations.count > 256 || bytes > CorrectionRecordingSeed.maximumBytes || observations.first.map({
                guard let first = $0.actorInput.fields?["cutoffNanos"]?.uint64 else { return true }
                return cutoff < first || cutoff - first > CorrectionRecordingSeed.maximumDurationNanos
            }) == true {
                bytes -= observations.removeFirst().frames.reduce(0) { $0 + $1.pixels.count }
            }
        }
    }
    func snapshot(through cutoff: UInt64) -> [CorrectionSourceObservation] {
        lock.withLock {
            observations.filter { ($0.actorInput.fields?["cutoffNanos"]?.uint64 ?? .max) <= cutoff }.map { observation in
                .init(actorInput: observation.actorInput, frames: observation.frames.map {
                    .init(metadata: $0.metadata, pixels: $0.pixels, coverage: $0.coverage)
                })
            }
        }
    }
    func clear() { lock.withLock { observations.removeAll(); bytes = 0 } }
}
