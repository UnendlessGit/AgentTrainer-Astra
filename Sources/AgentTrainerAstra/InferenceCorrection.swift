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
    private struct Entry {
        let observation: PolicyActorOwnedObservation
        let pixels: Int, metadata: Int
    }
    private struct FrameHeader: Encodable { let metadata: FrameMetadata; let coverage: CaptureFrameCoverage? }
    private var observations: [Entry] = []
    private var bytes = 0, metadataBytes = 0
    func append(_ observation: PolicyActorOwnedObservation) {
        // Reserve the eventual bounded sidecar as well as pixels. Large original
        // queue histories must evict whole old observations, never be truncated.
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let inputBytes = try? encoder.encode(observation.actorInput).count,
              let frameBytes = try? observation.frames.reduce(0, { total, frame in
                  try total + encoder.encode(FrameHeader(metadata: frame.metadata, coverage: frame.coverage)).count + 512
              }) else { clear(); return }
        let metadata = inputBytes + frameBytes + 128
        lock.withLock {
            guard let cutoff = observation.actorInput.fields?["cutoffNanos"]?.uint64 else { return }
            let cost = observation.frames.reduce(0) { $0 + $1.pixels.count }
            guard cost <= CorrectionRecordingSeed.maximumBytes, metadata <= 12 * 1024 * 1024 else {
                observations.removeAll(); bytes = 0; metadataBytes = 0; return
            }
            observations.append(.init(observation: observation, pixels: cost, metadata: metadata)); bytes += cost; metadataBytes += metadata
            while observations.count > 256 || bytes > CorrectionRecordingSeed.maximumBytes || metadataBytes > 12 * 1024 * 1024 || observations.first.map({
                guard let first = $0.observation.actorInput.fields?["cutoffNanos"]?.uint64 else { return true }
                return cutoff < first || cutoff - first > CorrectionRecordingSeed.maximumDurationNanos
            }) == true {
                let retired = observations.removeFirst(); bytes -= retired.pixels; metadataBytes -= retired.metadata
            }
        }
    }
    func snapshot(through cutoff: UInt64) -> [CorrectionSourceObservation] {
        lock.withLock {
            observations.filter { ($0.observation.actorInput.fields?["cutoffNanos"]?.uint64 ?? .max) <= cutoff }.map { entry in
                .init(actorInput: entry.observation.actorInput, frames: entry.observation.frames.map {
                    .init(metadata: $0.metadata, pixels: $0.pixels, coverage: $0.coverage)
                })
            }
        }
    }
    func clear() { lock.withLock { observations.removeAll(); bytes = 0; metadataBytes = 0 } }
}
