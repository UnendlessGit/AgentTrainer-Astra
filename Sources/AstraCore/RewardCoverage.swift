import Foundation

public enum RewardCoverageKind: String, Codable, Sendable { case frame, unchanged }

/// Only the owner of these source pixels may attest unchanged coverage. A
/// wall-clock poll or silence is not such evidence. Original source timestamps
/// and observation identity remain intact when coverage advances.
public struct RewardSourceCoverage: Codable, Hashable, Sendable {
    public let sourceObservationID: UUID
    public let surface: SurfaceDescriptor
    public let eventNanos: UInt64
    public let observedNanos: UInt64
    public let throughNanos: UInt64
    public let verifiedAtNanos: UInt64
    public let kind: RewardCoverageKind
    public init(sourceObservationID: UUID, surface: SurfaceDescriptor, eventNanos: UInt64, observedNanos: UInt64,
                throughNanos: UInt64, verifiedAtNanos: UInt64, kind: RewardCoverageKind) {
        self.sourceObservationID = sourceObservationID; self.surface = surface; self.eventNanos = eventNanos
        self.observedNanos = observedNanos; self.throughNanos = throughNanos; self.verifiedAtNanos = verifiedAtNanos; self.kind = kind
    }
    public func validated(scope: ControlScope, cutoffNanos: UInt64) throws -> Self {
        _ = try surface.validated()
        guard scope.surfaces.contains(surface), eventNanos <= observedNanos, eventNanos <= throughNanos,
              max(observedNanos, throughNanos) <= verifiedAtNanos, verifiedAtNanos <= cutoffNanos,
              kind == .unchanged || throughNanos == eventNanos else {
            throw AstraError("reward.coverage", "Observation coverage has a foreign surface, changed geometry or invalid source/evidence time.")
        }
        return self
    }
}

public struct RewardReadingCoverage: Codable, Hashable, Sendable {
    public let signalID: UUID
    public let source: RewardSourceCoverage
    public init(signalID: UUID, source: RewardSourceCoverage) { self.signalID = signalID; self.source = source }
}

/// Produced only by RewardEvaluator after source, signal, age and confidence
/// validation. Readiness and baseline reset consume this same immutable value.
public struct ResolvedRewardSnapshot: Sendable {
    let definition: RewardProgram
    public var programID: UUID { definition.id }
    public let episodeID: UUID
    public let cutoffNanos: UInt64
    public let readings: [SignalReading]
    public let coverage: [RewardReadingCoverage]?
    public let values: [UUID: SignalValue]
}
