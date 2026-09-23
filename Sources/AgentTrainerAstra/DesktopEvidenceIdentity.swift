import Foundation
import AstraCore
import AstraPlatform

struct DesktopEvidenceIdentity: Codable, Equatable, Sendable {
    let runID: UUID
    let clockID: UUID
    let environmentID: UUID
    let actorSourceID: UUID
    let environmentSourceID: UUID
}

/// Shared by successive physical episodes and collector rotations. Policy
/// activation does not restart this environment producer's wire sequence.
final class DesktopEnvironmentSequence: @unchecked Sendable {
    let identity: DesktopEvidenceIdentity
    private let lock = NSLock()
    private var next: UInt64
    init(identity: DesktopEvidenceIdentity, nextSequence: UInt64 = 0) throws {
        guard identity.actorSourceID != identity.environmentSourceID, nextSequence < UInt64.max else {
            throw AstraError("desktop.evidenceIdentity", "Environment evidence requires distinct producers and an available sequence.")
        }
        self.identity = identity; self.next = nextSequence
    }
    var nextSequence: UInt64 { lock.withLock { next } }

    func receipt(_ receipt: JSONValue, packetID: UUID, episodeID: UUID, cause: String?, collector: CollectorSession) throws {
        try lock.withLock {
            var fields: [String: JSONValue] = ["receipt": receipt]
            if let cause { fields["cancellationCause"] = .string(cause) }
            try emit("environment.receipt", payload: fields, packetID: packetID, episodeID: episodeID, collector: collector)
        }
    }
    func interval(_ evaluation: RewardEvaluation, packetID: UUID, collector: CollectorSession, deferredProgram: RewardProgram? = nil) throws {
        try lock.withLock {
            guard next <= UInt64.max - 3 else { throw AstraError("desktop.evidenceSequence", "The environment evidence sequence is exhausted.") }
            var reward: [String: JSONValue] = ["startNanos": .unsigned(evaluation.startNanos),
                "endNanos": .unsigned(evaluation.endNanos), "value": evaluation.value.map(JSONValue.number) ?? .null]
            if let program = deferredProgram {
                let deferred = program.rules.filter { $0.kind == .manualMarker }.map(\.id)
                guard deferred.allSatisfy({ evaluation.components[$0] == nil }) else {
                    throw AstraError("feedback.liveAndDeferred", "A reward interval cannot mix live and retrospective manual feedback.")
                }
                let automatic = program.rules.filter { $0.kind != .manualMarker }
                let known = evaluation.unknownRules.allSatisfy(Set(deferred).contains)
                reward["value"] = known ? .number(automatic.reduce(0) { $0 + (evaluation.components[$1.id] ?? 0) }) : .null
                reward["automaticComponents"] = .array(automatic.map { .object([
                    "ruleID": .string($0.id.uuidString.lowercased()), "value": evaluation.components[$0.id].map(JSONValue.number) ?? .null]) })
                reward["deferredManualRuleIDs"] = .array(deferred.map { .string($0.uuidString.lowercased()) })
            }
            try emit("environment.reward", payload: reward,
                packetID: packetID, episodeID: evaluation.episodeID, collector: collector)
            try emit("environment.outcome", payload: ["endNanos": .unsigned(evaluation.endNanos),
                "outcome": .string(evaluation.outcome.rawValue)], packetID: packetID, episodeID: evaluation.episodeID, collector: collector)
            try emit("environment.watermark", payload: ["throughNanos": .unsigned(evaluation.endNanos),
                "throughSequence": .unsigned(next - 1), "complete": .bool(true)],
                packetID: nil, episodeID: evaluation.episodeID, collector: collector)
        }
    }
    private func emit(_ kind: String, payload: [String: JSONValue], packetID: UUID?, episodeID: UUID, collector: CollectorSession) throws {
        guard collector.runID == identity.runID, next < UInt64.max else {
            throw AstraError("desktop.evidenceSequence", "The evidence destination or producer sequence is invalid.")
        }
        var fields = payload
        fields["clockID"] = .string(identity.clockID.uuidString.lowercased())
        fields["episodeID"] = .string(episodeID.uuidString.lowercased())
        let message = WireMessage(kind: kind, sequence: next, requestID: packetID, runID: identity.runID, payload: .object(fields))
        try collector.offer(.request("collector.evidence", .object([
            "sourceID": .string(identity.environmentSourceID.uuidString.lowercased()), "message": try .encode(message)])))
        next += 1
    }
}

struct DesktopManualSeal: Sendable {
    let observationID: UUID
    let episodeID: UUID
    let cutoffNanos: UInt64
    let readings: [SignalReading]
    let markers: [ManualRewardMarker]
    let coverage: ManualRewardCoverage?
}
struct DesktopEpisodeFault: Sendable {
    let generationID: UUID
    let episodeID: UUID
    let error: AstraError
}
struct DesktopTerminalEvidence: Sendable {
    let generationID: UUID
    let episodeID: UUID
    let cutoffNanos: UInt64
    let outcome: RewardOutcome
}
enum DesktopEpisodeStop: Codable, Sendable {
    case semanticBoundary(UInt64)
    case operatorAbort
    case failure(String)
}
struct DesktopEpisodeJoin: Codable, Sendable {
    let runID: UUID
    let episodeID: UUID
    let generationID: UUID
    let actorJoined: Bool
    let controlJoined: Bool
    let manualProducerJoined: Bool
    let predictionResolved: Bool
    let cleanupConfirmed: Bool
    let stoppedNanos: UInt64
    let lastProducedSequence: UInt64?
    let stop: DesktopEpisodeStop
}
enum DesktopEpisodeCompletion: Sendable {
    case empty
    case ended(producedDecisions: Int, learningDecisions: Int, lastProducedSequence: UInt64,
               terminal: DesktopTerminalEvidence?, auditOnly: Bool)
}
