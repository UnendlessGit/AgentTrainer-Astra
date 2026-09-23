import CryptoKit
import Foundation

public struct ClosedLoopTrial: Codable, Hashable, Sendable {
    public var seed: Int
    public var pixelWidth: Int
    public var pixelHeight: Int
    public var logicalBounds: [Int]
    public init(seed: Int, pixelWidth: Int = 1280, pixelHeight: Int = 720, logicalBounds: [Int] = [0, 0, 1280, 720]) {
        self.seed = seed; self.pixelWidth = pixelWidth; self.pixelHeight = pixelHeight; self.logicalBounds = logicalBounds
    }
    func validated() throws {
        guard (0...1_000_000_000).contains(seed), (32...8192).contains(pixelWidth), (32...8192).contains(pixelHeight),
              pixelWidth * pixelHeight * 4 <= 64 * 1024 * 1024, logicalBounds.count == 4,
              logicalBounds.prefix(2).allSatisfy({ (-1_000_000...1_000_000).contains($0) }),
              logicalBounds.suffix(2).allSatisfy({ (1...1_000_000).contains($0) }) else {
            throw AstraError("evaluation.layout", "Choose a bounded practice layout and evaluation seed.")
        }
    }
}

/// A task protocol can compare different model architectures, but preserves
/// environment timing, action semantics and ordered categorical meanings.
public struct ClosedLoopProtocol: Codable, Equatable, Sendable {
    public var schemaVersion = 1
    public var task = "pointing"
    public var periodMS = 100
    public var leadMS = 100
    public var delayMS = 2000
    public var cueMS = 500
    public var timeLimitMS = 10_000
    public var deterministic = true
    public var policySeed = 900_000
    public var trials: [ClosedLoopTrial]
    public var contextSizes: [Int] = []
    public var contextVocabulary: ContextVocabulary?
    public var contextIDs: [Int] = []
    public init(trials: [ClosedLoopTrial]) { self.trials = trials }
    public func validated() throws -> Self {
        guard schemaVersion == 1, ["pointing", "delayed_memory"].contains(task), (1...1000).contains(periodMS),
              (0...2000).contains(leadMS), (1...120_000).contains(delayMS), (1...120_000).contains(cueMS),
              (1...600_000).contains(timeLimitMS), task != "delayed_memory" || timeLimitMS > delayMS + cueMS,
              (0...1_000_000_000).contains(policySeed), (1...256).contains(trials.count), Set(trials).count == trials.count,
              trials.count * ((timeLimitMS + periodMS - 1) / periodMS) <= 200_000,
              contextSizes.count <= 32, contextSizes.allSatisfy({ (1...65_536).contains($0) }),
              contextIDs.count == contextSizes.count, zip(contextIDs, contextSizes).allSatisfy({ $0 >= 0 && $0 < $1 }) else {
            throw AstraError("evaluation.protocol", "Choose valid fixed trials, timing, contexts and a bounded evaluation duration.")
        }
        for trial in trials { try trial.validated() }
        if let contextVocabulary {
            _ = try contextVocabulary.validated()
            guard contextVocabulary.sizes == contextSizes else { throw AstraError("evaluation.contexts", "The evaluation context vocabulary does not match its dimensions.") }
        }
        return self
    }
    public var payload: JSONValue {
        get throws {
            var fields = try JSONValue.encode(self).fields!
            fields["contextVocabulary"] = contextVocabulary?.payload ?? .null
            return .object(fields)
        }
    }
    public var fingerprint: String {
        get throws {
            var value = try payload.fields!
            if deterministic { value["policySeed"] = .integer(0) }
            if let contextVocabulary {
                value["contextVocabulary"] = .array(contextVocabulary.fields.map { field in
                    .object(["id": .string(field.id.uuidString.lowercased()),
                        "values": .array(field.values.map { .string($0.id.uuidString.lowercased()) })])
                })
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            return SHA256.hash(data: try encoder.encode(JSONValue.object(value))).map { String(format: "%02x", $0) }.joined()
        }
    }
}

public struct ClosedLoopTrialResult: Codable, Equatable, Sendable, Identifiable {
    public var index: Int
    public var seed: Int
    public var outcome: String
    public var success: Bool
    public var returnValue: Double
    public var decisions: Int
    public var virtualDurationMS: Int
    public var fault: String?
    public var id: Int { index }
}
public struct ClosedLoopResult: Codable, Equatable, Sendable {
    public var checkpointID: UUID
    public var policySignature: String
    public var protocolFingerprint: String
    public var provenance: String
    public var trials: [ClosedLoopTrialResult]
    public var successes: Int { trials.filter(\.success).count }
    public var timeouts: Int { trials.filter { $0.outcome == "truncated" }.count }
    public var faults: Int { trials.filter { $0.outcome.hasSuffix("_fault") }.count }
    public var wrongChoices: Int { trials.filter { $0.outcome == "terminated" && !$0.success }.count }
    public var meanReturn: Double { trials.isEmpty ? 0 : trials.reduce(0) { $0 + $1.returnValue } / Double(trials.count) }
    public func validated(protocol definition: ClosedLoopProtocol, checkpointID expectedID: UUID, signature: String) throws -> Self {
        _ = try definition.validated()
        guard checkpointID == expectedID, policySignature == signature, provenance == "practice_closed_loop",
              protocolFingerprint == (try definition.fingerprint), trials.count == definition.trials.count else {
            throw AstraError("evaluation.resultIdentity", "The practice results differ from their checkpoint or frozen protocol.")
        }
        for (index, result) in trials.enumerated() {
            guard result.index == index, result.seed == definition.trials[index].seed,
                  ["terminated", "truncated", "action_fault", "numerical_fault"].contains(result.outcome),
                  !result.success || result.outcome == "terminated", result.returnValue.isFinite,
                  (0...200_000).contains(result.decisions), (0...definition.timeLimitMS).contains(result.virtualDurationMS),
                  result.virtualDurationMS == min(result.decisions * definition.periodMS, definition.timeLimitMS),
                  result.outcome.hasSuffix("_fault") || result.decisions > 0,
                  (result.fault?.utf8.count ?? 0) <= 16_384,
                  result.outcome.hasSuffix("_fault") == (result.fault != nil) else {
                throw AstraError("evaluation.trialResult", "A practice trial returned inconsistent or nonfinite outcomes.")
            }
        }
        guard trials.reduce(0.0, { $0 + $1.returnValue }).isFinite else {
            throw AstraError("evaluation.aggregate", "Practice returns exceed their finite aggregate bound.")
        }
        return self
    }
}

public struct ClosedLoopEvaluationDocument: Codable, Identifiable, Equatable, Sendable {
    public var schemaVersion = 1
    public var id: UUID
    public var comparisonID: UUID
    public var agentID: UUID
    public var checkpointID: UUID
    public var checkpointName: String
    public var checkpointPolicySignature: String
    public var protocolDefinition: ClosedLoopProtocol
    public var createdAt: Date
    public var finishedAt: Date?
    public var status: EvaluationStatus = .running
    public var result: ClosedLoopResult?
    public var issue: String?
    public init(id: UUID = UUID(), comparisonID: UUID, agentID: UUID, checkpoint: CheckpointDocument, protocolDefinition: ClosedLoopProtocol) {
        self.id = id; self.comparisonID = comparisonID; self.agentID = agentID
        checkpointID = checkpoint.id; checkpointName = checkpoint.name; checkpointPolicySignature = checkpoint.policySignature
        self.protocolDefinition = protocolDefinition; createdAt = Date()
    }
    public func validated() throws -> Self {
        _ = try protocolDefinition.validated(); _ = try DocumentNames.validated(checkpointName)
        guard schemaVersion == 1, EvaluationProtocol.isDigest(checkpointPolicySignature), createdAt.timeIntervalSince1970.isFinite,
              finishedAt.map({ $0.timeIntervalSince1970.isFinite && $0 >= createdAt }) ?? true,
              (status == .running) == (finishedAt == nil), (status == .completed) == (result != nil),
              (issue?.utf8.count ?? 0) <= 65_536 else { throw AstraError("evaluation.document", "Invalid practice evaluation history.") }
        if let result { _ = try result.validated(protocol: protocolDefinition, checkpointID: checkpointID, signature: checkpointPolicySignature) }
        return self
    }
    public func comparisonIssue(with other: Self) -> String? {
        guard status == .completed, other.status == .completed else { return "Both evaluations must finish before comparison." }
        guard (try? protocolDefinition.fingerprint) == (try? other.protocolDefinition.fingerprint) else {
            return "Different task, seeds, layouts, timing, action selection or context meanings."
        }
        return nil
    }
}
