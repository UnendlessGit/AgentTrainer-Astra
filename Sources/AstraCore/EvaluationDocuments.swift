import CryptoKit
import Foundation

/// One explicit, saved scoring protocol is shared by every checkpoint in a comparison.
/// Dataset fingerprints include the recorded revision (and its source/index digests),
/// or the complete generated demonstration specification and model configuration.
public struct EvaluationProtocol: Codable, Equatable, Sendable {
    public var schemaVersion = 1
    public var sourceRunID: UUID
    public var sourceCheckpointID: UUID
    public var sourceName: String
    public var dataset: JSONValue
    public var datasetFingerprint: String
    public var expectedDatasetID: UUID?
    public var provenance: String
    public var policySignature: String
    public var split: String
    public var sequenceLength: Int
    public var verificationMode: Bool

    public init(sourceRunID: UUID, sourceCheckpointID: UUID, sourceName: String, dataset: JSONValue,
                identity: JSONValue, expectedDatasetID: UUID?, provenance: String, policySignature: String,
                split: String, sequenceLength: Int = 64, verificationMode: Bool) throws {
        self.sourceRunID = sourceRunID; self.sourceCheckpointID = sourceCheckpointID; self.sourceName = sourceName
        self.dataset = dataset; self.expectedDatasetID = expectedDatasetID; self.provenance = provenance
        self.policySignature = policySignature; self.split = split; self.sequenceLength = sequenceLength
        self.verificationMode = verificationMode
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        datasetFingerprint = SHA256.hash(data: try encoder.encode(identity)).map { String(format: "%02x", $0) }.joined()
        _ = try validated()
    }

    public func validated() throws -> Self {
        guard schemaVersion == 1, ["train", "validation", "test"].contains(split),
              (1...512).contains(sequenceLength), Self.isDigest(datasetFingerprint), Self.isDigest(policySignature),
              ["recorded_demonstrations", "practice_oracle"].contains(provenance),
              dataset.fields?["kind"] == .string(provenance == "recorded_demonstrations" ? "recordings" : "practice_oracle"),
              provenance != "recorded_demonstrations" || expectedDatasetID != nil,
              (try? DocumentNames.validated(sourceName)) != nil else {
            throw AstraError("evaluation.protocol", "The saved evaluation protocol is invalid or unsupported.")
        }
        return self
    }

    static func isDigest(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    public func mismatch(with other: Self) -> String? {
        if policySignature != other.policySignature { return "Different model, action vocabulary, or observation/action timing." }
        if provenance != other.provenance || datasetFingerprint != other.datasetFingerprint { return "Different demonstration datasets or source revisions." }
        if split != other.split { return "Different demonstration splits (\(split) and \(other.split))." }
        if sequenceLength != other.sequenceLength || verificationMode != other.verificationMode { return "Different scoring protocols." }
        return nil
    }
}

public enum EvaluationStatus: String, Codable, Sendable {
    case running, completed, unavailable, failed, cancelled, interrupted
}

public struct EvaluationDocument: Codable, Identifiable, Equatable, Sendable {
    public var schemaVersion = 1
    public var id: UUID
    public var comparisonID: UUID
    public var agentID: UUID
    public var checkpointID: UUID
    public var checkpointName: String
    public var checkpointPolicySignature: String
    public var protocolDefinition: EvaluationProtocol
    public var createdAt: Date
    public var finishedAt: Date?
    public var status: EvaluationStatus = .running
    /// The identity returned by the evaluator after it opens the actual dataset.
    public var datasetID: UUID?
    public var decisions: Int?
    public var meanNLL: Double?
    public var issue: String?

    public init(id: UUID = UUID(), comparisonID: UUID = UUID(), agentID: UUID, checkpoint: CheckpointDocument,
                protocolDefinition: EvaluationProtocol, createdAt: Date = Date()) {
        self.id = id; self.comparisonID = comparisonID; self.agentID = agentID; checkpointID = checkpoint.id
        checkpointName = checkpoint.name; checkpointPolicySignature = checkpoint.policySignature
        self.protocolDefinition = protocolDefinition; self.createdAt = createdAt
    }

    public func validated() throws -> Self {
        _ = try protocolDefinition.validated()
        guard schemaVersion == 1, (try? DocumentNames.validated(checkpointName)) != nil,
              EvaluationProtocol.isDigest(checkpointPolicySignature), createdAt.timeIntervalSince1970.isFinite,
              finishedAt.map({ $0.timeIntervalSince1970.isFinite && $0 >= createdAt }) ?? true,
              (status == .running) == (finishedAt == nil), decisions.map({ $0 >= 0 }) ?? true,
              meanNLL.map({ $0.isFinite && $0 >= 0 }) ?? true, (issue?.utf8.count ?? 0) <= 65_536 else {
            throw AstraError("evaluation.document", "The evaluation history contains invalid metadata.")
        }
        if status == .completed {
            guard let decisions, decisions > 0, meanNLL != nil, datasetID != nil,
                  checkpointPolicySignature == protocolDefinition.policySignature else {
                throw AstraError("evaluation.result", "A completed evaluation needs a compatible policy, dataset identity, decision count and finite loss.")
            }
        } else if meanNLL != nil || (status != .unavailable && decisions != nil) {
            throw AstraError("evaluation.result", "Unfinished or unsuccessful evaluations cannot publish a loss or decision count.")
        }
        if let expected = protocolDefinition.expectedDatasetID, let datasetID, expected != datasetID {
            throw AstraError("evaluation.datasetIdentity", "The evaluator returned a different dataset revision.")
        }
        return self
    }

    public func comparisonIssue(with other: Self) -> String? {
        if let mismatch = protocolDefinition.mismatch(with: other.protocolDefinition) { return mismatch }
        guard status == .completed, other.status == .completed else { return "Both evaluations must finish with available results." }
        if datasetID != other.datasetID { return "The evaluator opened different dataset identities." }
        if decisions != other.decisions { return "The evaluations scored different numbers of decisions." }
        return nil
    }
}
