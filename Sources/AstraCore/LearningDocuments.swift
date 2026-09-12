import Foundation

public enum LearningKind: String, Codable, CaseIterable, Sendable { case behavioral, reinforcement }
public enum LearningStatus: String, Codable, Sendable { case preparing, running, cancelling, completed, cancelled, failed, interrupted }

public struct LearningRunDocument: Codable, Hashable, Identifiable, Sendable {
    public var schemaVersion = AstraVersion.dataVersion
    public var id: UUID
    public var agentID: UUID
    public var kind: LearningKind
    public var status: LearningStatus
    public var name: String
    public var createdAt: Date
    public var modifiedAt: Date
    public var epoch: Int = 0
    public var updates: Int = 0
    public var decisions: Int = 0
    public var meanNLL: Double?
    public var sourceRecordingIDs: [UUID] = []
    public var sourceKind: String
    public var datasetID: UUID?
    public var initialCheckpointID: UUID?
    public var checkpointID: UUID?
    public var issue: String?

    public init(id: UUID = UUID(), agentID: UUID, kind: LearningKind, name: String, sourceKind: String) {
        self.id = id; self.agentID = agentID; self.kind = kind; self.name = name; self.sourceKind = sourceKind
        self.status = .preparing; self.createdAt = Date(); self.modifiedAt = self.createdAt
    }
    public func validated() throws -> Self {
        var value = self; value.name = try DocumentNames.validated(name)
        guard schemaVersion == AstraVersion.dataVersion, createdAt.timeIntervalSince1970.isFinite,
              modifiedAt.timeIntervalSince1970.isFinite, epoch >= 0, updates >= 0, decisions >= 0,
              meanNLL.map({ $0.isFinite && $0 >= 0 }) ?? true,
              ["recordings", "practice_oracle", "practice_rollout", "desktop_rollout"].contains(sourceKind),
              sourceRecordingIDs.count <= 4096, Set(sourceRecordingIDs).count == sourceRecordingIDs.count,
              (issue?.utf8.count ?? 0) <= 65_536 else {
            throw AstraError("learning.document", "The learning run has invalid or unsupported metadata.")
        }
        if status == .completed, checkpointID == nil {
            throw AstraError("learning.checkpoint", "A completed learning run must publish its checkpoint.")
        }
        return value
    }
}

public struct CheckpointDocument: Codable, Hashable, Identifiable, Sendable {
    public var schemaVersion = AstraVersion.dataVersion
    public var id: UUID
    public var agentID: UUID
    public var runID: UUID?
    public var name: String
    public var kind: String
    public var createdAt: Date
    public var trainingStep: Int
    public var policySignature: String
    public var parameterCount: Int

    public init(id: UUID, agentID: UUID, runID: UUID?, name: String, kind: String,
                trainingStep: Int, policySignature: String, parameterCount: Int) {
        self.id = id; self.agentID = agentID; self.runID = runID; self.name = name; self.kind = kind
        self.createdAt = Date(); self.trainingStep = trainingStep; self.policySignature = policySignature; self.parameterCount = parameterCount
    }
    public func validated() throws -> Self {
        var value = self; value.name = try DocumentNames.validated(name)
        guard schemaVersion == AstraVersion.dataVersion, createdAt.timeIntervalSince1970.isFinite,
              ["initial", "behavioral", "reinforcement"].contains(kind), trainingStep >= 0,
              (1...500_000_000).contains(parameterCount), policySignature.count == 64,
              policySignature.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw AstraError("checkpoint.document", "The checkpoint has invalid or unsupported metadata.")
        }
        return value
    }
    /// Display names and serialized date precision do not change policy or
    /// provenance identity. Runtime admissions compare the immutable fields.
    public func matchesIdentity(of other: Self) -> Bool {
        schemaVersion == other.schemaVersion && id == other.id && agentID == other.agentID && runID == other.runID && kind == other.kind
            && trainingStep == other.trainingStep && policySignature == other.policySignature && parameterCount == other.parameterCount
    }
}
