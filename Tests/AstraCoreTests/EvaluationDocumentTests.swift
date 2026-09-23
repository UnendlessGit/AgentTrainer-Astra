import Foundation
import Testing
@testable import AstraCore

private func evaluationFixture() throws -> EvaluationDocument {
    let checkpoint = CheckpointDocument(id: UUID(), agentID: UUID(), runID: UUID(), name: "Candidate", kind: "behavioral",
        trainingStep: 20, policySignature: String(repeating: "a", count: 64), parameterCount: 10)
    let definition = try EvaluationProtocol(sourceRunID: UUID(), sourceCheckpointID: UUID(), sourceName: "Saved demonstrations",
        dataset: .object(["kind": .string("practice_oracle")]), identity: .object(["seed": .integer(3)]), expectedDatasetID: nil,
        provenance: "practice_oracle", policySignature: checkpoint.policySignature, split: "validation", verificationMode: true)
    return EvaluationDocument(agentID: checkpoint.agentID, checkpoint: checkpoint, protocolDefinition: definition,
                              createdAt: Date(timeIntervalSince1970: 1_000))
}

@Test func evaluationComparisonRequiresActualSharedIdentityAndScoringProtocol() throws {
    var first = try evaluationFixture()
    first.status = .completed; first.finishedAt = first.createdAt + 2
    first.datasetID = UUID(); first.decisions = 50; first.meanNLL = 1.2
    _ = try first.validated()
    var other = first; other.checkpointID = UUID(); other.meanNLL = 0.7
    #expect(first.comparisonIssue(with: other) == nil)
    other.protocolDefinition.split = "test"
    #expect(first.comparisonIssue(with: other)?.contains("splits") == true)
    other = first; other.datasetID = UUID()
    #expect(first.comparisonIssue(with: other)?.contains("identities") == true)
    other = first; other.decisions = 49
    #expect(first.comparisonIssue(with: other)?.contains("numbers") == true)
    other = first; other.protocolDefinition.policySignature = String(repeating: "b", count: 64)
    #expect(first.comparisonIssue(with: other)?.contains("timing") == true)
    other = first; other.meanNLL = .nan
    #expect(throws: AstraError.self) { try other.validated() }
}

@Test func evaluationHistoryFinalizesOnceAndRecoversInterruptedAttempts() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    var completed = try evaluationFixture()
    try await store.saveEvaluation(completed)
    completed.status = .completed; completed.finishedAt = completed.createdAt + 1
    completed.datasetID = UUID(); completed.meanNLL = 0.2; completed.decisions = 40
    try await store.saveEvaluation(completed)
    try await store.saveEvaluation(completed)
    var changed = completed; changed.meanNLL = 0.1
    await #expect(throws: AstraError.self) { try await store.saveEvaluation(changed) }
    let pending = try evaluationFixture()
    try await store.saveEvaluation(pending)
    try await store.markAbandonedLearningRunsInterrupted()
    let recovered = try await LibraryStore(root: root).snapshot().evaluations
    #expect(recovered.first { $0.id == completed.id }?.meanNLL == 0.2)
    #expect(recovered.first { $0.id == pending.id }?.status == .interrupted)
    #expect(recovered.first { $0.id == pending.id }?.meanNLL == nil)
}
