import Foundation
import Testing
@testable import AstraCore

@Test func closedLoopProtocolPersistsWithoutChangingNLLAndRetainsSemanticFingerprint() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), agent = AgentDocument(name: "Practice evaluations")
    try await store.save(agent)
    let checkpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: nil, name: "Fixture", kind: "initial", trainingStep: 0,
        policySignature: String(repeating: "a", count: 64), parameterCount: 1)
    var definition = ClosedLoopProtocol(trials: [.init(seed: 100_000, pixelWidth: 64, pixelHeight: 64, logicalBounds: [0, 0, 64, 64]),
        .init(seed: 100_000, pixelWidth: 96, pixelHeight: 64, logicalBounds: [-96, 0, 96, 64])])
    definition.timeLimitMS = 200
    definition.contextVocabulary = .init(fields: [.init(name: "Purpose", values: [.init(name: "Inspect")])])
    definition.contextSizes = [2]; definition.contextIDs = [1]
    let identity = try definition.fingerprint
    var renamed = definition; renamed.contextVocabulary?.fields[0].name = "Task"
    #expect(try renamed.fingerprint == identity)
    renamed.trials[0].seed += 1
    #expect(try renamed.fingerprint != identity)
    var document = ClosedLoopEvaluationDocument(comparisonID: UUID(), agentID: agent.id, checkpoint: checkpoint, protocolDefinition: definition)
    try await store.saveClosedLoopEvaluation(document)
    try await store.markAbandonedLearningRunsInterrupted()
    let reopened = try await LibraryStore(root: root).snapshot()
    #expect(reopened.evaluations.isEmpty && reopened.closedLoopEvaluations.count == 1)
    #expect(reopened.closedLoopEvaluations[0].status == .interrupted && reopened.closedLoopEvaluations[0].result == nil)
    document.status = .cancelled; document.finishedAt = Date()
    await #expect(throws: AstraError.self) { try await store.saveClosedLoopEvaluation(document) }
    if let path = ProcessInfo.processInfo.environment["ASTRA_CLOSED_LOOP_PROTOCOL"] {
        let payload = try JSONValue.object(["protocol": definition.payload, "fingerprint": .string(identity)])
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(payload).write(to: URL(fileURLWithPath: path))
    }
}
