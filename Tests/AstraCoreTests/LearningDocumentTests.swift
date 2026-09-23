import Foundation
import Testing
@testable import AstraCore

@Test func damagedRunHistoryDoesNotHideOtherLibraryArtifacts() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Existing agent", createdAt: Date(timeIntervalSince1970: 1000))
    try await store.save(agent)
    try Data("Invalid run directory".utf8).write(to: root.appendingPathComponent("Runs"))
    try await store.inspectPriorInferenceRuns()
    let snapshot = try await store.snapshot()
    #expect(snapshot.agents == [agent])
    #expect(snapshot.issues.contains { $0.id == "Runs.history" && $0.message.contains("could not be checked") })
}

@Test func unconfirmedInputCleanupRemainsVisibleAfterWorkspaceReopen() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let id = UUID()
    let run = root.appendingPathComponent("Runs/\(id.uuidString.lowercased())")
    try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
    try await store.inspectPriorInferenceRuns()
    #expect(try await store.snapshot().issues.contains { $0.controlHistory?.runID == id && $0.blocksLiveControl })
    let result: JSONValue = .object(["runID": .string(id.uuidString), "cleanupConfirmed": .bool(false)])
    try JSONEncoder().encode(result).write(to: run.appendingPathComponent("results.json"))
    try await store.inspectPriorInferenceRuns()
    #expect(try await store.snapshot().issues.contains { $0.controlHistory?.runID == id && $0.blocksLiveControl })
    let settled: JSONValue = .object(["runID": .string(id.uuidString), "cleanupConfirmed": .bool(true)])
    try JSONEncoder().encode(settled).write(to: run.appendingPathComponent("results.json"))
    try await store.inspectPriorInferenceRuns()
    #expect(try await store.snapshot().issues.isEmpty)
}

@Test func libraryCoordinatorLeaseExcludesAnotherOwnerAndRejectsLinkedLocks() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    do {
        let lease = try LibraryLease(root: root)
        _ = withExtendedLifetime(lease) {
            #expect(throws: AstraError.self) { try LibraryLease(root: root) }
        }
    }
    do { let lease = try LibraryLease(root: root); withExtendedLifetime(lease) {} }
    let lock = root.appendingPathComponent(".coordinator.lock")
    try FileManager.default.removeItem(at: lock)
    try FileManager.default.createSymbolicLink(at: lock, withDestinationURL: root.appendingPathComponent("unrelated"))
    #expect(throws: AstraError.self) { try LibraryLease(root: root) }
}

@Test func checkpointCatalogIsImmutableWhileRunMetricsRemainDurable() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Learner")
    try await store.save(agent)
    var run = LearningRunDocument(agentID: agent.id, kind: .behavioral, name: "First training", sourceKind: "recordings")
    run.createdAt = Date(timeIntervalSince1970: 1_000); run.modifiedAt = run.createdAt
    try await store.saveLearningRun(run)
    var checkpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: run.id, name: "Epoch 2", kind: "behavioral",
                                        trainingStep: 20, policySignature: String(repeating: "a", count: 64), parameterCount: 34_638_639)
    checkpoint.createdAt = Date(timeIntervalSince1970: 1_000)
    try await store.saveCheckpoint(checkpoint)
    run.status = .completed; run.checkpointID = checkpoint.id; run.epoch = 2; run.updates = 20; run.meanNLL = 1.3
    try await store.saveLearningRun(run)
    let snapshot = try await store.snapshot()
    #expect(snapshot.learningRuns == [run])
    #expect(snapshot.checkpoints == [checkpoint])
    var invalid = checkpoint; invalid.trainingStep += 1
    await #expect(throws: AstraError.self) { try await store.saveCheckpoint(invalid) }
    var incomplete = run; incomplete.checkpointID = nil
    await #expect(throws: AstraError.self) { try await store.saveLearningRun(incomplete) }
    #expect(try await store.snapshot().checkpoints == [checkpoint])
}
