import Foundation
import Testing
@testable import AstraCore

private func retentionCheckpoint(agent: UUID, name: String, step: Int) -> CheckpointDocument {
    var value = CheckpointDocument(id: UUID(), agentID: agent, runID: nil, name: name, kind: "initial",
        trainingStep: step, policySignature: String(repeating: "a", count: 64), parameterCount: 100)
    value.createdAt = Date(timeIntervalSince1970: Double(step))
    return value
}

@Test func checkpointRetentionProtectsDependenciesAndOnlyDeletesFinalOwnerFiles() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("astra-retention-\(UUID())")
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    var agent = AgentDocument(name: "First")
    try await store.save(agent)
    let names = ["Pinned", "Selected", "Feedback", "Shared", "Active", "Correction source"]
    let checkpoints = names.enumerated().map { retentionCheckpoint(agent: agent.id, name: $0.element, step: $0.offset + 1) }
    for checkpoint in checkpoints { try await store.saveCheckpoint(checkpoint) }
    let other = try await store.duplicateAgent(agent)
    agent.selectedCheckpointID = checkpoints[1].id; try await store.save(agent)
    try await store.updateCheckpointPresentation(id: checkpoints[0].id, name: "Keep forever", pinned: true)
    let pending = PendingFeedbackDocument(behaviorBatchID: UUID(), agentID: agent.id, checkpoint: checkpoints[2],
        createdAt: Date(timeIntervalSince1970: 1000), configuration: .object([:]), fragments: [.init(collectionID: UUID(), sourceDirectory: "/fixture/source",
            manifestSHA256: String(repeating: "b", count: 64), revisionDirectory: "/fixture/revisions", boundary: .object([:]))])
    try await store.savePendingFeedback(pending)
    var active = LearningRunDocument(agentID: agent.id, kind: .behavioral, name: "In progress", sourceKind: "recordings")
    active.initialCheckpointID = checkpoints[4].id; try await store.saveLearningRun(active)
    let recording = RecordingManifest(name: "Correction", environment: .init(name: "Window", kind: .window))
    let writer = try RecordingWriter(directory: store.recordingDirectory(id: recording.id), manifest: recording)
    try writer.attachCorrection(.init(sourceRunID: UUID(), sourceCheckpointID: checkpoints[5].id,
        sourcePolicySignature: checkpoints[5].policySignature, contextIDs: [], requestedAtNanos: 10,
        controlJoinedAtNanos: 11, observations: []), supervisionStartNanos: 12)
    try await store.saveRecording(writer.finish(at: 20, status: .interrupted), linkTo: agent.id)
    let disposable = retentionCheckpoint(agent: agent.id, name: "Disposable", step: 7)
    try await store.saveCheckpoint(disposable)
    for checkpoint in checkpoints + [disposable] {
        let folder = root.appendingPathComponent("Models/\(checkpoint.id.uuidString.lowercased())")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data([1, 2, 3, 4]).write(to: folder.appendingPathComponent("policy.safetensors"))
    }
    let preview = try await store.previewCheckpointRetention(agentID: agent.id, keepNewest: 0)
    #expect(Set(preview.items.map(\.id)) == [checkpoints[3].id, disposable.id])
    #expect(preview.items.first { $0.id == checkpoints[3].id }?.disposition == .unlinkShared)
    #expect(preview.bytesToDelete == 4)
    let result = try await store.applyCheckpointRetention(preview)
    #expect(result.deleted == 1 && result.unlinked == 1 && result.issues.isEmpty)
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("Models/\(disposable.id.uuidString.lowercased())").path))
    #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("Models/\(checkpoints[3].id.uuidString.lowercased())").path))
    #expect(try await store.checkpointIDs(for: other.id).contains(checkpoints[3].id))
    #expect(try await !store.checkpointIDs(for: agent.id).contains(checkpoints[3].id))
    // Reopening never reconstructs the explicitly removed creator link.
    let reopened = try LibraryStore(root: root)
    #expect(try await !reopened.checkpointIDs(for: agent.id).contains(checkpoints[3].id))
    let finalOwner = try await reopened.previewCheckpointRetention(agentID: other.id, keepNewest: 0)
    #expect(finalOwner.items.map(\.id) == [checkpoints[3].id])
    #expect(try await reopened.applyCheckpointRetention(finalOwner).deleted == 1)
    #expect(try await reopened.snapshot().pendingFeedback == [pending])
}

@Test func checkpointRetentionRevalidatesPreviewAndPreservesRenamedPinnedMetadata() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("astra-retention-edit-\(UUID())")
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), agent = AgentDocument(name: "Agent")
    try await store.save(agent)
    let checkpoint = retentionCheckpoint(agent: agent.id, name: "Before", step: 1)
    try await store.saveCheckpoint(checkpoint)
    let preview = try await store.previewCheckpointRetention(agentID: agent.id, keepNewest: 0)
    try await store.updateCheckpointPresentation(id: checkpoint.id, name: "Named policy", pinned: false)
    try await store.saveCheckpoint(checkpoint) // An idempotent publish must not undo presentation edits.
    await #expect(throws: AstraError.self) { try await store.applyCheckpointRetention(preview) }
    let retained = try #require(try await store.snapshot().checkpoints.first)
    #expect(retained.name == "Named policy" && retained.isPinned)
    #expect(retained.matchesIdentity(of: checkpoint))
    let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .millisecondsSince1970
    var old = try #require(JSONSerialization.jsonObject(with: encoder.encode(checkpoint)) as? [String: Any])
    old.removeValue(forKey: "pinned")
    let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
    #expect(try !decoder.decode(CheckpointDocument.self, from: JSONSerialization.data(withJSONObject: old)).isPinned)
}
