import Foundation
import Testing
@testable import AstraCore

@Test func contextVocabularyPreservesMeaningAcrossCatalogRenameAndSelectionRoundTrip() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Context fixture")
    try await store.save(agent)
    var field = ContextFieldDocument(name: "Task", values: [.init(name: "Organize"), .init(name: "Inspect")])
    try await store.saveContexts([field], selectedIDs: [field.id], for: agent.id)
    let snapshot = try await store.snapshot()
    let frozen = try ContextVocabulary(fields: snapshot.contextFields).validated()
    let selection = RecordingTrainingSelection(contextValues: [field.id: field.values[1].id])
    var recording = RecordingManifest(name: "Whole context selection", environment: .init(name: "Fixture", kind: .practice))
    recording.frameCount = 1; recording.firstObservedNanos = 100; recording.stoppedNanos = 200
    recording.storedBytes = 100; recording.status = .complete
    try await store.saveRecording(recording, linkTo: agent.id)
    try await store.saveRecordingSelection(selection, recordingID: recording.id, agentID: agent.id)
    #expect(try await LibraryStore(root: root).snapshot().recordingSelections[agent.id]?[recording.id] == selection)
    let decoded = try JSONDecoder().decode(RecordingTrainingSelection.self, from: JSONEncoder().encode(selection))
    #expect(try decoded.payload(recordingID: UUID(), vocabulary: frozen).required("context_ids") == .array([.integer(2)]))
    #expect(try frozen.indices(for: [:]) == [0])
    field.name = "Purpose"; field.values[1].name = "Review"
    try await store.saveContexts([field], selectedIDs: [field.id], for: agent.id)
    let reopened = try await LibraryStore(root: root).snapshot()
    #expect(reopened.contextFields.first?.name == "Purpose")
    #expect(reopened.agents.first?.contextFieldIDs == [field.id])
    #expect(frozen.fields[0].name == "Task" && frozen.fields[0].values[1].name == "Inspect")
    #expect(try ContextVocabulary.from(model: frozen.applying(to: .object([:]))).requireUnwrapped() == frozen)
    #expect(throws: AstraError.self) { try frozen.indices(for: [field.id: UUID()]) }
    #expect(throws: AstraError.self) { try ContextVocabulary(fields: [field, field]).validated() }
}

private extension Optional {
    func requireUnwrapped() throws -> Wrapped {
        guard let value = self else { throw AstraError("test.nil", "Expected a value") }
        return value
    }
}
