import Foundation
import Testing
@testable import AstraCore

@Test func artifactArchiveRoundTripAndIndependentStorageMigrationRecoverWithoutChangingSources() async throws {
    let base = FileManager.default.temporaryDirectory.appendingPathComponent("astra-artifact-transfer-\(UUID())", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: base) }
    let root = base.appendingPathComponent("Library", isDirectory: true)
    let store = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Portable fixture")
    try await store.save(agent)
    let initial = RecordingManifest(name: "Generated pixels", environment: .init(name: "Fixture", kind: .window),
        recordedForAgentID: agent.id, surfaceIDs: ["window:7"])
    let original = store.recordingDirectory(id: initial.id)
    let writer = try RecordingWriter(directory: original, manifest: initial)
    let surface = SurfaceDescriptor(id: "window:7", globalBounds: .init(x: 0, y: 0, width: 1, height: 1),
        pixelWidth: 1, pixelHeight: 1, nativeWindowID: 7)
    let metadata = FrameMetadata(eventNanos: 100, observedNanos: 101, surface: surface, byteCount: 4, codec: "raw")
    try writer.append(FrameArchive.prepare(pixels: Data([11, 22, 33, 255]), metadata: metadata))
    let sealed = try writer.finish(at: 1_000, status: .complete)
    try await store.saveRecording(sealed, linkTo: agent.id)
    let contexts = [ContextFieldDocument(name: "Layout", values: [.init(name: "Left")]),
                    ContextFieldDocument(name: "Mode", values: [.init(name: "Practice")])]
    try await store.saveContexts(contexts, selectedIDs: contexts.map(\.id), for: agent.id)
    try await store.saveRecordingSelection(.init(contextValues: Dictionary(uniqueKeysWithValues: contexts.map { ($0.id, $0.values[0].id) })),
        recordingID: sealed.id, agentID: agent.id)
    let originalInventory = try ArtifactTransferFiles.inventory(original)
    let export = try await store.previewArtifactExport(agentID: agent.id, recordingIDs: [sealed.id], checkpointIDs: [])
    let archive = base.appendingPathComponent("Fixture.astraarchive", isDirectory: true)
    await #expect(throws: CancellationError.self) { try await store.exportArtifactArchive(export, to: archive, cancelled: { true }) }
    let pendingExport = try #require(try await store.pendingArtifactTransfers().first?.exportPlan)
    #expect(pendingExport.archiveSHA256 == export.archiveSHA256)
    try await store.exportArtifactArchive(pendingExport, to: archive)
    try await store.exportArtifactArchive(pendingExport, to: archive) // Published retry is idempotent, never an overwrite.
    let importedStore = try LibraryStore(root: base.appendingPathComponent("Imported", isDirectory: true))
    let imported = try await importedStore.importArtifactArchive(importedStore.previewArtifactImport(from: archive))
    #expect(imported.recordingIDs == [sealed.id] && imported.importedCount == 1)
    #expect(try ArtifactTransferFiles.inventory(importedStore.recordingDirectory(id: sealed.id)) == originalInventory)
    #expect(try await importedStore.recordingIDs(for: agent.id) == [sealed.id])
    let duplicate = try await importedStore.importArtifactArchive(importedStore.previewArtifactImport(from: archive))
    #expect(duplicate.reusedCount == 1 && duplicate.importedCount == 0)

    let migration = try await store.previewStorageMigration(kind: .recordings, destination: base.appendingPathComponent("External recordings", isDirectory: true))
    await #expect(throws: CancellationError.self) { try await store.applyStorageMigration(migration, cancelled: { true }) }
    let recovery = try #require(try await store.pendingArtifactTransfers().first?.migration)
    let stage = migration.destination.deletingLastPathComponent().appendingPathComponent(".astra-transfer-" + migration.id.uuidString.lowercased())
    try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
    let orphan = stage.appendingPathComponent(".astra-copy-" + UUID().uuidString.lowercased())
    try Data([9, 8, 7]).write(to: orphan)
    let switched = try await store.applyStorageMigration(recovery)
    #expect(switched.recordingsRoot == migration.destination && switched.modelsRoot == store.layout.modelsRoot)
    #expect(try ArtifactTransferFiles.inventory(original) == originalInventory)
    let reopened = try LibraryStore(root: root)
    #expect(reopened.layout == switched)
    #expect(try ArtifactTransferFiles.inventory(reopened.recordingDirectory(id: sealed.id)) == originalInventory)
    #expect(try await reopened.pendingArtifactTransfers().isEmpty)

    let modelSource = reopened.layout.modelsRoot.appendingPathComponent("generated-storage-fixture.bin")
    let modelBytes = Data(repeating: 73, count: 8192)
    try modelBytes.write(to: modelSource)
    let models = try await reopened.previewStorageMigration(kind: .models, destination: base.appendingPathComponent("External models", isDirectory: true))
    let both = try await reopened.applyStorageMigration(models)
    #expect(both.recordingsRoot == switched.recordingsRoot)
    #expect(try Data(contentsOf: both.modelsRoot.appendingPathComponent(modelSource.lastPathComponent)) == modelBytes)
    #expect(try Data(contentsOf: modelSource) == modelBytes)
    #expect(try ArtifactTransferFiles.inventory(original) == originalInventory)
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_ARTIFACT_RESUME_FIXTURE"] != nil))
func artifactArchivePreservesRealCheckpointDatasetAndResumeConfiguration() async throws {
    let path = try #require(ProcessInfo.processInfo.environment["ASTRA_ARTIFACT_RESUME_FIXTURE"])
    let fixtureURL = URL(fileURLWithPath: path), fixture = try ArtifactArchive.json(fixtureURL)
    #expect(fixture.fields?["generatedOnly"] == .bool(true))
    let sourceRoot = URL(fileURLWithPath: try #require(fixture.fields?["library"]?.text), isDirectory: true)
    guard ArtifactTransferFiles.contains(fixtureURL.deletingLastPathComponent(), sourceRoot), fixture.fields?["generatedOnly"] == .bool(true) else {
        throw AstraError("fixture.path", "Use only the generated local archive-resume fixture.")
    }
    let store = try LibraryStore(root: sourceRoot)
    let agentID = try fixture.requiredUUID("agentID"), checkpointID = try fixture.requiredUUID("checkpointID")
    let recordingID = try fixture.requiredUUID("recordingID"), runID = try fixture.requiredUUID("runID")
    let agent = AgentDocument(id: agentID, name: "Archive resume fixture", selectedCheckpointID: checkpointID)
    try await store.save(agent)
    let checkpoint = CheckpointDocument(id: checkpointID, agentID: agentID, runID: runID, name: "Paused genuine BC fixture", kind: "behavioral",
        trainingStep: try #require(fixture.fields?["trainingStep"]?.int), policySignature: try #require(fixture.fields?["policySignature"]?.text),
        parameterCount: try #require(fixture.fields?["parameterCount"]?.int))
    try await store.saveCheckpoint(checkpoint)
    let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
    let recording = try decoder.decode(RecordingManifest.self,
        from: ArtifactTransferFiles.read(store.recordingDirectory(id: recordingID).appendingPathComponent("manifest.json"), limit: 262_144))
    try await store.saveRecording(recording, linkTo: agentID)
    var run = LearningRunDocument(id: runID, agentID: agentID, kind: .behavioral, name: "Paused generated behavior", sourceKind: "recordings")
    run.status = .cancelled; run.checkpointID = checkpointID; run.datasetID = try fixture.requiredUUID("datasetID")
    run.sourceRecordingIDs = [recordingID]; run.decisions = try #require(fixture.fields?["decisions"]?.int)
    try await store.saveLearningRun(run)
    let originalModel = try ArtifactTransferFiles.inventory(store.checkpointDirectory(id: checkpointID))
    let plan = try await store.previewArtifactExport(agentID: agentID, recordingIDs: [], checkpointIDs: [checkpointID])
    #expect(Set(plan.items.map(\.kind)) == [.checkpoint, .dataset, .recording, .runConfiguration])
    let output = fixtureURL.deletingLastPathComponent().appendingPathComponent("NativeRoundTrip-" + UUID().uuidString.lowercased(), isDirectory: true)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    let archive = output.appendingPathComponent("Checkpoint.astraarchive", isDirectory: true)
    try await store.exportArtifactArchive(plan, to: archive)
    let importedRoot = output.appendingPathComponent("ImportedLibrary", isDirectory: true)
    let importedStore = try LibraryStore(root: importedRoot)
    _ = try await importedStore.importArtifactArchive(importedStore.previewArtifactImport(from: archive))
    #expect(try ArtifactTransferFiles.inventory(importedStore.checkpointDirectory(id: checkpointID)) == originalModel)
    let reexport = try await importedStore.previewArtifactExport(agentID: agentID, recordingIDs: [], checkpointIDs: [checkpointID])
    #expect(reexport.items.map(\.files) == plan.items.map(\.files))
    #expect(try ArtifactTransferFiles.inventory(store.checkpointDirectory(id: checkpointID)) == originalModel)
    let result: JSONValue = .object(["importedLibrary": .string(importedRoot.path), "checkpointID": .string(checkpointID.uuidString.lowercased()),
        "datasetID": try fixture.required("datasetID"), "runID": .string(runID.uuidString.lowercased()), "generatedOnly": .bool(true)])
    try JSONEncoder().encode(result).write(to: fixtureURL.deletingLastPathComponent().appendingPathComponent("native-roundtrip-result.json"), options: .atomic)
}
