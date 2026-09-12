import Foundation
import Testing
@testable import AstraCore

private func selectionRecording(store: LibraryStore, creator: UUID? = nil) throws -> RecordingManifest {
    let manifest = RecordingManifest(name: "Shared source", environment: .init(name: "Fixture", kind: .desktop), recordedForAgentID: creator)
    let writer = try RecordingWriter(directory: store.recordingDirectory(id: manifest.id), manifest: manifest)
    let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
    try writer.append(FrameArchive.prepare(pixels: Data(repeating: 42, count: 16),
        metadata: .init(eventNanos: 1_000_000_000, observedNanos: 1_000_000_000, surface: surface, byteCount: 16)))
    return try writer.finish(at: 5_000_000_000, status: .complete)
}

@Test func recordingLinksKeepIndependentRangesAndUnlinkSurvivesRecovery() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let first = AgentDocument(name: "First"), second = AgentDocument(name: "Second")
    try await store.save(first); try await store.save(second)
    let recording = try selectionRecording(store: store, creator: first.id)
    try await store.saveRecording(recording, linkTo: first.id)
    let files = try FileManager.default.contentsOfDirectory(at: store.recordingDirectory(id: recording.id), includingPropertiesForKeys: nil)
    let original = try Dictionary(uniqueKeysWithValues: files.map { ($0, try Data(contentsOf: $0)) })
    try await store.linkRecordings([recording.id], to: second.id)
    let selection = RecordingTrainingSelection(ranges: [.init(startNanos: 1_100_000_000, endNanos: 2_000_000_000), .init(startNanos: 3_000_000_000, endNanos: 4_000_000_000)])
    try await store.saveRecordingSelection(selection, recordingID: recording.id, agentID: first.id)
    let copy = try await store.duplicateAgent(first)
    let snapshot = try await store.snapshot()
    #expect(snapshot.recordings.count == 1 && snapshot.recordingSelections[first.id]?[recording.id] == selection)
    #expect(snapshot.recordingSelections[second.id]?[recording.id] == .whole)
    #expect(snapshot.recordingSelections[copy.id]?[recording.id] == selection)
    try await store.unlinkRecording(recording.id, from: first.id)
    _ = try await store.recoverInterruptedRecordings()
    #expect(try await store.recordingIDs(for: first.id).isEmpty)
    #expect(try await store.recordingIDs(for: second.id) == [recording.id])
    for (file, bytes) in original { #expect(try Data(contentsOf: file) == bytes) }
}

@Test func recordingSelectionRejectsOverlapAndInvalidCoverageWithoutChangingSavedRanges() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), agent = AgentDocument(name: "Bounds")
    try await store.save(agent)
    let recording = try selectionRecording(store: store)
    try await store.saveRecording(recording, linkTo: agent.id)
    let adjacent = RecordingTrainingSelection(ranges: [.init(startNanos: 1_000_000_000, endNanos: 2_000_000_000), .init(startNanos: 2_000_000_000, endNanos: 3_000_000_000)])
    try await store.saveRecordingSelection(adjacent, recordingID: recording.id, agentID: agent.id)
    for ranges: [RecordingTimeRange] in [[], [.init(startNanos: 1, endNanos: 2)],
        [.init(startNanos: 1_000_000_000, endNanos: 2_500_000_000), .init(startNanos: 2_000_000_000, endNanos: 3_000_000_000)]] {
        await #expect(throws: AstraError.self) { try await store.saveRecordingSelection(.init(ranges: ranges), recordingID: recording.id, agentID: agent.id) }
    }
    #expect(try await store.snapshot().recordingSelections[agent.id]?[recording.id] == adjacent)
    let other = AgentDocument(name: "Atomic links"); try await store.save(other)
    await #expect(throws: AstraError.self) { try await store.linkRecordings([recording.id, UUID()], to: other.id) }
    #expect(try await store.recordingIDs(for: other.id).isEmpty)
    #expect(try await store.recordingIDs(for: agent.id) == [recording.id])
    var interrupted = recording; interrupted.status = .interrupted; interrupted.firstInvalidObservedNanos = 1_500_000_000
    try await store.saveRecording(interrupted)
    let snapshot = try await store.snapshot()
    #expect(snapshot.recordingSelections[agent.id]?[recording.id] == nil)
    #expect(snapshot.issues.contains { $0.collection == "recording selections" })
    #expect(try RecordingTrainingSelection.whole.resolved(for: interrupted).first?.endNanos == 1_500_000_000)
}

@Test func legacyCatalogMigrationPreservesLinksAndCorruptSelectionsRemainVisible() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), agent = AgentDocument(name: "Migration")
    try await store.save(agent)
    let recording = try selectionRecording(store: store)
    try await store.saveRecording(recording, linkTo: agent.id)
    let database = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    try database.execute("ALTER TABLE agent_recordings DROP COLUMN selection")
    try database.execute("UPDATE schema_info SET version=1")
    let reopened = try LibraryStore(root: root)
    #expect(try database.query("SELECT version FROM schema_info").first?["version"] == .integer(2))
    #expect(try await reopened.snapshot().recordingSelections[agent.id]?[recording.id] == .whole)
    try database.execute("UPDATE agent_recordings SET selection=?", [.blob(Data([255]))])
    let snapshot = try await reopened.snapshot()
    #expect(snapshot.recordingSelections[agent.id]?[recording.id] == nil && snapshot.issues.count == 1)
    #expect(try database.query("SELECT selection FROM agent_recordings").first?["selection"] == .blob(Data([255])))
    await #expect(throws: (any Error).self) { _ = try await reopened.duplicateAgent(agent) }
    #expect(try await reopened.snapshot().agents.count == 1) // Copy transaction rolled back.
}

@Test func failedCatalogMigrationRollsBackItsSchemaVersion() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let database = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    try database.execute("CREATE TABLE schema_info(version INTEGER NOT NULL)")
    try database.execute("INSERT INTO schema_info VALUES(1)")
    // Deliberately inconsistent legacy state: migration must fail atomically.
    try database.execute("CREATE TABLE agent_recordings(agent_id TEXT,recording_id TEXT,selection BLOB)")
    #expect(throws: AstraError.self) { _ = try LibraryStore(root: root) }
    #expect(try database.query("SELECT version FROM schema_info").first?["version"] == .integer(1))
    #expect(try database.query("SELECT name FROM sqlite_master WHERE name='agents'").isEmpty)
}

@Test func initialSchemaOneWorkspaceCanUpgradeBeforeRecordingLinksExisted() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Early workspace", createdAt: Date(timeIntervalSince1970: 1000))
    try await store.save(agent)
    let database = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    try database.execute("DROP TABLE agent_recordings")
    try database.execute("UPDATE schema_info SET version=1")
    let reopened = try LibraryStore(root: root)
    #expect(try await reopened.snapshot().agents == [agent])
    #expect(try database.query("SELECT version FROM schema_info").first?["version"] == .integer(2))
    #expect(try database.query("PRAGMA table_info(agent_recordings)").contains { $0["name"] == .text("selection") })
}
