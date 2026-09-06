import Foundation
import Testing
@testable import AstraCore

@Test func recordingPublishesExactFrameAndLateInputTimes() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = RecordingManifest(name: "Fixture", environment: .init(name: "Window", kind: .window))
    let writer = try RecordingWriter(directory: root, manifest: manifest)
    let surface = SurfaceDescriptor(id: "test", globalBounds: .init(x: 0, y: 0, width: 4, height: 4), pixelWidth: 4, pixelHeight: 4)
    let frame = try FrameArchive.prepare(pixels: Data(repeating: 42, count: 64),
                                        metadata: .init(eventNanos: 50, observedNanos: 100, surface: surface, byteCount: 64))
    try writer.append(frame)
    let events = [
        RawInputEvent(sequence: 0, eventNanos: 200, observedNanos: 210, origin: .physical, kind: .keyDown, keyCode: 13),
        RawInputEvent(sequence: 1, eventNanos: 180, observedNanos: 220, origin: .physical, kind: .pointer, x: 2, y: 3)
    ]
    try writer.append(events: events)
    let finished = try writer.finish(at: 1_000, status: .complete)
    #expect(finished.frameCount == 1)
    #expect(finished.eventCount == 2)
    #expect(finished.firstObservedNanos == 100)
    #expect(finished.status == .complete)
    let database = try SQLiteDatabase(url: root.appendingPathComponent("index.sqlite"), readOnly: true)
    let rows = try database.query("SELECT event FROM events ORDER BY sequence")
    let restored = try rows.map { try JSONDecoder().decode(RawInputEvent.self, from: $0["event"]!.data!) }
    #expect(restored == events)
    #expect(try FrameArchive.scan(url: root.appendingPathComponent("frames-00000.astraframes")).blocks.count == 1)
    #expect(try database.query("PRAGMA journal_mode").first?["journal_mode"]?.string == "delete")
    #expect(throws: AstraError.self) { try writer.append(frame) }
}

@Test func invalidIntervalIsRetainedSeparatelyFromRawSource() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let writer = try RecordingWriter(directory: root, manifest: .init(name: "Partial", environment: .init(name: "Window", kind: .window)))
    writer.markInvalid(from: 200, message: "Lost frame")
    writer.markInvalid(from: 150, message: "Earlier loss")
    try writer.append(events: [.init(sequence: 0, eventNanos: 300, observedNanos: 310, origin: .physical, kind: .keyUp, keyCode: 13)])
    let manifest = try writer.finish(at: 400, status: .interrupted, issue: "Capture interrupted")
    #expect(manifest.firstInvalidObservedNanos == 150)
    #expect(manifest.eventCount == 1)
    #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path))
}

@Test func recordingRejectsTheEntireInvalidInputBatch() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let writer = try RecordingWriter(directory: root, manifest: .init(name: "Atomic input", environment: .init(name: "Desktop", kind: .desktop)))
    let valid = RawInputEvent(sequence: 0, eventNanos: 10, observedNanos: 20, origin: .physical, kind: .keyDown, keyCode: 13)
    let invalid = RawInputEvent(sequence: 1, eventNanos: 30, observedNanos: 40, origin: .physical, kind: .pointer, x: .nan, y: 0)
    #expect(throws: AstraError.self) { try writer.append(events: [valid, invalid]) }
    #expect(writer.snapshot.eventCount == 0)
    try writer.append(events: [valid])
    #expect(throws: AstraError.self) { try writer.finish(at: 19, status: .complete) }
    let sealed = try writer.finish(at: 50, status: .complete)
    #expect(sealed.eventCount == 1)
    #expect(sealed.status == .failed)
    let database = try SQLiteDatabase(url: sealed.indexURL(in: root), readOnly: true)
    let rows = try database.query("SELECT event FROM events")
    #expect(rows.count == 1)
    #expect(try JSONDecoder().decode(RawInputEvent.self, from: rows[0]["event"]!.data!) == valid)
}

@Test func recordingManifestRejectsInconsistentStateAndEscapingIndexPaths() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    var manifest = RecordingManifest(name: "  Valid name  ", environment: .init(name: "Desktop", kind: .desktop))
    #expect(try manifest.validated().name == "Valid name")
    manifest.status = .complete; manifest.stoppedNanos = 100
    #expect(throws: AstraError.self) { try manifest.validated() }
    #expect(throws: AstraError.self) { try RecordingWriter(directory: root, manifest: manifest) }
    #expect(!FileManager.default.fileExists(atPath: root.path))
    manifest.status = .recording; manifest.stoppedNanos = nil; manifest.indexPath = "../outside.sqlite"
    #expect(throws: AstraError.self) { try manifest.indexURL(in: root) }
    manifest.indexPath = "Recovery/\(UUID().uuidString)/index.sqlite"
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("Recovery"), withDestinationURL: FileManager.default.temporaryDirectory)
    #expect(throws: AstraError.self) { try manifest.indexURL(in: root) }
}

@Test func failedSealNeverReportsCompletionAndLeavesRecoverableSource() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = RecordingManifest(name: "Failed seal", environment: .init(name: "Desktop", kind: .desktop))
    let writer = try RecordingWriter(directory: root, manifest: manifest)
    let surface = SurfaceDescriptor(id: "test", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
    let frame = try FrameArchive.prepare(pixels: Data(repeating: 42, count: 16), metadata: .init(eventNanos: 90, observedNanos: 100, surface: surface, byteCount: 16))
    try writer.append(frame); try writer.flush()
    let manifestURL = root.appendingPathComponent("manifest.json")
    let before = try Data(contentsOf: manifestURL)
    try FileManager.default.removeItem(at: manifestURL)
    try FileManager.default.createDirectory(at: manifestURL, withIntermediateDirectories: false)
    #expect(throws: AstraError.self) { try writer.finish(at: 200, status: .complete) }
    #expect(writer.snapshot.status == .failed)
    #expect(throws: AstraError.self) { try writer.finish(at: 200, status: .complete) }
    #expect(throws: AstraError.self) { try writer.append(frame) }
    try FileManager.default.removeItem(at: manifestURL)
    try before.write(to: manifestURL)
    let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
    #expect(try decoder.decode(RecordingManifest.self, from: before).status == .recording)
    let recovered = try RecordingRecovery.recover(directory: root, expectedID: manifest.id, fallback: nil)
    #expect(recovered.manifest.status == .interrupted)
    #expect(recovered.manifest.frameCount == 1)
    #expect(recovered.manifest.firstInvalidObservedNanos == 100)
}

@Test func recoveryPreservesRawWALAndPartialShardsWhileRestoringUnindexedFramesAndLinks() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let original = try LibraryStore(root: root.appendingPathComponent("live"))
    let restored = try LibraryStore(root: root.appendingPathComponent("restored"))
    let agent = AgentDocument(name: "Recovered agent")
    try await original.save(agent); try await restored.save(agent)
    let manifest = RecordingManifest(name: "Crash fixture", environment: .init(name: "Desktop", kind: .desktop), recordedForAgentID: agent.id)
    let liveDirectory = original.recordingDirectory(id: manifest.id)
    let crashDirectory = restored.recordingDirectory(id: manifest.id)
    let writer = try RecordingWriter(directory: liveDirectory, manifest: manifest)
    try await original.saveRecording(manifest, linkTo: agent.id)
    let surface = SurfaceDescriptor(id: "test", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
    for (index, time) in [UInt64(100), UInt64(300)].enumerated() {
        try writer.append(FrameArchive.prepare(pixels: Data(repeating: UInt8(index), count: 16),
                                             metadata: .init(eventNanos: time - 10, observedNanos: time, surface: surface, byteCount: 16)))
        if index == 0 {
            try writer.append(events: [.init(sequence: 0, eventNanos: 130, observedNanos: 150, origin: .physical, kind: .keyDown, keyCode: 13)])
            try writer.flush()
        }
    }
    // Copy a stable instant with one committed frame/input batch and a complete
    // but unindexed second frame, as they may remain after process termination.
    try FileManager.default.copyItem(at: liveDirectory, to: crashDirectory)
    let shardURL = crashDirectory.appendingPathComponent("frames-00000.astraframes")
    let tail = try FileHandle(forWritingTo: shardURL)
    try tail.seekToEnd(); try tail.write(contentsOf: Data("ASTRAF01partial".utf8)); try tail.close()
    let sourceNames = ["frames-00000.astraframes", "index.sqlite", "index.sqlite-wal", "index.sqlite-shm"]
    var sourceBytes: [String: Data] = [:]
    for name in sourceNames where FileManager.default.fileExists(atPath: crashDirectory.appendingPathComponent(name).path) {
        sourceBytes[name] = try Data(contentsOf: crashDirectory.appendingPathComponent(name))
    }
    #expect(sourceBytes["index.sqlite-wal"]?.isEmpty == false)
    #expect(try await original.recoverInterruptedRecordings().isEmpty)
    #expect(try await original.snapshot().recordings.first?.status == .recording)
    let issues = try await restored.recoverInterruptedRecordings()
    #expect(!issues.isEmpty)
    let recovered = try #require(try await restored.snapshot().recordings.first)
    #expect(recovered.status == .interrupted)
    #expect(recovered.frameCount == 2)
    #expect(recovered.eventCount == 1)
    #expect(recovered.firstInvalidObservedNanos == 150)
    #expect(recovered.stoppedNanos == 300)
    #expect(try await restored.recordingIDs(for: agent.id) == [manifest.id])
    let database = try SQLiteDatabase(url: recovered.indexURL(in: crashDirectory), readOnly: true)
    #expect(try database.query("PRAGMA journal_mode").first?["journal_mode"]?.string == "delete")
    #expect(try database.query("SELECT id FROM frames").count == 2)
    #expect(try database.query("SELECT sequence FROM events").count == 1)
    for (name, before) in sourceBytes { #expect(try Data(contentsOf: crashDirectory.appendingPathComponent(name)) == before) }
    try await restored.recoverInterruptedRecordings()
    #expect(try await restored.snapshot().recordings.first?.indexPath == recovered.indexPath)
    #expect(try FileManager.default.contentsOfDirectory(atPath: crashDirectory.appendingPathComponent("Recovery").path).count == 1)
    _ = try writer.finish(at: 400, status: .complete)
}
