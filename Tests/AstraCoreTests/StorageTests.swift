import Foundation
import Testing
@testable import AstraCore

@Test func databaseRollbackAndBackupPreserveConsistentRecords() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let database = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    try database.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, content BLOB NOT NULL)")
    try database.execute("INSERT INTO items VALUES (?, ?)", [.integer(1), .blob(Data([0, 1, 255]))])
    #expect(throws: AstraError.self) {
        try database.transaction {
            try database.execute("INSERT INTO items VALUES (?, ?)", [.integer(2), .blob(Data())])
            throw AstraError("test.failure", "Simulated publication failure")
        }
    }
    #expect(try database.query("SELECT id FROM items").count == 1)
    let backup = root.appendingPathComponent("backup.sqlite")
    try database.backup(to: backup)
    let restored = try SQLiteDatabase(url: backup)
    #expect(try restored.query("SELECT content FROM items").first?["content"]?.data == Data([0, 1, 255]))
}

@Test func losslessFramesRoundTripAndRecoverBeforePartialTail() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(FileManager.default.createFile(atPath: url.path, contents: nil))
    let writer = try FileHandle(forWritingTo: url)
    let pixels = Data((0..<(16 * 12 * 4)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
    let surface = SurfaceDescriptor(id: "source", globalBounds: .init(x: -80, y: 40, width: 8, height: 6),
                                    pixelWidth: 16, pixelHeight: 12)
    let metadata = FrameMetadata(eventNanos: 3, observedNanos: 9, surface: surface, byteCount: pixels.count)
    let block = try FrameArchive.append(pixels: pixels, metadata: metadata, to: writer)
    try writer.synchronize()
    let reader = try FileHandle(forReadingFrom: url)
    #expect(try FrameArchive.read(from: reader, offset: 0)?.1 == pixels)
    try reader.close()
    try writer.write(contentsOf: Data("ASTRAF01unfinished".utf8))
    try writer.close()
    let before = try Data(contentsOf: url)
    let recovered = try FrameArchive.scan(url: url)
    #expect(recovered.blocks.count == 1)
    #expect(recovered.validBytes == block.length)
    #expect(recovered.incompleteTail)
    #expect(try Data(contentsOf: url) == before)
}

@Test func corruptedFrameIsRejectedBeforeTraining() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(FileManager.default.createFile(atPath: url.path, contents: nil))
    let writer = try FileHandle(forWritingTo: url)
    let surface = SurfaceDescriptor(id: "source", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
    _ = try FrameArchive.append(pixels: Data(repeating: 17, count: 16),
                                metadata: .init(eventNanos: 1, observedNanos: 2, surface: surface, byteCount: 16), to: writer)
    try writer.close()
    var data = try Data(contentsOf: url); data[data.count - 1] ^= 1
    try data.write(to: url)
    let result = try FrameArchive.scan(url: url)
    #expect(result.blocks.isEmpty)
    #expect(result.incompleteTail)
    #expect(result.error?.contains("integrity") == true)
}
