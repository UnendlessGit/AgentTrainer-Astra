import Foundation
import Darwin

public struct RecordingInspection: Sendable {
    public let manifest: RecordingManifest
    public let firstFrameNanos: UInt64?
    public let lastFrameNanos: UInt64?
    public let surfaceIDs: [String]
}

public struct RecordingPreview: Sendable {
    public let frame: FrameMetadata
    public let pixels: Data
    public let events: [RawInputEvent]
    public let moreEvents: Bool
}

/// Read-only inspection of a sealed recording. A shared package lock excludes
/// recovery, and every displayed frame is decoded and checked against its index.
public final class RecordingReader: @unchecked Sendable {
    public let directory: URL
    public let manifest: RecordingManifest
    private let database: SQLiteDatabase
    private let access: Int32
    private let lock = NSLock()

    public init(directory: URL) throws {
        self.directory = directory
        let values = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else {
            throw AstraError("recording.package", "A recording must be a local package directory.")
        }
        let descriptor = open(directory.appendingPathComponent(".writer.lock").path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard descriptor >= 0 else { throw AstraError("recording.lock", "The recording's ownership file is missing.") }
        do {
            guard flock(descriptor, LOCK_SH | LOCK_NB) == 0 else {
                throw AstraError("recording.busy", "This recording is still being saved or recovered.")
            }
            let url = directory.appendingPathComponent("manifest.json")
            let properties = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard properties.isRegularFile == true, properties.isSymbolicLink != true, (properties.fileSize ?? .max) <= 262_144 else {
                throw AstraError("recording.manifest", "The recording manifest is missing, linked or oversized.")
            }
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
            let manifest = try decoder.decode(RecordingManifest.self, from: Data(contentsOf: url)).validated()
            guard manifest.status != .recording, UUID(uuidString: directory.deletingPathExtension().lastPathComponent) == manifest.id else {
                throw AstraError("recording.identity", "The recording is unsealed or its package identity has changed.")
            }
            let database = try SQLiteDatabase(url: manifest.indexURL(in: directory), readOnly: true)
            guard try database.query("PRAGMA quick_check").first?.values.first == .text("ok") else {
                throw AstraError("recording.index", "The recording index failed its integrity check.")
            }
            let counts = try database.query("SELECT (SELECT COUNT(*) FROM frames) AS frames,(SELECT COUNT(*) FROM events) AS events").first
            guard counts?["frames"]?.integer == Int64(manifest.frameCount), counts?["events"]?.integer == Int64(manifest.eventCount) else {
                throw AstraError("recording.indexCount", "The recording index does not match its sealed manifest.")
            }
            self.manifest = manifest; self.database = database; self.access = descriptor
        } catch {
            flock(descriptor, LOCK_UN); close(descriptor)
            throw error
        }
    }

    deinit { try? database.close(); flock(access, LOCK_UN); close(access) }

    public func inspect() throws -> RecordingInspection {
        try lock.withLock {
            let row = try database.query("SELECT MIN(observed) AS first,MAX(observed) AS last FROM frames").first
            return RecordingInspection(manifest: manifest, firstFrameNanos: row?["first"]?.integer.flatMap(UInt64.init(exactly:)),
                                       lastFrameNanos: row?["last"]?.integer.flatMap(UInt64.init(exactly:)),
                                       surfaceIDs: try manifest.surfaceIDs ?? database.query("SELECT DISTINCT json_extract(CAST(block AS TEXT),'$.metadata.surface.id') AS surface FROM frames ORDER BY surface").compactMap { $0["surface"]?.string })
        }
    }

    public func preview(at observedNanos: UInt64, surfaceID: String? = nil, eventRadiusNanos: UInt64 = 100_000_000) throws -> RecordingPreview? {
        try lock.withLock {
            guard observedNanos <= UInt64(Int64.max), eventRadiusNanos <= 5_000_000_000 else {
                throw AstraError("recording.previewTime", "The recording preview interval is outside supported bounds.")
            }
            let predicate = surfaceID == nil ? "" : " AND json_extract(CAST(block AS TEXT),'$.metadata.surface.id')=?"
            let arguments: [SQLValue] = [.integer(Int64(observedNanos))] + (surfaceID.map { [.text($0)] } ?? [])
            guard let row = try database.query("SELECT id,observed,source_time,shard,offset,length,CASE WHEN length(block)<=65536 THEN block ELSE NULL END AS block FROM frames WHERE observed<=?\(predicate) ORDER BY observed DESC,id DESC LIMIT 1", arguments).first else { return nil }
            guard let bytes = row["block"]?.data, let shard = row["shard"]?.string,
                  shard.range(of: #"^frames-[0-9]{5,}\.astraframes$"#, options: .regularExpression) != nil else {
                throw AstraError("recording.previewIndex", "The preview index contains an invalid block or shard path.")
            }
            let block = try JSONDecoder().decode(FrameBlock.self, from: bytes)
            _ = try block.metadata.validated()
            guard row["id"]?.string.flatMap(UUID.init(uuidString:)) == block.metadata.id,
                  row["observed"]?.integer.flatMap(UInt64.init(exactly:)) == block.metadata.observedNanos,
                  row["source_time"]?.integer.flatMap(UInt64.init(exactly:)) == block.metadata.eventNanos,
                  row["offset"]?.integer.flatMap(UInt64.init(exactly:)) == block.offset,
                  row["length"]?.integer.flatMap(UInt64.init(exactly:)) == block.length,
                  surfaceID == nil || block.metadata.surface.id == surfaceID else {
                throw AstraError("recording.previewIdentity", "The preview frame disagrees with its indexed time or identity.")
            }
            let descriptor = open(directory.appendingPathComponent(shard).path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            guard descriptor >= 0 else { throw AstraError("recording.previewFile", "The preview frame archive is missing or linked.") }
            let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
            defer { try? file.close() }
            guard let (decoded, pixels) = try FrameArchive.read(from: file, offset: block.offset), decoded == block else {
                throw AstraError("recording.previewMismatch", "The decoded frame does not match its saved index.")
            }
            let lower = observedNanos >= eventRadiusNanos ? observedNanos - eventRadiusNanos : 0
            let upper = min(UInt64(Int64.max), observedNanos + eventRadiusNanos)
            let rows = try database.query("SELECT sequence,observed,source_time,CASE WHEN length(event)<=262144 THEN event ELSE NULL END AS event FROM events WHERE observed>=? AND observed<=? ORDER BY observed,sequence LIMIT 257", [.integer(Int64(lower)), .integer(Int64(upper))])
            let events = try rows.prefix(256).map { row -> RawInputEvent in
                guard let bytes = row["event"]?.data else { throw AstraError("recording.previewEvent", "An input event exceeds its supported size.") }
                let event = try JSONDecoder().decode(RawInputEvent.self, from: bytes)
                guard row["sequence"]?.integer.flatMap(UInt64.init(exactly:)) == event.sequence,
                      row["observed"]?.integer.flatMap(UInt64.init(exactly:)) == event.observedNanos,
                      row["source_time"]?.integer.flatMap(UInt64.init(exactly:)) == event.eventNanos else {
                    throw AstraError("recording.previewEventIdentity", "A recorded input event disagrees with its index.")
                }
                return event
            }
            return RecordingPreview(frame: decoded.metadata, pixels: pixels, events: events, moreEvents: rows.count > 256)
        }
    }
}
