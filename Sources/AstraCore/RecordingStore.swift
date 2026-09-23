import Foundation
import Darwin

public enum RecordingStatus: String, Codable, Sendable { case recording, complete, interrupted, failed }
public struct RecordingManifest: Codable, Hashable, Identifiable, Sendable {
    public var schemaVersion = AstraVersion.dataVersion
    public var id: UUID
    public var name: String
    public var createdAt: Date
    public var environment: EnvironmentDocument
    public var status: RecordingStatus
    public var firstObservedNanos: UInt64?
    public var stoppedNanos: UInt64?
    public var firstInvalidObservedNanos: UInt64?
    public var frameCount: Int
    public var eventCount: Int
    public var storedBytes: UInt64
    public var issue: String?
    public var recordedForAgentID: UUID?
    /// Fixed model-role order for this recording; absent in legacy single-source files.
    public var surfaceIDs: [String]?
    public var correction: CorrectionReference?
    /// Recovery publishes a new index without altering the original source
    /// database or its WAL. Nil is the original `index.sqlite` location.
    public var indexPath: String?
    public var durationSeconds: Double {
        guard let firstObservedNanos, let stoppedNanos, stoppedNanos >= firstObservedNanos else { return 0 }
        return Double(stoppedNanos - firstObservedNanos) / 1_000_000_000
    }
    public init(id: UUID = UUID(), name: String, environment: EnvironmentDocument, recordedForAgentID: UUID? = nil,
                surfaceIDs: [String]? = nil) {
        self.id = id; self.name = name; self.environment = environment; self.recordedForAgentID = recordedForAgentID
        createdAt = Date(); status = .recording; frameCount = 0; eventCount = 0; storedBytes = 0
        self.surfaceIDs = surfaceIDs
    }

    public func validated() throws -> Self {
        var value = self
        value.name = try DocumentNames.validated(name)
        value.environment = try environment.validated()
        if let surfaceIDs {
            guard (1...16).contains(surfaceIDs.count), Set(surfaceIDs).count == surfaceIDs.count,
                  surfaceIDs.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 256 }) else {
                throw AstraError("recording.surfaces", "A recording needs a fixed, unique order of one to sixteen captured surfaces.")
            }
        }
        guard schemaVersion == AstraVersion.dataVersion, createdAt.timeIntervalSince1970.isFinite,
              frameCount >= 0, eventCount >= 0, (issue?.utf8.count ?? 0) <= 65_536,
              [firstObservedNanos, stoppedNanos, firstInvalidObservedNanos].compactMap({ $0 }).allSatisfy({ $0 <= UInt64(Int64.max) }),
              (frameCount == 0) == (firstObservedNanos == nil), (frameCount == 0) == (storedBytes == 0),
              status == .recording ? stoppedNanos == nil : stoppedNanos != nil else {
            throw AstraError("recording.manifest", "The recording manifest is invalid or unsupported.")
        }
        if let firstObservedNanos, let stoppedNanos, stoppedNanos < firstObservedNanos {
            throw AstraError("recording.duration", "The recording ends before its first frame.")
        }
        if status == .complete, frameCount == 0 || firstInvalidObservedNanos != nil || issue != nil {
            throw AstraError("recording.completion", "A complete recording must contain frames and have no invalid interval.")
        }
        _ = try correction?.validated()
        _ = try indexComponents()
        return value
    }

    public func indexURL(in directory: URL) throws -> URL {
        if (try? directory.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true {
            throw AstraError("recording.indexPath", "The recording package cannot be a symbolic link.")
        }
        var result = directory
        for component in try indexComponents() {
            result.appendPathComponent(component)
            if (try? result.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true {
                throw AstraError("recording.indexPath", "A recording index cannot resolve through a symbolic link.")
            }
        }
        return result
    }

    private func indexComponents() throws -> [String] {
        let parts = (indexPath ?? "index.sqlite").split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard parts == ["index.sqlite"] || (parts.count == 3 && parts[0] == "Recovery" && UUID(uuidString: parts[1]) != nil && parts[2] == "index.sqlite") else {
            throw AstraError("recording.indexPath", "The recording index path is invalid.")
        }
        return parts
    }
}

/// Held until sealing or process exit. Recovery never treats a still-running
/// writer as an abandoned recording, including writers in another process.
private final class RecordingAccessLock {
    private let descriptor: Int32
    init(directory: URL) throws {
        descriptor = open(directory.appendingPathComponent(".writer.lock").path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw AstraError("recording.lock", "Cannot open the recording writer lock.") }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            throw AstraError("recording.busy", "This recording still has an active writer.")
        }
    }
    deinit { flock(descriptor, LOCK_UN); close(descriptor) }
}

private enum RecordingFiles {
    static func writeManifest(_ manifest: RecordingManifest, in directory: URL) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .millisecondsSince1970
        let data = try encoder.encode(manifest.validated())
        let destination = directory.appendingPathComponent("manifest.json")
        let staging = directory.appendingPathComponent(".manifest-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: staging) }
        try data.write(to: staging, options: .withoutOverwriting)
        let file = try FileHandle(forWritingTo: staging)
        do { try file.synchronize(); try file.close() }
        catch { try? file.close(); throw error }
        guard rename(staging.path, destination.path) == 0 else {
            throw AstraError("recording.manifestWrite", "Cannot publish the recording manifest.")
        }
        let descriptor = open(directory.path, O_RDONLY | O_CLOEXEC | O_DIRECTORY)
        guard descriptor >= 0 else { throw AstraError("recording.directorySync", "Cannot synchronize the recording directory.") }
        defer { close(descriptor) }
        guard fsync(descriptor) == 0 else { throw AstraError("recording.directorySync", "Cannot synchronize the recording directory.") }
    }

    static func readManifest(in directory: URL) throws -> RecordingManifest {
        let url = directory.appendingPathComponent("manifest.json")
        try requireRegularFile(url)
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size <= 262_144 else { throw AstraError("recording.manifestSize", "The recording manifest exceeds its supported size.") }
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        let manifest = try decoder.decode(RecordingManifest.self, from: Data(contentsOf: url))
        guard manifest.schemaVersion == AstraVersion.dataVersion else {
            throw AstraError("recording.version", "This recording was created by an unsupported Astra version.")
        }
        return try manifest.validated()
    }

    static func createSchema(_ database: SQLiteDatabase) throws {
        try database.transaction {
            try database.execute("CREATE TABLE frames (id TEXT PRIMARY KEY, observed INTEGER NOT NULL, source_time INTEGER NOT NULL, shard TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL, block BLOB NOT NULL)")
            try database.execute("CREATE INDEX frames_observed ON frames(observed,id)")
            try database.execute("CREATE INDEX frames_surface_observed ON frames(json_extract(CAST(block AS TEXT),'$.metadata.surface.id'),observed,id)")
            try database.execute("CREATE TABLE events (sequence INTEGER PRIMARY KEY, observed INTEGER NOT NULL, source_time INTEGER NOT NULL, event BLOB NOT NULL)")
            try database.execute("CREATE INDEX events_observed ON events(observed,sequence)")
            try database.execute("CREATE TABLE health (observed INTEGER NOT NULL, status TEXT NOT NULL, message TEXT)")
            try database.execute("CREATE TABLE coverage (observed INTEGER NOT NULL, surface_id TEXT NOT NULL, frame_id TEXT NOT NULL, proof BLOB NOT NULL)")
            try database.execute("CREATE INDEX coverage_observed ON coverage(observed,surface_id)")
        }
    }

    static func validate(_ event: RawInputEvent) throws {
        guard [event.sequence, event.observedNanos, event.eventNanos].allSatisfy({ $0 <= UInt64(Int64.max) }),
              [event.x, event.y, event.dx, event.dy, event.scrollX, event.scrollY].compactMap({ $0 }).allSatisfy(\.isFinite),
              event.keyCode.map({ (0...127).contains($0) }) ?? true,
              event.button.map({ (0...31).contains($0) }) ?? true,
              (event.rawPlatformData?.count ?? 0) <= 65_536, (event.detail?.utf8.count ?? 0) <= 65_536,
              event.surfaceID.map({ !$0.isEmpty && $0.utf8.count <= 256 }) ?? true else {
            throw AstraError("recording.input", "Input contains invalid values or an oversized native event.")
        }
        switch event.kind {
        case .keyDown, .keyUp, .keyRepeat:
            guard event.keyCode != nil else { throw AstraError("recording.input", "A key event has no key code.") }
        case .buttonDown, .buttonUp:
            guard event.button != nil else { throw AstraError("recording.input", "A button event has no button number.") }
        case .pointer:
            guard event.x != nil, event.y != nil else { throw AstraError("recording.input", "A pointer event has no absolute location.") }
        case .scroll:
            guard event.scrollX != nil, event.scrollY != nil else { throw AstraError("recording.input", "A scroll event has no scroll deltas.") }
        case .flags:
            guard event.modifiers != nil else { throw AstraError("recording.input", "A flags event has no modifier snapshot.") }
        case .gap: break
        }
    }

    static func requireRegularFile(_ url: URL) throws {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw AstraError("recording.file", "A recording source must be a regular file, not a symbolic link.")
        }
    }
}

struct RecordingRecoveryResult {
    var manifest: RecordingManifest
    var issues: [String]
    var recovered: Bool
}

/// Recovery only rewrites the small manifest pointer. Source frame shards and
/// the previous index/WAL remain untouched, including corrupt and partial tails.
enum RecordingRecovery {
    static func recover(directory: URL, expectedID: UUID, fallback: RecordingManifest?) throws -> RecordingRecoveryResult {
        let values = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else {
            throw AstraError("recording.directory", "A recording package must be a directory, not a symbolic link.")
        }
        let access = try RecordingAccessLock(directory: directory)
        return try withExtendedLifetime(access) {
            var manifest: RecordingManifest
            var issues: [String] = []
            do { manifest = try RecordingFiles.readManifest(in: directory) }
            catch {
                if (error as? AstraError)?.code == "recording.version" { throw error }
                guard let fallback else { throw error }
                manifest = try fallback.validated()
                issues.append("The source manifest could not be read; recovery used the matching catalog document.")
            }
            guard manifest.id == expectedID else { throw AstraError("recording.identity", "The recording identity does not match its package name.") }
            let sourceIndex = try manifest.indexURL(in: directory)
            let hasIndex = (try? RecordingFiles.requireRegularFile(sourceIndex)) != nil
            if manifest.status != .recording && issues.isEmpty && hasIndex {
                return RecordingRecoveryResult(manifest: manifest, issues: [], recovered: false)
            }
            let recoveryID = UUID().uuidString
            let recoveryRoot = directory.appendingPathComponent("Recovery", isDirectory: true)
            if (try? recoveryRoot.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true {
                throw AstraError("recording.directory", "The recovery directory cannot be a symbolic link.")
            }
            let destination = recoveryRoot.appendingPathComponent(recoveryID, isDirectory: true)
            try FileManager.default.createDirectory(at: recoveryRoot, withIntermediateDirectories: true)
            guard mkdir(destination.path, S_IRWXU) == 0 else {
                throw AstraError("recording.recoveryDirectory", "Cannot create a new recovery destination.")
            }
            let publishedIndexPath = "Recovery/\(recoveryID)/index.sqlite"
            var published = false
            defer {
                // Failed derived work may be removed; every original remains
                // in place. Keep it if the manifest rename succeeded but its
                // subsequent directory synchronization reported an error.
                if !published, (try? RecordingFiles.readManifest(in: directory).indexPath) != publishedIndexPath {
                    try? FileManager.default.removeItem(at: destination)
                }
            }
            let source = destination.appendingPathComponent("source", isDirectory: true)
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            // Snapshot the original source for diagnosis. SQLite reads only this
            // copy, so its WAL recovery cannot modify original raw input data.
            let originalManifest = directory.appendingPathComponent("manifest.json")
            if (try? RecordingFiles.requireRegularFile(originalManifest)) != nil {
                try FileManager.default.copyItem(at: originalManifest, to: source.appendingPathComponent("manifest.json"))
            }
            for suffix in ["", "-wal", "-shm"] {
                let original = URL(fileURLWithPath: sourceIndex.path + suffix)
                if FileManager.default.fileExists(atPath: original.path) {
                    try RecordingFiles.requireRegularFile(original)
                    try FileManager.default.copyItem(at: original, to: source.appendingPathComponent("index.sqlite" + suffix))
                }
            }
            var rebuilt = try rebuild(directory: directory, source: source, destination: destination, manifest: manifest, initialIssues: issues)
            rebuilt.manifest.indexPath = publishedIndexPath
            rebuilt.manifest = try rebuilt.manifest.validated()
            let report: [String: String] = ["originalIndex": manifest.indexPath ?? "index.sqlite", "issues": rebuilt.issues.joined(separator: "\n")]
            try JSONEncoder().encode(report).write(to: destination.appendingPathComponent("report.json"), options: .atomic)
            // A crash before this rename leaves the previous pointer intact. A
            // retry may create another recovery, but never loses original data.
            try RecordingFiles.writeManifest(rebuilt.manifest, in: directory)
            published = true
            return rebuilt
        }
    }

    private static func rebuild(directory: URL, source: URL, destination: URL, manifest original: RecordingManifest,
                                initialIssues: [String]) throws -> RecordingRecoveryResult {
        let database = try SQLiteDatabase(url: destination.appendingPathComponent("index.sqlite"))
        try RecordingFiles.createSchema(database)
        var manifest = original
        manifest.frameCount = 0; manifest.eventCount = 0; manifest.storedBytes = 0; manifest.firstObservedNanos = nil
        var issues = initialIssues
        var latestObserved: UInt64 = 0
        var inputThrough: UInt64?
        var previousFrameObserved: UInt64?
        func invalid(_ message: String, at time: UInt64 = 0) {
            if issues.count < 128 { issues.append(message) }
            manifest.firstInvalidObservedNanos = min(manifest.firstInvalidObservedNanos ?? time, time)
        }
        let sourceURL = source.appendingPathComponent("index.sqlite")
        if FileManager.default.fileExists(atPath: sourceURL.path) {
            do {
                let previous = try SQLiteDatabase(url: sourceURL, readOnly: true)
                var cursor: Int64?
                while true {
                    let rows = try previous.query("SELECT sequence,observed,source_time,CASE WHEN length(event)<=262144 THEN event ELSE NULL END AS event FROM events \(cursor == nil ? "" : "WHERE sequence>?") ORDER BY sequence LIMIT 512", cursor.map { [.integer($0)] } ?? [])
                    if rows.isEmpty { break }
                    do { try database.transaction {
                        for row in rows {
                            guard let sequence = row["sequence"]?.integer else { throw AstraError("recording.eventIndex", "An input index row has no integer sequence.") }
                            cursor = sequence
                            let data: Data
                            let event: RawInputEvent
                            do {
                                guard let payload = row["event"]?.data, payload.count <= 262_144 else { throw AstraError("recording.eventIndex", "An input index row has missing or oversized data.") }
                                data = payload
                                event = try JSONDecoder().decode(RawInputEvent.self, from: data)
                                try RecordingFiles.validate(event)
                                guard Int64(event.sequence) == sequence, row["observed"]?.integer == Int64(event.observedNanos), row["source_time"]?.integer == Int64(event.eventNanos) else {
                                    throw AstraError("recording.eventIndex", "An input index row disagrees with its raw event.")
                                }
                            } catch { invalid("An invalid input index row was preserved in the source snapshot: \(error.localizedDescription)"); continue }
                            try database.execute("INSERT INTO events VALUES(?,?,?,?)", [.integer(sequence), .integer(Int64(event.observedNanos)), .integer(Int64(event.eventNanos)), .blob(data)])
                            manifest.eventCount += 1; latestObserved = max(latestObserved, event.observedNanos)
                            inputThrough = max(inputThrough ?? 0, event.observedNanos)
                            if event.kind == .gap { invalid("The durable input contains a recorded gap.", at: event.observedNanos) }
                        }
                    } } catch { throw AstraError("recording.recoveryWrite", "The recovery index could not be written: \(error.localizedDescription)") }
                }
                var healthCursor: Int64 = 0
                while true {
                    let rows = try previous.query("SELECT rowid,observed,status,message FROM health WHERE rowid>? ORDER BY rowid LIMIT 512", [.integer(healthCursor)])
                    if rows.isEmpty { break }
                    do { try database.transaction {
                        for row in rows {
                            guard let rowID = row["rowid"]?.integer else { throw AstraError("recording.healthIndex", "A health index row has no identifier.") }
                            healthCursor = rowID
                            guard let observed = row["observed"]?.integer, observed >= 0,
                                  let status = row["status"]?.string, !status.isEmpty, status.utf8.count <= 128,
                                  (row["message"]?.string?.utf8.count ?? 0) <= 65_536 else {
                                invalid("An invalid health entry was preserved in the source snapshot."); continue
                            }
                            try database.execute("INSERT INTO health VALUES(?,?,?)", [.integer(observed), .text(status), row["message"] ?? .null])
                        }
                    } } catch { throw AstraError("recording.recoveryWrite", "The recovery index could not be written: \(error.localizedDescription)") }
                }
            } catch {
                if (error as? AstraError)?.code == "recording.recoveryWrite" { throw error }
                invalid("The previous index could not be fully read: \(error.localizedDescription)")
            }
        } else { invalid("No durable input index was available.") }

        let shards = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            .filter { $0.lastPathComponent.hasPrefix("frames-") && $0.pathExtension == "astraframes" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        for shard in shards {
            let file: FileHandle
            do {
                try RecordingFiles.requireRegularFile(shard)
                file = try FileHandle(forReadingFrom: shard)
            } catch {
                invalid("\(shard.lastPathComponent): \(error.localizedDescription)", at: previousFrameObserved ?? 0)
                continue
            }
            defer { try? file.close() }
            var offset: UInt64 = 0
            try database.transaction {
                while true {
                    let block: FrameBlock
                    do {
                        guard let (value, _) = try FrameArchive.read(from: file, offset: offset) else { break }
                        block = value
                        guard block.metadata.observedNanos <= UInt64(Int64.max), block.metadata.eventNanos <= UInt64(Int64.max),
                              block.offset <= UInt64(Int64.max), block.length <= UInt64(Int64.max) else {
                            throw AstraError("recording.clockRange", "A recovered frame exceeds the indexed storage range.")
                        }
                    } catch {
                        invalid("\(shard.lastPathComponent): \(error.localizedDescription)", at: previousFrameObserved ?? 0)
                        break
                    }
                    if try !database.query("SELECT id FROM frames WHERE id=?", [.text(block.metadata.id.uuidString)]).isEmpty {
                        invalid("A duplicate frame identity was preserved in its original shard.", at: previousFrameObserved ?? 0)
                        break
                    }
                        try database.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?)", [
                            .text(block.metadata.id.uuidString), .integer(Int64(block.metadata.observedNanos)), .integer(Int64(block.metadata.eventNanos)),
                            .text(shard.lastPathComponent), .integer(Int64(block.offset)), .integer(Int64(block.length)), .blob(try JSONEncoder().encode(block))
                        ])
                        if let previousFrameObserved, block.metadata.observedNanos < previousFrameObserved {
                            invalid("Recovered frame availability times are out of order.", at: block.metadata.observedNanos)
                        }
                        previousFrameObserved = block.metadata.observedNanos
                        manifest.firstObservedNanos = min(manifest.firstObservedNanos ?? block.metadata.observedNanos, block.metadata.observedNanos)
                        manifest.frameCount += 1; manifest.storedBytes += block.length
                        latestObserved = max(latestObserved, block.metadata.observedNanos)
                        offset += block.length
                }
            }
        }
        if FileManager.default.fileExists(atPath: sourceURL.path) {
            do {
                let previous = try SQLiteDatabase(url: sourceURL, readOnly: true)
                if !(try previous.query("SELECT name FROM sqlite_master WHERE type='table' AND name='coverage'")).isEmpty {
                    var cursor: Int64 = 0
                    while true {
                        let rows = try previous.query("SELECT rowid,observed,surface_id,frame_id,CASE WHEN length(proof)<=16384 THEN proof ELSE NULL END AS proof FROM coverage WHERE rowid>? ORDER BY rowid LIMIT 512", [.integer(cursor)])
                        if rows.isEmpty { break }
                        for row in rows {
                            guard let rowID = row["rowid"]?.integer else { throw AstraError("recording.coverage", "A coverage row has no identity.") }
                            cursor = rowID
                            do {
                                guard let data = row["proof"]?.data else { throw AstraError("recording.coverage", "Capture proof is missing or oversized.") }
                                let proof = try JSONDecoder().decode(CaptureFrameCoverage.self, from: data)
                                guard proof.verifiedAtNanos <= UInt64(Int64.max), proof.kind == .unchanged,
                                      row["observed"]?.integer == Int64(proof.verifiedAtNanos), row["surface_id"]?.string == proof.surface.id,
                                      row["frame_id"]?.string == proof.frameID.uuidString,
                                      let block = try database.query("SELECT block FROM frames WHERE id=?", [.text(proof.frameID.uuidString)]).first?["block"]?.data else {
                                    throw AstraError("recording.coverage", "Capture proof has no matching recovered source frame.")
                                }
                                _ = try proof.validated(frame: JSONDecoder().decode(FrameBlock.self, from: block).metadata, cutoffNanos: proof.verifiedAtNanos)
                                try database.execute("INSERT INTO coverage VALUES(?,?,?,?)", [.integer(Int64(proof.verifiedAtNanos)), .text(proof.surface.id), .text(proof.frameID.uuidString), .blob(data)])
                                latestObserved = max(latestObserved, proof.verifiedAtNanos)
                            } catch { invalid("Capture coverage could not be recovered: \(error.localizedDescription)", at: UInt64(max(0, row["observed"]?.integer ?? 0))) }
                        }
                    }
                }
            } catch { invalid("The previous capture coverage could not be fully read: \(error.localizedDescription)") }
        }
        // A frame reaching disk does not prove input delivery had caught up.
        // Conservatively exclude the crash tail beyond the last durable event;
        // a future explicit input watermark may establish a stronger bound.
        if manifest.frameCount > 0, latestObserved > (inputThrough ?? 0) {
            invalid("Visual evidence after the last durable input evidence requires exclusion from training.", at: inputThrough ?? manifest.firstObservedNanos ?? 0)
        }
        manifest.stoppedNanos = max(original.stoppedNanos ?? 0, latestObserved)
        manifest.status = manifest.frameCount == 0 ? .failed : .interrupted
        manifest.issue = original.issue ?? "Recovered after an interruption. \(issues.isEmpty ? "The durable source prefix is available." : "Review the recovery report before creating a dataset.")"
        try database.execute("INSERT INTO health VALUES(?,?,?)", [.integer(Int64(latestObserved)), .text("recovered"), .text(manifest.issue!)])
        try database.checkpoint()
        try database.execute("PRAGMA journal_mode=DELETE")
        try database.close()
        return RecordingRecoveryResult(manifest: manifest, issues: issues, recovered: true)
    }
}

public struct StoredFrame: Sendable {
    public let shard: String
    public let block: FrameBlock
}

/// Per-recording source writer. Its owning session serializes writes; the lock
/// also protects finalization from incidental concurrent calls. The global
/// library catalog remains owned by LibraryStore.
public final class RecordingWriter: @unchecked Sendable {
    public let directory: URL
    private let database: SQLiteDatabase
    private let lock = NSRecursiveLock()
    private var manifest: RecordingManifest
    private var file: FileHandle?
    private var shardIndex = 0
    private var pendingFrames: [StoredFrame] = []
    private var pendingEvents: [RawInputEvent] = []
    private var pendingCoverage: [CaptureFrameCoverage] = []
    private var finished = false
    private var writeFailure: String?
    private var finalizationError: AstraError?
    private var access: RecordingAccessLock?
    private var lastEventSequence: UInt64?
    private var latestObservedNanos: UInt64 = 0
    private var latestFrameObservedNanos: UInt64?
    private static let maximumShardBytes: UInt64 = 256 * 1_024 * 1_024

    public init(directory: URL, manifest: RecordingManifest) throws {
        let value = try manifest.validated()
        guard value.status == .recording, value.frameCount == 0, value.eventCount == 0,
              value.firstInvalidObservedNanos == nil, value.issue == nil, value.indexPath == nil else {
            throw AstraError("recording.initialState", "A new recording must begin with an empty manifest.")
        }
        self.directory = directory; self.manifest = value
        try FileManager.default.createDirectory(at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(directory.path, S_IRWXU) == 0 else {
            throw AstraError("recording.exists", "A recording with this identifier already exists.")
        }
        access = try RecordingAccessLock(directory: directory)
        try RecordingFiles.writeManifest(value, in: directory)
        database = try SQLiteDatabase(url: directory.appendingPathComponent("index.sqlite"))
        try RecordingFiles.createSchema(database)
        try openShard()
    }

    deinit {
        try? file?.close()
        try? database.close()
        access = nil
    }

    public var snapshot: RecordingManifest { lock.withLock { manifest } }

    public func attachCorrection(_ seed: CorrectionRecordingSeed, supervisionStartNanos: UInt64) throws {
        try lock.withLock {
            try requireAccepting()
            guard manifest.correction == nil, manifest.frameCount == 0, manifest.eventCount == 0, pendingEvents.isEmpty else {
                throw AstraError("correction.started", "Correction provenance must be attached before expert observation starts.")
            }
            manifest.correction = try CorrectionPrelude.write(seed, supervisionStartNanos: supervisionStartNanos, in: directory)
            latestObservedNanos = max(latestObservedNanos, supervisionStartNanos)
            try RecordingFiles.writeManifest(manifest, in: directory)
        }
    }

    public func append(_ frame: PreparedFrame) throws {
        try lock.withLock {
            try requireAccepting()
            _ = try integer(frame.metadata.observedNanos); _ = try integer(frame.metadata.eventNanos)
            guard manifest.surfaceIDs.map({ $0.contains(frame.metadata.surface.id) }) ?? true else {
                throw AstraError("recording.surface", "The frame belongs to an unbound recording surface.")
            }
            guard latestFrameObservedNanos.map({ frame.metadata.observedNanos >= $0 }) ?? true,
                  !pendingFrames.contains(where: { $0.block.metadata.id == frame.metadata.id }),
                  try database.query("SELECT id FROM frames WHERE id=?", [.text(frame.metadata.id.uuidString)]).isEmpty else {
                throw AstraError("recording.frameOrder", "A frame identity was duplicated or its observation time was reordered.")
            }
            do {
            guard var file else { throw AstraError("recording.closed", "The frame archive is closed.") }
            if try file.offset() > 0, try file.offset() + UInt64(frame.byteCount) > Self.maximumShardBytes {
                try flush()
                try file.close(); self.file = nil; shardIndex += 1
                try openShard()
                guard let next = self.file else { throw AstraError("recording.closed", "The next frame archive could not open.") }
                file = next
            }
            let block = try FrameArchive.appendPrepared(frame, to: file)
            pendingFrames.append(StoredFrame(shard: shardName, block: block))
            if manifest.firstObservedNanos == nil { manifest.firstObservedNanos = frame.metadata.observedNanos }
            latestFrameObservedNanos = frame.metadata.observedNanos
            latestObservedNanos = max(latestObservedNanos, frame.metadata.observedNanos)
            manifest.frameCount += 1; manifest.storedBytes += block.length
            } catch {
                recordFailure(error, at: frame.metadata.observedNanos)
                throw error
            }
        }
    }

    public func append(events: [RawInputEvent]) throws {
        try lock.withLock {
            try requireAccepting()
            var sequence = lastEventSequence
            for event in events {
                guard sequence.map({ event.sequence > $0 }) ?? true else {
                    throw AstraError("recording.eventOrder", "An input sequence was duplicated or reordered.")
                }
                try RecordingFiles.validate(event)
                sequence = event.sequence
            }
            // Reject the whole batch before mutating counters or sequence state.
            lastEventSequence = sequence; pendingEvents.append(contentsOf: events); manifest.eventCount += events.count
            latestObservedNanos = max(latestObservedNanos, events.map(\.observedNanos).max() ?? 0)
        }
    }

    public func append(coverage: CaptureFrameCoverage) throws {
        try lock.withLock {
            try requireAccepting()
            guard pendingCoverage.count < 4_096, coverage.kind == .unchanged,
                  [coverage.eventNanos, coverage.observedNanos, coverage.throughNanos, coverage.verifiedAtNanos]
                    .allSatisfy({ $0 <= UInt64(Int64.max) }),
                  coverage.eventNanos <= coverage.observedNanos, coverage.observedNanos <= coverage.throughNanos,
                  coverage.throughNanos <= coverage.verifiedAtNanos,
                  manifest.surfaceIDs.map({ $0.contains(coverage.surface.id) }) ?? true else {
                throw AstraError("recording.coverage", "Capture coverage is invalid or its bounded recording queue is full.")
            }
            _ = try coverage.surface.validated()
            pendingCoverage.append(coverage)
            latestObservedNanos = max(latestObservedNanos, coverage.verifiedAtNanos)
        }
    }

    public func health(observedNanos: UInt64, status: String, message: String? = nil) throws {
        try lock.withLock {
            try requireAccepting()
            guard !status.isEmpty, status.utf8.count <= 128, (message?.utf8.count ?? 0) <= 65_536 else {
                throw AstraError("recording.health", "The recording health entry is invalid.")
            }
            do { try database.execute("INSERT INTO health VALUES(?,?,?)", [try integer(observedNanos), .text(status), message.map(SQLValue.text) ?? .null]) }
            catch { recordFailure(error, at: observedNanos); throw error }
        }
    }

    public func markInvalid(from observedNanos: UInt64, message: String) {
        lock.withLock {
            guard !finished else { return }
            manifest.firstInvalidObservedNanos = min(manifest.firstInvalidObservedNanos ?? observedNanos, observedNanos)
            manifest.issue = manifest.issue ?? String(message.prefix(8_192))
        }
    }

    public func flush() throws {
        try lock.withLock {
            try requireOpen()
            do {
            // Index references become durable only after their frame bytes.
            try file?.synchronize()
            var remainingCoverage: [CaptureFrameCoverage] = []
            try database.transaction {
                for frame in pendingFrames {
                    try database.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?)", [
                        .text(frame.block.metadata.id.uuidString), try integer(frame.block.metadata.observedNanos),
                        try integer(frame.block.metadata.eventNanos), .text(frame.shard),
                        try integer(frame.block.offset), try integer(frame.block.length), .blob(try JSONEncoder().encode(frame.block))
                    ])
                }
                for event in pendingEvents {
                    try database.execute("INSERT INTO events VALUES(?,?,?,?)", [
                        try integer(event.sequence), try integer(event.observedNanos), try integer(event.eventNanos),
                        .blob(try JSONEncoder().encode(event))
                    ])
                }
                for proof in pendingCoverage {
                    // Coverage can arrive while its immutable frame is still
                    // compressing. Publish only after that frame is durable.
                    guard let data = try database.query("SELECT block FROM frames WHERE id=?", [.text(proof.frameID.uuidString)]).first?["block"]?.data else {
                        remainingCoverage.append(proof); continue
                    }
                    let frame = try JSONDecoder().decode(FrameBlock.self, from: data).metadata
                    _ = try proof.validated(frame: frame, cutoffNanos: proof.verifiedAtNanos)
                    try database.execute("INSERT INTO coverage VALUES(?,?,?,?)", [try integer(proof.verifiedAtNanos),
                        .text(proof.surface.id), .text(proof.frameID.uuidString), .blob(try JSONEncoder().encode(proof))])
                }
            }
            pendingFrames.removeAll(keepingCapacity: true); pendingEvents.removeAll(keepingCapacity: true)
            pendingCoverage = remainingCoverage
            try RecordingFiles.writeManifest(manifest, in: directory)
            } catch { recordFailure(error, at: latestObservedNanos); throw error }
        }
    }

    public func finish(at time: UInt64, status: RecordingStatus, issue: String? = nil) throws -> RecordingManifest {
        try lock.withLock {
            if let finalizationError { throw finalizationError }
            if finished { return manifest }
            guard status != .recording, time <= UInt64(Int64.max), time >= latestObservedNanos else {
                throw AstraError("recording.finish", "A recording must end after its last accepted observation with a terminal status.")
            }
            do {
                // The on-disk manifest remains recoverable until every frame,
                // event and index transaction is durable and the archive closed.
                try flush()
                guard pendingCoverage.isEmpty else { throw AstraError("recording.coverage", "Capture proof references a frame that never reached storage.") }
                try file?.close(); file = nil
                try database.checkpoint()
                try database.execute("PRAGMA journal_mode=DELETE")
                try database.close()
                var sealed = manifest
                sealed.stoppedNanos = time
                sealed.issue = (issue ?? manifest.issue ?? writeFailure).map { String($0.prefix(8_192)) }
                if sealed.frameCount == 0 {
                    sealed.status = .failed
                    sealed.issue = sealed.issue ?? "No complete screen frames were received."
                } else if status == .complete && (sealed.firstInvalidObservedNanos != nil || sealed.issue != nil) {
                    sealed.status = .interrupted
                } else { sealed.status = status }
                sealed = try sealed.validated()
                try RecordingFiles.writeManifest(sealed, in: directory)
                manifest = sealed; finished = true; access = nil
                return sealed
            } catch {
                recordFailure(error, at: latestObservedNanos)
                manifest.status = .failed; manifest.stoppedNanos = time
                finalizationError = AstraError("recording.finalization", "The recording could not be sealed: \(error.localizedDescription) Its source files are preserved for recovery.")
                finished = true
                try? file?.close(); file = nil
                try? database.close(); access = nil
                throw finalizationError!
            }
        }
    }

    private var shardName: String { String(format: "frames-%05d.astraframes", shardIndex) }
    private func openShard() throws {
        let url = directory.appendingPathComponent(shardName)
        let descriptor = open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw AstraError("recording.shard", "A new recording shard could not be created.")
        }
        file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    }
    private func requireOpen() throws {
        guard !finished else { throw AstraError("recording.closed", "This recording has already been finalized.") }
    }
    private func requireAccepting() throws {
        try requireOpen()
        if let writeFailure { throw AstraError("recording.writeFailed", "Recording stopped after a storage failure: \(writeFailure)") }
    }
    private func recordFailure(_ error: any Error, at time: UInt64) {
        writeFailure = writeFailure ?? String(error.localizedDescription.prefix(8_192))
        markInvalid(from: time, message: writeFailure!)
    }
    private func integer(_ value: UInt64) throws -> SQLValue {
        guard value <= UInt64(Int64.max) else { throw AstraError("recording.clockRange", "The recording clock exceeds the indexed storage range.") }
        return .integer(Int64(value))
    }
}
