import Foundation

public struct AgentDocument: Codable, Hashable, Identifiable, Sendable {
    public var schemaVersion = AstraVersion.dataVersion
    public var id: UUID
    public var name: String
    public var notes: String
    public var createdAt: Date
    public var modifiedAt: Date
    public var environmentID: UUID?
    public var selectedCheckpointID: UUID?
    public var pinned: Bool

    public init(id: UUID = UUID(), name: String, notes: String = "", createdAt: Date = Date(),
                environmentID: UUID? = nil, selectedCheckpointID: UUID? = nil, pinned: Bool = false) {
        self.id = id; self.name = name; self.notes = notes; self.createdAt = createdAt; modifiedAt = createdAt
        self.environmentID = environmentID; self.selectedCheckpointID = selectedCheckpointID; self.pinned = pinned
    }
    public func validated() throws -> Self {
        var copy = self
        copy.name = try DocumentNames.validated(name)
        guard schemaVersion == AstraVersion.dataVersion, notes.utf8.count <= 65_536,
              createdAt.timeIntervalSince1970.isFinite, modifiedAt.timeIntervalSince1970.isFinite else {
            throw AstraError("library.agent", "The agent document is unsupported or invalid.")
        }
        return copy
    }
}

public enum TargetKind: String, Codable, CaseIterable, Sendable { case display, window, application, desktop, practice }
public struct EnvironmentDocument: Codable, Hashable, Identifiable, Sendable {
    public var schemaVersion = AstraVersion.dataVersion
    public var id: UUID
    public var name: String
    public var kind: TargetKind
    public var displayID: UInt32?
    public var windowID: UInt32?
    public var applicationBundleID: String?
    public var region: Rect2D?
    public var captureFPS: Int
    public var showsCursor: Bool
    public var capabilities: ActionCapabilities
    public var createdAt: Date

    public init(id: UUID = UUID(), name: String, kind: TargetKind, displayID: UInt32? = nil,
                windowID: UInt32? = nil, applicationBundleID: String? = nil, region: Rect2D? = nil,
                captureFPS: Int = 30, showsCursor: Bool = false, capabilities: ActionCapabilities = .init()) {
        self.id = id; self.name = name; self.kind = kind; self.displayID = displayID; self.windowID = windowID
        self.applicationBundleID = applicationBundleID; self.region = region; self.captureFPS = captureFPS
        self.showsCursor = showsCursor; self.capabilities = capabilities; self.createdAt = Date()
    }
    public func validated() throws -> Self {
        var copy = self; copy.name = try DocumentNames.validated(name)
        _ = try capabilities.validated()
        guard schemaVersion == AstraVersion.dataVersion, (1...120).contains(captureFPS),
              region.map(\.isValid) ?? true, createdAt.timeIntervalSince1970.isFinite else {
            throw AstraError("library.environment", "The environment settings are invalid or unsupported.")
        }
        return copy
    }
}

public struct LibraryIssue: Identifiable, Sendable {
    public let id: String
    public let collection: String
    public let message: String
}
public struct LibrarySnapshot: Sendable {
    public var agents: [AgentDocument]
    public var environments: [EnvironmentDocument]
    public var recordings: [RecordingManifest]
    public var issues: [LibraryIssue]
}

public enum DocumentNames {
    public static func validated(_ name: String) throws -> String {
        let value = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value.count <= 160,
              !value.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) else {
            throw AstraError("library.name", "Use a name with 1–160 visible characters.")
        }
        return value
    }
}

public actor LibraryStore {
    public nonisolated let root: URL
    private let database: SQLiteDatabase
    private var recoveryIssues: [LibraryIssue] = []

    public init(root: URL) throws {
        self.root = root.standardizedFileURL
        database = try SQLiteDatabase(url: self.root.appendingPathComponent("library.sqlite"))
        try database.transaction {
            try database.execute("CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL)")
            let versions = try database.query("SELECT version FROM schema_info")
            if versions.isEmpty { try database.execute("INSERT INTO schema_info VALUES (?)", [.integer(1)]) }
            else if versions.count != 1 || versions.first?["version"]?.integer != 1 {
                throw AstraError("library.version", "This library was created by an unsupported Astra version.")
            }
            try database.execute("CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS environments (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
            try database.execute("CREATE TABLE IF NOT EXISTS recordings (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS agent_recordings (agent_id TEXT NOT NULL REFERENCES agents(id), recording_id TEXT NOT NULL REFERENCES recordings(id), PRIMARY KEY(agent_id,recording_id))")
        }
        for folder in ["Recordings", "Models", "Jobs", "Caches", "Logs"] {
            try FileManager.default.createDirectory(at: self.root.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
    }

    public func snapshot() throws -> LibrarySnapshot {
        var issues = recoveryIssues
        let agents: [AgentDocument] = try documents(table: "agents", issues: &issues).compactMap { (document: AgentDocument) -> AgentDocument? in
            do { return try document.validated() }
            catch { issues.append(.init(id: document.id.uuidString, collection: "agents", message: error.localizedDescription)); return nil }
        }
        let environments: [EnvironmentDocument] = try documents(table: "environments", issues: &issues).compactMap { (document: EnvironmentDocument) -> EnvironmentDocument? in
            do { return try document.validated() }
            catch { issues.append(.init(id: document.id.uuidString, collection: "environments", message: error.localizedDescription)); return nil }
        }
        let recordings: [RecordingManifest] = try documents(table: "recordings", issues: &issues).compactMap { (document: RecordingManifest) -> RecordingManifest? in
            do { return try document.validated() }
            catch { issues.append(.init(id: document.id.uuidString, collection: "recordings", message: error.localizedDescription)); return nil }
        }
        return LibrarySnapshot(agents: agents, environments: environments, recordings: recordings, issues: issues)
    }

    public func save(_ document: AgentDocument) throws {
        let value = try document.validated()
        try database.execute("INSERT INTO agents(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
            .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
        ])
    }
    public func save(_ document: EnvironmentDocument) throws {
        let value = try document.validated()
        try database.execute("INSERT INTO environments(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
            .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
        ])
    }

    public nonisolated func recordingDirectory(id: UUID) -> URL {
        root.appendingPathComponent("Recordings", isDirectory: true)
            .appendingPathComponent(id.uuidString + ".astrarecord", isDirectory: true)
    }

    public func saveRecording(_ manifest: RecordingManifest, linkTo agentID: UUID? = nil) throws {
        let value = try manifest.validated()
        try database.transaction {
            try database.execute("INSERT INTO recordings(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
                .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
            ])
            if let agentID {
                try database.execute("INSERT OR IGNORE INTO agent_recordings VALUES(?,?)", [.text(agentID.uuidString), .text(value.id.uuidString)])
            }
        }
    }

    /// Call once when opening the workspace, before starting new jobs. Sealed
    /// packages only need their manifest read; abandoned writers are recovered
    /// under an exclusive package lock. Snapshot does not scan the filesystem.
    @discardableResult
    public func recoverInterruptedRecordings() throws -> [LibraryIssue] {
        var issues: [LibraryIssue] = []
        let documents: [RecordingManifest] = try documents(table: "recordings", includeArchived: true, issues: &issues)
        let catalog = Dictionary(uniqueKeysWithValues: documents.map { ($0.id, $0) })
        let packages = try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent("Recordings"),
                                                                 includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        var seen: Set<UUID> = []
        for directory in packages.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where directory.pathExtension == "astrarecord" {
            guard let id = UUID(uuidString: directory.deletingPathExtension().lastPathComponent) else {
                issues.append(.init(id: directory.lastPathComponent, collection: "recordings", message: "A recording package has an unrecognized name and was left untouched."))
                continue
            }
            seen.insert(id)
            do {
                let result = try RecordingRecovery.recover(directory: directory, expectedID: id, fallback: catalog[id])
                var agentID = result.manifest.recordedForAgentID
                if let candidate = agentID, try database.query("SELECT id FROM agents WHERE id=?", [.text(candidate.uuidString)]).isEmpty {
                    agentID = nil
                    issues.append(.init(id: id.uuidString + ".agent", collection: "recordings", message: "The recording's original agent is unavailable; the shared source remains in the library."))
                }
                try saveRecording(result.manifest, linkTo: agentID)
                if result.recovered || result.manifest.status == .failed || result.manifest.status == .interrupted {
                    let detail = result.issues.first ?? result.manifest.issue ?? "The interrupted recording's durable source prefix was recovered."
                    issues.append(.init(id: id.uuidString + ".recovery", collection: "recordings", message: "\(result.manifest.name): \(detail)"))
                }
            } catch {
                if (error as? AstraError)?.code == "recording.busy" { continue }
                issues.append(.init(id: id.uuidString + ".recovery", collection: "recordings", message: error.localizedDescription))
            }
        }
        for document in documents where !seen.contains(document.id) {
            issues.append(.init(id: document.id.uuidString + ".missing", collection: "recordings", message: "The source package for \(document.name) is unavailable. Its catalog entry and agent links were preserved."))
        }
        recoveryIssues = issues
        return issues
    }

    public func recordingIDs(for agentID: UUID) throws -> Set<UUID> {
        Set(try database.query("SELECT recording_id FROM agent_recordings WHERE agent_id=?", [.text(agentID.uuidString)])
            .compactMap { $0["recording_id"]?.string.flatMap(UUID.init(uuidString:)) })
    }

    /// Archiving is reversible and leaves shared artifacts untouched.
    public func archiveAgent(id: UUID, archived: Bool) throws {
        try database.execute("UPDATE agents SET archived=? WHERE id=?", [.integer(archived ? 1 : 0), .text(id.uuidString)])
    }

    public func duplicateAgent(_ source: AgentDocument) throws -> AgentDocument {
        var copy = source
        copy.id = UUID(); copy.name = String(source.name.prefix(150)) + " copy"
        copy.createdAt = Date(); copy.modifiedAt = copy.createdAt; copy.pinned = false
        try save(copy)
        return copy
    }

    public func checkpoint() throws { try database.checkpoint() }

    private func documents<T: Decodable & Identifiable>(table: String, includeArchived: Bool = false, issues: inout [LibraryIssue]) throws -> [T] where T.ID == UUID {
        // Table is selected exclusively by private call sites above.
        let rows = try database.query("SELECT id,document FROM \(table) \(includeArchived ? "" : "WHERE archived=0") ORDER BY created DESC,id ASC")
        var identities: Set<UUID> = []
        return rows.compactMap { row in
            do {
                guard let data = row["document"]?.data else { throw AstraError("library.document", "An item has no document data.") }
                let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
                let value = try decoder.decode(T.self, from: data)
                guard row["id"]?.string.flatMap(UUID.init(uuidString:)) == value.id else {
                    throw AstraError("library.identity", "An item's document identity does not match its catalog entry.")
                }
                guard identities.insert(value.id).inserted else {
                    throw AstraError("library.identity", "A duplicate document identity was preserved as a catalog issue.")
                }
                return value
            } catch {
                issues.append(.init(id: row["id"]?.string ?? UUID().uuidString, collection: table, message: error.localizedDescription))
                return nil
            }
        }
    }

    private func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return try encoder.encode(value)
    }
}
