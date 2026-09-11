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
    public var rewardProgramID: UUID?
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
    public var learningRuns: [LearningRunDocument]
    public var checkpoints: [CheckpointDocument]
    public var issues: [LibraryIssue]
    public var rewardPrograms: [RewardProgram] = []
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
    private var inferenceIssues: [LibraryIssue] = []

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
            try database.execute("CREATE TABLE IF NOT EXISTS learning_runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS agent_checkpoints (agent_id TEXT NOT NULL REFERENCES agents(id), checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id), PRIMARY KEY(agent_id,checkpoint_id))")
            try database.execute("CREATE TABLE IF NOT EXISTS reward_programs (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
        }
        for folder in ["Recordings", "Models", "Datasets", "Jobs", "Caches", "Logs"] {
            try FileManager.default.createDirectory(at: self.root.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
    }

    public func snapshot() throws -> LibrarySnapshot {
        var issues = recoveryIssues + inferenceIssues
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
        let runs: [LearningRunDocument] = try documents(table: "learning_runs", issues: &issues).compactMap { (value: LearningRunDocument) -> LearningRunDocument? in
            do { return try value.validated() }
            catch { issues.append(.init(id: value.id.uuidString, collection: "learning_runs", message: error.localizedDescription)); return nil }
        }
        let checkpoints: [CheckpointDocument] = try documents(table: "checkpoints", issues: &issues).compactMap { (value: CheckpointDocument) -> CheckpointDocument? in
            do { return try value.validated() }
            catch { issues.append(.init(id: value.id.uuidString, collection: "checkpoints", message: error.localizedDescription)); return nil }
        }
        let rewards: [RewardProgram] = try documents(table: "reward_programs", issues: &issues).compactMap { (value: RewardProgram) -> RewardProgram? in
            do { return try value.validated() }
            catch { issues.append(.init(id: value.id.uuidString, collection: "reward_programs", message: error.localizedDescription)); return nil }
        }
        return LibrarySnapshot(agents: agents, environments: environments, recordings: recordings, learningRuns: runs, checkpoints: checkpoints, issues: issues, rewardPrograms: rewards)
    }

    public func saveRewardProgram(_ document: RewardProgram, for agentID: UUID) throws {
        let value = try document.validated()
        for digest in Set(value.signals.compactMap(\.templateDigest)) { _ = try RewardAssets.read(digest, root: root) }
        try database.transaction {
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
            guard let agentBytes = try database.query("SELECT document FROM agents WHERE id=?", [.text(agentID.uuidString)]).first?["document"]?.data else {
                throw AstraError("reward.agent", "The agent was removed before its reward definition could be saved.")
            }
            var agent = try decoder.decode(AgentDocument.self, from: agentBytes).validated()
            if let bytes = try database.query("SELECT document FROM reward_programs WHERE id=?", [.text(value.id.uuidString)]).first?["document"]?.data {
                guard try decoder.decode(RewardProgram.self, from: bytes) == value else {
                    throw AstraError("reward.immutable", "Saved reward definitions cannot change. Save the edit as a new definition.")
                }
            } else {
                try database.execute("INSERT INTO reward_programs(id,name,document,created) VALUES(?,?,?,?)", [
                    .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(Date().timeIntervalSince1970)])
            }
            agent.rewardProgramID = value.id; agent.modifiedAt = Date()
            try save(agent)
        }
    }

    public func saveLearningRun(_ document: LearningRunDocument) throws {
        let value = try document.validated()
        try database.execute("INSERT INTO learning_runs(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
            .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
        ])
    }

    public func saveCheckpoint(_ document: CheckpointDocument) throws {
        let value = try document.validated()
        try database.transaction {
            let existing = try database.query("SELECT document FROM checkpoints WHERE id=?", [.text(value.id.uuidString)])
            if let bytes = existing.first?["document"]?.data {
                let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
                let previous = try decoder.decode(CheckpointDocument.self, from: bytes)
                guard previous.agentID == value.agentID, previous.policySignature == value.policySignature,
                      previous.trainingStep == value.trainingStep, previous.parameterCount == value.parameterCount,
                      previous.kind == value.kind, previous.runID == value.runID else {
                    throw AstraError("checkpoint.immutable", "A checkpoint's identity and model metadata cannot be overwritten.")
                }
            }
            try database.execute("INSERT INTO checkpoints(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
                .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
            ])
            try database.execute("INSERT OR IGNORE INTO agent_checkpoints VALUES(?,?)", [.text(value.agentID.uuidString), .text(value.id.uuidString)])
        }
    }

    public func checkpointIDs(for agentID: UUID) throws -> Set<UUID> {
        // Creator rows remain usable when opening an older development catalog
        // that predates explicit shared links.
        let rows = try database.query("SELECT checkpoint_id AS id FROM agent_checkpoints WHERE agent_id=? UNION SELECT id FROM checkpoints WHERE json_extract(CAST(document AS TEXT),'$.agentID')=?",
                                      [.text(agentID.uuidString), .text(agentID.uuidString)])
        return Set(rows.compactMap { $0["id"]?.string.flatMap(UUID.init(uuidString:)) })
    }

    /// Call only after the coordinator has established an exclusive library
    /// lease. Running children of a different coordinator must not be relabeled.
    public func markAbandonedLearningRunsInterrupted() throws {
        var issues: [LibraryIssue] = []
        let runs: [LearningRunDocument] = try documents(table: "learning_runs", issues: &issues)
        for var run in runs where [.preparing, .running, .cancelling].contains(run.status) {
            run.status = .interrupted; run.modifiedAt = Date()
            run.issue = "The previous compute session ended before this run was finalized. Any published checkpoint is preserved."
            try saveLearningRun(run)
        }
    }

    /// Restore persisted control-cleanup warnings after acquiring the library
    /// lease. This is evidence presentation, not an assertion of OS key state.
    public func inspectPriorInferenceRuns() throws {
        inferenceIssues = []
        do { try inspectInferenceHistory() }
        catch {
            inferenceIssues.append(.init(id: "inference.history", collection: "runs",
                                        message: "Previous control cleanup could not be checked. " + error.localizedDescription))
        }
    }

    private func inspectInferenceHistory() throws {
        let folder = root.appendingPathComponent("Runs", isDirectory: true)
        guard FileManager.default.fileExists(atPath: folder.path) else { return }
        let values = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else {
            throw AstraError("inference.history", "The local run history must be a regular directory.")
        }
        let keys: [URLResourceKey] = [.isDirectoryKey, .isSymbolicLinkKey]
        let runs = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: keys)
        guard runs.count <= 4096 else {
            inferenceIssues = [.init(id: "inference.historyLimit", collection: "runs", message: "Run history is too large to check automatically. Review prior run results before starting live control.")]
            return
        }
        for run in runs {
            guard let identifier = UUID(uuidString: run.lastPathComponent) else { continue }
            let values = try run.resourceValues(forKeys: Set(keys))
            guard values.isDirectory == true, values.isSymbolicLink != true else { continue }
            let result = run.appendingPathComponent("results.json")
            var message: String?
            if !FileManager.default.fileExists(atPath: result.path) {
                message = "A previous agent run was interrupted before its cleanup result was saved. Verify that no controls remain held before starting live control."
            } else {
                do {
                    let properties = try result.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
                    guard properties.isRegularFile == true, properties.isSymbolicLink != true, (properties.fileSize ?? .max) <= 262_144 else {
                        throw AstraError("inference.history", "A previous run result is missing, linked or oversized.")
                    }
                    let value = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: result))
                    guard case .object(let fields) = value, case .string(let runID) = fields["runID"], UUID(uuidString: runID) == identifier else {
                        throw AstraError("inference.history", "A previous run result has an inconsistent identity.")
                    }
                    if fields["cleanupConfirmed"] != .bool(true) {
                        message = "A previous control helper ended without confirmed cleanup. Release any remaining held controls manually before starting another run."
                    }
                } catch { message = error.localizedDescription }
            }
            if let message { inferenceIssues.append(.init(id: identifier.uuidString + ".controlCleanup", collection: "runs", message: message)) }
        }
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
        try database.transaction {
            try save(copy)
            for identifier in try checkpointIDs(for: source.id) {
                try database.execute("INSERT OR IGNORE INTO agent_checkpoints VALUES(?,?)", [.text(copy.id.uuidString), .text(identifier.uuidString)])
            }
            for identifier in try recordingIDs(for: source.id) {
                try database.execute("INSERT OR IGNORE INTO agent_recordings VALUES(?,?)", [.text(copy.id.uuidString), .text(identifier.uuidString)])
            }
        }
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
