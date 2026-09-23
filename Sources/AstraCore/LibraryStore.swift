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
    public var contextFieldIDs: [UUID]?

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
        if let contextFieldIDs {
            guard contextFieldIDs.count <= 32, Set(contextFieldIDs).count == contextFieldIDs.count else {
                throw AstraError("context.agent", "An agent can use up to 32 distinct context fields.")
            }
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
    public let controlHistory: ControlHistoryReview?
    public let blocksLiveControl: Bool
    public init(id: String, collection: String, message: String, controlHistory: ControlHistoryReview? = nil, blocksLiveControl: Bool = false) {
        self.id = id; self.collection = collection; self.message = message
        self.controlHistory = controlHistory; self.blocksLiveControl = blocksLiveControl
    }
}
public struct LibrarySnapshot: Sendable {
    public var agents: [AgentDocument]
    public var environments: [EnvironmentDocument]
    public var recordings: [RecordingManifest]
    public var learningRuns: [LearningRunDocument]
    public var checkpoints: [CheckpointDocument]
    public var issues: [LibraryIssue]
    public var rewardPrograms: [RewardProgram] = []
    public var recordingSelections: [UUID: [UUID: RecordingTrainingSelection]] = [:]
    public var evaluations: [EvaluationDocument] = []
    public var closedLoopEvaluations: [ClosedLoopEvaluationDocument] = []
    public var pendingFeedback: [PendingFeedbackDocument] = []
    public var contextFields: [ContextFieldDocument] = []
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
    public nonisolated let layout: ArtifactStorageLayout
    // Module-internal for same-actor artifact transaction extensions.
    let database: SQLiteDatabase
    private var recoveryIssues: [LibraryIssue] = []
    private var inferenceIssues: [LibraryIssue] = []

    public init(root: URL) throws {
        self.root = root.standardizedFileURL
        let database = try SQLiteDatabase(url: self.root.appendingPathComponent("library.sqlite"))
        self.database = database
        try database.transaction {
            try database.execute("CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL)")
            let versions = try database.query("SELECT version FROM schema_info")
            let previousVersion = versions.first?["version"]?.integer
            if versions.isEmpty { try database.execute("INSERT INTO schema_info VALUES (?)", [.integer(2)]) }
            else if versions.count != 1 || ![Int64(1), 2].contains(previousVersion ?? -1) {
                throw AstraError("library.version", "This library was created by an unsupported Astra version.")
            }
            try database.execute("CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS environments (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS context_fields (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
            try database.execute("CREATE TABLE IF NOT EXISTS recordings (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            if previousVersion == 1 {
                // Early schema-1 workspaces predate recording links entirely.
                // Create the old shape first, then apply the same atomic upgrade.
                try database.execute("CREATE TABLE IF NOT EXISTS agent_recordings (agent_id TEXT NOT NULL REFERENCES agents(id), recording_id TEXT NOT NULL REFERENCES recordings(id), PRIMARY KEY(agent_id,recording_id))")
                try database.execute("ALTER TABLE agent_recordings ADD COLUMN selection BLOB")
                try database.execute("UPDATE schema_info SET version=2")
            } else {
                try database.execute("CREATE TABLE IF NOT EXISTS agent_recordings (agent_id TEXT NOT NULL REFERENCES agents(id), recording_id TEXT NOT NULL REFERENCES recordings(id), selection BLOB, PRIMARY KEY(agent_id,recording_id))")
            }
            try database.execute("CREATE TABLE IF NOT EXISTS learning_runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS agent_checkpoints (agent_id TEXT NOT NULL REFERENCES agents(id), checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id), PRIMARY KEY(agent_id,checkpoint_id))")
            try database.execute("CREATE TABLE IF NOT EXISTS control_history_acknowledgements (location TEXT NOT NULL, directory_name TEXT NOT NULL, run_id TEXT NOT NULL, fingerprint TEXT NOT NULL, operator_acknowledged_at REAL NOT NULL, PRIMARY KEY(location,directory_name,fingerprint))")
            try database.execute("CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS closed_loop_evaluations (id TEXT PRIMARY KEY, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS reward_programs (id TEXT PRIMARY KEY, name TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS pending_feedback (id TEXT PRIMARY KEY, document BLOB NOT NULL, created REAL NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            try database.execute("CREATE TABLE IF NOT EXISTS checkpoint_cleanup (checkpoint_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, document BLOB NOT NULL, created REAL NOT NULL, issue TEXT)")
            try database.execute("CREATE TABLE IF NOT EXISTS artifact_transfers (id TEXT PRIMARY KEY, operation TEXT NOT NULL, status TEXT NOT NULL, document BLOB NOT NULL, issue TEXT, created REAL NOT NULL)")
            if try database.query("SELECT key FROM settings WHERE key='checkpointLinksMigrated'").isEmpty {
                // Materialize old creator ownership once; repeated migration
                // must never reattach an explicitly removed creator link.
                try database.execute("INSERT OR IGNORE INTO agent_checkpoints(agent_id,checkpoint_id) SELECT agents.id,checkpoints.id FROM checkpoints JOIN agents ON agents.id=CASE WHEN json_valid(CAST(checkpoints.document AS TEXT)) THEN json_extract(CAST(checkpoints.document AS TEXT),'$.agentID') END WHERE checkpoints.archived=0")
                try database.execute("INSERT INTO settings(key,value) VALUES('checkpointLinksMigrated',?)", [.blob(Data([1]))])
            }
        }
        if let bytes = try database.query("SELECT value FROM settings WHERE key='artifactStorageLayout'").first?["value"]?.data {
            guard bytes.count <= 32_768 else { throw AstraError("storage.layout", "The saved storage locations exceed their metadata limit.") }
            let saved = try JSONDecoder().decode(ArtifactStorageLayout.self, from: bytes).validated()
            guard saved.catalogRoot == self.root else { throw AstraError("storage.catalogMoved", "The catalog moved independently of its storage settings. Open it from its original location or import a portable archive.") }
            layout = saved
        } else {
            let initial = try ArtifactStorageLayout.initial(catalogRoot: self.root)
            for kind in ArtifactStorageKind.allCases {
                try FileManager.default.createDirectory(at: initial.root(for: kind), withIntermediateDirectories: true)
                try ArtifactTransferFiles.requireDirectory(initial.root(for: kind))
                try ArtifactStorageMarker.create(layout: initial, kind: kind)
            }
            try ArtifactTransferFiles.sync(self.root)
            try database.execute("INSERT INTO settings(key,value) VALUES('artifactStorageLayout',?)", [.blob(try JSONEncoder().encode(initial))])
            layout = initial
        }
        for folder in ["Datasets", "Jobs", "Caches", "Logs"] {
            try FileManager.default.createDirectory(at: self.root.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
    }

    public func snapshot() throws -> LibrarySnapshot {
        var issues = recoveryIssues + inferenceIssues
        for kind in ArtifactStorageKind.allCases {
            do { try layout.requireAvailable(kind) }
            catch { issues.append(.init(id: "storage." + kind.rawValue, collection: "storage", message: error.localizedDescription)) }
        }
        for row in try database.query("SELECT checkpoint_id,issue FROM checkpoint_cleanup") {
            issues.append(.init(id: (row["checkpoint_id"]?.string ?? "unknown") + ".cleanup", collection: "checkpoint cleanup",
                message: row["issue"]?.string ?? "A previously requested checkpoint cleanup is unfinished. Retry it from Manage Checkpoints."))
        }
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
        let evaluations: [EvaluationDocument] = try documents(table: "evaluations", issues: &issues).compactMap { (value: EvaluationDocument) -> EvaluationDocument? in
            do { return try value.validated() }
            catch { issues.append(.init(id: value.id.uuidString, collection: "evaluations", message: error.localizedDescription)); return nil }
        }
        let closedLoop: [ClosedLoopEvaluationDocument] = try documents(table: "closed_loop_evaluations", issues: &issues).compactMap { (value: ClosedLoopEvaluationDocument) -> ClosedLoopEvaluationDocument? in
            do { return try value.validated() }
            catch { issues.append(.init(id: value.id.uuidString, collection: "closed-loop evaluations", message: error.localizedDescription)); return nil }
        }
        let pendingFeedback: [PendingFeedbackDocument] = try documents(table: "pending_feedback", maximumBytes: PendingFeedbackDocument.maximumBytes, issues: &issues).compactMap { (value: PendingFeedbackDocument) -> PendingFeedbackDocument? in
            do { return try value.validated() }
            catch { issues.append(.init(id: value.id.uuidString, collection: "pending_feedback", message: error.localizedDescription)); return nil }
        }
        let contexts: [ContextFieldDocument] = try documents(table: "context_fields", issues: &issues).compactMap { (value: ContextFieldDocument) -> ContextFieldDocument? in
            do { return try value.validated() }
            catch { issues.append(.init(id: value.id.uuidString, collection: "context fields", message: error.localizedDescription)); return nil }
        }
        var selections: [UUID: [UUID: RecordingTrainingSelection]] = [:]
        let recordingsByID = Dictionary(uniqueKeysWithValues: recordings.map { ($0.id, $0) })
        for agent in agents {
            for row in try database.query("SELECT recording_id,CASE WHEN selection IS NULL OR (typeof(selection)='blob' AND length(selection)<=65536) THEN selection ELSE 0 END AS selection FROM agent_recordings WHERE agent_id=?", [.text(agent.id.uuidString)]) {
                let identifier = row["recording_id"]?.string.flatMap(UUID.init(uuidString:))
                do {
                    guard let identifier, let recording = recordingsByID[identifier] else {
                        throw AstraError("selection.source", "A linked recording is unavailable in the catalog.")
                    }
                    let selection = try decodeRecordingSelection(row["selection"])
                    if selection.ranges != nil { _ = try selection.resolved(for: recording) }
                    selections[agent.id, default: [:]][identifier] = selection
                } catch {
                    issues.append(.init(id: "\(agent.id).\(identifier?.uuidString ?? "unknown").selection", collection: "recording selections", message: error.localizedDescription))
                }
            }
        }
        return LibrarySnapshot(agents: agents, environments: environments, recordings: recordings, learningRuns: runs, checkpoints: checkpoints, issues: issues, rewardPrograms: rewards, recordingSelections: selections, evaluations: evaluations, closedLoopEvaluations: closedLoop, pendingFeedback: pendingFeedback, contextFields: contexts)
    }

    public func saveContexts(_ fields: [ContextFieldDocument], selectedIDs: [UUID], for agentID: UUID) throws {
        let checked = try fields.map { try $0.validated() }
        guard checked.count <= 256, Set(checked.map(\.id)).count == checked.count else {
            throw AstraError("context.catalog", "Save at most 256 distinct context fields at once.")
        }
        _ = try ContextVocabulary(fields: selectedIDs.map { id in
            guard let field = checked.first(where: { $0.id == id }) else { throw AstraError("context.field", "A selected context field is missing.") }
            return field
        }).validated()
        try database.transaction {
            guard let bytes = try database.query("SELECT document FROM agents WHERE id=? AND archived=0", [.text(agentID.uuidString)]).first?["document"]?.data else {
                throw AstraError("context.agent", "The agent is no longer available.")
            }
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
            var agent = try decoder.decode(AgentDocument.self, from: bytes).validated()
            for field in checked { try saveContextField(field) }
            agent.contextFieldIDs = selectedIDs; agent.modifiedAt = Date(); try save(agent)
        }
    }

    public func saveContextField(_ document: ContextFieldDocument) throws {
        let value = try document.validated()
        try database.execute("INSERT INTO context_fields(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
            .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(Date().timeIntervalSince1970)])
    }

    public func savePendingFeedback(_ document: PendingFeedbackDocument) throws {
        let value = try document.validated()
        try database.transaction {
            if let row = try database.query("SELECT CASE WHEN typeof(document)='blob' AND length(document)<=? THEN document END AS document FROM pending_feedback WHERE id=?", [
                .integer(Int64(PendingFeedbackDocument.maximumBytes)), .text(value.id.uuidString)
            ]).first {
                guard let bytes = row["document"]?.data else {
                    throw AstraError("feedback.pending", "The existing feedback catalog entry is invalid or exceeds its size limit.")
                }
                let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
                let previous = try decoder.decode(PendingFeedbackDocument.self, from: bytes).validated()
                try value.validateUpdate(from: previous)
            }
            try database.execute("INSERT INTO pending_feedback(id,document,created) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET document=excluded.document", [
                .text(value.id.uuidString), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
            ])
        }
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

    /// A running attempt may be finalized once; published metrics/provenance remain immutable.
    public func saveEvaluation(_ document: EvaluationDocument) throws {
        let value = try document.validated()
        let encoded = try encode(value)
        try database.transaction {
            if let bytes = try database.query("SELECT document FROM evaluations WHERE id=?", [.text(value.id.uuidString)]).first?["document"]?.data {
                let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
                let previous = try decoder.decode(EvaluationDocument.self, from: bytes).validated()
                guard previous.comparisonID == value.comparisonID, previous.agentID == value.agentID,
                      previous.checkpointID == value.checkpointID, previous.checkpointName == value.checkpointName,
                      previous.checkpointPolicySignature == value.checkpointPolicySignature,
                      previous.protocolDefinition == value.protocolDefinition,
                      abs(previous.createdAt.timeIntervalSince(value.createdAt)) < 0.001,
                      previous.status == .running || bytes == encoded else {
                    throw AstraError("evaluation.immutable", "A saved evaluation result or protocol cannot be overwritten.")
                }
            }
            try database.execute("INSERT INTO evaluations(id,document,created) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET document=excluded.document", [
                .text(value.id.uuidString), .blob(encoded), .real(value.createdAt.timeIntervalSince1970)])
        }
    }

    public func saveLearningRun(_ document: LearningRunDocument) throws {
        let value = try document.validated()
        try database.execute("INSERT INTO learning_runs(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
            .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
        ])
    }

    public func saveClosedLoopEvaluation(_ document: ClosedLoopEvaluationDocument) throws {
        let value = try document.validated(), encoded = try encode(document)
        try database.transaction {
            if let bytes = try database.query("SELECT document FROM closed_loop_evaluations WHERE id=?", [.text(value.id.uuidString)]).first?["document"]?.data {
                let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
                let previous = try decoder.decode(ClosedLoopEvaluationDocument.self, from: bytes).validated()
                guard previous.comparisonID == value.comparisonID, previous.agentID == value.agentID,
                      previous.checkpointID == value.checkpointID, previous.checkpointName == value.checkpointName,
                      previous.checkpointPolicySignature == value.checkpointPolicySignature,
                      previous.protocolDefinition == value.protocolDefinition,
                      abs(previous.createdAt.timeIntervalSince(value.createdAt)) < 0.001,
                      previous.status == .running || bytes == encoded else {
                    throw AstraError("evaluation.immutable", "A saved practice evaluation or its protocol cannot be overwritten.")
                }
            }
            try database.execute("INSERT INTO closed_loop_evaluations(id,document,created) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET document=excluded.document", [
                .text(value.id.uuidString), .blob(encoded), .real(value.createdAt.timeIntervalSince1970)])
        }
    }

    public func saveCheckpoint(_ document: CheckpointDocument) throws {
        var value = try document.validated()
        try database.transaction {
            let existing = try database.query("SELECT document,archived FROM checkpoints WHERE id=?", [.text(value.id.uuidString)])
            if let bytes = existing.first?["document"]?.data {
                let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
                let previous = try decoder.decode(CheckpointDocument.self, from: bytes)
                guard existing.first?["archived"]?.integer == 0, previous.agentID == value.agentID, previous.policySignature == value.policySignature,
                      previous.trainingStep == value.trainingStep, previous.parameterCount == value.parameterCount,
                      previous.kind == value.kind, previous.runID == value.runID else {
                    throw AstraError("checkpoint.immutable", "A checkpoint's identity and model metadata cannot be overwritten.")
                }
                value.name = previous.name; value.pinned = previous.pinned; value.createdAt = previous.createdAt
            }
            try database.execute("INSERT INTO checkpoints(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
                .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
            ])
            if existing.isEmpty {
                try database.execute("INSERT OR IGNORE INTO agent_checkpoints VALUES(?,?)", [.text(value.agentID.uuidString), .text(value.id.uuidString)])
            }
        }
    }

    public func checkpointIDs(for agentID: UUID) throws -> Set<UUID> {
        let rows = try database.query("SELECT checkpoint_id AS id FROM agent_checkpoints JOIN checkpoints ON checkpoints.id=checkpoint_id WHERE agent_id=? AND checkpoints.archived=0",
                                      [.text(agentID.uuidString)])
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
        let evaluations: [EvaluationDocument] = try documents(table: "evaluations", issues: &issues)
        for var evaluation in evaluations where evaluation.status == .running {
            evaluation.status = .interrupted; evaluation.finishedAt = Date()
            evaluation.issue = "The previous compute session ended before this evaluation finished. No partial score was published."
            try saveEvaluation(evaluation)
        }
        let closedLoop: [ClosedLoopEvaluationDocument] = try documents(table: "closed_loop_evaluations", issues: &issues)
        for var value in closedLoop where value.status == .running {
            value.status = .interrupted; value.finishedAt = Date(); value.issue = "Astra closed before this practice evaluation finished."
            try saveClosedLoopEvaluation(value)
        }
    }

    /// Call with the library lease held and outside an active control workflow.
    /// Operator review is separate from the immutable native cleanup evidence.
    public func inspectPriorInferenceRuns() throws {
        inferenceIssues = []
        for location in [ControlHistoryLocation.inference, .desktop] {
            do {
                for name in try ControlHistoryReader.names(root: root, location: location) {
                    try Task.checkCancellation()
                    guard UUID(uuidString: name) != nil else { continue }
                    do {
                        guard let review = try ControlHistoryReader.review(root: root, location: location, directoryName: name),
                              try !historyAcknowledged(review) else { continue }
                        inferenceIssues.append(.init(id: review.id, collection: "runs", message: review.message,
                            controlHistory: review, blocksLiveControl: true))
                    } catch is CancellationError { throw CancellationError() }
                    catch {
                        inferenceIssues.append(.init(id: location.rawValue + "." + name + ".history", collection: "runs",
                            message: error.localizedDescription, blocksLiveControl: true))
                    }
                }
            } catch is CancellationError { throw CancellationError() }
            catch {
                inferenceIssues.append(.init(id: location.rawValue + ".history", collection: "runs",
                    message: "Previous control cleanup could not be checked. " + error.localizedDescription, blocksLiveControl: true))
            }
        }
    }

    public func controlHistoryReview(location: ControlHistoryLocation, runID: UUID) throws -> ControlHistoryReview? {
        try ControlHistoryReader.review(root: root, location: location, directoryName: runID.uuidString.lowercased())
    }

    /// The caller must have joined its current workflows and hold an exclusive
    /// desktop-control lease throughout this operation. This records only the
    /// user's manual-release acknowledgement, never a successful native release.
    public func acknowledgeControlHistory(_ expected: ControlHistoryReview) throws {
        guard let current = try ControlHistoryReader.review(root: root, location: expected.location, directoryName: expected.directoryName),
              current.runID == expected.runID, current.fingerprint == expected.fingerprint else {
            throw AstraError("history.changed", "This run history changed since it was shown. Refresh and review the current cleanup warning.")
        }
        try Task.checkCancellation()
        try database.execute("INSERT INTO control_history_acknowledgements(location,directory_name,run_id,fingerprint,operator_acknowledged_at) VALUES(?,?,?,?,?) ON CONFLICT(location,directory_name,fingerprint) DO NOTHING", [
            .text(current.location.rawValue), .text(current.directoryName), .text(current.runID.uuidString.lowercased()),
            .text(current.fingerprint), .real(Date().timeIntervalSince1970)])
        inferenceIssues.removeAll { $0.controlHistory?.id == current.id }
    }
    private func historyAcknowledged(_ review: ControlHistoryReview) throws -> Bool {
        try !database.query("SELECT run_id FROM control_history_acknowledgements WHERE location=? AND directory_name=? AND fingerprint=? AND run_id=?", [
            .text(review.location.rawValue), .text(review.directoryName), .text(review.fingerprint), .text(review.runID.uuidString.lowercased())]).isEmpty
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
        layout.recordingDirectory(id: id)
    }
    public nonisolated func checkpointDirectory(id: UUID) -> URL { layout.checkpointDirectory(id: id) }

    public func saveRecording(_ manifest: RecordingManifest, linkTo agentID: UUID? = nil) throws {
        let value = try manifest.validated()
        try database.transaction {
            try database.execute("INSERT INTO recordings(id,name,document,created) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,document=excluded.document", [
                .text(value.id.uuidString), .text(value.name), .blob(try encode(value)), .real(value.createdAt.timeIntervalSince1970)
            ])
            if let agentID {
                try database.execute("INSERT OR IGNORE INTO agent_recordings(agent_id,recording_id) VALUES(?,?)", [.text(agentID.uuidString), .text(value.id.uuidString)])
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
        try layout.requireAvailable(.recordings)
        let packages = try FileManager.default.contentsOfDirectory(at: layout.recordingsRoot,
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
                // A known recording may have been intentionally unlinked. Its
                // immutable creator identity is provenance, not current membership.
                try saveRecording(result.manifest, linkTo: catalog[id] == nil ? agentID : nil)
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

    public func linkRecordings(_ identifiers: Set<UUID>, to agentID: UUID) throws {
        guard !identifiers.isEmpty, identifiers.count <= 4096 else { throw AstraError("selection.sources", "Choose between 1 and 4,096 recordings.") }
        try database.transaction {
            guard try !database.query("SELECT id FROM agents WHERE id=? AND archived=0", [.text(agentID.uuidString)]).isEmpty else {
                throw AstraError("selection.agent", "The destination agent is unavailable.")
            }
            for id in identifiers {
                let recording = try recordingDocument(id)
                _ = try RecordingTrainingSelection.whole.resolved(for: recording)
                try database.execute("INSERT OR IGNORE INTO agent_recordings(agent_id,recording_id) VALUES(?,?)", [.text(agentID.uuidString), .text(id.uuidString)])
            }
        }
    }

    public func unlinkRecording(_ recordingID: UUID, from agentID: UUID) throws {
        try database.execute("DELETE FROM agent_recordings WHERE agent_id=? AND recording_id=?", [.text(agentID.uuidString), .text(recordingID.uuidString)])
    }

    public func saveRecordingSelection(_ selection: RecordingTrainingSelection, recordingID: UUID, agentID: UUID) throws {
        try database.transaction {
            _ = try selection.resolved(for: recordingDocument(recordingID))
            guard try !database.query("SELECT recording_id FROM agent_recordings JOIN agents ON agents.id=agent_recordings.agent_id WHERE agent_id=? AND recording_id=? AND agents.archived=0", [.text(agentID.uuidString), .text(recordingID.uuidString)]).isEmpty else {
                throw AstraError("selection.link", "This recording is no longer linked to the agent.")
            }
            try database.execute("UPDATE agent_recordings SET selection=? WHERE agent_id=? AND recording_id=?", [
                (selection.ranges == nil && (selection.contextValues?.isEmpty ?? true)) ? .null : .blob(try encode(selection)), .text(agentID.uuidString), .text(recordingID.uuidString)])
        }
    }

    private func recordingDocument(_ id: UUID) throws -> RecordingManifest {
        guard let bytes = try database.query("SELECT document FROM recordings WHERE id=? AND archived=0", [.text(id.uuidString)]).first?["document"]?.data else {
            throw AstraError("selection.source", "The selected recording is unavailable.")
        }
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        let value = try decoder.decode(RecordingManifest.self, from: bytes).validated()
        guard value.id == id else { throw AstraError("selection.source", "The recording's catalog identity changed.") }
        return value
    }

    private func decodeRecordingSelection(_ value: SQLValue?) throws -> RecordingTrainingSelection {
        if value == .null { return .whole }
        guard let bytes = value?.data, bytes.count <= 65_536 else { throw AstraError("selection.document", "The saved selection is missing, invalid or oversized.") }
        return try JSONDecoder().decode(RecordingTrainingSelection.self, from: bytes).validated()
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
            for row in try database.query("SELECT recording_id,CASE WHEN selection IS NULL OR (typeof(selection)='blob' AND length(selection)<=65536) THEN selection ELSE 0 END AS selection FROM agent_recordings WHERE agent_id=?", [.text(source.id.uuidString)]) {
                guard let identifier = row["recording_id"]?.string.flatMap(UUID.init(uuidString:)) else { throw AstraError("selection.source", "A recording link has an invalid identity.") }
                let selection = try decodeRecordingSelection(row["selection"])
                if selection.ranges != nil { _ = try selection.resolved(for: recordingDocument(identifier)) }
                try database.execute("INSERT INTO agent_recordings(agent_id,recording_id,selection) VALUES(?,?,?)", [.text(copy.id.uuidString), .text(identifier.uuidString), row["selection"] ?? .null])
            }
        }
        return copy
    }

    public func checkpoint() throws { try database.checkpoint() }

    func documents<T: Decodable & Identifiable>(table: String, includeArchived: Bool = false, maximumBytes: Int? = nil, issues: inout [LibraryIssue]) throws -> [T] where T.ID == UUID {
        // Table is selected exclusively by private call sites above.
        let projection = maximumBytes == nil ? "document" : "CASE WHEN typeof(document)='blob' AND length(document)<=? THEN document END AS document"
        let rows = try database.query("SELECT id,\(projection) FROM \(table) \(includeArchived ? "" : "WHERE archived=0") ORDER BY created DESC,id ASC", maximumBytes.map { [.integer(Int64($0))] } ?? [])
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

    func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return try encoder.encode(value)
    }
}

extension LibraryStore {
    public func updateCheckpointPresentation(id: UUID, name: String, pinned: Bool) throws {
        guard let bytes = try database.query("SELECT document FROM checkpoints WHERE id=? AND archived=0", [.text(id.uuidString)]).first?["document"]?.data else {
            throw AstraError("checkpoint.missing", "This checkpoint is no longer available in the catalog.")
        }
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        var value = try decoder.decode(CheckpointDocument.self, from: bytes).validated()
        guard value.id == id else { throw AstraError("checkpoint.identity", "The checkpoint catalog identity is inconsistent.") }
        let updatedName = try DocumentNames.validated(name)
        value.pinned = pinned || updatedName != value.name
        value.name = updatedName
        try database.execute("UPDATE checkpoints SET name=?,document=? WHERE id=? AND archived=0", [.text(value.name), .blob(try encode(value)), .text(id.uuidString)])
    }

    /// The caller owns the library lease and blocks new runtime work while
    /// previewing/applying. Persisted dependencies are independently rechecked.
    public func previewCheckpointRetention(agentID: UUID, keepNewest: Int, activeCheckpointIDs: Set<UUID> = []) throws -> CheckpointRetentionPreview {
        guard (0...100).contains(keepNewest) else { throw AstraError("checkpoint.retention", "Keep between zero and one hundred recent checkpoints.") }
        var issues: [LibraryIssue] = []
        let agents: [AgentDocument] = try documents(table: "agents", includeArchived: true, maximumBytes: 1_048_576, issues: &issues)
        let checkpoints: [CheckpointDocument] = try documents(table: "checkpoints", includeArchived: true, maximumBytes: 1_048_576, issues: &issues)
        let runs: [LearningRunDocument] = try documents(table: "learning_runs", includeArchived: true, maximumBytes: 1_048_576, issues: &issues)
        let evaluations: [EvaluationDocument] = try documents(table: "evaluations", includeArchived: true, maximumBytes: 1_048_576, issues: &issues)
        let closedLoop: [ClosedLoopEvaluationDocument] = try documents(table: "closed_loop_evaluations", includeArchived: true, maximumBytes: 1_048_576, issues: &issues)
        let feedback: [PendingFeedbackDocument] = try documents(table: "pending_feedback", includeArchived: true, maximumBytes: PendingFeedbackDocument.maximumBytes, issues: &issues)
        let recordings: [RecordingManifest] = try documents(table: "recordings", includeArchived: true, maximumBytes: 1_048_576, issues: &issues)
        guard issues.isEmpty, agents.contains(where: { $0.id == agentID }) else {
            throw AstraError("checkpoint.retentionCatalog", "Checkpoint references could not all be read. Resolve catalog issues before removing models.")
        }
        for value in agents { _ = try value.validated() }
        for value in checkpoints { _ = try value.validated() }
        for value in runs { _ = try value.validated() }
        for value in evaluations { _ = try value.validated() }
        for value in closedLoop { _ = try value.validated() }
        for value in feedback { _ = try value.validated() }
        for value in recordings { _ = try value.validated() }
        let byID = Dictionary(uniqueKeysWithValues: checkpoints.map { ($0.id, $0) })
        let agentNames = Dictionary(uniqueKeysWithValues: agents.map { ($0.id, $0.name) })
        var owners: [UUID: Set<UUID>] = [:]
        for row in try database.query("SELECT agent_id,checkpoint_id FROM agent_checkpoints") {
            guard let agent = row["agent_id"]?.string.flatMap(UUID.init(uuidString:)), agentNames[agent] != nil,
                  let checkpoint = row["checkpoint_id"]?.string.flatMap(UUID.init(uuidString:)), byID[checkpoint] != nil else {
                throw AstraError("checkpoint.retentionCatalog", "An owning checkpoint link is inconsistent. No model was removed.")
            }
            owners[checkpoint, default: []].insert(agent)
        }
        var protected: [UUID: String] = [:]
        for value in checkpoints where value.isPinned { protected[value.id] = "Pinned" }
        for agent in agents { if let id = agent.selectedCheckpointID { protected[id] = "Selected by \(agent.name)" } }
        for value in feedback where ![.completed, .discarded].contains(value.status) { protected[value.checkpoint.id] = "Needed by saved feedback" }
        for recording in recordings {
            if let reference = recording.correction {
                let prelude = try CorrectionPrelude.load(in: recordingDirectory(id: recording.id), reference: reference)
                protected[prelude.sourceCheckpointID] = "Needed by correction recording \(recording.name)"
            }
        }
        for value in runs where [.preparing, .running, .cancelling].contains(value.status) {
            for id in [value.initialCheckpointID, value.checkpointID].compactMap({ $0 }) { protected[id] = "Used by active learning" }
        }
        for value in evaluations where value.status == .running {
            protected[value.checkpointID] = "Used by active evaluation"
            protected[value.protocolDefinition.sourceCheckpointID] = "Supplies an active evaluation protocol"
        }
        for value in closedLoop where value.status == .running { protected[value.checkpointID] = "Used by active closed-loop evaluation" }
        for id in activeCheckpointIDs { protected[id] = "Retained by an active workflow" }
        let linked = checkpoints.filter { owners[$0.id]?.contains(agentID) == true }
            .sorted { $0.createdAt == $1.createdAt ? $0.id.uuidString < $1.id.uuidString : $0.createdAt > $1.createdAt }
        for value in linked.filter({ !$0.isPinned }).prefix(keepNewest) where protected[value.id] == nil {
            protected[value.id] = "Among the newest \(keepNewest) unpinned checkpoints"
        }
        var candidates: [(CheckpointDocument, CheckpointCleanupDisposition, [String])] = []
        for value in linked where protected[value.id] == nil {
            let retained = (owners[value.id] ?? []).subtracting([agentID]).compactMap { agentNames[$0] }.sorted()
            candidates.append((value, retained.isEmpty ? .deleteFiles : .unlinkShared, retained))
        }
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        let retired = Set(try database.query("SELECT id FROM checkpoints WHERE archived=1").compactMap { $0["id"]?.string.flatMap(UUID.init(uuidString:)) })
        for row in try database.query("SELECT checkpoint_id,agent_id,document FROM checkpoint_cleanup ORDER BY created,checkpoint_id") {
            guard let data = row["document"]?.data, data.count <= 1_048_576,
                  let id = row["checkpoint_id"]?.string.flatMap(UUID.init(uuidString:)), retired.contains(id),
                  let requestingAgent = row["agent_id"]?.string.flatMap(UUID.init(uuidString:)), agentNames[requestingAgent] != nil else {
                throw AstraError("checkpoint.cleanupJournal", "A pending cleanup entry needs manual review.")
            }
            let checkpoint = try decoder.decode(CheckpointDocument.self, from: data).validated()
            guard checkpoint.id == id, byID[id]?.matchesIdentity(of: checkpoint) == true, owners[id, default: []].isEmpty else {
                throw AstraError("checkpoint.cleanupJournal", "A pending cleanup entry regained ownership or has inconsistent identity.")
            }
            if row["agent_id"]?.string == agentID.uuidString, protected[id] == nil { candidates.append((checkpoint, .retryFiles, [])) }
        }
        let chosen = candidates.prefix(256)
        let items = try chosen.map { value, disposition, retained in
            CheckpointCleanupItem(checkpoint: value, disposition: disposition, retainedByAgents: retained,
                bytes: disposition == .unlinkShared ? 0 : try CheckpointArtifactFiles.bytes(modelsRoot: layout.modelsRoot, id: value.id))
        }
        return .init(agentID: agentID, keepNewest: keepNewest, items: items, protected: protected,
            remainingCandidates: max(0, candidates.count - items.count))
    }

    public func applyCheckpointRetention(_ preview: CheckpointRetentionPreview, activeCheckpointIDs: Set<UUID> = []) throws -> CheckpointCleanupResult {
        let current = try previewCheckpointRetention(agentID: preview.agentID, keepNewest: preview.keepNewest, activeCheckpointIDs: activeCheckpointIDs)
        guard current == preview else { throw AstraError("checkpoint.retentionChanged", "Checkpoint ownership or selections changed. Review a refreshed cleanup preview before continuing.") }
        guard !current.items.isEmpty else { return .init(unlinked: 0, deleted: 0, issues: []) }
        try database.transaction {
            for item in current.items where item.disposition != .retryFiles {
                try database.execute("DELETE FROM agent_checkpoints WHERE agent_id=? AND checkpoint_id=?", [.text(current.agentID.uuidString), .text(item.id.uuidString)])
                if item.disposition == .deleteFiles {
                    guard try database.query("SELECT agent_id FROM agent_checkpoints WHERE checkpoint_id=?", [.text(item.id.uuidString)]).isEmpty else {
                        throw AstraError("checkpoint.retentionOwned", "Another agent still owns this checkpoint.")
                    }
                    try database.execute("UPDATE checkpoints SET archived=1 WHERE id=?", [.text(item.id.uuidString)])
                    try database.execute("INSERT INTO checkpoint_cleanup(checkpoint_id,agent_id,document,created) VALUES(?,?,?,?)", [
                        .text(item.id.uuidString), .text(current.agentID.uuidString), .blob(try encode(item.checkpoint)), .real(Date().timeIntervalSince1970)])
                }
            }
        }
        var deleted = 0, failures: [String] = []
        for item in current.items where item.disposition != .unlinkShared {
            do {
                // This synchronous actor segment admits no new catalog owner.
                // Recheck filesystem shape; no symlink is followed for cleanup.
                _ = try CheckpointArtifactFiles.bytes(modelsRoot: layout.modelsRoot, id: item.id)
                let directory = try CheckpointArtifactFiles.directory(modelsRoot: layout.modelsRoot, id: item.id)
                if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
                try database.execute("DELETE FROM checkpoint_cleanup WHERE checkpoint_id=?", [.text(item.id.uuidString)])
                deleted += 1
            } catch {
                let message = "\(item.checkpoint.name): \(error.localizedDescription) Retry the pending cleanup from Manage Checkpoints."
                failures.append(message)
                try database.execute("UPDATE checkpoint_cleanup SET issue=? WHERE checkpoint_id=?", [.text(message), .text(item.id.uuidString)])
            }
        }
        return .init(unlinked: current.items.filter { $0.disposition == .unlinkShared }.count, deleted: deleted, issues: failures)
    }
}
