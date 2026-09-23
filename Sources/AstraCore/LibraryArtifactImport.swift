import Foundation

extension LibraryStore {
    public func previewArtifactImport(from source: URL, linkToAgentID: UUID? = nil) async throws -> ArtifactImportPlan {
        let (archive, digest) = try ArtifactArchive.read(source)
        if let id = linkToAgentID,
           try database.query("SELECT id FROM agents WHERE id=? AND archived=0", [.text(id.uuidString)]).isEmpty {
            throw ArtifactArchive.error("The selected destination agent is unavailable.")
        }
        for item in archive.items { _ = try importDestination(item); _ = try existingImportMatches(item) }
        try validateImportCatalog(archive.catalog)
        return .init(id: UUID(), source: source.standardizedFileURL, linkToAgentID: linkToAgentID, archive: archive, archiveSHA256: digest)
    }

    public func importArtifactArchive(_ plan: ArtifactImportPlan,
        progress: @Sendable (ArtifactTransferProgress) -> Void = { _ in },
        cancelled: @Sendable () -> Bool = { false }) async throws -> ArtifactImportResult {
        try ArtifactArchive.validate(plan.archive)
        progress(.init(phase: "Verifying archive", completedBytes: 0, totalBytes: plan.totalBytes, name: plan.source.lastPathComponent))
        let (archive, digest) = try ArtifactArchive.read(plan.source, cancelled: cancelled)
        guard digest == plan.archiveSHA256, archive.id == plan.archive.id else {
            throw ArtifactArchive.error("The archive changed after preview. Review it again before importing.")
        }
        if let id = plan.linkToAgentID,
           try database.query("SELECT id FROM agents WHERE id=? AND archived=0", [.text(id.uuidString)]).isEmpty {
            throw ArtifactArchive.error("The selected destination agent is unavailable.")
        }
        try validateImportCatalog(archive.catalog)
        try recordTransfer(id: plan.id, operation: "import", status: "copying", document: plan, createdAt: archive.createdAt)
        var reused = 0, imported = 0, completed: UInt64 = 0
        do {
            for item in archive.items {
                try ArtifactTransferFiles.check(cancelled)
                if try existingImportMatches(item) {
                    reused += 1; completed += item.bytes
                    progress(.init(phase: "Reusing verified artifact", completedBytes: completed, totalBytes: plan.totalBytes, name: item.name))
                    continue
                }
                let destination = try importDestination(item)
                let parent = destination.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                try ArtifactTransferFiles.requireDirectory(parent)
                try ArtifactTransferFiles.sync(parent.deletingLastPathComponent())
                let stage = parent.appendingPathComponent(".astra-import-\(plan.id.uuidString.lowercased())-\(item.kind.rawValue)-\(item.identity)", isDirectory: true)
                if !FileManager.default.fileExists(atPath: stage.path) {
                    try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
                }
                try ArtifactTransferFiles.requireDirectory(stage)
                try ArtifactTransferFiles.recoverStagingTemps(stage, planID: plan.id, entries: item.files)
                let alreadyCopied = completed
                try ArtifactTransferFiles.copy(item.files, from: plan.source.appendingPathComponent(item.relativePath), to: stage,
                    progress: { value in progress(.init(phase: value.phase, completedBytes: alreadyCopied + value.completedBytes,
                        totalBytes: plan.totalBytes, name: item.name)) }, cancelled: cancelled)
                guard try ArtifactTransferFiles.inventory(stage, cancelled: cancelled) == item.files else {
                    throw ArtifactArchive.error("A staged artifact contains missing, changed or unexpected files.")
                }
                try ArtifactArchive.validateContent(item, directory: stage, catalog: archive.catalog, items: archive.items)
                try ArtifactTransferFiles.check(cancelled)
                if item.kind == .rewardAsset {
                    try ArtifactTransferFiles.publish(stage.appendingPathComponent(item.files[0].path), to: destination)
                    try FileManager.default.removeItem(at: stage)
                } else { try ArtifactTransferFiles.publish(stage, to: destination) }
                imported += 1; completed += item.bytes
            }
            try ArtifactTransferFiles.check(cancelled)
            // Immutable bytes are all published before any catalog link appears.
            // A crash before this transaction leaves only recoverable orphans.
            try database.transaction {
                try validateImportCatalog(archive.catalog)
                try publishImportCatalog(archive.catalog, linkToAgentID: plan.linkToAgentID)
                try database.execute("UPDATE artifact_transfers SET status='completed',issue=NULL WHERE id=?", [.text(plan.id.uuidString)])
            }
            progress(.init(phase: "Import complete", completedBytes: plan.totalBytes, totalBytes: plan.totalBytes, name: plan.source.lastPathComponent))
            return .init(agentIDs: archive.catalog.agents.map(\.id), recordingIDs: archive.catalog.recordings.map(\.id),
                checkpointIDs: archive.catalog.checkpoints.map(\.id), importedCount: imported, reusedCount: reused, notices: archive.notices)
        } catch {
            try? transferFailed(plan.id, issue: error is CancellationError ? "Import cancelled. Published copies are intact but not linked; retry this same archive to finish." : error.localizedDescription)
            throw error
        }
    }

    func importDestination(_ item: ArtifactTransferItem) throws -> URL {
        switch item.kind {
        case .recording:
            try layout.requireAvailable(.recordings)
            guard let id = UUID(uuidString: item.identity) else { throw ArtifactArchive.error("Invalid recording identity.") }
            return recordingDirectory(id: id)
        case .checkpoint:
            try layout.requireAvailable(.models)
            guard let id = UUID(uuidString: item.identity) else { throw ArtifactArchive.error("Invalid checkpoint identity.") }
            return checkpointDirectory(id: id)
        case .dataset: return root.appendingPathComponent("Datasets/\(item.identity)", isDirectory: true)
        case .runConfiguration: return root.appendingPathComponent("Jobs/\(item.identity)", isDirectory: true)
        case .desktopConfiguration: return root.appendingPathComponent("DesktopRuns/\(item.identity)", isDirectory: true)
        case .rewardAsset: return root.appendingPathComponent("RewardTemplates/\(item.identity).image")
        }
    }
    func existingImportMatches(_ item: ArtifactTransferItem) throws -> Bool {
        let destination = try importDestination(item)
        guard FileManager.default.fileExists(atPath: destination.path) else { return false }
        if item.kind == .rewardAsset {
            guard try ArtifactTransferFiles.fingerprint(destination, relativePath: item.files[0].path, cancelled: { false }) == item.files[0] else {
                throw ArtifactArchive.error("An existing reward image has the same address but different bytes.")
            }
        } else if item.kind == .runConfiguration || item.kind == .desktopConfiguration {
            try ArtifactTransferFiles.requireDirectory(destination)
            guard try ArtifactTransferFiles.fingerprint(destination.appendingPathComponent("configuration.json"), relativePath: "configuration.json", cancelled: { false }) == item.files[0] else {
                throw ArtifactArchive.error("An existing run configuration has the same identity but different bytes.")
            }
        } else {
            guard try ArtifactTransferFiles.inventory(destination) == item.files else {
                throw ArtifactArchive.error("An existing \(item.kind.rawValue) has the same identity but different bytes. Nothing was overwritten.")
            }
        }
        return true
    }

    private func validateImportCatalog(_ catalog: ArtifactCatalogBundle) throws {
        // Mutable agent presentation is retained on collision. Immutable source
        // documents must agree; no source UUID is rewritten to evade a conflict.
        for value in catalog.environments { try sameImported(value, table: "environments") }
        for value in catalog.contexts { try sameImported(value, table: "context_fields") }
        for value in catalog.rewards { try sameImported(value, table: "reward_programs") }
        for value in catalog.recordings { try sameImported(value, table: "recordings") }
        for value in catalog.runs { try sameImported(value, table: "learning_runs") }
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        for value in catalog.checkpoints {
            if let data = try database.query("SELECT document FROM checkpoints WHERE id=?", [.text(value.id.uuidString)]).first?["document"]?.data {
                guard try decoder.decode(CheckpointDocument.self, from: data).matchesIdentity(of: value) else {
                    throw ArtifactArchive.error("An existing checkpoint has different policy or provenance metadata.")
                }
            }
            guard try database.query("SELECT checkpoint_id FROM checkpoint_cleanup WHERE checkpoint_id=?", [.text(value.id.uuidString)]).isEmpty else {
                throw ArtifactArchive.error("Finish this checkpoint's previously requested cleanup before importing it again.")
            }
        }
        for link in catalog.links {
            for (id, selection) in link.recordingSelections {
                if let row = try database.query("SELECT selection FROM agent_recordings WHERE agent_id=? AND recording_id=?", [.text(link.agentID.uuidString), .text(id.uuidString)]).first {
                    let old: RecordingTrainingSelection
                    if row["selection"] == .null { old = .whole }
                    else if let data = row["selection"]?.data, data.count <= 65_536 { old = try decoder.decode(RecordingTrainingSelection.self, from: data).validated() }
                    else { throw ArtifactArchive.error("An existing recording selection is invalid. No catalog metadata was changed.") }
                    guard old == selection else { throw ArtifactArchive.error("A recording's saved ranges or contexts differ from this archive. Existing training selections were preserved.") }
                }
            }
        }
    }
    private func sameImported<T: Codable & Identifiable>(_ value: T, table: String) throws where T.ID == UUID {
        if let bytes = try database.query("SELECT document FROM \(table) WHERE id=?", [.text(value.id.uuidString)]).first?["document"]?.data {
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
            let previous = try decoder.decode(T.self, from: bytes)
            guard try ArtifactArchive.sameMetadata(previous, value) else { throw ArtifactArchive.error("An existing \(table) item has the same identity but different metadata. Existing data was preserved.") }
        }
    }
    private func publishImportCatalog(_ catalog: ArtifactCatalogBundle, linkToAgentID: UUID?) throws {
        func insert<T: Encodable & Identifiable>(_ value: T, table: String, name: String, created: Date) throws where T.ID == UUID {
            try database.execute("INSERT OR IGNORE INTO \(table)(id,name,document,created) VALUES(?,?,?,?)", [
                .text(value.id.uuidString), .text(name), .blob(try encode(value)), .real(created.timeIntervalSince1970)])
            try database.execute("UPDATE \(table) SET archived=0 WHERE id=?", [.text(value.id.uuidString)])
        }
        for value in catalog.environments { try insert(value, table: "environments", name: value.name, created: value.createdAt) }
        for value in catalog.contexts { try insert(value, table: "context_fields", name: value.name, created: Date()) }
        for value in catalog.rewards { try insert(value, table: "reward_programs", name: value.name, created: Date()) }
        for value in catalog.agents { try insert(value, table: "agents", name: value.name, created: value.createdAt) }
        for value in catalog.recordings { try insert(value, table: "recordings", name: value.name, created: value.createdAt) }
        for value in catalog.runs { try insert(value, table: "learning_runs", name: value.name, created: value.createdAt) }
        for value in catalog.checkpoints { try insert(value, table: "checkpoints", name: value.name, created: value.createdAt) }
        for link in catalog.links {
            for (id, selection) in link.recordingSelections {
                try database.execute("INSERT OR IGNORE INTO agent_recordings(agent_id,recording_id,selection) VALUES(?,?,?)", [
                    .text(link.agentID.uuidString), .text(id.uuidString), .blob(try encode(selection))])
            }
            for id in link.checkpointIDs { try database.execute("INSERT OR IGNORE INTO agent_checkpoints VALUES(?,?)", [.text(link.agentID.uuidString), .text(id.uuidString)]) }
        }
        if let agentID = linkToAgentID {
            for recording in catalog.recordings {
                // Extra destination links never overwrite an existing agent's
                // independently chosen ranges or context values.
                try database.execute("INSERT OR IGNORE INTO agent_recordings(agent_id,recording_id) VALUES(?,?)", [.text(agentID.uuidString), .text(recording.id.uuidString)])
            }
            for checkpoint in catalog.checkpoints { try database.execute("INSERT OR IGNORE INTO agent_checkpoints VALUES(?,?)", [.text(agentID.uuidString), .text(checkpoint.id.uuidString)]) }
        }
    }
}
