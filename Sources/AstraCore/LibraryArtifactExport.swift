import Foundation

extension LibraryStore {
    public func previewArtifactExport(agentID: UUID, recordingIDs: Set<UUID>, checkpointIDs: Set<UUID>) async throws -> ArtifactExportPlan {
        guard !recordingIDs.isEmpty || !checkpointIDs.isEmpty, recordingIDs.count + checkpointIDs.count <= 4096 else {
            throw ArtifactArchive.error("Choose between one and 4,096 recordings or checkpoints to export.")
        }
        let snapshot = try snapshot()
        guard snapshot.agents.contains(where: { $0.id == agentID }),
              recordingIDs.isSubset(of: try self.recordingIDs(for: agentID)),
              checkpointIDs.isSubset(of: try self.checkpointIDs(for: agentID)) else {
            throw ArtifactArchive.error("Choose artifacts linked to the selected agent.")
        }
        var issues: [LibraryIssue] = []
        let allAgents: [AgentDocument] = try documents(table: "agents", includeArchived: true, maximumBytes: 1_048_576, issues: &issues)
        var recordings = recordingIDs, checkpoints = checkpointIDs, visitedRecordings: Set<UUID> = [], visitedCheckpoints: Set<UUID> = []
        var datasets: Set<UUID> = [], runs: Set<UUID> = [], rewards: [UUID: RewardProgram] = [:]
        var catalog = ArtifactCatalogBundle(), items: [ArtifactTransferItem] = [], sources: [String: URL] = [:]
        func add(_ kind: ArtifactTransferKind, _ identity: String, _ name: String, _ directory: URL, only: Set<String>? = nil) throws {
            let key = kind.rawValue + ":" + identity
            guard sources[key] == nil else { return }
            let files: [ArtifactFileEntry]
            if let only {
                try ArtifactTransferFiles.requireDirectory(directory)
                files = try only.sorted().map { path in try ArtifactTransferFiles.fingerprint(directory.appendingPathComponent(path), relativePath: path, cancelled: { false }) }
            } else { files = try ArtifactTransferFiles.inventory(directory) }
            let item = ArtifactTransferItem(kind: kind, identity: identity, name: String(name.prefix(160)),
                relativePath: "payload/\(kind.rawValue)/\(identity)", files: files)
            items.append(item); sources[key] = directory
        }
        while visitedRecordings != recordings || visitedCheckpoints != checkpoints {
            try Task.checkCancellation()
            guard recordings.count + checkpoints.count <= 4096 else { throw ArtifactArchive.error("The dependency closure exceeds this archive's item limit.") }
            for id in checkpoints.subtracting(visitedCheckpoints).sorted(by: { $0.uuidString < $1.uuidString }) {
                guard let checkpoint = snapshot.checkpoints.first(where: { $0.id == id }) else { throw ArtifactArchive.error("A required source checkpoint is unavailable: \(id).") }
                try layout.requireAvailable(.models)
                let directory = checkpointDirectory(id: id), manifest = try ArtifactArchive.json(directory.appendingPathComponent("manifest.json"))
                try add(.checkpoint, id.uuidString.lowercased(), checkpoint.name, directory)
                catalog.checkpoints.append(checkpoint); visitedCheckpoints.insert(id)
                if checkpoint.kind != "initial" {
                    guard let runID = checkpoint.runID, let run = snapshot.learningRuns.first(where: { $0.id == runID }),
                          ![.preparing, .running, .cancelling].contains(run.status) else {
                        throw ArtifactArchive.error("\(checkpoint.name) has no sealed learning run. This archive requires its complete resume dependencies.")
                    }
                    let runDirectory = root.appendingPathComponent("Jobs/\(runID.uuidString.lowercased())", isDirectory: true)
                    let configuration = try ArtifactArchive.json(runDirectory.appendingPathComponent("configuration.json"))
                    guard configuration.fields?["runID"]?.uuid == runID, configuration.fields?["agentID"]?.uuid == checkpoint.agentID else {
                        throw ArtifactArchive.error("A checkpoint's saved learning configuration has a different identity.")
                    }
                    if runs.insert(runID).inserted {
                        try add(.runConfiguration, runID.uuidString.lowercased(), run.name, runDirectory, only: ["configuration.json"])
                        catalog.runs.append(run)
                    }
                    if let datasetID = manifest.fields?["datasetID"]?.uuid, configuration.fields?["dataset"]?.fields?["kind"]?.text == "recordings",
                       datasets.insert(datasetID).inserted {
                        let directory = root.appendingPathComponent("Datasets/\(datasetID.uuidString.lowercased())", isDirectory: true)
                        let dataset = try ArtifactArchive.json(directory.appendingPathComponent("manifest.json"))
                        guard dataset.fields?["id"]?.uuid == datasetID,
                              let dependencies = try? dataset.required("sources").decode([JSONValue].self) else { throw ArtifactArchive.error("A saved dataset has invalid dependency metadata.") }
                        for dependency in dependencies { recordings.insert(try dependency.requiredUUID("id")) }
                        try add(.dataset, datasetID.uuidString.lowercased(), "Dataset · " + String(datasetID.uuidString.prefix(8)), directory)
                    }
                    if let actorRunID = configuration.fields?["actorRunID"]?.uuid {
                        let desktopDirectory = root.appendingPathComponent("DesktopRuns/\(actorRunID.uuidString.lowercased())", isDirectory: true)
                        let desktop = try ArtifactArchive.json(desktopDirectory.appendingPathComponent("configuration.json"))
                        let binding = try desktop.required("rewardBinding").decode(RewardProgramBinding.self)
                        let scope = try desktop.required("scope").decode(ControlScope.self)
                        _ = try binding.validated(scope: scope)
                        guard configuration.fields?["environment"]?.fields?["reward_signature"]?.text == binding.definitionSignature else {
                            throw ArtifactArchive.error("The desktop checkpoint's reward definition does not match its saved task.")
                        }
                        rewards[binding.definition.id] = binding.definition
                        try add(.desktopConfiguration, actorRunID.uuidString.lowercased(), "Desktop task · " + String(actorRunID.uuidString.prefix(8)),
                            desktopDirectory, only: ["configuration.json"])
                    }
                }
            }
            for id in recordings.subtracting(visitedRecordings).sorted(by: { $0.uuidString < $1.uuidString }) {
                guard let recording = snapshot.recordings.first(where: { $0.id == id }), recording.status != .recording else {
                    throw ArtifactArchive.error("A required recording is missing or still active: \(id).")
                }
                try layout.requireAvailable(.recordings)
                let directory = recordingDirectory(id: id)
                try add(.recording, id.uuidString.lowercased(), recording.name, directory)
                catalog.recordings.append(recording); visitedRecordings.insert(id)
                if let reference = recording.correction {
                    let prelude = try CorrectionPrelude.load(in: directory, reference: reference)
                    checkpoints.insert(prelude.sourceCheckpointID)
                }
            }
        }
        var agentIDs = Set(catalog.checkpoints.map(\.agentID)); agentIDs.insert(agentID)
        agentIDs.formUnion(catalog.recordings.compactMap(\.recordedForAgentID))
        agentIDs.formUnion(catalog.runs.map(\.agentID))
        for id in agentIDs.sorted(by: { $0.uuidString < $1.uuidString }) {
            guard var agent = allAgents.first(where: { $0.id == id }) else { throw ArtifactArchive.error("A source creator agent is unavailable in the catalog.") }
            if let selected = agent.selectedCheckpointID, !checkpoints.contains(selected) { agent.selectedCheckpointID = nil }
            if let rewardID = agent.rewardProgramID, let reward = snapshot.rewardPrograms.first(where: { $0.id == rewardID }) {
                if let frozen = rewards[rewardID], frozen != reward { throw ArtifactArchive.error("A current reward definition conflicts with the checkpoint's frozen task identity.") }
                rewards[rewardID] = reward
            }
            else { agent.rewardProgramID = nil }
            catalog.agents.append(agent)
            if let environmentID = agent.environmentID, let environment = snapshot.environments.first(where: { $0.id == environmentID }), !catalog.environments.contains(where: { $0.id == environmentID }) {
                catalog.environments.append(environment)
            }
            for fieldID in agent.contextFieldIDs ?? [] {
                guard let field = snapshot.contextFields.first(where: { $0.id == fieldID }) else { throw ArtifactArchive.error("A selected context field is unavailable.") }
                if !catalog.contexts.contains(where: { $0.id == fieldID }) { catalog.contexts.append(field) }
            }
            let sourceRecordings = try self.recordingIDs(for: id).intersection(recordings)
            let sourceCheckpoints = try self.checkpointIDs(for: id).intersection(checkpoints)
            catalog.links.append(.init(agentID: id,
                recordingSelections: Dictionary(uniqueKeysWithValues: sourceRecordings.map { ($0, snapshot.recordingSelections[id]?[$0] ?? .whole) }),
                checkpointIDs: sourceCheckpoints.sorted { $0.uuidString < $1.uuidString }))
        }
        catalog.rewards = rewards.values.sorted { $0.id.uuidString < $1.id.uuidString }
        for digest in Set(catalog.rewards.flatMap { $0.signals.compactMap(\.templateDigest) }).sorted() {
            _ = try RewardAssets.read(digest, root: root)
            try add(.rewardAsset, digest, "Reward image · " + String(digest.prefix(8)), root.appendingPathComponent("RewardTemplates"), only: [digest + ".image"])
        }
        let id = UUID()
        let notices = ["The preview includes all saved dataset, recording, correction-policy and reward-image dependencies needed by these checkpoints.",
            "Optimizer and random-state bytes are preserved. Saved configurations retain their original provenance paths; runtime resolution uses artifact identities in the receiving library.",
            "Unfinished feedback, raw desktop rollouts, claim files and inference logs are not exported. Continue those workflows in their original library. Historical training logs are not included.",
            "Recordings contain original screen content and input evidence. Export only to a destination where you want those data copied."]
        let archive = ArtifactArchiveManifest(id: id, createdAt: Date(timeIntervalSince1970: floor(Date().timeIntervalSince1970)),
            items: items.sorted { $0.id < $1.id }, catalog: catalog, notices: notices)
        try ArtifactArchive.validate(archive)
        let archiveBytes = try ArtifactArchive.encoded(archive)
        for item in archive.items { try ArtifactArchive.validateContent(item, directory: sources[item.id]!, catalog: catalog, items: archive.items) }
        return .init(id: id, archive: archive, sources: sources, archiveSHA256: ArtifactArchive.digest(archiveBytes))
    }

    public func exportArtifactArchive(_ plan: ArtifactExportPlan, to destination: URL,
        progress: @Sendable (ArtifactTransferProgress) -> Void = { _ in },
        cancelled: @Sendable () -> Bool = { false }) async throws {
        try ArtifactArchive.validate(plan.archive)
        let header = try ArtifactArchive.encoded(plan.archive)
        guard plan.id == plan.archive.id, ArtifactArchive.digest(header) == plan.archiveSHA256,
              destination.isFileURL, Set(plan.sources.keys) == Set(plan.items.map(\.id)) else {
            throw ArtifactArchive.error("Choose a new archive destination and a complete export preview.")
        }
        try ArtifactTransferFiles.requireDirectory(destination.deletingLastPathComponent())
        let stage = destination.deletingLastPathComponent().appendingPathComponent(".astra-export-" + plan.id.uuidString.lowercased(), isDirectory: true)
        try recordTransfer(id: plan.id, operation: "export", status: "copying",
            document: ArtifactExportJournal(plan: plan, destination: destination), createdAt: plan.archive.createdAt)
        do {
            try ArtifactTransferFiles.check(cancelled)
            if FileManager.default.fileExists(atPath: destination.path) {
                let (_, digest) = try ArtifactArchive.read(destination, cancelled: cancelled)
                guard digest == plan.archiveSHA256 else { throw ArtifactArchive.error("The export destination already contains different archive bytes. Nothing was overwritten.") }
            } else {
                if !FileManager.default.fileExists(atPath: stage.path) { try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false) }
                var expected = [ArtifactFileEntry(path: ArtifactArchive.filename, byteCount: UInt64(header.count), sha256: plan.archiveSHA256)]
                expected += plan.items.flatMap { item in item.files.map { .init(path: item.relativePath + "/" + $0.path, byteCount: $0.byteCount, sha256: $0.sha256) } }
                try ArtifactTransferFiles.recoverStagingTemps(stage, planID: plan.id, entries: expected)
                var completed: UInt64 = 0
                for item in plan.items {
                    try ArtifactTransferFiles.check(cancelled)
                    let target = stage.appendingPathComponent(item.relativePath, isDirectory: true)
                    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
                    let alreadyCopied = completed
                    try ArtifactTransferFiles.copy(item.files, from: plan.sources[item.id]!, to: target,
                        progress: { value in progress(.init(phase: value.phase, completedBytes: alreadyCopied + value.completedBytes,
                            totalBytes: plan.totalBytes, name: item.name)) }, cancelled: cancelled)
                    completed += item.bytes
                }
                let metadata = stage.appendingPathComponent(ArtifactArchive.filename)
                if FileManager.default.fileExists(atPath: metadata.path) {
                    guard try ArtifactTransferFiles.read(metadata, limit: ArtifactArchive.metadataLimit) == header else {
                        throw ArtifactArchive.error("The staged archive metadata differs from the approved export plan.")
                    }
                } else {
                    let temporary = stage.appendingPathComponent(".astra-copy-" + UUID().uuidString.lowercased())
                    defer { try? FileManager.default.removeItem(at: temporary) }
                    try ArtifactTransferFiles.write(header, to: temporary)
                    try ArtifactTransferFiles.publish(temporary, to: metadata)
                }
                progress(.init(phase: "Verifying archive", completedBytes: plan.totalBytes, totalBytes: plan.totalBytes, name: destination.lastPathComponent))
                let (_, digest) = try ArtifactArchive.read(stage, cancelled: cancelled)
                guard digest == plan.archiveSHA256 else { throw ArtifactArchive.error("The staged archive header failed verification.") }
                try ArtifactTransferFiles.syncTreeDirectories(stage)
                try ArtifactTransferFiles.check(cancelled)
                try ArtifactTransferFiles.publish(stage, to: destination)
            }
            try database.execute("UPDATE artifact_transfers SET status='completed',issue=NULL WHERE id=?", [.text(plan.id.uuidString)])
            progress(.init(phase: "Archive exported", completedBytes: plan.totalBytes, totalBytes: plan.totalBytes, name: destination.lastPathComponent))
        } catch {
            try? transferFailed(plan.id, issue: error is CancellationError ? "Export cancelled. Original artifacts and the verified partial copy are retained; retry when ready." : error.localizedDescription)
            throw error
        }
    }
}
