import Foundation

extension LibraryStore {
    public func previewStorageMigration(kind: ArtifactStorageKind, destination: URL) async throws -> StorageMigrationPlan {
        try layout.requireAvailable(kind)
        let destination = destination.standardizedFileURL
        guard destination.isFileURL, destination.path != "/", !FileManager.default.fileExists(atPath: destination.path),
              ![layout.recordingsRoot, layout.modelsRoot].contains(where: { $0 == destination
                  || ArtifactTransferFiles.contains($0, destination) || ArtifactTransferFiles.contains(destination, $0) }),
              destination != root, !ArtifactTransferFiles.contains(destination, root) else {
            throw AstraError("storage.destination", "Choose an unused destination folder outside the current artifact folders.")
        }
        try ArtifactTransferFiles.requireDirectory(destination.deletingLastPathComponent())
        let files = try ArtifactTransferFiles.inventory(layout.root(for: kind), excluding: [ArtifactStorageMarker.filename])
        return .init(id: UUID(), kind: kind, source: layout.root(for: kind), destination: destination,
            originalLayout: layout, resultingLayout: layout.replacing(kind, root: destination), files: files,
            createdAt: Date(timeIntervalSince1970: floor(Date().timeIntervalSince1970)))
    }

    /// Leaves every original byte in place. After success, reopen this library
    /// and all idle coordinators before admitting another workflow.
    public func applyStorageMigration(_ plan: StorageMigrationPlan,
        progress: @Sendable (ArtifactTransferProgress) -> Void = { _ in },
        cancelled: @Sendable () -> Bool = { false }) async throws -> ArtifactStorageLayout {
        _ = try plan.originalLayout.validated(); _ = try plan.resultingLayout.validated()
        guard plan.source == plan.originalLayout.root(for: plan.kind), plan.destination == plan.resultingLayout.root(for: plan.kind),
              plan.originalLayout.catalogRoot == root, plan.resultingLayout.catalogRoot == root,
              plan.originalLayout.identity(for: plan.kind) == plan.resultingLayout.identity(for: plan.kind),
              plan.originalLayout.root(for: plan.kind == .models ? .recordings : .models)
                == plan.resultingLayout.root(for: plan.kind == .models ? .recordings : .models) else {
            throw AstraError("storage.migrationIdentity", "The migration no longer matches this library's artifact ownership.")
        }
        let persisted = try currentSavedLayout()
        if persisted == plan.resultingLayout {
            try plan.resultingLayout.requireAvailable(plan.kind)
            return persisted
        }
        guard persisted == plan.originalLayout, layout == plan.originalLayout else {
            throw AstraError("storage.migrationChanged", "Storage locations changed after preview. Reopen the workspace and preview again.")
        }
        try plan.originalLayout.requireAvailable(plan.kind)
        let stage = plan.destination.deletingLastPathComponent().appendingPathComponent(".astra-transfer-" + plan.id.uuidString.lowercased(), isDirectory: true)
        try recordTransfer(id: plan.id, operation: "migration", status: "copying", document: plan, createdAt: plan.createdAt)
        do {
            try ArtifactTransferFiles.check(cancelled)
            guard try ArtifactTransferFiles.inventory(plan.source, excluding: [ArtifactStorageMarker.filename], cancelled: cancelled) == plan.files else {
                throw AstraError("storage.migrationChanged", "Source files changed after preview. No storage location was switched.")
            }
            if FileManager.default.fileExists(atPath: plan.destination.path) {
                try plan.resultingLayout.requireAvailable(plan.kind)
                guard try ArtifactTransferFiles.inventory(plan.destination, excluding: [ArtifactStorageMarker.filename], cancelled: cancelled) == plan.files else {
                    throw AstraError("storage.migrationCollision", "The existing destination differs from this interrupted migration.")
                }
            } else {
                if !FileManager.default.fileExists(atPath: stage.path) {
                    try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
                }
                try ArtifactTransferFiles.requireDirectory(stage)
                try ArtifactTransferFiles.recoverStagingTemps(stage, planID: plan.id, entries: plan.files)
                try ArtifactTransferFiles.copy(plan.files, from: plan.source, to: stage, progress: progress, cancelled: cancelled)
                guard try ArtifactTransferFiles.inventory(stage, excluding: [ArtifactStorageMarker.filename], cancelled: cancelled) == plan.files else {
                    throw AstraError("storage.migrationIntegrity", "The staged storage copy does not match the original files.")
                }
                try ArtifactStorageMarker.create(layout: plan.resultingLayout, kind: plan.kind, at: stage)
                try ArtifactTransferFiles.sync(stage)
                try ArtifactTransferFiles.check(cancelled)
                try ArtifactTransferFiles.publish(stage, to: plan.destination)
            }
            // Recheck originals before changing the routing pointer. The source
            // copy is retained even after successful publication and switching.
            guard try ArtifactTransferFiles.inventory(plan.source, excluding: [ArtifactStorageMarker.filename], cancelled: cancelled) == plan.files else {
                throw AstraError("storage.migrationChanged", "The original storage changed during copying. It remains the active location.")
            }
            try ArtifactTransferFiles.check(cancelled)
            try database.transaction {
                guard try currentSavedLayout() == plan.originalLayout else { throw AstraError("storage.migrationChanged", "The active storage locations changed.") }
                try database.execute("UPDATE settings SET value=? WHERE key='artifactStorageLayout'", [.blob(try JSONEncoder().encode(plan.resultingLayout))])
                try database.execute("UPDATE artifact_transfers SET status='completed',issue=NULL WHERE id=?", [.text(plan.id.uuidString)])
            }
            progress(.init(phase: "Storage switched · original files retained", completedBytes: plan.totalBytes, totalBytes: plan.totalBytes, name: plan.kind.rawValue))
            return plan.resultingLayout
        } catch {
            try? transferFailed(plan.id, issue: error is CancellationError ? "Copy cancelled. Original storage remains intact; retry this transfer when ready." : error.localizedDescription)
            throw error
        }
    }

    public func pendingArtifactTransfers() throws -> [ArtifactTransferRecord] {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        return try database.query("SELECT id,operation,status,CASE WHEN length(document)<=33554432 THEN document END AS document,issue FROM artifact_transfers WHERE status!='completed' ORDER BY created,id").map { row in
            guard let id = row["id"]?.string.flatMap(UUID.init(uuidString:)), let operation = row["operation"]?.string,
                  let status = row["status"]?.string, let data = row["document"]?.data else {
                throw AstraError("artifact.journal", "An interrupted artifact transfer has invalid recovery metadata.")
            }
            let exporting = operation == "export" ? try decoder.decode(ArtifactExportJournal.self, from: data) : nil
            return .init(id: id, operation: operation, status: status, issue: row["issue"]?.string,
                migration: operation == "migration" ? try decoder.decode(StorageMigrationPlan.self, from: data) : nil,
                importPlan: operation == "import" ? try decoder.decode(ArtifactImportPlan.self, from: data) : nil,
                exportPlan: exporting?.plan, exportDestination: exporting?.destination)
        }
    }
    func currentSavedLayout() throws -> ArtifactStorageLayout {
        guard let bytes = try database.query("SELECT value FROM settings WHERE key='artifactStorageLayout'").first?["value"]?.data else {
            throw AstraError("storage.layout", "The library has no saved storage routing.")
        }
        return try JSONDecoder().decode(ArtifactStorageLayout.self, from: bytes).validated()
    }
    func recordTransfer<T: Encodable>(id: UUID, operation: String, status: String, document: T, createdAt: Date) throws {
        let data = try encode(document)
        guard data.count <= 32 * 1024 * 1024 else { throw AstraError("artifact.journal", "The transfer metadata exceeds its recovery limit.") }
        if let previous = try database.query("SELECT operation,document FROM artifact_transfers WHERE id=?", [.text(id.uuidString)]).first {
            guard previous["operation"]?.string == operation, let saved = previous["document"]?.data,
                  try sameTransferPlan(saved, data, operation: operation) else {
                throw AstraError("artifact.journalIdentity", "A transfer retry differs from its original recovery plan.")
            }
            try database.execute("UPDATE artifact_transfers SET status=?,issue=NULL WHERE id=?", [.text(status), .text(id.uuidString)])
        } else {
            try database.execute("INSERT INTO artifact_transfers(id,operation,status,document,created) VALUES(?,?,?,?,?)", [
                .text(id.uuidString), .text(operation), .text(status), .blob(data), .real(createdAt.timeIntervalSince1970)])
        }
    }
    private func sameTransferPlan(_ lhs: Data, _ rhs: Data, operation: String) throws -> Bool {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        if operation == "migration" {
            let one = try decoder.decode(StorageMigrationPlan.self, from: lhs), two = try decoder.decode(StorageMigrationPlan.self, from: rhs)
            return one.id == two.id && one.kind == two.kind && one.source == two.source && one.destination == two.destination
                && one.originalLayout == two.originalLayout && one.resultingLayout == two.resultingLayout && one.files == two.files
        }
        if operation == "import" {
            let one = try decoder.decode(ArtifactImportPlan.self, from: lhs), two = try decoder.decode(ArtifactImportPlan.self, from: rhs)
            return one.id == two.id && one.source == two.source && one.linkToAgentID == two.linkToAgentID
                && one.archiveSHA256 == two.archiveSHA256 && one.archive.id == two.archive.id
        }
        if operation == "export" {
            let one = try decoder.decode(ArtifactExportJournal.self, from: lhs), two = try decoder.decode(ArtifactExportJournal.self, from: rhs)
            return one.plan.id == two.plan.id && one.destination == two.destination && one.plan.archiveSHA256 == two.plan.archiveSHA256
                && one.plan.sources == two.plan.sources && one.plan.archive.id == two.plan.archive.id
        }
        return lhs == rhs
    }
    func transferFailed(_ id: UUID, issue: String) throws {
        try database.execute("UPDATE artifact_transfers SET status='needsAttention',issue=? WHERE id=?", [.text(String(issue.prefix(8192))), .text(id.uuidString)])
    }
}
