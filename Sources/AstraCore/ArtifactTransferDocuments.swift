import Foundation

public enum ArtifactTransferKind: String, Codable, Sendable {
    case recording, checkpoint, dataset, runConfiguration, desktopConfiguration, rewardAsset
}
public struct ArtifactTransferItem: Codable, Hashable, Sendable, Identifiable {
    public let kind: ArtifactTransferKind
    public let identity: String
    public let name: String
    public let relativePath: String
    public let files: [ArtifactFileEntry]
    public var id: String { kind.rawValue + ":" + identity }
    public var bytes: UInt64 { artifactByteTotal(files.map(\.byteCount)) }
}
public struct StorageMigrationPlan: Codable, Hashable, Sendable, Identifiable {
    public let id: UUID
    public let kind: ArtifactStorageKind
    public let source: URL
    public let destination: URL
    public let originalLayout: ArtifactStorageLayout
    public let resultingLayout: ArtifactStorageLayout
    public let files: [ArtifactFileEntry]
    public let createdAt: Date
    public var items: [ArtifactFileEntry] { files }
    public var totalBytes: UInt64 { artifactByteTotal(files.map(\.byteCount)) }
}
public struct ArtifactCatalogLinks: Codable, Sendable {
    public let agentID: UUID
    public let recordingSelections: [UUID: RecordingTrainingSelection]
    public let checkpointIDs: [UUID]
}
public struct ArtifactCatalogBundle: Codable, Sendable {
    public var agents: [AgentDocument] = []
    public var environments: [EnvironmentDocument] = []
    public var contexts: [ContextFieldDocument] = []
    public var rewards: [RewardProgram] = []
    public var recordings: [RecordingManifest] = []
    public var checkpoints: [CheckpointDocument] = []
    public var runs: [LearningRunDocument] = []
    public var links: [ArtifactCatalogLinks] = []
}
public struct ArtifactArchiveManifest: Codable, Sendable {
    public var schemaVersion = 1
    public let id: UUID
    public let createdAt: Date
    public let items: [ArtifactTransferItem]
    public let catalog: ArtifactCatalogBundle
    public let notices: [String]
}
public struct ArtifactExportPlan: Codable, Sendable, Identifiable {
    public let id: UUID
    public let archive: ArtifactArchiveManifest
    public let sources: [String: URL]
    public let archiveSHA256: String
    public var items: [ArtifactTransferItem] { archive.items }
    public var totalBytes: UInt64 { artifactByteTotal(items.map(\.bytes)) }
    public var notices: [String] { archive.notices }
}
public struct ArtifactImportPlan: Codable, Sendable, Identifiable {
    public let id: UUID
    public let source: URL
    public let linkToAgentID: UUID?
    public let archive: ArtifactArchiveManifest
    public let archiveSHA256: String
    public var items: [ArtifactTransferItem] { archive.items }
    public var totalBytes: UInt64 { artifactByteTotal(items.map(\.bytes)) }
    public var notices: [String] { archive.notices }
}
public struct ArtifactImportResult: Sendable {
    public let agentIDs: [UUID]
    public let recordingIDs: [UUID]
    public let checkpointIDs: [UUID]
    public let importedCount: Int
    public let reusedCount: Int
    public let notices: [String]
}
public struct ArtifactTransferRecord: Identifiable, Sendable {
    public let id: UUID
    public let operation: String
    public let status: String
    public let issue: String?
    public let migration: StorageMigrationPlan?
    public let importPlan: ArtifactImportPlan?
    public let exportPlan: ArtifactExportPlan?
    public let exportDestination: URL?
}

struct ArtifactExportJournal: Codable {
    let plan: ArtifactExportPlan
    let destination: URL
}

private func artifactByteTotal(_ values: [UInt64]) -> UInt64 {
    values.reduce(0) { total, next in
        let sum = total.addingReportingOverflow(next)
        return sum.overflow ? UInt64.max : sum.partialValue
    }
}
