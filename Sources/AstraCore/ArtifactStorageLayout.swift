import Foundation

public enum ArtifactStorageKind: String, Codable, CaseIterable, Sendable { case recordings, models }

/// Captured once by each coordinator. A successful migration requires reopening
/// the store/coordinators before another workflow may begin.
public struct ArtifactStorageLayout: Codable, Hashable, Sendable {
    public var schemaVersion = 1
    public let catalogRoot: URL
    public let recordingsRoot: URL
    public let modelsRoot: URL
    public let recordingsIdentity: UUID
    public let modelsIdentity: UUID
    public let revision: UUID

    public static func defaults(catalogRoot: URL) -> Self {
        Self(catalogRoot: catalogRoot.standardizedFileURL,
            recordingsRoot: catalogRoot.appendingPathComponent("Recordings", isDirectory: true).standardizedFileURL,
            modelsRoot: catalogRoot.appendingPathComponent("Models", isDirectory: true).standardizedFileURL,
            recordingsIdentity: UUID(), modelsIdentity: UUID(), revision: UUID())
    }
    static func initial(catalogRoot: URL) throws -> Self {
        let candidate = defaults(catalogRoot: catalogRoot)
        func identity(_ kind: ArtifactStorageKind) throws -> UUID {
            let path = candidate.root(for: kind).appendingPathComponent(ArtifactStorageMarker.filename)
            guard FileManager.default.fileExists(atPath: path.path) else { return candidate.identity(for: kind) }
            let marker = try JSONDecoder().decode(ArtifactStorageMarker.self, from: ArtifactTransferFiles.read(path, limit: 4096))
            guard marker.schemaVersion == 1, marker.kind == kind else { throw AstraError("storage.identity", "An existing storage marker belongs to another artifact kind.") }
            return marker.identity
        }
        return Self(catalogRoot: candidate.catalogRoot, recordingsRoot: candidate.recordingsRoot, modelsRoot: candidate.modelsRoot,
            recordingsIdentity: try identity(.recordings), modelsIdentity: try identity(.models), revision: candidate.revision)
    }
    public func root(for kind: ArtifactStorageKind) -> URL { kind == .recordings ? recordingsRoot : modelsRoot }
    public func identity(for kind: ArtifactStorageKind) -> UUID { kind == .recordings ? recordingsIdentity : modelsIdentity }
    public func recordingDirectory(id: UUID) -> URL { recordingsRoot.appendingPathComponent(id.uuidString + ".astrarecord", isDirectory: true) }
    public func checkpointDirectory(id: UUID) -> URL { modelsRoot.appendingPathComponent(id.uuidString.lowercased(), isDirectory: true) }
    func replacing(_ kind: ArtifactStorageKind, root: URL) -> Self {
        Self(catalogRoot: catalogRoot, recordingsRoot: kind == .recordings ? root : recordingsRoot,
            modelsRoot: kind == .models ? root : modelsRoot, recordingsIdentity: recordingsIdentity,
            modelsIdentity: modelsIdentity, revision: UUID())
    }
    @discardableResult public func validated() throws -> Self {
        guard schemaVersion == 1, [catalogRoot, recordingsRoot, modelsRoot].allSatisfy({
            $0.isFileURL && $0.path.hasPrefix("/") && $0.path != "/" && $0.path.utf8.count <= 4096
                && $0.standardizedFileURL.path == $0.path
        }), recordingsRoot != modelsRoot,
              !ArtifactTransferFiles.contains(recordingsRoot, modelsRoot), !ArtifactTransferFiles.contains(modelsRoot, recordingsRoot) else {
            throw AstraError("storage.layout", "The artifact storage locations are invalid or overlap.")
        }
        return self
    }
    public func requireAvailable(_ selected: ArtifactStorageKind? = nil) throws {
        _ = try validated()
        for kind in selected.map({ [$0] }) ?? ArtifactStorageKind.allCases { try ArtifactStorageMarker.verify(layout: self, kind: kind) }
    }
}

struct ArtifactStorageMarker: Codable, Equatable {
    static let filename = ".astra-storage.json"
    var schemaVersion = 1
    let identity: UUID
    let kind: ArtifactStorageKind

    static func verify(layout: ArtifactStorageLayout, kind: ArtifactStorageKind) throws {
        let root = layout.root(for: kind)
        try ArtifactTransferFiles.requireDirectory(root)
        let marker = try JSONDecoder().decode(Self.self, from: ArtifactTransferFiles.read(root.appendingPathComponent(filename), limit: 4096))
        guard marker == Self(identity: layout.identity(for: kind), kind: kind) else {
            throw AstraError("storage.identity", "The connected \(kind.rawValue) folder is not the storage selected by this library.")
        }
    }
    static func create(layout: ArtifactStorageLayout, kind: ArtifactStorageKind, at root: URL? = nil) throws {
        let destination = (root ?? layout.root(for: kind)).appendingPathComponent(filename)
        let marker = Self(identity: layout.identity(for: kind), kind: kind)
        if FileManager.default.fileExists(atPath: destination.path) {
            guard try JSONDecoder().decode(Self.self, from: ArtifactTransferFiles.read(destination, limit: 4096)) == marker else {
                throw AstraError("storage.identity", "A storage folder has a different identity.")
            }
            return
        }
        try ArtifactTransferFiles.write(JSONEncoder().encode(marker), to: destination)
        try ArtifactTransferFiles.sync(destination.deletingLastPathComponent())
    }
}
