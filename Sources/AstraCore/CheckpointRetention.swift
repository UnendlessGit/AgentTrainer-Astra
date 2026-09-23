import Foundation

public enum CheckpointCleanupDisposition: String, Codable, Sendable {
    case unlinkShared, deleteFiles, retryFiles
}

public struct CheckpointCleanupItem: Codable, Equatable, Sendable, Identifiable {
    public let checkpoint: CheckpointDocument
    public var id: UUID { checkpoint.id }
    public let disposition: CheckpointCleanupDisposition
    public let retainedByAgents: [String]
    public let bytes: UInt64
}

public struct CheckpointRetentionPreview: Equatable, Sendable {
    public let agentID: UUID
    public let keepNewest: Int
    public let items: [CheckpointCleanupItem]
    public let protected: [UUID: String]
    public let remainingCandidates: Int
    public var bytesToDelete: UInt64 {
        items.filter { $0.disposition != .unlinkShared }.reduce(0) {
            let sum = $0.addingReportingOverflow($1.bytes)
            return sum.overflow ? UInt64.max : sum.partialValue
        }
    }
}

public struct CheckpointCleanupResult: Sendable {
    public let unlinked: Int
    public let deleted: Int
    public let issues: [String]
}

enum CheckpointArtifactFiles {
    static func directory(modelsRoot: URL, id: UUID) throws -> URL {
        let models = modelsRoot
        let values = try models.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else {
            throw AstraError("checkpoint.cleanupPath", "The Models folder is unavailable or resolves through a symbolic link.")
        }
        let url = models.appendingPathComponent(id.uuidString.lowercased(), isDirectory: true)
        if FileManager.default.fileExists(atPath: url.path) {
            let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else {
                throw AstraError("checkpoint.cleanupPath", "A checkpoint folder is not a regular local directory.")
            }
        }
        return url
    }

    static func bytes(modelsRoot: URL, id: UUID) throws -> UInt64 {
        let directory = try directory(modelsRoot: modelsRoot, id: id)
        guard FileManager.default.fileExists(atPath: directory.path) else { return 0 }
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey, .fileSizeKey]
        var stack = [directory], count = 0, bytes: UInt64 = 0
        while let parent = stack.popLast() {
            for url in try FileManager.default.contentsOfDirectory(at: parent, includingPropertiesForKeys: Array(keys)) {
                count += 1
                guard count <= 4_096 else { throw AstraError("checkpoint.cleanupSize", "A checkpoint contains too many files for bounded cleanup. Inspect its folder manually.") }
                let value = try url.resourceValues(forKeys: keys)
                guard value.isSymbolicLink != true else { throw AstraError("checkpoint.cleanupPath", "A checkpoint contains a symbolic link and needs manual review.") }
                if value.isDirectory == true { stack.append(url) }
                else if value.isRegularFile == true, let size = value.fileSize, size >= 0 {
                    let total = bytes.addingReportingOverflow(UInt64(size))
                    guard !total.overflow else { throw AstraError("checkpoint.cleanupSize", "Checkpoint storage exceeds the supported range.") }
                    bytes = total.partialValue
                } else { throw AstraError("checkpoint.cleanupPath", "A checkpoint contains a nonregular file and needs manual review.") }
            }
        }
        return bytes
    }
}
