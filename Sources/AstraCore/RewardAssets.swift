import Foundation
import CryptoKit
import Darwin

/// Content-addressed reward templates are outside the signed app. They are
/// verified on every read and never overwritten by an edited definition.
public enum RewardAssets {
    private static func validatedDigest(_ digest: String) throws {
        guard digest.count == 64, digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw AstraError("reward.assetID", "A reward template has an invalid identity.")
        }
    }
    public static func save(_ data: Data, root: URL) throws -> String {
        guard !data.isEmpty, data.count <= 16 * 1024 * 1024 else { throw AstraError("reward.assetSize", "A reward template must be smaller than 16 MB.") }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let folder = root.appendingPathComponent("RewardTemplates", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let properties = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard properties.isDirectory == true, properties.isSymbolicLink != true else { throw AstraError("reward.assetFolder", "Reward templates require a regular local directory.") }
        let final = folder.appendingPathComponent(digest + ".image")
        if FileManager.default.fileExists(atPath: final.path) { _ = try read(digest, root: root); return digest }
        let staging = folder.appendingPathComponent(".template-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: staging) }
        try data.write(to: staging, options: .withoutOverwriting)
        let descriptor = open(staging.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw AstraError("reward.assetWrite", "The staged reward template could not be opened.") }
        defer { close(descriptor) }
        guard fsync(descriptor) == 0 else { throw AstraError("reward.assetWrite", "The reward template could not be synchronized.") }
        if link(staging.path, final.path) != 0 {
            guard errno == EEXIST else { throw AstraError("reward.assetWrite", "The reward template could not be published.") }
            _ = try read(digest, root: root)
        }
        return digest
    }
    public static func read(_ digest: String, root: URL) throws -> Data {
        try validatedDigest(digest)
        let url = root.appendingPathComponent("RewardTemplates").appendingPathComponent(digest + ".image")
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw AstraError("reward.assetMissing", "A saved reward image template is missing or linked.") }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true); defer { try? handle.close() }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size > 0, info.st_size <= 16 * 1024 * 1024 else {
            throw AstraError("reward.assetSize", "A reward template is not a regular image file within the size limit.")
        }
        guard let data = try handle.read(upToCount: 16 * 1024 * 1024 + 1), data.count == Int(info.st_size),
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == digest else {
            throw AstraError("reward.assetIntegrity", "A reward template failed its integrity check.")
        }
        return data
    }
}
