import Foundation
import CryptoKit
import Darwin

public struct ArtifactTransferProgress: Sendable {
    public let phase: String
    public let completedBytes: UInt64
    public let totalBytes: UInt64
    public let name: String
}

public struct ArtifactFileEntry: Codable, Hashable, Sendable {
    public let path: String
    public let byteCount: UInt64
    public let sha256: String
}

enum ArtifactTransferFiles {
    static let maximumFiles = 100_000
    static let maximumBytes: UInt64 = 16 * 1024 * 1024 * 1024 * 1024
    static func contains(_ root: URL, _ other: URL) -> Bool {
        other.standardizedFileURL.path.hasPrefix(root.standardizedFileURL.path + "/")
    }
    static func validPath(_ path: String) -> Bool {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return !path.isEmpty && path.utf8.count <= 4096 && parts.count <= 32
            && parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." && $0.utf8.count <= 255
                && !$0.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) }
    }
    static func requireDirectory(_ url: URL) throws {
        guard url.isFileURL else { throw error("Artifact storage must be a local folder.") }
        let parts = url.standardizedFileURL.pathComponents
        var path = URL(fileURLWithPath: "/", isDirectory: true)
        for (index, part) in parts.dropFirst().enumerated() {
            path.appendPathComponent(part, isDirectory: true)
            // macOS itself aliases these roots into /private. Resolve only
            // that known system alias; package/user directory links still fail.
            if index == 0, ["var", "tmp"].contains(part),
               ["private/" + part, "/private/" + part].contains((try? FileManager.default.destinationOfSymbolicLink(atPath: path.path)) ?? "") {
                path = URL(fileURLWithPath: "/private/" + part, isDirectory: true)
            }
            var information = stat()
            guard lstat(path.path, &information) == 0, information.st_mode & S_IFMT == S_IFDIR else {
                throw error("The local folder is missing, disconnected or resolves through a symbolic link: \(url.lastPathComponent).")
            }
        }
    }
    static func read(_ url: URL, limit: Int) throws -> Data {
        let fd = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw error("A required artifact file is unavailable: \(url.lastPathComponent).") }
        let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? file.close() }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size >= 0, info.st_size <= limit,
              let bytes = try file.read(upToCount: limit + 1), bytes.count == info.st_size else {
            throw error("An artifact metadata file is invalid or exceeds its size limit.")
        }
        return bytes
    }
    static func write(_ data: Data, to url: URL) throws {
        let fd = open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw error("The destination already exists or cannot be written: \(url.lastPathComponent).") }
        let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? file.close() }
        try file.write(contentsOf: data); try file.synchronize()
    }
    static func sync(_ directory: URL) throws {
        let fd = open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw error("The artifact directory could not be synchronized.") }
        defer { close(fd) }
        guard fsync(fd) == 0 else { throw error("The artifact directory could not be synchronized.") }
    }
    static func publish(_ staged: URL, to destination: URL) throws {
        guard renamex_np(staged.path, destination.path, UInt32(RENAME_EXCL)) == 0 else {
            throw error("The artifact destination already exists or publication failed: \(destination.lastPathComponent).")
        }
        try sync(destination.deletingLastPathComponent())
    }
    static func inventory(_ root: URL, excluding: Set<String> = [], cancelled: @Sendable () -> Bool = { false }) throws -> [ArtifactFileEntry] {
        try requireDirectory(root)
        var entries: [ArtifactFileEntry] = [], folders = [(root, "")], total: UInt64 = 0
        while let (folder, prefix) = folders.popLast() {
            try check(cancelled)
            guard let listing = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: nil, options: [.skipsSubdirectoryDescendants]) else {
                throw error("The artifact directory could not be enumerated.")
            }
            for case let url as URL in listing {
                let path = prefix + url.lastPathComponent
                if excluding.contains(path) { continue }
                guard validPath(path), entries.count + folders.count < maximumFiles else { throw error("The artifact tree exceeds its path or entry limits.") }
                var info = stat()
                guard lstat(url.path, &info) == 0 else { throw error("An artifact changed while it was being inspected.") }
                if info.st_mode & S_IFMT == S_IFDIR { folders.append((url, path + "/")) }
                else {
                    guard info.st_mode & S_IFMT == S_IFREG else { throw error("Artifact packages cannot contain symbolic links or special files.") }
                    let entry = try fingerprint(url, relativePath: path, cancelled: cancelled)
                    let sum = total.addingReportingOverflow(entry.byteCount)
                    guard !sum.overflow, sum.partialValue <= maximumBytes else { throw error("Artifact storage exceeds the supported transfer size.") }
                    total = sum.partialValue; entries.append(entry)
                }
            }
        }
        return entries.sorted { $0.path < $1.path }
    }
    static func fingerprint(_ url: URL, relativePath: String, cancelled: @Sendable () -> Bool) throws -> ArtifactFileEntry {
        let fd = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw error("An artifact file became unavailable.") }
        let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? file.close() }
        var before = stat(), after = stat(), path = stat()
        guard fstat(fd, &before) == 0, before.st_mode & S_IFMT == S_IFREG, before.st_size >= 0,
              UInt64(before.st_size) <= maximumBytes else { throw error("An artifact file has invalid type or size.") }
        var hash = SHA256(), count: UInt64 = 0
        while let bytes = try file.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty {
            try check(cancelled); hash.update(data: bytes); count += UInt64(bytes.count)
        }
        guard fstat(fd, &after) == 0, lstat(url.path, &path) == 0,
              before.st_dev == after.st_dev, before.st_ino == after.st_ino, before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec, before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              path.st_dev == before.st_dev, path.st_ino == before.st_ino, count == UInt64(before.st_size) else {
            throw error("An artifact changed during checksum verification.")
        }
        return .init(path: relativePath, byteCount: count, sha256: hash.finalize().map { String(format: "%02x", $0) }.joined())
    }
    static func copy(_ entries: [ArtifactFileEntry], from source: URL, to destination: URL,
                     progress: @Sendable (ArtifactTransferProgress) -> Void, cancelled: @Sendable () -> Bool) throws {
        try requireDirectory(source); try requireDirectory(destination)
        let total = entries.reduce(UInt64(0)) { $0 + $1.byteCount }
        var completed: UInt64 = 0
        for entry in entries {
            try check(cancelled)
            guard validPath(entry.path) else { throw error("The archive contains a path outside its artifact.") }
            let from = source.appendingPathComponent(entry.path), to = destination.appendingPathComponent(entry.path)
            try FileManager.default.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            try requireDirectory(from.deletingLastPathComponent()); try requireDirectory(to.deletingLastPathComponent())
            if FileManager.default.fileExists(atPath: to.path) {
                guard try fingerprint(to, relativePath: entry.path, cancelled: cancelled) == entry else { throw error("A staged or existing artifact differs from this transfer.") }
                completed += entry.byteCount; progress(.init(phase: "Copying", completedBytes: completed, totalBytes: total, name: entry.path)); continue
            }
            let inputFD = open(from.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
            guard inputFD >= 0 else { throw error("The source artifact is unavailable.") }
            let input = FileHandle(fileDescriptor: inputFD, closeOnDealloc: true); defer { try? input.close() }
            var info = stat()
            guard fstat(inputFD, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size >= 0,
                  UInt64(info.st_size) == entry.byteCount else { throw error("The source artifact changed after preview.") }
            let temporary = to.deletingLastPathComponent().appendingPathComponent(".astra-copy-" + UUID().uuidString.lowercased())
            let outputFD = open(temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard outputFD >= 0 else { throw error("The destination cannot create an artifact copy.") }
            let output = FileHandle(fileDescriptor: outputFD, closeOnDealloc: true)
            defer { try? output.close(); try? FileManager.default.removeItem(at: temporary) }
            var hash = SHA256(), count: UInt64 = 0
            while let bytes = try input.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty {
                try check(cancelled); count += UInt64(bytes.count)
                guard count <= entry.byteCount else { throw error("The source artifact grew during copying.") }
                hash.update(data: bytes); try output.write(contentsOf: bytes)
                progress(.init(phase: "Copying", completedBytes: completed + count, totalBytes: total, name: entry.path))
            }
            guard count == entry.byteCount, hash.finalize().map({ String(format: "%02x", $0) }).joined() == entry.sha256 else {
                throw error("The copied artifact failed its checksum. The original remains intact.")
            }
            try output.synchronize(); try output.close()
            try publish(temporary, to: to); completed += count
        }
        try syncTreeDirectories(destination)
    }
    static func syncTreeDirectories(_ root: URL) throws {
        try requireDirectory(root)
        guard let listing = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey]) else { throw error("The copied directory cannot be synchronized.") }
        var folders = [root], count = 0
        for case let url as URL in listing {
            count += 1
            guard count <= maximumFiles else { throw error("The copied directory exceeds its entry limit.") }
            let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isSymbolicLink != true else { throw error("The copied directory contains a symbolic link.") }
            if values.isDirectory == true { folders.append(url) }
        }
        for folder in folders.sorted(by: { $0.pathComponents.count > $1.pathComponents.count }) { try sync(folder) }
    }
    /// Call only after the catalog has matched this exact durable transfer plan.
    /// A hard crash skips copy's defer; only our UUID-named partial files in the
    /// plan's staging directory may be removed. Expected source names survive.
    static func recoverStagingTemps(_ stage: URL, planID: UUID, entries: [ArtifactFileEntry]) throws {
        guard stage.lastPathComponent.hasPrefix(".astra-"), stage.lastPathComponent.contains(planID.uuidString.lowercased()) else {
            throw error("Temporary-file recovery requires the journal-owned staging directory.")
        }
        try requireDirectory(stage)
        let expected = Set(entries.map(\.path))
        var folders = [(stage, "")], visited = 0
        while let (folder, prefix) = folders.popLast() {
            try requireDirectory(folder)
            guard let listing = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: nil, options: [.skipsSubdirectoryDescendants]) else { throw error("The transfer staging directory cannot be read.") }
            for case let url as URL in listing {
                visited += 1
                guard visited <= maximumFiles else { throw error("The staging directory exceeds its entry limit.") }
                var info = stat()
                guard lstat(url.path, &info) == 0 else { throw error("The staging directory changed during recovery.") }
                let relative = prefix + url.lastPathComponent
                if info.st_mode & S_IFMT == S_IFDIR { folders.append((url, relative + "/")); continue }
                guard info.st_mode & S_IFMT == S_IFREG else { throw error("Transfer staging contains a linked or special file.") }
                if !expected.contains(relative), url.lastPathComponent.hasPrefix(".astra-copy-"),
                   UUID(uuidString: String(url.lastPathComponent.dropFirst(".astra-copy-".count))) != nil {
                    guard info.st_nlink == 1, unlink(url.path) == 0 else { throw error("An interrupted temporary copy could not be removed safely.") }
                }
            }
        }
    }
    static func check(_ cancelled: @Sendable () -> Bool) throws { if cancelled() || Task.isCancelled { throw CancellationError() } }
    static func error(_ message: String) -> AstraError { .init("artifact.transfer", message) }
}
