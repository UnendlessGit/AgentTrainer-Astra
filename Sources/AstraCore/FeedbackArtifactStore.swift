import Foundation
import Darwin

public struct LoadedFeedbackRevision: Sendable {
    public let reference: FeedbackRevisionReference
    public let document: FeedbackRewardRevision
    fileprivate init(reference: FeedbackRevisionReference, document: FeedbackRewardRevision) {
        self.reference = reference; self.document = document
    }
}

/// Separate, immutable reward metadata. This store never edits source frames,
/// observations, actions, execution/outcome evidence or existing revisions.
public enum FeedbackArtifactStore {
    public static func encoded<T: Encodable>(_ value: T) throws -> Data { try FeedbackChecks.encode(value) }
    public static func digest(_ data: Data) -> String { FeedbackChecks.sha256(data) }

    @discardableResult
    public static func publish(_ revision: FeedbackRewardRevision, source: VerifiedFeedbackSource, directory: URL,
                               parent: LoadedFeedbackRevision? = nil) throws -> LoadedFeedbackRevision {
        try revision.validate(source: source, parent: parent)
        let data = try FeedbackChecks.encode(revision)
        let reference = FeedbackRevisionReference(id: revision.id, sha256: FeedbackChecks.sha256(data))
        guard directory.isFileURL else { throw FeedbackChecks.error("path", "Feedback artifacts require a local directory.") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let folder = try openDirectory(directory)
        defer { close(folder) }
        let staging = ".feedback-" + UUID().uuidString.lowercased(), name = filename(revision.id)
        let file = openat(folder, staging, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard file >= 0 else { throw FeedbackChecks.error("write", "The feedback staging file could not be created.") }
        defer { close(file); unlinkat(folder, staging, 0) }
        try data.withUnsafeBytes { bytes in
            var written = 0
            while written < bytes.count {
                let result = Darwin.write(file, bytes.baseAddress!.advanced(by: written), bytes.count - written)
                if result < 0 && errno == EINTR { continue }
                guard result > 0 else { throw FeedbackChecks.error("write", "The feedback artifact could not be written completely.") }
                written += result
            }
        }
        guard fsync(file) == 0 else { throw FeedbackChecks.error("sync", "The feedback artifact could not be synchronized.") }
        // linkat is atomic and cannot replace an existing destination, including
        // an existing symlink. The published inode is complete before exposure.
        guard linkat(folder, staging, folder, name, 0) == 0 else {
            throw FeedbackChecks.error(errno == EEXIST ? "immutable" : "publish", "A feedback revision cannot replace an existing artifact.")
        }
        guard fsync(folder) == 0 else { throw FeedbackChecks.error("sync", "The published feedback directory could not be synchronized.") }
        return .init(reference: reference, document: revision)
    }

    public static func load(_ reference: FeedbackRevisionReference, source: VerifiedFeedbackSource, directory: URL,
                            parent: LoadedFeedbackRevision? = nil) throws -> LoadedFeedbackRevision {
        try FeedbackChecks.digest(reference.sha256)
        let folder = try openDirectory(directory)
        defer { close(folder) }
        let file = openat(folder, filename(reference.id), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard file >= 0 else { throw FeedbackChecks.error("missing", "The feedback artifact is missing or is a symbolic link.") }
        let handle = FileHandle(fileDescriptor: file, closeOnDealloc: true)
        defer { try? handle.close() }
        var info = stat()
        guard fstat(file, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_size > 0, info.st_size <= FeedbackLimits.maximumBytes,
              let data = try handle.read(upToCount: FeedbackLimits.maximumBytes + 1), data.count == info.st_size,
              FeedbackChecks.sha256(data) == reference.sha256 else {
            throw FeedbackChecks.error("integrity", "The feedback artifact failed its size, file type or expected digest check.")
        }
        let document = try FeedbackChecks.decode(FeedbackRewardRevision.self, data: data)
        guard document.id == reference.id else { throw FeedbackChecks.error("identity", "The feedback artifact has a different revision identity.") }
        try document.validate(source: source, parent: parent)
        return .init(reference: reference, document: document)
    }

    public static func filename(_ id: UUID) -> String { id.uuidString.lowercased() + ".json" }
    private static func openDirectory(_ url: URL) throws -> Int32 {
        guard url.isFileURL else { throw FeedbackChecks.error("path", "Feedback artifacts require a local directory.") }
        let file = open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard file >= 0 else { throw FeedbackChecks.error("directory", "Feedback artifacts require a regular, unlinked directory.") }
        return file
    }
}
