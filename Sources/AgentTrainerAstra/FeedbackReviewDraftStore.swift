import Foundation
import Darwin
import AstraCore

/// Drafts are a separate non-learning artifact, including counts not yet
/// reviewed. Every checkpoint has a new ID; cancellation never overwrites a
/// revision or interprets an unfinished cell as zero.
enum FeedbackReviewDraftStore {
    static func save(_ draft: FeedbackReviewDraft, source: VerifiedFeedbackSource, parent: LoadedFeedbackRevision?,
                     directory: URL) async throws -> FeedbackReviewDraftReference {
        try await Task.detached {
            _ = try draft.validated(source: source, parent: parent)
            let data = try FeedbackArtifactStore.encoded(draft)
            let folder = directory.appendingPathComponent("Drafts", isDirectory: true)
            guard folder.isFileURL else { throw AstraError("feedback.draftPath", "Review drafts require local storage.") }
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let descriptor = open(folder.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else { throw AstraError("feedback.draftPath", "The review draft directory is unavailable or linked.") }
            defer { close(descriptor) }
            let staging = ".draft-" + UUID().uuidString.lowercased(), name = draft.id.uuidString.lowercased() + ".json"
            let file = openat(descriptor, staging, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
            guard file >= 0 else { throw AstraError("feedback.draftWrite", "The review draft could not be created.") }
            defer { close(file); unlinkat(descriptor, staging, 0) }
            try data.withUnsafeBytes { bytes in
                var offset = 0
                while offset < bytes.count {
                    let written = Darwin.write(file, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                    if written < 0 && errno == EINTR { continue }
                    guard written > 0 else { throw AstraError("feedback.draftWrite", "The review draft could not be fully written.") }
                    offset += written
                }
            }
            guard fsync(file) == 0, linkat(descriptor, staging, descriptor, name, 0) == 0, fsync(descriptor) == 0 else {
                throw AstraError("feedback.draftPublish", "The review draft could not be durably published. Your edits remain in this window.")
            }
            return .init(id: draft.id, sha256: FeedbackArtifactStore.digest(data), url: folder.appendingPathComponent(name))
        }.value
    }

    static func load(_ reference: FeedbackReviewDraftReference, source: VerifiedFeedbackSource,
                     parent: LoadedFeedbackRevision?) async throws -> FeedbackReviewDraft {
        try await Task.detached {
            guard reference.url.isFileURL, reference.url.lastPathComponent == reference.id.uuidString.lowercased() + ".json" else {
                throw AstraError("feedback.draftIdentity", "The draft has an inconsistent file identity.")
            }
            let fd = open(reference.url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
            guard fd >= 0 else { throw AstraError("feedback.draftMissing", "The review draft is missing or linked.") }
            let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? handle.close() }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size > 0, info.st_size <= FeedbackLimits.maximumBytes,
                  let data = try handle.read(upToCount: FeedbackLimits.maximumBytes + 1), data.count == info.st_size,
                  FeedbackArtifactStore.digest(data) == reference.sha256 else { throw AstraError("feedback.draftIntegrity", "The review draft failed its integrity check.") }
            let draft = try JSONDecoder().decode(FeedbackReviewDraft.self, from: data)
            guard draft.id == reference.id else { throw AstraError("feedback.draftIdentity", "The draft belongs to another review.") }
            return try draft.validated(source: source, parent: parent)
        }.value
    }
}
