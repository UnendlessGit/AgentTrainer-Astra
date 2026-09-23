import Foundation

public enum PendingFeedbackStatus: String, Codable, Sendable {
    case awaitingReview, reviewing, readyToContinue, learning, completed, failed, discarded
}

/// References metadata artifacts only. The catalog never owns mutable copies of
/// source frames, policy weights, or learner state.
public struct PendingFeedbackArtifactReference: Codable, Equatable, Sendable {
    public let id: UUID
    public let sha256: String
    public let path: String

    public init(id: UUID, sha256: String, path: String) {
        self.id = id; self.sha256 = sha256; self.path = path
    }

    fileprivate func validate() throws {
        try FeedbackChecks.digest(sha256)
        try PendingFeedbackChecks.path(path)
    }
}

public struct PendingFeedbackFragment: Codable, Equatable, Sendable {
    public let collectionID: UUID
    public let sourceDirectory: String
    public let manifestSHA256: String
    public let revisionDirectory: String
    /// Parent-first order, checked against the actual artifacts when reopened.
    public var revisions: [FeedbackRevisionReference]
    public var draft: PendingFeedbackArtifactReference?
    public var reviewedPackage: PendingFeedbackArtifactReference?
    /// Original native join/progress evidence; never re-created after reopening.
    public let boundary: JSONValue

    public init(collectionID: UUID, sourceDirectory: String, manifestSHA256: String, revisionDirectory: String,
                revisions: [FeedbackRevisionReference] = [], draft: PendingFeedbackArtifactReference? = nil,
                reviewedPackage: PendingFeedbackArtifactReference? = nil, boundary: JSONValue) {
        self.collectionID = collectionID; self.sourceDirectory = sourceDirectory; self.manifestSHA256 = manifestSHA256
        self.revisionDirectory = revisionDirectory; self.revisions = revisions; self.draft = draft
        self.reviewedPackage = reviewedPackage; self.boundary = boundary
    }

    fileprivate func validate() throws {
        try PendingFeedbackChecks.path(sourceDirectory)
        try PendingFeedbackChecks.path(revisionDirectory)
        try FeedbackChecks.digest(manifestSHA256)
        guard revisions.count <= 4_096, Set(revisions.map(\.id)).count == revisions.count,
              reviewedPackage == nil || !revisions.isEmpty else {
            throw PendingFeedbackChecks.error("The saved feedback revision chain is duplicated, unbounded, or missing from a reviewed package.")
        }
        for revision in revisions { try FeedbackChecks.digest(revision.sha256) }
        try draft?.validate(); try reviewedPackage?.validate()
        try PendingFeedbackChecks.snapshot(boundary, maximumBytes: 8 * 1024 * 1024)
    }
}

/// Durable navigation/provenance for one behavior batch. The referenced source,
/// revisions and packages must still pass their own integrity/admission checks.
/// A saved workflow status is never evidence that learning or control completed.
public struct PendingFeedbackDocument: Codable, Equatable, Identifiable, Sendable {
    public static let maximumBytes = 32 * 1024 * 1024
    public var schemaVersion = 1
    public let behaviorBatchID: UUID
    public var id: UUID { behaviorBatchID }
    public let agentID: UUID
    public let checkpoint: CheckpointDocument
    public let createdAt: Date
    public var modifiedAt: Date
    public var status: PendingFeedbackStatus
    /// Environment, training, context, target and resume-learner configuration.
    public let configuration: JSONValue
    public var fragments: [PendingFeedbackFragment]

    public init(behaviorBatchID: UUID, agentID: UUID, checkpoint: CheckpointDocument,
                createdAt: Date = Date(), modifiedAt: Date? = nil, status: PendingFeedbackStatus = .awaitingReview,
                configuration: JSONValue, fragments: [PendingFeedbackFragment]) {
        self.behaviorBatchID = behaviorBatchID; self.agentID = agentID; self.checkpoint = checkpoint
        self.createdAt = createdAt; self.modifiedAt = modifiedAt ?? createdAt; self.status = status
        self.configuration = configuration; self.fragments = fragments
    }

    public func validated() throws -> Self {
        _ = try checkpoint.validated()
        guard schemaVersion == 1, createdAt.timeIntervalSince1970.isFinite,
              modifiedAt.timeIntervalSince1970.isFinite, modifiedAt >= createdAt,
              (1...1_024).contains(fragments.count), Set(fragments.map(\.collectionID)).count == fragments.count,
              Set(fragments.map(\.sourceDirectory)).count == fragments.count,
              Set(fragments.map(\.revisionDirectory)).count == fragments.count else {
            throw PendingFeedbackChecks.error("The pending feedback catalog contains invalid dates or duplicated source identities.")
        }
        try PendingFeedbackChecks.snapshot(configuration, maximumBytes: 256 * 1024)
        var revisionIDs: Set<UUID> = [], draftIDs: Set<UUID> = [], packageIDs: Set<UUID> = []
        for fragment in fragments {
            try fragment.validate()
            for reference in fragment.revisions {
                guard revisionIDs.insert(reference.id).inserted else {
                    throw PendingFeedbackChecks.error("A feedback revision belongs to more than one fragment.")
                }
            }
            if let draft = fragment.draft, !draftIDs.insert(draft.id).inserted {
                throw PendingFeedbackChecks.error("A feedback draft belongs to more than one fragment.")
            }
            if let package = fragment.reviewedPackage, !packageIDs.insert(package.id).inserted {
                throw PendingFeedbackChecks.error("A reviewed package belongs to more than one fragment.")
            }
        }
        guard try PendingFeedbackChecks.encoded(self).count <= Self.maximumBytes else {
            throw PendingFeedbackChecks.error("The pending feedback document exceeds its metadata size limit.")
        }
        return self
    }

    /// Only new fragments/revisions and review workflow state can be appended.
    func validateUpdate(from previous: Self) throws {
        guard behaviorBatchID == previous.behaviorBatchID, agentID == previous.agentID,
              checkpoint.matchesIdentity(of: previous.checkpoint),
              abs(createdAt.timeIntervalSince(previous.createdAt)) < 0.001,
              modifiedAt.timeIntervalSince(previous.modifiedAt) > -0.001,
              try PendingFeedbackChecks.encoded(configuration) == PendingFeedbackChecks.encoded(previous.configuration),
              fragments.count >= previous.fragments.count else {
            throw PendingFeedbackChecks.error("A pending feedback update cannot change its behavior batch, checkpoint, configuration or original creation time.")
        }
        for (old, new) in zip(previous.fragments, fragments) {
            guard old.collectionID == new.collectionID, old.sourceDirectory == new.sourceDirectory,
                  old.manifestSHA256 == new.manifestSHA256, old.revisionDirectory == new.revisionDirectory,
                  try PendingFeedbackChecks.encoded(old.boundary) == PendingFeedbackChecks.encoded(new.boundary),
                  new.revisions.starts(with: old.revisions) else {
                throw PendingFeedbackChecks.error("A pending feedback update cannot remove, reorder or rewrite frozen source fragments or saved revisions.")
            }
            if old.reviewedPackage != nil, old.reviewedPackage != new.reviewedPackage,
               new.revisions.count == old.revisions.count {
                throw PendingFeedbackChecks.error("Replacing a reviewed package requires an appended feedback revision.")
            }
            for (oldReference, newReference) in [(old.draft, new.draft), (old.reviewedPackage, new.reviewedPackage)] {
                if let oldReference, let newReference,
                   (oldReference.id == newReference.id || oldReference.path == newReference.path), oldReference != newReference {
                    throw PendingFeedbackChecks.error("A changed feedback artifact requires a new identity and path.")
                }
            }
        }
    }
}

private enum PendingFeedbackChecks {
    static func error(_ message: String) -> AstraError { AstraError("feedback.pending", message) }

    static func path(_ value: String) throws {
        guard value.hasPrefix("/"), value.utf8.count <= 4_096, value != "/",
              !value.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains),
              !value.split(separator: "/", omittingEmptySubsequences: false).dropFirst().contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
            throw error("Feedback references require bounded, absolute local paths without traversal or empty components.")
        }
    }

    static func encoded<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return try encoder.encode(value)
    }

    static func snapshot(_ value: JSONValue, maximumBytes: Int) throws {
        guard value.fields != nil else { throw error("A saved feedback snapshot must be a JSON metadata object.") }
        var nodes = 0
        func visit(_ item: JSONValue, depth: Int) throws {
            nodes += 1
            guard nodes <= 524_288, depth <= 32 else { throw error("A saved feedback snapshot exceeds its structure limit.") }
            switch item {
            case .object(let fields):
                for (key, value) in fields {
                    guard key.utf8.count <= 256 else { throw error("A saved feedback snapshot contains an oversized field name.") }
                    try visit(value, depth: depth + 1)
                }
            case .array(let values): for value in values { try visit(value, depth: depth + 1) }
            case .string(let text):
                guard text.utf8.count <= 16_384 else { throw error("A saved feedback snapshot contains an oversized metadata string.") }
            case .number(let number):
                guard number.isFinite else { throw error("A saved feedback snapshot contains a nonfinite value.") }
            default: break
            }
        }
        try visit(value, depth: 0)
        guard try encoded(value).count <= maximumBytes else { throw error("A saved feedback snapshot exceeds its metadata size limit.") }
    }
}
