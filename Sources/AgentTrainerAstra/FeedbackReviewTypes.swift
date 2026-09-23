import Foundation
import AstraCore

struct FeedbackObservationRequest: Sendable, Equatable {
    let sourceSHA256: String
    let trajectorySHA256: String
    let episodeID: UUID
    let observationID: UUID
    let cutoffNanos: UInt64
    let maximumBytes: Int
    let maximumFrames: Int
}

struct FeedbackReviewObservation: Sendable {
    struct Frame: Sendable {
        let metadata: FrameMetadata
        let pixels: Data
        /// Expected checksum from the verified immutable source frame index,
        /// not a new checksum substituted for a corrupt source by the loader.
        let pixelSHA256: String
    }
    let sourceSHA256: String
    let trajectorySHA256: String
    let episodeID: UUID
    let observationID: UUID
    let cutoffNanos: UInt64
    let frames: [Frame]
}

struct FeedbackReviewDependencies: Sendable {
    /// Parent retains the actor-pause token and joined cleanup/source ownership
    /// for the whole review lifetime. A successful read is not a new lease.
    let validateBoundary: @Sendable () async throws -> Void
    /// Returns verified original observation/frame identity and owned BGRA.
    /// Completion/cancellation must join the loader's I/O. No synthetic fallback.
    let loadObservation: @Sendable (FeedbackObservationRequest) async throws -> FeedbackReviewObservation
    let authoredNow: @Sendable () -> FeedbackAuthorship
    var maximumPreviewBytes = FrameArchive.maximumFrameBytes
}

enum FeedbackObservationPoint: String, Codable, CaseIterable, Sendable { case before, after }

struct FeedbackReviewDraft: Codable, Sendable {
    struct Cell: Codable, Sendable {
        let packetID: UUID, ruleID: UUID
        var count: Int
        var reviewed: Bool
        var pair: FeedbackReviewPair { .init(packetID: packetID, ruleID: ruleID) }
    }
    let schemaVersion: Int
    let id: UUID
    let sourceSHA256: String
    let parent: FeedbackRevisionReference?
    let selectedPacketID: UUID
    let point: FeedbackObservationPoint
    let cells: [Cell]

    func validated(source: VerifiedFeedbackSource, parent expectedParent: LoadedFeedbackRevision?) throws -> Self {
        let packets = Set(source.metadata.intervals.map(\.target.packetID)), rules = Set(source.metadata.manualRules.map(\.id))
        guard schemaVersion == 1, sourceSHA256 == source.sha256, parent == expectedParent?.reference,
              packets.contains(selectedPacketID), cells.count <= FeedbackLimits.maximumPairs,
              Set(cells.map(\.pair)).count == cells.count,
              cells.allSatisfy({ packets.contains($0.packetID) && rules.contains($0.ruleID) && (0...FeedbackLimits.maximumCount).contains($0.count) }) else {
            throw AstraError("feedback.draft", "The saved review draft differs from this source, revision or supported count range.")
        }
        return self
    }
}

struct FeedbackReviewDraftReference: Codable, Sendable {
    let id: UUID
    let sha256: String
    let url: URL
}

enum FeedbackReviewOutcome: Sendable {
    case saved(LoadedFeedbackRevision, complete: Bool, draft: FeedbackReviewDraftReference)
    case cancelled(FeedbackReviewDraftReference, retainedRevision: FeedbackRevisionReference?)
}
