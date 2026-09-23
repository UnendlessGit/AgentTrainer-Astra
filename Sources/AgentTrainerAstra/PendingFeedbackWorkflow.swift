import Foundation
import AstraCore
import AstraPlatform

/// Review and reward publication are separate from the physical collector.
/// This owner may run with a live paused actor or a suspended immutable source.
@MainActor final class PendingFeedbackWorkflow {
    private let store: LibraryStore
    private let learner: LearningCoordinator
    private let root: URL
    private let validate: @Sendable () async throws -> Void
    private let present: @MainActor (FeedbackReviewPresentation?) -> Void
    private let changed: @MainActor () async -> Void
    private var presentation: FeedbackReviewPresentation?
    private var cancelled = false
    private var computeActive = false
    private let authorSession = UUID(), authorClock = UUID()

    init(store: LibraryStore, learner: LearningCoordinator, root: URL,
         validate: @escaping @Sendable () async throws -> Void,
         present: @escaping @MainActor (FeedbackReviewPresentation?) -> Void,
         changed: @escaping @MainActor () async -> Void) {
        self.store = store; self.learner = learner; self.root = root; self.validate = validate
        self.present = present; self.changed = changed
    }
    func cancel() async {
        cancelled = true
        if computeActive { await learner.requestStop() }
        await presentation?.cancelAndJoin()
    }
    private func job(_ kind: String, _ payload: JSONValue, agent: UUID) async throws -> JSONValue {
        try await validate()
        if cancelled { throw CancellationError() }
        computeActive = true; defer { computeActive = false }
        return try await learner.feedbackOperation(agentID: agent, kind: kind, payload: payload)
    }
    private func save(_ value: inout PendingFeedbackDocument) async throws {
        value.modifiedAt = Date(); try await store.savePendingFeedback(value); await changed()
    }

    /// Returns a verified materialized batch only when every fragment has
    /// explicit complete judgments and the original learning minimum is met.
    func run(_ original: PendingFeedbackDocument, showReview: Bool = true) async throws -> JSONValue? {
        var document = original
        var total = 0, minimum = 0
        for index in document.fragments.indices {
            let fragment = document.fragments[index]
            let inspected = try await job("feedback.inspect", .object(["sourcePath": .string(fragment.sourceDirectory),
                "manifestSHA256": .string(fragment.manifestSHA256)]), agent: document.agentID)
            guard inspected.fields?["behaviorBatchID"]?.uuid == document.id,
                  let count = inspected.fields?["decisions"]?.int, let required = inspected.fields?["minimumDecisions"]?.int,
                  count > 0, required > 0, minimum == 0 || minimum == required else {
                throw AstraError("feedback.pendingIdentity", "The saved review does not match its immutable behavior batch.")
            }
            total += count; minimum = required
            let reader = try await FeedbackSourceReader.open(inspection: inspected)
            guard reader.source.metadata.collectionID == fragment.collectionID,
                  reader.source.metadata.policyID == document.checkpoint.id,
                  reader.source.metadata.policySignature == document.checkpoint.policySignature else {
                throw AstraError("feedback.policy", "Review source and original checkpoint differ.")
            }
            let revisionDirectory = URL(fileURLWithPath: fragment.revisionDirectory, isDirectory: true)
            var parent: LoadedFeedbackRevision?
            for reference in fragment.revisions {
                let prior = parent
                parent = try await Task.detached { try FeedbackArtifactStore.load(reference, source: reader.source, directory: revisionDirectory, parent: prior) }.value
            }
            var draft: FeedbackReviewDraft?
            if let reference = fragment.draft {
                draft = try await FeedbackReviewDraftStore.load(.init(id: reference.id, sha256: reference.sha256,
                    url: URL(fileURLWithPath: reference.path)), source: reader.source, parent: parent)
            }
            while parent?.document.resolved.allSatisfy({ $0.manualReward != nil }) != true {
                if cancelled || !showReview { document.status = .awaitingReview; try await save(&document); return nil }
                try await validate()
                document.status = .reviewing; try await save(&document)
                let authorSession = authorSession, authorClock = authorClock
                let model = try FeedbackReviewModel(source: reader.source, parent: parent, artifactDirectory: revisionDirectory,
                    dependencies: .init(validateBoundary: validate, loadObservation: { try await reader.load($0) }, authoredNow: {
                        .init(sessionID: authorSession, clockID: authorClock, observedNanos: MonotonicClock.now,
                            wallTimeMS: UInt64(Date().timeIntervalSince1970 * 1000))
                    }), draft: draft)
                let review = FeedbackReviewPresentation(model: model)
                presentation = review; present(review)
                let outcome = await review.wait()
                // The model completes only after joining loaders/publication.
                present(nil); presentation = nil
                let reference: FeedbackReviewDraftReference
                var newRevision: LoadedFeedbackRevision?
                switch outcome {
                case .saved(let loaded, _, let savedDraft): reference = savedDraft; newRevision = loaded
                case .cancelled(let savedDraft, let retained):
                    reference = savedDraft; cancelled = true
                    if let retained {
                        let old = parent
                        newRevision = try await Task.detached { try FeedbackArtifactStore.load(retained, source: reader.source, directory: revisionDirectory, parent: old) }.value
                    }
                }
                let oldDraft = try await FeedbackReviewDraftStore.load(reference, source: reader.source, parent: parent)
                if let loaded = newRevision {
                    document.fragments[index].revisions.append(loaded.reference)
                    // The UI's durable pre-publication draft names its previous
                    // parent. Carry unreviewed edits into a new immutable draft.
                    let rebound = FeedbackReviewDraft(schemaVersion: 1, id: UUID(), sourceSHA256: oldDraft.sourceSHA256,
                        parent: loaded.reference, selectedPacketID: oldDraft.selectedPacketID, point: oldDraft.point, cells: oldDraft.cells)
                    let saved = try await FeedbackReviewDraftStore.save(rebound, source: reader.source, parent: loaded, directory: revisionDirectory)
                    document.fragments[index].draft = .init(id: saved.id, sha256: saved.sha256, path: saved.url.path)
                    parent = loaded; draft = rebound
                } else {
                    document.fragments[index].draft = .init(id: reference.id, sha256: reference.sha256, path: reference.url.path)
                    draft = oldDraft
                }
                document.status = cancelled ? .awaitingReview : .reviewing; try await save(&document)
                if cancelled { return nil }
            }
            if cancelled || !showReview { document.status = .readyToContinue; try await save(&document); return nil }
            if document.fragments[index].reviewedPackage == nil {
                let parentDirectory = root.appendingPathComponent("ReviewedExperience", isDirectory: true)
                try await Task.detached { try FileManager.default.createDirectory(at: parentDirectory, withIntermediateDirectories: true) }.value
                let destination = parentDirectory.appendingPathComponent(UUID().uuidString.lowercased(), isDirectory: true)
                let result = try await job("feedback.materialize", .object(["sourcePath": .string(fragment.sourceDirectory),
                    "manifestSHA256": .string(fragment.manifestSHA256), "revisionDirectory": .string(fragment.revisionDirectory),
                    "revisionChain": try .encode(document.fragments[index].revisions), "destination": .string(destination.path)]), agent: document.agentID)
                document.fragments[index].reviewedPackage = .init(id: try result.required("manifest").requiredUUID("id"),
                    sha256: try result.required("manifestSHA256").decode(String.self), path: try result.required("rolloutPath").decode(String.self))
                try await save(&document)
            }
        }
        document.status = .readyToContinue; try await save(&document)
        guard !cancelled, total >= minimum else { return nil }
        let directory = root.appendingPathComponent("ReviewedExperience", isDirectory: true)
        let destination = directory.appendingPathComponent(UUID().uuidString.lowercased(), isDirectory: true)
        let fragments = try document.fragments.map { fragment -> JSONValue in
            guard let package = fragment.reviewedPackage else { throw AstraError("feedback.incomplete", "A feedback fragment has not been fully reviewed.") }
            return .object(["path": .string(package.path), "manifestSHA256": .string(package.sha256)])
        }
        return try await job("feedback.combine", .object(["fragments": .array(fragments), "destination": .string(destination.path)]), agent: document.agentID)
    }
}
