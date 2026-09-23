import Foundation
import AstraCore
import AstraPlatform

struct DesktopLearningProgress: Sendable {
    let phase: String
    let completedUpdates: Int
    let completedEpisodes: Int
    let learningDecisions: Int
}

struct DesktopStoppedCollection: Sendable {
    let result: WireMessage
    let checkpoint: PolicyActorCheckpoint
    let joins: [DesktopEpisodeJoin]
    let pause: PolicyActorLearningPause
}
struct DesktopReviewBatch: Sendable {
    let collection: DesktopStoppedCollection
    let completedUpdates: Int
    let resumeLearner: Bool
}

/// Concrete owners are supplied by the native host. Every operation completes
/// only after its corresponding helper/source ownership has joined. These
/// closures also allow the complete loop to run against owned virtual targets.
struct DesktopLearningOperations: Sendable {
    let collector: @Sendable (CheckpointDocument, JSONValue?, UUID) async throws -> CollectorSession
    let reset: @Sendable (UUID, ResetCancellation) async throws -> ResetResult
    let warmup: @Sendable (ResetResult) async throws -> Void
    let episode: @Sendable (ResetResult, CheckpointDocument, CollectorSession) throws -> DesktopEpisodeRunner
    let learn: @Sendable (DesktopLearningBatch, Bool, @escaping @Sendable () async throws -> Void) async throws -> ExternalLearningResult
    let stopLearning: @Sendable () async -> Void
    let preserveStopped: @Sendable (DesktopStoppedCollection, @escaping @Sendable () async throws -> Void) async throws -> CheckpointDocument?
    var review: (@Sendable (DesktopReviewBatch) async throws -> DesktopLearningBatch?)? = nil
}

struct DesktopLearningLoopResult: Sendable {
    let checkpoint: PolicyActorCheckpoint
    let completedUpdates: Int
    let completedEpisodes: Int
    let stopped: Bool
    let issue: AstraError?
    var pendingReviewID: UUID? = nil
}

/// Complete episodes share an unchanged categorical actor until their rollout
/// reaches its minimum. Learning and checkpoint publication hold an exclusive
/// idle-actor pause. New weights activate only after the next physical reset.
/// The host owns final actor/capture shutdown and workspace/UI lifecycle.
final class DesktopLearningLoop: @unchecked Sendable {
    private let actor: PolicyActorSession
    private let identity: DesktopEvidenceIdentity
    private let configuration: DesktopLearningConfiguration
    private let initial: PolicyActorCheckpoint
    private let initialProgress: JSONValue?
    private let firstIteration: Int, targetIterations: Int
    private let resume: Bool
    private let carriedDecisions: Int
    private let operations: DesktopLearningOperations
    private let onProgress: @Sendable (DesktopLearningProgress) -> Void
    private let lock = NSLock()
    private var work: Task<DesktopLearningLoopResult, Never>?
    private var stopping = false
    private var finished = false
    private var issue: AstraError?
    private var currentEpisode: DesktopEpisodeRunner?
    private var currentResetCancellation: ResetCancellation?
    private var learning = false
    private var stopLearningWork: Task<Void, Never>?
    var activeEpisodeID: UUID? { lock.withLock { currentEpisode?.episodeID } }

    init(actor: PolicyActorSession, identity: DesktopEvidenceIdentity, configuration: DesktopLearningConfiguration,
         checkpoint: PolicyActorCheckpoint, previousActorProgress: JSONValue?, completedUpdates: Int,
         targetUpdates: Int, resume: Bool, operations: DesktopLearningOperations,
         carriedDecisions: Int = 0,
         onProgress: @escaping @Sendable (DesktopLearningProgress) -> Void = { _ in }) throws {
        guard (0..<100_000).contains(completedUpdates), (completedUpdates + 1...100_000).contains(targetUpdates),
              identity.environmentID == configuration.environmentID,
              !configuration.retrospective || operations.review != nil else {
            throw AstraError("desktop.updateTarget", "Choose a total update target greater than the saved iteration count.")
        }
        self.actor = actor; self.identity = identity; self.configuration = configuration; initial = checkpoint
        initialProgress = previousActorProgress; firstIteration = completedUpdates; targetIterations = targetUpdates
        self.resume = resume; self.operations = operations; self.onProgress = onProgress
        self.carriedDecisions = max(0, carriedDecisions)
    }

    func requestStop() {
        let admission = lock.withLock { () -> (Bool, DesktopEpisodeRunner?, ResetCancellation?) in
            guard !finished else { return (false, nil, nil) }
            stopping = true; return (true, currentEpisode, currentResetCancellation)
        }
        guard admission.0 else { return }
        admission.1?.requestStop(); admission.2?.cancel()
        lock.withLock {
            if learning, stopLearningWork == nil { stopLearningWork = Task { await operations.stopLearning() } }
        }
    }
    func fail(_ error: AstraError) {
        let episode = lock.withLock { issue = issue ?? error; stopping = true; return currentEpisode }
        if let episode { episode.sourceFailed(error, generation: episode.generationID) }
        requestStop()
    }
    func run() async -> DesktopLearningLoopResult {
        let task = lock.withLock { () -> Task<DesktopLearningLoopResult, Never> in
            if let work { return work }
            let task = Task { await self.perform() }; work = task; return task
        }
        return await withTaskCancellationHandler { await task.value } onCancel: { self.requestStop() }
    }

    private func perform() async -> DesktopLearningLoopResult {
        var current = initial, previousProgress = initialProgress
        var updates = firstIteration, episodes = 0, shouldResume = resume
        var carried = carriedDecisions
        var collector: CollectorSession?
        var heldPause: PolicyActorLearningPause?
        var pendingReviewID: UUID?
        func report(_ phase: String, decisions: Int = 0) {
            onProgress(.init(phase: phase, completedUpdates: updates, completedEpisodes: episodes, learningDecisions: decisions))
        }
        do {
            while updates < targetIterations, !lock.withLock({ stopping }) {
                report("Preparing episode collection…")
                let collectionID = UUID()
                let collecting = try await operations.collector(current.document, previousProgress, collectionID)
                collector = collecting
                var joins: [DesktopEpisodeJoin] = [], decisions = carried, produced = 0, auditOnly = false
                while decisions < configuration.minimumDecisions, !lock.withLock({ stopping }) {
                    report("Preparing the next episode…", decisions: decisions)
                    let cancellation = ResetCancellation()
                    let stopReset = lock.withLock { currentResetCancellation = cancellation; return stopping }
                    if stopReset { cancellation.cancel() }
                    let ready = try await operations.reset(UUID(), cancellation)
                    lock.withLock { currentResetCancellation = nil }
                    guard ready.cleanupConfirmed else {
                        throw ready.issue ?? AstraError("desktop.resetCleanup", "Reset control cleanup remains unconfirmed.")
                    }
                    if lock.withLock({ stopping }) { break }
                    guard ready.status == .ready else {
                        throw ready.issue ?? AstraError("desktop.reset", "The environment did not reach a confirmed starting condition.")
                    }
                    _ = try await actor.reset(confirmedEpisodeID: ready.context.nextEpisodeID,
                        contextIDs: configuration.contextIDs, activating: current)
                    if lock.withLock({ stopping }) { break }
                    do { try await operations.warmup(ready) }
                    catch is CancellationError where lock.withLock({ stopping }) { break }
                    if lock.withLock({ stopping }) { break }
                    let episode = try operations.episode(ready, current.document, collecting)
                    let stopNow = lock.withLock { currentEpisode = episode; return stopping }
                    if stopNow { episode.requestStop() }
                    report("Collecting desktop experience…", decisions: decisions)
                    let result = await episode.run()
                    lock.withLock { currentEpisode = nil }
                    produced += result.producedDecisions
                    if let proof = result.join { joins.append(proof) }
                    guard result.cleanupConfirmed, result.actorState.sampledProgressKnown else {
                        throw result.issue ?? AstraError("desktop.episodeJoin", "The episode did not join with known actor progress and released controls.")
                    }
                    if let error = result.issue { throw error }
                    if case .ended(_, let count, _, _, let audited) = result.evidence {
                        episodes += 1; decisions += count; auditOnly = auditOnly || audited
                    }
                    switch result.stop {
                    case .semanticBoundary: break
                    case .operatorAbort: requestStop()
                    case .failure(let reason): throw AstraError("desktop.episode", reason)
                    }
                }
                if let issue = lock.withLock({ issue }) { throw issue }
                if produced == 0 {
                    await collecting.abandon(reason: "No actor decision was produced in this collection.")
                    collector = nil
                    if lock.withLock({ stopping }) { break }
                    throw AstraError("desktop.emptyCollection", "The environment produced no complete policy episode.")
                }
                let pause = try await actor.acquireLearningPause(); heldPause = pause
                let stopped = lock.withLock { stopping }
                if stopped && !auditOnly {
                    try collecting.offer(.request("collector.abort", .object(["reason": .string("The operator stopped before the next learner update.")])))
                }
                report(stopped ? "Saving the stopped actor boundary…" : "Sealing completed episodes…", decisions: decisions)
                let sealed = try await collecting.finish(); collector = nil
                var reviewedBatch: DesktopLearningBatch?
                try await actor.validateLearningPause(pause)
                if sealed.payload.fields?["manifest"]?.fields?["status"] == .string("awaiting_manual_review") {
                    pendingReviewID = sealed.payload.fields?["manifest"]?.fields?["behaviorBatchID"]?.uuid
                    guard let review = operations.review, pendingReviewID != nil else {
                        throw AstraError("desktop.reviewOwner", "The saved experience requires its feedback-review owner.")
                    }
                    report("Waiting for your review…", decisions: decisions)
                    guard let reviewed = try await review(.init(collection: .init(result: sealed, checkpoint: current, joins: joins, pause: pause),
                        completedUpdates: updates, resumeLearner: shouldResume)) else { requestStop(); break }
                    reviewedBatch = reviewed
                }
                if lock.withLock({ stopping }) {
                    if pendingReviewID != nil { break } // Retain original policy identity for suspended reviewed experience.
                    let boundary = DesktopStoppedCollection(result: sealed, checkpoint: current, joins: joins, pause: pause)
                    let saved = try await operations.preserveStopped(boundary, { [actor] in try await actor.validateLearningPause(pause) })
                    if let saved { current = nextCheckpoint(saved, beside: current) }
                    try await actor.releaseLearningPause(pause); heldPause = nil
                    break
                }
                let batch = try reviewedBatch ?? DesktopLearningBatch(identity: identity, checkpoint: current.document,
                    destination: URL(fileURLWithPath: try sealed.payload.required("path").decode(String.self)),
                    result: sealed, joins: joins, actorState: pause.binding.state)
                try await batch.validateHeldPause(actor: actor, token: pause)
                report("Updating the learner with controls released…", decisions: decisions)
                lock.withLock { learning = true; stopLearningWork = nil }
                let learned: ExternalLearningResult?
                do {
                    learned = try await operations.learn(batch, shouldResume, { [self, actor] in
                        try await batch.validateHeldPause(actor: actor, token: pause)
                        // Stop may precede the child coordinator's begin call.
                        // Recheck from its admitted boundary callback so that
                        // an early no-op cancellation cannot start new work.
                        if lock.withLock({ stopping }) { await operations.stopLearning() }
                    })
                } catch is CancellationError where lock.withLock({ stopping }) {
                    learned = nil
                } catch {
                    // The immutable source checkpoint remains the last known
                    // optimizer/weight state. Preserve its newer real actor
                    // cursor when the sealed package still authenticates, but
                    // retain the learner error even if that recovery succeeds.
                    lock.withLock { learning = false }
                    await lock.withLock({ stopLearningWork })?.value
                    do {
                        try await batch.validateHeldPause(actor: actor, token: pause)
                        if pendingReviewID == nil {
                            let boundary = DesktopStoppedCollection(result: sealed, checkpoint: current, joins: joins, pause: pause)
                            if let saved = try await operations.preserveStopped(boundary, { [actor] in try await batch.validateHeldPause(actor: actor, token: pause) }) {
                                current = nextCheckpoint(saved, beside: current)
                            }
                        }
                    } catch let preservationError {
                        lock.withLock { issue = .init("desktop.learningAndBoundary", "\(error.localizedDescription)\nThe latest actor boundary also could not be saved: \(preservationError.localizedDescription)") }
                    }
                    throw error
                }
                lock.withLock { learning = false }
                await lock.withLock({ stopLearningWork })?.value
                guard let learned, let saved = learned.checkpoint, let progress = learned.actorProgress else {
                    if learned == nil || learned?.cancelled == true {
                        requestStop()
                        if pendingReviewID != nil { break }
                        let boundary = DesktopStoppedCollection(result: sealed, checkpoint: current, joins: joins, pause: pause)
                        if let saved = try await operations.preserveStopped(boundary, { [actor] in try await actor.validateLearningPause(pause) }) {
                            current = nextCheckpoint(saved, beside: current)
                        }
                        break
                    }
                    throw AstraError("desktop.checkpoint", "The learner did not publish a resumable desktop checkpoint.")
                }
                current = nextCheckpoint(saved, beside: current); previousProgress = progress; shouldResume = true
                carried = 0
                pendingReviewID = nil
                if !learned.cancelled { updates += 1 }
                try await actor.releaseLearningPause(pause); heldPause = nil
                if learned.cancelled { requestStop() }
            }
        } catch {
            if !(error is CancellationError && lock.withLock({ stopping })) {
                lock.withLock { issue = issue ?? (error as? AstraError) ?? .init("desktop.learning", error.localizedDescription) }
            }
            requestStop()
        }
        await lock.withLock({ stopLearningWork })?.value
        if let collector { await collector.abandon(reason: lock.withLock { issue?.message ?? "Desktop learning stopped before collection closure." }) }
        if let heldPause { try? await actor.releaseLearningPause(heldPause) }
        lock.withLock { finished = true; learning = false; currentEpisode = nil; currentResetCancellation = nil }
        return .init(checkpoint: current, completedUpdates: updates, completedEpisodes: episodes,
                     stopped: lock.withLock { stopping }, issue: lock.withLock { issue }, pendingReviewID: pendingReviewID)
    }

    private func nextCheckpoint(_ document: CheckpointDocument, beside old: PolicyActorCheckpoint) -> PolicyActorCheckpoint {
        .init(document: document, directory: old.directory.deletingLastPathComponent().appendingPathComponent(document.id.uuidString.lowercased(), isDirectory: true))
    }
}
