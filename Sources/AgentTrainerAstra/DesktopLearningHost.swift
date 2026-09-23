import Foundation
import Observation
import AstraCore
import AstraPlatform

/// Native workspace ownership around the environment-neutral episode/learner
/// loop. Capture stays alive, while each reset/episode owns fresh control.
@MainActor @Observable final class DesktopLearningHost {
    private(set) var isBusy = false
    private(set) var isStopping = false
    private(set) var agentID: UUID?
    private(set) var phase = ""
    private(set) var progress: DesktopLearningProgress?
    private(set) var metrics: [ReinforcementMetric] = []
    private(set) var failure: String?
    private(set) var checkpoint: CheckpointDocument?
    private(set) var awaitingReady: UUID?
    private(set) var countdown: Int?
    private(set) var cleanupWarning: NativeControlCompletion?
    private(set) var completedReportRunID: UUID?
    private(set) var reviewPresentation: FeedbackReviewPresentation?
    private var feedbackWorkflow: PendingFeedbackWorkflow?
    private var feedbackCancellation: Task<Void, Never>?
    private let store: LibraryStore
    private let root: URL
    private let learner: LearningCoordinator
    private let dependencies: InferenceDependencies
    private let controlFactory: NativeControlRuntimeFactory
    private let collectorFactory: CollectorRuntime.Factory
    private let verifyScope: NativeResetDriver.ScopeVerifier
    private let detector: RewardAnalysisQueue.Detector
    private let changed: @MainActor () async -> Void
    private var generation = UUID()
    private var work: Task<Void, Never>?
    private var readyWork: Task<Void, Never>?
    private var signals: DesktopHostSignals?
    private var resetRunner: ResetRunner?
    private var resetSource: CaptureSource?
    private var currentResetID: UUID?

    init(store: LibraryStore, root: URL, learner: LearningCoordinator, bundle: Bundle = .main,
         dependencies: InferenceDependencies? = nil, controlFactory: NativeControlRuntimeFactory? = nil,
         collectorFactory: CollectorRuntime.Factory? = nil,
         scopeVerifier: @escaping NativeResetDriver.ScopeVerifier = NativeResetDriver.verifyLiveScope,
         detector: @escaping RewardAnalysisQueue.Detector = { try VisualRewardDetector.read(signals: $0, frames: $1, episodeID: $2, templates: $3) },
         changed: @escaping @MainActor () async -> Void) {
        self.store = store; self.root = root; self.learner = learner; self.dependencies = dependencies ?? .live(bundle: bundle)
        self.controlFactory = controlFactory ?? .live(executable: bundle.bundleURL.appendingPathComponent("Contents/Helpers/AstraControl.app/Contents/MacOS/AstraControl"))
        self.collectorFactory = collectorFactory ?? CollectorRuntime.live(bundle: bundle); verifyScope = scopeVerifier
        self.detector = detector; self.changed = changed
    }

    func start(agent: AgentDocument, source: CaptureSource, program: RewardProgram, options: DesktopLearningOptions,
               continuation: PendingFeedbackDocument? = nil) throws {
        guard !isBusy, !learner.isBusy, dependencies.controlOwner.priorCleanupJoined else {
            throw AstraError("desktop.busy", "Finish learning and resolve any previous control cleanup before starting desktop training.")
        }
        _ = try options.validated(); _ = try program.validated()
        // The live manual source and retrospective-review workflow are wired
        // separately; missing feedback must never silently become zero reward.
        guard !program.signals.contains(where: { $0.kind == .manual }) else {
            throw AstraError("desktop.feedbackSource", "Manual state signals need a connected value source. Use visual signals and reviewable feedback rules for this session.")
        }
        if controlFactory.protectsPhysicalInputs {
            let privacy = PermissionSnapshot.current()
            guard privacy.screenRecording && privacy.inputMonitoring && privacy.accessibility && privacy.eventPosting else {
                throw AstraError("desktop.permissions", "Allow Screen Recording, Input Monitoring and Accessibility for Astra before starting desktop training.")
            }
        }
        generation = UUID(); let generation = generation
        let signals = DesktopHostSignals(); self.signals = signals
        isBusy = true; isStopping = false; agentID = agent.id; failure = nil; progress = nil; metrics = []; checkpoint = nil
        awaitingReady = nil; countdown = nil; cleanupWarning = nil; completedReportRunID = nil; phase = "Preparing desktop learning…"
        work = Task { await perform(agent: agent, source: source, program: program, options: options,
            generation: generation, signals: signals, continuation: continuation) }
    }

    func requestStop() {
        guard isBusy else { return }
        isStopping = true; phase = "Stopping and saving the joined actor boundary…"
        signals?.stop(); readyWork?.cancel()
        if let workflow = feedbackWorkflow, feedbackCancellation == nil {
            feedbackCancellation = Task { await workflow.cancel() }
        }
        // Before loop creation, this host can own policy preparation. Once the
        // loop exists it cancels its own learner operation at a safe boundary.
        if signals?.hasLoop != true {
            let expected = generation
            Task { [weak self] in
                guard let self, generation == expected, isBusy, signals?.hasLoop != true else { return }
                await learner.requestStop()
            }
        }
    }
    func stopAndWait() async {
        let pending = work
        requestStop(); await pending?.value
    }
    func confirmReady() {
        guard let id = awaitingReady, let runner = resetRunner, let source = resetSource,
              let signals, !isStopping, readyWork == nil else { return }
        let expected = generation
        readyWork = Task { [weak self] in
            guard let self else { return }
            defer { if generation == expected { readyWork = nil; countdown = nil } }
            do {
                try await activateWithCountdown(source, signals: signals)
                try signals.check()
                guard generation == expected, awaitingReady == id else { return }
                guard runner.confirmManualReady(resetID: id) else { throw AstraError("desktop.readyExpired", "That reset is no longer waiting for Ready.") }
                awaitingReady = nil
            } catch {
                if !(error is CancellationError) { signals.fail((error as? AstraError) ?? .init("desktop.ready", error.localizedDescription)) }
            }
        }
    }
    func acknowledgeManualCleanup() throws {
        guard !isBusy, let warning = cleanupWarning else { throw AstraError("desktop.cleanup", "There is no completed cleanup warning to acknowledge.") }
        try dependencies.controlOwner.acknowledgeManualCleanup(sessionID: warning.sessionID)
        cleanupWarning = nil
    }
    func cancelFeedbackReview() {
        guard let review = reviewPresentation else { return }
        Task { await review.cancelAndJoin() }
    }

    private func perform(agent: AgentDocument, source: CaptureSource, program: RewardProgram, options: DesktopLearningOptions,
                         generation expected: UUID, signals: DesktopHostSignals, continuation: PendingFeedbackDocument? = nil) async {
        let runID = UUID()
        let directory = root.appendingPathComponent("DesktopRuns/\(runID.uuidString.lowercased())", isDirectory: true)
        let inbox = InferenceFrameInbox()
        var capture: InferenceCapture?
        var actor: PolicyActorSession?
        var completed: DesktopLearningLoopResult?
        do {
            let selected: CheckpointDocument?
            if let id = options.initialCheckpointID {
                selected = try await store.snapshot().checkpoints.first { $0.id == id }
                guard selected != nil else { throw AstraError("desktop.checkpoint", "The selected starting checkpoint is unavailable.") }
            } else { selected = nil }
            try signals.check()
            let prepared = try await learner.prepareDesktopPolicy(agent: agent, checkpoint: selected,
                model: options.model, actions: options.actions, seed: options.training.seed)
            checkpoint = prepared.checkpoint.document; try signals.check()
            let startIteration: Int
            if let continuation {
                startIteration = continuation.configuration.fields?["completedUpdates"]?.int ?? 0
            } else if options.resume {
                guard let iteration = prepared.manifest.fields?["metrics"]?.fields?["iteration"]?.int,
                      (0..<100_000).contains(iteration), options.iterations > iteration else {
                    throw AstraError("desktop.updateTarget", "Choose a total update target greater than the checkpoint's saved iteration count.")
                }
                startIteration = iteration
            } else { startIteration = 0 }
            phase = "Observing the selected environment…"
            let stream = dependencies.capture(source); capture = stream
            try await stream.start({ inbox.receive($0) }, { inbox.health($0) })
            let frame = try await firstFrame(inbox, signals: signals)
            let scope = try ControlScope(surfaces: [frame.metadata.surface], applicationPID: source.applicationPID, windowID: source.windowID,
                wholeDesktop: source.kind == .desktop, stopOnPhysicalInput: true, geometryRevision: frame.metadata.surface.geometryRevision).validated()
            let reward = try RewardProgramBinding.singleSource(program, surface: frame.metadata.surface, scope: scope)
            var resumed: JSONValue?
            if options.resume, let savedRun = prepared.checkpoint.document.runID {
                resumed = try await LearningFiles.read(root.appendingPathComponent("Jobs/\(savedRun.uuidString.lowercased())/configuration.json"))
            }
            let configuration = try DesktopLearningConfiguration(prepared: prepared, reward: reward, scope: scope,
                options: options, resumedConfiguration: resumed, suspended: continuation)
            let identity = DesktopEvidenceIdentity(runID: runID, clockID: UUID(), environmentID: configuration.environmentID,
                actorSourceID: UUID(), environmentSourceID: UUID())
            let sequence = try DesktopEnvironmentSequence(identity: identity)
            let metadata: JSONValue = .object(["schemaVersion": .integer(1), "runID": .string(runID.uuidString.lowercased()),
                "agentID": .string(agent.id.uuidString.lowercased()), "environment": configuration.environment,
                "rewardBinding": try .encode(reward), "scope": try .encode(scope), "training": configuration.training,
                "model": configuration.model, "contextIDs": try .encode(configuration.contextIDs),
                "initialCheckpointID": .string(prepared.checkpoint.document.id.uuidString.lowercased()),
                "resume": .bool(options.resume), "targetUpdates": .integer(Int64(options.iterations))])
            try await LearningFiles.write(metadata, to: directory.appendingPathComponent("configuration.json"), exclusive: true)
            let collectionsDirectory = directory.appendingPathComponent("Collections", isDirectory: true)
            try await Task.detached { try FileManager.default.createDirectory(at: collectionsDirectory, withIntermediateDirectories: false) }.value
            try signals.check()
            let runtime = dependencies.runtime("actor", { _ in }, { signals.fail($0) })
            let policyActor = PolicyActorSession(runID: runID, ringURL: directory.appendingPathComponent("actor.astraring"),
                slotCapacity: frame.metadata.byteCount, runtime: runtime)
            actor = policyActor
            let previous: JSONValue?
            let mode: PolicyActorPreparation
            if let last = continuation?.fragments.last {
                let original = try last.boundary.decode(PendingReviewBoundary.self)
                previous = try original.result.payload.required("actorProgress")
                mode = .resumeCollection(sourcePath: URL(fileURLWithPath: last.sourceDirectory), manifestSHA256: last.manifestSHA256,
                    expectedProgress: previous!)
            } else {
                previous = options.resume ? try prepared.manifest.required("metrics").required("actorProgress") : nil
                mode = options.resume ? .resume(expectedProgress: previous) : .fresh(seed: UInt64(options.training.seed))
            }
            _ = try await policyActor.prepare(checkpoint: prepared.checkpoint, collection: true, mode: mode)
            try signals.check()
            let observations = try await DesktopResetObservations.prepare(program: configuration.program, scope: scope,
                assetRoot: root.appendingPathComponent("RewardAssets"), frames: {
                    guard let image = try inbox.read() else { throw AstraError("desktop.capture", "The reset has no current frame.") }
                    return [image]
                }, detector: detector)
            let owner = dependencies.controlOwner, controlFactory = controlFactory, verify = verifyScope
            let collectorFactory = collectorFactory, detector = detector, assetRoot = root.appendingPathComponent("RewardAssets")
            var operations = DesktopLearningOperations(collector: { checkpoint, prior, id in
                let destination = collectionsDirectory.appendingPathComponent(id.uuidString.lowercased(), isDirectory: true)
                return try await CollectorSession.start(runID: runID,
                    configuration: configuration.collector(identity: identity, checkpoint: checkpoint, destination: destination,
                        previousActorProgress: prior, continuation: checkpoint.id == continuation?.checkpoint.id ? continuation : nil),
                    journalURL: directory.appendingPathComponent("collector-\(id.uuidString.lowercased()).ndjson"),
                    ringURL: directory.appendingPathComponent("collector-\(id.uuidString.lowercased()).astraring"), slotCapacity: frame.metadata.byteCount,
                    factory: collectorFactory, onFault: { signals.fail($0) })
            }, reset: { [weak self] episode, cancellation in
                guard let self else { throw CancellationError() }
                return try await self.performReset(episode: episode, cancellation: cancellation, source: source, configuration: configuration,
                    scope: scope, observations: observations, actor: policyActor, directory: directory, signals: signals, generation: expected)
            }, warmup: { ready in
                try await Self.warm(policyActor, inbox: inbox, scope: scope, signals: signals)
            }, episode: { [weak self] ready, checkpoint, collector in
                DesktopEpisodeRunner(actor: policyActor, checkpoint: checkpoint, reset: ready, identity: identity,
                    collector: collector, sequence: sequence, program: configuration.program, assetRoot: assetRoot,
                    captureRead: { try inbox.read() }, verifyScope: { try await verify(source, scope) }, controlFactory: controlFactory,
                    controlOwner: owner, recoveryDirectory: directory, deferManualFeedback: configuration.retrospective,
                    detector: detector, onPhase: { [weak self] value in
                        Task { @MainActor [weak self] in
                            guard let self, generation == expected, isBusy,
                                  signals.activeEpisodeID == ready.context.nextEpisodeID else { return }
                            switch value {
                            case .preparing: phase = "Preparing episode observations and reward analysis…"
                            case .arming: phase = "Verifying protected control…"
                            case .running: phase = "Collecting desktop experience…"
                            case .drainingPrediction: phase = "Waiting for the final policy result…"
                            case .releasingControl: phase = "Releasing and joining input control…"
                            case .joiningEvidence: phase = "Saving the episode’s execution and reward evidence…"
                            case .finished: break
                            }
                        }
                    })
            }, learn: { [weak self, learner] batch, resume, validate in
                let result = try await learner.updateExternal(agent: agent, batch: batch, resume: resume, validateBoundary: {
                    try await validate()
                    guard owner.priorCleanupJoined else { throw AstraError("desktop.controlBoundary", "Control ownership changed before learner publication.") }
                })
                await self?.recordCompletedMetrics(generation: expected)
                if result.checkpoint != nil { try await self?.markFeedbackConsumed(batch.rolloutID) }
                return result
            }, stopLearning: { [learner] in await learner.requestStop() }, preserveStopped: { [learner] stopped, validate in
                let boundary = try DesktopCollectionBoundary(identity: identity, checkpoint: stopped.checkpoint.document,
                    destination: URL(fileURLWithPath: try stopped.result.payload.required("path").decode(String.self)),
                    result: stopped.result, joins: stopped.joins, actorState: stopped.pause.binding.state)
                try await boundary.validateHeldPause(actor: policyActor, token: stopped.pause)
                let manifest = try await LearningFiles.read(stopped.checkpoint.directory.appendingPathComponent("manifest.json"))
                let resume = manifest.fields?["metrics"]?.fields?["sourceKind"] == .string("external_rollout")
                let result = try await learner.preserveExternalBoundary(agent: agent, boundary: boundary, resume: resume, validateBoundary: {
                    try await validate(); try await boundary.validateHeldPause(actor: policyActor, token: stopped.pause)
                    guard owner.priorCleanupJoined else { throw AstraError("desktop.controlBoundary", "Stopped control ownership has not joined.") }
                })
                guard let checkpoint = result.checkpoint else { throw AstraError("desktop.stoppedCheckpoint", "The latest joined actor state could not be saved.") }
                return checkpoint
            })
            if configuration.retrospective {
                operations.review = { [weak self] batch in
                    guard let self else { throw CancellationError() }
                    return try await reviewCollected(batch, agent: agent, identity: identity, actor: policyActor,
                        configuration: metadata, signals: signals)
                }
            }
            let loop = try DesktopLearningLoop(actor: policyActor, identity: identity, configuration: configuration,
                checkpoint: prepared.checkpoint, previousActorProgress: previous, completedUpdates: startIteration,
                targetUpdates: options.iterations, resume: continuation?.configuration.fields?["resumeLearner"] == .bool(true) || options.resume,
                operations: operations, carriedDecisions: continuation?.fragments.reduce(0, { total, fragment in
                    total + (fragment.boundary.fields?["result"]?.fields?["payload"]?.fields?["manifest"]?.fields?["decisions"]?.int ?? 0)
                }) ?? 0, onProgress: { [weak self] value in
                    Task { @MainActor [weak self] in
                        guard let self, generation == expected, isBusy else { return }
                        progress = value; if !isStopping { phase = value.phase }
                    }
                })
            signals.bind(loop)
            let result = await loop.run()
            completed = result
            checkpoint = result.checkpoint.document
            if let issue = result.issue { throw issue }
            phase = result.pendingReviewID != nil ? "Experience saved · feedback review can continue later"
                : result.stopped ? "Desktop learning stopped" : "Desktop learning complete"
        } catch {
            if error is CancellationError { phase = "Desktop learning stopped" }
            else { failure = error.localizedDescription; phase = "Desktop learning needs attention" }
        }
        readyWork?.cancel(); await readyWork?.value; readyWork = nil
        await feedbackCancellation?.value; feedbackCancellation = nil
        await feedbackWorkflow?.cancel(); feedbackWorkflow = nil; reviewPresentation = nil
        await actor?.shutdown()
        inbox.close(); await capture?.stop()
        cleanupWarning = dependencies.controlOwner.pendingManualCleanup
        signals.finish()
        do {
            try await LearningFiles.write(.object(["schemaVersion": .integer(1), "runID": .string(runID.uuidString.lowercased()),
                "checkpointID": checkpoint.map { .string($0.id.uuidString.lowercased()) } ?? .null,
                "completedUpdates": .integer(Int64(completed?.completedUpdates ?? 0)), "episodes": .integer(Int64(completed?.completedEpisodes ?? 0)),
                "stopped": .bool(completed?.stopped ?? isStopping), "issue": failure.map(JSONValue.string) ?? .null,
                "pendingReviewID": completed?.pendingReviewID.map { .string($0.uuidString.lowercased()) } ?? .null,
                "controlCleanupConfirmed": .bool(dependencies.controlOwner.priorCleanupJoined)]),
                to: directory.appendingPathComponent("results.json"), exclusive: true)
        } catch { failure = [failure, "The session report could not be saved: \(error.localizedDescription)"].compactMap { $0 }.joined(separator: "\n") }
        completedReportRunID = runID
        resetRunner = nil; resetSource = nil; currentResetID = nil; awaitingReady = nil; countdown = nil
        isBusy = false; isStopping = false; work = nil; self.signals = nil
        await changed()
    }

    private func recordCompletedMetrics(generation expected: UUID) {
        guard generation == expected, isBusy else { return }
        for point in learner.reinforcementMetrics {
            if let index = metrics.firstIndex(where: { $0.iteration == point.iteration }) { metrics[index] = point }
            else { metrics.append(point) }
        }
        metrics.sort { $0.iteration < $1.iteration }
        if metrics.count > 1000 { metrics.removeFirst(metrics.count - 1000) }
    }

    private func markFeedbackConsumed(_ id: UUID) async throws {
        guard var pending = try await store.snapshot().pendingFeedback.first(where: { $0.id == id }) else { return }
        pending.status = .completed; pending.modifiedAt = Date(); try await store.savePendingFeedback(pending)
    }

    private func reviewCollected(_ input: DesktopReviewBatch, agent: AgentDocument, identity: DesktopEvidenceIdentity,
                                 actor: PolicyActorSession, configuration: JSONValue, signals: DesktopHostSignals) async throws -> DesktopLearningBatch? {
        let collected = input.collection
        let proof = PendingReviewBoundary(identity: identity, joins: collected.joins, actor: collected.pause.binding.state, result: collected.result)
        let boundary = try proof.validated(checkpoint: collected.checkpoint.document)
        try await boundary.validateHeldPause(actor: actor, token: collected.pause)
        let batchID = try boundary.manifest.requiredUUID("behaviorBatchID")
        var saved = configuration.fields ?? [:]
        saved["resumeLearner"] = .bool(input.resumeLearner)
        saved["completedUpdates"] = .integer(Int64(input.completedUpdates))
        let fragment = PendingFeedbackFragment(collectionID: boundary.collectionID, sourceDirectory: boundary.path.path,
            manifestSHA256: try collected.result.payload.required("manifestSHA256").decode(String.self),
            revisionDirectory: root.appendingPathComponent("FeedbackRevisions/\(boundary.collectionID.uuidString.lowercased())").path,
            boundary: try .encode(proof))
        var document: PendingFeedbackDocument
        if let existing = try await store.snapshot().pendingFeedback.first(where: { $0.id == batchID }) {
            document = existing; document.fragments.append(fragment); document.status = .awaitingReview; document.modifiedAt = Date()
        } else {
            document = .init(behaviorBatchID: batchID, agentID: agent.id, checkpoint: collected.checkpoint.document,
                configuration: .object(saved), fragments: [fragment])
        }
        try await store.savePendingFeedback(document); await changed()
        if isStopping { return nil }
        let owner = dependencies.controlOwner
        let lease = try DesktopControlLock()
        defer { withExtendedLifetime(lease) {} }
        let pause = collected.pause
        let workflow = PendingFeedbackWorkflow(store: store, learner: learner, root: root, validate: {
            try await boundary.validateHeldPause(actor: actor, token: pause)
            guard owner.priorCleanupJoined else { throw AstraError("feedback.controls", "Feedback review requires joined input control.") }
        }, present: { [weak self] review in
            if review != nil { self?.dependencies.showOperator() }
            self?.reviewPresentation = review
        }, changed: changed)
        feedbackWorkflow = workflow
        defer { feedbackWorkflow = nil; reviewPresentation = nil }
        let result = try await workflow.run(document)
        if let result, !isStopping {
            return try DesktopLearningBatch(reviewed: result, boundary: boundary)
        }
        return nil
    }

    /// Reviewing suspended experience does not start capture or an actor. A
    /// complete batch trains against its exact original behavior checkpoint.
    func resumeFeedback(_ document: PendingFeedbackDocument, agent: AgentDocument) throws {
        guard !isBusy, !learner.isBusy, dependencies.controlOwner.priorCleanupJoined,
              document.agentID == agent.id, ![.completed, .discarded].contains(document.status) else {
            throw AstraError("feedback.busy", "Finish the current workflow before reopening saved feedback.")
        }
        generation = UUID(); isBusy = true; isStopping = false; agentID = agent.id; failure = nil
        phase = "Opening saved experience…"; checkpoint = document.checkpoint; feedbackCancellation = nil
        let signals = DesktopHostSignals(); self.signals = signals
        work = Task { [self] in
            do {
                let lease = try DesktopControlLock(); defer { withExtendedLifetime(lease) {} }
                let owner = dependencies.controlOwner
                let validate: @Sendable () async throws -> Void = {
                    guard owner.priorCleanupJoined else { throw AstraError("feedback.controls", "Input control must remain released during review and learning.") }
                }
                try signals.check()
                let workflow = PendingFeedbackWorkflow(store: store, learner: learner, root: root, validate: validate,
                    present: { [weak self] review in
                        if review != nil { self?.dependencies.showOperator() }
                        self?.reviewPresentation = review
                    }, changed: changed)
                feedbackWorkflow = workflow
                if let reviewed = try await workflow.run(document), !isStopping {
                    guard let last = document.fragments.last else { throw AstraError("feedback.source", "The review has no original source.") }
                    let original = try last.boundary.decode(PendingReviewBoundary.self)
                    let boundary = try original.validated(checkpoint: document.checkpoint)
                    let batch = try DesktopLearningBatch(reviewed: reviewed, boundary: boundary)
                    phase = "Learning from reviewed experience…"
                    let result = try await learner.updateExternal(agent: agent, batch: batch,
                        resume: document.configuration.fields?["resumeLearner"] == .bool(true), validateBoundary: validate)
                    checkpoint = result.checkpoint ?? document.checkpoint
                    if result.checkpoint != nil { try await markFeedbackConsumed(document.id) }
                    phase = result.cancelled ? "Learning stopped · checkpoint saved" : "Reviewed experience learned · checkpoint saved"
                } else { phase = "Review saved · continue when ready" }
            } catch is CancellationError { phase = "Review saved · learning stopped" }
            catch { failure = error.localizedDescription; phase = "Feedback review needs attention" }
            await feedbackCancellation?.value; feedbackCancellation = nil
            await feedbackWorkflow?.cancel(); feedbackWorkflow = nil; reviewPresentation = nil
            signals.finish(); self.signals = nil; isBusy = false; isStopping = false; work = nil
            await changed()
        }
    }

    func continueFeedback(_ document: PendingFeedbackDocument, agent: AgentDocument, source: CaptureSource) throws {
        guard document.agentID == agent.id, ![.completed, .discarded].contains(document.status),
              let minimum = document.configuration.fields?["training"]?.fields?["rollout_decisions"]?.int,
              document.collectedDecisions < minimum else {
            throw AstraError("feedback.collectionComplete", "This batch already has enough experience. Review it to continue learning.")
        }
        let binding = try document.configuration.required("rewardBinding").decode(RewardProgramBinding.self)
        var options = DesktopLearningOptions()
        options.initialCheckpointID = document.checkpoint.id
        options.iterations = document.configuration.fields?["targetUpdates"]?.int ?? 20
        try start(agent: agent, source: source, program: binding.definition, options: options, continuation: document)
    }

    private func performReset(episode: UUID, cancellation: ResetCancellation, source: CaptureSource,
                              configuration: DesktopLearningConfiguration, scope: ControlScope, observations: DesktopResetObservations,
                              actor: PolicyActorSession, directory: URL, signals: DesktopHostSignals, generation expected: UUID) async throws -> ResetResult {
        let owner = dependencies.controlOwner
        let driver = NativeResetDriver(environmentID: configuration.environmentID, source: source,
            observations: { try await observations.observe(context: $0) }, priorOwnersJoined: {
                let available = await actor.binding?.canReset == true
                return owner.priorCleanupJoined && available
            }, owner: owner, runtimeFactory: controlFactory, recoveryDirectory: directory, scopeVerifier: verifyScope)
        let runner = ResetRunner(driver: driver, progress: { [weak self] value in
            Task { @MainActor [weak self] in
                guard let self, generation == expected, isBusy, currentResetID == value.resetID else { return }
                if value.phase == .awaitingManualReady { dependencies.showOperator(); awaitingReady = value.resetID }
                else if awaitingReady == value.resetID { awaitingReady = nil }
                if !isStopping { phase = value.message }
            }
        })
        resetRunner = runner; resetSource = source; currentResetID = cancellation.resetID
        defer {
            if currentResetID == cancellation.resetID { resetRunner = nil; resetSource = nil; currentResetID = nil; awaitingReady = nil }
        }
        if configuration.program.resetPlan != nil {
            do { try await activateWithCountdown(source, signals: signals) }
            catch is CancellationError { cancellation.cancel() }
        }
        let context = try ResetContext(resetID: cancellation.resetID, nextEpisodeID: episode, environmentID: configuration.environmentID, scope: scope)
        return try await runner.run(context: context, program: configuration.program, cancellation: cancellation)
    }

    private func activateWithCountdown(_ source: CaptureSource, signals: DesktopHostSignals) async throws {
        if dependencies.countdownSeconds > 0 {
            for remaining in (1...dependencies.countdownSeconds).reversed() {
                try signals.check(); try Task.checkCancellation()
                countdown = remaining
                if remaining == min(2, dependencies.countdownSeconds) { try dependencies.activate(source) }
                try await Task.sleep(for: .seconds(1))
            }
        } else { try dependencies.activate(source) }
        countdown = nil; try signals.check()
    }
    private func firstFrame(_ inbox: InferenceFrameInbox, signals: DesktopHostSignals) async throws -> InferenceImage {
        let deadline = ContinuousClock.now + .seconds(5)
        while ContinuousClock.now < deadline {
            try signals.check()
            if let image = try inbox.read() { return image }
            try await Task.sleep(for: .milliseconds(20))
        }
        throw AstraError("desktop.captureTimeout", "The selected environment did not produce a screen observation.")
    }
    private nonisolated static func warm(_ actor: PolicyActorSession, inbox: InferenceFrameInbox, scope: ControlScope,
                                        signals: DesktopHostSignals) async throws {
        guard let binding = await actor.binding, binding.isAvailable else { throw AstraError("desktop.warmup", "The actor is not ready for isolated warmup.") }
        for index in 0..<3 {
            try signals.check()
            guard let image = try inbox.read(), scope.surfaces.contains(image.metadata.surface) else { throw AstraError("desktop.capture", "The environment changed before policy warmup.") }
            let cutoff = MonotonicClock.now, bounds = image.metadata.surface.globalBounds
            var warmState = ControlState(); warmState.valid = true; warmState.observedNanos = cutoff
            warmState.pointer = .init(x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2)
            let observation = ControlObservation(controlState: warmState, executedEvents: [], intervalCovered: true,
                cutoffNanos: cutoff, lastSequence: nil)
            let started = MonotonicClock.now
            _ = try await actor.warmup(.init(frame: image, controls: observation))
            let elapsed = Double(MonotonicClock.now - started) / 1_000_000
            let budget = Double(min(binding.policy.periodMS, binding.policy.leadMS))
            if index > 0, elapsed > budget - max(5, budget * 0.1) {
                throw AstraError("desktop.warmupTiming", "The policy needs \(Int(elapsed.rounded(.up))) ms per decision, leaving insufficient headroom for its trained timing.")
            }
        }
    }
}

extension PendingFeedbackDocument {
    var collectedDecisions: Int {
        fragments.reduce(0) { $0 + ($1.boundary.fields?["result"]?.fields?["payload"]?.fields?["manifest"]?.fields?["decisions"]?.int ?? 0) }
    }
}

private final class DesktopHostSignals: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    private var finished = false
    private var error: AstraError?
    private var loop: DesktopLearningLoop?
    var hasLoop: Bool { lock.withLock { loop != nil } }
    var activeEpisodeID: UUID? { lock.withLock { loop }?.activeEpisodeID }
    func bind(_ loop: DesktopLearningLoop) {
        let prior = lock.withLock { self.loop = loop; return (stopped, error) }
        if let error = prior.1 { loop.fail(error) } else if prior.0 { loop.requestStop() }
    }
    func stop() { let loop = lock.withLock { stopped = true; return loop }; loop?.requestStop() }
    func fail(_ error: AstraError) { let loop = lock.withLock { () -> DesktopLearningLoop? in
        guard !finished else { return nil }; self.error = self.error ?? error; return self.loop
    }; loop?.fail(error) }
    func finish() { lock.withLock { finished = true; loop = nil } }
    func check() throws { try lock.withLock { if let error { throw error }; if stopped { throw CancellationError() } } }
}
