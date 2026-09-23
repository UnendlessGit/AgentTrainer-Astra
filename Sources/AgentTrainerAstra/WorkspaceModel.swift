import AppKit
import AstraCore
import Observation
import AstraPlatform

enum WorkspaceDestination: Hashable { case agent(UUID), library, activity }
struct RecordingLinkRequest: Identifiable {
    let id = UUID()
    var agentID: UUID?
    var recordingID: UUID?
}
enum AgentSection: String, CaseIterable, Identifiable {
    case demonstrations = "Demonstrations", training = "Training", evaluation = "Evaluation", run = "Run"
    var id: String { rawValue }
}

@MainActor @Observable final class WorkspaceModel {
    private(set) var agents: [AgentDocument] = []
    private(set) var environments: [EnvironmentDocument] = []
    private(set) var issues: [LibraryIssue] = []
    private(set) var recordings: [RecordingManifest] = []
    private(set) var learningRuns: [LearningRunDocument] = []
    private(set) var checkpoints: [CheckpointDocument] = []
    private(set) var evaluations: [EvaluationDocument] = []
    private(set) var rewardPrograms: [RewardProgram] = []
    private(set) var learning: LearningCoordinator?
    private(set) var inference: InferenceCoordinator?
    private(set) var desktopLearning: DesktopLearningHost?
    private(set) var pendingFeedback: [PendingFeedbackDocument] = []
    private(set) var refreshingSources = false
    private(set) var sourceIssue: String?
    private(set) var checkpointContextSizes: [Int] = []
    private var checkpointContextGeneration = UUID()
    private(set) var isClosing = false
    private(set) var sources: [CaptureSource] = []
    private(set) var permissions = PermissionSnapshot.current()
    private(set) var recordingProgress: RecordingProgress?
    private(set) var recordingStarting = false
    private(set) var recordingCountdown: Int?
    private(set) var recordingStopping = false
    var showingRecorder = false
    var recordingToInspect: RecordingManifest?
    var recordingInspectionAgentID: UUID?
    var recordingLinkRequest: RecordingLinkRequest?
    private(set) var loading = true
    private(set) var saving = false
    var destination: WorkspaceDestination? = .library
    var section: AgentSection = .demonstrations
    var errorMessage: String?
    var showingNewAgent = false
    private var store: LibraryStore?
    private let historyRoot: URL?
    private let controlOwner: NativeControlOwner
    private let desktopLeaseURL: URL
    private(set) var controlHistoryBusy = false
    private var controlStartupWork: Task<Void, Never>?
    private var controlStartupGeneration: UUID?
    private let controlPreflightInspection: @Sendable (LibraryStore) async throws -> Void
    private var libraryLease: LibraryLease?
    private var started = false
    private var acknowledgedUnconfirmedControlRun: UUID?
    private var recorder: RecordingSession?
    private var recordingAgentID: UUID?
    private var activeRecordingID: UUID?
    private(set) var recordingLinks: [UUID: Set<UUID>] = [:]
    private(set) var recordingSelections: [UUID: [UUID: RecordingTrainingSelection]] = [:]
    private(set) var checkpointLinks: [UUID: Set<UUID>] = [:]

    init(inferenceCoordinator: InferenceCoordinator? = nil, historyStore: LibraryStore? = nil, historyRoot: URL? = nil,
         controlOwner: NativeControlOwner = .shared, desktopLeaseURL: URL = DesktopControlLock.standardURL,
         controlPreflightInspection: @escaping @Sendable (LibraryStore) async throws -> Void = WorkspaceModel.inspectControlHistory) {
        self.inference = inferenceCoordinator; store = historyStore; self.historyRoot = historyRoot
        self.controlOwner = controlOwner; self.desktopLeaseURL = desktopLeaseURL
        self.controlPreflightInspection = controlPreflightInspection
        if historyStore != nil { started = true; loading = false }
    }

    nonisolated static func inspectControlHistory(_ store: LibraryStore) async throws {
        try await store.inspectPriorInferenceRuns()
    }

    var selectedAgent: AgentDocument? {
        guard case .agent(let id) = destination else { return nil }
        return agents.first { $0.id == id }
    }

    var supportRoot: URL {
        if let historyRoot { return historyRoot }
        if let path = ProcessInfo.processInfo.environment["ASTRA_WORKSPACE_ROOT"], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AgentTrainer Astra", isDirectory: true)
    }

    func start() async {
        guard !started else { return }
        started = true
        do {
            let root = supportRoot
            let opened = try await Task.detached {
                let lease = try LibraryLease(root: root)
                return (lease, try LibraryStore(root: root))
            }.value
            libraryLease = opened.0; store = opened.1
            learning = LearningCoordinator(store: opened.1, root: root) { [weak self] in
                do { try await self?.refresh() }
                catch { self?.errorMessage = error.localizedDescription }
            }
            inference = InferenceCoordinator(store: opened.1, root: root)
            if let learning {
                desktopLearning = DesktopLearningHost(store: opened.1, root: root, learner: learning) { [weak self] in
                    do { try await self?.refresh() }
                    catch { self?.errorMessage = error.localizedDescription }
                }
            }
            _ = try await store?.recoverInterruptedRecordings()
            try await store?.markAbandonedLearningRunsInterrupted()
            try await store?.inspectPriorInferenceRuns()
            try await refresh()
            if let first = agents.first { destination = .agent(first.id) }
        } catch { errorMessage = error.localizedDescription }
        loading = false
    }

    func createAgent(name: String) async {
        guard let store else { return }
        saving = true
        defer { saving = false }
        do {
            let agent = try AgentDocument(name: name).validated()
            try await store.save(agent)
            try await refresh()
            destination = .agent(agent.id); section = .demonstrations
            showingNewAgent = false
        } catch { errorMessage = error.localizedDescription }
    }

    func saveAgent(_ document: AgentDocument) async {
        guard let store else { return }
        saving = true
        defer { saving = false }
        do {
            var value = document; value.modifiedAt = Date()
            try await store.save(value)
            try await refresh()
        } catch { errorMessage = error.localizedDescription }
    }

    func saveRewardProgram(_ document: RewardProgram, for agentID: UUID) async throws {
        guard let store, !isClosing else { throw AstraError("reward.workspace", "The workspace is not ready to save this definition.") }
        try await store.saveRewardProgram(document, for: agentID)
        try await refresh()
    }

    func refreshPermissionsAndSources() async {
        guard !refreshingSources else { return }
        refreshingSources = true
        defer { refreshingSources = false }
        sourceIssue = nil
        permissions = PermissionSnapshot.current()
        guard permissions.screenRecording else {
            sources = []; sourceIssue = "Allow Screen Recording for Astra in System Settings, then refresh environments."
            return
        }
        do { sources = try await CaptureDiscovery.sources() }
        catch { sources = []; sourceIssue = error.localizedDescription }
    }

    func startRecording(source: CaptureSource, name: String, fps: Int) async {
        guard let store, !isClosing, !isRunningAgent, !recordingStarting, !recordingStopping, recorder == nil else { return }
        permissions = PermissionSnapshot.current()
        guard permissions.screenRecording && permissions.inputMonitoring else {
            errorMessage = "Screen Recording and Input Monitoring permissions are required to record a demonstration."
            return
        }
        recordingStarting = true
        defer { recordingStarting = false; recordingCountdown = nil }
        let agentID = selectedAgent?.id
        let environment = EnvironmentDocument(name: String(source.name.prefix(160)), kind: source.kind, displayID: source.displayID,
                                              windowID: source.windowID, applicationBundleID: source.applicationBundleID, captureFPS: fps)
        let manifest = RecordingManifest(name: name, environment: environment, recordedForAgentID: agentID)
        activeRecordingID = manifest.id
        do {
            let directory = store.recordingDirectory(id: manifest.id)
            let progressHandler: @Sendable (RecordingProgress) -> Void = { [weak self] progress in
                Task { @MainActor [weak self] in
                    guard self?.activeRecordingID == progress.id else { return }
                    self?.recordingProgress = progress
                }
            }
            let faultHandler: @Sendable (String) -> Void = { [weak self] message in
                Task { @MainActor [weak self] in
                    guard self?.activeRecordingID == manifest.id else { return }
                    await self?.stopRecording(issue: message)
                }
            }
            let created = try await Task.detached {
                try RecordingSession(directory: directory, manifest: manifest, source: source,
                                     onProgress: progressHandler, onFault: faultHandler)
            }.value
            guard activeRecordingID == manifest.id else {
                let cancelled = try await created.stop(issue: "Cancelled before capture started.")
                try await store.saveRecording(cancelled, linkTo: agentID)
                try await refresh()
                return
            }
            recorder = created; recordingAgentID = agentID
            try await store.save(environment)
            if let agentID, var agent = agents.first(where: { $0.id == agentID }) {
                agent.environmentID = environment.id; agent.modifiedAt = Date()
                try await store.save(agent)
            }
            try await store.saveRecording(manifest, linkTo: agentID)
            try await refresh()
            showingRecorder = false
            // Dismiss the operator sheet before enabling input observation.
            // The visible countdown lets a window settle and provides a stop
            // path while the selected environment is brought to the front.
            for remaining in (1...3).reversed() {
                guard activeRecordingID == manifest.id, !recordingStopping else { return }
                recordingCountdown = remaining
                if remaining == 2, let pid = source.applicationPID {
                    guard let application = NSRunningApplication(processIdentifier: pid),
                          !application.isTerminated,
                          source.applicationLaunchDate == nil || source.applicationLaunchDate == application.launchDate else {
                        throw AstraError("recording.targetChanged", "The selected application closed or restarted. Choose it again.")
                    }
                    application.activate()
                }
                try await Task.sleep(for: .seconds(1))
            }
            recordingCountdown = nil
            guard activeRecordingID == manifest.id, !recordingStopping else { return }
            try await created.start()
        } catch {
            errorMessage = error.localizedDescription
            if recorder != nil { await stopRecording(issue: error.localizedDescription) }
        }
    }

    func stopRecording(issue: String? = nil) async {
        guard let store, !recordingStopping else { return }
        guard let recorder else { activeRecordingID = nil; return }
        recordingStopping = true
        defer { recordingStopping = false }
        do {
            let manifest = try await recorder.stop(issue: issue)
            try await store.saveRecording(manifest, linkTo: recordingAgentID)
            self.recorder = nil; recordingProgress = nil; recordingAgentID = nil; activeRecordingID = nil
            try await refresh()
            if let issue { errorMessage = issue }
        } catch {
            errorMessage = "Recording data was preserved, but finalization needs recovery: \(error.localizedDescription)"
            self.recorder = nil; recordingProgress = nil; recordingAgentID = nil; activeRecordingID = nil
            try? await refresh()
        }
    }

    var isRecording: Bool { recorder != nil }

    var isLearning: Bool { learning?.isBusy == true || desktopLearning?.isBusy == true }
    var isRunningAgent: Bool { inference?.isBusy == true || desktopLearning?.isBusy == true || controlHistoryBusy }
    var pendingControlHistory: [ControlHistoryReview] { issues.compactMap(\.controlHistory) }
    var controlHistoryBlockReason: String? { issues.first(where: \.blocksLiveControl)?.message }

    var inferenceUnavailableReason: String? {
        if isClosing { return "The workspace is closing." }
        if controlHistoryBusy { return "Checking previous control cleanup…" }
        if let controlHistoryBlockReason { return controlHistoryBlockReason }
        if isRecording || recordingStarting || recordingStopping { return "Finish recording before running an agent." }
        if isLearning { return "Finish learning before starting live control." }
        return nil
    }

    func startInference(agent: AgentDocument, checkpoint: CheckpointDocument, source: CaptureSource, options: InferenceOptions) {
        do {
            guard inferenceUnavailableReason == nil else { throw AstraError("inference.busy", inferenceUnavailableReason!) }
            guard let inference else { throw AstraError("inference.workspace", "The workspace is still opening.") }
            try startAfterHistoryReview {
                try inference.start(agent: agent, checkpoint: checkpoint, source: source, options: options)
            }
        } catch { errorMessage = error.localizedDescription }
    }

    func inspectCheckpointContexts(_ id: UUID?, agentID: UUID) async {
        let generation = UUID(); checkpointContextGeneration = generation; checkpointContextSizes = []
        guard let id, checkpointLinks[agentID]?.contains(id) == true else { return }
        do {
            let metadata = try await LearningFiles.read(supportRoot.appendingPathComponent("Models").appendingPathComponent(id.uuidString.lowercased()).appendingPathComponent("manifest.json"))
            guard case .array(let sizes) = metadata.fields?["model"]?.fields?["context_sizes"], sizes.count <= 32,
                  sizes.allSatisfy({ $0.int.map { (1...65_536).contains($0) } == true }) else {
                throw AstraError("inference.contexts", "The selected checkpoint's context configuration is invalid.")
            }
            if checkpointContextGeneration == generation { checkpointContextSizes = sizes.compactMap(\.int) }
        } catch {
            if checkpointContextGeneration == generation { errorMessage = error.localizedDescription }
        }
    }

    func startBehaviorTraining(agent: AgentDocument, options: BehaviorOptions) {
        do {
            guard !isClosing, !isRunningAgent, !saving else { throw AstraError("learning.closing", "Finish saving library changes and stop live control before starting another learning operation.") }
            guard let learning else { throw AstraError("learning.workspace", "The workspace is still opening.") }
            try learning.start(agent: agent, options: options,
                               recordings: recordings.filter { recordingLinks[agent.id]?.contains($0.id) == true },
                               selections: recordingSelections[agent.id] ?? [:])
        } catch { errorMessage = error.localizedDescription }
    }

    func evaluateCheckpoint(_ checkpoint: CheckpointDocument, agentID: UUID, split: String) {
        do {
            guard !isClosing, !isRunningAgent else { throw AstraError("learning.closing", "Stop live control before starting evaluation.") }
            guard let learning else { throw AstraError("learning.workspace", "The workspace is still opening.") }
            try learning.evaluate(checkpoint: checkpoint, agentID: agentID, split: split)
        }
        catch { errorMessage = error.localizedDescription }
    }

    func evaluateCheckpoints(_ checkpoints: [CheckpointDocument], agentID: UUID, datasetCheckpoint: CheckpointDocument, split: String) {
        do {
            guard !isClosing, !isRunningAgent else { throw AstraError("learning.closing", "Stop live control before starting evaluation.") }
            guard let learning else { throw AstraError("learning.workspace", "The workspace is still opening.") }
            try learning.evaluate(checkpoints: checkpoints, agentID: agentID, datasetCheckpoint: datasetCheckpoint, split: split)
        } catch { errorMessage = error.localizedDescription }
    }

    func startReinforcementTraining(agent: AgentDocument, options: ReinforcementOptions) throws {
        guard !isClosing, !isRunningAgent else { throw AstraError("learning.closing", "Stop live control before starting learning.") }
        guard let learning else { throw AstraError("learning.workspace", "The workspace is still opening.") }
        try learning.startReinforcement(agent: agent, options: options)
    }

    func startDesktopTraining(agent: AgentDocument, source: CaptureSource, program: RewardProgram, options: DesktopLearningOptions) throws {
        guard !isClosing, !isRunningAgent, !isLearning, !isRecording, !recordingStarting, !recordingStopping, !saving else {
            throw AstraError("desktop.busy", "Finish recording, learning and live control before starting desktop training.")
        }
        guard let desktopLearning else { throw AstraError("desktop.workspace", "The workspace is still opening.") }
        if let controlHistoryBlockReason { throw AstraError("history.reviewRequired", controlHistoryBlockReason) }
        try startAfterHistoryReview {
            try desktopLearning.start(agent: agent, source: source, program: program, options: options)
        }
    }

    func reopenFeedback(_ document: PendingFeedbackDocument, source: CaptureSource? = nil) throws {
        guard !isClosing, !isRunningAgent, !isLearning, !isRecording, !saving,
              let desktopLearning, let agent = agents.first(where: { $0.id == document.agentID }),
              let saved = pendingFeedback.first(where: { $0.id == document.id }), saved == document else {
            throw AstraError("feedback.workspace", "Finish the current workflow and choose the latest saved feedback item.")
        }
        try startAfterHistoryReview {
            if let source { try desktopLearning.continueFeedback(saved, agent: agent, source: source) }
            else { try desktopLearning.resumeFeedback(saved, agent: agent) }
        }
    }

    func selectCheckpoint(_ id: UUID?, agentID: UUID) async {
        guard let store, var agent = agents.first(where: { $0.id == agentID }) else { return }
        if let id, checkpointLinks[agentID]?.contains(id) != true || !checkpoints.contains(where: { $0.id == id }) {
            errorMessage = "This checkpoint is not linked to the selected agent."; return
        }
        do {
            agent.selectedCheckpointID = id; agent.modifiedAt = Date()
            try await store.save(agent); try await refresh()
            await inspectCheckpointContexts(id, agentID: agentID)
        } catch { errorMessage = error.localizedDescription }
    }

    func stopLearningAndWait() async {
        if desktopLearning?.isBusy == true { await desktopLearning?.stopAndWait() }
        else { await learning?.stopAndWait() }
    }

    @discardableResult
    func prepareForTermination() async -> Bool {
        isClosing = true
        let starting = controlStartupWork; starting?.cancel()
        // Both owners receive stop promptly. The application exits only after
        // capture is sealed and learning has published its terminal checkpoint.
        async let recording: Void = stopRecording()
        async let learning: Void = stopLearningAndWait()
        async let inferenceStop: Void = inference?.stopAndWait() ?? ()
        _ = await (recording, learning, inferenceStop)
        await starting?.value
        while recordingStarting || recordingStopping || saving || controlHistoryBusy {
            try? await Task.sleep(for: .milliseconds(50))
        }
        if let inference, let run = inference.runID, inference.requiresManualControlCleanupAcknowledgement,
           acknowledgedUnconfirmedControlRun != run {
            acknowledgedUnconfirmedControlRun = run
            isClosing = false
            errorMessage = "The control helper exited before input cleanup could be confirmed. Release any held controls manually. Quit again when you are ready to close Astra."
            return false
        }
        if let warning = desktopLearning?.cleanupWarning, acknowledgedUnconfirmedControlRun != warning.sessionID {
            acknowledgedUnconfirmedControlRun = warning.sessionID
            isClosing = false
            errorMessage = "Desktop control cleanup could not be confirmed. Release any held controls manually. Quit again when you are ready to close Astra."
            return false
        }
        return true
    }

    func duplicateSelectedAgent() async {
        guard let store, let selectedAgent else { return }
        do {
            let copy = try await store.duplicateAgent(selectedAgent)
            try await refresh(); destination = .agent(copy.id)
        } catch { errorMessage = error.localizedDescription }
    }

    func inspectRecording(_ recording: RecordingManifest, for agentID: UUID? = nil) {
        recordingInspectionAgentID = agentID; recordingToInspect = recording
    }

    func linkRecordings(_ identifiers: Set<UUID>, to agentID: UUID) async -> Bool {
        guard let store, !isClosing, !saving else { return false }
        saving = true; defer { saving = false }
        do { try await store.linkRecordings(identifiers, to: agentID); try await refresh(); return true }
        catch { errorMessage = error.localizedDescription; return false }
    }

    func unlinkRecording(_ recordingID: UUID, from agentID: UUID) async {
        guard let store, !isClosing, !saving else { return }
        saving = true; defer { saving = false }
        do { try await store.unlinkRecording(recordingID, from: agentID); try await refresh() }
        catch { errorMessage = error.localizedDescription }
    }

    func saveRecordingSelection(_ selection: RecordingTrainingSelection, recordingID: UUID, agentID: UUID) async -> Bool {
        guard let store, !isClosing, !saving else { return false }
        saving = true; defer { saving = false }
        do { try await store.saveRecordingSelection(selection, recordingID: recordingID, agentID: agentID); try await refresh(); return true }
        catch { errorMessage = error.localizedDescription; return false }
    }

    func trainingSelectionSummary(recording: RecordingManifest, agentID: UUID) -> String {
        guard let selection = recordingSelections[agentID]?[recording.id], let ranges = try? selection.resolved(for: recording) else { return "Review selection" }
        let seconds = ranges.reduce(0) { $0 + $1.durationSeconds }.formatted(.number.precision(.fractionLength(1)))
        return selection.ranges == nil ? "All usable · \(seconds) s" : "\(ranges.count) \(ranges.count == 1 ? "interval" : "intervals") · \(seconds) s"
    }

    func refreshControlHistory() async {
        guard !controlHistoryBusy else { return }
        do {
            try requireIdleForHistory(allowClosing: false)
            guard let store else { throw AstraError("history.workspace", "The workspace is still opening.") }
            controlHistoryBusy = true
            defer { controlHistoryBusy = false }
            try await store.inspectPriorInferenceRuns(); try await refresh()
        } catch { errorMessage = error.localizedDescription }
    }

    func historyReviewUnavailableReason(_ review: ControlHistoryReview) -> String? {
        if controlHistoryBusy { return "Another cleanup review is still finishing." }
        do { try requireIdleForHistory(allowClosing: false); _ = try cleanupRoute(review); return nil }
        catch { return error.localizedDescription }
    }

    func acknowledgeControlHistory(_ review: ControlHistoryReview) async throws {
        try await acknowledgeHistory(location: review.location, runID: review.runID, expected: review)
    }
    func acknowledgeInferenceCleanup() async throws {
        guard let inference, inference.requiresManualControlCleanupAcknowledgement, let run = inference.runID else {
            throw AstraError("history.currentRun", "There is no completed agent cleanup warning to acknowledge.")
        }
        try await acknowledgeHistory(location: .inference, runID: run, expected: nil)
    }
    func acknowledgeDesktopCleanup() async throws {
        guard let desktopLearning, !desktopLearning.isBusy, desktopLearning.cleanupWarning != nil,
              let run = desktopLearning.completedReportRunID else {
            throw AstraError("history.currentRun", "There is no completed desktop cleanup warning to acknowledge.")
        }
        try await acknowledgeHistory(location: .desktop, runID: run, expected: nil)
    }

    private enum CleanupRoute { case history, inference(UUID), desktop(UUID) }
    private func cleanupRoute(_ review: ControlHistoryReview) throws -> CleanupRoute {
        if let warning = controlOwner.pendingManualCleanup {
            if review.location == .inference, warning.runID == review.runID,
               inference?.runID == review.runID, inference?.requiresManualControlCleanupAcknowledgement == true {
                return .inference(warning.sessionID)
            }
            if review.location == .desktop, desktopLearning?.completedReportRunID == review.runID,
               desktopLearning?.cleanupWarning?.sessionID == warning.sessionID {
                return .desktop(warning.sessionID)
            }
            throw AstraError("history.currentOwner", "Resolve the current run’s cleanup warning before acknowledging other history.")
        }
        guard controlOwner.priorCleanupJoined else {
            throw AstraError("history.activeOwner", "A current control owner is still running or joining cleanup.")
        }
        return .history
    }
    private func requireIdleForHistory(allowClosing: Bool) throws {
        guard allowClosing || !isClosing, inference?.isBusy != true, desktopLearning?.isBusy != true,
              learning?.isBusy != true, !isRecording, !recordingStarting, !recordingStopping, !saving else {
            throw AstraError("history.activeWorkflow", "Finish current recording, learning and control work before reviewing previous cleanup.")
        }
    }
    private func acknowledgeHistory(location: ControlHistoryLocation, runID: UUID, expected: ControlHistoryReview?) async throws {
        guard !controlHistoryBusy, let store else { throw AstraError("history.busy", "Wait for the workspace or current cleanup review to finish.") }
        try requireIdleForHistory(allowClosing: false)
        controlHistoryBusy = true
        defer { controlHistoryBusy = false }
        // A surviving orphaned helper/guardian holds this same flock even when
        // its old app has gone. History never authorizes stopping that owner.
        let lease = try DesktopControlLock(url: desktopLeaseURL)
        defer { withExtendedLifetime(lease) {} }
        let review: ControlHistoryReview
        if let expected { review = expected }
        else {
            guard let current = try await store.controlHistoryReview(location: location, runID: runID) else {
                throw AstraError("history.changed", "This report no longer contains the cleanup warning that was shown.")
            }
            review = current
        }
        let route = try cleanupRoute(review)
        do { try await store.acknowledgeControlHistory(review) }
        catch {
            try? await store.inspectPriorInferenceRuns(); try? await refresh()
            throw error
        }
        // The global lease remains held through both the durable operator
        // record and the matching in-memory acknowledgement.
        try requireIdleForHistory(allowClosing: true)
        switch route {
        case .history:
            guard controlOwner.priorCleanupJoined else { throw AstraError("history.activeOwner", "Control ownership changed while this history was reviewed.") }
        case .inference(let session):
            guard controlOwner.pendingManualCleanup?.sessionID == session else { throw AstraError("history.changedOwner", "The current cleanup warning changed.") }
            try inference?.acknowledgeManualControlCleanup()
        case .desktop(let session):
            guard controlOwner.pendingManualCleanup?.sessionID == session else { throw AstraError("history.changedOwner", "The current cleanup warning changed.") }
            try desktopLearning?.acknowledgeManualCleanup()
        }
        try await refresh()
    }

    /// Recheck disk history immediately before each new live workflow. This
    /// keeps a replaced/new report from inheriting a previously cached consent.
    private func startAfterHistoryReview(_ start: @escaping @MainActor () throws -> Void) throws {
        guard !controlHistoryBusy, let store else { throw AstraError("history.busy", "Wait for the workspace or current cleanup review to finish.") }
        try requireIdleForHistory(allowClosing: false)
        controlHistoryBusy = true
        let generation = UUID(); controlStartupGeneration = generation
        controlStartupWork = Task {
            defer {
                if controlStartupGeneration == generation { controlHistoryBusy = false; controlStartupWork = nil; controlStartupGeneration = nil }
            }
            do {
                try await controlPreflightInspection(store); try await refresh()
                try Task.checkCancellation(); try requireIdleForHistory(allowClosing: false)
                guard controlStartupGeneration == generation else { throw CancellationError() }
                guard controlHistoryBlockReason == nil else {
                    throw AstraError("history.reviewRequired", "Review the previous control-cleanup warning before starting live control.")
                }
                try start()
            } catch is CancellationError { }
            catch { errorMessage = error.localizedDescription }
        }
    }

    private func refresh() async throws {
        guard let store else { return }
        let snapshot = try await store.snapshot()
        agents = snapshot.agents; environments = snapshot.environments; recordings = snapshot.recordings; issues = snapshot.issues
        learningRuns = snapshot.learningRuns; checkpoints = snapshot.checkpoints; evaluations = snapshot.evaluations
        pendingFeedback = snapshot.pendingFeedback
        rewardPrograms = snapshot.rewardPrograms
        recordingSelections = snapshot.recordingSelections
        var links: [UUID: Set<UUID>] = [:]
        for agent in agents { links[agent.id] = try await store.recordingIDs(for: agent.id) }
        recordingLinks = links
        var models: [UUID: Set<UUID>] = [:]
        for agent in agents { models[agent.id] = try await store.checkpointIDs(for: agent.id) }
        checkpointLinks = models
    }
}
