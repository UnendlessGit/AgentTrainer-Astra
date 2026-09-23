import AppKit
import AstraCore
import AstraPlatform
import Observation

struct InferenceOptions: Sendable {
    var deterministic = true
    var seed = 0
    var contextIDs: [Int] = []
}

/// Image production and process I/O are injected independently. Tests use the
/// same coordinator, leased transport, validation and stop paths without TCC.
struct InferenceImage: Sendable {
    let metadata: FrameMetadata
    let pixels: @Sendable () throws -> Data
    var coverage: CaptureFrameCoverage? = nil
}

struct InferenceCapture: Sendable {
    let start: @Sendable (@escaping @Sendable (InferenceImage) -> Void, @escaping @Sendable (CaptureHealth) -> Void) async throws -> Void
    let stop: @Sendable () async -> Void
}

struct InferenceRuntime: Sendable {
    let start: @Sendable () async throws -> WireMessage
    let request: @Sendable (String, JSONValue, UUID, Duration, Bool) async throws -> WireMessage
    /// Completion proves that the process has exited and no mapped bytes remain
    /// in its ownership. Implementations may not return on shutdown ack alone.
    let shutdown: @Sendable () async -> Int32?
}

struct InferenceDependencies: Sendable {
    let runtime: @Sendable (String, @escaping ComputeProcess.EventHandler, @escaping ComputeProcess.FailureHandler) -> InferenceRuntime
    let capture: @Sendable (CaptureSource) -> InferenceCapture
    let activate: @MainActor @Sendable (CaptureSource) throws -> Void
    var countdownSeconds = 3
    /// Live control always uses the paired recovery protocol. Injected virtual
    /// runtimes have no physical input ownership and are tested independently.
    var protectsPhysicalInputs = false
    var controlOwner = NativeControlOwner()
    var showOperator: @MainActor @Sendable () -> Void = {}

    static func live(bundle: Bundle) -> Self {
        let helpers = bundle.bundleURL.appendingPathComponent("Contents/Helpers")
        return Self(runtime: { role, events, failure in
            let path = role == "actor" ? "AstraCompute.app/Contents/MacOS/AstraCompute" : "AstraControl.app/Contents/MacOS/AstraControl"
            let process = ComputeProcess(executable: helpers.appendingPathComponent(path), arguments: role == "actor" ? ["--role", "actor"] : [],
                                         expectedRole: role, allowedEvents: role == "control" ? ["control.receipt", "control.stopped", "control.error"] : [],
                                         onEvent: events, onFailure: failure)
            return InferenceRuntime(start: { try await process.start() }, request: { kind, payload, run, timeout, acceptingError in
                try await process.request(kind: kind, payload: payload, runID: run, timeout: timeout, acceptingError: acceptingError)
            }, shutdown: { await process.shutdown(); return await process.terminationStatus() })
        }, capture: { source in
            let capture = ScreenCaptureGroup()
            return InferenceCapture(start: { onFrame, onHealth in
                try await capture.start(source: source, fps: 30, showsCursor: false, onFrame: { frame in
                    let metadata = FrameMetadata(id: frame.id, eventNanos: frame.eventNanos, observedNanos: frame.observedNanos,
                        surface: frame.surface, byteCount: frame.surface.pixelWidth * frame.surface.pixelHeight * 4, codec: "raw")
                    onFrame(InferenceImage(metadata: metadata, pixels: { try frame.copyCompactPixels() }, coverage: frame.coverage))
                }, onHealth: { _, health in onHealth(health) })
            }, stop: { await capture.stop() })
        }, activate: { source in
            if let pid = source.applicationPID {
                guard let application = NSRunningApplication(processIdentifier: pid), !application.isTerminated,
                      source.applicationLaunchDate == nil || source.applicationLaunchDate == application.launchDate else {
                    throw AstraError("inference.targetChanged", "The selected application closed or restarted. Choose it again.")
                }
                application.activate()
            } else {
                // Display observation must match the actual unobstructed
                // desktop, not pixels with an invisible control panel removed.
                NSApp?.hide(nil)
            }
        }, protectsPhysicalInputs: true, controlOwner: .shared, showOperator: {
            NSApp?.unhide(nil); NSApp?.activate(ignoringOtherApps: true)
        })
    }
}

/// Capture retains one frame per ordered source role. Reading a group is atomic;
/// partial startup never becomes a smaller observation with different slot meaning.
final class InferenceFrameInbox: @unchecked Sendable {
    private let lock = NSLock()
    private var latest: [String: InferenceImage] = [:]
    private var sourceIDs: [String]
    private var issue: String?
    private var closed = false
    init(sourceIDs: [String] = []) {
        self.sourceIDs = sourceIDs
        if sourceIDs.count > 16 || Set(sourceIDs).count != sourceIDs.count { issue = "Invalid capture source membership." }
    }
    func receive(_ image: InferenceImage) {
        lock.withLock {
            guard !closed else { return }
            let id = image.metadata.surface.id
            // Unbound legacy fixtures acquire one source; production supplies
            // the full fixed group before starting any streams.
            if sourceIDs.isEmpty { sourceIDs = [id] }
            guard sourceIDs.contains(id) else { issue = "Capture returned an unexpected source."; return }
            if let previous = latest[id], image.metadata.observedNanos < previous.metadata.observedNanos {
                issue = "Capture observation time moved backwards."; return
            }
            latest[id] = image
        }
    }
    func health(_ value: CaptureHealth) {
        lock.withLock {
            guard !closed else { return }
            switch value {
            case .unavailable(let message): issue = message
            case .stopped: issue = "The capture stream stopped unexpectedly."
            case .coverage(let evidence):
                let id = evidence.surface.id
                guard var frame = latest[id] else { issue = "Capture coverage arrived without its source frame."; return }
                do {
                    try evidence.validated(frame: frame.metadata, cutoffNanos: evidence.verifiedAtNanos)
                    if let previous = frame.coverage {
                        guard previous.streamID == evidence.streamID, evidence.throughNanos >= previous.throughNanos,
                              evidence.verifiedAtNanos >= previous.verifiedAtNanos else {
                            throw AstraError("capture.coverageOrder", "Capture coverage changed stream or moved backwards.")
                        }
                    }
                    frame.coverage = evidence; latest[id] = frame
                } catch { issue = error.localizedDescription }
            default: break
            }
        }
    }
    func readAll() throws -> [InferenceImage]? {
        try lock.withLock {
            if let issue { throw AstraError("inference.capture", issue) }
            guard !closed, !sourceIDs.isEmpty, latest.count == sourceIDs.count else { return nil }
            return sourceIDs.compactMap { latest[$0] }
        }
    }
    func read() throws -> InferenceImage? {
        guard let frames = try readAll() else { return nil }
        guard frames.count == 1 else { throw AstraError("inference.sourceGroup", "This operation requires the complete source group.") }
        return frames[0]
    }
    func close() { lock.withLock { closed = true; latest.removeAll() } }
}

struct InferencePolicyDetails: Sendable {
    let periodMS: Int
    let leadMS: Int
    let capacity: Int
    let capabilities: ActionCapabilities
    let contextSizes: [Int]

    init(_ payload: JSONValue, checkpoint: CheckpointDocument, runID: UUID, ringID: UUID) throws {
        guard payload.fields?["runID"]?.uuid == runID, payload.fields?["checkpointID"]?.uuid == checkpoint.id,
              payload.fields?["policySignature"]?.text == checkpoint.policySignature,
              payload.fields?["ringID"]?.uuid == ringID, let model = payload.fields?["model"]?.fields,
              model["schema_version"]?.int == 2, let period = model["period_ms"]?.int, (1...1000).contains(period),
              let lead = model["lead_ms"]?.int, (1...2000).contains(lead),
              let capacity = model["packet_capacity"]?.int, [16, 32, 64].contains(capacity),
              case .array(let contexts) = model["context_sizes"], contexts.count <= 32,
              contexts.allSatisfy({ $0.int.map { (1...65_536).contains($0) } == true }) else {
            throw AstraError("inference.policy", "The selected checkpoint has an incompatible identity, context vocabulary or execution timing. Live control requires a positive execution lead.")
        }
        capabilities = try payload.required("actions").decode(ActionCapabilities.self)
        guard !capabilities.isEmpty else { throw AstraError("inference.controls", "This checkpoint has no controls available to execute.") }
        periodMS = period; leadMS = lead; self.capacity = capacity; contextSizes = contexts.compactMap(\.int)
    }
}

/// One run owns capture, actor state, frame leases and a NativeControlSession.
/// The session maintains control liveness independently of MLX. Stop disarms
/// before waiting for the actor; missed action deadlines are never retimed.
@MainActor @Observable final class InferenceCoordinator {
    private(set) var isBusy = false
    private(set) var isStopping = false
    private(set) var activeAgentID: UUID?
    private(set) var runID: UUID?
    private(set) var phase = ""
    private(set) var failure: String?
    private(set) var stopReason: String?
    private(set) var countdown: Int?
    /// Packets admitted by the control helper; production and execution are separate.
    private(set) var decisions = 0
    private(set) var producedPackets: UInt64 = 0
    private(set) var executedPackets = 0
    private(set) var lastLatencyMS: Double?
    private(set) var maximumLatencyMS: Double?
    private(set) var warmupLatencyMS: [Double] = []
    private(set) var policy: InferencePolicyDetails?
    private(set) var resultsURL: URL?
    private(set) var cleanupConfirmed = false
    private(set) var cleanupRecoveredByGuardian = false
    var requiresManualControlCleanupAcknowledgement: Bool { !isBusy && unconfirmedControlSessionID != nil }
    private let store: LibraryStore
    private let root: URL
    private let dependencies: InferenceDependencies
    private var work: Task<Void, Never>?
    private var urgentStop: Task<Void, Never>?
    private var actor: PolicyActorSession?
    private var control: NativeControlSession?
    private var capture: InferenceCapture?
    private var inbox: InferenceFrameInbox?
    private var stopRequested = false
    private var userOrInterventionStop = false
    private var unconfirmedControlSessionID: UUID?
    private var collectionSink: InferenceCollectionSink?
    private var collectionFault: InferenceCollectionFault?
    private var collectionNextSequence: UInt64 = 0
    private let correctionBuffer = InferenceCorrectionBuffer()
    private var correctionSource: CaptureSource?
    private var correctionCheckpoint: CheckpointDocument?
    private var correctionAgentID: UUID?
    private var correctionContexts: [Int] = []
    private var correctionContextValues: [UUID: UUID] = [:]
    private var controlJoinedAtNanos: UInt64?
    var retainedCorrectionCheckpointID: UUID? { correctionSource == nil ? nil : correctionCheckpoint?.id }
    var correctionOwnerID: UUID? { correctionAgentID }
    var canRecordCorrection: Bool { !isBusy && cleanupConfirmed && producedPackets > 0 && controlJoinedAtNanos != nil && correctionSource != nil }

    func prepareCorrection(for requestingAgentID: UUID, requestedAtNanos: UInt64) async throws -> InferenceCorrection {
        guard correctionAgentID == requestingAgentID else { throw AstraError("correction.agent", "Open the agent that produced this run to record its correction.") }
        await stopAndWait()
        guard canRecordCorrection, let source = correctionSource, let checkpoint = correctionCheckpoint,
              let agentID = correctionAgentID, let runID, let joined = controlJoinedAtNanos else {
            throw AstraError("correction.cleanup", "Finish the agent run with confirmed control release before recording a correction.")
        }
        let seed = CorrectionRecordingSeed(sourceRunID: runID, sourceCheckpointID: checkpoint.id,
            sourcePolicySignature: checkpoint.policySignature, contextIDs: correctionContexts,
            requestedAtNanos: requestedAtNanos, controlJoinedAtNanos: joined,
            observations: correctionBuffer.snapshot(through: min(requestedAtNanos, joined)))
        return .init(agentID: agentID, checkpoint: checkpoint, source: source, seed: seed, contextValues: correctionContextValues)
    }
    func discardCorrectionHistory() { correctionBuffer.clear(); correctionSource = nil; correctionCheckpoint = nil; correctionAgentID = nil }

    init(store: LibraryStore, root: URL, bundle: Bundle = .main, dependencies: InferenceDependencies? = nil) {
        self.store = store; self.root = root; self.dependencies = dependencies ?? .live(bundle: bundle)
    }

    func start(agent: AgentDocument, checkpoint: CheckpointDocument, source: CaptureSource, options: InferenceOptions,
               collection: InferenceCollectionSink? = nil) throws {
        guard !isBusy else { throw AstraError("inference.busy", "Stop the current agent before starting another run.") }
        guard dependencies.controlOwner.priorCleanupJoined else { throw AstraError("inference.previousControl", "The previous control owner has not joined confirmed cleanup. Resolve its cleanup warning before starting another run.") }
        guard collection == nil || !options.deterministic else { throw AstraError("inference.collectionMode", "Reinforcement collection requires categorical policy sampling.") }
        guard options.seed >= 0, options.seed <= 1_000_000_000, options.contextIDs.count <= 32,
              options.contextIDs.allSatisfy({ $0 >= 0 && $0 < 65_536 }) else {
            throw AstraError("inference.options", "Choose a supported environment and valid inference settings.")
        }
        _ = try source.captureBindings()
        try store.layout.requireAvailable(.models)
        let run = UUID()
        correctionBuffer.clear(); correctionSource = source; correctionCheckpoint = checkpoint; correctionAgentID = agent.id
        correctionContexts = options.contextIDs; correctionContextValues = [:]; controlJoinedAtNanos = nil
        isBusy = true; isStopping = false; activeAgentID = agent.id; runID = run; stopRequested = false; userOrInterventionStop = false
        failure = nil; stopReason = nil; phase = "Opening the local policy…"; decisions = 0; producedPackets = 0; executedPackets = 0
        lastLatencyMS = nil; maximumLatencyMS = nil; warmupLatencyMS = []; policy = nil; resultsURL = nil
        cleanupConfirmed = false; cleanupRecoveredByGuardian = false; unconfirmedControlSessionID = nil
        collectionSink = collection
        collectionFault = collection == nil ? nil : InferenceCollectionFault()
        collectionNextSequence = 0
        work = Task { [weak self] in await self?.perform(agent: agent, checkpoint: checkpoint, source: source, options: options, run: run) }
    }

    func stopAndWait() async {
        requestStop()
        await work?.value
    }

    /// Explicit user acknowledgement unlocks future control; it does not
    /// rewrite the previous run's persisted or displayed cleanup evidence.
    func acknowledgeManualControlCleanup() throws {
        guard !isBusy, let id = unconfirmedControlSessionID else { throw AstraError("inference.cleanupWarning", "There is no completed control cleanup warning to acknowledge.") }
        try dependencies.controlOwner.acknowledgeManualCleanup(sessionID: id)
        unconfirmedControlSessionID = nil
    }

    private func requestStop(reason: String? = nil) {
        guard isBusy, !stopRequested else { return }
        if let reason, failure == nil { failure = reason }
        userOrInterventionStop = reason == nil
        stopRequested = true; isStopping = true; phase = "Stopping and releasing controls…"
        control?.requestStop(); work?.cancel()
        inbox?.close()
        if urgentStop == nil {
            let control = self.control, actor = self.actor, capture = self.capture
            urgentStop = Task {
                if let control { _ = try? await control.disarm() }
                // Shutdown interrupts a blocked actor request. Its completion
                // joins process exit before its session retires mapped leases.
                async let stoppedCapture: Void = capture?.stop() ?? ()
                await actor?.shutdown(interruptPending: true)
                await stoppedCapture
            }
        }
    }

    private func checkRunning() throws {
        if stopRequested { throw CancellationError() }
        if let failure { throw AstraError("inference.stopped", failure) }
        try Task.checkCancellation()
    }

    private func makeActorRuntime(run: UUID) -> InferenceRuntime {
        dependencies.runtime("actor", { [weak self] message in
            Task { @MainActor [weak self] in
                guard let self, runID == run, isBusy, !stopRequested else { return }
                requestStop(reason: "The actor returned an unexpected event: \(message.kind).")
            }
        }, { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self, runID == run, isBusy, !stopRequested else { return }
                requestStop(reason: error.localizedDescription)
            }
        })
    }

    private func makeControl(configuration: NativeControlConfiguration) -> NativeControlSession {
        let dependencies = dependencies, sink = collectionSink, fault = collectionFault, run = configuration.runID
        let factory = NativeControlRuntimeFactory(protectsPhysicalInputs: dependencies.protectsPhysicalInputs) { events, failure in
            let runtime = dependencies.runtime("control", events, failure)
            return NativeControlRuntime(start: runtime.start, request: { kind, payload, run, timeout in
                try await runtime.request(kind, payload, run, timeout, true)
            }, shutdown: runtime.shutdown)
        }
        return NativeControlSession(configuration: configuration, owner: dependencies.controlOwner, runtimeFactory: factory,
            onEvent: { [weak self] event in
                // Synchronous offer precedes UI dispatch. The session joins
                // callbacks before collector finalization, including Stop.
                do {
                    if let sink {
                        guard event.message.runID == run else { throw AstraError("inference.collectionRun", "Control collection received a foreign run.") }
                        try sink.offer(.control(event.message))
                    }
                } catch { fault?.record(error); throw error }
                Task { @MainActor [weak self] in
                    guard let self, runID == run, control?.sessionID == event.sessionID, isBusy, !stopRequested else { return }
                    receive(event.message)
                }
            }, onFailure: { [weak self] id, error in
                Task { @MainActor [weak self] in
                    guard let self, runID == run, control?.sessionID == id, isBusy, !stopRequested else { return }
                    if Self.isIntervention(error) { stopReason = error.message; requestStop() }
                    else { requestStop(reason: error.localizedDescription) }
                }
            })
    }

    private func perform(agent: AgentDocument, checkpoint: CheckpointDocument, source: CaptureSource, options: InferenceOptions, run: UUID) async {
        let directory = root.appendingPathComponent("Runs", isDirectory: true).appendingPathComponent(run.uuidString.lowercased(), isDirectory: true)
        let startedAt = Date()
        do {
            guard try await store.checkpointIDs(for: agent.id).contains(checkpoint.id),
                  let saved = try await store.snapshot().checkpoints.first(where: { $0.id == checkpoint.id }),
                  saved.matchesIdentity(of: checkpoint) else {
                throw AstraError("inference.checkpoint", "This checkpoint is no longer linked to the selected agent.")
            }
            try await LearningFiles.write(.object(["runID": .string(run.uuidString.lowercased()), "agentID": .string(agent.id.uuidString.lowercased()),
                "checkpointID": .string(checkpoint.id.uuidString.lowercased()), "policySignature": .string(checkpoint.policySignature),
                "sourceID": .string(source.id), "sourceName": .string(source.name), "deterministic": .bool(options.deterministic),
                "seed": .integer(Int64(options.seed)), "contextIDs": .array(options.contextIDs.map { .integer(Int64($0)) }),
                "startedAt": .string(startedAt.ISO8601Format())]), to: directory.appendingPathComponent("configuration.json"), exclusive: true)
            try checkRunning()
            let inbox = InferenceFrameInbox(sourceIDs: try source.captureBindings().map(\.id)); self.inbox = inbox
            let capture = dependencies.capture(source); self.capture = capture
            phase = "Observing the environment…"
            try await capture.start({ inbox.receive($0) }, { inbox.health($0) })
            let first = try await waitForFrames(inbox)
            let initialSurfaces = try first.map { try $0.metadata.surface.validated() }
            let initialSurface = initialSurfaces[0]
            let geometryRevision = initialSurfaces.count == 1 ? initialSurface.geometryRevision : 0
            try checkRunning()
            let actor = PolicyActorSession(runID: run, ringURL: directory.appendingPathComponent("frames.astraring"),
                slotCapacity: first[0].metadata.byteCount, slotCapacities: first.map(\.metadata.byteCount), runtime: makeActorRuntime(run: run))
            // Publish ownership before preparation can suspend. Stop explicitly
            // interrupts this session's owned task and joins its mapped leases.
            self.actor = actor
            let selected = PolicyActorCheckpoint(document: checkpoint,
                directory: store.checkpointDirectory(id: checkpoint.id))
            let ready = try await actor.prepare(checkpoint: selected, collection: collectionSink != nil,
                deterministic: options.deterministic, mode: .fresh(seed: UInt64(options.seed)))
            try checkRunning()
            guard let details = await actor.policy else { throw AstraError("inference.policy", "The actor did not prepare its policy.") }
            try checkRunning()
            policy = details
            if let model = ready.payload.fields?["model"], let vocabulary = try ContextVocabulary.from(model: model) {
                for (field, index) in zip(vocabulary.fields, options.contextIDs) where index > 0 {
                    guard field.values.indices.contains(index - 1) else { throw AstraError("correction.context", "The run's named context value is unavailable.") }
                    correctionContextValues[field.id] = field.values[index - 1].id
                }
            }
            if let collectionSink { try collectionSink.offer(.prepared(runID: run, actor: ready.payload)) }
            phase = "Warming the local policy…"
            // Warmup runs the actual mapped image/preprocessing/policy/decoder
            // path, but the actor restores recurrent, sequence and RNG state.
            // The first step pays compilation cost. Two measured warm steps
            // must leave headroom within both immutable cadence and lead.
            let warmEpisode = UUID()
            _ = try await actor.reset(confirmedEpisodeID: warmEpisode, contextIDs: options.contextIDs, timeout: .seconds(20))
            var warmState = ControlState(); warmState.valid = true
            warmState.pointer = Point2D(x: initialSurface.globalBounds.x + initialSurface.globalBounds.width / 2,
                                        y: initialSurface.globalBounds.y + initialSurface.globalBounds.height / 2)
            for index in 0..<3 {
                try checkRunning()
                guard let frames = try inbox.readAll(), frames.map(\.metadata.surface) == initialSurfaces else {
                    throw AstraError("inference.geometryChanged", "The environment changed while warming the policy. Choose it again and restart.")
                }
                let cutoff = MonotonicClock.now
                guard frames.allSatisfy({ $0.metadata.observedNanos <= cutoff }) else { throw AstraError("inference.causality", "Capture returned an image from a future observation.") }
                warmState.observedNanos = cutoff
                let begin = MonotonicClock.now
                let observation = ControlObservation(controlState: warmState, executedEvents: [], intervalCovered: true,
                    cutoffNanos: cutoff, lastSequence: nil)
                _ = try await actor.warmup(.init(frames: frames, geometryRevision: geometryRevision, controls: observation))
                try checkRunning()
                let elapsed = Double(MonotonicClock.now - begin) / 1_000_000
                warmupLatencyMS.append(elapsed)
                if index > 0 {
                    let budget = Double(min(details.periodMS, details.leadMS))
                    guard elapsed <= budget - max(5, budget * 0.1) else {
                        throw AstraError("inference.warmupTiming", "The local policy needs \(Int(elapsed.rounded(.up))) ms per decision in this environment. Its \(details.periodMS) ms cadence and \(details.leadMS) ms execution lead leave insufficient headroom. Stop other GPU work or use a checkpoint trained with suitable timing.")
                    }
                }
            }
            // Warmup is not an environment episode and does not consume the
            // run's action sequence. Confirm the real initial state only now.
            let episode = UUID()
            // Preparation seeds the actor once; all isolated warmups restore
            // that key. Reset clears recurrence without replacing this stream.
            _ = try await actor.reset(confirmedEpisodeID: episode, contextIDs: options.contextIDs, timeout: .seconds(20))
            try checkRunning()
            if dependencies.countdownSeconds > 0 {
                for remaining in (1...dependencies.countdownSeconds).reversed() {
                    countdown = remaining; phase = "Agent starts in \(remaining)…"
                    if remaining == min(2, dependencies.countdownSeconds) { try dependencies.activate(source) }
                    try await Task.sleep(for: .seconds(1)); try checkRunning()
                }
            } else { try dependencies.activate(source) }
            countdown = nil
            guard let fresh = try inbox.readAll(), fresh.map(\.metadata.surface) == initialSurfaces else {
                throw AstraError("inference.geometryChanged", "The environment changed size or position while preparing. Choose it again and restart.")
            }
            let scope = ControlScope(surfaces: initialSurfaces, applicationPID: source.applicationPID, windowID: source.windowID,
                                     wholeDesktop: source.kind == .desktop, stopOnPhysicalInput: true, geometryRevision: geometryRevision)
            try checkRunning()
            let configuration = try NativeControlConfiguration(runID: run, scope: scope, capabilities: details.capabilities,
                packetCapacity: details.capacity, recoveryDirectory: directory)
            let control = makeControl(configuration: configuration); self.control = control
            _ = try await control.start()
            try checkRunning()
            phase = "Agent running · move the pointer or press a key to take over"
            var nextDecision = MonotonicClock.now
            var lastEventSequence: UInt64?
            while true {
                try checkRunning()
                try await sleep(until: nextDecision)
                try checkRunning()
                // Select the image first, then obtain an atomic input cutoff.
                // A newer image never enters an earlier control observation.
                guard let images = try inbox.readAll() else { throw AstraError("inference.capture", "The capture stream has no current frame.") }
                guard images.map(\.metadata.surface) == initialSurfaces else {
                    throw AstraError("inference.geometryChanged", "The environment changed size or position. Select the environment again before restarting.")
                }
                let observed = try await observation(control, after: lastEventSequence)
                try checkRunning()
                let begin = MonotonicClock.now
                let sink = collectionSink, correctionBuffer = correctionBuffer
                let ticket = try await actor.beginPrediction(.init(frames: images, geometryRevision: geometryRevision, controls: observed), onObservation: { owned in
                    // This bounded offer completes before sampling and retains
                    // the exact CPU copy published by the actor session.
                    if let sink { try sink.offer(.observation(owned.collected())) }
                    correctionBuffer.append(owned)
                })
                let result = try await ticket.value()
                producedPackets = result.packet.sequence + 1
                try checkRunning()
                guard let current = try inbox.readAll(), current.map(\.metadata.surface) == initialSurfaces else {
                    throw AstraError("inference.geometryChanged", "The environment changed while the policy was deciding. Select it again before restarting.")
                }
                let packet = result.packet
                let now = MonotonicClock.now
                lastLatencyMS = Double(now - begin) / 1_000_000
                maximumLatencyMS = max(maximumLatencyMS ?? 0, lastLatencyMS ?? 0)
                guard now < packet.executeAtNanos else {
                    throw AstraError("inference.deadline", "The local policy missed its \(details.leadMS) ms execution lead. Use a checkpoint trained with a longer lead, or stop other GPU work before restarting.")
                }
                if let collectionSink {
                    try collectionSink.offer(.decision(result.response.payload))
                    collectionNextSequence = packet.sequence + 1
                }
                let submitted = try await control.submit(packet)
                guard submitted.admitted else { throw AstraError("inference.admission", "The control helper did not admit the action packet.") }
                decisions += 1; lastEventSequence = observed.lastSequence
                let addition = observed.cutoffNanos.addingReportingOverflow(UInt64(details.periodMS) * 1_000_000)
                guard !addition.overflow else { throw AstraError("inference.clock", "The decision clock is exhausted.") }
                nextDecision = addition.partialValue
            }
        } catch is CancellationError { /* Stop already records any originating failure. */ }
        catch {
            if !stopRequested {
                if let error = error as? AstraError, Self.isIntervention(error) { stopReason = error.message; requestStop() }
                else { requestStop(reason: error.localizedDescription) }
            }
        }
        await finish()
        if let message = collectionFault?.message {
            failure = [failure, "Collection evidence is incomplete: \(message)"].compactMap { $0 }.joined(separator: "\n")
        }
        var summary: JSONValue = .object(["runID": .string(run.uuidString.lowercased()), "status": .string(failure == nil ? "stopped" : "failed"),
            "decisions": .integer(Int64(decisions)), "producedPackets": .unsigned(producedPackets), "executedPackets": .integer(Int64(executedPackets)),
            "warmupLatencyMS": .array(warmupLatencyMS.map(JSONValue.number)),
            "elapsedSeconds": .number(Date().timeIntervalSince(startedAt)), "maximumLatencyMS": maximumLatencyMS.map(JSONValue.number) ?? .null,
            "issue": failure.map(JSONValue.string) ?? .null, "stopReason": stopReason.map(JSONValue.string) ?? .null,
            "cleanupConfirmed": .bool(cleanupConfirmed), "cleanupRecoveredByGuardian": .bool(cleanupRecoveredByGuardian),
            "finishedAt": .string(Date().ISO8601Format())])
        if let collectionSink {
            phase = "Saving collected experience…"
            do { try await collectionSink.finish(summary) }
            catch {
                failure = [failure, "Collection could not be finalized: \(error.localizedDescription)"].compactMap { $0 }.joined(separator: "\n")
                var fields = summary.fields ?? [:]; fields["status"] = .string("failed"); fields["issue"] = failure.map(JSONValue.string)
                summary = .object(fields)
            }
        }
        do {
            let destination = directory.appendingPathComponent("results.json")
            try await LearningFiles.write(summary, to: destination, exclusive: true); resultsURL = destination
        } catch { failure = [failure, "The run summary could not be saved: \(error.localizedDescription)"].compactMap { $0 }.joined(separator: "\n") }
        isBusy = false; isStopping = false; activeAgentID = nil; countdown = nil; work = nil
        collectionSink = nil
        collectionFault = nil
        phase = failure == nil && cleanupConfirmed
            ? (cleanupRecoveredByGuardian ? "Agent stopped · guardian recovered controls" : "Agent stopped · controls released")
            : "Agent stopped · needs attention"
    }

    private func waitForFrames(_ inbox: InferenceFrameInbox) async throws -> [InferenceImage] {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            try checkRunning()
            if let images = try inbox.readAll() {
                for image in images {
                    _ = try image.metadata.validated()
                    guard image.metadata.eventNanos <= image.metadata.observedNanos,
                          image.metadata.observedNanos <= MonotonicClock.now else { throw AstraError("inference.causality", "Capture returned an invalid image timestamp.") }
                }
                return images
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw AstraError("inference.captureTimeout", "The selected environment did not produce a frame in time.")
    }

    private func observation(_ control: NativeControlSession, after: UInt64?) async throws -> AstraPlatform.ControlObservation {
        let deadline = MonotonicClock.now + 20_000_000
        while true {
            let value = try await control.observation(afterSequence: after)
            guard value.intervalCovered, value.cutoffNanos <= MonotonicClock.now, value.controlState.observedNanos <= value.cutoffNanos, value.controlState.pointer.isFinite,
                  value.executedEvents.count <= 2048 else { throw AstraError("inference.inputCoverage", "Executed input history is unavailable or incomplete.") }
            var previous = after
            for event in value.executedEvents {
                guard previous.map({ event.sequence > $0 }) ?? true, event.eventNanos <= event.observedNanos,
                      event.observedNanos <= value.cutoffNanos else { throw AstraError("inference.inputCausality", "Executed inputs have invalid sequence or availability timestamps.") }
                previous = event.sequence
            }
            guard previous == value.lastSequence else { throw AstraError("inference.inputCursor", "Executed input history does not match its cursor.") }
            if value.controlState.valid { try value.validateControlCoverage(); return value }
            try checkRunning()
            guard MonotonicClock.now < deadline else { throw AstraError("inference.inputBusy", "The control helper could not provide a settled input observation in time.") }
            try await Task.sleep(for: .milliseconds(1))
        }
    }

    private func receive(_ message: WireMessage) {
        guard message.runID == runID else { requestStop(reason: "The control helper reported an event for another run."); return }
        switch message.kind {
        case "control.stopped" where ["physicalTakeover", "emergencyStop"].contains(message.payload.fields?["cause"]?.text ?? ""):
            stopReason = message.payload.fields?["reason"]?.text ?? "You took over control."
            requestStop()
        case "control.stopped", "control.error":
            requestStop(reason: message.payload.fields?["reason"]?.text ?? message.payload.fields?["message"]?.text ?? "Desktop control stopped.")
        case "control.receipt":
            do {
                let receipt = try message.payload.required("receipt").decode(ExecutionReceipt.self)
                if receipt.status == .admitted { return }
                // Identity, command evidence and terminal uniqueness have
                // already passed NativeControlSession before this callback.
                guard receipt.status == .executed else { throw AstraError("inference.execution", "An action was late, cancelled or could not be posted.") }
                executedPackets += 1
            } catch { requestStop(reason: error.localizedDescription) }
        default: requestStop(reason: "The control helper returned an unsupported event.")
        }
    }

    private func sleep(until nanos: UInt64) async throws {
        let now = MonotonicClock.now
        if nanos > now { try await Task.sleep(for: .nanoseconds(Int64(min(nanos - now, UInt64(Int64.max))))) }
    }

    private static func isIntervention(_ issue: AstraError) -> Bool {
        ["control.physicalTakeover", "control.emergencyStop"].contains(issue.code)
    }

    private func finish() async {
        stopRequested = true; isStopping = true; countdown = nil
        control?.requestStop()
        await urgentStop?.value; urgentStop = nil
        if let control {
            phase = "Waiting for owned controls to release…"
            let completion = await control.shutdown()
            executedPackets = control.executedPacketCount
            cleanupConfirmed = completion.cleanupConfirmed
            cleanupRecoveredByGuardian = completion.recoveredByGuardian
            if !completion.cleanupConfirmed { unconfirmedControlSessionID = completion.sessionID }
            if let issue = completion.issue {
                if Self.isIntervention(issue) { stopReason = stopReason ?? issue.message }
                // An already-forwarded packet may receive the helper's real
                // rejection after user Stop disarms it. Its raw evidence is
                // retained; that expected race does not make Stop a failure.
                else if failure == nil, !(userOrInterventionStop && issue.code == "control.rejected") { failure = issue.message }
            }
        } else { cleanupConfirmed = true }
        if !cleanupConfirmed {
            let message = "The control helper exited before input cleanup could be confirmed. Release any held controls manually before starting another run."
            if failure?.contains("input cleanup could be confirmed") != true { failure = [failure, message].compactMap { $0 }.joined(separator: "\n") }
        }
        if cleanupConfirmed { controlJoinedAtNanos = MonotonicClock.now }
        self.control = nil
        inbox?.close()
        await capture?.stop(); capture = nil; inbox = nil
        if let actor {
            await actor.shutdown(interruptPending: true)
            let state = await actor.state
            producedPackets = state.nextPacketSequence
            // Actor knowledge alone does not prove that the sink retained the
            // result: Stop can arrive after sampling but before live admission.
            if collectionSink != nil, !state.sampledProgressKnown || state.nextPacketSequence != collectionNextSequence {
                collectionFault?.record(AstraError("inference.unresolvedPrediction", "The actor ended before its in-flight sampled result could be fully retained. Its final random-stream progress is unverified."))
            }
        }
        actor = nil
    }
}
