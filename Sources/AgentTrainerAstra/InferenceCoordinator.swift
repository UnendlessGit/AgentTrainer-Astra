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
            let capture = ScreenCapture()
            return InferenceCapture(start: { onFrame, onHealth in
                try await capture.start(source: source, fps: 30, showsCursor: false, onFrame: { frame in
                    let metadata = FrameMetadata(id: frame.id, eventNanos: frame.eventNanos, observedNanos: frame.observedNanos,
                        surface: frame.surface, byteCount: frame.surface.pixelWidth * frame.surface.pixelHeight * 4, codec: "raw")
                    onFrame(InferenceImage(metadata: metadata, pixels: { try frame.copyCompactPixels() }, coverage: frame.coverage))
                }, onHealth: onHealth)
            }, stop: { await capture.stop() })
        }, activate: { source in
            if let pid = source.applicationPID {
                guard let application = NSRunningApplication(processIdentifier: pid), !application.isTerminated,
                      source.applicationLaunchDate == nil || source.applicationLaunchDate == application.launchDate else {
                    throw AstraError("inference.targetChanged", "The selected application closed or restarted. Choose it again.")
                }
                application.activate()
            }
        }, protectsPhysicalInputs: true)
    }
}

/// The producer replaces one retained frame synchronously on the capture queue.
/// No per-frame Task, compressed archive or UI actor queue retains SCK buffers.
final class InferenceFrameInbox: @unchecked Sendable {
    private let lock = NSLock()
    private var latest: InferenceImage?
    private var issue: String?
    private var closed = false
    func receive(_ image: InferenceImage) {
        lock.withLock {
            guard !closed else { return }
            if let latest, image.metadata.observedNanos < latest.metadata.observedNanos {
                issue = "Capture observation time moved backwards."; return
            }
            latest = image
        }
    }
    func health(_ value: CaptureHealth) {
        lock.withLock {
            guard !closed else { return }
            switch value {
            case .unavailable(let message): issue = message
            case .stopped: issue = "The capture stream stopped unexpectedly."
            case .coverage(let evidence):
                guard var latest else { issue = "Capture coverage arrived without its source frame."; return }
                do {
                    try evidence.validated(frame: latest.metadata, cutoffNanos: evidence.verifiedAtNanos)
                    if let previous = latest.coverage {
                        guard previous.streamID == evidence.streamID, evidence.throughNanos >= previous.throughNanos,
                              evidence.verifiedAtNanos >= previous.verifiedAtNanos else {
                            throw AstraError("capture.coverageOrder", "Capture coverage changed stream or moved backwards.")
                        }
                    }
                    latest.coverage = evidence; self.latest = latest
                } catch { issue = error.localizedDescription }
            default: break
            }
        }
    }
    func read() throws -> InferenceImage? {
        try lock.withLock {
            if let issue { throw AstraError("inference.capture", issue) }
            return latest
        }
    }
    func close() { lock.withLock { closed = true; latest = nil } }
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

/// One run owns capture, actor state, frame leases and the desktop helper. An
/// independent heartbeat task runs while awaiting MLX, and Stop disarms before
/// waiting for the actor. A missed action deadline is terminal, never retimed.
@MainActor @Observable final class InferenceCoordinator {
    private(set) var isBusy = false
    private(set) var isStopping = false
    private(set) var activeAgentID: UUID?
    private(set) var runID: UUID?
    private(set) var phase = ""
    private(set) var failure: String?
    private(set) var stopReason: String?
    private(set) var countdown: Int?
    private(set) var decisions = 0
    private(set) var executedPackets = 0
    private(set) var lastLatencyMS: Double?
    private(set) var maximumLatencyMS: Double?
    private(set) var warmupLatencyMS: [Double] = []
    private(set) var policy: InferencePolicyDetails?
    private(set) var resultsURL: URL?
    private(set) var cleanupConfirmed = false
    private(set) var cleanupRecoveredByGuardian = false
    private let store: LibraryStore
    private let root: URL
    private let dependencies: InferenceDependencies
    private var work: Task<Void, Never>?
    private var heartbeat: Task<Void, Never>?
    private var urgentStop: Task<Void, Never>?
    private var actor: InferenceRuntime?
    private var control: InferenceRuntime?
    private var capture: InferenceCapture?
    private var ring: SharedFrameRing?
    private var inbox: InferenceFrameInbox?
    private var stopRequested = false
    private var controlAdmissionAttempted = false
    private var controlRecovery: ControlRecoveryLedger?
    private var pending: [UUID: ActionPacket] = [:]
    private var collectionSink: InferenceCollectionSink?
    private var collectionFault: InferenceCollectionFault?

    init(store: LibraryStore, root: URL, bundle: Bundle = .main, dependencies: InferenceDependencies? = nil) {
        self.store = store; self.root = root; self.dependencies = dependencies ?? .live(bundle: bundle)
    }

    func start(agent: AgentDocument, checkpoint: CheckpointDocument, source: CaptureSource, options: InferenceOptions,
               collection: InferenceCollectionSink? = nil) throws {
        guard !isBusy else { throw AstraError("inference.busy", "Stop the current agent before starting another run.") }
        guard collection == nil || !options.deterministic else { throw AstraError("inference.collectionMode", "Reinforcement collection requires categorical policy sampling.") }
        guard options.seed >= 0, options.seed <= 1_000_000_000, options.contextIDs.count <= 32,
              options.contextIDs.allSatisfy({ $0 >= 0 && $0 < 65_536 }), [.window, .display].contains(source.kind) else {
            throw AstraError("inference.options", "Choose a supported environment and valid inference settings.")
        }
        let run = UUID()
        isBusy = true; isStopping = false; activeAgentID = agent.id; runID = run; stopRequested = false
        failure = nil; stopReason = nil; phase = "Opening the local policy…"; decisions = 0; executedPackets = 0
        lastLatencyMS = nil; maximumLatencyMS = nil; warmupLatencyMS = []; policy = nil; resultsURL = nil; pending = [:]
        controlAdmissionAttempted = false; cleanupConfirmed = false
        cleanupRecoveredByGuardian = false
        controlRecovery = nil
        collectionSink = collection
        collectionFault = collection == nil ? nil : InferenceCollectionFault()
        work = Task { [weak self] in await self?.perform(agent: agent, checkpoint: checkpoint, source: source, options: options, run: run) }
    }

    func stopAndWait() async {
        requestStop()
        await work?.value
    }

    private func requestStop(reason: String? = nil) {
        guard isBusy, !stopRequested else { return }
        if let reason, failure == nil { failure = reason }
        stopRequested = true; isStopping = true; phase = "Stopping and releasing controls…"
        heartbeat?.cancel(); work?.cancel()
        inbox?.close()
        if urgentStop == nil, let run = runID {
            let control = self.control, actor = self.actor, capture = self.capture
            urgentStop = Task {
                if let control { _ = try? await control.request("disarm", .object([:]), run, .seconds(2), false) }
                // Shutdown interrupts a blocked actor request. Its completion
                // joins process exit; the ring remains mapped until finish.
                async let stoppedCapture: Void = capture?.stop() ?? ()
                _ = await actor?.shutdown()
                await stoppedCapture
            }
        }
    }

    private func checkRunning() throws {
        if stopRequested { throw CancellationError() }
        if let failure { throw AstraError("inference.stopped", failure) }
        try Task.checkCancellation()
    }

    private func makeRuntime(_ role: String, run: UUID) -> InferenceRuntime {
        let sink = collectionSink, fault = collectionFault
        return dependencies.runtime(role, { [weak self] message in
            // Offer on the process I/O callback before dispatching UI updates.
            // Shutdown joins that callback, so final receipts cannot overtake
            // collector finalization or disappear behind stopRequested.
            if role == "control", let sink {
                do {
                    guard message.runID == run else { throw AstraError("inference.collectionRun", "Control collection received a foreign run.") }
                    try sink.offer(.control(message))
                } catch {
                    fault?.record(error)
                    Task { @MainActor [weak self] in
                        guard let self, runID == run, isBusy else { return }
                        requestStop(reason: error.localizedDescription)
                    }
                }
            }
            Task { @MainActor [weak self] in
                guard let self, runID == run, isBusy, !stopRequested else { return }
                receive(message)
            }
        }, { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self, runID == run, isBusy, !stopRequested else { return }
                requestStop(reason: error.localizedDescription)
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
            let inbox = InferenceFrameInbox(); self.inbox = inbox
            let capture = dependencies.capture(source); self.capture = capture
            phase = "Observing the environment…"
            try await capture.start({ inbox.receive($0) }, { inbox.health($0) })
            let first = try await waitForFrame(inbox)
            let initialSurface = try first.metadata.surface.validated()
            let ring = try await Task.detached {
                try SharedFrameRing(url: directory.appendingPathComponent("frames.astraring"), runID: run, slotCount: 2,
                                    slotCapacity: first.metadata.byteCount)
            }.value
            self.ring = ring
            try checkRunning()
            let actor = makeRuntime("actor", run: run), control = makeRuntime("control", run: run)
            self.actor = actor; self.control = control
            try validateHello(try await actor.start(), role: "actor"); try checkRunning()
            let checkpointPath = root.appendingPathComponent("Models").appendingPathComponent(checkpoint.id.uuidString.lowercased()).path
            var preparation: [String: JSONValue] = ["checkpointPath": .string(checkpointPath),
                "ring": .object(["path": .string(ring.url.path), "ringID": .string(ring.ringID.uuidString.lowercased())]),
                "seed": .integer(Int64(options.seed)), "deterministic": .bool(options.deterministic)]
            if collectionSink != nil { preparation["collection"] = .bool(true) }
            let ready = try await actor.request("inference.prepare", .object(preparation), run, .seconds(120), false)
            guard ready.kind == "ack", ready.runID == run else { throw AstraError("inference.run", "The actor prepared a different run.") }
            let details = try InferencePolicyDetails(ready.payload, checkpoint: checkpoint, runID: run, ringID: ring.ringID)
            guard options.contextIDs.count == details.contextSizes.count,
                  zip(options.contextIDs, details.contextSizes).allSatisfy({ $0 < $1 }) else {
                throw AstraError("inference.contexts", "Choose one valid value for every context in this checkpoint.")
            }
            policy = details; try checkRunning()
            if let collectionSink {
                guard ready.payload.fields?["collection"] == .bool(true), ready.payload.fields?["collectionVersion"] == .integer(1),
                      ready.payload.fields?["rngStreamID"]?.uuid != nil, ready.payload.fields?["deterministic"] == .bool(false) else {
                    throw AstraError("inference.collectionVersion", "The actor cannot provide the required on-policy collection evidence.")
                }
                try collectionSink.offer(.prepared(runID: run, actor: ready.payload))
            }
            phase = "Warming the local policy…"
            // Warmup runs the actual mapped image/preprocessing/policy/decoder
            // path, but the actor restores recurrent, sequence and RNG state.
            // The first step pays compilation cost. Two measured warm steps
            // must leave headroom within both immutable cadence and lead.
            let warmEpisode = UUID()
            let warmReset = try await reset(actor, run: run, episode: warmEpisode, contexts: options.contextIDs)
            var warmState = ControlState(); warmState.valid = true
            warmState.pointer = Point2D(x: initialSurface.globalBounds.x + initialSurface.globalBounds.width / 2,
                                        y: initialSurface.globalBounds.y + initialSurface.globalBounds.height / 2)
            for index in 0..<3 {
                try checkRunning()
                guard let frame = try inbox.read(), frame.metadata.surface == initialSurface else {
                    throw AstraError("inference.geometryChanged", "The environment changed while warming the policy. Choose it again and restart.")
                }
                let cutoff = MonotonicClock.now, observationID = UUID()
                guard frame.metadata.observedNanos <= cutoff else { throw AstraError("inference.causality", "Capture returned an image from a future observation.") }
                warmState.observedNanos = cutoff
                let begin = MonotonicClock.now
                let warm = try await predict(actor, ring: ring, frame: frame, run: run, episode: warmEpisode,
                    previousState: warmReset, cutoff: cutoff, controls: warmState, events: [], contexts: options.contextIDs,
                    observationID: observationID, warmup: true)
                try requireSuccess(warm); try checkRunning()
                guard warm.payload.fields?["warmup"] == .bool(true) else { throw AstraError("inference.warmup", "The actor did not isolate its warmup state.") }
                _ = try validateResult(warm.payload, checkpoint: checkpoint, run: run, episode: warmEpisode,
                    previousState: warmReset, observationID: observationID, cutoff: cutoff, sequence: 0, surface: initialSurface, details: details)
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
            var previousState = try await reset(actor, run: run, episode: episode, contexts: options.contextIDs,
                                                seed: collectionSink == nil ? options.seed : nil)
            if dependencies.protectsPhysicalInputs {
                controlRecovery = try await Task.detached {
                    try ControlRecoveryLedger(createAt: directory.appendingPathComponent("control-recovery.astracontrol"), runID: run, capabilities: details.capabilities)
                }.value
            }
            try validateHello(try await control.start(), role: "control"); try checkRunning()
            if dependencies.countdownSeconds > 0 {
                for remaining in (1...dependencies.countdownSeconds).reversed() {
                    countdown = remaining; phase = "Agent starts in \(remaining)…"
                    if remaining == min(2, dependencies.countdownSeconds) { try dependencies.activate(source) }
                    try await Task.sleep(for: .seconds(1)); try checkRunning()
                }
            } else { try dependencies.activate(source) }
            countdown = nil
            guard let fresh = try inbox.read(), fresh.metadata.surface == initialSurface else {
                throw AstraError("inference.geometryChanged", "The environment changed size or position while preparing. Choose it again and restart.")
            }
            let scope = ControlScope(surfaces: [initialSurface], applicationPID: source.applicationPID, windowID: source.windowID,
                                     wholeDesktop: false, stopOnPhysicalInput: true, geometryRevision: initialSurface.geometryRevision)
            try checkRunning()
            controlAdmissionAttempted = true
            let armed = try await control.request("arm", .encode(ArmRequest(runID: run, scope: scope, capabilities: details.capabilities,
                                                                           packetCapacity: details.capacity, recovery: controlRecovery?.descriptor)), run, .seconds(20), false)
            guard armed.kind == "ack", armed.runID == run, armed.payload.fields?["armed"] == .bool(true) else { throw AstraError("inference.arm", "The control helper did not arm the selected environment.") }
            if let recovery = controlRecovery {
                let snapshot = try recovery.snapshot()
                guard armed.payload.fields?["recoveryLedgerID"]?.uuid == recovery.descriptor.ledgerID,
                      armed.payload.fields?["guardianPID"]?.int == Int(snapshot.guardianPID), snapshot.guardianReady,
                      snapshot.everArmed, snapshot.executorPID > 0 else {
                    throw AstraError("inference.recoveryHandshake", "The control helper did not establish the requested independent cleanup protection.")
                }
            }
            // Stop may arrive while arm is in flight. Disarm again after its
            // response so an earlier stop request cannot leave a late arm live.
            if stopRequested { _ = try? await control.request("disarm", .object([:]), run, .seconds(2), false); throw CancellationError() }
            heartbeat = Task { [weak self] in
                do {
                    while !Task.isCancelled {
                        let beat = try await control.request("heartbeat", .object([:]), run, .seconds(1), false)
                        guard beat.kind == "ack", beat.runID == run else { throw AstraError("inference.heartbeat", "The control heartbeat changed run identity.") }
                        try await Task.sleep(for: .milliseconds(100))
                    }
                } catch {
                    guard !Task.isCancelled else { return }
                    self?.requestStop(reason: "The desktop control heartbeat stopped: \(error.localizedDescription)")
                }
            }
            phase = "Agent running · move the pointer or press a key to take over"
            var nextDecision = MonotonicClock.now
            var lastEventSequence: UInt64?
            var previousCutoff: UInt64?
            while true {
                try checkRunning()
                try await sleep(until: nextDecision)
                try checkRunning()
                // Select the image first, then obtain an atomic input cutoff.
                // A newer image never enters an earlier control observation.
                guard let image = try inbox.read() else { throw AstraError("inference.capture", "The capture stream has no current frame.") }
                guard image.metadata.surface == initialSurface else {
                    throw AstraError("inference.geometryChanged", "The environment changed size or position. Select the environment again before restarting.")
                }
                let observed = try await observation(control, run: run, after: lastEventSequence)
                guard image.metadata.observedNanos <= observed.cutoffNanos, image.metadata.eventNanos <= image.metadata.observedNanos,
                      previousCutoff.map({ observed.cutoffNanos >= $0 + UInt64(details.periodMS) * 1_000_000 }) ?? true else {
                    throw AstraError("inference.causality", "Capture and input timestamps do not form a valid causal observation.")
                }
                for packet in pending.values {
                    guard observed.cutoffNanos <= packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000 + 250_000_000 else {
                        throw AstraError("inference.receiptTimeout", "The control helper did not finish an admitted packet in time.")
                    }
                }
                let begin = MonotonicClock.now
                let observationID = UUID()
                let result = try await predict(actor, ring: ring, frame: image, run: run, episode: episode,
                    previousState: previousState, cutoff: observed.cutoffNanos, controls: observed.controlState,
                    events: observed.executedEvents, contexts: options.contextIDs, observationID: observationID)
                try requireSuccess(result); try checkRunning()
                guard let current = try inbox.read(), current.metadata.surface == initialSurface else {
                    throw AstraError("inference.geometryChanged", "The environment changed while the policy was deciding. Select it again before restarting.")
                }
                let packet = try validateResult(result.payload, checkpoint: checkpoint, run: run, episode: episode,
                    previousState: previousState, observationID: observationID, cutoff: observed.cutoffNanos,
                    sequence: UInt64(decisions), surface: initialSurface, details: details)
                let now = MonotonicClock.now
                lastLatencyMS = Double(now - begin) / 1_000_000
                maximumLatencyMS = max(maximumLatencyMS ?? 0, lastLatencyMS ?? 0)
                guard now < packet.executeAtNanos else {
                    throw AstraError("inference.deadline", "The local policy missed its \(details.leadMS) ms execution lead. Use a checkpoint trained with a longer lead, or stop other GPU work before restarting.")
                }
                guard pending.count < 32 else { throw AstraError("inference.receipts", "Too many action packets are awaiting execution receipts.") }
                pending[packet.id] = packet
                if let collectionSink {
                    let record = try result.payload.required("collectionRecord")
                    guard record.fields?["schemaVersion"] == .integer(1), record.fields?["checkpointID"]?.uuid == checkpoint.id,
                          record.fields?["policySignature"] == .string(checkpoint.policySignature), record.fields?["episodeID"]?.uuid == episode,
                          record.fields?["observationID"]?.uuid == observationID, record.fields?["previousStateID"]?.uuid == previousState,
                          record.fields?["nextStateID"]?.uuid == result.payload.fields?["stateID"]?.uuid,
                          try record.required("cutoffNanos").decode(UInt64.self) == observed.cutoffNanos,
                          try record.required("frameIDs").decode([UUID].self) == [image.metadata.id],
                          try record.required("contextIDs").decode([Int].self) == options.contextIDs,
                          try record.required("episodeStep").decode(UInt64.self) == UInt64(decisions),
                          record.fields?["recurrentReset"] == .bool(decisions == 0),
                          record.fields?["logProbability"]?.double == result.payload.fields?["logProbability"]?.double,
                          record.fields?["value"]?.double == result.payload.fields?["value"]?.double,
                          record.fields?["sampler"]?.fields?["rngStreamID"]?.uuid == ready.payload.fields?["rngStreamID"]?.uuid,
                          try record.required("sampler").required("drawIndex").decode(UInt64.self) == UInt64(decisions),
                          record.fields?["sampler"]?.fields?["kind"] == .string("categorical"),
                          record.fields?["sampler"]?.fields?["temperature"]?.double == 1,
                          record.fields?["sampler"]?.fields?["mixture"] == .string("none") else {
                        throw AstraError("inference.collectionIdentity", "The actor returned inconsistent collection evidence.")
                    }
                    try collectionSink.offer(.decision(result.payload))
                }
                let accepted = try await control.request("execute", .encode(packet), run, .seconds(2), false)
                guard accepted.kind == "ack", accepted.runID == run, accepted.payload.fields?["admitted"] == .bool(true) else { throw AstraError("inference.admission", "The control helper did not admit the action packet.") }
                decisions += 1; previousState = try result.payload.requiredUUID("stateID")
                previousCutoff = observed.cutoffNanos; lastEventSequence = observed.lastSequence
                let addition = observed.cutoffNanos.addingReportingOverflow(UInt64(details.periodMS) * 1_000_000)
                guard !addition.overflow else { throw AstraError("inference.clock", "The decision clock is exhausted.") }
                nextDecision = addition.partialValue
            }
        } catch is CancellationError { /* Stop already records any originating failure. */ }
        catch { if !stopRequested { requestStop(reason: error.localizedDescription) } }
        await finish(run: run)
        if let message = collectionFault?.message {
            failure = [failure, "Collection lost control evidence: \(message)"].compactMap { $0 }.joined(separator: "\n")
        }
        var summary: JSONValue = .object(["runID": .string(run.uuidString.lowercased()), "status": .string(failure == nil ? "stopped" : "failed"),
            "decisions": .integer(Int64(decisions)), "executedPackets": .integer(Int64(executedPackets)),
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

    private func waitForFrame(_ inbox: InferenceFrameInbox) async throws -> InferenceImage {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            try checkRunning()
            if let image = try inbox.read() {
                _ = try image.metadata.validated()
                guard image.metadata.eventNanos <= image.metadata.observedNanos,
                      image.metadata.observedNanos <= MonotonicClock.now else { throw AstraError("inference.causality", "Capture returned an invalid image timestamp.") }
                return image
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw AstraError("inference.captureTimeout", "The selected environment did not produce a frame in time.")
    }

    private func reset(_ actor: InferenceRuntime, run: UUID, episode: UUID, contexts: [Int], seed: Int? = nil) async throws -> UUID {
        var payload: [String: JSONValue] = ["confirmed": .bool(true), "episodeID": .string(episode.uuidString.lowercased()),
            "contextIDs": .array(contexts.map { .integer(Int64($0)) })]
        if let seed { payload["seed"] = .integer(Int64(seed)) }
        let response = try await actor.request("inference.reset", .object(payload), run, .seconds(20), false)
        guard response.kind == "ack", response.runID == run, response.payload.fields?["runID"]?.uuid == run, response.payload.fields?["episodeID"]?.uuid == episode,
              response.payload.fields?["needsReset"] == .bool(false) else { throw AstraError("inference.reset", "The actor did not confirm a fresh recurrent episode.") }
        return try response.payload.requiredUUID("stateID")
    }

    private func predict(_ actor: InferenceRuntime, ring: SharedFrameRing, frame: InferenceImage, run: UUID, episode: UUID,
                         previousState: UUID, cutoff: UInt64, controls: ControlState, events: [RawInputEvent], contexts: [Int],
                         observationID: UUID = UUID(), warmup: Bool = false) async throws -> WireMessage {
        try checkRunning()
        let collecting = collectionSink != nil && !warmup
        if let coverage = frame.coverage { try coverage.validated(frame: frame.metadata, cutoffNanos: cutoff, maximumAgeMS: 250) }
        else {
            guard frame.metadata.eventNanos <= frame.metadata.observedNanos, frame.metadata.observedNanos <= cutoff,
                  cutoff - frame.metadata.eventNanos <= 250_000_000 else {
                throw AstraError("capture.stale", "The actor observation has no recent source coverage.")
            }
        }
        let (reference, collectedPixels) = try await Task.detached {
            let pixels = try frame.pixels()
            return (try ring.publish(pixels: pixels, metadata: frame.metadata), collecting ? pixels : nil)
        }.value
        try checkRunning()
        let payload: JSONValue = .object(["observationID": .string(observationID.uuidString.lowercased()), "episodeID": .string(episode.uuidString.lowercased()),
            "previousStateID": .string(previousState.uuidString.lowercased()), "cutoffNanos": .unsigned(cutoff),
            "geometryRevision": .unsigned(frame.metadata.surface.geometryRevision), "frames": .array([try .encode(reference)]),
            "controlState": try .encode(controls), "executedEvents": try .encode(events), "intervalCovered": .bool(true),
            "contextIDs": .array(contexts.map { .integer(Int64($0)) })])
        if let collectedPixels, let collectionSink, var input = payload.fields {
            input.removeValue(forKey: "frames")
            try collectionSink.offer(.observation(.init(runID: run, actorInput: .object(input), frame: frame.metadata,
                pixels: collectedPixels, coverage: frame.coverage)))
        }
        let response = try await actor.request(warmup ? "inference.warmup" : "inference.step", payload, run, .seconds(120), true)
        guard response.runID == run else { throw AstraError("inference.run", "The actor replied for another run.") }
        let released = try response.payload.required("releasedFrames").decode([SharedFrameAcknowledgement].self)
        guard released == [reference.acknowledgement] || (response.kind == "error" && released.isEmpty) else {
            throw AstraError("inference.frameAcknowledgement", "The actor acknowledged a different or repeated image lease.")
        }
        for acknowledgement in released { try ring.release(acknowledgement) }
        return response
    }

    private struct ControlObservation: Decodable {
        let controlState: ControlState
        let executedEvents: [RawInputEvent]
        let intervalCovered: Bool
        let cutoffNanos: UInt64
        let lastSequence: UInt64?
    }
    private func observation(_ control: InferenceRuntime, run: UUID, after: UInt64?) async throws -> ControlObservation {
        let deadline = MonotonicClock.now + 20_000_000
        while true {
            let response = try await control.request("observation", .object(after.map { ["afterSequence": .unsigned($0)] } ?? [:]), run, .seconds(2), false)
            guard response.kind == "ack", response.runID == run else { throw AstraError("inference.run", "The control observation belongs to another run.") }
            let value = try response.payload.decode(ControlObservation.self)
            guard value.intervalCovered, value.cutoffNanos <= MonotonicClock.now, value.controlState.observedNanos <= value.cutoffNanos, value.controlState.pointer.isFinite,
                  value.executedEvents.count <= 2048 else { throw AstraError("inference.inputCoverage", "Executed input history is unavailable or incomplete.") }
            var previous = after
            for event in value.executedEvents {
                guard previous.map({ event.sequence > $0 }) ?? true, event.eventNanos <= event.observedNanos,
                      event.observedNanos <= value.cutoffNanos else { throw AstraError("inference.inputCausality", "Executed inputs have invalid sequence or availability timestamps.") }
                previous = event.sequence
            }
            guard previous == value.lastSequence else { throw AstraError("inference.inputCursor", "Executed input history does not match its cursor.") }
            if value.controlState.valid { return value }
            try checkRunning()
            guard MonotonicClock.now < deadline else { throw AstraError("inference.inputBusy", "The control helper could not provide a settled input observation in time.") }
            try await Task.sleep(for: .milliseconds(1))
        }
    }

    private func validateHello(_ message: WireMessage, role: String) throws {
        guard message.kind == "hello", message.version == AstraVersion.protocolVersion,
              message.payload.fields?["role"]?.text == role,
              message.payload.fields?["protocolVersion"]?.int == AstraVersion.protocolVersion else {
            throw AstraError("inference.runtime", "The local \(role) runtime reported an incompatible identity.")
        }
        if role == "control", dependencies.protectsPhysicalInputs, message.payload.fields?["recoveryVersion"] != .integer(1) {
            throw AstraError("inference.recoveryVersion", "This control helper does not support the required independent cleanup protection. Rebuild or reinstall the complete application.")
        }
    }

    private func requireSuccess(_ response: WireMessage) throws {
        guard response.kind == "ack" else {
            throw AstraError(response.payload.fields?["code"]?.text ?? "inference.actor", response.payload.fields?["message"]?.text ?? "The local actor rejected its observation.")
        }
    }

    private func validateResult(_ value: JSONValue, checkpoint: CheckpointDocument, run: UUID, episode: UUID, previousState: UUID,
                                observationID: UUID, cutoff: UInt64, sequence: UInt64, surface: SurfaceDescriptor,
                                details: InferencePolicyDetails) throws -> ActionPacket {
        let packet = try value.required("packet").decode(ActionPacket.self)
        guard value.fields?["runID"]?.uuid == run, value.fields?["checkpointID"]?.uuid == checkpoint.id,
              value.fields?["policySignature"]?.text == checkpoint.policySignature, value.fields?["episodeID"]?.uuid == episode,
              value.fields?["stateID"]?.uuid != nil, value.fields?["stateID"]?.uuid != previousState, value.fields?["needsReset"] == .bool(false),
              ["logProbability", "conditionalEntropy", "value"].allSatisfy({ value.fields?[$0]?.double?.isFinite == true }),
              try value.required("surfaces").decode([SurfaceDescriptor].self) == [surface], packet.runID == run,
              packet.observationID == observationID, packet.sequence == sequence, packet.geometryRevision == surface.geometryRevision,
              !cutoff.addingReportingOverflow(UInt64(details.leadMS) * 1_000_000).overflow,
              packet.executeAtNanos == cutoff.addingReportingOverflow(UInt64(details.leadMS) * 1_000_000).partialValue, packet.durationMs == details.periodMS else {
            throw AstraError("inference.resultIdentity", "The actor returned an invalid policy, recurrent state, observation or action identity.")
        }
        return try packet.validated(capabilities: details.capabilities, surfaces: [surface], capacity: details.capacity)
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
                guard let packet = pending[receipt.packetID], packet.runID == receipt.runID, packet.sequence == receipt.sequence,
                      receipt.resultingState.pointer.isFinite else { throw AstraError("inference.receipt", "The control helper returned an unknown or inconsistent packet receipt.") }
                if receipt.status == .admitted { return }
                // Removing the identity also rejects duplicate terminal replies.
                pending.removeValue(forKey: receipt.packetID)
                guard receipt.status == .executed, receipt.commandResults.count == packet.commands.count,
                      Set(receipt.commandResults.map(\.commandIndex)) == Set(packet.commands.indices),
                      receipt.commandResults.allSatisfy({ result in
                          guard packet.commands.indices.contains(result.commandIndex) else { return false }
                          let scheduled = packet.executeAtNanos + UInt64(packet.commands[result.commandIndex].offsetMs) * 1_000_000
                          return result.scheduledNanos == scheduled && (result.status == .posted || result.status == .noOp)
                              && (result.status != .posted || result.postedNanos != nil)
                              && (result.postedNanos.map { $0 >= scheduled && $0 <= receipt.observedNanos } ?? true)
                      }) else {
                    throw AstraError("inference.execution", "An action was late, cancelled or could not be posted.")
                }
                executedPackets += 1
            } catch { requestStop(reason: error.localizedDescription) }
        default: requestStop(reason: "The control helper returned an unsupported event.")
        }
    }

    private func sleep(until nanos: UInt64) async throws {
        let now = MonotonicClock.now
        if nanos > now { try await Task.sleep(for: .nanoseconds(Int64(min(nanos - now, UInt64(Int64.max))))) }
    }

    private func finish(run: UUID) async {
        stopRequested = true; isStopping = true; countdown = nil
        heartbeat?.cancel(); await heartbeat?.value; heartbeat = nil
        await urgentStop?.value; urgentStop = nil
        // Control teardown begins before capture/actor waits. Joined helper
        // shutdown is the ownership boundary; all late results are ignored.
        if let control {
            _ = try? await control.request("disarm", .object([:]), run, .seconds(2), false)
            phase = "Waiting for owned controls to release…"
            let status = await control.shutdown()
            if let recovery = controlRecovery {
                cleanupConfirmed = await confirmRecoveryAfterExecutorExit(recovery)
            } else { cleanupConfirmed = !controlAdmissionAttempted || status == 0 }
        } else {
            cleanupConfirmed = !controlAdmissionAttempted
        }
        if !cleanupConfirmed {
            let message = "The control helper exited before input cleanup could be confirmed. Release any held controls manually before starting another run."
            failure = [failure, message].compactMap { $0 }.joined(separator: "\n")
        }
        self.control = nil
        inbox?.close()
        await capture?.stop(); capture = nil; inbox = nil
        _ = await actor?.shutdown(); actor = nil
        ring?.closeAfterConsumerExit(); ring = nil
        pending = [:]
    }

    private func confirmRecoveryAfterExecutorExit(_ ledger: ControlRecoveryLedger) async -> Bool {
        while true {
            do {
                let snapshot = try ledger.snapshot()
                if snapshot.cleanupConfirmed { cleanupRecoveredByGuardian = snapshot.recoveredByGuardian; return true }
                // Native posting requires the sticky armed marker before any
                // reservation. A command process dying before that point never
                // gained input authority, even when the arm request was in flight.
                if !snapshot.everArmed, snapshot.possibleKeys.isEmpty, snapshot.possibleButtons.isEmpty, snapshot.inFlight == 0 { return true }
                guard ledger.matchingGuardianIsAlive(snapshot) else { return false }
                phase = snapshot.errorCode == 0 ? "Recovering owned controls after the executor stopped…"
                    : "Waiting for the cleanup guardian to release owned controls…"
                try? await Task.sleep(for: .milliseconds(25))
            } catch { return false }
        }
    }
}

extension JSONValue {
    var uuid: UUID? { text.flatMap(UUID.init(uuidString:)) }
    func requiredUUID(_ key: String) throws -> UUID {
        guard let value = fields?[key]?.uuid else { throw AstraError("inference.identity", "The runtime did not provide a valid \(key).") }
        return value
    }
}
