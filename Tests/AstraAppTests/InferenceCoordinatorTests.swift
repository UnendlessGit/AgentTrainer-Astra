import Foundation
import Testing
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

/// The production coordinator and mapped ring, with no GPU, event posting or
/// privacy access. These doubles only implement the documented process boundary.
private actor InferenceHarness {
    let mode: String
    let checkpoint: CheckpointDocument
    var requests: [String] = []
    var captureStopped = false
    var actorExited = false
    var controlExited = false
    var allClosed: Bool { actorExited && controlExited && captureStopped }
    var actorCaptureClosed: Bool { actorExited && captureStopped }
    var processesClosed: Bool { actorExited && controlExited }
    var armWaiting = false
    var stepWaiting = false
    var actorExitWaiting = false
    var guardianRecoveryWaiting = false
    var recoveryLedger: ControlRecoveryLedger?
    var guardianProcess: Process?
    var guardianInput: Pipe?
    var controlArmed = false
    var controlRun: UUID?
    var admissionWaiting = false
    var suspendedAdmission: (ActionPacket, UUID, CheckedContinuation<WireMessage, any Error>)?
    var pendingControlPackets: [(ActionPacket, UUID)] = []
    var oldControlCallbacks: [(UUID, ComputeProcess.EventHandler)] = []
    var ringPath: String?
    var references: [SharedFrameReference] = []
    var sequence: UInt64 = 0
    var resetGeneration: UInt64 = 0
    var rng: [UInt32] = [0, 0]
    var preparationSeed: UInt64?
    var resetSeeds: [JSONValue] = []
    var resetTimeouts: [Duration] = []
    var sampledResponse: WireMessage?
    var stateID = UUID()
    var episodeID = UUID()
    let rngStreamID = UUID()
    var collecting = false
    var events: ComputeProcess.EventHandler?
    var failure: ComputeProcess.FailureHandler?
    var frameCallback: (@Sendable (InferenceImage) -> Void)?
    var healthCallback: (@Sendable (CaptureHealth) -> Void)?
    var suspendedArm: CheckedContinuation<WireMessage, any Error>?
    var suspendedStep: CheckedContinuation<WireMessage, any Error>?
    var suspendedExit: CheckedContinuation<Void, Never>?
    var currentRun: UUID?
    var surface = SurfaceDescriptor(id: "surface", globalBounds: .init(x: 0, y: 0, width: 100, height: 100),
                                    pixelWidth: 32, pixelHeight: 32, contentBounds: .init(x: 0, y: 0, width: 32, height: 32), geometryRevision: 0)

    init(mode: String, checkpoint: CheckpointDocument) { self.mode = mode; self.checkpoint = checkpoint }
    nonisolated func dependencies() -> InferenceDependencies {
        .init(runtime: { role, events, failure in
            InferenceRuntime(start: { await self.start(role, events: events, failure: failure) },
                             request: { kind, payload, run, timeout, _ in try await self.request(role, kind, payload, run, timeout: timeout) },
                             shutdown: { await self.shutdown(role) })
        }, capture: { _ in
            InferenceCapture(start: { frames, health in await self.capture(frames, health: health) }, stop: { await self.stopCapture() })
        }, activate: { _ in }, countdownSeconds: 0)
    }
    func capture(_ callback: @escaping @Sendable (InferenceImage) -> Void, health: @escaping @Sendable (CaptureHealth) -> Void) {
        frameCallback = callback; healthCallback = health; captureStopped = false; publishFrame()
    }
    func publishFrame() {
        let now = MonotonicClock.now
        let metadata = FrameMetadata(eventNanos: now, observedNanos: now, surface: surface,
                                     byteCount: surface.pixelWidth * surface.pixelHeight * 4, codec: "raw")
        let size = metadata.byteCount
        frameCallback?(InferenceImage(metadata: metadata, pixels: { Data(repeating: 255, count: size) }))
    }
    func stopCapture() { captureStopped = true }
    func start(_ role: String, events: @escaping ComputeProcess.EventHandler, failure: @escaping ComputeProcess.FailureHandler) -> WireMessage {
        requests.append(role + ".start")
        if role == "control" {
            if let previous = self.events, let run = controlRun { oldControlCallbacks.append((run, previous)) }
            self.events = events; controlExited = false
        } else { self.failure = failure; actorExited = false }
        var payload: [String: JSONValue] = ["role": .string(mode == "wrongRuntime" && role == "actor" ? "compute" : role), "protocolVersion": .integer(1)]
        if role == "control" { payload["initialPacketSequenceVersion"] = .integer(1) }
        if role == "control", mode == "protectedGuardianRecovery" { payload["recoveryVersion"] = .integer(1) }
        return WireMessage(kind: "hello", sequence: 0, payload: .object(payload))
    }
    func ack(_ payload: JSONValue = .object([:]), run: UUID, requestID: UUID = UUID()) -> WireMessage {
        WireMessage(kind: "ack", sequence: 1, requestID: requestID, runID: run, payload: payload)
    }
    func request(_ role: String, _ kind: String, _ payload: JSONValue, _ run: UUID, timeout: Duration) async throws -> WireMessage {
        requests.append(kind); currentRun = run
        let requestID = UUID()
        switch kind {
        case "inference.prepare":
            sequence = 0; resetGeneration = 0; preparationSeed = payload.fields?["seed"]?.uint64
            rng = [0, UInt32(preparationSeed ?? 0)]
            collecting = payload.fields?["collection"] == .bool(true)
            ringPath = payload.fields?["ring"]?.fields?["path"]?.text
            return ack(.object(["runID": .string(run.uuidString), "checkpointID": .string(checkpoint.id.uuidString),
                "policySignature": .string(mode == "wrongPolicy" ? String(repeating: "b", count: 64) : checkpoint.policySignature),
                "ringID": payload.fields?["ring"]?.fields?["ringID"] ?? .null,
                "model": .object(["schema_version": .integer(2), "period_ms": .integer(100), "lead_ms": .integer(200),
                                   "packet_capacity": .integer(16), "context_sizes": .array([])]),
                "actions": try .encode(ActionCapabilities(keyCodes: [0])),
                "collection": .bool(collecting && mode != "oldCollection"), "collectionVersion": .integer(1),
                "rngStreamID": .string(rngStreamID.uuidString), "deterministic": payload.fields?["deterministic"] ?? .bool(false),
                "needsReset": .bool(true), "resumedActor": .bool(false), "nextPacketSequence": .unsigned(sequence),
                "nextDrawIndex": .unsigned(sequence), "actorResetGeneration": .unsigned(resetGeneration)]), run: run)
        case "inference.reset":
            if collecting, payload.fields?["seed"] != nil { throw AstraError("fixture.reseed", "Collecting resets cannot reseed") }
            resetTimeouts.append(timeout)
            if let seed = payload.fields?["seed"] { resetSeeds.append(seed) }
            resetGeneration += 1
            episodeID = try payload.required("episodeID").decode(UUID.self); stateID = UUID()
            return ack(.object(["runID": .string(run.uuidString), "episodeID": .string(episodeID.uuidString),
                "checkpointID": .string(checkpoint.id.uuidString), "policySignature": .string(checkpoint.policySignature),
                "stateID": .string(stateID.uuidString), "needsReset": .bool(false), "nextPacketSequence": .unsigned(sequence),
                "nextDrawIndex": .unsigned(sequence), "actorResetGeneration": .unsigned(resetGeneration)]), run: run)
        case "inference.warmup", "inference.step":
            let warming = kind == "inference.warmup"
            let reference = try payload.required("frames").decode([SharedFrameReference].self)[0]
            references.append(reference)
            if mode == "slowWarmup", warming { try await Task.sleep(for: .milliseconds(115)) }
            if mode == "lateAction", !warming { try await Task.sleep(for: .milliseconds(220)) }
            if mode == "captureStops", !warming { healthCallback?(.stopped) }
            if mode == "geometryDuringStep", !warming {
                surface.geometryRevision += 1; surface.globalBounds.x += 1; publishFrame()
            }
            if (mode == "blockedActor" || mode == "workerCrash"), !warming {
                stepWaiting = true
                if mode == "workerCrash" { failure?(AstraError("compute.exited", "Fixture actor exited unexpectedly.")) }
                return try await withCheckedThrowingContinuation { suspendedStep = $0 }
            }
            let cutoff = try payload.required("cutoffNanos").decode(UInt64.self)
            let packet = ActionPacket(runID: mode == "wrongPacket" && !warming ? UUID() : run,
                sequence: sequence, observationID: try payload.required("observationID").decode(UUID.self),
                geometryRevision: reference.metadata.surface.geometryRevision, executeAtNanos: cutoff + 200_000_000,
                durationMs: 100, commands: [TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)])
            let nextState = UUID()
            var fields: [String: JSONValue] = ["runID": .string(run.uuidString), "checkpointID": .string(checkpoint.id.uuidString),
                "policySignature": .string(checkpoint.policySignature), "episodeID": .string(episodeID.uuidString),
                "stateID": .string(mode == "staleState" && !warming ? stateID.uuidString : nextState.uuidString), "needsReset": .bool(false),
                "surfaces": try .encode([reference.metadata.surface]), "packet": try .encode(packet),
                "logProbability": .number(-0.5), "conditionalEntropy": .number(0.5), "value": .number(0),
                "releasedFrames": try .encode([reference.acknowledgement])]
            if mode == "wrongLease", !warming {
                let wrong = SharedFrameAcknowledgement(runID: run, ringID: reference.ringID, slot: reference.slot,
                                                      leaseID: UUID(), sequence: reference.sequence)
                fields["releasedFrames"] = try .encode([wrong])
            }
            if collecting && !warming && mode != "missingCollection" {
                let after = rng.map { $0 &+ 1 }
                let sampler: [String: JSONValue] = ["kind": .string("categorical"), "temperature": .integer(1), "mixture": .string("none"),
                    "rngStreamID": .string(rngStreamID.uuidString), "drawIndex": .unsigned(sequence), "version": .integer(1),
                    "stateBefore": try .encode(rng), "stateAfter": try .encode(after), "sampleKey": try .encode([UInt32(17), 23])]
                var record: [String: JSONValue] = ["schemaVersion": .integer(1),
                    "checkpointID": .string(checkpoint.id.uuidString), "policySignature": .string(checkpoint.policySignature),
                    "episodeID": .string(episodeID.uuidString), "sampler": .object(sampler)]
                record["observationID"] = payload.fields?["observationID"]
                record["previousStateID"] = payload.fields?["previousStateID"]
                record["nextStateID"] = fields["stateID"]
                record["cutoffNanos"] = .unsigned(cutoff)
                record["frameIDs"] = try .encode([reference.metadata.id])
                record["geometryRevision"] = try payload.required("geometryRevision")
                record["environmentResets"] = .unsigned(resetGeneration)
                record["controlCoverageNanos"] = payload.fields?["controlCoverageNanos"]
                record["contextIDs"] = .array([])
                record["episodeStep"] = .unsigned(sequence)
                record["recurrentReset"] = .bool(sequence == 0)
                record["logProbability"] = fields["logProbability"]
                record["value"] = fields["value"]
                fields["collectionRecord"] = .object(record)
                rng = after
            }
            if warming { fields["warmup"] = .bool(true) }
            else { sequence += 1; stateID = nextState }
            let response = ack(.object(fields), run: run)
            if mode == "sampledDuringStop", !warming {
                sampledResponse = response; stepWaiting = true
                return try await withCheckedThrowingContinuation { suspendedStep = $0 }
            }
            return response
        case "arm":
            controlRun = run
            if ["armTakeoverEvent", "armEmergencyEvent", "armTakeoverError", "armEmergencyError"].contains(mode) {
                let cause = mode.contains("Emergency") ? "emergencyStop" : "physicalTakeover"
                if mode.hasSuffix("Event") {
                    events?(WireMessage(kind: "control.stopped", sequence: 1, runID: run,
                        payload: .object(["cause": .string(cause), "reason": .string("Pre-arm intervention.")])))
                }
                return WireMessage(kind: "error", sequence: 2, requestID: requestID, runID: run,
                    payload: .object(["code": .string(mode.hasSuffix("Event") ? "control.cancelled" : "control." + cause),
                                      "message": .string("Pre-arm intervention."), "recoverable": .bool(true)]))
            }
            if mode == "protectedGuardianRecovery" {
                let request = try payload.decode(ArmRequest.self)
                let ledger = try ControlRecoveryLedger(open: #require(request.recovery))
                let guardian = Process(), input = Pipe()
                guardian.executableURL = URL(fileURLWithPath: "/bin/cat")
                guardian.standardInput = input; guardian.standardOutput = FileHandle.nullDevice; guardian.standardError = FileHandle.nullDevice
                try guardian.run(); guardianProcess = guardian; guardianInput = input
                guard ledger.registerExecutor(pid: getpid()), ledger.registerGuardian(pid: guardian.processIdentifier, now: MonotonicClock.now), ledger.arm() else {
                    throw AstraError("fixture.recovery", "The virtual recovery interface did not arm.")
                }
                recoveryLedger = ledger
            }
            if mode == "blockedArm" {
                armWaiting = true
                return try await withCheckedThrowingContinuation { suspendedArm = $0 }
            }
            controlArmed = true
            if let recoveryLedger, let guardianProcess {
                return ack(.object(["armed": .bool(true), "nextPacketSequence": .integer(0), "recoveryLedgerID": .string(recoveryLedger.descriptor.ledgerID.uuidString), "guardianPID": .integer(Int64(guardianProcess.processIdentifier))]), run: run, requestID: requestID)
            }
            return ack(.object(["armed": .bool(true), "nextPacketSequence": .integer(0)]), run: run, requestID: requestID)
        case "disarm":
            controlArmed = false
            if mode != "receiptOnShutdown" {
                for (packet, id) in pendingControlPackets { try emitReceipt(packet, requestID: id, status: .cancelled) }
                pendingControlPackets = []
            }
            return ack(.object(["stopped": .bool(true), "cleanupSettled": .bool(mode != "protectedGuardianRecovery")]), run: run, requestID: requestID)
        case "observation":
            let now = MonotonicClock.now
            var controls = ControlState(); controls.valid = true; controls.observedNanos = now
            return ack(.object(["controlState": try .encode(controls), "executedEvents": .array([]),
                               "intervalCovered": .bool(mode != "inputGap"), "cutoffNanos": .unsigned(now)]),
                       run: mode == "wrongControlRun" ? UUID() : run)
        case "execute":
            let packet = try payload.decode(ActionPacket.self)
            if mode == "blockedAdmission" {
                admissionWaiting = true
                return try await withCheckedThrowingContinuation { suspendedAdmission = (packet, requestID, $0) }
            }
            if let recoveryLedger {
                guard recoveryLedger.beginPost(operation: .keyDown, keyCode: 0, button: nil) else { throw AstraError("fixture.recovery", "The virtual reservation was rejected.") }
                recoveryLedger.endPost(operation: .keyDown, keyCode: 0, button: nil, success: true)
            }
            if mode == "delayedReceipt" || mode == "receiptOnShutdown" {
                pendingControlPackets.append((packet, requestID))
                try emitReceipt(packet, requestID: requestID, status: .admitted)
            } else { try emitReceipt(packet, requestID: requestID, status: .executed) }
            if mode == "takeover" || mode == "emergency" {
                events?(WireMessage(kind: "control.stopped", sequence: 2, runID: run,
                    payload: .object(["cause": .string(mode == "takeover" ? "physicalTakeover" : "emergencyStop"),
                                      "reason": .string(mode == "takeover" ? "Physical input took over." : "Emergency stop pressed.")])))
            }
            return ack(.object(["admitted": .bool(true)]), run: run, requestID: requestID)
        default: return ack(run: run)
        }
    }
    private func emitReceipt(_ packet: ActionPacket, requestID: UUID, status: ReceiptStatus) throws {
        var controls = ControlState(); controls.valid = true
        let results: [CommandResult] = status == .executed ? packet.commands.enumerated().map { index, command in
            CommandResult(commandIndex: index, scheduledNanos: packet.executeAtNanos + UInt64(command.offsetMs) * 1_000_000,
                          postedNanos: packet.executeAtNanos, status: .posted)
        } : []
        var receipt = ExecutionReceipt(packet: packet, status: status, observedNanos: packet.executeAtNanos + 100_000_000,
            commandResults: mode == "partialReceipt" ? [] : results, resultingState: controls)
        if mode == "wrongReceipt" { receipt.sequence += 1 }
        events?(WireMessage(kind: "control.receipt", sequence: 1, requestID: requestID, runID: packet.runID,
                            payload: .object(["receipt": try .encode(receipt)])))
    }
    func completePendingReceipts() throws {
        for (packet, id) in pendingControlPackets { try emitReceipt(packet, requestID: id, status: .executed) }
        pendingControlPackets = []
    }
    func emitOldControlFaults() {
        for (run, callback) in oldControlCallbacks {
            callback(WireMessage(kind: "control.stopped", sequence: 9, runID: run,
                payload: .object(["cause": .string("fault"), "reason": .string("Old control callback escaped") ])))
        }
    }
    func rejectSuspendedAdmission() throws {
        guard let (packet, request, continuation) = suspendedAdmission else { return }
        suspendedAdmission = nil
        #expect(!controlArmed)
        try emitReceipt(packet, requestID: request, status: .rejected)
        continuation.resume(returning: WireMessage(kind: "error", sequence: 2, requestID: request, runID: packet.runID,
            payload: .object(["code": .string("control.session"), "message": .string("The helper is disarmed."), "recoverable": .bool(true)])))
    }
    func resumeArm() {
        if let suspendedArm, let currentRun { self.suspendedArm = nil; controlArmed = true; suspendedArm.resume(returning: ack(.object(["armed": .bool(true), "nextPacketSequence": .integer(0)]), run: currentRun)) }
    }
    func resumeExit() { suspendedExit?.resume(); suspendedExit = nil }
    func settleGuardianRecovery() {
        recoveryLedger?.releaseKey(0); recoveryLedger?.settleGuardian(now: MonotonicClock.now)
    }
    func exitGuardian() async {
        try? guardianInput?.fileHandleForWriting.close()
        while guardianProcess?.isRunning == true { try? await Task.sleep(for: .milliseconds(5)) }
        guardianInput = nil; guardianProcess = nil
    }
    func shutdown(_ role: String) async -> Int32? {
        requests.append(role + ".shutdown")
        if role == "actor" {
            if let sampledResponse { suspendedStep?.resume(returning: sampledResponse) }
            else { suspendedStep?.resume(throwing: AstraError("compute.closed", "Fixture actor stopped.")) }
            suspendedStep = nil
            if mode == "blockedActor", !actorExited {
                actorExitWaiting = true
                await withCheckedContinuation { suspendedExit = $0 }
            }
            actorExited = true
        } else {
            if mode == "receiptOnShutdown" { try? completePendingReceipts() }
            if collecting, let currentRun {
                events?(WireMessage(kind: "control.stopped", sequence: 3, runID: currentRun,
                                    payload: .object(["cause": .string("shutdown"), "reason": .string("Fixture finalized control") ])))
            }
            controlExited = true
            if mode == "protectedGuardianRecovery", let recoveryLedger {
                recoveryLedger.stop()
                _ = recoveryLedger.claimAfterExecutorExit() // Virtual process boundary only; real death watches are tested in GuardianTests.
                guardianRecoveryWaiting = true
                return 15
            }
        }
        return role == "control" && mode == "controlCrashOnShutdown" ? 15 : 0
    }
}

private final class CollectionSpy: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [InferenceCollectionEvent] = []
    private var closed = false
    private var seenLateEvent = false
    var rejectObservations = false
    var failFinish = false
    var rejectShutdownEvidence = false
    var rejectExecutedEvidence = false
    var events: [InferenceCollectionEvent] { lock.withLock { values } }
    var completed: Bool { lock.withLock { closed } }
    var lateEvent: Bool { lock.withLock { seenLateEvent } }
    func sink() -> InferenceCollectionSink {
        .init(offer: { [self] event in
            try lock.withLock {
                if closed { seenLateEvent = true }
                if rejectObservations, case .observation = event { throw AstraError("collector.full", "Fixture collector capacity is full") }
                if rejectExecutedEvidence, case .control(let message) = event,
                   let payload = message.payload.fields?["receipt"], let receipt = try? payload.decode(ExecutionReceipt.self), receipt.status == .executed {
                    throw AstraError("collector.full", "Fixture could not retain terminal execution evidence")
                }
                if rejectShutdownEvidence, case .control(let message) = event, message.kind == "control.stopped" {
                    throw AstraError("collector.closed", "Fixture collector could not retain final control evidence")
                }
                values.append(event)
            }
        }, finish: { [self] _ in
            try lock.withLock {
                closed = true
                if failFinish { throw AstraError("collector.finish", "Fixture collector could not persist its audit") }
            }
        })
    }
}

@MainActor private final class InferenceTestCase {
    let root: URL
    let agent: AgentDocument
    let checkpoint: CheckpointDocument
    let harness: InferenceHarness
    let coordinator: InferenceCoordinator
    let controlOwner: NativeControlOwner
    let source = CaptureSource(id: "display:1", name: "Fixture display", kind: .display, displayID: 1,
                               bounds: .init(x: 0, y: 0, width: 100, height: 100), pixelWidth: 32, pixelHeight: 32)
    init(mode: String) async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraInference-" + UUID().uuidString)
        agent = AgentDocument(name: "Inference fixture")
        checkpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: nil, name: "Checkpoint", kind: "behavioral",
                                        trainingStep: 1, policySignature: String(repeating: "a", count: 64), parameterCount: 1)
        let store = try LibraryStore(root: root)
        try await store.save(agent); try await store.saveCheckpoint(checkpoint)
        harness = InferenceHarness(mode: mode, checkpoint: checkpoint)
        var dependencies = harness.dependencies()
        dependencies.protectsPhysicalInputs = mode == "protectedGuardianRecovery" || mode == "oldControlProtection"
        controlOwner = dependencies.controlOwner
        coordinator = InferenceCoordinator(store: store, root: root, dependencies: dependencies)
    }
    deinit { try? FileManager.default.removeItem(at: root) }
    func start() throws { try coordinator.start(agent: agent, checkpoint: checkpoint, source: source, options: InferenceOptions()) }
    func wait(_ condition: @MainActor () async -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !(await condition()) {
            guard ContinuousClock.now < deadline else { throw AstraError("test.timeout", "Inference fixture did not reach its expected state. \(coordinator.failure ?? coordinator.phase) Requests: \(await harness.requests)") }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
}

@MainActor struct InferenceCoordinatorTests {
    @Test func collectionOwnsExactPixelsAndReceivesShutdownEvidenceBeforeFinalization() async throws {
        let test = try await InferenceTestCase(mode: "success"), spy = CollectionSpy()
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { test.coordinator.executedPackets >= 2 }
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.failure == nil && test.coordinator.cleanupConfirmed && spy.completed && !spy.lateEvent)
        let events = spy.events
        #expect(events.filter { if case .prepared = $0 { true } else { false } }.count == 1)
        let observations = events.compactMap { if case .observation(let value) = $0 { value } else { nil } }
        let decisions = events.compactMap { if case .decision(let value) = $0 { value } else { nil } }
        #expect(decisions.count >= 2 && observations.count >= decisions.count)
        for value in observations {
            #expect(value.pixels == Data(repeating: 255, count: 32 * 32 * 4))
            #expect(value.actorInput.fields?["frames"] == nil)
            #expect(value.actorInput.fields?["observationID"]?.text.flatMap(UUID.init(uuidString:)) != nil)
        }
        #expect(events.contains { event in
            if case .control(let message) = event { return message.kind == "control.stopped" && message.payload.fields?["cause"] == .string("shutdown") }
            return false
        })
        #expect(await test.harness.allClosed)
        #expect(!FileManager.default.fileExists(atPath: try #require(await test.harness.ringPath)))
    }

    @Test(arguments: ["full", "missingCollection", "oldCollection"])
    func collectionAdmissionFailuresStopBeforePosting(_ failure: String) async throws {
        let test = try await InferenceTestCase(mode: failure), spy = CollectionSpy()
        spy.rejectObservations = failure == "full"
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure != nil && test.coordinator.cleanupConfirmed && spy.completed)
        #expect(!(await test.harness.requests).contains("execute"))
    }

    @Test func collectionFinalizationFailureIsSavedWithoutLosingControlCleanupProof() async throws {
        let test = try await InferenceTestCase(mode: "success"), spy = CollectionSpy()
        spy.failFinish = true
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.cleanupConfirmed && test.coordinator.failure?.contains("Collection could not be finalized") == true)
        let result = try await LearningFiles.read(#require(test.coordinator.resultsURL))
        #expect(result.fields?["status"] == .string("failed") && result.fields?["cleanupConfirmed"] == .bool(true))
    }

    @Test func shutdownOfferFailureCannotRaceASuccessfulCollectionSummary() async throws {
        let test = try await InferenceTestCase(mode: "success"), spy = CollectionSpy()
        spy.rejectShutdownEvidence = true
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        let result = try await LearningFiles.read(#require(test.coordinator.resultsURL))
        #expect(result.fields?["status"] == .string("failed"))
        #expect(result.fields?["issue"]?.text?.contains("Collection evidence is incomplete") == true)
        #expect(test.coordinator.cleanupConfirmed && spy.completed)
    }

    @Test func stoppingInFlightCollectionNeverClaimsVerifiedActorProgress() async throws {
        let test = try await InferenceTestCase(mode: "blockedActor"), spy = CollectionSpy()
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { await test.harness.stepWaiting }
        let stopping = Task { await test.coordinator.stopAndWait() }
        try await test.wait { await test.harness.actorExitWaiting }
        await test.harness.resumeExit(); await stopping.value
        let result = try await LearningFiles.read(#require(test.coordinator.resultsURL))
        #expect(result.fields?["status"] == .string("failed"))
        #expect(result.fields?["issue"]?.text?.contains("random-stream progress is unverified") == true)
        #expect(test.coordinator.cleanupConfirmed && spy.completed)
        #expect(await test.harness.allClosed)
    }

    @Test func stopAfterSamplingRequiresTheSinkToHaveRetainedTheResult() async throws {
        let test = try await InferenceTestCase(mode: "sampledDuringStop"), spy = CollectionSpy()
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { await test.harness.stepWaiting }
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.producedPackets == 1 && test.coordinator.decisions == 0 && test.coordinator.executedPackets == 0)
        #expect(test.coordinator.failure?.contains("random-stream progress is unverified") == true)
        #expect(!spy.events.contains { if case .decision = $0 { true } else { false } })
        #expect(test.coordinator.cleanupConfirmed && spy.completed && !spy.lateEvent)
    }

    @Test func successfulRunWarmsWithoutControlAndReleasesResources() async throws {
        let test = try await InferenceTestCase(mode: "success")
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 2 }
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.failure == nil && !test.coordinator.isBusy && test.coordinator.cleanupConfirmed)
        #expect(test.coordinator.warmupLatencyMS.count == 3)
        #expect(await test.harness.preparationSeed == 0)
        #expect(await test.harness.resetSeeds.isEmpty)
        #expect(await test.harness.resetTimeouts == [.seconds(20), .seconds(20)])
        let requests = await test.harness.requests
        #expect(requests.filter { $0 == "inference.prepare" }.count == 1)
        #expect(requests.filter { $0 == "inference.warmup" }.count == 3)
        #expect(try #require(requests.lastIndex(of: "inference.warmup")) < #require(requests.firstIndex(of: "arm")))
        let refs = await test.harness.references
        #expect(Set(refs.map(\.leaseID)).count == refs.count)
        #expect(await test.harness.allClosed)
        #expect(!FileManager.default.fileExists(atPath: try #require(await test.harness.ringPath)))
        #expect(test.coordinator.resultsURL.map { FileManager.default.fileExists(atPath: $0.path) } == true)
        let previousRun = test.coordinator.runID
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.failure == nil && test.coordinator.runID != previousRun)
    }

    @Test(arguments: ["wrongRuntime", "wrongPolicy", "slowWarmup", "oldControlProtection"])
    func qualificationFailuresNeverArm(mode: String) async throws {
        let test = try await InferenceTestCase(mode: mode)
        try test.start(); try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure != nil)
        #expect(await !test.harness.requests.contains("arm"))
        #expect(await test.harness.actorCaptureClosed)
    }

    @Test(arguments: ["wrongPacket", "staleState", "wrongLease", "geometryDuringStep", "captureStops", "lateAction", "inputGap", "wrongControlRun"])
    func invalidObservationsAndActionsDisarmWithoutExecution(mode: String) async throws {
        let test = try await InferenceTestCase(mode: mode)
        try test.start(); try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure != nil)
        #expect(await !test.harness.requests.contains("execute"))
        #expect(await test.harness.requests.contains("disarm"), "\(test.coordinator.failure ?? "No failure")")
        #expect(await test.harness.processesClosed)
    }

    @Test(arguments: ["wrongReceipt", "partialReceipt"])
    func invalidReceiptsDoNotCountAsExecuted(mode: String) async throws {
        let test = try await InferenceTestCase(mode: mode)
        try test.start(); try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure != nil && test.coordinator.executedPackets == 0)
    }

    @Test func stopDuringArmDisarmsAgainAfterLateArmResponse() async throws {
        let test = try await InferenceTestCase(mode: "blockedArm")
        try test.start(); try await test.wait { await test.harness.armWaiting }
        let stopping = Task { await test.coordinator.stopAndWait() }
        try await test.wait { await test.harness.requests.contains("disarm") }
        await test.harness.resumeArm(); await stopping.value
        #expect(test.coordinator.failure == nil && !test.coordinator.isBusy)
        #expect(await test.harness.requests.filter { $0 == "disarm" }.count >= 2)
        #expect(await !test.harness.requests.contains("execute"))
    }

    @Test func stopRetainsUnacknowledgedRingUntilActorHasActuallyExited() async throws {
        let test = try await InferenceTestCase(mode: "blockedActor")
        try test.start(); try await test.wait { await test.harness.stepWaiting }
        let stopping = Task { await test.coordinator.stopAndWait() }
        try await test.wait { await test.harness.actorExitWaiting }
        let stoppingAgain = Task { await test.coordinator.stopAndWait() }
        let ring = try #require(await test.harness.ringPath)
        #expect(await test.harness.requests.filter { $0 == "actor.shutdown" }.count == 1)
        #expect(test.coordinator.isBusy && FileManager.default.fileExists(atPath: ring))
        let reference = try #require(await test.harness.references.last)
        let bytes = try Data(contentsOf: URL(fileURLWithPath: ring))
        let slotOffset = reference.offset - SharedFrameRing.slotHeaderBytes
        #expect(bytes[slotOffset] == 2) // The last consumer lease is still published.
        await test.harness.resumeExit(); await stopping.value; await stoppingAgain.value
        #expect(!FileManager.default.fileExists(atPath: ring) && !test.coordinator.isBusy)
        #expect(test.coordinator.failure == nil)
    }

    @Test func workerCrashInterruptsPendingPredictionAndJoinsCleanup() async throws {
        let test = try await InferenceTestCase(mode: "workerCrash")
        try test.start(); try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure?.contains("exited unexpectedly") == true)
        #expect(await test.harness.allClosed)
        #expect(await !test.harness.requests.contains("execute"))
    }

    @Test(arguments: ["takeover", "emergency"])
    func physicalTakeoverIsAnInterventionRatherThanAnActorFailure(mode: String) async throws {
        let test = try await InferenceTestCase(mode: mode)
        try test.start(); try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure == nil)
        #expect(test.coordinator.stopReason == (mode == "takeover" ? "Physical input took over." : "Emergency stop pressed."))
    }

    @Test func controlCrashDuringShutdownDoesNotClaimOwnedInputsWereReleased() async throws {
        let test = try await InferenceTestCase(mode: "controlCrashOnShutdown")
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        #expect(!test.coordinator.isBusy && !test.coordinator.cleanupConfirmed)
        #expect(test.coordinator.failure?.contains("input cleanup could be confirmed") == true)
        let result = try await LearningFiles.read(try #require(test.coordinator.resultsURL))
        #expect(result.fields?["status"] == .string("failed") && result.fields?["cleanupConfirmed"] == .bool(false))
    }

    @Test func quitPresentsUnconfirmedCleanupOnceAndCanCloseAfterTheWarning() async throws {
        let test = try await InferenceTestCase(mode: "controlCrashOnShutdown")
        let workspace = WorkspaceModel(inferenceCoordinator: test.coordinator)
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 1 }
        let firstQuit = await workspace.prepareForTermination()
        #expect(!firstQuit && !workspace.isClosing)
        #expect(workspace.errorMessage?.contains("Release any held controls manually") == true)
        #expect(!test.coordinator.cleanupConfirmed && !test.coordinator.isBusy)
        let secondQuit = await workspace.prepareForTermination()
        #expect(secondQuit && workspace.isClosing)
        #expect(!test.coordinator.cleanupConfirmed)
    }

    @Test func hostWaitsForMappedGuardianProofAfterExecutorExit() async throws {
        let test = try await InferenceTestCase(mode: "protectedGuardianRecovery")
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 1 }
        let stop = Task { await test.coordinator.stopAndWait() }
        try await test.wait { await test.harness.guardianRecoveryWaiting }
        #expect(test.coordinator.isBusy && !test.coordinator.cleanupConfirmed)
        #expect(test.coordinator.resultsURL == nil)
        await test.harness.settleGuardianRecovery()
        try await Task.sleep(for: .milliseconds(20))
        #expect(test.coordinator.isBusy && !test.coordinator.cleanupConfirmed && test.coordinator.resultsURL == nil)
        await test.harness.exitGuardian()
        await stop.value
        #expect(!test.coordinator.isBusy && test.coordinator.cleanupConfirmed && test.coordinator.cleanupRecoveredByGuardian)
        let result = try await LearningFiles.read(try #require(test.coordinator.resultsURL))
        #expect(result.fields?["cleanupRecoveredByGuardian"] == .bool(true))
    }
    @Test func actorDecisionsContinueAfterAdmissionBeforeTerminalControlReceipts() async throws {
        let test = try await InferenceTestCase(mode: "delayedReceipt")
        try test.start(); try await test.wait { test.coordinator.decisions >= 2 }
        #expect(test.coordinator.executedPackets == 0 && test.coordinator.isBusy)
        try await test.harness.completePendingReceipts()
        try await test.wait { test.coordinator.executedPackets >= 2 }
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.failure == nil && test.coordinator.cleanupConfirmed)
    }

    @Test func manualCleanupAcknowledgementUnlocksFutureRunsWithoutRewritingTheOldProof() async throws {
        let test = try await InferenceTestCase(mode: "controlCrashOnShutdown")
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        let original = try #require(test.coordinator.resultsURL)
        #expect(throws: AstraError.self) { try test.start() }
        #expect(test.coordinator.requiresManualControlCleanupAcknowledgement)
        try test.coordinator.acknowledgeManualControlCleanup()
        #expect(!test.coordinator.cleanupConfirmed && !test.coordinator.requiresManualControlCleanupAcknowledgement)
        let summary = try await LearningFiles.read(original)
        #expect(summary.fields?["cleanupConfirmed"] == .bool(false))
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        #expect(await test.harness.requests.filter { $0 == "control.start" }.count == 2)
    }

    @Test func previousControlCallbacksCannotStopTheNextRunOrReopenTheOldCollection() async throws {
        let test = try await InferenceTestCase(mode: "success"), first = CollectionSpy(), second = CollectionSpy()
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: first.sink())
        try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: second.sink())
        try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.harness.emitOldControlFaults()
        try await test.wait { test.coordinator.executedPackets >= 2 }
        #expect(test.coordinator.isBusy && test.coordinator.failure == nil && !first.lateEvent)
        await test.coordinator.stopAndWait()
        #expect(second.completed && !second.lateEvent && test.coordinator.failure == nil)
    }

    @Test func stoppingAnAlreadyForwardedPacketRetainsRealRejectionWithoutReportingAFault() async throws {
        let test = try await InferenceTestCase(mode: "blockedAdmission"), spy = CollectionSpy()
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { await test.harness.admissionWaiting }
        let stop = Task { await test.coordinator.stopAndWait() }
        try await test.wait { await test.harness.requests.contains("disarm") }
        try await test.harness.rejectSuspendedAdmission()
        await stop.value
        #expect(test.coordinator.failure == nil && test.coordinator.cleanupConfirmed && spy.completed)
        #expect(spy.events.contains { event in
            if case .control(let message) = event, let payload = message.payload.fields?["receipt"],
               let receipt = try? payload.decode(ExecutionReceipt.self) { return receipt.status == .rejected }
            return false
        })
    }

    @Test(arguments: ["armTakeoverEvent", "armEmergencyEvent", "armTakeoverError", "armEmergencyError"])
    func interventionDuringArmingRetainsRunIdentityAndDoesNotFailCollection(mode: String) async throws {
        let test = try await InferenceTestCase(mode: mode), spy = CollectionSpy()
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure == nil && test.coordinator.stopReason == "Pre-arm intervention.")
        #expect(test.coordinator.cleanupConfirmed && spy.completed && !spy.lateEvent)
        #expect(!(await test.harness.requests).contains("execute"))
        if mode.hasSuffix("Event") {
            #expect(spy.events.contains { event in
                if case .control(let message) = event { return message.kind == "control.stopped" && message.runID == test.coordinator.runID }
                return false
            })
        }
    }

    @Test(arguments: [false, true])
    func closingExecutionReceiptsCountEvenAfterStopOrAnAuditFailure(rejectAudit: Bool) async throws {
        let test = try await InferenceTestCase(mode: "receiptOnShutdown"), spy = CollectionSpy()
        spy.rejectExecutedEvidence = rejectAudit
        var options = InferenceOptions(); options.deterministic = false
        try test.coordinator.start(agent: test.agent, checkpoint: test.checkpoint, source: test.source, options: options, collection: spy.sink())
        try await test.wait { test.coordinator.decisions >= 1 }
        #expect(test.coordinator.executedPackets == 0)
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.executedPackets >= 1 && test.coordinator.cleanupConfirmed && spy.completed)
        #expect((test.coordinator.failure != nil) == rejectAudit)
        let summary = try await LearningFiles.read(#require(test.coordinator.resultsURL))
        #expect(summary.fields?["executedPackets"]?.int == test.coordinator.executedPackets)
    }

    @Test func currentCleanupAcknowledgementPersistsExactHistoryWithoutUpgradingTheRunProof() async throws {
        let test = try await InferenceTestCase(mode: "controlCrashOnShutdown")
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 1 }
        await test.coordinator.stopAndWait()
        let report = try #require(test.coordinator.resultsURL), original = try Data(contentsOf: report)
        let store = try LibraryStore(root: test.root)
        let model = WorkspaceModel(inferenceCoordinator: test.coordinator, historyStore: store, historyRoot: test.root,
            controlOwner: test.controlOwner, desktopLeaseURL: test.root.appendingPathComponent("history-control.lock"))
        try await model.acknowledgeInferenceCleanup()
        #expect(!test.coordinator.cleanupConfirmed && !test.coordinator.requiresManualControlCleanupAcknowledgement)
        let unchanged = try Data(contentsOf: report)
        #expect(test.controlOwner.priorCleanupJoined && unchanged == original)
        let reopened = try LibraryStore(root: test.root)
        try await reopened.inspectPriorInferenceRuns()
        #expect(try await reopened.snapshot().issues.compactMap(\.controlHistory).isEmpty)
        #expect(await model.prepareForTermination())
    }

}
