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
    var ringPath: String?
    var references: [SharedFrameReference] = []
    var sequence: UInt64 = 0
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
                             request: { kind, payload, run, _, _ in try await self.request(role, kind, payload, run) },
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
        if role == "control" { self.events = events }
        else { self.failure = failure }
        var payload: [String: JSONValue] = ["role": .string(mode == "wrongRuntime" && role == "actor" ? "compute" : role), "protocolVersion": .integer(1)]
        if role == "control", mode == "protectedGuardianRecovery" { payload["recoveryVersion"] = .integer(1) }
        return WireMessage(kind: "hello", sequence: 0, payload: .object(payload))
    }
    func ack(_ payload: JSONValue = .object([:]), run: UUID) -> WireMessage {
        WireMessage(kind: "ack", sequence: 1, requestID: UUID(), runID: run, payload: payload)
    }
    func request(_ role: String, _ kind: String, _ payload: JSONValue, _ run: UUID) async throws -> WireMessage {
        requests.append(kind); currentRun = run
        switch kind {
        case "inference.prepare":
            sequence = 0
            collecting = payload.fields?["collection"] == .bool(true)
            ringPath = payload.fields?["ring"]?.fields?["path"]?.text
            return ack(.object(["runID": .string(run.uuidString), "checkpointID": .string(checkpoint.id.uuidString),
                "policySignature": .string(mode == "wrongPolicy" ? String(repeating: "b", count: 64) : checkpoint.policySignature),
                "ringID": payload.fields?["ring"]?.fields?["ringID"] ?? .null,
                "model": .object(["schema_version": .integer(2), "period_ms": .integer(100), "lead_ms": .integer(200),
                                   "packet_capacity": .integer(16), "context_sizes": .array([])]),
                "actions": try .encode(ActionCapabilities(keyCodes: [0])),
                "collection": .bool(collecting && mode != "oldCollection"), "collectionVersion": .integer(1),
                "rngStreamID": .string(rngStreamID.uuidString), "deterministic": .bool(!collecting)]), run: run)
        case "inference.reset":
            if collecting, payload.fields?["seed"] != nil { throw AstraError("fixture.reseed", "Collecting resets cannot reseed") }
            episodeID = try payload.required("episodeID").decode(UUID.self); stateID = UUID()
            return ack(.object(["runID": .string(run.uuidString), "episodeID": .string(episodeID.uuidString),
                               "stateID": .string(stateID.uuidString), "needsReset": .bool(false)]), run: run)
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
                fields["collectionRecord"] = .object([
                    "schemaVersion": .integer(1), "checkpointID": .string(checkpoint.id.uuidString),
                    "policySignature": .string(checkpoint.policySignature), "episodeID": .string(episodeID.uuidString),
                    "observationID": payload.fields?["observationID"] ?? .null,
                    "previousStateID": payload.fields?["previousStateID"] ?? .null, "nextStateID": fields["stateID"] ?? .null,
                    "cutoffNanos": .unsigned(cutoff), "frameIDs": try .encode([reference.metadata.id]),
                    "contextIDs": .array([]), "episodeStep": .unsigned(sequence), "recurrentReset": .bool(sequence == 0),
                    "logProbability": fields["logProbability"] ?? .null, "value": fields["value"] ?? .null,
                    "sampler": .object(["kind": .string("categorical"), "temperature": .integer(1), "mixture": .string("none"),
                                         "rngStreamID": .string(rngStreamID.uuidString), "drawIndex": .unsigned(sequence)])])
            }
            if warming { fields["warmup"] = .bool(true) }
            else { sequence += 1; stateID = nextState }
            return ack(.object(fields), run: run)
        case "arm":
            if mode == "protectedGuardianRecovery" {
                let request = try payload.decode(ArmRequest.self)
                let ledger = try ControlRecoveryLedger(open: #require(request.recovery))
                guard ledger.registerExecutor(pid: getpid()), ledger.registerGuardian(pid: getpid(), now: MonotonicClock.now), ledger.arm() else {
                    throw AstraError("fixture.recovery", "The virtual recovery interface did not arm.")
                }
                recoveryLedger = ledger
            }
            if mode == "blockedArm" {
                armWaiting = true
                return try await withCheckedThrowingContinuation { suspendedArm = $0 }
            }
            if let recoveryLedger {
                return ack(.object(["armed": .bool(true), "recoveryLedgerID": .string(recoveryLedger.descriptor.ledgerID.uuidString), "guardianPID": .integer(Int64(getpid()))]), run: run)
            }
            return ack(.object(["armed": .bool(true)]), run: run)
        case "observation":
            let now = MonotonicClock.now
            var controls = ControlState(); controls.valid = true; controls.observedNanos = now
            return ack(.object(["controlState": try .encode(controls), "executedEvents": .array([]),
                               "intervalCovered": .bool(mode != "inputGap"), "cutoffNanos": .unsigned(now)]),
                       run: mode == "wrongControlRun" ? UUID() : run)
        case "execute":
            let packet = try payload.decode(ActionPacket.self)
            if let recoveryLedger {
                guard recoveryLedger.beginPost(operation: .keyDown, keyCode: 0, button: nil) else { throw AstraError("fixture.recovery", "The virtual reservation was rejected.") }
                recoveryLedger.endPost(operation: .keyDown, keyCode: 0, button: nil, success: true)
            }
            var controls = ControlState(); controls.valid = true
            let results = packet.commands.enumerated().map { index, command in
                CommandResult(commandIndex: index, scheduledNanos: packet.executeAtNanos + UInt64(command.offsetMs) * 1_000_000,
                              postedNanos: packet.executeAtNanos, status: .posted)
            }
            var receipt = ExecutionReceipt(packet: packet, status: .executed, observedNanos: packet.executeAtNanos + 100_000_000,
                                            commandResults: mode == "partialReceipt" ? [] : results, resultingState: controls)
            if mode == "wrongReceipt" { receipt.sequence += 1 }
            events?(WireMessage(kind: "control.receipt", sequence: 1, runID: run, payload: .object(["receipt": try .encode(receipt)])))
            if mode == "takeover" {
                events?(WireMessage(kind: "control.stopped", sequence: 2, runID: run,
                                    payload: .object(["cause": .string("physicalTakeover"), "reason": .string("Physical input took over.")])))
            }
            return ack(.object(["admitted": .bool(true)]), run: run)
        default: return ack(run: run)
        }
    }
    func resumeArm() {
        if let suspendedArm, let currentRun { self.suspendedArm = nil; suspendedArm.resume(returning: ack(.object(["armed": .bool(true)]), run: currentRun)) }
    }
    func resumeExit() { suspendedExit?.resume(); suspendedExit = nil }
    func completeGuardianRecovery() {
        recoveryLedger?.releaseKey(0); recoveryLedger?.settleGuardian(now: MonotonicClock.now)
    }
    func shutdown(_ role: String) async -> Int32? {
        requests.append(role + ".shutdown")
        if role == "actor" {
            suspendedStep?.resume(throwing: AstraError("compute.closed", "Fixture actor stopped.")); suspendedStep = nil
            if mode == "blockedActor", !actorExited {
                actorExitWaiting = true
                await withCheckedContinuation { suspendedExit = $0 }
            }
            actorExited = true
        } else {
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
    var events: [InferenceCollectionEvent] { lock.withLock { values } }
    var completed: Bool { lock.withLock { closed } }
    var lateEvent: Bool { lock.withLock { seenLateEvent } }
    func sink() -> InferenceCollectionSink {
        .init(offer: { [self] event in
            try lock.withLock {
                if closed { seenLateEvent = true }
                if rejectObservations, case .observation = event { throw AstraError("collector.full", "Fixture collector capacity is full") }
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
        #expect(result.fields?["issue"]?.text?.contains("lost control evidence") == true)
        #expect(test.coordinator.cleanupConfirmed && spy.completed)
    }

    @Test func successfulRunWarmsWithoutControlAndReleasesResources() async throws {
        let test = try await InferenceTestCase(mode: "success")
        try test.start(); try await test.wait { test.coordinator.executedPackets >= 2 }
        await test.coordinator.stopAndWait()
        #expect(test.coordinator.failure == nil && !test.coordinator.isBusy && test.coordinator.cleanupConfirmed)
        #expect(test.coordinator.warmupLatencyMS.count == 3)
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

    @Test func physicalTakeoverIsAnInterventionRatherThanAnActorFailure() async throws {
        let test = try await InferenceTestCase(mode: "takeover")
        try test.start(); try await test.wait { !test.coordinator.isBusy }
        #expect(test.coordinator.failure == nil)
        #expect(test.coordinator.stopReason == "Physical input took over.")
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
        await test.harness.completeGuardianRecovery()
        await stop.value
        #expect(!test.coordinator.isBusy && test.coordinator.cleanupConfirmed && test.coordinator.cleanupRecoveredByGuardian)
        let result = try await LearningFiles.read(try #require(test.coordinator.resultsURL))
        #expect(result.fields?["cleanupRecoveredByGuardian"] == .bool(true))
    }
}
