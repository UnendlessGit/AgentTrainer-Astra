import Foundation
import Testing
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

private final class EpisodeGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async { await withCheckedContinuation { continuation in
        let ready = lock.withLock { if opened { return true }; waiters.append(continuation); return false }
        if ready { continuation.resume() }
    } }
    func open() { let callbacks = lock.withLock { opened = true; let result = waiters; waiters = []; return result }; callbacks.forEach { $0.resume() } }
}
private final class EpisodeFlag: @unchecked Sendable {
    private let lock = NSLock(); private var value = false
    var set: Bool { lock.withLock { value } }
    func mark() { lock.withLock { value = true } }
}
private func episodeWait(_ condition: () async -> Bool) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(4))
    while !(await condition()) {
        guard ContinuousClock.now < deadline else { throw AstraError("fixture.timeout", "Episode runner fixture did not reach its boundary.") }
        try await Task.sleep(for: .milliseconds(2))
    }
}

private actor EpisodeActorRuntime {
    let checkpoint: CheckpointDocument
    var checkpointID: UUID
    var activations: [UUID] = []
    let streamID = UUID()
    let blockedAt: UInt64?
    let crash: Bool
    let gate = EpisodeGate()
    var waiting = false, closed = false
    var next: UInt64 = 0, generation: UInt64 = 0, step: UInt64 = 0
    var stateID = UUID(), episodeID = UUID()
    var rng: [UInt32] = [1, 2]
    var packets: [ActionPacket] = []
    var requests: [String] = []
    init(checkpoint: CheckpointDocument, blockedAt: UInt64? = nil, crash: Bool = false) {
        self.checkpoint = checkpoint; checkpointID = checkpoint.id; self.blockedAt = blockedAt; self.crash = crash
    }
    nonisolated var runtime: InferenceRuntime {
        .init(start: { WireMessage(kind: "hello", sequence: 0, payload: .object(["role": .string("actor"), "protocolVersion": .integer(1)])) },
              request: { kind, input, run, _, _ in try await self.request(kind, input, run) }, shutdown: { await self.shutdown() })
    }
    func request(_ kind: String, _ input: JSONValue, _ run: UUID) async throws -> WireMessage {
        requests.append(kind)
        var fields: [String: JSONValue] = ["runID": .string(run.uuidString), "checkpointID": .string(checkpointID.uuidString),
            "policySignature": .string(checkpoint.policySignature), "needsReset": .bool(false)]
        switch kind {
        case "inference.prepare":
            fields.merge(["ringID": try input.required("ring").required("ringID"), "needsReset": .bool(true),
                "model": .object(["schema_version": .integer(2), "period_ms": .integer(100), "lead_ms": .integer(400),
                                  "packet_capacity": .integer(16), "context_sizes": .array([])]),
                "actions": try .encode(ActionCapabilities(keyCodes: [0])), "collection": .bool(true), "collectionVersion": .integer(1),
                "rngStreamID": .string(streamID.uuidString), "deterministic": .bool(false), "resumedActor": .bool(false),
                "nextPacketSequence": .unsigned(next), "nextDrawIndex": .unsigned(next), "actorResetGeneration": .unsigned(generation)], uniquingKeysWith: { _, new in new })
        case "inference.reset":
            if let path = input.fields?["checkpointPath"]?.text { checkpointID = UUID(uuidString: URL(fileURLWithPath: path).lastPathComponent)! }
            fields["checkpointID"] = .string(checkpointID.uuidString); activations.append(checkpointID)
            episodeID = try input.requiredUUID("episodeID"); stateID = UUID(); generation += 1; step = 0
            fields.merge(["episodeID": .string(episodeID.uuidString), "stateID": .string(stateID.uuidString),
                "nextPacketSequence": .unsigned(next), "nextDrawIndex": .unsigned(next), "actorResetGeneration": .unsigned(generation)], uniquingKeysWith: { _, new in new })
        case "inference.warmup", "inference.step":
            let warm = kind == "inference.warmup"
            let references = try input.required("frames").decode([SharedFrameReference].self)
            if !warm, next == blockedAt { waiting = true; await gate.wait(); waiting = false }
            if !warm, crash { throw AstraError("compute.exited", "Fixture actor lost its sampled result.") }
            let cutoff = try input.required("cutoffNanos").decode(UInt64.self)
            let packet = ActionPacket(runID: run, sequence: next, observationID: try input.requiredUUID("observationID"),
                geometryRevision: references[0].metadata.surface.geometryRevision, executeAtNanos: cutoff + 400_000_000,
                durationMs: 100, commands: [])
            let after = UUID()
            fields.merge(["episodeID": .string(episodeID.uuidString), "stateID": .string(after.uuidString), "packet": try .encode(packet),
                "surfaces": try .encode(references.map(\.metadata.surface)), "releasedFrames": try .encode(references.map(\.acknowledgement)),
                "logProbability": .number(-0.5), "conditionalEntropy": .number(0.5), "value": .number(1)], uniquingKeysWith: { _, new in new })
            if warm { fields["warmup"] = .bool(true) }
            else {
                let afterRNG: [UInt32] = [rng[0] + 1, rng[1] + 1]
                let sampler: JSONValue = .object(["version": .integer(1), "kind": .string("categorical"), "temperature": .integer(1),
                    "mixture": .string("none"), "rngStreamID": .string(streamID.uuidString), "drawIndex": .unsigned(next),
                    "stateBefore": try .encode(rng), "sampleKey": try .encode([UInt32(8), 9]), "stateAfter": try .encode(afterRNG)])
                fields["collectionRecord"] = .object(["schemaVersion": .integer(1), "checkpointID": .string(checkpointID.uuidString),
                    "policySignature": .string(checkpoint.policySignature), "episodeID": .string(episodeID.uuidString),
                    "observationID": try input.required("observationID"), "previousStateID": try input.required("previousStateID"),
                    "nextStateID": .string(after.uuidString), "cutoffNanos": .unsigned(cutoff), "geometryRevision": try input.required("geometryRevision"),
                    "frameIDs": try .encode(references.map(\.metadata.id)), "contextIDs": .array([]), "episodeStep": .unsigned(step),
                    "recurrentReset": .bool(step == 0), "environmentResets": .unsigned(generation), "logProbability": .number(-0.5),
                    "value": .number(1), "sampler": sampler])
                if let coverage = input.fields?["controlCoverageNanos"], var record = fields["collectionRecord"]?.fields {
                    record["controlCoverageNanos"] = coverage; fields["collectionRecord"] = .object(record)
                }
                packets.append(packet); next += 1; step += 1; stateID = after; rng = afterRNG
            }
        default: throw AstraError("fixture.actorOperation", kind)
        }
        return .init(kind: "ack", sequence: 1, requestID: UUID(), runID: run, payload: .object(fields))
    }
    func shutdown() -> Int32 { closed = true; gate.open(); return 0 }
}

private final class EpisodeCollectorRuntime: @unchecked Sendable {
    private let lock = NSLock()
    private var callback: ComputeProcess.EventHandler?
    private var id: UUID?
    private var sequence: UInt64 = 0
    private var recorded: [(String, JSONValue)] = []
    private let boundaries: Bool
    let finishGate: EpisodeGate?
    let finishEntered = EpisodeFlag()
    private var preparation: JSONValue = .object([:])
    private var progress: JSONValue?
    private var labels = 0
    private var aborted = false
    init(boundaries: Bool = false, finishGate: EpisodeGate? = nil) { self.boundaries = boundaries; self.finishGate = finishGate }
    var requests: [(String, JSONValue)] { lock.withLock { recorded } }
    var factory: CollectorRuntime.Factory {
        { events, _ in
            self.lock.withLock { self.callback = events }
            return .init(start: { .init(kind: "hello", sequence: 0, payload: .object(["role": .string("collector"), "protocolVersion": .integer(1)])) },
                request: { try await self.request($0, $1, $2) }, shutdown: {})
        }
    }
    private func request(_ kind: String, _ payload: JSONValue, _ run: UUID) async throws -> WireMessage {
        lock.withLock { recorded.append((kind, payload)) }
        if kind == "collector.prepare" {
            id = UUID(uuidString: URL(fileURLWithPath: try payload.required("destination").decode(String.self)).lastPathComponent); preparation = payload
        }
        if kind == "collector.actor" {
            let record = try payload.required("response").required("collectionRecord"), sampler = try record.required("sampler")
            progress = .object(["schemaVersion": .integer(1), "runID": .string(run.uuidString.lowercased()),
                "rngStreamID": .string(try sampler.requiredUUID("rngStreamID").uuidString.lowercased()),
                "drawIndex": try sampler.required("drawIndex"), "rngState": try sampler.required("stateAfter"),
                "actorResetGeneration": try record.required("environmentResets")])
            let observation = try payload.required("observation")
            let frames = try observation.required("frames").decode([JSONValue].self)
            let ref = try frames[0].required("reference").decode(SharedFrameReference.self)
            emit("collector.framesConsumed", run: run, fields: ["observationID": try observation.required("id"), "acknowledgements": try .encode([ref.acknowledgement])])
        }
        if kind == "collector.evidence", try payload.required("message").decode(WireMessage.self).kind == "environment.reward" { labels += 1 }
        if kind == "collector.abort" { aborted = true; emit("collector.fault", run: run, fields: ["learningAborted": .bool(true), "auditContinuable": .bool(true)]) }
        if kind == "collector.finish" {
            finishEntered.mark(); if let finishGate { await finishGate.wait() }
            if boundaries, let progress {
                let binding: JSONValue = .object(["runID": .string(run.uuidString), "clockID": try preparation.required("clockID"),
                    "actorSourceID": try preparation.required("actorSourceID"), "environmentSourceID": try preparation.required("environmentSourceID"),
                    "policyID": try preparation.required("policyID"), "policySignature": try preparation.required("policySignature"),
                    "purpose": .string("learning")])
                let manifest: JSONValue = .object(["schemaVersion": .integer(1), "id": .string(id!.uuidString), "rolloutID": .string(UUID().uuidString),
                    "status": .string(aborted ? "audited" : "sealed"), "controlClosureKnown": .bool(true), "actorProgress": progress,
                    "decisions": .integer(Int64(labels)), "binding": binding,
                    "actorSampling": .array([.string("categorical"), .integer(1), .string("none"), .integer(1)])])
                emit(aborted ? "collector.audited" : "collector.sealed", run: run, fields: ["path": try preparation.required("destination"),
                    "manifest": manifest, "actorProgress": progress, "learningEligible": .bool(!aborted), "controlClosureKnown": .bool(true)])
            } else { emit("collector.audited", run: run, fields: ["learningEligible": .bool(false)]) }
        }
        return .init(kind: "ack", sequence: 0, requestID: UUID(), runID: run, payload: .object([
            "collectionID": .string(id!.uuidString), "status": .string(kind == "collector.prepare" ? "ready" : "queued"), "collectionVersion": .integer(1)]))
    }
    private func emit(_ kind: String, run: UUID, fields: [String: JSONValue]) {
        let value = lock.withLock { () -> (ComputeProcess.EventHandler?, WireMessage) in
            defer { sequence += 1 }
            return (callback, .init(kind: kind, sequence: sequence, runID: run, payload: .object(fields.merging(["collectionID": .string(id!.uuidString)]) { old, _ in old })))
        }
        value.0?(value.1)
    }
}

private final class EpisodeControlChild: @unchecked Sendable {
    let shutdownGate = EpisodeGate(), shutdownEntered = EpisodeFlag()
    let blockShutdown: Bool, omitCoverage: Bool
    private let lock = NSLock(), output = NSLock()
    private let callback: ComputeProcess.EventHandler
    private var run = UUID(), wireSequence: UInt64 = 1
    private var live = false
    private var next: UInt64 = 0
    private var stateAt: UInt64 = 0
    private var pending: [UUID: (ActionPacket, UUID, Task<Void, Never>)] = [:]
    private var original: [ActionPacket] = []
    private var receipts: [ExecutionReceipt] = []
    private var ops: [String] = []
    private(set) var origin: UInt64 = 0
    var packets: [ActionPacket] { lock.withLock { original } }
    var results: [ExecutionReceipt] { lock.withLock { receipts } }
    var requests: [String] { lock.withLock { ops } }
    init(callback: @escaping ComputeProcess.EventHandler, blockShutdown: Bool, omitCoverage: Bool) {
        self.callback = callback; self.blockShutdown = blockShutdown; self.omitCoverage = omitCoverage
    }
    var runtime: NativeControlRuntime {
        .init(start: { .init(kind: "hello", sequence: 0, payload: .object(["role": .string("control"), "protocolVersion": .integer(1), "initialPacketSequenceVersion": .integer(1)])) },
            request: { kind, payload, run, _ in try await self.request(kind, payload, run) }, shutdown: {
                self.shutdownEntered.mark()
                if self.blockShutdown { await self.shutdownGate.wait() }
                return 0
            })
    }
    private func request(_ kind: String, _ payload: JSONValue, _ run: UUID) async throws -> WireMessage {
        let request = UUID()
        lock.withLock { ops.append(kind) }
        var fields: [String: JSONValue] = [:]
        switch kind {
        case "arm":
            let request = try payload.decode(ArmRequest.self)
            lock.withLock { self.run = run; origin = request.initialPacketSequence ?? 0; next = origin; stateAt = MonotonicClock.now; live = true }
            fields = ["armed": .bool(true), "nextPacketSequence": .unsigned(origin)]
        case "observation":
            let now = MonotonicClock.now
            var state = ControlState(); state.valid = lock.withLock { live }; state.observedNanos = stateAt
            fields = ["controlState": try .encode(state), "executedEvents": .array([]), "intervalCovered": .bool(true), "cutoffNanos": .unsigned(now)]
            if !omitCoverage { fields["controlCoverageNanos"] = .unsigned(now) }
        case "execute":
            let packet = try payload.decode(ActionPacket.self)
            let admitted = lock.withLock { () -> Bool in
                original.append(packet)
                guard live, packet.sequence == next else { return false }
                next += 1; return true
            }
            if !admitted {
                emit(packet, request: request, status: .rejected)
                return .init(kind: "error", sequence: 0, requestID: request, runID: run,
                    payload: .object(["code": .string("control.session"), "message": .string("The virtual helper is disarmed."), "recoverable": .bool(true)]))
            }
            let task = Task { [weak self] in
                do {
                    let now = MonotonicClock.now, end = packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000
                    if end > now { try await Task.sleep(nanoseconds: end - now) }
                    self?.complete(packet.id)
                } catch {}
            }
            lock.withLock { pending[packet.id] = (packet, request, task) }
            emit(packet, request: request, status: .admitted)
            fields = ["admitted": .bool(true)]
        case "disarm":
            let values = lock.withLock { let wasLive = live; live = false; let values = Array(pending.values); pending = [:]; return (wasLive, values) }
            for (packet, request, task) in values.1 { task.cancel(); emit(packet, request: request, status: .cancelled) }
            if values.0 { send("control.stopped", request: nil, fields: ["cause": .string("requested"), "reason": .string("Desktop control stopped.")]) }
            fields = ["stopped": .bool(true), "cleanupSettled": .bool(true)]
        default: break
        }
        return .init(kind: "ack", sequence: 0, requestID: request, runID: run, payload: .object(fields))
    }
    private func complete(_ id: UUID) {
        if let value = lock.withLock({ pending.removeValue(forKey: id) }) { emit(value.0, request: value.1, status: .executed) }
    }
    private func emit(_ packet: ActionPacket, request: UUID, status: ReceiptStatus) {
        let now = MonotonicClock.now
        var state = ControlState(); state.valid = status == .admitted || status == .executed; state.observedNanos = stateAt
        let receipt = ExecutionReceipt(packet: packet, status: status, observedNanos: now, resultingState: state)
        lock.withLock { receipts.append(receipt) }
        send("control.receipt", request: request, fields: ["receipt": try! .encode(receipt)])
    }
    private func send(_ kind: String, request: UUID?, fields: [String: JSONValue]) {
        output.withLock {
            let message = lock.withLock { () -> WireMessage in
                defer { wireSequence += 1 }
                return .init(kind: kind, sequence: wireSequence, requestID: request, runID: run, payload: .object(fields))
            }
            callback(message)
        }
    }
    func sendRetiredFault() { send("control.stopped", request: nil, fields: ["cause": .string("fault"), "reason": .string("A retired helper callback")]) }
}

private final class EpisodeControlFactory: @unchecked Sendable {
    private let lock = NSLock()
    private var children: [EpisodeControlChild] = []
    let blockShutdown: Bool, omitCoverage: Bool
    init(blockShutdown: Bool = false, omitCoverage: Bool = false) { self.blockShutdown = blockShutdown; self.omitCoverage = omitCoverage }
    var all: [EpisodeControlChild] { lock.withLock { children } }
    var factory: NativeControlRuntimeFactory {
        .init(protectsPhysicalInputs: false) { events, _ in
            let child = EpisodeControlChild(callback: events, blockShutdown: self.blockShutdown, omitCoverage: self.omitCoverage)
            self.lock.withLock { self.children.append(child) }
            return child.runtime
        }
    }
}

private struct EpisodeRunnerFixture {
    let root: URL
    let checkpoint: CheckpointDocument
    let identity: DesktopEvidenceIdentity
    let sequence: DesktopEnvironmentSequence
    let scope: ControlScope
    let actorRuntime: EpisodeActorRuntime
    let actor: PolicyActorSession
    let collectorRuntime: EpisodeCollectorRuntime
    let collector: CollectorSession
    let controls: EpisodeControlFactory
    let owner = NativeControlOwner()
    static func make(blockedAt: UInt64? = nil, actorCrash: Bool = false, blockCleanup: Bool = false,
                     omitCoverage: Bool = false) async throws -> Self {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraEpisodeRunner-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let checkpoint = CheckpointDocument(id: UUID(), agentID: UUID(), runID: nil, name: "Runner fixture", kind: "initial",
            trainingStep: 0, policySignature: String(repeating: "a", count: 64), parameterCount: 10)
        let identity = DesktopEvidenceIdentity(runID: UUID(), clockID: UUID(), environmentID: UUID(), actorSourceID: UUID(), environmentSourceID: UUID())
        let runtime = EpisodeActorRuntime(checkpoint: checkpoint, blockedAt: blockedAt, crash: actorCrash)
        let actor = PolicyActorSession(runID: identity.runID, ringURL: root.appendingPathComponent("actor.ring"), slotCapacity: 4096, runtime: runtime.runtime)
        _ = try await actor.prepare(checkpoint: .init(document: checkpoint, directory: root.appendingPathComponent(checkpoint.id.uuidString)), collection: true)
        let collectorRuntime = EpisodeCollectorRuntime()
        let collector = try await CollectorSession.start(runID: identity.runID,
            configuration: .object(["destination": .string(root.appendingPathComponent(UUID().uuidString.lowercased()).path)]),
            journalURL: root.appendingPathComponent("journal.jsonl"), ringURL: root.appendingPathComponent("collector.ring"), slotCapacity: 4096,
            factory: collectorRuntime.factory, onFault: { _ in })
        let surface = SurfaceDescriptor(id: "runner", globalBounds: .init(x: 0, y: 0, width: 32, height: 32), pixelWidth: 32, pixelHeight: 32)
        return .init(root: root, checkpoint: checkpoint, identity: identity, sequence: try .init(identity: identity),
            scope: .init(surfaces: [surface], wholeDesktop: true), actorRuntime: runtime, actor: actor,
            collectorRuntime: collectorRuntime, collector: collector, controls: .init(blockShutdown: blockCleanup, omitCoverage: omitCoverage))
    }
    func image() -> InferenceImage {
        let now = MonotonicClock.now
        let frame = FrameMetadata(eventNanos: now, observedNanos: now, surface: scope.surfaces[0], byteCount: 4096, codec: "raw")
        return .init(metadata: frame, pixels: { Data(repeating: 127, count: 4096) })
    }
    func reset(_ program: RewardProgram) async throws -> ResetResult {
        let context = try ResetContext(nextEpisodeID: UUID(), environmentID: identity.environmentID, scope: scope)
        let frame = image(), now = MonotonicClock.now
        let resolved = try RewardEvaluator(program: program).resolveSnapshot(episodeID: context.nextEpisodeID, cutoffNanos: now, readings: [])
        let ready = ResetResult(context: context, status: .ready, attempts: 1,
            cleanup: .init(resetID: context.resetID, observedNanos: frame.metadata.eventNanos, confirmed: true),
            readyObservation: .init(context: context, observedNanos: now, sourceCoverage: [
                .init(sourceObservationID: frame.metadata.id, surface: frame.metadata.surface, eventNanos: frame.metadata.eventNanos,
                    observedNanos: frame.metadata.observedNanos, throughNanos: frame.metadata.eventNanos, verifiedAtNanos: now, kind: .frame)
            ], readings: []), readySignals: resolved, issue: nil)
        _ = try await actor.reset(confirmedEpisodeID: context.nextEpisodeID, contextIDs: [])
        let warmFrame = image(), cutoff = MonotonicClock.now
        var state = ControlState(); state.valid = true; state.observedNanos = cutoff
        let controls = try JSONValue.object(["controlState": try .encode(state), "executedEvents": .array([]), "intervalCovered": .bool(true),
            "cutoffNanos": .unsigned(cutoff), "controlCoverageNanos": .unsigned(cutoff)]).decode(ControlObservation.self)
        _ = try await actor.warmup(.init(frame: warmFrame, controls: controls))
        return ready
    }
    func runner(reset: ResetResult, program: RewardProgram, manual: DesktopEpisodeManualProducer? = nil,
                phase: @escaping @Sendable (DesktopEpisodePhase) -> Void = { _ in }) -> DesktopEpisodeRunner {
        .init(actor: actor, checkpoint: checkpoint, reset: reset, identity: identity, collector: collector, sequence: sequence,
            program: program, assetRoot: root, captureRead: { image() }, verifyScope: { MonotonicClock.now },
            controlFactory: controls.factory, controlOwner: owner, recoveryDirectory: root,
            manualProducer: manual, detector: { _, _, _, _ in [] }, onPhase: phase)
    }
    func close() async {
        actorRuntime.gate.open()
        controls.all.forEach { $0.shutdownGate.open() }
        await actor.shutdown(interruptPending: true)
        await collector.abandon(reason: "Permission-free fixture completed.")
        try? FileManager.default.removeItem(at: root)
    }
}
private func withEpisodeRunnerFixture(blockedAt: UInt64? = nil, actorCrash: Bool = false, blockCleanup: Bool = false,
                                      omitCoverage: Bool = false, _ body: (EpisodeRunnerFixture) async throws -> Void) async throws {
    let fixture = try await EpisodeRunnerFixture.make(blockedAt: blockedAt, actorCrash: actorCrash, blockCleanup: blockCleanup, omitCoverage: omitCoverage)
    do { try await body(fixture); await fixture.close() }
    catch { await fixture.close(); throw error }
}
private func shortProgram() -> RewardProgram {
    var value = RewardProgram(name: "Timed fixture", rules: [.init(name: "Time", kind: .ratePerSecond, amount: 1)])
    value.maximumEpisodeMS = 100; return value
}

@Test func desktopEpisodeRunnerPreservesTheTerminalActorRowAndItsUnchangedRejectedSuffix() async throws {
    try await withEpisodeRunnerFixture(blockedAt: 1) { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program)
        let runner = fixture.runner(reset: reset, program: program)
        let running = Task { await runner.run() }
        try await episodeWait { await fixture.actorRuntime.waiting && fixture.controls.all.first?.requests.contains("disarm") == true }
        #expect(!(await fixture.actorRuntime.closed))
        fixture.actorRuntime.gate.open()
        let result = await running.value
        #expect(result.issue == nil && result.cleanupConfirmed && result.actorState.sampledProgressKnown)
        #expect(result.terminal?.outcome == .truncated)
        #expect(result.producedDecisions == 2 && result.admittedPackets == 1)
        let packets = await fixture.actorRuntime.packets
        #expect(fixture.controls.all.first?.packets == packets)
        #expect(fixture.controls.all.first?.results.last?.status == .rejected)
        let join = try #require(result.join)
        #expect(join.actorJoined && join.controlJoined && join.manualProducerJoined && join.predictionResolved && join.cleanupConfirmed)
        #expect(join.lastProducedSequence == packets.last?.sequence)
        guard case .ended(let produced, let learning, _, let terminal, let audit)? = result.evidence else { Issue.record("Expected an ended learning episode"); return }
        #expect(produced == 2 && learning == 1 && terminal?.outcome == .truncated && !audit)
        _ = try await fixture.collector.finish()
        let retained = fixture.collectorRuntime.requests.filter { $0.0 == "collector.actor" }
        #expect(retained.count == 2)
        let endpoint = try #require(retained.last).1.required("response")
        #expect(try endpoint.required("collectionRecord").required("cutoffNanos").decode(UInt64.self) == result.terminal?.cutoffNanos)
        #expect(endpoint.fields?["value"]?.double == 1) // Exact endpoint bootstrap remains available to the assembler.
        #expect(!(await fixture.actorRuntime.closed))
    }
}

@Test func desktopEpisodeRunnerOperatorStopDrainsTheSampleWithoutStoppingParentOwners() async throws {
    try await withEpisodeRunnerFixture(blockedAt: 0) { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program)
        let runner = fixture.runner(reset: reset, program: program)
        let resultTask = Task { await runner.run() }
        try await episodeWait { await fixture.actorRuntime.waiting }
        runner.requestStop()
        try await episodeWait { fixture.controls.all.first?.requests.contains("disarm") == true }
        #expect(!(await fixture.actorRuntime.closed))
        fixture.actorRuntime.gate.open()
        let result = await resultTask.value
        #expect(result.issue == nil, "\(String(describing: result.issue))")
        #expect(result.producedDecisions == 1 && result.admittedPackets == 0 && result.executedPackets == 0)
        guard case .operatorAbort = result.stop else { Issue.record("Expected explicit operator abort"); return }
        guard case .ended(_, let learning, _, _, let audit)? = result.evidence else { Issue.record("Missing truthful actor audit"); return }
        #expect(learning == 0 && audit && result.cleanupConfirmed && result.actorState.nextPacketSequence == 1)
        #expect(fixture.controls.all.first?.packets == (await fixture.actorRuntime.packets))
        #expect(fixture.controls.all.first?.results.last?.resultingState.valid == false)
        #expect(!fixture.collectorRuntime.requests.contains { $0.0 == "collector.finish" })
        #expect(!(await fixture.actorRuntime.closed) && !result.actorState.stopped && !result.actorState.joined)
    }
}

@Test func desktopEpisodeRunnerEmptyCancellationDoesNotInventAnEpisodeOrAdvanceRNG() async throws {
    try await withEpisodeRunnerFixture { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program)
        let runner = fixture.runner(reset: reset, program: program)
        runner.requestStop()
        let result = await runner.run()
        #expect(result.issue == nil && result.producedDecisions == 0 && result.lastProducedSequence == nil)
        #expect(result.actorState.nextPacketSequence == 0 && result.actorState.actorProgress == nil && result.cleanupConfirmed)
        guard case .empty? = result.evidence else { Issue.record("Zero-result stop manufactured an episode"); return }
        #expect(result.join == nil && fixture.controls.all.isEmpty)
        #expect(!fixture.collectorRuntime.requests.contains { $0.0 == "collector.begin" })
    }
}

@Test func desktopEpisodeRunnerCannotDeclareUnknownSampleProgressEmptyOrReleasedLearning() async throws {
    try await withEpisodeRunnerFixture(actorCrash: true) { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program)
        let result = await fixture.runner(reset: reset, program: program).run()
        #expect(result.issue != nil && result.producedDecisions == 0 && !result.actorState.sampledProgressKnown)
        #expect(result.evidence == nil && result.join?.predictionResolved == false && result.cleanupConfirmed)
        #expect(!(await fixture.actorRuntime.closed))
    }
}

@Test func desktopEpisodeRunnerWaitsForControlJoinEvenWhenItsActorIsIdle() async throws {
    try await withEpisodeRunnerFixture(blockCleanup: true) { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program)
        let runner = fixture.runner(reset: reset, program: program)
        let finished = EpisodeFlag()
        let work = Task { defer { finished.mark() }; return await runner.run() }
        try await episodeWait { runner.phase == .running }
        runner.requestStop()
        try await episodeWait { fixture.controls.all.first?.shutdownEntered.set == true }
        let actorClosed = await fixture.actorRuntime.closed
        #expect(!finished.set && !fixture.owner.priorCleanupJoined && !actorClosed)
        fixture.controls.all.first?.shutdownGate.open()
        let result = await work.value
        #expect(result.cleanupConfirmed && fixture.owner.priorCleanupJoined && result.join?.controlJoined == true)
    }
}

@Test func desktopEpisodeRunnerRefusesUnprovedControlCoverageBeforeSampling() async throws {
    try await withEpisodeRunnerFixture(omitCoverage: true) { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program)
        let result = await fixture.runner(reset: reset, program: program).run()
        #expect(result.issue?.code == "desktop.controlCoverage" && result.producedDecisions == 0 && result.cleanupConfirmed)
        #expect(await fixture.actorRuntime.packets.isEmpty)
    }
}

@Test func desktopEpisodeRunnerDeadlineClosesControlWhileWaitingForTheOriginalSample() async throws {
    try await withEpisodeRunnerFixture(blockedAt: 0) { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program)
        let runner = fixture.runner(reset: reset, program: program)
        let running = Task { await runner.run() }
        try await episodeWait { await fixture.actorRuntime.waiting }
        try await episodeWait { fixture.controls.all.first?.requests.contains("disarm") == true }
        #expect(await fixture.actorRuntime.waiting)
        #expect(!(await fixture.actorRuntime.closed))
        fixture.actorRuntime.gate.open()
        let result = await running.value
        #expect(result.issue?.code == "desktop.policyDeadline" && result.cleanupConfirmed && result.actorState.sampledProgressKnown)
        #expect(result.producedDecisions == 1 && result.admittedPackets == 0)
        #expect(fixture.controls.all.first?.packets == (await fixture.actorRuntime.packets))
        #expect(fixture.controls.all.first?.results.last?.status == .rejected)
    }
}

@Test func desktopEpisodeRunnerKeepsGlobalActorAndEnvironmentCountersAcrossFreshControlEpisodes() async throws {
    try await withEpisodeRunnerFixture { fixture in
        let program = shortProgram(), initial = try await fixture.reset(program)
        let oldRunner = fixture.runner(reset: initial, program: program)
        let first = await oldRunner.run()
        #expect(first.issue == nil && first.cleanupConfirmed)
        let previousEnvironmentSequence = fixture.sequence.nextSequence
        let next = try await fixture.reset(program), current = fixture.runner(reset: next, program: program)
        let work = Task { await current.run() }
        try await episodeWait { current.phase == .running }
        fixture.controls.all.first?.sendRetiredFault()
        let second = await work.value
        #expect(second.issue == nil && second.cleanupConfirmed && second.episodeID != first.episodeID)
        #expect(fixture.controls.all.count == 2 && fixture.controls.all[1].origin == first.actorState.nextPacketSequence)
        #expect(second.actorState.nextPacketSequence == first.actorState.nextPacketSequence + UInt64(second.producedDecisions))
        let actorClosed = await fixture.actorRuntime.closed
        #expect(fixture.sequence.nextSequence > previousEnvironmentSequence && !actorClosed)
        _ = try await fixture.collector.finish()
        let messages = try fixture.collectorRuntime.requests.filter { $0.0 == "collector.evidence" }.map { try $0.1.required("message").decode(WireMessage.self) }
        #expect(messages.map(\.sequence) == Array(0..<UInt64(messages.count)))
    }
}

@Test func desktopEpisodeRunnerJoinsManualProducerAndKeepsUnknownRewardsUnknown() async throws {
    try await withEpisodeRunnerFixture { fixture in
        var program = shortProgram()
        program.rules = [.init(name: "Manual", kind: .manualMarker, amount: 1)]
        let reset = try await fixture.reset(program), gate = EpisodeGate(), joining = EpisodeFlag(), ended = EpisodeFlag()
        let manual = DesktopEpisodeManualProducer(observe: { observation, seal in
            try seal(.init(observationID: observation.observationID, episodeID: observation.episodeID,
                           cutoffNanos: observation.cutoffNanos, readings: [], markers: [], coverage: nil))
        }, closeAndJoin: { joining.mark(); await gate.wait() })
        let runner = fixture.runner(reset: reset, program: program, manual: manual)
        let task = Task { defer { ended.mark() }; return await runner.run() }
        try await episodeWait { joining.set }
        #expect(!ended.set && fixture.owner.priorCleanupJoined && runner.phase == .joiningEvidence)
        gate.open()
        let result = await task.value
        #expect(result.issue == nil && result.join?.manualProducerJoined == true && result.cleanupConfirmed)
        _ = try await fixture.collector.finish()
        let messages = try fixture.collectorRuntime.requests.filter { $0.0 == "collector.evidence" }.map { try $0.1.required("message").decode(WireMessage.self) }
        let rewards = messages.filter { $0.kind == "environment.reward" }
        #expect(!rewards.isEmpty && rewards.allSatisfy { $0.payload.fields?["value"] == .null })
    }
}

private final class LoopResetDriver: ResetControlDriver, @unchecked Sendable {
    private let lock = NSLock()
    private var preparations = 0
    let unconfirmed: Bool
    init(unconfirmed: Bool) { self.unconfirmed = unconfirmed }
    var prepared: Int { lock.withLock { preparations } }
    func prepare(context: ResetContext, capabilities: ActionCapabilities) async throws -> ResetBinding {
        #expect(capabilities.isEmpty)
        lock.withLock { preparations += 1 }
        return .init(context: context, priorOwnersJoined: true, verifiedAtNanos: MonotonicClock.now)
    }
    func observe(context: ResetContext) async throws -> ResetObservationSnapshot {
        let now = MonotonicClock.now
        return .init(context: context, observedNanos: now, sourceCoverage: [
            .init(sourceObservationID: UUID(), surface: context.scope.surfaces[0], eventNanos: now, observedNanos: now,
                  throughNanos: now, verifiedAtNanos: now, kind: .frame)
        ], readings: [])
    }
    func execute(_ packet: ActionPacket, context: ResetContext) async throws -> ExecutionReceipt {
        throw AstraError("fixture.resetAction", "Manual reset fixture cannot post actions.")
    }
    func checkHealth(resetID: UUID) throws {}
    func requestStop(resetID: UUID) {}
    func release(context: ResetContext) async -> ResetReleaseProof {
        .init(resetID: context.resetID, observedNanos: MonotonicClock.now, confirmed: !unconfirmed,
              issue: unconfirmed ? "Virtual cleanup remains unconfirmed." : nil)
    }
}
private final class LoopResetReference: @unchecked Sendable {
    private let lock = NSLock()
    private weak var value: ResetRunner?
    func set(_ runner: ResetRunner) { lock.withLock { value = runner } }
    func ready(_ id: UUID) { let runner = lock.withLock { value }; _ = runner?.confirmManualReady(resetID: id) }
}

private final class EpisodeLoopHost: @unchecked Sendable {
    let fixture: EpisodeRunnerFixture
    let configuration: DesktopLearningConfiguration
    let identity: DesktopEvidenceIdentity
    let sequence: DesktopEnvironmentSequence
    let mode: String
    let resetEntered = EpisodeFlag(), resetGate = EpisodeGate()
    let collectorFinishGate: EpisodeGate?
    private let lock = NSLock()
    private var collectors: [(CollectorSession, EpisodeCollectorRuntime)] = []
    private var resetDrivers: [LoopResetDriver] = []
    private var resetResults: [ResetResult] = []
    private var checkpointsDuringReadiness: [UUID] = []
    private var learnedPolicies: [UUID] = []
    private var published: [CheckpointDocument] = []
    private var resumes: [Bool] = []
    private var stoppedBoundaries: [DesktopStoppedCollection] = []
    private var warmups = 0
    private var learningStopped = false
    weak var loop: DesktopLearningLoop?
    init(fixture: EpisodeRunnerFixture, minimum: Int, mode: String = "normal") throws {
        self.fixture = fixture; self.mode = mode
        collectorFinishGate = mode == "blockCollectorFinish" ? EpisodeGate() : nil
        let program = shortProgram()
        var options = DesktopLearningOptions(); options.initialCheckpointID = fixture.checkpoint.id; options.training.rolloutDecisions = minimum
        let manifest: JSONValue = .object(["model": .object(["schema_version": .integer(2), "period_ms": .integer(100),
            "lead_ms": .integer(400), "packet_capacity": .integer(16), "context_sizes": .array([])]),
            "actions": try ActionCapabilities(keyCodes: [0]).policyVocabulary()])
        let prepared = PreparedDesktopPolicy(checkpoint: .init(document: fixture.checkpoint,
            directory: fixture.root.appendingPathComponent(fixture.checkpoint.id.uuidString.lowercased())), manifest: manifest)
        configuration = try .init(prepared: prepared, reward: .init(definition: program, surfaces: [:], scope: fixture.scope),
                                  scope: fixture.scope, options: options)
        identity = .init(runID: fixture.identity.runID, clockID: fixture.identity.clockID, environmentID: configuration.environmentID,
                         actorSourceID: fixture.identity.actorSourceID, environmentSourceID: fixture.identity.environmentSourceID)
        sequence = try .init(identity: identity)
    }
    var learned: [UUID] { lock.withLock { learnedPolicies } }
    var saved: [CheckpointDocument] { lock.withLock { published } }
    var resumeFlags: [Bool] { lock.withLock { resumes } }
    var boundaries: [DesktopStoppedCollection] { lock.withLock { stoppedBoundaries } }
    var readinessPolicies: [UUID] { lock.withLock { checkpointsDuringReadiness } }
    var resetPreparationCount: Int { lock.withLock { resetDrivers.reduce(0) { $0 + $1.prepared } } }
    var resetProofs: [ResetResult] { lock.withLock { resetResults } }
    var warmupCount: Int { lock.withLock { warmups } }
    var runtimes: [EpisodeCollectorRuntime] { lock.withLock { collectors.map(\.1) } }
    var operations: DesktopLearningOperations {
        .init(collector: { try await self.makeCollector($0, progress: $1, id: $2) },
            reset: { try await self.reset(episodeID: $0, cancellation: $1) }, warmup: { try await self.warmup($0) },
            episode: { ready, checkpoint, collector in
                DesktopEpisodeRunner(actor: self.fixture.actor, checkpoint: checkpoint, reset: ready, identity: self.identity,
                    collector: collector, sequence: self.sequence, program: self.configuration.program, assetRoot: self.fixture.root,
                    captureRead: { self.fixture.image() }, verifyScope: { MonotonicClock.now }, controlFactory: self.fixture.controls.factory,
                    controlOwner: self.fixture.owner, recoveryDirectory: self.fixture.root, detector: { _, _, _, _ in [] })
            }, learn: { try await self.learn($0, resume: $1, admitted: $2) },
            stopLearning: { self.lock.withLock { self.learningStopped = true } },
            preserveStopped: { try await self.preserve($0, validate: $1) })
    }
    private func makeCollector(_ checkpoint: CheckpointDocument, progress: JSONValue?, id: UUID) async throws -> CollectorSession {
        let runtime = EpisodeCollectorRuntime(boundaries: true, finishGate: collectorFinishGate)
        let destination = fixture.root.appendingPathComponent(id.uuidString.lowercased())
        let configuration = try configuration.collector(identity: identity, checkpoint: checkpoint, destination: destination, previousActorProgress: progress)
        let session = try await CollectorSession.start(runID: identity.runID, configuration: configuration,
            journalURL: fixture.root.appendingPathComponent(id.uuidString + ".jsonl"), ringURL: fixture.root.appendingPathComponent(id.uuidString + ".ring"),
            slotCapacity: 4096, factory: runtime.factory, onFault: { _ in })
        lock.withLock { collectors.append((session, runtime)) }; return session
    }
    private func reset(episodeID: UUID, cancellation: ResetCancellation) async throws -> ResetResult {
        resetEntered.mark()
        if mode == "blockResetAdmission" { await resetGate.wait() }
        let context = try ResetContext(resetID: cancellation.resetID, nextEpisodeID: episodeID, environmentID: identity.environmentID, scope: fixture.scope)
        let driver = LoopResetDriver(unconfirmed: mode == "unconfirmedReset"), reference = LoopResetReference()
        lock.withLock { resetDrivers.append(driver) }
        let runner = ResetRunner(driver: driver, progress: { if $0.phase == .awaitingManualReady { reference.ready($0.resetID) } })
        reference.set(runner)
        let result = try await runner.run(context: context, program: configuration.program, cancellation: cancellation)
        let checkpoint = await fixture.actor.binding?.checkpoint.id
        lock.withLock {
            resetResults.append(result)
            if result.status == .ready, let checkpoint { checkpointsDuringReadiness.append(checkpoint) }
        }
        return result
    }
    private func warmup(_ ready: ResetResult) async throws {
        lock.withLock { warmups += 1 }
        let frame = fixture.image(), cutoff = MonotonicClock.now
        var state = ControlState(); state.valid = true; state.observedNanos = cutoff
        let controls = try JSONValue.object(["controlState": try .encode(state), "executedEvents": .array([]), "intervalCovered": .bool(true),
            "cutoffNanos": .unsigned(cutoff), "controlCoverageNanos": .unsigned(cutoff)]).decode(ControlObservation.self)
        _ = try await fixture.actor.warmup(.init(frame: frame, controls: controls))
    }
    private func learn(_ batch: DesktopLearningBatch, resume: Bool, admitted: @escaping @Sendable () async throws -> Void) async throws -> ExternalLearningResult {
        try await admitted()
        let before = try #require(await fixture.actor.binding)
        #expect(!before.isAvailable && before.checkpoint.id == batch.checkpoint.id && fixture.owner.priorCleanupJoined)
        await #expect(throws: AstraError.self) { try await self.fixture.actor.reset(confirmedEpisodeID: UUID(), contextIDs: []) }
        lock.withLock { learnedPolicies.append(batch.checkpoint.id); resumes.append(resume) }
        if mode == "learnerFailure" { throw AstraError("fixture.learner", "Injected learner failure after a verified source boundary.") }
        if lock.withLock({ learningStopped }) {
            return .init(runID: UUID(), checkpoint: nil, cancelled: true, actorProgress: batch.actorProgress)
        }
        var next = batch.checkpoint; next.id = UUID(); next.runID = UUID(); next.kind = "reinforcement"; next.trainingStep += 1
        lock.withLock { published.append(next) }
        #expect(await fixture.actor.binding?.checkpoint.id == batch.checkpoint.id)
        return .init(runID: next.runID!, checkpoint: next, cancelled: false, actorProgress: batch.actorProgress)
    }
    private func preserve(_ stopped: DesktopStoppedCollection, validate: @escaping @Sendable () async throws -> Void) async throws -> CheckpointDocument? {
        try await validate()
        #expect(fixture.owner.priorCleanupJoined && stopped.joins.allSatisfy { $0.actorJoined && $0.controlJoined && $0.cleanupConfirmed && $0.predictionResolved })
        let binding = try #require(await fixture.actor.binding)
        #expect(!binding.isAvailable && binding.checkpoint.matchesIdentity(of: stopped.checkpoint.document))
        #expect(try JSONValue.encode(stopped.result.payload.required("actorProgress")) == binding.state.actorProgress.map(JSONValue.encode))
        lock.withLock { stoppedBoundaries.append(stopped) }
        var copy = stopped.checkpoint.document; copy.id = UUID(); copy.runID = UUID(); copy.kind = "reinforcement"
        return copy
    }
    func makeLoop(targetUpdates: Int = 1, stopBeforeMinimum: Bool = false) throws -> DesktopLearningLoop {
        let loop = try DesktopLearningLoop(actor: fixture.actor, identity: identity, configuration: configuration,
            checkpoint: .init(document: fixture.checkpoint, directory: fixture.root.appendingPathComponent(fixture.checkpoint.id.uuidString.lowercased())),
            previousActorProgress: nil, completedUpdates: 0, targetUpdates: targetUpdates, resume: false, operations: operations,
            onProgress: { [weak self] progress in
                if stopBeforeMinimum, progress.completedEpisodes == 1, progress.phase == "Preparing the next episode…" { self?.loop?.requestStop() }
            })
        self.loop = loop; return loop
    }
    func close() async {
        resetGate.open(); collectorFinishGate?.open()
        let owned = lock.withLock { collectors.map(\.0) }
        for collector in owned { await collector.abandon(reason: "Loop fixture completed.") }
    }
}

@Test func desktopLearningLoopLearnsAfterTwoEpisodesAndActivatesWeightsOnlyAfterTheNextReadyReset() async throws {
    try await withEpisodeRunnerFixture { fixture in
        let host = try EpisodeLoopHost(fixture: fixture, minimum: 2)
        let loop = try host.makeLoop(targetUpdates: 2)
        let result = await loop.run()
        #expect(result.issue == nil, "\(String(describing: result.issue))")
        #expect(result.completedUpdates == 2 && result.completedEpisodes == 4 && !result.stopped)
        try #require(host.learned.count == 2 && host.saved.count == 2)
        #expect(host.resumeFlags == [false, true])
        #expect(host.learned == [fixture.checkpoint.id, host.saved[0].id])
        let activations = await fixture.actorRuntime.activations
        #expect(activations == [fixture.checkpoint.id, fixture.checkpoint.id, host.saved[0].id, host.saved[0].id])
        #expect(host.readinessPolicies == [fixture.checkpoint.id, fixture.checkpoint.id, fixture.checkpoint.id, host.saved[0].id])
        #expect(result.checkpoint.document.id == host.saved[1].id)
        #expect(await fixture.actor.binding?.checkpoint.id == host.saved[0].id)
        #expect(await fixture.actorRuntime.requests.filter { $0 == "inference.prepare" }.count == 1)
        #expect(!(await fixture.actorRuntime.closed))
        await host.close()
    }
}

@Test func desktopLearningLoopStopBeforeMinimumPreservesTheOriginalJoinedAudit() async throws {
    try await withEpisodeRunnerFixture { fixture in
        let host = try EpisodeLoopHost(fixture: fixture, minimum: 10)
        let loop = try host.makeLoop(stopBeforeMinimum: true)
        let result = await loop.run()
        #expect(result.issue == nil && result.stopped && result.completedUpdates == 0 && result.completedEpisodes == 1)
        #expect(host.learned.isEmpty && host.boundaries.count == 1)
        let boundary = try #require(host.boundaries.first)
        #expect(boundary.result.kind == "collector.audited" && boundary.checkpoint.document.id == fixture.checkpoint.id)
        #expect(boundary.joins.count == 1 && result.checkpoint.document.id != fixture.checkpoint.id)
        #expect(!(await fixture.actorRuntime.closed))
        await host.close()
    }
}

@Test(arguments: ["blockResetAdmission", "unconfirmedReset"])
func desktopLearningLoopResetCancellationAndFailureKeepCleanupTruth(mode: String) async throws {
    try await withEpisodeRunnerFixture { fixture in
        let host = try EpisodeLoopHost(fixture: fixture, minimum: 1, mode: mode)
        let loop = try host.makeLoop()
        let work = Task { await loop.run() }
        if mode == "blockResetAdmission" {
            try await episodeWait { host.resetEntered.set }
            loop.requestStop(); host.resetGate.open()
        }
        let result = await work.value
        #expect(host.learned.isEmpty && host.boundaries.isEmpty && fixture.controls.all.isEmpty && host.warmupCount == 0)
        if mode == "blockResetAdmission" {
            #expect(result.issue == nil && result.stopped && host.resetPreparationCount == 0)
            #expect(host.resetProofs.last?.status == .cancelled && host.resetProofs.last?.cleanupConfirmed == true)
        } else {
            #expect(result.issue != nil && host.resetProofs.last?.cleanupConfirmed == false)
        }
        #expect(await fixture.actorRuntime.requests.filter { $0 == "inference.reset" }.isEmpty)
        await host.close()
    }
}

@Test func desktopLearningLoopStopDuringCollectorFinalizationDoesNotStartAnUpdate() async throws {
    try await withEpisodeRunnerFixture { fixture in
        let host = try EpisodeLoopHost(fixture: fixture, minimum: 1, mode: "blockCollectorFinish")
        let loop = try host.makeLoop()
        let work = Task { await loop.run() }
        try await episodeWait { host.runtimes.first?.finishEntered.set == true }
        loop.requestStop(); host.collectorFinishGate?.open()
        let result = await work.value
        #expect(result.issue == nil && result.stopped && host.learned.isEmpty && host.boundaries.count == 1)
        #expect(host.boundaries.first?.result.kind == "collector.sealed")
        #expect(host.runtimes.first?.requests.contains { $0.0 == "collector.abort" } == false)
        await host.close()
    }
}

@Test func desktopLearningLoopRetainsItsPrimaryLearnerFailureWhilePreservingKnownProgress() async throws {
    try await withEpisodeRunnerFixture { fixture in
        let host = try EpisodeLoopHost(fixture: fixture, minimum: 1, mode: "learnerFailure")
        let result = try await host.makeLoop().run()
        #expect(result.issue?.code == "fixture.learner", "\(String(describing: result.issue))")
        #expect(result.completedUpdates == 0 && result.stopped)
        #expect(host.boundaries.count == 1 && result.checkpoint.document.id != fixture.checkpoint.id)
        #expect(!(await fixture.actorRuntime.closed))
        await host.close()
    }
}


private final class EpisodeRunnerReference: @unchecked Sendable {
    private let lock = NSLock()
    private weak var value: DesktopEpisodeRunner?
    func set(_ runner: DesktopEpisodeRunner) { lock.withLock { value = runner } }
    func stop() { let runner = lock.withLock { value }; runner?.requestStop() }
}
@Test func desktopEpisodeRunnerStopBeforePublishingControlCannotStartALateHelper() async throws {
    try await withEpisodeRunnerFixture { fixture in
        let program = shortProgram(), reset = try await fixture.reset(program), reference = EpisodeRunnerReference()
        let runner = fixture.runner(reset: reset, program: program, phase: { if $0 == .arming { reference.stop() } })
        reference.set(runner)
        let result = await runner.run()
        #expect(result.issue == nil && result.cleanupConfirmed && result.producedDecisions == 0)
        let packets = await fixture.actorRuntime.packets
        #expect(fixture.controls.all.isEmpty && packets.isEmpty)
        guard case .operatorAbort = result.stop, case .empty? = result.evidence else {
            Issue.record("An early Stop started control or fabricated an episode"); return
        }
    }
}
