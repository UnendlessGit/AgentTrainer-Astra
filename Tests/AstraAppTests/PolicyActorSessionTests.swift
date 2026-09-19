import Foundation
import Testing
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

private actor PolicyActorHarness {
    let mode: String
    let original: CheckpointDocument
    var checkpointID: UUID
    let stream = UUID()
    var next: UInt64 = 0
    var generation: UInt64 = 0
    var episodeStep: UInt64 = 0
    var state = UUID()
    var episode = UUID()
    var collecting = true
    var rng: [UInt32] = [7, 9]
    var ringPath: String?
    var lastPacket: ActionPacket?
    var lastInput: JSONValue?
    var requests: [String] = []
    var blocked = false
    var joined = false
    var joining = false
    var stepContinuation: CheckedContinuation<Void, Never>?
    var exitContinuation: CheckedContinuation<Void, Never>?
    var shutdownStarted = false
    var shutdownCalls = 0
    var startBlocked = false
    var startContinuation: CheckedContinuation<Void, Never>?

    init(_ checkpoint: CheckpointDocument, mode: String) { original = checkpoint; checkpointID = checkpoint.id; self.mode = mode }
    nonisolated func runtime() -> InferenceRuntime {
        .init(start: { await self.start() },
            request: { kind, payload, run, _, _ in try await self.request(kind, payload, run) },
            shutdown: { await self.shutdown() })
    }
    func start() async -> WireMessage {
        if mode == "blockedStart" { startBlocked = true; await withCheckedContinuation { startContinuation = $0 } }
        return WireMessage(kind: "hello", sequence: 0, payload: .object(["role": .string("actor"), "protocolVersion": .integer(1)]))
    }
    func ack(_ fields: [String: JSONValue], run: UUID, kind: String = "ack") -> WireMessage {
        WireMessage(kind: kind, sequence: 1, runID: run, payload: .object(fields))
    }
    func identity(_ run: UUID) -> [String: JSONValue] {
        ["runID": .string(run.uuidString), "checkpointID": .string(checkpointID.uuidString),
         "policySignature": .string(original.policySignature), "episodeID": .string(episode.uuidString),
         "stateID": .string(state.uuidString), "needsReset": .bool(false)]
    }
    func request(_ kind: String, _ payload: JSONValue, _ run: UUID) async throws -> WireMessage {
        requests.append(kind)
        switch kind {
        case "inference.prepare":
            collecting = payload.fields?["collection"] == .bool(true)
            ringPath = payload.fields?["ring"]?.fields?["path"]?.text
            let resumed = payload.fields?["resumeActor"] == .bool(true)
            if resumed { next = mode == "lastCounter" ? UInt64.max - 1 : 8; generation = 3; rng = [11, 12] }
            var fields = identity(run)
            fields.merge(["ringID": payload.fields?["ring"]?.fields?["ringID"] ?? .null,
                "model": .object(["schema_version": .integer(2), "period_ms": .integer(100), "lead_ms": .integer(200),
                                  "packet_capacity": .integer(16), "context_sizes": .array([])]),
                "actions": try .encode(ActionCapabilities(keyCodes: [0])), "needsReset": .bool(true),
                "collection": .bool(collecting), "collectionVersion": .integer(1), "rngStreamID": .string(stream.uuidString),
                "deterministic": payload.fields?["deterministic"] ?? .bool(false), "resumedActor": .bool(resumed),
                "nextPacketSequence": .unsigned(next), "nextDrawIndex": .unsigned(next), "actorResetGeneration": .unsigned(generation)], uniquingKeysWith: { _, new in new })
            return ack(fields, run: run)
        case "inference.reset":
            episode = try payload.requiredUUID("episodeID"); state = UUID(); generation += 1; episodeStep = 0
            if let path = payload.fields?["checkpointPath"]?.text { checkpointID = UUID(uuidString: URL(fileURLWithPath: path).lastPathComponent)! }
            var fields = identity(run)
            fields.merge(["nextPacketSequence": .unsigned(next), "nextDrawIndex": .unsigned(mode == "wrongResetCounter" ? next + 1 : next),
                          "actorResetGeneration": .unsigned(generation)], uniquingKeysWith: { _, new in new })
            return ack(fields, run: run)
        case "inference.step", "inference.warmup":
            let warm = kind == "inference.warmup"
            let refs = try payload.required("frames").decode([SharedFrameReference].self)
            lastInput = payload
            if !warm && ["blocked", "forceExit"].contains(mode) {
                blocked = true
                await withCheckedContinuation { stepContinuation = $0 }
                if shutdownStarted { throw AstraError("compute.exited", "Fixture actor exited before a result.") }
            }
            if mode == "workerCrash" { throw AstraError("compute.exited", "Fixture actor crashed.") }
            let packet = ActionPacket(runID: run, sequence: next, observationID: try payload.requiredUUID("observationID"),
                geometryRevision: refs[0].metadata.surface.geometryRevision,
                executeAtNanos: try payload.required("cutoffNanos").decode(UInt64.self) + 200_000_000,
                durationMs: 100, commands: [TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)])
            let newState = UUID()
            var fields = identity(run)
            fields.merge(["stateID": .string(newState.uuidString), "packet": try .encode(packet),
                          "surfaces": try .encode(refs.map(\.metadata.surface)), "releasedFrames": try .encode(refs.map(\.acknowledgement)),
                          "logProbability": .number(-0.5), "conditionalEntropy": .number(0.5), "value": .number(1)], uniquingKeysWith: { _, new in new })
            if mode == "wrongLease" {
                fields["releasedFrames"] = try .encode([SharedFrameAcknowledgement(runID: run, ringID: refs[0].ringID,
                    slot: refs[0].slot, leaseID: UUID(), sequence: refs[0].sequence)])
            }
            if warm { fields["warmup"] = .bool(true) }
            else if collecting {
                let after: [UInt32] = [rng[0] + 1, rng[1] + 1]
                let before: [UInt32] = mode == "wrongRNG" ? [88, 99] : rng
                let sampler: [String: JSONValue] = ["version": .integer(1), "kind": .string("categorical"), "temperature": .integer(1),
                    "mixture": .string("none"), "rngStreamID": .string(stream.uuidString),
                    "drawIndex": .unsigned(mode == "wrongDraw" ? next + 1 : next),
                    "stateBefore": try .encode(before), "sampleKey": try .encode([UInt32(13), 15]), "stateAfter": try .encode(after)]
                fields["collectionRecord"] = .object([
                    "schemaVersion": .integer(1), "checkpointID": .string(checkpointID.uuidString),
                    "policySignature": .string(original.policySignature), "episodeID": .string(episode.uuidString),
                    "observationID": try payload.required("observationID"), "previousStateID": try payload.required("previousStateID"),
                    "nextStateID": .string(newState.uuidString), "cutoffNanos": try payload.required("cutoffNanos"),
                    "geometryRevision": try payload.required("geometryRevision"), "frameIDs": try .encode(refs.map(\.metadata.id)),
                    "contextIDs": .array([]), "episodeStep": .unsigned(episodeStep), "recurrentReset": .bool(episodeStep == 0),
                    "environmentResets": .unsigned(generation), "logProbability": .number(-0.5), "value": .number(1),
                    "sampler": .object(sampler)])
                rng = after
            }
            if !warm { next += 1; episodeStep += 1; state = newState; lastPacket = packet }
            return ack(fields, run: run)
        default: throw AstraError("fixture.operation", kind)
        }
    }
    func releaseStep() { stepContinuation?.resume(); stepContinuation = nil }
    func releaseExit() { exitContinuation?.resume(); exitContinuation = nil }
    func shutdown() async -> Int32? {
        shutdownCalls += 1; shutdownStarted = true; releaseStep()
        startContinuation?.resume(); startContinuation = nil
        if mode == "forceExit" { joining = true; await withCheckedContinuation { exitContinuation = $0 } }
        joined = true; return 0
    }
    func expectedProgress() -> JSONValue {
        .object(["schemaVersion": .integer(1), "runID": .string(UUID().uuidString), "rngStreamID": .string(stream.uuidString),
                 "drawIndex": .unsigned(mode == "lastCounter" ? UInt64.max - 2 : 7), "rngState": .array([.integer(11), .integer(12)]), "actorResetGeneration": .integer(3)])
    }
}

private struct PolicyActorFixture {
    let directory: URL
    let checkpoint: PolicyActorCheckpoint
    let harness: PolicyActorHarness
    let session: PolicyActorSession
    let ringURL: URL
    let now: UInt64 = 10_000_000_000
    init(mode: String = "normal") throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("astra-policy-session-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let document = CheckpointDocument(id: UUID(), agentID: UUID(), runID: nil, name: "Policy fixture", kind: "initial",
                                          trainingStep: 0, policySignature: String(repeating: "a", count: 64), parameterCount: 100)
        checkpoint = .init(document: document, directory: directory.appendingPathComponent(document.id.uuidString))
        harness = PolicyActorHarness(document, mode: mode)
        ringURL = directory.appendingPathComponent("frames.astraring")
        session = PolicyActorSession(runID: UUID(), ringURL: ringURL, slotCapacity: 4096, runtime: harness.runtime(), clock: { 10_000_000_000 })
    }
    func snapshot(cutoff: UInt64 = 1_000_000_000, revision: UInt64 = 0, frameAge: UInt64 = 0, events: [RawInputEvent] = []) throws -> PolicyActorSnapshot {
        let surface = SurfaceDescriptor(id: "surface", globalBounds: .init(x: -100, y: 0, width: 100, height: 100),
            pixelWidth: 32, pixelHeight: 32, contentBounds: .init(x: 0, y: 0, width: 32, height: 32), geometryRevision: revision)
        let metadata = FrameMetadata(eventNanos: cutoff - frameAge, observedNanos: cutoff - frameAge, surface: surface, byteCount: 4096)
        var controls = ControlState(); controls.valid = true; controls.observedNanos = cutoff
        let observed = try JSONValue.object(["controlState": try .encode(controls), "executedEvents": try .encode(events),
            "intervalCovered": .bool(true), "cutoffNanos": .unsigned(cutoff), "lastSequence": events.last.map { .unsigned($0.sequence) } ?? .null]).decode(ControlObservation.self)
        return .init(frame: .init(metadata: metadata, pixels: { Data(repeating: 101, count: 4096) }), controls: observed)
    }
    func prepare(mode: PolicyActorPreparation = .fresh(seed: 7)) async throws {
        _ = try await session.prepare(checkpoint: checkpoint, collection: true, mode: mode)
        _ = try await session.reset(confirmedEpisodeID: UUID(), contextIDs: [])
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
}

private func actorEventually(_ condition: @escaping @Sendable () async -> Bool) async throws {
    let limit = ContinuousClock.now.advanced(by: .seconds(3))
    while !(await condition()) {
        guard ContinuousClock.now < limit else { throw AstraError("fixture.timeout", "Actor fixture did not reach its boundary.") }
        try await Task.sleep(for: .milliseconds(1))
    }
}

@Suite struct PolicyActorSessionTests {
@Test func persistentActorKeepsExactPacketsAndGlobalCountersAcrossWarmupResetAndActivation() async throws {
    let fixture = try PolicyActorFixture(); defer { fixture.remove() }
    try await fixture.prepare()
    let initial = await fixture.session.state
    #expect(initial.actorProgress == nil)
    let warm = try await fixture.session.warmup(fixture.snapshot())
    #expect(warm.actorProgress == nil)
    #expect(await fixture.session.state.nextPacketSequence == 0)
    let first = try await fixture.session.beginPrediction(fixture.snapshot()).value()
    #expect(first.packet == (await fixture.harness.lastPacket))
    #expect(first.observation.frames[0].metadata.codec == "raw")
    #expect(first.observation.frames[0].pixels == Data(repeating: 101, count: 4096))
    #expect(first.actorProgress?.fields?["drawIndex"]?.uint64 == 0)
    var replacement = fixture.checkpoint.document; replacement.id = UUID(); replacement.trainingStep = 1
    _ = try await fixture.session.reset(confirmedEpisodeID: UUID(), contextIDs: [],
        activating: .init(document: replacement, directory: fixture.directory.appendingPathComponent(replacement.id.uuidString)))
    let reset = await fixture.session.state
    #expect(reset.nextPacketSequence == 1 && reset.nextDrawIndex == 1 && reset.episodeStep == 0 && reset.actorResetGeneration == 2)
    #expect(reset.actorProgress == first.actorProgress)
    _ = try await fixture.session.warmup(fixture.snapshot(cutoff: 1_100_000_000, revision: 2))
    let second = try await fixture.session.beginPrediction(fixture.snapshot(cutoff: 1_100_000_000, revision: 2)).value()
    #expect(second.packet.sequence == 1)
    #expect(second.response.payload.fields?["checkpointID"]?.uuid == replacement.id)
    #expect(second.actorProgress?.fields?["actorResetGeneration"]?.uint64 == 2)
    await fixture.session.shutdown()
    #expect(await fixture.session.state.joined)
    #expect(!FileManager.default.fileExists(atPath: fixture.ringURL.path))
}

@Test func stopAndAwaiterCancellationPreserveTheInFlightOriginalResult() async throws {
    let fixture = try PolicyActorFixture(mode: "blocked"); defer { fixture.remove() }
    try await fixture.prepare()
    let ticket = try await fixture.session.beginPrediction(fixture.snapshot())
    let waiter = Task { try await ticket.value() }
    try await actorEventually { await fixture.harness.blocked }
    waiter.cancel(); await fixture.session.requestStop()
    await #expect(throws: (any Error).self) { try await fixture.session.beginPrediction(fixture.snapshot(cutoff: 1_100_000_000)) }
    #expect(!(await fixture.harness.joined))
    await fixture.harness.releaseStep()
    let result = try await waiter.value
    #expect(result.packet == (await fixture.harness.lastPacket))
    let drained = try await fixture.session.drainPrediction()
    #expect(drained?.packet == result.packet)
    #expect(await fixture.session.state.sampledProgressKnown)
    await fixture.session.shutdown()
}

@Test func forcedActorExitKeepsUnacknowledgedRingUntilActualJoinAndRejectsResumeProgress() async throws {
    let fixture = try PolicyActorFixture(mode: "forceExit"); defer { fixture.remove() }
    try await fixture.prepare()
    let ticket = try await fixture.session.beginPrediction(fixture.snapshot())
    try await actorEventually { await fixture.harness.blocked }
    let graceful = Task { await fixture.session.shutdown() }
    try await actorEventually { await fixture.session.state.stopped }
    #expect(!(await fixture.harness.shutdownStarted))
    let stopping = Task { await fixture.session.shutdown(interruptPending: true) }
    try await actorEventually { await fixture.harness.joining }
    #expect(FileManager.default.fileExists(atPath: fixture.ringURL.path))
    await #expect(throws: (any Error).self) { try await ticket.value() }
    #expect(!(await fixture.session.state.sampledProgressKnown))
    await fixture.harness.releaseExit(); await stopping.value; await graceful.value
    #expect(await fixture.harness.shutdownCalls == 1)
    #expect(await fixture.session.state.joined)
    #expect(!FileManager.default.fileExists(atPath: fixture.ringURL.path))
}

@Test(arguments: ["wrongLease", "wrongDraw", "workerCrash"])
func invalidActorResultCannotAdvanceVerifiedProgress(mode: String) async throws {
    let fixture = try PolicyActorFixture(mode: mode); defer { fixture.remove() }
    try await fixture.prepare()
    await #expect(throws: (any Error).self) { try await fixture.session.beginPrediction(fixture.snapshot()).value() }
    let state = await fixture.session.state
    #expect(state.nextPacketSequence == 0 && !state.sampledProgressKnown && state.actorProgress == nil)
    await fixture.session.shutdown()
}

@Test func staleFramesAndWithinEpisodeGeometryChangesNeverEnterTheActor() async throws {
    for stale in [true, false] {
        let fixture = try PolicyActorFixture(); defer { fixture.remove() }
        try await fixture.prepare()
        if !stale { _ = try await fixture.session.beginPrediction(fixture.snapshot()).value() }
        let before = await fixture.harness.requests.count
        await #expect(throws: (any Error).self) {
            try await fixture.session.beginPrediction(fixture.snapshot(cutoff: 1_100_000_000, revision: stale ? 0 : 1,
                                                                      frameAge: stale ? 300_000_000 : 0)).value()
        }
        #expect(await fixture.harness.requests.count == before)
        #expect(await fixture.session.state.sampledProgressKnown)
        await fixture.session.shutdown()
    }
}

@Test(arguments: [false, true])
func resumeAuthenticatesCountersAndFirstRealRNGWithoutManufacturingProgress(wrongRNG: Bool) async throws {
    let fixture = try PolicyActorFixture(mode: wrongRNG ? "wrongRNG" : "normal"); defer { fixture.remove() }
    try await fixture.prepare(mode: .resume(expectedProgress: await fixture.harness.expectedProgress()))
    let prepared = await fixture.session.state
    #expect(prepared.nextPacketSequence == 8 && prepared.nextDrawIndex == 8 && prepared.actorResetGeneration == 4)
    #expect(prepared.actorProgress == nil)
    if wrongRNG {
        await #expect(throws: (any Error).self) { try await fixture.session.beginPrediction(fixture.snapshot()).value() }
        #expect(!(await fixture.session.state.sampledProgressKnown))
    } else {
        let result = try await fixture.session.beginPrediction(fixture.snapshot()).value()
        #expect(result.packet.sequence == 8)
        #expect(result.actorProgress?.fields?["drawIndex"]?.uint64 == 8)
    }
    await fixture.session.shutdown()
}

@Test func administrativeCounterMismatchCannotEstablishAnEpisode() async throws {
    let fixture = try PolicyActorFixture(mode: "wrongResetCounter"); defer { fixture.remove() }
    await #expect(throws: (any Error).self) { try await fixture.prepare() }
    let state = await fixture.session.state
    #expect(state.episodeID == nil && state.actorProgress == nil && state.stopped)
    await fixture.session.shutdown()
}

@Test func forcedStopDuringPrepareCannotRestartAJoinedWorker() async throws {
    let fixture = try PolicyActorFixture(mode: "blockedStart"); defer { fixture.remove() }
    let preparing = Task { try await fixture.session.prepare(checkpoint: fixture.checkpoint, collection: true) }
    try await actorEventually { await fixture.harness.startBlocked }
    preparing.cancel()
    await fixture.session.shutdown(interruptPending: true)
    await #expect(throws: (any Error).self) { try await preparing.value }
    #expect(await fixture.harness.requests.isEmpty)
    #expect(await fixture.harness.shutdownCalls == 1)
    #expect(await fixture.session.state.joined)
    #expect(!FileManager.default.fileExists(atPath: fixture.ringURL.path))
}

@Test func resetAndWarmupDoNotDrainThePreviousEpisodesPacket() async throws {
    let fixture = try PolicyActorFixture(); defer { fixture.remove() }
    try await fixture.prepare()
    let first = try await fixture.session.beginPrediction(fixture.snapshot()).value()
    _ = try await fixture.session.reset(confirmedEpisodeID: UUID(), contextIDs: [])
    _ = try await fixture.session.warmup(fixture.snapshot(cutoff: 1_100_000_000))
    await fixture.session.requestStop()
    #expect(try await fixture.session.drainPrediction() == nil)
    #expect(await fixture.session.state.actorProgress == first.actorProgress)
    await fixture.session.shutdown()
}

@Test(arguments: ["initial", "physical", "boundary", "gap"])
func collectionRejectsIneligibleHistoryBeforeSampling(kind: String) async throws {
    let fixture = try PolicyActorFixture(); defer { fixture.remove() }
    try await fixture.prepare()
    if kind != "initial" { _ = try await fixture.session.beginPrediction(fixture.snapshot()).value() }
    let requests = await fixture.harness.requests.count
    let cutoff: UInt64 = 1_100_000_000
    let event = RawInputEvent(sequence: 0, eventNanos: cutoff, observedNanos: cutoff,
        origin: kind == "physical" ? .physical : (kind == "boundary" ? .boundary : .agent), kind: kind == "gap" ? .gap : .pointer)
    await #expect(throws: (any Error).self) {
        try await fixture.session.beginPrediction(fixture.snapshot(cutoff: cutoff, events: [event])).value()
    }
    #expect(await fixture.harness.requests.count == requests)
    #expect(await fixture.session.state.sampledProgressKnown)
    await fixture.session.shutdown()
}

@Test func lastLegalPacketCounterIsUsableAndNeverWraps() async throws {
    let fixture = try PolicyActorFixture(mode: "lastCounter"); defer { fixture.remove() }
    try await fixture.prepare(mode: .resume(expectedProgress: await fixture.harness.expectedProgress()))
    _ = try await fixture.session.warmup(fixture.snapshot())
    let result = try await fixture.session.beginPrediction(fixture.snapshot()).value()
    #expect(result.packet.sequence == UInt64.max - 1)
    #expect(await fixture.session.state.nextPacketSequence == UInt64.max)
    do {
        _ = try await fixture.session.beginPrediction(fixture.snapshot(cutoff: 1_100_000_000)).value()
        Issue.record("An exhausted packet counter was reused")
    } catch let error as AstraError { #expect(error.code == "inference.counterExhausted") }
    await fixture.session.shutdown()
}

}
