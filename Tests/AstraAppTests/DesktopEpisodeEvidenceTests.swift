import Foundation
import Testing
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

private final class EpisodeCollectorHarness: @unchecked Sendable {
    private let lock = NSLock()
    private var events: ComputeProcess.EventHandler?
    private var collection: UUID?
    private var values: [(String, JSONValue)] = []
    private var eventSequence: UInt64 = 0
    var requests: [(String, JSONValue)] { lock.withLock { values } }
    var factory: CollectorRuntime.Factory {
        { callback, _ in
            self.lock.withLock { self.events = callback }
            return CollectorRuntime(start: {
                WireMessage(kind: "hello", sequence: 0, payload: .object(["role": .string("collector"), "protocolVersion": .integer(1)]))
            }, request: { kind, payload, run in try self.request(kind, payload, run) }, shutdown: {})
        }
    }
    private func request(_ kind: String, _ payload: JSONValue, _ run: UUID) throws -> WireMessage {
        lock.withLock { values.append((kind, payload)) }
        if kind == "collector.prepare" {
            let destination = try payload.required("destination").decode(String.self)
            collection = UUID(uuidString: URL(fileURLWithPath: destination).lastPathComponent)!
            return acknowledgement(run, status: "ready", extra: ["collectionVersion": .integer(1)])
        }
        if kind == "collector.actor" {
            let snapshot = try payload.required("observation")
            let frames = try snapshot.required("frames").decode([JSONValue].self)
            let reference = try frames[0].required("reference").decode(SharedFrameReference.self)
            emit("collector.framesConsumed", run: run, fields: ["observationID": try snapshot.required("id"),
                "acknowledgements": try .encode([reference.acknowledgement])])
        }
        if kind == "collector.abort" { emit("collector.fault", run: run, fields: ["learningAborted": .bool(true), "auditContinuable": .bool(true)]) }
        if kind == "collector.finish" { emit("collector.audited", run: run, fields: ["learningEligible": .bool(false)]) }
        return acknowledgement(run, status: "queued")
    }
    private func acknowledgement(_ run: UUID, status: String, extra: [String: JSONValue] = [:]) -> WireMessage {
        WireMessage(kind: "ack", sequence: 0, runID: run, payload: .object(extra.merging([
            "collectionID": .string(collection!.uuidString), "status": .string(status)]) { _, new in new }))
    }
    private func emit(_ kind: String, run: UUID, fields: [String: JSONValue]) {
        let value = lock.withLock { () -> (ComputeProcess.EventHandler?, WireMessage) in
            defer { eventSequence += 1 }
            return (events, WireMessage(kind: kind, sequence: eventSequence, runID: run,
                payload: .object(fields.merging(["collectionID": .string(collection!.uuidString)]) { old, _ in old })))
        }
        value.0?(value.1)
    }
}
private final class EpisodeCallbacks: @unchecked Sendable {
    private let lock = NSLock()
    private var issues: [DesktopEpisodeFault] = []
    private var ends: [DesktopTerminalEvidence] = []
    var faults: [DesktopEpisodeFault] { lock.withLock { issues } }
    var terminals: [DesktopTerminalEvidence] { lock.withLock { ends } }
    func fault(_ error: DesktopEpisodeFault) { lock.withLock { issues.append(error) } }
    func terminal(_ value: DesktopTerminalEvidence) { lock.withLock { ends.append(value) } }
}
private final class BlockingEpisodeDetector: @unchecked Sendable {
    let release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var calls = 0
    private var analysisStarted = false
    func waitForEntry() async throws -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while ContinuousClock.now < deadline {
            if lock.withLock({ analysisStarted }) { return true }
            try await Task.sleep(for: .milliseconds(5))
        }
        return lock.withLock { analysisStarted }
    }
    var detector: RewardAnalysisQueue.Detector {
        { _, _, _, _ in
            let block = self.lock.withLock { self.calls += 1; return self.calls == 2 }
            if block {
                self.lock.withLock { self.analysisStarted = true }
                guard self.release.wait(timeout: .now() + 5) == .success else { throw AstraError("fixture.detector", "The detector fixture was not released.") }
            }
            return []
        }
    }
}
private struct EpisodeFixture {
    let root: URL
    let collector: CollectorSession
    let harness: EpisodeCollectorHarness
    let identity: DesktopEvidenceIdentity
    let sequence: DesktopEnvironmentSequence
    let scope: ControlScope
    let policyID = UUID()
    let signature = String(repeating: "9", count: 64)
    static func make(identity existingIdentity: DesktopEvidenceIdentity? = nil,
                     sequence existingSequence: DesktopEnvironmentSequence? = nil) async throws -> EpisodeFixture {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let identity = existingIdentity ?? DesktopEvidenceIdentity(runID: UUID(), clockID: UUID(), environmentID: UUID(), actorSourceID: UUID(), environmentSourceID: UUID())
        let harness = EpisodeCollectorHarness()
        let collector = try await CollectorSession.start(runID: identity.runID,
            configuration: .object(["destination": .string(root.appendingPathComponent(UUID().uuidString.lowercased()).path)]),
            journalURL: root.appendingPathComponent("native.jsonl"), ringURL: root.appendingPathComponent("frames.ring"),
            slotCapacity: 4096, factory: harness.factory, onFault: { _ in })
        let surface = SurfaceDescriptor(id: "episode-source", globalBounds: .init(x: 0, y: 0, width: 32, height: 32), pixelWidth: 32, pixelHeight: 32)
        return Self(root: root, collector: collector, harness: harness, identity: identity,
            sequence: try existingSequence ?? .init(identity: identity), scope: .init(surfaces: [surface]))
    }
    func frame(_ time: UInt64) -> RewardImageFrame {
        .init(metadata: .init(eventNanos: time, observedNanos: time, surface: scope.surfaces[0], byteCount: 4096, codec: "raw"),
              pixels: Data(repeating: 127, count: 4096))
    }
    func bridge(episode: UUID, program: RewardProgram, callbacks: EpisodeCallbacks,
                limits: DesktopEpisodeEvidence.Limits = .init(), detector: @escaping RewardAnalysisQueue.Detector = { _, _, _, _ in [] }) async throws -> DesktopEpisodeEvidence {
        try await DesktopEpisodeEvidence.prepare(identity: identity, episodeID: episode, policyID: policyID, policySignature: signature,
            collector: collector, sequence: sequence, program: program, scope: scope, assetRoot: root, warmupFrames: [frame(1)],
            limits: limits, detector: detector, onTerminal: callbacks.terminal, onFault: callbacks.fault)
    }
    func ready(_ bridge: DesktopEpisodeEvidence, program: RewardProgram, at time: UInt64) async throws {
        let context = try ResetContext(nextEpisodeID: bridge.episodeID, environmentID: identity.environmentID, scope: scope)
        let snapshot = try RewardEvaluator(program: program).resolveSnapshot(episodeID: bridge.episodeID, cutoffNanos: time, readings: [])
        let result = ResetResult(context: context, status: .ready, attempts: 1,
            cleanup: .init(resetID: context.resetID, observedNanos: time, confirmed: true),
            readyObservation: .init(context: context, observedNanos: time, sourceCoverage: [.init(sourceObservationID: UUID(),
                surface: scope.surfaces[0], eventNanos: time, observedNanos: time, throughNanos: time, verifiedAtNanos: time, kind: .frame)],
                readings: []), readySignals: snapshot, issue: nil)
        try await bridge.confirmReady(result)
    }
    func actor(_ bridge: DesktopEpisodeEvidence, cutoff: UInt64, step: Int, sequence: UInt64, previousState: UUID) throws -> (InferenceCollectedObservation, JSONValue, ActionPacket, UUID) {
        let frame = frame(cutoff), id = UUID(), after = UUID()
        var controls = ControlState(); controls.valid = true; controls.observedNanos = cutoff
        let input: JSONValue = .object(["observationID": .string(id.uuidString), "episodeID": .string(bridge.episodeID.uuidString),
            "cutoffNanos": .unsigned(cutoff), "geometryRevision": .integer(0), "controlState": try .encode(controls), "executedEvents": .array([])])
        let observation = InferenceCollectedObservation(runID: identity.runID, actorInput: input, frame: frame.metadata, pixels: frame.pixels)
        let packet = ActionPacket(runID: identity.runID, sequence: sequence, observationID: id, geometryRevision: 0,
                                  executeAtNanos: cutoff + 100_000_000, durationMs: 100, commands: [])
        let record: JSONValue = .object(["schemaVersion": .integer(1), "checkpointID": .string(policyID.uuidString),
            "policySignature": .string(signature), "modelSignature": .string(signature), "episodeID": .string(bridge.episodeID.uuidString),
            "episodeStep": .integer(Int64(step)), "observationID": .string(id.uuidString), "cutoffNanos": .unsigned(cutoff),
            "geometryRevision": .integer(0), "frameIDs": try .encode([frame.metadata.id]), "contextIDs": .array([]),
            "previousStateID": .string(previousState.uuidString), "nextStateID": .string(after.uuidString),
            "recurrentReset": .bool(step == 0), "elapsedSeconds": .number(0.1), "stateBefore": .array([.array([.number(0)])]),
            "packetFields": .object(["operation": .array([.integer(0)])]), "logProbability": .number(-1), "value": .number(0.25),
            "environmentResets": .integer(1), "sampler": .object(["kind": .string("categorical"), "temperature": .integer(1),
                "mixture": .string("none"), "version": .integer(1), "drawIndex": .unsigned(sequence),
                "rngStreamID": .string(identity.actorSourceID.uuidString), "stateBefore": .array([.integer(1), .integer(2)]),
                "sampleKey": .array([.integer(3), .integer(4)]), "stateAfter": .array([.integer(5), .integer(6)])])])
        return (observation, .object(["packet": try .encode(packet), "collectionRecord": record]), packet, after)
    }
    func receipt(_ packet: ActionPacket, status: ReceiptStatus, controlSequence: UInt64) throws -> WireMessage {
        var state = ControlState(); state.valid = status == .admitted || status == .executed; state.observedNanos = packet.executeAtNanos + 100_000_000
        return WireMessage(kind: "control.receipt", sequence: controlSequence, runID: identity.runID,
            payload: .object(["receipt": try .encode(ExecutionReceipt(packet: packet, status: status, observedNanos: state.observedNanos, resultingState: state))]))
    }
    func join(_ bridge: DesktopEpisodeEvidence, last: UInt64?, time: UInt64, stop: DesktopEpisodeStop) -> DesktopEpisodeJoin {
        .init(runID: identity.runID, episodeID: bridge.episodeID, generationID: bridge.generationID, actorJoined: true,
            controlJoined: true, manualProducerJoined: true, predictionResolved: true, cleanupConfirmed: true,
            stoppedNanos: time, lastProducedSequence: last, stop: stop)
    }
    func environmentMessages() throws -> [WireMessage] {
        try harness.requests.filter { $0.0 == "collector.evidence" }.map { try $0.1.required("message").decode(WireMessage.self) }
    }
}

@Test func desktopEvidenceJoinsOriginalActorsDelayedRewardsAndInactiveTerminalSuffix() async throws {
    let fixture = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks(), detector = BlockingEpisodeDetector()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    var program = RewardProgram(name: "Timed", rules: [.init(name: "Duration", kind: .ratePerSecond, amount: 2)])
    program.maximumEpisodeMS = 100
    let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks, detector: detector.detector)
    try await fixture.ready(bridge, program: program, at: 90_000_000)
    var state = UUID(), original: [JSONValue] = []
    for step in 0..<3 {
        let (observation, response, packet, next) = try fixture.actor(bridge, cutoff: 100_000_000 + UInt64(step) * 100_000_000,
            step: step, sequence: UInt64(step), previousState: state)
        original.append(response); state = next
        // Native receipts can arrive before pairing the exact actor response.
        try bridge.offer(.control(fixture.receipt(packet, status: .admitted, controlSequence: UInt64(step * 2))), generation: bridge.generationID)
        try bridge.offer(.observation(observation), generation: bridge.generationID)
        try bridge.offer(.decision(response), generation: bridge.generationID)
        try bridge.offer(.control(fixture.receipt(packet, status: step == 0 ? .executed : .cancelled,
            controlSequence: UInt64(step * 2 + 1))), generation: bridge.generationID)
    }
    #expect(try await detector.waitForEntry())
    #expect(callbacks.terminals.isEmpty) // All actor offers completed while the detector was blocked.
    try bridge.offer(.control(WireMessage(kind: "control.stopped", sequence: 6, runID: fixture.identity.runID,
        payload: .object(["cause": .string("requested")]))), generation: bridge.generationID)
    detector.release.signal()
    let completed = try await bridge.finish(joined: fixture.join(bridge, last: 2, time: 500_000_000, stop: .semanticBoundary(200_000_000)))
    guard case .ended(let produced, let learning, let last, _, let auditOnly) = completed else { Issue.record("Expected a complete episode"); return }
    #expect(produced == 3 && learning == 1 && last == 2 && !auditOnly)
    #expect(!fixture.harness.requests.contains { $0.0 == "collector.finish" })
    _ = try await fixture.collector.finish()
    let sent = fixture.harness.requests.filter { $0.0 == "collector.actor" }
    #expect(try sent.map { try $0.1.required("response") } == original)
    let messages = try fixture.environmentMessages()
    #expect(messages.map(\.sequence) == Array(0..<UInt64(messages.count)))
    let rewards = messages.filter { $0.kind == "environment.reward" }
    #expect(rewards.count == 1 && rewards[0].payload.fields?["value"]?.double == 0.2)
    let firstPacket = try original[0].required("packet").decode(ActionPacket.self)
    #expect(rewards[0].requestID == firstPacket.id)
    #expect(try rewards[0].payload.required("startNanos").decode(UInt64.self) == 100_000_000)
    for (index, message) in messages.enumerated() where message.kind == "environment.watermark" {
        #expect(try message.payload.required("throughSequence").decode(UInt64.self) == messages[index - 1].sequence)
    }
    let inactive = messages.filter { $0.payload.fields?["receipt"]?.fields?["status"] == .string("cancelled") }
    #expect(inactive.count == 2 && inactive.allSatisfy { $0.payload.fields?["receipt"]?.fields?["resultingState"]?.fields?["valid"] == .bool(false) })
    #expect(inactive.allSatisfy { $0.payload.fields?["cancellationCause"] == .string("episodeBoundary") })
    #expect(callbacks.faults.isEmpty)
}

@Test func desktopManualCoverageIsExplicitAndUnknownNeverBecomesZero() async throws {
    let fixture = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    var program = RewardProgram(name: "Manual", rules: [.init(name: "Feedback", kind: .manualMarker, amount: 1)])
    program.maximumEpisodeMS = 100
    let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks)
    try await fixture.ready(bridge, program: program, at: 90_000_000)
    var state = UUID()
    for step in 0..<2 {
        let time: UInt64 = 100_000_000 + UInt64(step) * 100_000_000
        let (observation, response, packet, next) = try fixture.actor(bridge, cutoff: time, step: step, sequence: UInt64(step), previousState: state)
        state = next
        let id = try observation.actorInput.required("observationID").decode(UUID.self)
        try bridge.offerManualSeal(.init(observationID: id, episodeID: bridge.episodeID, cutoffNanos: time,
            readings: [], markers: [], coverage: nil), generation: bridge.generationID)
        try bridge.offer(.decision(response), generation: bridge.generationID)
        try bridge.offer(.observation(observation), generation: bridge.generationID)
        try bridge.offer(.control(fixture.receipt(packet, status: .admitted, controlSequence: UInt64(step * 2))), generation: bridge.generationID)
        try bridge.offer(.control(fixture.receipt(packet, status: .executed, controlSequence: UInt64(step * 2 + 1))), generation: bridge.generationID)
    }
    _ = try await bridge.finish(joined: fixture.join(bridge, last: 1, time: 400_000_000, stop: .semanticBoundary(200_000_000)))
    _ = try await fixture.collector.finish()
    let rewards = try fixture.environmentMessages().filter { $0.kind == "environment.reward" }
    #expect(rewards.count == 1 && rewards[0].payload.fields?["value"] == .null)
}

@Test func desktopEmptyEpisodeDoesNotPoisonSharedCollectorOrRebindLateCallbacks() async throws {
    let fixture = try await EpisodeFixture.make(), old = EpisodeCallbacks(), current = EpisodeCallbacks()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let program = RewardProgram(name: "Empty")
    let first = try await fixture.bridge(episode: UUID(), program: program, callbacks: old)
    try await fixture.ready(first, program: program, at: 1)
    let empty = try await first.finish(joined: fixture.join(first, last: nil, time: 2, stop: .operatorAbort))
    guard case .empty = empty else { Issue.record("Empty episode fabricated progress"); return }
    let second = try await fixture.bridge(episode: UUID(), program: program, callbacks: current)
    try await fixture.ready(second, program: program, at: 3)
    #expect(throws: AstraError.self) {
        try first.offerManualSeal(.init(observationID: UUID(), episodeID: first.episodeID, cutoffNanos: 4,
            readings: [], markers: [], coverage: nil), generation: first.generationID)
    }
    #expect(throws: AstraError.self) {
        try second.offer(.control(WireMessage(kind: "control.stopped", sequence: 0, runID: fixture.identity.runID,
            payload: .object(["cause": .string("requested")]))), generation: first.generationID)
    }
    #expect(fixture.sequence.nextSequence == 0 && current.faults.isEmpty)
    #expect(old.faults.first?.generationID == first.generationID)
    _ = try await second.finish(joined: fixture.join(second, last: nil, time: 5, stop: .operatorAbort))
    await fixture.collector.abandon(reason: "No actors were produced by either physical episode")
    #expect(!fixture.harness.requests.contains { ["collector.begin", "collector.abort", "collector.end", "collector.finish"].contains($0.0) })
    let journal = try String(contentsOf: fixture.collector.journalURL, encoding: .utf8)
    #expect(!journal.contains("native.control")) // Wrong generations never enter the new collector's audit.
}

@Test func desktopFinalControlAuditKeepsHeadroomAfterObservationBackpressure() async throws {
    let fixture = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let program = RewardProgram(name: "Pending manual", rules: [.init(name: "Feedback", kind: .manualMarker, amount: 1)])
    let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks,
        limits: .init(maximumBytes: 2 * AstraVersion.maximumMessageBytes + 4096, maximumItems: 1))
    try await fixture.ready(bridge, program: program, at: 1)
    let (observation, response, packet, _) = try fixture.actor(bridge, cutoff: 100_000_000, step: 0, sequence: 0, previousState: UUID())
    try bridge.offer(.observation(observation), generation: bridge.generationID)
    #expect(throws: AstraError.self) { try bridge.offer(.decision(response), generation: bridge.generationID) }
    try bridge.offer(.control(fixture.receipt(packet, status: .cancelled, controlSequence: 0)), generation: bridge.generationID)
    await #expect(throws: AstraError.self) { try await bridge.finish(joined: fixture.join(bridge, last: 0, time: 300_000_000, stop: .operatorAbort)) }
    await fixture.collector.abandon(reason: "Pairing budget exhausted")
    let journal = try String(contentsOf: fixture.collector.journalURL, encoding: .utf8)
    #expect(journal.contains("native.control") && journal.contains("cancelled"))
    #expect(callbacks.faults.contains { $0.error.code == "desktop.evidenceBackpressure" })
}


private func offerTwoActorEpisode(_ fixture: EpisodeFixture, _ bridge: DesktopEpisodeEvidence,
                                 firstCutoff: UInt64, firstSequence: UInt64, rejectedSuffix: Bool = false) throws -> [UUID] {
    var state = UUID(), ids: [UUID] = []
    for step in 0..<2 {
        let (observed, response, packet, next) = try fixture.actor(bridge, cutoff: firstCutoff + UInt64(step) * 100_000_000,
            step: step, sequence: firstSequence + UInt64(step), previousState: state)
        state = next; ids.append(packet.observationID)
        try bridge.offer(.observation(observed), generation: bridge.generationID)
        try bridge.offer(.decision(response), generation: bridge.generationID)
        if !rejectedSuffix || step == 0 {
            try bridge.offer(.control(fixture.receipt(packet, status: .admitted, controlSequence: UInt64(step * 2))), generation: bridge.generationID)
        }
        try bridge.offer(.control(fixture.receipt(packet, status: rejectedSuffix && step == 1 ? .rejected : .executed,
            controlSequence: UInt64(step * 2 + 1))), generation: bridge.generationID)
    }
    return ids
}

@Test func desktopSharedCollectorKeepsGlobalEnvironmentAndProducedPacketCounters() async throws {
    let fixture = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    var program = RewardProgram(name: "Episodes", rules: [.init(name: "Time", kind: .ratePerSecond, amount: 1)])
    program.maximumEpisodeMS = 100
    var episodeIDs: [UUID] = []
    for index in 0..<2 {
        let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks)
        let cutoff: UInt64 = 100_000_000 + UInt64(index) * 400_000_000
        episodeIDs.append(bridge.episodeID)
        try await fixture.ready(bridge, program: program, at: cutoff - 1)
        _ = try offerTwoActorEpisode(fixture, bridge, firstCutoff: cutoff, firstSequence: UInt64(index * 2), rejectedSuffix: index == 1)
        let result = try await bridge.finish(joined: fixture.join(bridge, last: UInt64(index * 2 + 1), time: cutoff + 300_000_000,
            stop: .semanticBoundary(cutoff + 100_000_000)))
        guard case .ended(let produced, let learning, let last, _, _) = result else { Issue.record("Missing completed episode"); return }
        #expect(produced == 2 && learning == 1 && last == UInt64(index * 2 + 1))
        // An empty episode between useful ones cannot abort their shared collector.
        if index == 0 {
            let empty = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks)
            try await fixture.ready(empty, program: program, at: cutoff + 350_000_000)
            let result = try await empty.finish(joined: fixture.join(empty, last: nil, time: cutoff + 360_000_000, stop: .operatorAbort))
            guard case .empty = result else { Issue.record("Empty episode produced progress"); return }
        }
    }
    _ = try await fixture.collector.finish()
    let requests = fixture.harness.requests
    #expect(requests.filter { $0.0 == "collector.begin" }.count == 2)
    #expect(!requests.contains { $0.0 == "collector.abort" })
    let ends = requests.filter { $0.0 == "collector.end" }
    #expect(try ends.map { try $0.1.required("lastActorSequence").decode(UInt64.self) } == [1, 3])
    let messages = try fixture.environmentMessages()
    #expect(messages.map(\.sequence) == Array(0..<UInt64(messages.count)))
    #expect(try messages.filter { $0.kind == "environment.reward" }.map { try $0.payload.required("episodeID").decode(UUID.self) } == episodeIDs)
    let rejected = messages.filter { $0.payload.fields?["receipt"]?.fields?["status"] == .string("rejected") }
    #expect(rejected.count == 1 && rejected[0].payload.fields?["receipt"]?.fields?["resultingState"]?.fields?["valid"] == .bool(false))
    #expect(callbacks.faults.isEmpty)
}

@Test func desktopOperatorAbortIgnoresLateManualSealsWithoutInventingReward() async throws {
    let fixture = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let program = RewardProgram(name: "Stopped feedback", rules: [.init(name: "Manual", kind: .manualMarker, amount: 1)])
    let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks)
    try await fixture.ready(bridge, program: program, at: 1)
    let (observation, response, packet, _) = try fixture.actor(bridge, cutoff: 100_000_000, step: 0, sequence: 4, previousState: UUID())
    try bridge.offer(.observation(observation), generation: bridge.generationID)
    try bridge.offer(.decision(response), generation: bridge.generationID)
    try bridge.requestAuditAbort(reason: "Operator ended learning", generation: bridge.generationID)
    try bridge.offerManualSeal(.init(observationID: packet.observationID, episodeID: bridge.episodeID, cutoffNanos: 100_000_000,
        readings: [], markers: [], coverage: nil), generation: bridge.generationID)
    try bridge.offer(.control(fixture.receipt(packet, status: .admitted, controlSequence: 0)), generation: bridge.generationID)
    try bridge.offer(.control(fixture.receipt(packet, status: .cancelled, controlSequence: 1)), generation: bridge.generationID)
    let result = try await bridge.finish(joined: fixture.join(bridge, last: 4, time: 400_000_000, stop: .operatorAbort))
    guard case .ended(let count, let learning, let last, _, let auditOnly) = result else { Issue.record("Audit did not close"); return }
    #expect(count == 1 && learning == 0 && last == 4 && auditOnly)
    _ = try await fixture.collector.finish()
    #expect(fixture.harness.requests.filter { $0.0 == "collector.abort" }.count == 1)
    let messages = try fixture.environmentMessages()
    #expect(!messages.contains { $0.kind == "environment.reward" || $0.kind == "environment.watermark" })
    #expect(messages.last?.payload.fields?["cancellationCause"] == .string("requested"))
    #expect(callbacks.faults.isEmpty)
}

@Test func desktopJoinCannotPromoteUnconfirmedCleanupToEpisodeEnd() async throws {
    let fixture = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let program = RewardProgram(name: "Stop")
    let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks)
    try await fixture.ready(bridge, program: program, at: 1)
    let badJoin = DesktopEpisodeJoin(runID: fixture.identity.runID, episodeID: bridge.episodeID, generationID: bridge.generationID,
        actorJoined: true, controlJoined: true, manualProducerJoined: true, predictionResolved: true, cleanupConfirmed: false,
        stoppedNanos: 2, lastProducedSequence: nil, stop: .operatorAbort)
    await #expect(throws: AstraError.self) { try await bridge.finish(joined: badJoin) }
    await fixture.collector.abandon(reason: "Cleanup remains unconfirmed")
    #expect(!fixture.harness.requests.contains { $0.0 == "collector.end" || $0.0 == "collector.finish" })
}


@Test func desktopCollectorRotationPreservesEnvironmentProducerSequence() async throws {
    let first = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks()
    let second = try await EpisodeFixture.make(identity: first.identity, sequence: first.sequence)
    defer { try? FileManager.default.removeItem(at: first.root); try? FileManager.default.removeItem(at: second.root) }
    var program = RewardProgram(name: "Rotated", rules: [.init(name: "Time", kind: .ratePerSecond, amount: 1)])
    program.maximumEpisodeMS = 100
    for (index, fixture) in [first, second].enumerated() {
        let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks)
        let start: UInt64 = 100_000_000 + UInt64(index) * 400_000_000
        try await fixture.ready(bridge, program: program, at: start - 1)
        _ = try offerTwoActorEpisode(fixture, bridge, firstCutoff: start, firstSequence: UInt64(index * 2))
        _ = try await bridge.finish(joined: fixture.join(bridge, last: UInt64(index * 2 + 1), time: start + 300_000_000,
            stop: .semanticBoundary(start + 100_000_000)))
        _ = try await fixture.collector.finish()
    }
    let one = try first.environmentMessages(), two = try second.environmentMessages()
    #expect(one.first?.sequence == 0 && two.first?.sequence == one.last!.sequence + 1)
    #expect((one + two).map(\.sequence) == Array(0..<UInt64(one.count + two.count)))
    #expect(one.allSatisfy { $0.runID == first.identity.runID } && two.allSatisfy { $0.runID == first.identity.runID })
    #expect(callbacks.faults.isEmpty)
}

@Test func desktopLateManualSuffixSealDoesNotInvalidateKnownTerminalEvidence() async throws {
    let fixture = try await EpisodeFixture.make(), callbacks = EpisodeCallbacks()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    var program = RewardProgram(name: "Manual tail", rules: [.init(name: "Feedback", kind: .manualMarker, amount: 1)])
    program.maximumEpisodeMS = 100
    let bridge = try await fixture.bridge(episode: UUID(), program: program, callbacks: callbacks)
    try await fixture.ready(bridge, program: program, at: 90_000_000)
    var state = UUID()
    for step in 0..<3 {
        if step == 2 {
            let deadline = ContinuousClock.now.advanced(by: .seconds(2))
            while callbacks.terminals.isEmpty && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
            #expect(callbacks.terminals.count == 1)
        }
        let cutoff: UInt64 = 100_000_000 + UInt64(step) * 100_000_000
        let (observation, response, packet, next) = try fixture.actor(bridge, cutoff: cutoff, step: step, sequence: UInt64(step), previousState: state)
        state = next
        try bridge.offer(.observation(observation), generation: bridge.generationID)
        try bridge.offer(.decision(response), generation: bridge.generationID)
        try bridge.offerManualSeal(.init(observationID: packet.observationID, episodeID: bridge.episodeID, cutoffNanos: cutoff,
            readings: [], markers: [], coverage: step == 0 ? nil : .init(episodeID: bridge.episodeID,
                startNanos: cutoff - 100_000_000, endNanos: cutoff, lastSequence: nil)), generation: bridge.generationID)
        try bridge.offer(.control(fixture.receipt(packet, status: .admitted, controlSequence: UInt64(step * 2))), generation: bridge.generationID)
        try bridge.offer(.control(fixture.receipt(packet, status: .executed, controlSequence: UInt64(step * 2 + 1))), generation: bridge.generationID)
    }
    _ = try await bridge.finish(joined: fixture.join(bridge, last: 2, time: 500_000_000, stop: .semanticBoundary(200_000_000)))
    _ = try await fixture.collector.finish()
    let rewards = try fixture.environmentMessages().filter { $0.kind == "environment.reward" }
    #expect(rewards.count == 1 && rewards[0].payload.fields?["value"]?.double == 0)
    #expect(callbacks.faults.isEmpty)
}
