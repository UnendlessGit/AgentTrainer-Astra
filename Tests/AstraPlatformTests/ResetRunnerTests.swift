import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

private final class ResetGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        await withCheckedContinuation { continuation in
            let ready = lock.withLock { if opened { return true }; waiters.append(continuation); return false }
            if ready { continuation.resume() }
        }
    }
    func open() {
        let pending = lock.withLock { opened = true; let result = waiters; waiters = []; return result }
        pending.forEach { $0.resume() }
    }
}

private final class ResetTestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var time: UInt64 = 1_000_000_000
    var now: UInt64 { lock.withLock { time } }
    func advance(_ amount: UInt64) { lock.withLock { time += amount } }
    func advance(to value: UInt64) { lock.withLock { time = max(time, value) } }
    func sleep(_ amount: UInt64) async throws {
        try Task.checkCancellation(); advance(amount); try await Task.sleep(for: .milliseconds(1))
    }
}

private final class ResetResultFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var finished: Bool { lock.withLock { value } }
    func finish() { lock.withLock { value = true } }
}

private enum ResetFixtureMode: Sendable {
    case normal, missing, stale, unchanged, wrongScope, wrongEpisode, incompleteReceipt, admissionOnly, futureRelease, unconfirmedRelease
    case blockPrepare, blockExecute, blockRelease, clockAdvancesDuringHealth, priorOwnerPending
}

private final class ResetFixtureDriver: ResetControlDriver, @unchecked Sendable {
    let clock = ResetTestClock()
    let gate = ResetGate()
    let mode: ResetFixtureMode
    let signal: RewardSignal
    let readyAfterAttempt: Int
    private let lock = NSLock()
    private var prepareCount = 0, releaseCount = 0, stopCount = 0, inFlight = 0
    private var packets: [ActionPacket] = []
    private var owned: Set<Int> = []
    private var caps: [ActionCapabilities] = []
    private var stopped = false
    private var events: [ResetProgress] = []
    private let originalFrameID = UUID()
    init(mode: ResetFixtureMode = .normal, readyAfterAttempt: Int = 1) {
        self.mode = mode; self.readyAfterAttempt = readyAfterAttempt
        if mode == .stale || mode == .unchanged {
            var value = RewardSignal(name: "Ready text", kind: .ocrText, surfaceID: "fixture", region: .init(x: 0, y: 0, width: 1, height: 1))
            value.maximumAgeMS = 1; signal = value
        } else { signal = RewardSignal(name: "Ready", kind: .manual) }
    }
    var prepared: Int { lock.withLock { prepareCount } }
    var released: Int { lock.withLock { releaseCount } }
    var stops: Int { lock.withLock { stopCount } }
    var executing: Int { lock.withLock { inFlight } }
    var posted: [ActionPacket] { lock.withLock { packets } }
    var held: Set<Int> { lock.withLock { owned } }
    var capabilities: [ActionCapabilities] { lock.withLock { caps } }
    var progress: [ResetProgress] { lock.withLock { events } }
    func record(_ progress: ResetProgress) { lock.withLock { events.append(progress) } }
    func context() throws -> ResetContext {
        try .init(nextEpisodeID: UUID(), environmentID: UUID(), scope: .init(surfaces: [
            .init(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 100, height: 100), pixelWidth: 100, pixelHeight: 100)
        ], wholeDesktop: true))
    }
    func program(automatic: Bool = true, attempts: Int = 1, hold: Bool = false) -> RewardProgram {
        var program = RewardProgram(name: "Reset fixture", signals: [signal])
        program.ready = .init(conditions: [signal.kind.isVisual
            ? .init(signalID: signal.id, comparison: .equalText, text: "Ready")
            : .init(signalID: signal.id, comparison: .isTrue)])
        if automatic {
            var commands = [TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 36)]
            if !hold { commands.append(.init(offsetMs: 80, operation: .keyUp, keyCode: 36)) }
            var plan = ResetPlan(steps: [.init(name: "Key press", packet: .init(durationMS: 100, commands: commands))])
            plan.maximumAttempts = attempts; plan.maximumDurationMS = 5_000; plan.readinessTimeoutMS = 100
            program.resetPlan = plan
        }
        return program
    }
    func runner() -> ResetRunner {
        ResetRunner(driver: self, clock: { self.clock.now }, sleep: { try await self.clock.sleep($0) }, progress: { self.record($0) })
    }
    func prepare(context: ResetContext, capabilities: ActionCapabilities) async throws -> ResetBinding {
        lock.withLock { prepareCount += 1; caps.append(capabilities); stopped = false }
        if mode == .blockPrepare { await gate.wait() }
        return .init(context: context, priorOwnersJoined: mode != .priorOwnerPending, verifiedAtNanos: clock.now)
    }
    func observe(context: ResetContext) async throws -> ResetObservationSnapshot {
        clock.advance(1_000_000)
        let current = clock.now
        let stale = mode == .stale || mode == .unchanged
        let sourceID = stale ? originalFrameID : UUID()
        let original: UInt64 = stale ? 1_000_000_000 : current
        let source = RewardSourceCoverage(sourceObservationID: sourceID, surface: context.scope.surfaces[0], eventNanos: original,
            observedNanos: original, throughNanos: mode == .unchanged ? current : original, verifiedAtNanos: current,
            kind: mode == .unchanged ? .unchanged : .frame)
        let ready = prepared >= readyAfterAttempt
        let reading = SignalReading(signalID: signal.id, episodeID: mode == .wrongEpisode ? UUID() : context.nextEpisodeID,
            eventNanos: original, observedNanos: original, value: signal.kind.isVisual ? .text("Ready") : .flag(ready),
            sourceObservationID: signal.kind.isVisual ? sourceID : nil)
        let owner = mode == .wrongScope ? try ResetContext(nextEpisodeID: context.nextEpisodeID, environmentID: UUID(), scope: context.scope) : context
        return .init(context: owner, observedNanos: current, sourceCoverage: [source], readings: mode == .missing ? [] : [reading])
    }
    func execute(_ packet: ActionPacket, context: ResetContext) async throws -> ExecutionReceipt {
        #expect(packet.runID == context.resetID && packet.runID != context.nextEpisodeID)
        lock.withLock { inFlight += 1 }
        if mode == .blockExecute { await gate.wait() } // Intentionally ignores cancellation like an in-flight post.
        clock.advance(to: packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000)
        lock.withLock {
            packets.append(packet); inFlight -= 1
            for command in packet.commands {
                if command.operation == .keyDown, let key = command.keyCode { owned.insert(key) }
                if command.operation == .keyUp, let key = command.keyCode { owned.remove(key) }
            }
        }
        var state = ControlState(); state.valid = true; state.keys = held; state.observedNanos = clock.now
        let results = packet.commands.enumerated().map { index, command in
            let time = packet.executeAtNanos + UInt64(command.offsetMs) * 1_000_000
            return CommandResult(commandIndex: index, scheduledNanos: time, postedNanos: time, status: .posted)
        }
        return .init(packet: packet, status: mode == .admissionOnly ? .admitted : .executed, observedNanos: clock.now,
                     commandResults: mode == .incompleteReceipt ? [] : results, resultingState: state)
    }
    func checkHealth(resetID: UUID) throws {
        if mode == .clockAdvancesDuringHealth { clock.advance(2_000_000) }
        if lock.withLock({ stopped }) { throw AstraError("fixture.stopped", "The virtual controller is disarmed.") }
    }
    func requestStop(resetID: UUID) { lock.withLock { stopCount += 1; stopped = true } }
    func release(context: ResetContext) async -> ResetReleaseProof {
        lock.withLock { releaseCount += 1; stopped = true }
        #expect(executing == 0) // The runner must join an in-flight submission before this proof.
        if mode == .blockRelease { await gate.wait() }
        let confirmed = mode != .unconfirmedRelease
        if confirmed { lock.withLock { owned = [] } }
        clock.advance(1_000_000)
        return .init(resetID: context.resetID, observedNanos: clock.now + (mode == .futureRelease ? 1_000_000_000 : 0),
                     confirmed: confirmed, issue: confirmed ? nil : "Virtual release remains unconfirmed.")
    }
}

private func waitForReset(_ condition: () -> Bool) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while !condition() {
        guard ContinuousClock.now < deadline else { throw AstraError("fixture.timeout", "Reset fixture did not reach its expected state.") }
        try await Task.sleep(for: .milliseconds(2))
    }
}

@Test func resetRunnerManualReadyRequiresTheCurrentAcknowledgementAndNeverArmsControls() async throws {
    let driver = ResetFixtureDriver(), runner = driver.runner(), context = try driver.context()
    var program = driver.program(automatic: false); program.ready = nil
    let task = Task { try await runner.run(context: context, program: program) }
    try await waitForReset { driver.progress.contains { $0.phase == .awaitingManualReady } }
    #expect(driver.prepared == 0 && driver.posted.isEmpty)
    #expect(!runner.confirmManualReady(resetID: UUID()))
    #expect(runner.confirmManualReady(resetID: context.resetID))
    let result = try await task.value
    #expect(result.status == .ready && result.cleanupConfirmed && result.readySignals?.episodeID == context.nextEpisodeID)
    #expect(driver.capabilities == [ActionCapabilities()] && driver.posted.isEmpty)
    #expect(!runner.confirmManualReady(resetID: context.resetID) && !runner.cancel(resetID: context.resetID))
}

@Test func resetRunnerRetriesOnlyAConditionTimeoutAfterJoinedCleanup() async throws {
    let driver = ResetFixtureDriver(readyAfterAttempt: 2), context = try driver.context()
    let result = try await driver.runner().run(context: context, program: driver.program(attempts: 2))
    #expect(result.status == .ready && result.attempts == 2 && result.cleanupConfirmed)
    #expect(driver.prepared == 2 && driver.posted.count == 2 && driver.released >= 2)
    #expect(driver.posted.allSatisfy { $0.runID == context.resetID && $0.sequence == 0 })
    #expect(Set(driver.posted.map(\.id)).count == 2 && driver.held.isEmpty)
}

@Test func resetRunnerPauseBoundaryCannotUnderflowWhenHealthWorkCrossesItsDeadline() async throws {
    let driver = ResetFixtureDriver(mode: .clockAdvancesDuringHealth)
    var program = driver.program()
    program.resetPlan?.steps.append(.init(pauseMS: 1))
    program.resetPlan?.steps.append(.init(condition: program.ready!, timeoutMS: 100))
    let result = try await driver.runner().run(context: driver.context(), program: program)
    #expect(result.status == .ready && driver.progress.contains { $0.phase == .pausing })
}

@Test(arguments: [ResetFixtureMode.incompleteReceipt, .admissionOnly, .wrongScope, .wrongEpisode, .futureRelease, .unconfirmedRelease, .priorOwnerPending])
private func resetRunnerRejectsUnprovenExecutionAndForeignOrUnconfirmedEvidence(mode: ResetFixtureMode) async throws {
    let driver = ResetFixtureDriver(mode: mode)
    let result = try await driver.runner().run(context: driver.context(), program: driver.program(attempts: 3, hold: true))
    #expect(result.status == .failed && result.readySignals == nil && driver.prepared == 1)
    #expect(driver.released == 1)
    if mode == .priorOwnerPending { #expect(driver.posted.isEmpty) }
    if mode == .futureRelease || mode == .unconfirmedRelease { #expect(!result.cleanupConfirmed && !result.cleanup.confirmed) }
    else { #expect(result.cleanupConfirmed) }
}

@Test func resetRunnerNeverTreatsMissingSignalsOrANewerWallTimeAsReadiness() async throws {
    for mode in [ResetFixtureMode.missing, .stale] {
        let driver = ResetFixtureDriver(mode: mode)
        let result = try await driver.runner().run(context: driver.context(), program: driver.program())
        #expect(result.status == .failed && result.readySignals == nil && result.issue?.code == "reset.conditionTimeout")
    }
    let driver = ResetFixtureDriver(mode: .unchanged)
    let result = try await driver.runner().run(context: driver.context(), program: driver.program())
    #expect(result.status == .ready && result.readySignals?.readings.first?.eventNanos == 1_000_000_000)
    #expect(result.readySignals?.readings.first?.observedNanos == 1_000_000_000)
    #expect(result.readySignals?.coverage?.first?.source.throughNanos ?? 0 > 1_000_000_000)
}

@Test func resetRunnerCancellationJoinsLatePreparationWithoutExecutingAnyStep() async throws {
    let driver = ResetFixtureDriver(mode: .blockPrepare), runner = driver.runner(), context = try driver.context()
    let flag = ResetResultFlag()
    let task = Task { defer { flag.finish() }; return try await runner.run(context: context, program: driver.program()) }
    try await waitForReset { driver.prepared == 1 }
    #expect(runner.cancel(resetID: context.resetID) && !runner.cancel(resetID: UUID()))
    #expect(driver.stops > 0 && !flag.finished && driver.posted.isEmpty)
    driver.gate.open()
    let result = try await task.value
    #expect(result.status == .cancelled && result.cleanupConfirmed && driver.posted.isEmpty)
}

@Test func resetRunnerDeadlineStopsAdmissionIndependentlyOfABlockedPostAndWaitsForItsCleanup() async throws {
    let driver = ResetFixtureDriver(mode: .blockExecute), runner = driver.runner(), context = try driver.context()
    var program = driver.program(hold: true); program.resetPlan?.maximumDurationMS = 1_000
    let flag = ResetResultFlag()
    let task = Task { defer { flag.finish() }; return try await runner.run(context: context, program: program) }
    try await waitForReset { driver.executing == 1 }
    driver.clock.advance(2_000_000_000)
    try await waitForReset { driver.stops > 0 }
    #expect(!flag.finished && driver.released == 0)
    driver.gate.open()
    let result = try await task.value
    #expect(result.status == .failed && result.issue?.code == "reset.timeout" && result.cleanupConfirmed && driver.held.isEmpty)
}

@Test func resetRunnerCallerCancellationDoesNotAbandonAnOwnedRelease() async throws {
    let driver = ResetFixtureDriver(mode: .blockRelease), runner = driver.runner(), context = try driver.context()
    let flag = ResetResultFlag()
    let task = Task { defer { flag.finish() }; return try await runner.run(context: context, program: driver.program(hold: true)) }
    try await waitForReset { driver.released > 0 }
    task.cancel()
    try await waitForReset { driver.stops > 0 }
    #expect(!flag.finished && driver.held == [36])
    driver.gate.open()
    let result = try await task.value
    #expect(result.status == .cancelled && result.cleanupConfirmed && driver.held.isEmpty)
}
