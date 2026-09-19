import Foundation
import AstraCore

public struct ResetBinding: Sendable {
    public let context: ResetContext
    public let priorOwnersJoined: Bool
    public let verifiedAtNanos: UInt64
    public init(context: ResetContext, priorOwnersJoined: Bool, verifiedAtNanos: UInt64) {
        self.context = context; self.priorOwnersJoined = priorOwnersJoined; self.verifiedAtNanos = verifiedAtNanos
    }
}

public struct ResetObservationSnapshot: Sendable {
    public let id: UUID
    public let context: ResetContext
    public let observedNanos: UInt64
    public let sourceCoverage: [RewardSourceCoverage]
    public let readings: [SignalReading]
    public init(id: UUID = UUID(), context: ResetContext, observedNanos: UInt64,
                sourceCoverage: [RewardSourceCoverage], readings: [SignalReading]) {
        self.id = id; self.context = context; self.observedNanos = observedNanos
        self.sourceCoverage = sourceCoverage; self.readings = readings
    }
}

public struct ResetReleaseProof: Sendable {
    public let resetID: UUID
    public let observedNanos: UInt64
    public let confirmed: Bool
    public let issue: String?
    public init(resetID: UUID, observedNanos: UInt64, confirmed: Bool, issue: String? = nil) {
        self.resetID = resetID; self.observedNanos = observedNanos; self.confirmed = confirmed; self.issue = issue
    }
}

/// The orchestration owner binds these operations to the actual scoped helper
/// and observation producer. Empty capabilities must not arm a controller.
/// Stop closes admission synchronously, cancels pending operation waits, and
/// keeps paired cleanup alive. Release joins every possible post before returning
/// confirmed, or reports unconfirmed dead owners; it never times out as success.
public protocol ResetControlDriver: Sendable {
    func prepare(context: ResetContext, capabilities: ActionCapabilities) async throws -> ResetBinding
    func observe(context: ResetContext) async throws -> ResetObservationSnapshot
    func execute(_ packet: ActionPacket, context: ResetContext) async throws -> ExecutionReceipt
    func checkHealth(resetID: UUID) throws
    func requestStop(resetID: UUID)
    func release(context: ResetContext) async -> ResetReleaseProof
}

public enum ResetPhase: String, Sendable { case awaitingManualReady, preparing, executing, pausing, waitingCondition, releasing, ready, failed, cancelled }
public struct ResetProgress: Sendable {
    public let resetID: UUID
    public let phase: ResetPhase
    public let attempt: Int
    public let stepIndex: Int?
    public let message: String
}
public enum ResetStatus: String, Sendable { case ready, failed, cancelled }
public struct ResetResult: Sendable {
    public let context: ResetContext
    public let status: ResetStatus
    public let attempts: Int
    public let cleanup: ResetReleaseProof
    public let readyObservation: ResetObservationSnapshot?
    public let readySignals: ResolvedRewardSnapshot?
    public let issue: AstraError?
    public var cleanupConfirmed: Bool { cleanup.resetID == context.resetID && cleanup.confirmed }
}

private final class ResetRunState: @unchecked Sendable {
    private let lock = NSLock()
    let context: ResetContext
    let manual: Bool
    let globalDeadline: UInt64?
    private let stop: @Sendable () -> Void
    private var cancelled = false, finished = false
    private var failure: AstraError?
    private var manualAt: UInt64?
    private var awaitingManual = false
    private var healthEnabled = false
    private var operationID: UUID?
    private var operationDeadline: UInt64?
    private var operationError: AstraError?
    private var cancelOperation: (@Sendable () -> Void)?
    init(context: ResetContext, manual: Bool, globalDeadline: UInt64?, stop: @escaping @Sendable () -> Void) {
        self.context = context; self.manual = manual; self.globalDeadline = globalDeadline; self.stop = stop
    }
    var isCancelled: Bool { lock.withLock { cancelled } }
    var error: AstraError? { lock.withLock { cancelled ? .init("reset.cancelled", "Reset cancelled.") : failure } }
    var manualReadyAt: UInt64? { lock.withLock { manualAt } }
    var checksHealth: Bool { lock.withLock { healthEnabled && !finished && !cancelled && failure == nil } }
    func setHealth(_ enabled: Bool) { lock.withLock { healthEnabled = enabled } }
    func beginManualWait() { lock.withLock { awaitingManual = true } }
    func confirm(at now: UInt64) -> Bool {
        lock.withLock {
            guard manual, awaitingManual, !cancelled, !finished, failure == nil, manualAt == nil else { return false }
            manualAt = now; awaitingManual = false; return true
        }
    }
    func cancel() -> Bool {
        let callback: (@Sendable () -> Void)? = lock.withLock {
            guard !finished, !cancelled else { return nil }
            cancelled = true; return cancelOperation ?? {}
        }
        guard let callback else { return false }
        stop(); callback(); return true
    }
    func fail(_ error: AstraError, operation: UUID? = nil) {
        let callback: (@Sendable () -> Void)? = lock.withLock {
            guard !finished, !cancelled, failure == nil, operation == nil || operationID == operation else { return nil }
            failure = error; return cancelOperation ?? {}
        }
        if let callback { stop(); callback() }
    }
    func poll(now: UInt64) {
        if let globalDeadline, now >= globalDeadline { fail(.init("reset.timeout", "The reset exceeded its total time limit.")); return }
        let item = lock.withLock { (operationID, operationDeadline, operationError) }
        if let id = item.0, let deadline = item.1, now >= deadline, let error = item.2 { fail(error, operation: id) }
    }
    func check(now: UInt64) throws { poll(now: now); if let error { throw error } }
    func beginAttempt(now: UInt64) throws {
        lock.withLock { if failure?.code == "reset.conditionTimeout" { failure = nil } }
        try check(now: now)
    }
    func installOperation(id: UUID, deadline: UInt64?, error: AstraError, cancel: @escaping @Sendable () -> Void) {
        let rejected = lock.withLock {
            guard !finished, !cancelled, failure == nil else { return true }
            operationID = id; operationDeadline = deadline; operationError = error; cancelOperation = cancel; return false
        }
        if rejected { cancel() }
    }
    func endOperation(_ id: UUID) {
        lock.withLock {
            if operationID == id { operationID = nil; operationDeadline = nil; operationError = nil; cancelOperation = nil }
        }
    }
    func finish() { lock.withLock { finished = true; awaitingManual = false; cancelOperation = nil; operationID = nil } }
}

/// Executes author-defined control only between policy episodes. The runner
/// owns cancellation/deadline state, never a second input backend or OS lease.
public final class ResetRunner: @unchecked Sendable {
    private let driver: any ResetControlDriver
    private let now: @Sendable () -> UInt64
    private let sleep: @Sendable (UInt64) async throws -> Void
    private let progress: @Sendable (ResetProgress) -> Void
    private let lock = NSLock()
    private var active: ResetRunState?
    public static let maximumSnapshotAgeNanos: UInt64 = 500_000_000
    public init(driver: any ResetControlDriver, clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now },
                sleep: @escaping @Sendable (UInt64) async throws -> Void = { try await Task.sleep(nanoseconds: $0) },
                progress: @escaping @Sendable (ResetProgress) -> Void = { _ in }) {
        self.driver = driver; now = clock; self.sleep = sleep; self.progress = progress
    }
    @discardableResult public func confirmManualReady(resetID: UUID) -> Bool {
        guard let state = lock.withLock({ active }), state.context.resetID == resetID else { return false }
        return state.confirm(at: now())
    }
    @discardableResult public func cancel(resetID: UUID) -> Bool {
        guard let state = lock.withLock({ active }), state.context.resetID == resetID else { return false }
        return state.cancel()
    }
    public func run(context: ResetContext, program: RewardProgram) async throws -> ResetResult {
        let program = try program.validated()
        let evaluator = try RewardEvaluator(program: program)
        if let plan = program.resetPlan {
            for step in plan.steps { if let packet = step.packet { _ = try packet.packet(context: context, observationID: UUID(), sequence: 0, executeAtNanos: 0) } }
        }
        let started = now(), deadline = try program.resetPlan.map { try adding(started, milliseconds: $0.maximumDurationMS) }
        let driver = driver
        let state = ResetRunState(context: context, manual: program.resetPlan == nil, globalDeadline: deadline,
                                  stop: { driver.requestStop(resetID: context.resetID) })
        try lock.withLock {
            guard active == nil else { throw AstraError("reset.busy", "A reset is already running or joining cleanup.") }
            active = state
        }
        let watchdog = Task { [now] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .milliseconds(20)) } catch { return }
                state.poll(now: now())
                if state.checksHealth {
                    do { try driver.checkHealth(resetID: context.resetID) }
                    catch { state.fail((error as? AstraError) ?? .init("reset.health", error.localizedDescription)) }
                }
            }
        }
        let result = await withTaskCancellationHandler {
            await perform(context: context, program: program, evaluator: evaluator, state: state)
        } onCancel: { _ = state.cancel() }
        state.finish(); watchdog.cancel(); await watchdog.value
        lock.withLock { if active === state { active = nil } }
        return result
    }

    private func perform(context: ResetContext, program: RewardProgram, evaluator: RewardEvaluator, state: ResetRunState) async -> ResetResult {
        var attempt = 0
        var release = ResetReleaseProof(resetID: context.resetID, observedNanos: now(), confirmed: false)
        var releasedForAttempt = false
        do {
            var firstBarrier = now()
            if state.manual {
                state.beginManualWait(); emit(state, .awaitingManualReady, attempt: 0, "Reset the environment, then confirm Ready.")
                while state.manualReadyAt == nil {
                    try state.check(now: now())
                    try await operation(state, deadline: nil, error: .init("reset.cancelled", "Reset cancelled.")) { [sleep] in try await sleep(50_000_000) }
                }
                firstBarrier = state.manualReadyAt!
            }
            let maximumAttempts = program.resetPlan?.maximumAttempts ?? 1
            for number in 1...maximumAttempts {
                attempt = number; try state.beginAttempt(now: now())
                do {
                    releasedForAttempt = false
                    emit(state, .preparing, attempt: attempt, "Verifying the current environment and previous cleanup.")
                    let prepareDeadline = try adding(now(), milliseconds: 5_000)
                    let binding = try await operation(state, deadline: prepareDeadline, error: .init("reset.prepareTimeout", "Reset preparation timed out.")) { [driver] in
                        try await driver.prepare(context: context, capabilities: program.resetPlan?.capabilities ?? .init())
                    }
                    guard binding.context == context, binding.priorOwnersJoined, binding.verifiedAtNanos <= now(),
                          now() - binding.verifiedAtNanos <= Self.maximumSnapshotAgeNanos else {
                        throw AstraError("reset.binding", "Reset control did not verify the same environment and joined prior owners.")
                    }
                    state.setHealth(true)
                    var barrier = max(firstBarrier, binding.verifiedAtNanos), sequence: UInt64 = 0
                    if let plan = program.resetPlan {
                        for (index, step) in plan.steps.enumerated() {
                            try check(state)
                            switch step.kind {
                            case .packet:
                                emit(state, .executing, attempt: attempt, step: index, step.name)
                                let observed = try await freshObservation(context: context, state: state, minimum: barrier,
                                    deadline: adding(now(), milliseconds: 5_000), timeout: .init("reset.observationTimeout", "No fresh scoped observation arrived for reset control."))
                                let packet = try step.packet!.packet(context: context, observationID: observed.id, sequence: sequence,
                                                                    executeAtNanos: adding(now(), milliseconds: 50))
                                let receipt = try await operation(state, deadline: adding(packet.executeAtNanos, milliseconds: packet.durationMs + 2_000),
                                    error: .init("reset.receiptTimeout", "The reset packet did not complete in time.")) { [driver] in
                                    try await driver.execute(packet, context: context)
                                }
                                try validate(receipt, packet: packet)
                                sequence += 1; barrier = receipt.observedNanos
                            case .pause:
                                emit(state, .pausing, attempt: attempt, step: index, step.name)
                                let end = try adding(now(), milliseconds: step.pauseMS!)
                                while true {
                                    try check(state)
                                    let current = now()
                                    guard current < end else { break }
                                    let duration = min(50_000_000, end - current)
                                    try await operation(state, deadline: nil, error: .init("reset.cancelled", "Reset cancelled.")) { [sleep] in try await sleep(duration) }
                                }
                                barrier = now()
                            case .wait:
                                _ = try await waitForCondition(step.condition!, evaluator: evaluator, context: context, state: state,
                                    minimum: barrier, deadline: adding(now(), milliseconds: step.timeoutMS!), attempt: attempt, step: index, name: step.name)
                                barrier = now()
                            }
                        }
                    }
                    emit(state, .releasing, attempt: attempt, "Waiting for owned controls and pending posts to release.")
                    state.setHealth(false); release = await releaseOwned(context: context)
                    releasedForAttempt = true
                    try validate(release, context: context)
                    try state.check(now: now())
                    let finalDeadline = try adding(now(), milliseconds: program.resetPlan?.readinessTimeoutMS ?? 30_000)
                    let minimum = max(barrier, release.observedNanos)
                    let ready: (ResetObservationSnapshot, ResolvedRewardSnapshot)
                    if let predicate = program.ready {
                        ready = try await waitForCondition(predicate, evaluator: evaluator, context: context, state: state,
                            minimum: minimum, deadline: finalDeadline, attempt: attempt, name: "Waiting for the starting condition.", checkControl: false)
                    } else {
                        let observed = try await freshObservation(context: context, state: state, minimum: minimum,
                            deadline: finalDeadline, timeout: .init("reset.conditionTimeout", "The starting observation was not verified in time."), checkControl: false)
                        ready = (observed, try resolved(observed, evaluator: evaluator, minimum: minimum))
                    }
                    try state.check(now: now())
                    emit(state, .ready, attempt: attempt, "Ready for a new policy episode.")
                    return .init(context: context, status: .ready, attempts: attempt, cleanup: release,
                                 readyObservation: ready.0, readySignals: ready.1, issue: nil)
                } catch {
                    let failure = state.error ?? (error as? AstraError) ?? .init("reset.operation", error.localizedDescription)
                    driver.requestStop(resetID: context.resetID)
                    emit(state, .releasing, attempt: attempt, "Stopping reset; waiting for owned controls to release.")
                    state.setHealth(false)
                    if !releasedForAttempt { release = await releaseOwned(context: context); releasedForAttempt = true }
                    try validate(release, context: context)
                    guard !state.isCancelled, failure.code == "reset.conditionTimeout", attempt < maximumAttempts else { throw failure }
                    firstBarrier = release.observedNanos
                }
            }
            throw AstraError("reset.attempts", "The reset exhausted its authored attempts.")
        } catch {
            driver.requestStop(resetID: context.resetID)
            emit(state, .releasing, attempt: attempt, "Waiting for owned controls and pending posts to release.")
            state.setHealth(false)
            if !releasedForAttempt { release = await releaseOwned(context: context) }
            let failure = state.error ?? (error as? AstraError) ?? .init("reset.operation", error.localizedDescription)
            let status: ResetStatus = state.isCancelled ? .cancelled : .failed
            emit(state, status == .cancelled ? .cancelled : .failed, attempt: attempt,
                 release.confirmed && release.resetID == context.resetID ? failure.message : "Reset stopped, but owned input cleanup is unconfirmed.")
            return .init(context: context, status: status, attempts: attempt, cleanup: release, readyObservation: nil, readySignals: nil, issue: failure)
        }
    }

    private func check(_ state: ResetRunState) throws { try state.check(now: now()); try driver.checkHealth(resetID: state.context.resetID) }
    private func operation<T: Sendable>(_ state: ResetRunState, deadline: UInt64?, error: AstraError,
                                       body: @escaping @Sendable () async throws -> T) async throws -> T {
        try state.check(now: now())
        let id = UUID()
        let work = Task { try state.check(now: now()); return try await body() }
        state.installOperation(id: id, deadline: deadline, error: error, cancel: { work.cancel() })
        defer { state.endOperation(id) }
        do { let value = try await work.value; try state.check(now: now()); return value }
        catch { throw state.error ?? error }
    }
    private func freshObservation(context: ResetContext, state: ResetRunState, minimum: UInt64, deadline: UInt64,
                                  timeout: AstraError, checkControl: Bool = true) async throws -> ResetObservationSnapshot {
        while true {
            if checkControl { try check(state) } else { try state.check(now: now()) }
            guard now() < deadline else { state.fail(timeout); throw timeout }
            let observed = try await operation(state, deadline: deadline, error: timeout) { [driver] in try await driver.observe(context: context) }
            guard observed.context == context, observed.observedNanos <= now(), observed.sourceCoverage.count == context.scope.surfaces.count,
                  Set(observed.sourceCoverage.map { $0.surface.id }).count == observed.sourceCoverage.count,
                  Set(observed.sourceCoverage.map(\.sourceObservationID)).count == observed.sourceCoverage.count else {
                throw AstraError("reset.observation", "The reset observation belongs to another environment, geometry or clock.")
            }
            for source in observed.sourceCoverage { _ = try source.validated(scope: context.scope, cutoffNanos: observed.observedNanos) }
            if now() - observed.observedNanos <= Self.maximumSnapshotAgeNanos,
               observed.observedNanos >= minimum, observed.sourceCoverage.allSatisfy({ $0.throughNanos >= minimum }) { return observed }
            try await operation(state, deadline: deadline, error: timeout) { [sleep] in try await sleep(50_000_000) }
        }
    }
    private func resolved(_ observation: ResetObservationSnapshot, evaluator: RewardEvaluator, minimum: UInt64) throws -> ResolvedRewardSnapshot {
        let sources = Dictionary(uniqueKeysWithValues: observation.sourceCoverage.map { ($0.sourceObservationID, $0) })
        let coverage = observation.readings.compactMap { reading -> RewardReadingCoverage? in
            guard let id = reading.sourceObservationID, let source = sources[id] else { return nil }
            return .init(signalID: reading.signalID, source: source)
        }
        return try evaluator.resolveSnapshot(episodeID: observation.context.nextEpisodeID, cutoffNanos: observation.observedNanos,
            readings: observation.readings, coverage: coverage, scope: observation.context.scope, minimumEvidenceNanos: minimum)
    }
    private func waitForCondition(_ predicate: RewardPredicate, evaluator: RewardEvaluator, context: ResetContext, state: ResetRunState,
                                  minimum: UInt64, deadline: UInt64, attempt: Int, step: Int? = nil, name: String,
                                  checkControl: Bool = true) async throws -> (ResetObservationSnapshot, ResolvedRewardSnapshot) {
        let timeout = AstraError("reset.conditionTimeout", "\(name) did not become true before its timeout.")
        var previous: RewardTruth?
        while true {
            let observation = try await freshObservation(context: context, state: state, minimum: minimum, deadline: deadline, timeout: timeout, checkControl: checkControl)
            let values = try resolved(observation, evaluator: evaluator, minimum: minimum)
            let truth = try evaluator.predicate(predicate, snapshot: values)
            if truth == .yes { return (observation, values) }
            if previous != truth {
                emit(state, .waitingCondition, attempt: attempt, step: step, truth == .unknown ? "\(name) is not currently readable." : name)
                previous = truth
            }
            try await operation(state, deadline: deadline, error: timeout) { [sleep] in try await sleep(50_000_000) }
        }
    }
    private func validate(_ proof: ResetReleaseProof, context: ResetContext) throws {
        guard proof.resetID == context.resetID, proof.confirmed, proof.observedNanos <= now() else {
            throw AstraError("reset.cleanupUnconfirmed", proof.issue ?? "Reset input cleanup could not be confirmed. Verify held controls before continuing.")
        }
    }
    private func releaseOwned(context: ResetContext) async -> ResetReleaseProof {
        // Joining owned input release is not cancelled with the policy/reset
        // caller. This task is always awaited and never abandoned on timeout.
        let driver = driver
        let work = Task { await driver.release(context: context) }
        let proof = await work.value
        guard proof.resetID == context.resetID, proof.observedNanos <= now() else {
            return .init(resetID: proof.resetID, observedNanos: proof.observedNanos, confirmed: false,
                         issue: "Cleanup returned an invalid reset identity or future timestamp.")
        }
        return proof
    }
    private func validate(_ receipt: ExecutionReceipt, packet: ActionPacket) throws {
        guard receipt.runID == packet.runID, receipt.packetID == packet.id, receipt.sequence == packet.sequence,
              receipt.status == .executed, receipt.observedNanos <= now(), receipt.observedNanos >= packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000,
              receipt.resultingState.pointer.isFinite, receipt.commandResults.count == packet.commands.count,
              Set(receipt.commandResults.map(\.commandIndex)) == Set(packet.commands.indices), receipt.commandResults.allSatisfy({ result in
                  guard packet.commands.indices.contains(result.commandIndex) else { return false }
                  let time = packet.executeAtNanos + UInt64(packet.commands[result.commandIndex].offsetMs) * 1_000_000
                  return result.scheduledNanos == time && (result.status == .posted || result.status == .noOp)
                    && (result.status != .posted || result.postedNanos != nil)
                    && (result.postedNanos.map { $0 >= time && $0 <= receipt.observedNanos } ?? true)
              }) else { throw AstraError("reset.receipt", "A reset packet was rejected, late, incomplete or did not match its execution receipt.") }
    }
    private func adding(_ nanos: UInt64, milliseconds: Int) throws -> UInt64 {
        let sum = nanos.addingReportingOverflow(UInt64(milliseconds) * 1_000_000)
        guard !sum.overflow else { throw AstraError("reset.clock", "The reset deadline exceeds the monotonic clock range.") }
        return sum.partialValue
    }
    private func emit(_ state: ResetRunState, _ phase: ResetPhase, attempt: Int, step: Int? = nil, _ message: String) {
        progress(.init(resetID: state.context.resetID, phase: phase, attempt: attempt, stepIndex: step, message: message))
    }
}
