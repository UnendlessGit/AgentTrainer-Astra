import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

final class ControlTestGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var waits: [CheckedContinuation<Void, Never>] = []
    func wait() async { await withCheckedContinuation { continuation in
        let immediate = lock.withLock { if opened { return true }; waits.append(continuation); return false }
        if immediate { continuation.resume() }
    } }
    func open() { let pending = lock.withLock { opened = true; let result = waits; waits = []; return result }; pending.forEach { $0.resume() } }
}
final class ControlTestFlag: @unchecked Sendable {
    private let lock = NSLock(); private var stored = false
    var value: Bool { lock.withLock { stored } }
    func set() { lock.withLock { stored = true } }
}

final class ControlRuntimeFixture: @unchecked Sendable {
    let armGate = ControlTestGate(), heartbeatGate = ControlTestGate(), shutdownGate = ControlTestGate(), executeGate = ControlTestGate()
    private let lock = NSLock()
    private var events: ComputeProcess.EventHandler?
    private var failure: ComputeProcess.FailureHandler?
    private var activeRun: UUID?
    private var outputSequence: UInt64 = 1
    private var pending: [UUID: (ActionPacket, UUID)] = [:]
    private var operations: [String] = []
    private var starts = 0, heartbeats = 0, cancellations = 0
    private var forwarded: [UUID] = []
    private var joined = false
    let blockArm: Bool, blockHeartbeat: Bool, blockShutdown: Bool, earlyTerminal: Bool, blockExecute: Bool, rejectOrdinary: Bool, exitStatus: Int32
    var wrongOrigin = false
    init(blockArm: Bool = false, blockHeartbeat: Bool = false, blockShutdown: Bool = false, earlyTerminal: Bool = false, blockExecute: Bool = false, rejectOrdinary: Bool = false, exitStatus: Int32 = 0) {
        self.blockArm = blockArm; self.blockHeartbeat = blockHeartbeat; self.blockShutdown = blockShutdown
        self.earlyTerminal = earlyTerminal; self.blockExecute = blockExecute; self.rejectOrdinary = rejectOrdinary; self.exitStatus = exitStatus
    }
    var log: [String] { lock.withLock { operations } }
    var startCount: Int { lock.withLock { starts } }
    var heartbeatCount: Int { lock.withLock { heartbeats } }
    var cancelledRequests: Int { lock.withLock { cancellations } }
    var sentPackets: [UUID] { lock.withLock { forwarded } }
    var hasJoined: Bool { lock.withLock { joined } }
    var armed: Bool { lock.withLock { activeRun != nil } }
    var factory: NativeControlRuntimeFactory {
        .init(protectsPhysicalInputs: false) { events, failure in
            self.lock.withLock { self.events = events; self.failure = failure }
            return .init(start: {
                self.lock.withLock { self.starts += 1 }
                return WireMessage(kind: "hello", sequence: 0, payload: .object(["role": .string("control"), "protocolVersion": .integer(1),
                    "recoveryVersion": .integer(1), "initialPacketSequenceVersion": .integer(1)]))
            }, request: { kind, payload, run, _ in
                try await withTaskCancellationHandler { try await self.request(kind, payload, run) } onCancel: { self.lock.withLock { self.cancellations += 1 } }
            }, shutdown: {
                self.lock.withLock { self.operations.append("shutdown") }
                if self.blockShutdown { await self.shutdownGate.wait() }
                let identifiers = self.lock.withLock { Array(self.pending.keys) }
                for id in identifiers { self.complete(id, status: .cancelled) }
                self.lock.withLock { self.activeRun = nil; self.joined = true }
                return self.exitStatus
            })
        }
    }
    private func request(_ kind: String, _ payload: JSONValue, _ run: UUID) async throws -> WireMessage {
        lock.withLock { operations.append(kind) }
        let request = UUID()
        var value: JSONValue = .object([:]); var reply = "ack"
        switch kind {
        case "arm":
            let arm = try payload.decode(ArmRequest.self)
            if blockArm { await armGate.wait() }
            lock.withLock { activeRun = run }
            value = .object(["armed": .bool(true), "nextPacketSequence": .unsigned((arm.initialPacketSequence ?? 0) + (wrongOrigin ? 1 : 0))])
        case "heartbeat":
            lock.withLock { heartbeats += 1 }
            if blockHeartbeat { await heartbeatGate.wait() }
            value = .object(["alive": .bool(true)])
        case "execute":
            let packet = try payload.decode(ActionPacket.self)
            lock.withLock { forwarded.append(packet.id) }
            if blockExecute { await executeGate.wait() }
            let live = lock.withLock { pending[packet.id] = (packet, request); return activeRun == run && !rejectOrdinary }
            if live {
                sendReceipt(packet, request: request, status: .admitted)
                if earlyTerminal { try await Task.sleep(for: .milliseconds(3)); complete(packet.id) }
                value = .object(["admitted": .bool(true)])
            } else {
                complete(packet.id, status: .rejected); reply = "error"
                value = .object(["code": .string("control.session"), "message": .string("The helper is disarmed."), "recoverable": .bool(true)])
            }
        case "disarm":
            lock.withLock { activeRun = nil }
            let identifiers = lock.withLock { Array(pending.keys) }
            for id in identifiers { complete(id, status: .cancelled) }
            value = .object(["stopped": .bool(true), "cleanupSettled": .bool(true)])
        case "observation":
            var state = ControlState(); state.valid = true; state.observedNanos = MonotonicClock.now
            value = try .encode(ControlObservation(controlState: state, executedEvents: [], intervalCovered: true, cutoffNanos: state.observedNanos, lastSequence: nil))
        default: break
        }
        return WireMessage(kind: reply, sequence: 0, requestID: request, runID: run, payload: value)
    }
    func complete(_ id: UUID, status: ReceiptStatus = .executed) {
        guard let value = lock.withLock({ pending.removeValue(forKey: id) }) else { return }
        sendReceipt(value.0, request: value.1, status: status)
    }
    private func sendReceipt(_ packet: ActionPacket, request: UUID, status: ReceiptStatus) {
        var state = ControlState(); state.valid = true; state.observedNanos = MonotonicClock.now
        let results: [CommandResult] = status == .executed ? packet.commands.enumerated().map { index, command in
            let time = packet.executeAtNanos + UInt64(command.offsetMs) * 1_000_000
            return CommandResult(commandIndex: index, scheduledNanos: time, postedNanos: time, status: .posted)
        } : []
        let receipt = ExecutionReceipt(packet: packet, status: status, observedNanos: max(MonotonicClock.now, packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000),
            commandResults: results, resultingState: state)
        let metadata = lock.withLock { let sequence = outputSequence; outputSequence += 1; return (events, sequence) }
        metadata.0?(WireMessage(kind: "control.receipt", sequence: metadata.1, requestID: request, runID: packet.runID,
                               payload: .object(["receipt": try! .encode(receipt)])))
    }
    func lateFailure() { lock.withLock { failure }?(AstraError("fixture.late", "A closed instance emitted a late callback.")) }
}

func controlConfiguration(root: URL, run: UUID = UUID(), origin: UInt64 = 0) throws -> NativeControlConfiguration {
    try .init(runID: run, scope: .init(surfaces: [.init(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 10, height: 10), pixelWidth: 10, pixelHeight: 10)], wholeDesktop: true),
              capabilities: .init(keyCodes: [36]), initialPacketSequence: origin, recoveryDirectory: root)
}
func controlPacket(_ config: NativeControlConfiguration, sequence: UInt64? = nil) -> ActionPacket {
    .init(runID: config.runID, sequence: sequence ?? config.initialPacketSequence, observationID: UUID(), geometryRevision: 0,
          executeAtNanos: MonotonicClock.now, durationMs: 1, commands: [.init(offsetMs: 0, operation: .keyDown, keyCode: 36)])
}
func controlWait(_ condition: () -> Bool) async throws {
    let until = ContinuousClock.now.advanced(by: .seconds(3))
    while !condition() { guard ContinuousClock.now < until else { throw AstraError("fixture.timeout", "Native control fixture timed out.") }; try await Task.sleep(for: .milliseconds(2)) }
}

@Test func nativeControlAdmissionReturnsBeforeTerminalAndHandlesEarlyReceipts() async throws {
    for early in [false, true] {
        let fixture = ControlRuntimeFixture(earlyTerminal: early), owner = NativeControlOwner()
        let config = try controlConfiguration(root: FileManager.default.temporaryDirectory, origin: 917)
        let session = NativeControlSession(configuration: config, owner: owner, runtimeFactory: fixture.factory)
        _ = try await session.start()
        let packet = controlPacket(config), submission = try await session.submit(packet)
        #expect(submission.admitted)
        let flag = ControlTestFlag()
        let terminal = Task { defer { flag.set() }; return try await submission.terminalReceipt() }
        if !early { #expect(!flag.value); fixture.complete(packet.id) }
        #expect(try await terminal.value.status == .executed)
        #expect(await session.shutdown().cleanupConfirmed && owner.priorCleanupJoined)
    }
}

@Test func nativeControlDisarmKeepsTheHelperForRealLateRejectionAndDrainsHeartbeat() async throws {
    let fixture = ControlRuntimeFixture(blockHeartbeat: true), owner = NativeControlOwner()
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let session = NativeControlSession(configuration: config, owner: owner, runtimeFactory: fixture.factory)
    _ = try await session.start(); try await controlWait { fixture.heartbeatCount == 1 }
    session.requestStop(); _ = try await session.disarm()
    #expect(!fixture.armed && !fixture.hasJoined)
    let packet = controlPacket(config)
    let rejection = try await session.rejectLatePacket(packet)
    #expect(!rejection.admitted && rejection.reply.kind == "error")
    #expect(try await rejection.terminalReceipt().status == .rejected)
    #expect(fixture.sentPackets == [packet.id])
    let flag = ControlTestFlag()
    let closing = Task { defer { flag.set() }; return await session.shutdown() }
    try await Task.sleep(for: .milliseconds(10))
    #expect(!flag.value && fixture.cancelledRequests == 0)
    fixture.heartbeatGate.open()
    #expect(await closing.value.cleanupConfirmed && fixture.hasJoined)
}

@Test func nativeControlStopDuringArmRepeatsDisarmAfterTheLateArmReply() async throws {
    let fixture = ControlRuntimeFixture(blockArm: true), owner = NativeControlOwner()
    let session = NativeControlSession(configuration: try controlConfiguration(root: FileManager.default.temporaryDirectory), owner: owner, runtimeFactory: fixture.factory)
    let starting = Task { try await session.start() }
    try await controlWait { fixture.log.contains("arm") }
    session.requestStop(); try await controlWait { fixture.log.contains("disarm") }
    fixture.armGate.open()
    do { _ = try await starting.value; Issue.record("Stopped arming unexpectedly succeeded") } catch {}
    #expect(await session.shutdown().cleanupConfirmed && !fixture.armed)
    #expect(fixture.log.filter { $0 == "disarm" }.count >= 2 && fixture.sentPackets.isEmpty)
}

@Test func nativeControlOwnerPreventsNewHelperUntilJoinedAndDropsOldInstanceCallbacks() async throws {
    let first = ControlRuntimeFixture(blockShutdown: true), second = ControlRuntimeFixture(), owner = NativeControlOwner()
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let old = NativeControlSession(configuration: config, owner: owner, runtimeFactory: first.factory)
    _ = try await old.start(); _ = try await old.disarm()
    let closing = Task { await old.shutdown() }
    try await controlWait { first.log.contains("shutdown") }
    let rejected = NativeControlSession(configuration: config, owner: owner, runtimeFactory: second.factory)
    do { _ = try await rejected.start(); Issue.record("A helper started before its predecessor joined") } catch {}
    #expect(second.startCount == 0)
    _ = await rejected.shutdown()
    first.shutdownGate.open(); #expect(await closing.value.cleanupConfirmed)
    let next = NativeControlSession(configuration: config, owner: owner, runtimeFactory: second.factory)
    _ = try await next.start(); first.lateFailure(); try next.checkHealth()
    #expect(old.sessionID != next.sessionID)
    _ = await next.shutdown()
}

@Test func nativeControlThrowingTerminalAuditStillCompletesItsRemovedPromise() async throws {
    let fixture = ControlRuntimeFixture(), owner = NativeControlOwner()
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let session = NativeControlSession(configuration: config, owner: owner, runtimeFactory: fixture.factory, onEvent: { event in
        let receipt = try event.message.payload.decode([String: ExecutionReceipt].self)["receipt"]
        if receipt?.status == .executed { throw AstraError("fixture.backpressure", "Audit queue is full.") }
    })
    _ = try await session.start()
    let packet = controlPacket(config), submission = try await session.submit(packet)
    let flag = ControlTestFlag()
    let terminal = Task { defer { flag.set() }; return try await submission.terminalReceipt() }
    fixture.complete(packet.id)
    try await controlWait { flag.value }
    do { _ = try await terminal.value; Issue.record("Audit failure was not propagated") }
    catch { #expect((error as? AstraError)?.code == "fixture.backpressure") }
    #expect(await session.shutdown().cleanupConfirmed)
    #expect(session.executedPacketCount == 1)
}

@Test func nativeControlUnconfirmedCleanupBlocksSuccessorUntilExplicitAcknowledgement() async throws {
    let fixture = ControlRuntimeFixture(exitStatus: 7), owner = NativeControlOwner()
    let session = NativeControlSession(configuration: try controlConfiguration(root: FileManager.default.temporaryDirectory), owner: owner, runtimeFactory: fixture.factory)
    _ = try await session.start()
    let done = await session.shutdown()
    #expect(!done.cleanupConfirmed && !owner.priorCleanupJoined)
    try owner.acknowledgeManualCleanup(sessionID: done.sessionID)
    #expect(owner.priorCleanupJoined && !done.cleanupConfirmed)
}

@Test func nativeControlWrongInitialSequenceFailsBeforeAnyPacket() async throws {
    let fixture = ControlRuntimeFixture(); fixture.wrongOrigin = true
    let session = NativeControlSession(configuration: try controlConfiguration(root: FileManager.default.temporaryDirectory, origin: 917), owner: NativeControlOwner(), runtimeFactory: fixture.factory)
    do { _ = try await session.start(); Issue.record("Wrong arm origin was accepted") } catch {}
    #expect(fixture.sentPackets.isEmpty)
    _ = await session.shutdown()
}

@Test func nativeControlReservedSubmissionAndCallerCancellationNeverResendThePacket() async throws {
    let fixture = ControlRuntimeFixture(blockExecute: true), owner = NativeControlOwner()
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let session = NativeControlSession(configuration: config, owner: owner, runtimeFactory: fixture.factory)
    _ = try await session.start()
    let first = controlPacket(config), second = controlPacket(config, sequence: 1)
    let sending = Task { try await session.submit(first) }
    try await controlWait { fixture.sentPackets == [first.id] }
    let queued = Task { try await session.submit(second) }
    // Let the next call reserve behind the blocked, already-forwarded request.
    try await Task.sleep(for: .milliseconds(20))
    sending.cancel(); _ = try await session.disarm()
    fixture.executeGate.open()
    let replies = try await [sending.value, queued.value]
    for reply in replies { #expect(!reply.admitted); #expect(try await reply.terminalReceipt().status == .rejected) }
    #expect(fixture.sentPackets == [first.id, second.id] && fixture.cancelledRequests == 0)
    #expect(await session.shutdown().cleanupConfirmed)
}

@Test func nativeControlUnexpectedNonAdmissionStopsCountersWithoutLosingRejectionEvidence() async throws {
    let fixture = ControlRuntimeFixture(rejectOrdinary: true)
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let session = NativeControlSession(configuration: config, owner: NativeControlOwner(), runtimeFactory: fixture.factory)
    _ = try await session.start()
    let result = try await session.submit(controlPacket(config))
    let receipt = try await result.terminalReceipt()
    #expect(!result.admitted && receipt.status == .rejected)
    #expect(throws: AstraError.self) { try session.checkHealth() }
    await #expect(throws: AstraError.self) { try await session.submit(controlPacket(config, sequence: 1)) }
    #expect(await session.shutdown().cleanupConfirmed && fixture.sentPackets.count == 1)
}

@Test func nativeControlRejectsTerminalEvidenceThatDisagreesWithAdmission() async throws {
    let fixture = ControlRuntimeFixture()
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let session = NativeControlSession(configuration: config, owner: NativeControlOwner(), runtimeFactory: fixture.factory)
    _ = try await session.start()
    let packet = controlPacket(config), result = try await session.submit(packet)
    fixture.complete(packet.id, status: .rejected)
    await #expect(throws: AstraError.self) { try await result.terminalReceipt() }
    #expect(throws: AstraError.self) { try session.checkHealth() }
    _ = await session.shutdown()
}

@Test func nativeControlMissingTerminalReceiptFailsAndStopsWithoutAbandoningTheHelper() async throws {
    let fixture = ControlRuntimeFixture()
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let session = NativeControlSession(configuration: config, owner: NativeControlOwner(), runtimeFactory: fixture.factory)
    _ = try await session.start()
    let result = try await session.submit(controlPacket(config))
    do { _ = try await result.terminalReceipt(); Issue.record("A missing terminal receipt succeeded") }
    catch { #expect((error as? AstraError)?.code == "control.receiptTimeout") }
    #expect(throws: AstraError.self) { try session.checkHealth() }
    #expect(await session.shutdown().cleanupConfirmed && fixture.hasJoined)
}

@Test func nativeControlAbandonedSessionStillJoinsItsOwnedHelper() async throws {
    let fixture = ControlRuntimeFixture(), owner = NativeControlOwner()
    var session: NativeControlSession? = NativeControlSession(configuration: try controlConfiguration(root: FileManager.default.temporaryDirectory),
        owner: owner, runtimeFactory: fixture.factory)
    _ = try await session?.start()
    let deallocated = { [weak session] in session == nil }
    session = nil
    try await controlWait { deallocated() && fixture.hasJoined && owner.priorCleanupJoined }
    #expect(fixture.log.contains("disarm") && fixture.cancelledRequests == 0)
}

@Test func nativeControlReceiptDeadlineIncludesBothMaximumLookaheadAndPacketDuration() throws {
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    var packet = controlPacket(config)
    let now: UInt64 = 9_000_000_000
    packet.executeAtNanos = now + ControlLease.maximumLookaheadNanos; packet.durationMs = 1_000
    _ = try packet.validated(capabilities: config.capabilities, surfaces: config.scope.surfaces, capacity: config.packetCapacity)
    #expect(NativeControlSession.receiptWaitNanos(packet, now: now) == 3_500_000_000)
    #expect(NativeControlSession.receiptWaitNanos(packet, now: now + 3_000_000_000) == 500_000_000)
}

@Test func nativeControlAbandonmentDrainsBlockedHeartbeatBeforeJoiningItsOwner() async throws {
    let fixture = ControlRuntimeFixture(blockHeartbeat: true), owner = NativeControlOwner()
    var session: NativeControlSession? = NativeControlSession(configuration: try controlConfiguration(root: FileManager.default.temporaryDirectory),
        owner: owner, runtimeFactory: fixture.factory)
    _ = try await session?.start()
    try await controlWait { fixture.heartbeatCount == 1 }
    let deallocated = { [weak session] in session == nil }
    session = nil
    try await controlWait { deallocated() && fixture.log.contains("disarm") }
    #expect(!owner.priorCleanupJoined && !fixture.hasJoined && fixture.cancelledRequests == 0)
    fixture.heartbeatGate.open()
    try await controlWait { fixture.hasJoined && owner.priorCleanupJoined }
}

@Test func nativeControlCannotRestartAStoppedSessionFromItsCachedArmAcknowledgement() async throws {
    let fixture = ControlRuntimeFixture()
    let session = NativeControlSession(configuration: try controlConfiguration(root: FileManager.default.temporaryDirectory),
        owner: NativeControlOwner(), runtimeFactory: fixture.factory)
    _ = try await session.start()
    _ = try await session.disarm()
    await #expect(throws: AstraError.self) { try await session.start() }
    _ = await session.shutdown()
    await #expect(throws: AstraError.self) { try await session.start() }
    #expect(fixture.startCount == 1)
}

@Test func nativeControlAtomicDispatchRetainsARealRejectionWhenStopWinsAfterTheCallerChecks() async throws {
    let fixture = ControlRuntimeFixture(), owner = NativeControlOwner()
    let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
    let session = NativeControlSession(configuration: config, owner: owner, runtimeFactory: fixture.factory)
    _ = try await session.start()
    try session.checkHealth() // Caller selected live dispatch from this state.
    let packet = controlPacket(config)
    session.requestStop() // Deterministic stop before packet reservation.
    let submitted = try await session.submitPreservingStoppedPacket(packet)
    #expect(!submitted.admitted && submitted.reply.kind == "error")
    #expect(try await submitted.terminalReceipt().status == .rejected)
    #expect(fixture.sentPackets == [packet.id])
    await #expect(throws: AstraError.self) { try await session.submitPreservingStoppedPacket(packet) }
    var foreign = controlPacket(config, sequence: 1); foreign.runID = UUID()
    await #expect(throws: AstraError.self) { try await session.submitPreservingStoppedPacket(foreign) }
    #expect(fixture.sentPackets == [packet.id])
    #expect(await session.shutdown().cleanupConfirmed)
    await #expect(throws: AstraError.self) { try await session.submitPreservingStoppedPacket(controlPacket(config, sequence: 1)) }
}
