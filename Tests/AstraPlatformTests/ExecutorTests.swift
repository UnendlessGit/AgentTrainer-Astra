import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

private final class ExecutorClock: @unchecked Sendable {
    private let lock = NSLock()
    private var value: UInt64 = 1_000_000_000
    var now: UInt64 { lock.withLock { value } }
    func advance(ms: UInt64) { lock.withLock { value += ms * 1_000_000 } }
}
private final class VirtualControl: ControlInputBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var events: [InputEmission] = []
    private var physical = ControlState()
    private var heldKeys: Set<Int> = []
    private var heldButtons: Set<Int> = []
    private var validScope = true
    private var healthy = true
    private var failures: Set<CommandOperation> = []
    let released = DispatchSemaphore(value: 0)
    let entering = DispatchSemaphore(value: 0)
    let unblock = DispatchSemaphore(value: 0)
    var blockDown = false // Configured before starting the driver.
    var blockCleanup = false
    var blockPrepare = false
    var blockValidation = false
    var failAfterDown = false
    init() { physical.valid = true; physical.pointer = Point2D(x: 10, y: 10) }
    func prepare(_ request: ArmRequest) throws -> ControlState {
        try checkHealth()
        if blockPrepare { try waitForRelease() }
        return physicalState()
    }
    private func waitForRelease() throws {
        entering.signal()
        guard unblock.wait(timeout: .now() + 5) == .success else { throw AstraError("fixture.timeout", "Fixture operation was not released.") }
    }
    func physicalState() -> ControlState { lock.withLock { physical } }
    func checkHealth() throws {
        guard lock.withLock({ healthy }) else { throw AstraError("fixture.permission", "Permission verification expired or was denied.") }
    }
    func setHealthy(_ value: Bool) { lock.withLock { healthy = value } }
    func setPhysical(keys: Set<Int>, buttons: Set<Int> = []) { lock.withLock { physical.keys = keys; physical.buttons = buttons } }
    func invalidatePhysical() { lock.withLock { physical.valid = false } }
    func warpPointer(_ pointer: Point2D) { lock.withLock { physical.pointer = pointer } }
    func invalidateScope() { lock.withLock { validScope = false } }
    func failNext(_ operations: Set<CommandOperation>) { lock.withLock { failures = operations } }
    func validate(_ scope: ControlScope, pointer: Point2D?) throws {
        if blockValidation { try waitForRelease() }
        guard lock.withLock({ validScope }) else { throw AstraError("fixture.geometry", "Target geometry changed.") }
    }
    func post(_ emission: InputEmission) throws {
        if blockDown && emission.operation == .keyDown {
            try waitForRelease()
        }
        if blockCleanup && emission.cleanup { try waitForRelease() }
        try lock.withLock {
            if failures.remove(emission.operation) != nil { throw AstraError("fixture.failure", "Injected posting failure.") }
            events.append(emission)
            if emission.operation.isMotion { physical.pointer = emission.location }
            if let key = emission.keyCode {
                if emission.operation == .keyDown { heldKeys.insert(key) }
                if emission.operation == .keyUp { heldKeys.remove(key) }
            }
            if let button = emission.button {
                if emission.operation == .buttonDown { heldButtons.insert(button) }
                if emission.operation == .buttonUp { heldButtons.remove(button) }
            }
        }
        if emission.cleanup { released.signal() }
        if failAfterDown && emission.operation == .keyDown { throw AstraError("fixture.afterPost", "Posting reported failure after the input may have taken effect.") }
    }
    var posted: [InputEmission] { lock.withLock { events } }
    var ownedKeys: Set<Int> { lock.withLock { heldKeys } }
}
private final class Receipts: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [ExecutionReceipt] = []
    func append(_ receipt: ExecutionReceipt) { lock.withLock { values.append(receipt) } }
    var all: [ExecutionReceipt] { lock.withLock { values } }
}
private final class ControlStops: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [(UUID?, ControlStopCause, String)] = []
    func append(_ run: UUID?, _ cause: ControlStopCause, _ reason: String) { lock.withLock { stored.append((run, cause, reason)) } }
    var all: [(UUID?, ControlStopCause, String)] { lock.withLock { stored } }
}
private func waitForControl(_ predicate: () -> Bool) -> Bool {
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while !predicate() && ContinuousClock.now < deadline { Thread.sleep(forTimeInterval: 0.001) }
    return predicate()
}
private struct ExecutorFixture {
    let clock = ExecutorClock()
    let backend = VirtualControl()
    let receipts = Receipts()
    let stops = ControlStops()
    let run = UUID()
    let executor: InputExecutor
    init(automaticScheduling: Bool = false, historyCapacity: Int = 2048) {
        let clock = clock, backend = backend, receipts = receipts, stops = stops
        executor = InputExecutor(backend: backend, clock: { clock.now }, automaticScheduling: automaticScheduling, historyCapacity: historyCapacity,
                                 onReceipt: { receipts.append($0) }, onStop: { stops.append($0, $1, $2) })
    }
    var arm: ArmRequest {
        ArmRequest(runID: run, scope: ControlScope(surfaces: [SurfaceDescriptor(id: "fixture", globalBounds: Rect2D(x: 0, y: 0, width: 100, height: 100), pixelWidth: 200, pixelHeight: 200)], wholeDesktop: true),
                   capabilities: ActionCapabilities(keyCodes: [0, 55, 56, 60], mouseButtons: [0, 1], absolutePointer: true, relativePointer: true, scroll: true))
    }
    func packet(_ commands: [TimedCommand], sequence: UInt64 = 0, duration: Int = 10, leadMS: UInt64 = 0) -> ActionPacket {
        ActionPacket(runID: run, sequence: sequence, observationID: UUID(), geometryRevision: 0,
                     executeAtNanos: clock.now + leadMS * 1_000_000, durationMs: duration, commands: commands)
    }
}

@Test func executorPreservesTapsAndIdempotentOwnershipWithExactReceipts() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    let packet = f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0),
                           TimedCommand(offsetMs: 1, operation: .keyDown, keyCode: 0),
                           TimedCommand(offsetMs: 2, operation: .keyRepeat, keyCode: 0),
                           TimedCommand(offsetMs: 3, operation: .keyUp, keyCode: 0),
                           TimedCommand(offsetMs: 4, operation: .keyUp, keyCode: 0)])
    try f.executor.execute(packet)
    for _ in 0...10 { f.executor.service(); f.clock.advance(ms: 1) }
    #expect(f.backend.posted.map(\.operation) == [.keyDown, .keyRepeat, .keyUp])
    let receipt = try #require(f.receipts.all.last)
    #expect(receipt.status == .executed && receipt.commandResults.count == 5)
    #expect(receipt.commandResults.map(\.status) == [.posted, .noOp, .posted, .posted, .noOp])
    #expect(receipt.commandResults[2].postedNanos == packet.executeAtNanos + 2_000_000)
    #expect(f.backend.ownedKeys.isEmpty)
    f.executor.disarm()
}

@Test func executorRejectsWholeInvalidPacketsAndDoesNotConsumeTheirSequence() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    let invalid = f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0), TimedCommand(offsetMs: 1, operation: .keyDown, keyCode: 13)])
    #expect(throws: AstraError.self) { try f.executor.execute(invalid) }
    var moved = f.packet([]); moved.geometryRevision = 2
    #expect(throws: AstraError.self) { try f.executor.execute(moved) }
    let edge = f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0),
                         TimedCommand(offsetMs: 1, operation: .pointerAbsolute, surfaceID: "fixture", x: 1, y: 0.5)])
    #expect(throws: AstraError.self) { try f.executor.execute(edge) }
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    f.executor.service()
    #expect(f.backend.posted.count == 1 && f.backend.ownedKeys == [0])
    f.executor.disarm()
    #expect(f.backend.ownedKeys.isEmpty)
}

@Test func executorRelativeInterpolationPreservesCumulativeRawCountsAndOrder() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .pointerRelative, dx: 0, dy: 0),
                                    TimedCommand(offsetMs: 2, operation: .buttonDown, button: 0),
                                    TimedCommand(offsetMs: 5, operation: .pointerRelative, dx: 7, dy: -3)], duration: 5))
    for _ in 0...5 { f.executor.service(); f.clock.advance(ms: 1) }
    let motion = f.backend.posted.filter { $0.operation == .pointerRelative }
    #expect(motion.map { $0.delta.x } == [0, 1, 2, 1, 2, 1])
    #expect(motion.map { $0.delta.y }.reduce(0, +) == -3)
    #expect(f.backend.posted[2].operation == .pointerRelative && f.backend.posted[3].operation == .buttonDown)
    #expect(f.executor.state().pointer == Point2D(x: 17, y: 7))
    f.executor.disarm()
}

@Test func executorAbsoluteInterpolationUsesSurfaceGeometryAndNeverReleasesAPhysicalTakeover() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .pointerAbsolute, surfaceID: "fixture", x: 0.1, y: 0.2),
                                    TimedCommand(offsetMs: 5, operation: .pointerAbsolute, surfaceID: "fixture", x: 0.6, y: 0.8)], duration: 5))
    for _ in 0...5 { f.executor.service(); f.clock.advance(ms: 1) }
    let points = f.backend.posted.map(\.location)
    #expect(points.count == 6 && points.first == Point2D(x: 10, y: 20) && points.last == Point2D(x: 60, y: 80))
    #expect(abs(points[2].x - 30) < 1e-10 && abs(points[2].y - 44) < 1e-10)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)], sequence: 1))
    f.executor.service(); f.backend.setPhysical(keys: [0]); f.executor.disarm(reason: "Physical takeover")
    #expect(f.backend.posted.filter { $0.operation == .keyUp }.isEmpty)
    #expect(!f.executor.state().valid && f.executor.state().keys.isEmpty)
}

@Test func executorExpiresItsLeaseAndCancelsQueuedCommandsWithoutReleasingPhysicalHolds() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)], duration: 100))
    f.executor.service()
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 55)], sequence: 1, leadMS: 200))
    f.backend.setPhysical(keys: [56])
    f.clock.advance(ms: 500); f.executor.checkWatchdog(); f.executor.service()
    #expect(waitForControl { f.receipts.all.filter { $0.status == .cancelled }.count == 2 })
    #expect(f.backend.posted.map(\.keyCode) == [0, 0])
    #expect(f.backend.ownedKeys.isEmpty && !f.executor.state().valid)
    #expect(throws: AstraError.self) { try f.executor.heartbeat(runID: f.run) }
    #expect(f.receipts.all.filter { $0.status == .cancelled }.count == 2)
}

@Test func executorWatchdogCannotBeBlockedByPostingOrReviveAnInflightHold() throws {
    let f = ExecutorFixture(); f.backend.blockDown = true; try f.executor.arm(f.arm)
    let packet = f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)])
    try f.executor.execute(packet)
    let finished = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { f.executor.service(); finished.signal() }
    defer { f.backend.unblock.signal() }
    #expect(f.backend.entering.wait(timeout: .now() + 3) == .success)
    f.clock.advance(ms: 500); f.executor.checkWatchdog()
    #expect(!f.executor.state().valid)
    #expect(f.backend.released.wait(timeout: .now() + 3) == .success)
    #expect(f.backend.posted.map(\.operation) == [.keyUp])
    #expect(f.receipts.all.filter { $0.status == .cancelled }.isEmpty) // The in-flight result remains unsettled.
    #expect(throws: AstraError.self) { try f.executor.arm(f.arm) }
    f.backend.unblock.signal()
    #expect(finished.wait(timeout: .now() + 3) == .success)
    #expect(f.backend.posted.map(\.operation) == [.keyUp, .keyDown, .keyUp])
    #expect(f.backend.ownedKeys.isEmpty)
    let terminal = try #require(f.receipts.all.last)
    #expect(terminal.status == .cancelled && terminal.commandResults[0].status == .posted)
    #expect(terminal.commandResults[0].postedNanos == f.clock.now)
}

@Test func executorStopsAtGeometryChangesAndLateDeadlinesAndBoundsItsQueue() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)], leadMS: 10))
    f.backend.invalidateScope(); f.clock.advance(ms: 10); f.executor.service()
    #expect(f.backend.posted.isEmpty && !f.executor.state().valid)
    let late = ExecutorFixture(); try late.executor.arm(late.arm)
    try late.executor.execute(late.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    late.clock.advance(ms: 21); late.executor.service()
    #expect(late.backend.posted.isEmpty && late.receipts.all.last?.status == .late)
    let full = ExecutorFixture(); try full.executor.arm(full.arm)
    for number in 0..<32 { try full.executor.execute(full.packet([], sequence: UInt64(number), duration: 1, leadMS: UInt64(number))) }
    #expect(throws: AstraError.self) { try full.executor.execute(full.packet([], sequence: 32, duration: 1, leadMS: 32)) }
    full.executor.disarm()
    #expect(full.receipts.all.filter { $0.status == .cancelled }.count == 32)
}

@Test func executorRetainsFailedReleasesForRetryAndRejectsPhysicalHoldsAtArm() throws {
    let f = ExecutorFixture(); f.backend.setPhysical(keys: [55])
    #expect(throws: AstraError.self) { try f.executor.arm(f.arm) }
    f.backend.setPhysical(keys: []); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)])); f.executor.service()
    f.backend.failNext([.keyUp]); f.executor.disarm()
    #expect(f.executor.state().keys == [0] && !f.executor.state().valid)
    #expect(throws: AstraError.self) { try f.executor.arm(f.arm) }
    f.executor.checkWatchdog()
    #expect(waitForControl { f.executor.cleanupSettled })
    #expect(f.executor.state().keys.isEmpty && f.backend.ownedKeys.isEmpty)
}

@Test func executorPermissionWatchdogReleasesHoldsWithoutWaitingForAnotherPacket() throws {
    let f = ExecutorFixture(); f.backend.setHealthy(false)
    #expect(throws: AstraError.self) { try f.executor.arm(f.arm) }
    f.backend.setHealthy(true); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)], duration: 1))
    f.executor.service(); f.clock.advance(ms: 1); f.executor.service()
    #expect(f.backend.ownedKeys == [0])
    f.backend.setHealthy(false); f.executor.checkWatchdog()
    #expect(waitForControl { f.executor.cleanupSettled })
    #expect(!f.executor.state().valid && f.backend.ownedKeys.isEmpty)
    #expect(f.backend.posted.last?.cleanup == true)
}

@Test func executorIgnoresStaleMonitorEventsAndInvalidatesCurrentAdmissionImmediately() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    f.executor.requestDisarm(runID: UUID(), reason: "Old monitor callback")
    #expect(f.executor.state().valid)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 1, operation: .keyDown, keyCode: 0)]))
    f.executor.requestDisarm(runID: f.run, reason: "Current physical takeover")
    #expect(!f.executor.state().valid)
    f.clock.advance(ms: 1); f.executor.service()
    #expect(f.backend.posted.isEmpty)
    f.executor.disarm()
}

@Test func desktopControlLockExcludesOtherLibrariesAndModifierFlagsRetainSides() throws {
    let path = FileManager.default.temporaryDirectory.appendingPathComponent("AstraDesktopLock-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: path) }
    var first: DesktopControlLock? = try DesktopControlLock(url: path)
    #expect(first != nil)
    #expect(throws: AstraError.self) { _ = try DesktopControlLock(url: path) }
    first = nil
    let next = try DesktopControlLock(url: path)
    withExtendedLifetime(next) { }
    let left = CGEventControlBackend.modifierFlags(keys: [56])
    let right = CGEventControlBackend.modifierFlags(keys: [60])
    #expect(left.contains(.maskShift) && right.contains(.maskShift) && left != right)
    #expect(CGEventControlBackend.modifierFlags(keys: []).isEmpty)
}


@Test func executorAllowsBoundedSchedulingJitterWithoutInvalidatingExecution() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    let packet = f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)], duration: 10)
    try f.executor.execute(packet)
    f.clock.advance(ms: 1); f.executor.service()
    f.clock.advance(ms: 10); f.executor.service()
    let receipt = try #require(f.receipts.all.last)
    #expect(receipt.status == .executed)
    #expect(receipt.commandResults[0].postedNanos == packet.executeAtNanos + 1_000_000)
    #expect(f.executor.state().valid)
    f.executor.disarm()
}

@Test func executorCausalObservationAcknowledgesHistoryAndPreservesSilentInputAge() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    let initial = try f.executor.observation()
    #expect(initial.executedEvents.isEmpty && initial.intervalCovered && initial.lastSequence == nil)
    let packet = f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0),
                           TimedCommand(offsetMs: 1, operation: .keyDown, keyCode: 0),
                           TimedCommand(offsetMs: 2, operation: .pointerRelative, dx: 3, dy: -2),
                           TimedCommand(offsetMs: 3, operation: .scroll, dx: 0.125, dy: -0.375),
                           TimedCommand(offsetMs: 4, operation: .keyUp, keyCode: 0)])
    try f.executor.execute(packet)
    f.clock.advance(ms: 1); f.executor.service()
    let first = try f.executor.observation()
    #expect(first.lastSequence == 0 && first.executedEvents.count == 1)
    #expect(first.controlState.keys == [0] && first.controlState.valid)
    #expect(first.executedEvents[0].origin == .agent && first.executedEvents[0].kind == .keyDown)
    for _ in 0..<4 { f.clock.advance(ms: 1); f.executor.service() }
    let second = try f.executor.observation(afterSequence: first.lastSequence)
    #expect(second.lastSequence == 3 && second.executedEvents.map(\.kind) == [.pointer, .scroll, .keyUp])
    #expect(second.executedEvents.allSatisfy { $0.eventNanos <= $0.observedNanos && $0.observedNanos > first.cutoffNanos && $0.observedNanos <= second.cutoffNanos })
    #expect(second.executedEvents[0].x == 13 && second.executedEvents[0].dx == 3)
    #expect(second.executedEvents[1].scrollY == -0.375)
    #expect(second.controlState.keys.isEmpty && second.controlState.pointer == Point2D(x: 13, y: 8))
    f.clock.advance(ms: 1)
    let silent = try f.executor.observation(afterSequence: second.lastSequence)
    #expect(silent.executedEvents.isEmpty && silent.intervalCovered && silent.lastSequence == second.lastSequence)
    #expect(silent.controlState.observedNanos == second.controlState.observedNanos)
    #expect(throws: AstraError.self) { _ = try f.executor.observation(afterSequence: 100) }
    #expect(throws: AstraError.self) { _ = try f.executor.observation(afterSequence: 0) }
    #expect(throws: AstraError.self) { _ = try f.executor.observation() }
    f.executor.disarm()
}

@Test func executorHistoryOverflowStopsControlInsteadOfLosingCausalCoverage() throws {
    let f = ExecutorFixture(historyCapacity: 2); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0),
                                    TimedCommand(offsetMs: 1, operation: .keyRepeat, keyCode: 0),
                                    TimedCommand(offsetMs: 2, operation: .keyRepeat, keyCode: 0)]))
    for _ in 0...2 { f.executor.service(); f.clock.advance(ms: 1) }
    let snapshot = try f.executor.observation()
    #expect(snapshot.executedEvents.count == 2 && !snapshot.intervalCovered && !snapshot.controlState.valid)
    #expect(f.executor.currentRunID == nil && f.backend.ownedKeys.isEmpty && f.executor.cleanupSettled)
    #expect(f.receipts.all.last?.status == .cancelled)
}

@Test func executorAutomaticWatchdogAndShutdownSettlementAreIndependentOfBlockedPosting() throws {
    let f = ExecutorFixture(automaticScheduling: true); f.backend.blockDown = true
    try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    defer { f.backend.unblock.signal() }
    #expect(f.backend.entering.wait(timeout: .now() + 3) == .success)
    #expect(!f.executor.cleanupSettled)
    #expect(try !f.executor.observation().controlState.valid)
    f.clock.advance(ms: 500)
    // No manual service/checkWatchdog call: this exercises the separate timer queue.
    #expect(f.backend.released.wait(timeout: .now() + 3) == .success)
    #expect(f.executor.currentRunID == nil && !f.executor.cleanupSettled)
    #expect(f.receipts.all.filter { $0.status == .cancelled }.isEmpty)
    f.backend.unblock.signal()
    #expect(f.backend.released.wait(timeout: .now() + 3) == .success)
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while !f.executor.cleanupSettled && ContinuousClock.now < deadline { Thread.sleep(forTimeInterval: 0.001) }
    #expect(f.executor.cleanupSettled && f.backend.ownedKeys.isEmpty)
    #expect(f.receipts.all.last?.status == .cancelled)
}

@Test func executorWatchdogReturnsWhileCleanupIsBlockedAndRetainsOwnership() throws {
    let f = ExecutorFixture(); f.backend.blockCleanup = true
    try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    f.executor.service(); f.clock.advance(ms: 500)
    let checked = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { f.executor.checkWatchdog(); checked.signal() }
    defer { f.backend.unblock.signal() }
    #expect(checked.wait(timeout: .now() + 1) == .success)
    #expect(f.backend.entering.wait(timeout: .now() + 3) == .success)
    #expect(!f.executor.state().valid && !f.executor.cleanupSettled)
    #expect(f.backend.ownedKeys == [0])
    #expect(throws: AstraError.self) { try f.executor.arm(f.arm) }
    DispatchQueue.global().async { f.executor.checkWatchdog(); checked.signal() }
    #expect(checked.wait(timeout: .now() + 1) == .success)
    f.backend.unblock.signal()
    #expect(waitForControl { f.executor.cleanupSettled && f.receipts.all.last?.status == .cancelled })
    #expect(f.backend.ownedKeys.isEmpty)
}

@Test(arguments: [ControlStopCause.physicalTakeover, .emergencyStop])
func executorStoppedArmingCannotPublishALateLease(cause: ControlStopCause) throws {
    let f = ExecutorFixture(); f.backend.blockPrepare = true
    let completed = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
        #expect(throws: AstraError.self) { try f.executor.arm(f.arm) }
        completed.signal()
    }
    defer { f.backend.unblock.signal() }
    #expect(f.backend.entering.wait(timeout: .now() + 3) == .success)
    f.executor.requestDisarm(runID: f.run, reason: "Intervention during prepare", cause: cause)
    #expect(!f.executor.cleanupSettled && !f.executor.state().valid)
    f.backend.unblock.signal()
    #expect(completed.wait(timeout: .now() + 3) == .success)
    #expect(waitForControl { f.executor.cleanupSettled })
    #expect(f.executor.currentRunID == nil && f.backend.posted.isEmpty)
    #expect(waitForControl { f.stops.all.contains { $0.0 == f.run && $0.1 == cause } })
}

@Test func executorRechecksHealthAfterArmingPreparationAndRejectsUnknownPhysicalState() throws {
    let f = ExecutorFixture(); f.backend.blockPrepare = true
    let completed = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
        #expect(throws: AstraError.self) { try f.executor.arm(f.arm) }
        completed.signal()
    }
    defer { f.backend.unblock.signal() }
    #expect(f.backend.entering.wait(timeout: .now() + 3) == .success)
    f.backend.setHealthy(false); f.backend.unblock.signal()
    #expect(completed.wait(timeout: .now() + 3) == .success)
    #expect(f.executor.cleanupSettled && f.executor.currentRunID == nil)
    let invalid = ExecutorFixture(); try invalid.executor.arm(invalid.arm)
    try invalid.executor.execute(invalid.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    invalid.backend.invalidatePhysical(); invalid.executor.service()
    #expect(invalid.backend.posted.isEmpty && invalid.executor.cleanupSettled)
    #expect(invalid.receipts.all.last?.status == .cancelled)
    #expect(invalid.stops.all.last?.1 == .fault)
}

@Test func executorReportsPhysicalTakeoverAsAnInterventionInsteadOfAControlFault() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    f.backend.setPhysical(keys: [55]); f.executor.service()
    let stop = try #require(f.stops.all.last)
    #expect(stop.0 == f.run && stop.1 == .physicalTakeover)
    #expect(f.backend.posted.isEmpty && f.executor.cleanupSettled)
}

@Test func executorStopDuringScopeValidationNeverPostsAndAReportedPostFailureStillCleans() throws {
    let f = ExecutorFixture(); f.backend.blockValidation = true
    try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    let completed = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { f.executor.service(); completed.signal() }
    defer { f.backend.unblock.signal() }
    #expect(f.backend.entering.wait(timeout: .now() + 3) == .success)
    f.executor.disarm(); f.backend.unblock.signal()
    #expect(completed.wait(timeout: .now() + 3) == .success)
    #expect(f.backend.posted.isEmpty && f.executor.cleanupSettled)
    #expect(f.receipts.all.filter { $0.status == .cancelled }.count == 1)
    let failed = ExecutorFixture(); failed.backend.failAfterDown = true
    try failed.executor.arm(failed.arm)
    try failed.executor.execute(failed.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    failed.executor.service()
    #expect(failed.backend.posted.map(\.operation) == [.keyDown, .keyUp])
    #expect(failed.backend.ownedKeys.isEmpty && failed.executor.cleanupSettled)
    #expect(failed.receipts.all.last?.commandResults[0].status == .failed)
}

@Test func executorLatePostingPreservesItsResultAndReleasesTheHold() throws {
    let f = ExecutorFixture(); f.backend.blockDown = true; try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    let completed = DispatchSemaphore(value: 0)
    DispatchQueue.global().async { f.executor.service(); completed.signal() }
    defer { f.backend.unblock.signal() }
    #expect(f.backend.entering.wait(timeout: .now() + 3) == .success)
    f.clock.advance(ms: 21); f.backend.unblock.signal()
    #expect(completed.wait(timeout: .now() + 3) == .success)
    #expect(f.backend.posted.map(\.operation) == [.keyDown, .keyUp])
    #expect(f.backend.ownedKeys.isEmpty && f.executor.cleanupSettled)
    let receipt = try #require(f.receipts.all.last)
    #expect(receipt.status == .late && receipt.commandResults[0].status == .posted)
    #expect(receipt.commandResults[0].postedNanos == f.clock.now)
}

@Test func executorRelativeInputAndClicksFollowAnApplicationCursorWarp() throws {
    let f = ExecutorFixture(); try f.executor.arm(f.arm)
    try f.executor.execute(f.packet([TimedCommand(offsetMs: 0, operation: .pointerRelative, dx: 20, dy: 0),
                                    TimedCommand(offsetMs: 1, operation: .buttonDown, button: 0)]))
    f.backend.warpPointer(Point2D(x: 50, y: 50)); f.executor.service()
    #expect(f.backend.posted.last?.location == Point2D(x: 70, y: 50))
    f.backend.warpPointer(Point2D(x: 40, y: 40)); f.clock.advance(ms: 1); f.executor.service()
    #expect(f.backend.posted.last?.location == Point2D(x: 40, y: 40))
    #expect(f.executor.state().pointer == Point2D(x: 40, y: 40))
    f.backend.warpPointer(Point2D(x: 20, y: 20))
    f.executor.disarm()
    #expect(f.backend.posted.last?.operation == .buttonUp && f.backend.posted.last?.location == Point2D(x: 20, y: 20))
}
