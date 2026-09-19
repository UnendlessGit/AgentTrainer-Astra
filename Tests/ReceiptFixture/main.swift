import Foundation
import AstraCore
import AstraPlatform

// Non-shipping interoperability fixture. The real scheduler emits its receipts;
// the backend has no OS input/capture APIs and keeps effects in private memory.
private final class FixtureClock: @unchecked Sendable {
    private let lock = NSLock()
    private var nanos: UInt64 = 1_000_000_000
    var now: UInt64 { lock.withLock { nanos } }
    func set(_ value: UInt64) { lock.withLock { nanos = value } }
}
private struct Effect: Codable { let operation: CommandOperation; let keyCode: Int?; let cleanup: Bool }
private final class Backend: ControlInputBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var effects: [Effect] = []
    let failPosting: Bool
    let failCleanup: Bool
    init(failPosting: Bool, failCleanup: Bool) { self.failPosting = failPosting; self.failCleanup = failCleanup }
    func prepare(_ request: ArmRequest) throws -> ControlState { physicalState() }
    func physicalState() -> ControlState {
        var result = ControlState(); result.valid = true; result.pointer = .init(x: 4, y: 4)
        result.observedNanos = 1_000_000_000; return result
    }
    func checkHealth() throws {}
    func validate(_ scope: ControlScope, pointer: Point2D?) throws {}
    func post(_ emission: InputEmission) throws {
        if (failPosting && !emission.cleanup) || (failCleanup && emission.cleanup) {
            throw AstraError("fixture.postFailure", "Virtual backend rejected the requested effect.")
        }
        lock.withLock { effects.append(Effect(operation: emission.operation, keyCode: emission.keyCode, cleanup: emission.cleanup)) }
    }
    var posted: [Effect] { lock.withLock { effects } }
}
private final class ReceiptLog: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [ExecutionReceipt] = []
    func append(_ value: ExecutionReceipt) { lock.withLock { values.append(value) } }
    var all: [ExecutionReceipt] { lock.withLock { values } }
}
private struct Report: Codable {
    let name: String
    let runID: UUID
    let decisionNanos: UInt64
    let boundaryNanos: UInt64
    let stoppedNanos: UInt64
    let packet: ActionPacket
    let rejectedPacket: ActionPacket
    let receipts: [ExecutionReceipt]
    let cleanupSettled: Bool
    let finalState: ControlState
    let effects: [Effect]
}
private func run(_ name: String) throws -> Report {
    let clock = FixtureClock(), log = ReceiptLog()
    let backend = Backend(failPosting: name == "postingFailure", failCleanup: name == "cleanupFailure")
    let executor = InputExecutor(backend: backend, clock: { clock.now }, automaticScheduling: false,
                                 onReceipt: { log.append($0) })
    let runID = UUID(), start = clock.now
    let boundary = start + (name == "overdueCancellation" ? 160_000_000 : 50_000_000)
    let surface = SurfaceDescriptor(id: "receipt-fixture", globalBounds: .init(x: 0, y: 0, width: 10, height: 10),
                                    pixelWidth: 10, pixelHeight: 10)
    try executor.arm(ArmRequest(runID: runID, scope: ControlScope(surfaces: [surface], wholeDesktop: true),
                                capabilities: ActionCapabilities(keyCodes: [0])))
    let packet = ActionPacket(runID: runID, sequence: 0, observationID: UUID(), geometryRevision: 0,
                              executeAtNanos: start + 100_000_000, durationMs: 100,
                              commands: [TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0),
                                         TimedCommand(offsetMs: 50, operation: .keyUp, keyCode: 0)])
    try executor.execute(packet)
    if name == "overdueCancellation" {
        clock.set(start + 170_000_000) // Delayed scheduling must not become on-policy cancellation.
    } else if name != "futureCancellation" {
        clock.set(start + 100_000_000); executor.service()
        if name == "completedPacket" {
            clock.set(start + 150_000_000); executor.service()
            clock.set(start + 200_000_000); executor.service()
        } else { clock.set(start + 120_000_000) }
    } else { clock.set(boundary) }
    executor.disarm(reason: "Fixture observed an earlier episode boundary.", cause: .requested)
    let rejected = ActionPacket(runID: runID, sequence: 1, observationID: UUID(), geometryRevision: 0,
                                executeAtNanos: clock.now + 100_000_000, durationMs: 100,
                                commands: [TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0)])
    do {
        try executor.execute(rejected)
        throw AstraError("fixture.unexpectedAdmission", "Disarmed executor admitted a packet.")
    } catch let error as AstraError where error.code == "fixture.unexpectedAdmission" { throw error }
    catch { /* The actual rejected receipt was delivered by InputExecutor. */ }
    return Report(name: name, runID: runID, decisionNanos: start, boundaryNanos: boundary, stoppedNanos: clock.now,
                  packet: packet, rejectedPacket: rejected, receipts: log.all, cleanupSettled: executor.cleanupSettled,
                  finalState: executor.state(), effects: backend.posted)
}

do {
    let reports = try ["futureCancellation", "postedCancellation", "postingFailure", "cleanupFailure", "overdueCancellation", "completedPacket"].map(run)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    try FileHandle.standardOutput.write(contentsOf: encoder.encode(reports) + Data([10]))
} catch {
    FileHandle.standardError.write(Data(error.localizedDescription.utf8)); Foundation.exit(1)
}
