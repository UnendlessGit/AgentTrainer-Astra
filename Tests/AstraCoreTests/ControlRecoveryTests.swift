import Foundation
import Testing
import Darwin
@testable import AstraCore

@Test func controlRecoveryRejectsReplacedIdentityAndProtectsConservativeReservations() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraLedger-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let first = try ControlRecoveryLedger(createAt: root.appendingPathComponent("first"), runID: UUID(), capabilities: .init(keyCodes: [0]))
    let second = try ControlRecoveryLedger(createAt: root.appendingPathComponent("second"), runID: first.descriptor.runID, capabilities: .init(keyCodes: [0]))
    let replaced = ControlRecoveryDescriptor(runID: first.descriptor.runID, ledgerID: first.descriptor.ledgerID,
        path: second.descriptor.path, device: first.descriptor.device, inode: first.descriptor.inode)
    #expect(throws: AstraError.self) { _ = try ControlRecoveryLedger(open: replaced) }
    let reader = try ControlRecoveryLedger(open: first.descriptor)
    #expect(first.registerExecutor(pid: getpid()) && first.registerGuardian(pid: getpid(), now: 1_000_000_000))
    #expect(first.arm() && first.guardianIsFresh(now: 1_000_000_001))
    #expect(!first.guardianIsFresh(now: 1_150_000_000))
    #expect(!first.beginPost(operation: .keyDown, keyCode: 55, button: nil))
    #expect(first.beginPost(operation: .keyDown, keyCode: 0, button: nil))
    #expect(try reader.snapshot().possibleKeys == [0] && reader.snapshot().inFlight == 1)
    first.stop()
    #expect(!first.settleLocally(now: 1_010_000_000))
    first.endPost(operation: .keyDown, keyCode: 0, button: nil, success: true)
    #expect(try reader.snapshot().possibleKeys == [0])
    // This call represents the local executor's separately verified final
    // release pass after all driver and provisional-cleanup work has joined.
    #expect(first.settleLocally(now: 1_020_000_000))
    #expect(try reader.snapshot().cleanupConfirmed)
    #expect(!first.beginPost(operation: .keyDown, keyCode: 0, button: nil))
}

private final class ReservationRaceResult: @unchecked Sendable {
    private let lock = NSLock()
    private var success = false
    func record(_ value: Bool) { lock.withLock { success = value } }
    var reserved: Bool { lock.withLock { success } }
}

@Test func controlRecoveryReservationRacingStopNeverPostsAfterSettlement() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraLedgerRace-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for index in 0..<128 {
        let ledger = try ControlRecoveryLedger(createAt: root.appendingPathComponent(String(index)), runID: UUID(), capabilities: .init(keyCodes: [0]))
        #expect(ledger.registerExecutor(pid: getpid()) && ledger.registerGuardian(pid: getpid(), now: 1))
        #expect(ledger.arm())
        let entered = DispatchSemaphore(value: 0), completed = DispatchSemaphore(value: 0)
        let result = ReservationRaceResult()
        DispatchQueue.global().async {
            entered.signal()
            let reserved = ledger.beginPost(operation: .keyDown, keyCode: 0, button: nil)
            result.record(reserved)
            if reserved {
                // A successful reservation protects this interval against a
                // terminal proof until endPost, even if stop races its first read.
                #expect((try? ledger.snapshot().cleanupConfirmed) == false)
                ledger.endPost(operation: .keyDown, keyCode: 0, button: nil, success: true)
            }
            completed.signal()
        }
        #expect(entered.wait(timeout: .now() + 1) == .success)
        ledger.stop()
        _ = ledger.settleLocally(now: 2)
        #expect(completed.wait(timeout: .now() + 1) == .success)
        // The public begin operation rechecks phase after incrementing its
        // in-flight count; a losing reservation never reaches event posting.
        if !result.reserved { #expect(try ledger.snapshot().possibleKeys.isEmpty) }
        _ = ledger.settleLocally(now: 3)
        #expect(try ledger.snapshot().cleanupConfirmed)
    }
}
