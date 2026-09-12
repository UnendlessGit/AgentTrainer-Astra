import Foundation
import Testing
import Darwin
import AstraCore
@testable import AstraPlatform

private final class GuardianBundleAnchor: NSObject {}

private final class RecoveryProcessFixture {
    let root: URL
    let ledger: ControlRecoveryLedger
    let process = Process()
    init(mode: String, flags: [String] = []) throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraRecovery-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ledger = try ControlRecoveryLedger(createAt: root.appendingPathComponent("ownership.ledger"), runID: UUID(), capabilities: .init(keyCodes: [0, 55], mouseButtons: [0]))
        try JSONEncoder().encode(ledger.descriptor).write(to: root.appendingPathComponent("descriptor.json"))
        try Data().write(to: root.appendingPathComponent("effects.jsonl"))
        for flag in flags { try Data().write(to: root.appendingPathComponent(flag)) }
        var directory = Bundle(for: GuardianBundleAnchor.self).bundleURL.deletingLastPathComponent()
        var executable: URL?
        for _ in 0..<6 {
            let candidate = directory.appendingPathComponent("AstraRecoveryFixture")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { executable = candidate; break }
            directory.deleteLastPathComponent()
        }
        guard let executable else {
            try? FileManager.default.removeItem(at: root)
            throw AstraError("fixture.executable", "The built recovery fixture was not found beside \(Bundle(for: GuardianBundleAnchor.self).bundleURL.path).")
        }
        process.executableURL = executable
        process.environment = ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "ASTRA_RECOVERY_FIXTURE_ROOT": root.path, "ASTRA_RECOVERY_FIXTURE_MODE": mode]
        process.standardInput = FileHandle.nullDevice; process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        try process.run()
    }
    func exists(_ name: String) -> Bool { FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path) }
    func write(_ name: String) throws { try Data().write(to: root.appendingPathComponent(name)) }
    func wait(_ condition: () throws -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while try !condition() {
            guard ContinuousClock.now < deadline else {
                let failure = (try? String(contentsOf: root.appendingPathComponent("error.txt"), encoding: .utf8)) ?? "none"
                let status = (try? String(contentsOf: root.appendingPathComponent("guardian-status.txt"), encoding: .utf8)) ?? "running"
                let guardianError = (try? String(contentsOf: root.appendingPathComponent("guardian-error.txt"), encoding: .utf8)) ?? "none"
                throw AstraError("fixture.timeout", "Guardian fixture timed out. Executor error: \(failure); guardian status: \(status); guardian error: \(guardianError); ledger: \(String(describing: try? ledger.snapshot())).")
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
    func effects() throws -> [[String: JSONValue]] {
        try Data(contentsOf: root.appendingPathComponent("effects.jsonl")).split(separator: 10).map { try JSONDecoder().decode([String: JSONValue].self, from: Data($0)) }
    }
    func assertLeaseHeld() {
        #expect(throws: AstraError.self) { _ = try DesktopControlLock(url: root.appendingPathComponent("desktop.lock")) }
    }
    func close() async {
        for name in ["deny-release", "invalid-physical", "deny-ready"] { try? FileManager.default.removeItem(at: root.appendingPathComponent(name)) }
        for name in ["allow-post", "allow-ready", "allow-bootstrap", "stop"] { try? write(name) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while process.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(5)) }
        if process.isRunning { process.terminate() }
        while process.isRunning { try? await Task.sleep(for: .milliseconds(5)) }
        try? await wait {
            let value = try self.ledger.snapshot()
            return value.cleanupConfirmed || (value.guardianPID > 0 && !self.ledger.matchingGuardianIsAlive(value))
        }
        try? await wait { let value = try self.ledger.snapshot(); return !self.ledger.matchingGuardianIsAlive(value) }
        try? FileManager.default.removeItem(at: root)
    }
}

private func withRecoveryFixture(mode: String, flags: [String] = [], body: (RecoveryProcessFixture) async throws -> Void) async throws {
    let fixture = try RecoveryProcessFixture(mode: mode, flags: flags)
    do { try await body(fixture); await fixture.close() }
    catch { await fixture.close(); throw error }
}

@Test(arguments: ["reserved", "posted", "inflight", "after-release"])
func guardianRecoversEveryPossiblyPostedBoundaryAfterExecutorKill(mode: String) async throws {
    try await withRecoveryFixture(mode: mode) { fixture in
        let marker = mode == "reserved" || mode == "inflight" ? "post-entered" : mode == "after-release" ? "release-posted" : "down-posted"
        try await fixture.wait { fixture.exists(marker) }
        fixture.assertLeaseHeld()
        #expect(try fixture.ledger.snapshot().possibleKeys.contains(0))
        #expect(kill(fixture.process.processIdentifier, SIGKILL) == 0)
        try await fixture.wait { try fixture.ledger.snapshot().cleanupConfirmed }
        let snapshot = try fixture.ledger.snapshot()
        #expect(snapshot.recoveredByGuardian && snapshot.possibleKeys.isEmpty && snapshot.possibleButtons.isEmpty)
        let effects = try fixture.effects()
        #expect(effects.contains { $0["operation"] == .string("keyUp") && $0["cleanup"] == .bool(true) })
        #expect(!effects.contains { $0["pid"] == .integer(Int64(snapshot.guardianPID)) && $0["operation"] == .string("keyDown") })
    }
}

@Test func guardianRetainsFailedReleasesAndDoesNotReleaseAPhysicalHold() async throws {
    try await withRecoveryFixture(mode: "posted", flags: ["deny-release"]) { fixture in
        try await fixture.wait { try fixture.ledger.snapshot().possibleButtons.contains(0) }
        try JSONSerialization.data(withJSONObject: ["keys": [0], "buttons": []]).write(to: fixture.root.appendingPathComponent("physical.json"))
        #expect(kill(fixture.process.processIdentifier, SIGKILL) == 0)
        try await fixture.wait { try fixture.ledger.snapshot().phase == .recovering }
        try await Task.sleep(for: .milliseconds(40))
        fixture.assertLeaseHeld()
        #expect(try !fixture.ledger.snapshot().cleanupConfirmed)
        #expect(try !fixture.effects().contains { $0["operation"] == .string("keyUp") })
        try FileManager.default.removeItem(at: fixture.root.appendingPathComponent("deny-release"))
        try await fixture.wait { try fixture.ledger.snapshot().cleanupConfirmed }
        #expect(try fixture.effects().contains { $0["operation"] == .string("buttonUp") && $0["cleanup"] == .bool(true) })
    }
}

@Test func executorCleansLocallyWhenItsGuardianIsKilled() async throws {
    try await withRecoveryFixture(mode: "posted") { fixture in
        try await fixture.wait { try fixture.ledger.snapshot().possibleButtons.contains(0) }
        let guardian = try fixture.ledger.snapshot().guardianPID
        #expect(kill(guardian, SIGKILL) == 0)
        try await fixture.wait { fixture.exists("executor-settled") }
        #expect(try fixture.ledger.snapshot().cleanupConfirmed)
        #expect(try !fixture.ledger.snapshot().recoveredByGuardian)
    }
}

@Test func guardianRegistersDeathWatchBeforeReadinessAndHandlesParentAlreadyDead() async throws {
    try await withRecoveryFixture(mode: "posted", flags: ["block-ready"]) { fixture in
        try await fixture.wait { fixture.exists("watch-registered") }
        #expect(try !fixture.ledger.snapshot().everArmed)
        #expect(kill(fixture.process.processIdentifier, SIGKILL) == 0)
        try fixture.write("allow-ready")
        try await fixture.wait { try fixture.ledger.snapshot().cleanupConfirmed }
        #expect(try fixture.effects().isEmpty)
    }
}

@Test func guardianNeverTakesOverWhileExecutorPostingIsStillLive() async throws {
    try await withRecoveryFixture(mode: "inflight") { fixture in
        try await fixture.wait { fixture.exists("post-entered") }
        try await Task.sleep(for: .milliseconds(100))
        #expect(try fixture.ledger.snapshot().inFlight == 1)
        #expect(try !fixture.effects().contains { $0["cleanup"] == .bool(true) })
        fixture.assertLeaseHeld()
        try fixture.write("allow-post"); try fixture.write("stop")
        try await fixture.wait { fixture.exists("executor-settled") }
        #expect(try fixture.ledger.snapshot().cleanupConfirmed)
    }
}

@Test func guardianHandlesExecutorDeathBeforeItsExitWatchCanBeInstalled() async throws {
    try await withRecoveryFixture(mode: "posted", flags: ["block-bootstrap"]) { fixture in
        try await fixture.wait { fixture.exists("bootstrap-waiting") }
        #expect(kill(fixture.process.processIdentifier, SIGKILL) == 0)
        try fixture.write("allow-bootstrap")
        try await fixture.wait { try fixture.ledger.snapshot().cleanupConfirmed }
        #expect(try !fixture.ledger.snapshot().everArmed && fixture.effects().isEmpty)
    }
}

@Test func guardianPermissionDenialPreventsArmingWithoutPosting() async throws {
    try await withRecoveryFixture(mode: "posted", flags: ["deny-ready"]) { fixture in
        try await fixture.wait { !fixture.process.isRunning }
        #expect(try !fixture.ledger.snapshot().everArmed)
        #expect(try fixture.ledger.snapshot().cleanupConfirmed)
        #expect(try fixture.effects().isEmpty)
    }
}

@Test func guardianFailedExitWatchCancelsUnarmedAdmissionWhileParentIsStillAlive() async throws {
    try await withRecoveryFixture(mode: "posted", flags: ["deny-watch"]) { fixture in
        try await fixture.wait { fixture.exists("guardian-status.txt") }
        let snapshot = try fixture.ledger.snapshot()
        #expect(snapshot.cleanupConfirmed && !snapshot.everArmed && !snapshot.guardianReady)
        #expect(fixture.exists("watch-parent-alive"))
        #expect(try fixture.effects().isEmpty)
        #expect(!fixture.exists("watch-registered")) // Permission/readiness never follows a failed watch.
        try await fixture.wait { !fixture.process.isRunning }
        let acquired = try DesktopControlLock(url: fixture.root.appendingPathComponent("desktop.lock"))
        withExtendedLifetime(acquired) {}
    }
}

@Test func mutualGuardianAndExecutorDeathDoesNotProduceFalseCleanupProof() async throws {
    try await withRecoveryFixture(mode: "inflight", flags: ["deny-release"]) { fixture in
        try await fixture.wait { fixture.exists("post-entered") }
        let guardian = try fixture.ledger.snapshot().guardianPID
        #expect(kill(guardian, SIGKILL) == 0)
        try await fixture.wait { try fixture.ledger.snapshot().phase == .stopping }
        #expect(kill(fixture.process.processIdentifier, SIGKILL) == 0)
        try await fixture.wait { let value = try fixture.ledger.snapshot(); return !fixture.process.isRunning && !fixture.ledger.matchingGuardianIsAlive(value) }
        let result = try fixture.ledger.snapshot()
        #expect(!result.cleanupConfirmed && result.possibleKeys.contains(0))
    }
}

@Test func guardianDoesNotTransferOwnershipFromAnUnverifiedPhysicalSnapshot() async throws {
    try await withRecoveryFixture(mode: "posted") { fixture in
        try await fixture.wait { try fixture.ledger.snapshot().possibleButtons.contains(0) }
        try fixture.write("invalid-physical")
        #expect(kill(fixture.process.processIdentifier, SIGKILL) == 0)
        try await fixture.wait { try fixture.ledger.snapshot().phase == .recovering }
        try await Task.sleep(for: .milliseconds(30))
        fixture.assertLeaseHeld()
        #expect(try !fixture.ledger.snapshot().cleanupConfirmed)
        #expect(try !fixture.effects().contains { $0["cleanup"] == .bool(true) })
        try FileManager.default.removeItem(at: fixture.root.appendingPathComponent("invalid-physical"))
        try await fixture.wait { try fixture.ledger.snapshot().cleanupConfirmed }
    }
}
