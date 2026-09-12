import Foundation
import Darwin
import AstraCore
import AstraPlatform

/// A non-shipping process fault fixture. Effects are private JSON files, never
/// Core Graphics events; it does not call privacy preflight or event-tap APIs.
private final class FileInputBackend: ControlInputBackend, @unchecked Sendable {
    let root: URL
    let mode: String
    private let lock = NSLock()
    init(root: URL, mode: String) { self.root = root; self.mode = mode }
    func prepare(_ request: ArmRequest) throws -> ControlState { physicalState() }
    func physicalState() -> ControlState {
        var value = ControlState(); value.valid = !exists("invalid-physical"); value.pointer = .init(x: 50, y: 50)
        if let data = try? Data(contentsOf: root.appendingPathComponent("physical.json")),
           let fields = try? JSONSerialization.jsonObject(with: data) as? [String: [Int]] {
            value.keys = Set(fields["keys"] ?? []); value.buttons = Set(fields["buttons"] ?? [])
        }
        value.observedNanos = MonotonicClock.now; return value
    }
    func checkHealth() throws {}
    func validate(_ scope: ControlScope, pointer: Point2D?) throws {}
    func exists(_ name: String) -> Bool { FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path) }
    func waitForGate(_ name: String) throws {
        let deadline = Date().addingTimeInterval(10)
        while !exists(name), Date() < deadline { Thread.sleep(forTimeInterval: 0.005) }
        guard exists(name) else { throw AstraError("fixture.gateTimeout", "The private fixture gate was not released.") }
    }
    func mark(_ name: String) throws { try Data().write(to: root.appendingPathComponent(name)) }
    func post(_ emission: InputEmission) throws {
        if emission.cleanup, exists("deny-release") { throw AstraError("fixture.releaseDenied", "Fixture cleanup permission is denied.") }
        if emission.operation == .keyDown, !emission.cleanup, mode == "reserved" {
            try mark("post-entered"); try waitForGate("allow-post")
        }
        try lock.withLock {
            let destination = root.appendingPathComponent("effects.jsonl")
            let handle = try FileHandle(forWritingTo: destination); defer { try? handle.close() }
            try handle.seekToEnd()
            var effect: [String: Any] = ["operation": emission.operation.rawValue, "cleanup": emission.cleanup, "pid": getpid()]
            if let key = emission.keyCode { effect["key"] = key }
            if let button = emission.button { effect["button"] = button }
            var data = try JSONSerialization.data(withJSONObject: effect, options: [.sortedKeys]); data.append(10)
            try handle.write(contentsOf: data)
        }
        if emission.operation == .keyDown, !emission.cleanup {
            try mark("down-posted")
            if mode == "inflight" { try mark("post-entered"); try waitForGate("allow-post") }
        }
        if emission.operation == .keyUp, !emission.cleanup, mode == "after-release" {
            try mark("release-posted"); try waitForGate("allow-post")
        }
    }
}

@main struct RecoveryFixtureMain {
    static func main() async {
        guard let path = ProcessInfo.processInfo.environment["ASTRA_RECOVERY_FIXTURE_ROOT"] else { exit(20) }
        let root = URL(fileURLWithPath: path)
        let mode = ProcessInfo.processInfo.environment["ASTRA_RECOVERY_FIXTURE_MODE"] ?? "posted"
        let backend = FileInputBackend(root: root, mode: mode)
        if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--guardian", let owner = Int32(CommandLine.arguments[2]) {
            if backend.exists("block-bootstrap") { try? backend.mark("bootstrap-waiting"); try? backend.waitForGate("allow-bootstrap") }
            let status = ControlGuardian.run(ownerPID: owner, backend: backend, permissionsReady: {
                try? backend.mark("watch-registered")
                if backend.exists("block-ready") { try? backend.waitForGate("allow-ready") }
                return !backend.exists("deny-ready")
            }, onFailure: {
                try? Data($0.localizedDescription.utf8).write(to: root.appendingPathComponent("guardian-error.txt"))
            }, registerExitWatch: backend.exists("deny-watch") ? { _, parent in
                if getppid() == parent { try? backend.mark("watch-parent-alive") }
                return ESRCH
            } : nil)
            try? Data(String(status).utf8).write(to: root.appendingPathComponent("guardian-status.txt"))
            exit(status)
        }
        do {
            let descriptor = try JSONDecoder().decode(ControlRecoveryDescriptor.self, from: Data(contentsOf: root.appendingPathComponent("descriptor.json")))
            let ledger = try ControlRecoveryLedger(open: descriptor)
            let lease = try DesktopControlLock(url: root.appendingPathComponent("desktop.lock"))
            let executor = InputExecutor(backend: backend)
            let pair = try ControlGuardianPair(ledger: ledger, desktopLease: lease,
                executable: URL(fileURLWithPath: CommandLine.arguments[0]), onExit: {
                    executor.requestDisarm(runID: descriptor.runID, reason: "Fixture guardian exited.")
                })
            try JSONSerialization.data(withJSONObject: ["executorPID": getpid(), "guardianPID": pair.guardianPID])
                .write(to: root.appendingPathComponent("processes.json"))
            do {
                try await pair.waitUntilReady()
                let request = ArmRequest(runID: descriptor.runID, scope: .init(surfaces: [
                    SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 100, height: 100), pixelWidth: 100, pixelHeight: 100)
                ], wholeDesktop: true), capabilities: .init(keyCodes: [0, 55], mouseButtons: [0]), recovery: descriptor)
                try executor.arm(request, recovery: ledger, desktopLease: lease)
                var commands = [TimedCommand(offsetMs: 0, operation: .keyDown, keyCode: 0), TimedCommand(offsetMs: 2, operation: .buttonDown, button: 0)]
                if mode == "after-release" { commands = [commands[0], TimedCommand(offsetMs: 20, operation: .keyUp, keyCode: 0)] }
                try executor.execute(ActionPacket(runID: descriptor.runID, sequence: 0, observationID: UUID(), geometryRevision: 0,
                    executeAtNanos: MonotonicClock.now + 30_000_000, durationMs: 100, commands: commands))
                while executor.currentRunID != nil, !backend.exists("stop") {
                    try? executor.heartbeat(runID: descriptor.runID)
                    try await Task.sleep(for: .milliseconds(20))
                }
            } catch { try? Data(error.localizedDescription.utf8).write(to: root.appendingPathComponent("error.txt")) }
            executor.requestDisarm(reason: "Fixture stopping.")
            while !executor.cleanupSettled { try? await Task.sleep(for: .milliseconds(5)) }
            ledger.stop(); ledger.settleLocally(now: MonotonicClock.now)
            await pair.finishAfterLocalSettlement()
            try? backend.mark("executor-settled")
        } catch { try? Data(error.localizedDescription.utf8).write(to: root.appendingPathComponent("error.txt")); exit(21) }
    }
}
