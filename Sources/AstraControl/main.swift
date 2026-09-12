import Foundation
import AppKit
import Darwin
import AstraCore
import AstraPlatform

/// Nonblocking stdout cannot stall input cleanup. Overflow is terminal and
/// invokes the executor's stop path instead of retaining unbounded receipts.
private final class ControlOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var queue: [Data] = []
    private var offset = 0
    private var bytes = 0
    private var sequence: UInt64 = 0
    private var failed = false
    private var failureHandler: @Sendable () -> Void = {}
    private let timer: DispatchSourceTimer
    init() {
        let flags = fcntl(STDOUT_FILENO, F_GETFL)
        failed = flags < 0 || fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) < 0
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "astra.control.output", qos: .userInteractive))
        timer.schedule(deadline: .now(), repeating: .milliseconds(2), leeway: .milliseconds(1))
        timer.setEventHandler { [weak self] in self?.drain() }; timer.resume()
    }
    deinit { timer.cancel() }
    var healthy: Bool { lock.withLock { !failed } }
    func onFailure(_ handler: @escaping @Sendable () -> Void) { lock.withLock { failureHandler = handler } }
    func send(_ kind: String, _ payload: JSONValue, request: WireMessage? = nil, runID: UUID? = nil) {
        let failure = lock.withLock { () -> (@Sendable () -> Void)? in
            guard !failed else { return nil }
            do {
                guard sequence < UInt64.max else { throw AstraError("control.sequence", "Output sequence exhausted.") }
                let data = try WireMessage(kind: kind, sequence: sequence, requestID: request?.requestID,
                                           runID: runID ?? request?.runID, payload: payload).framed()
                guard bytes + data.count <= 4 * AstraVersion.maximumMessageBytes else { throw AstraError("control.output", "The receipt consumer stopped reading.") }
                sequence += 1; bytes += data.count; queue.append(data)
                return nil
            } catch { failed = true; queue.removeAll(); bytes = 0; return failureHandler }
        }
        failure?()
    }
    private func drain() {
        let failure = lock.withLock { () -> (@Sendable () -> Void)? in
            guard !failed else { return nil }
            var budget = 65_536
            while let first = queue.first, budget > 0 {
                let written = first.withUnsafeBytes { raw in
                    Darwin.write(STDOUT_FILENO, raw.baseAddress!.advanced(by: offset), min(first.count - offset, budget))
                }
                if written < 0 {
                    if errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR { return nil }
                    failed = true; queue.removeAll(); bytes = 0; return failureHandler
                }
                if written == 0 { return nil }
                offset += written; bytes -= written; budget -= written
                if offset == first.count { queue.removeFirst(); offset = 0 }
            }
            return nil
        }
        failure?()
    }
    func finish() async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(1))
        while lock.withLock({ !failed && bytes > 0 }), ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(5))
        }
    }
}

private final class ReceiptRoutes: @unchecked Sendable {
    private let lock = NSLock()
    private var requests: [UUID: WireMessage] = [:]
    func remember(_ packet: ActionPacket, request: WireMessage) throws {
        try lock.withLock {
            guard requests.count < 32, requests[packet.id] == nil else { throw AstraError("control.receipts", "The receipt queue is full or packet identity was repeated.") }
            requests[packet.id] = request
        }
    }
    func request(for receipt: ExecutionReceipt) -> WireMessage? {
        lock.withLock {
            let request = requests[receipt.packetID]
            if receipt.status != .admitted { requests.removeValue(forKey: receipt.packetID) }
            return request
        }
    }
}

private final class MonitorFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: AstraError?
    func record(_ error: AstraError) { lock.withLock { stored = stored ?? error } }
    func check() throws { if let error = lock.withLock({ stored }) { throw error } }
}

private struct ObservationRequest: Decodable {
    var afterSequence: UInt64?
}

private actor ControlServer {
    private let executor: InputExecutor
    private let output: ControlOutput
    private let routes: ReceiptRoutes
    private let monitor = PhysicalInputMonitor()
    private var protection: ControlGuardianPair?
    private var sequence: UInt64 = 0
    private var closing = false

    init(output: ControlOutput) throws {
        self.output = output
        let routes = ReceiptRoutes(); self.routes = routes
        executor = InputExecutor(backend: try CGEventControlBackend(), desktopLockURL: DesktopControlLock.standardURL,
            onReceipt: { receipt in
                if let value = try? JSONValue.encode(receipt) {
                    output.send("control.receipt", .object(["receipt": value]), request: routes.request(for: receipt), runID: receipt.runID)
                }
            }, onStop: { runID, cause, reason in output.send("control.stopped", .object(["cause": .string(cause.rawValue), "reason": .string(reason)]), runID: runID) })
        let executor = executor
        output.onFailure { [weak executor] in executor?.requestDisarm(reason: "The control output channel failed.") }
    }
    // Signals and sleep must invalidate an in-progress arm even if the server
    // actor is occupied by a synchronous WindowServer or permission query.
    nonisolated func stopImmediately(reason: String, cause: ControlStopCause) {
        executor.requestDisarm(reason: reason, cause: cause)
    }
    func handle(_ message: WireMessage) async -> Bool {
        guard !closing, output.healthy else { await disconnect(); return false }
        do {
            guard message.sequence == sequence, sequence < UInt64.max, message.requestID != nil else {
                throw AstraError("control.protocol", "Control requests require ordered sequence and request identity.", recoverable: false)
            }
            sequence += 1
            let payload: JSONValue
            switch message.kind {
            case "ping": payload = .object(["alive": .bool(true)])
            case "permissions": payload = try .encode(PermissionSnapshot.current())
            case "arm":
                let request = try message.payload.decode(ArmRequest.self)
                guard message.runID == request.runID else { throw AstraError("control.session", "Arm request identity does not match its envelope.") }
                guard executor.currentRunID == nil, protection == nil else { throw AstraError("control.busy", "Disarm the current control run before arming another.") }
                guard let recovery = request.recovery, recovery.runID == request.runID else {
                    throw AstraError("control.recoveryRequired", "A protected recovery ledger is required before desktop control can arm.")
                }
                _ = try request.scope.validated(); _ = try request.capabilities.validated()
                let journal = try ControlRecoveryLedger(open: recovery)
                guard journal.matches(request.capabilities) else { throw AstraError("control.recoveryCapabilities", "Recovery controls do not match this run's capabilities.") }
                let lease = try DesktopControlLock()
                let executor = executor
                let pair = try ControlGuardianPair(ledger: journal, desktopLease: lease,
                    executable: URL(fileURLWithPath: CommandLine.arguments[0]), onExit: {
                        executor.requestDisarm(runID: request.runID, reason: "The independent cleanup guardian exited.")
                    })
                protection = pair
                await monitor.stop()
                do {
                    try await pair.waitUntilReady()
                    let monitoring = MonitorFailure()
                    try await monitor.start(onEvents: { events in
                        if request.scope.stopOnPhysicalInput, events.contains(where: { $0.origin == .physical }) {
                            monitoring.record(AstraError("control.physicalTakeover", "Physical input interrupted control arming or execution."))
                            executor.requestDisarm(runID: request.runID, reason: "Physical input took over desktop control.", cause: .physicalTakeover)
                        }
                    }, onFault: { error in monitoring.record(error); executor.requestDisarm(runID: request.runID, reason: error.localizedDescription) },
                       onEmergency: {
                           monitoring.record(AstraError("control.emergency", "Emergency stop interrupted control arming or execution."))
                           executor.requestDisarm(runID: request.runID, reason: "Emergency stop.", cause: .emergencyStop)
                       })
                    try monitoring.check()
                    guard !closing else { throw AstraError("control.closing", "The control helper is shutting down.") }
                    try executor.arm(request, recovery: journal, desktopLease: lease)
                    try monitoring.check()
                } catch {
                    executor.requestDisarm(reason: "Control arming failed."); await monitor.stop()
                    journal.stop()
                    if executor.cleanupSettled { journal.settleLocally(now: MonotonicClock.now) }
                    if (try? journal.snapshot().cleanupConfirmed) == true {
                        await pair.finishAfterLocalSettlement(); protection = nil
                    }
                    throw error
                }
                payload = .object(["armed": .bool(true), "leaseNanos": .unsigned(ControlLease.durationNanos),
                                   "recoveryLedgerID": .string(journal.descriptor.ledgerID.uuidString),
                                   "guardianPID": .integer(Int64(pair.guardianPID))])
            case "heartbeat":
                guard let runID = message.runID else { throw AstraError("control.session", "Heartbeat requires a run identity.") }
                try executor.heartbeat(runID: runID); payload = .object(["alive": .bool(true)])
            case "execute":
                let packet = try message.payload.decode(ActionPacket.self)
                guard packet.runID == message.runID else { throw AstraError("control.session", "Packet identity does not match its envelope.") }
                try routes.remember(packet, request: message)
                try executor.execute(packet); payload = .object(["admitted": .bool(true)])
            case "disarm":
                if let expected = message.runID, let current = executor.currentRunID, expected != current {
                    throw AstraError("control.session", "Disarm belongs to a different active control run.")
                }
                executor.requestDisarm(reason: "Desktop control stopped.", cause: .requested); await monitor.stop()
                try await awaitSettlement()
                payload = .object(["stopped": .bool(true), "cleanupSettled": .bool(true)])
            case "observation":
                guard let runID = message.runID, executor.currentRunID == runID else {
                    throw AstraError("control.session", "Input observation requires the active control run.")
                }
                let request = try message.payload.decode(ObservationRequest.self)
                payload = try .encode(executor.observation(afterSequence: request.afterSequence))
            case "state": payload = .object(["state": try .encode(executor.state()), "cleanupSettled": .bool(executor.cleanupSettled)])
            case "shutdown":
                executor.requestDisarm(reason: "Control helper shutdown.", cause: .shutdown); await monitor.stop()
                try await awaitSettlement(); closing = true
                output.send("ack", .object(["stopped": .bool(true), "cleanupSettled": .bool(true)]), request: message); return false
            default: throw AstraError("protocol.unsupportedOperation", "Unsupported control operation.")
            }
            output.send("ack", payload, request: message)
        } catch {
            let value = (error as? AstraError) ?? AstraError("control.request", error.localizedDescription)
            output.send("error", (try? .encode(value)) ?? .object(["message": .string("The control request failed.")]), request: message)
            if !value.recoverable { await disconnect(); return false }
        }
        return true
    }
    private func awaitSettlement() async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while !executor.cleanupSettled {
            guard ContinuousClock.now < deadline else {
                throw AstraError("control.cleanupPending", "Control admission stopped, but an input post or owned release is still pending. Keep the helper alive until cleanup settles.")
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        if let protection {
            protection.ledger.stop(); protection.ledger.settleLocally(now: MonotonicClock.now)
            await protection.finishAfterLocalSettlement(); self.protection = nil
        }
    }
    func disconnect() async {
        closing = true; executor.requestDisarm(reason: "The control owner disconnected.", cause: .disconnected); await monitor.stop()
        // The parent disappearing must not kill a helper while a late post can
        // still revive a hold. The independent watchdog continues retrying failed
        // releases and retains the desktop lock until settlement, even after EOF.
        while !executor.cleanupSettled { try? await Task.sleep(for: .milliseconds(25)) }
        if let protection {
            protection.ledger.stop(); protection.ledger.settleLocally(now: MonotonicClock.now)
            await protection.finishAfterLocalSettlement(); self.protection = nil
        }
    }
}

@main struct AstraControlMain {
    static func main() async {
        if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--guardian", let owner = Int32(CommandLine.arguments[2]) {
            exit(ControlGuardian.runNative(ownerPID: owner))
        }
        signal(SIGPIPE, SIG_IGN)
        let output = ControlOutput()
        do {
            let server = try ControlServer(output: output)
            output.send("hello", .object(["role": .string("control"), "protocolVersion": .integer(1), "recoveryVersion": .integer(1),
                "capabilities": .array(["ping", "permissions", "arm", "heartbeat", "execute", "disarm", "observation", "state", "shutdown"].map(JSONValue.string))]))
            let sleep = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.willSleepNotification, object: nil, queue: nil) { @Sendable _ in
                server.stopImmediately(reason: "The Mac is going to sleep.", cause: .sleep)
            }
            let terminationSignals = [SIGTERM, SIGINT, SIGHUP].map { number in
                signal(number, SIG_IGN)
                let source = DispatchSource.makeSignalSource(signal: number, queue: DispatchQueue.global(qos: .userInteractive))
                source.setEventHandler { @Sendable in
                    server.stopImmediately(reason: "The control helper received a termination signal.", cause: .shutdown)
                    Task {
                        await server.disconnect()
                        await output.finish()
                        exit(0)
                    }
                }
                source.resume()
                return source
            }
            await Task.detached {
                var framer = MessageFramer()
                do {
                    var buffer = [UInt8](repeating: 0, count: 16_384)
                    reading: while true {
                        let count = Darwin.read(STDIN_FILENO, &buffer, buffer.count)
                        if count == 0 { try framer.finish(); break }
                        if count < 0 { if errno == EINTR { continue }; throw AstraError("control.input", "The command pipe failed.") }
                        for message in try framer.append(Data(buffer.prefix(count))) {
                            if !(await server.handle(message)) { break reading }
                        }
                    }
                } catch { output.send("control.error", .object(["message": .string("The control command stream was closed because it was invalid.")])) }
                await server.disconnect()
            }.value
            terminationSignals.forEach { $0.cancel() }
            NSWorkspace.shared.notificationCenter.removeObserver(sleep)
        } catch { output.send("control.error", .object(["message": .string(error.localizedDescription)])) }
        await output.finish()
    }
}
