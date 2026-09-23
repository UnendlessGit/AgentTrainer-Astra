import Foundation
import AstraCore

public struct NativeControlConfiguration: Sendable {
    public let runID: UUID
    public let scope: ControlScope
    public let capabilities: ActionCapabilities
    public let packetCapacity: Int
    public let initialPacketSequence: UInt64
    public let recoveryDirectory: URL
    public init(runID: UUID, scope: ControlScope, capabilities: ActionCapabilities, packetCapacity: Int = 16,
                initialPacketSequence: UInt64 = 0, recoveryDirectory: URL) throws {
        self.runID = runID; self.scope = try scope.validated(); self.capabilities = try capabilities.validated()
        guard !capabilities.isEmpty, [16, 32, 64].contains(packetCapacity), initialPacketSequence < UInt64.max else {
            throw AstraError("control.configuration", "A control session needs supported actions, packet capacity and sequence origin.")
        }
        self.packetCapacity = packetCapacity; self.initialPacketSequence = initialPacketSequence; self.recoveryDirectory = recoveryDirectory
    }
}

public struct NativeControlEvent: Sendable {
    public let sessionID: UUID
    public let runID: UUID
    public let message: WireMessage
}
public struct NativeControlCompletion: Sendable {
    public let sessionID: UUID
    public let runID: UUID
    public let exitStatus: Int32?
    public let cleanupConfirmed: Bool
    public let recoveredByGuardian: Bool
    public let issue: AstraError?
}

/// Shared by the app's inference/reset/learning owners. A disarmed but still
/// connected helper keeps this gate until its entire lifetime has joined.
public final class NativeControlOwner: @unchecked Sendable {
    public static let shared = NativeControlOwner()
    private let lock = NSLock()
    private var owner: UUID?
    private var unconfirmed: NativeControlCompletion?
    public init() {}
    public var priorCleanupJoined: Bool { lock.withLock { owner == nil && unconfirmed == nil } }
    /// Immutable evidence from a joined, unconfirmed helper. Reading this never
    /// acknowledges the warning or changes the historical cleanup result.
    public var pendingManualCleanup: NativeControlCompletion? { lock.withLock { unconfirmed } }
    fileprivate func acquire(_ id: UUID) throws {
        try lock.withLock {
            guard owner == nil, unconfirmed == nil else { throw AstraError("control.previousOwner", "The previous control owner has not joined confirmed cleanup.") }
            owner = id
        }
    }
    fileprivate func finish(_ value: NativeControlCompletion) {
        lock.withLock {
            guard owner == value.sessionID else { return }
            owner = nil; if !value.cleanupConfirmed { unconfirmed = value }
        }
    }
    /// Call only for an explicit human acknowledgement after an unconfirmed
    /// dead-owner result. This does not alter or upgrade the recorded proof.
    public func acknowledgeManualCleanup(sessionID: UUID) throws {
        try lock.withLock {
            guard owner == nil, unconfirmed?.sessionID == sessionID else { throw AstraError("control.manualCleanup", "There is no matching completed cleanup warning to acknowledge.") }
            unconfirmed = nil
        }
    }
}

public struct NativeControlRuntime: Sendable {
    public let start: @Sendable () async throws -> WireMessage
    /// Must preserve error envelopes and join its own I/O during shutdown.
    public let request: @Sendable (String, JSONValue, UUID, Duration) async throws -> WireMessage
    public let shutdown: @Sendable () async -> Int32?
    public init(start: @escaping @Sendable () async throws -> WireMessage,
                request: @escaping @Sendable (String, JSONValue, UUID, Duration) async throws -> WireMessage,
                shutdown: @escaping @Sendable () async -> Int32?) {
        self.start = start; self.request = request; self.shutdown = shutdown
    }
}
public struct NativeControlRuntimeFactory: Sendable {
    public let protectsPhysicalInputs: Bool
    public let make: @Sendable (@escaping ComputeProcess.EventHandler, @escaping ComputeProcess.FailureHandler) -> NativeControlRuntime
    /// Unprotected factories are exclusively for virtual, permission-free
    /// fixtures. Production callers use live(executable:), which always pairs.
    public init(protectsPhysicalInputs: Bool,
                make: @escaping @Sendable (@escaping ComputeProcess.EventHandler, @escaping ComputeProcess.FailureHandler) -> NativeControlRuntime) {
        self.protectsPhysicalInputs = protectsPhysicalInputs; self.make = make
    }
    public static func live(executable: URL) -> Self {
        .init(protectsPhysicalInputs: true) { events, failure in
            let process = ComputeProcess(executable: executable, expectedRole: "control",
                allowedEvents: ["control.receipt", "control.stopped", "control.error"], onEvent: events, onFailure: failure)
            return NativeControlRuntime(start: { try await process.start() }, request: { kind, payload, run, timeout in
                try await process.request(kind: kind, payload: payload, runID: run, timeout: timeout, acceptingError: true)
            }, shutdown: { await process.shutdown(); return await process.terminationStatus() })
        }
    }
}

private final class NativeReceiptPromise: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<ExecutionReceipt, any Error>?
    private var waiters: [CheckedContinuation<ExecutionReceipt, any Error>] = []
    func wait() async throws -> ExecutionReceipt {
        try await withCheckedThrowingContinuation { continuation in
            let result: Result<ExecutionReceipt, any Error>? = lock.withLock { if let result = self.result { return result }; waiters.append(continuation); return nil }
            if let result { continuation.resume(with: result) }
        }
    }
    func finish(_ result: Result<ExecutionReceipt, any Error>) {
        let callbacks = lock.withLock {
            guard self.result == nil else { return [CheckedContinuation<ExecutionReceipt, any Error>]() }
            self.result = result; let callbacks = waiters; waiters = []; return callbacks
        }
        callbacks.forEach { $0.resume(with: result) }
    }
}
public struct NativeControlSubmission: Sendable {
    public let packet: ActionPacket
    public let reply: WireMessage
    private let promise: NativeReceiptPromise
    public var admitted: Bool { reply.kind == "ack" && reply.payload.fields?["admitted"] == .bool(true) }
    fileprivate init(packet: ActionPacket, reply: WireMessage, promise: NativeReceiptPromise) { self.packet = packet; self.reply = reply; self.promise = promise }
    /// Independent from admission; cancellation cannot erase a real receipt.
    public func terminalReceipt() async throws -> ExecutionReceipt { try await promise.wait() }
}

private final class NativeHeartbeatFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var running = true
    var active: Bool { lock.withLock { running } }
    func stop() { lock.withLock { running = false } }
}

/// One fresh helper for one physical episode. All pending requests are drained
/// independently of caller cancellation; Stop disarms without destroying the
/// helper needed to reject a known-unsent late policy packet truthfully.
public final class NativeControlSession: @unchecked Sendable {
    public let sessionID = UUID()
    public let configuration: NativeControlConfiguration
    private let owner: NativeControlOwner
    private let factory: NativeControlRuntimeFactory
    private let onEvent: @Sendable (NativeControlEvent) throws -> Void
    private let onFailure: @Sendable (UUID, AstraError) -> Void
    private let lock = NSLock()
    private var runtime: NativeControlRuntime?
    private var recovery: ControlRecoveryLedger?
    private var ownsGate = false, armAttempted = false, armed = false, stopped = false, closing = false, callbacksOpen = true
    private var issue: AstraError?
    private var completedExecutionCount = 0
    public var executedPacketCount: Int { lock.withLock { completedExecutionCount } }
    private var nextSequence: UInt64
    private var startTask: Task<WireMessage, any Error>?
    private var disarmTask: Task<WireMessage, any Error>?
    private var shutdownTask: Task<NativeControlCompletion, Never>?
    private var submissionTail: Task<Void, Never>?
    private var heartbeat: Task<Void, Never>?
    private let heartbeatFlag = NativeHeartbeatFlag()
    private struct Pending {
        let packet: ActionPacket
        let promise: NativeReceiptPromise
        var timer: Task<Void, Never>?
        var requestID: UUID?
        var terminal: ExecutionReceipt?
        var replyAdmitted: Bool?
        var mustReject = false
        var admittedSeen = false
    }
    private var pending: [UUID: Pending] = [:]
    private var recent: [UUID] = []
    private var recentSet: Set<UUID> = []
    public init(configuration: NativeControlConfiguration, owner: NativeControlOwner = .shared,
                runtimeFactory: NativeControlRuntimeFactory,
                onEvent: @escaping @Sendable (NativeControlEvent) throws -> Void = { _ in },
                onFailure: @escaping @Sendable (UUID, AstraError) -> Void = { _, _ in }) {
        self.configuration = configuration; self.owner = owner; factory = runtimeFactory
        self.onEvent = onEvent; self.onFailure = onFailure; nextSequence = configuration.initialPacketSequence
    }
    deinit {
        heartbeatFlag.stop()
        let error = AstraError("control.abandoned", "The control-session owner was abandoned.")
        for value in pending.values { value.timer?.cancel(); value.promise.finish(.failure(error)) }
        guard ownsGate else { return }
        let runtime = runtime, recovery = recovery, owner = owner, id = sessionID, config = configuration, attempted = armAttempted, beating = heartbeat
        Task {
            if let runtime { _ = try? await runtime.request("disarm", .object([:]), config.runID, .seconds(5)) }
            await beating?.value
            let status = await runtime?.shutdown()
            let proof = await Self.recoveryProof(recovery, attempted: attempted, status: status)
            owner.finish(.init(sessionID: id, runID: config.runID, exitStatus: status, cleanupConfirmed: proof.0, recoveredByGuardian: proof.1, issue: error))
        }
    }
    public func checkHealth() throws {
        try lock.withLock {
            if let issue { throw issue }
            guard armed, !stopped, !closing else { throw AstraError("control.inactive", "This control session is not accepting policy actions.") }
        }
    }
    public func start() async throws -> WireMessage {
        let task = try lock.withLock { () throws -> Task<WireMessage, any Error> in
            guard !stopped, !closing else { throw AstraError("control.closed", "A stopped session cannot start a helper.") }
            if let startTask { return startTask }
            let task = Task { try await self.startWork() }; startTask = task; return task
        }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { self.requestStop() }
    }
    private func startWork() async throws -> WireMessage {
        do {
            try owner.acquire(sessionID); lock.withLock { ownsGate = true }
            try requireOpen()
            if factory.protectsPhysicalInputs {
                let directory = configuration.recoveryDirectory.appendingPathComponent(sessionID.uuidString.lowercased())
                let config = configuration
                let ledger = try await Task.detached {
                    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                    return try ControlRecoveryLedger(createAt: directory.appendingPathComponent("ownership.astracontrol"), runID: config.runID, capabilities: config.capabilities)
                }.value
                lock.withLock { recovery = ledger }
            }
            try requireOpen()
            let runtime = factory.make({ [weak self] in self?.receive($0) }, { [weak self] in self?.fail($0) })
            lock.withLock { self.runtime = runtime }
            try requireOpen()
            let hello = try await runtime.start()
            guard hello.kind == "hello", hello.payload.fields?["role"] == .string("control"), hello.payload.fields?["protocolVersion"] == .integer(1),
                  !factory.protectsPhysicalInputs || hello.payload.fields?["recoveryVersion"] == .integer(1),
                  configuration.initialPacketSequence == 0 || hello.payload.fields?["initialPacketSequenceVersion"] == .integer(1) else {
                throw AstraError("control.handshake", "The helper does not support this protected control session or packet sequence origin.")
            }
            try requireOpen()
            lock.withLock { armAttempted = true }
            let request = ArmRequest(runID: configuration.runID, scope: configuration.scope, capabilities: configuration.capabilities,
                packetCapacity: configuration.packetCapacity, recovery: lock.withLock { recovery?.descriptor },
                initialPacketSequence: configuration.initialPacketSequence == 0 ? nil : configuration.initialPacketSequence)
            let reply = try await runtime.request("arm", .encode(request), configuration.runID, .seconds(20))
            guard reply.runID == configuration.runID else { throw AstraError("control.armIdentity", "The helper replied to arm for a different run.") }
            if reply.kind == "error" {
                throw AstraError(reply.payload.fields?["code"]?.text ?? "control.arm", reply.payload.fields?["message"]?.text ?? "The helper rejected control arming.")
            }
            guard reply.kind == "ack", reply.payload.fields?["armed"] == .bool(true),
                  reply.payload.fields?["nextPacketSequence"]?.uint64 == configuration.initialPacketSequence else {
                throw AstraError("control.arm", "The helper did not arm with the requested packet sequence.")
            }
            if let recovery = lock.withLock({ recovery }) {
                let value = try recovery.snapshot()
                guard reply.payload.fields?["recoveryLedgerID"]?.uuid == recovery.descriptor.ledgerID,
                      reply.payload.fields?["guardianPID"]?.uint64 == UInt64(value.guardianPID), value.guardianReady,
                      value.everArmed, value.phase == .armed, value.executorPID > 0 else { throw AstraError("control.recoveryHandshake", "The helper did not establish independent cleanup protection.") }
            }
            lock.withLock { armed = true }
            if lock.withLock({ stopped || closing }) {
                _ = try? await rawDisarm(runtime); throw CancellationError()
            }
            startHeartbeat(runtime)
            return reply
        } catch {
            // A stop event can arrive before the arm rejection. Keep its
            // original cause even when the native arm unwinds as cancelled.
            if let issue = lock.withLock({ issue }) { throw issue }
            if !lock.withLock({ stopped }) { fail((error as? AstraError) ?? .init("control.start", error.localizedDescription)) }
            throw error
        }
    }
    private func startHeartbeat(_ runtime: NativeControlRuntime) {
        let flag = heartbeatFlag, run = configuration.runID
        let task = Task { [weak self] in
            while flag.active {
                do {
                    let reply = try await runtime.request("heartbeat", .object([:]), run, .seconds(1))
                    if !flag.active { return }
                    guard reply.kind == "ack", reply.runID == run else { throw AstraError("control.heartbeat", "Control heartbeat was rejected or changed identity.") }
                    try await Task.sleep(for: .milliseconds(100))
                } catch {
                    if flag.active { self?.fail((error as? AstraError) ?? .init("control.heartbeat", error.localizedDescription)) }
                    return
                }
            }
        }
        lock.withLock { heartbeat = task }
    }
    private enum SubmissionMode: Equatable { case liveOnly, lateOnly, preservingStop }
    public func submit(_ packet: ActionPacket) async throws -> NativeControlSubmission { try await submit(packet, mode: .liveOnly) }
    public func rejectLatePacket(_ packet: ActionPacket) async throws -> NativeControlSubmission { try await submit(packet, mode: .lateOnly) }
    /// One reservation decides whether this never-submitted produced packet
    /// enters live admission or obtains a real rejection after disarm. Never
    /// retry this API after an uncertain transport outcome.
    public func submitPreservingStoppedPacket(_ packet: ActionPacket) async throws -> NativeControlSubmission {
        try await submit(packet, mode: .preservingStop)
    }
    private func submit(_ packet: ActionPacket, mode: SubmissionMode) async throws -> NativeControlSubmission {
        _ = try packet.validated(capabilities: configuration.capabilities, surfaces: configuration.scope.surfaces, capacity: configuration.packetCapacity)
        let promise = NativeReceiptPromise()
        let task = try lock.withLock { () throws -> Task<NativeControlSubmission, any Error> in
            let late = mode == .lateOnly || (mode == .preservingStop && stopped)
            guard !closing, runtime != nil, packet.runID == configuration.runID, pending.count < 32,
                  pending[packet.id] == nil, !recentSet.contains(packet.id),
                  mode == .lateOnly || packet.sequence == nextSequence,
                  late ? stopped : (armed && !stopped) else {
                throw AstraError("control.admission", "The packet is stale, duplicated, out of order or belongs to closed control admission.")
            }
            if mode != .lateOnly { guard nextSequence < UInt64.max else { throw AstraError("control.sequence", "The control sequence is exhausted.") }; nextSequence += 1 }
            pending[packet.id] = Pending(packet: packet, promise: promise)
            recent.append(packet.id); recentSet.insert(packet.id)
            if recent.count > 2048 { recentSet.remove(recent.removeFirst()) }
            let previous = submissionTail
            let task = Task { [self] in
                await previous?.value
                do {
                    let stopped = lock.withLock { self.stopped }
                    if late || stopped { _ = try await disarm() }
                    lock.withLock { pending[packet.id]?.mustReject = late || stopped }
                    guard let runtime = lock.withLock({ runtime }) else { throw AstraError("control.closed", "The control helper has joined shutdown.") }
                    let reply = try await runtime.request("execute", .encode(packet), configuration.runID, .seconds(2))
                    try acceptReply(reply, packet: packet, mustReject: late || stopped)
                    return NativeControlSubmission(packet: packet, reply: reply, promise: promise)
                } catch {
                    promise.finish(.failure(error)); fail((error as? AstraError) ?? .init("control.submission", error.localizedDescription)); throw error
                }
            }
            submissionTail = Task { _ = try? await task.value }
            return task
        }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { self.requestStop() }
    }
    public func execute(_ packet: ActionPacket) async throws -> ExecutionReceipt {
        try await withTaskCancellationHandler {
            let submitted = try await submit(packet)
            let receipt = try await submitted.terminalReceipt()
            guard submitted.admitted, receipt.status == .executed else { throw AstraError("control.execution", "The reset packet was not executed successfully.") }
            return receipt
        } onCancel: { self.requestStop() }
    }
    public func observation(afterSequence: UInt64? = nil) async throws -> ControlObservation {
        try checkHealth()
        let payload: JSONValue = .object(afterSequence.map { ["afterSequence": .unsigned($0)] } ?? [:])
        guard let runtime = lock.withLock({ runtime }) else { throw AstraError("control.closed", "The control helper is unavailable.") }
        let run = configuration.runID
        let request = Task { try await runtime.request("observation", payload, run, .seconds(2)) }
        let reply = try await withTaskCancellationHandler { try await request.value } onCancel: { self.requestStop() }
        guard reply.kind == "ack", reply.runID == run else { throw AstraError("control.observation", "The helper rejected input observation.") }
        return try reply.payload.decode(ControlObservation.self)
    }
    public func requestStop() {
        let schedule = lock.withLock { stopped = true; return disarmTask == nil && !closing }
        heartbeatFlag.stop()
        if schedule { _ = ensureDisarmTask() }
    }
    public func disarm() async throws -> WireMessage {
        lock.withLock { stopped = true }; heartbeatFlag.stop()
        return try await ensureDisarmTask().value
    }
    private func ensureDisarmTask() -> Task<WireMessage, any Error> {
        lock.withLock {
            if let disarmTask { return disarmTask }
            let starting = startTask
            let task = Task { [self] in
                // Stop now, then again after an in-flight arm joins.
                if let runtime = lock.withLock({ runtime }) { _ = try? await rawDisarm(runtime) }
                _ = try? await starting?.value
                guard let runtime = lock.withLock({ runtime }) else { throw AstraError("control.notStarted", "No helper was started for this session.") }
                let reply = try await rawDisarm(runtime)
                lock.withLock { armed = false }
                return reply
            }
            disarmTask = task; return task
        }
    }
    private func rawDisarm(_ runtime: NativeControlRuntime) async throws -> WireMessage {
        let reply = try await runtime.request("disarm", .object([:]), configuration.runID, .seconds(5))
        guard reply.kind == "ack", reply.runID == configuration.runID, reply.payload.fields?["stopped"] == .bool(true),
              reply.payload.fields?["cleanupSettled"] == .bool(true) else { throw AstraError("control.disarm", "The helper has not acknowledged settled disarm.") }
        return reply
    }
    public func shutdown() async -> NativeControlCompletion {
        let task = lock.withLock { () -> Task<NativeControlCompletion, Never> in
            if let shutdownTask { return shutdownTask }
            stopped = true; closing = true; heartbeatFlag.stop()
            let task = Task { await self.shutdownWork() }; shutdownTask = task; return task
        }
        return await task.value
    }
    private func shutdownWork() async -> NativeControlCompletion {
        let starting = lock.withLock { startTask }; _ = try? await starting?.value
        let disarming = ensureDisarmTask(); _ = try? await disarming.value
        let sending = lock.withLock { submissionTail }; await sending?.value
        let beating = lock.withLock { heartbeat }; await beating?.value
        let runtime = lock.withLock { self.runtime }
        let status = await runtime?.shutdown()
        let proof = await Self.recoveryProof(lock.withLock { recovery }, attempted: lock.withLock { armAttempted }, status: status)
        let leftovers = lock.withLock { callbacksOpen = false; let values = Array(pending.values); pending = [:]; self.runtime = nil; return values }
        for value in leftovers { value.timer?.cancel(); value.promise.finish(.failure(AstraError("control.missingReceipt", "The joined helper did not finish this packet's receipt."))) }
        let failure = lock.withLock { issue }
        let finalIssue = proof.0 ? failure : AstraError("control.cleanupUnconfirmed", [failure?.message,
            "The helper exited before owned input cleanup could be confirmed. Release held controls manually before continuing."].compactMap { $0 }.joined(separator: "\n"))
        let result = NativeControlCompletion(sessionID: sessionID, runID: configuration.runID, exitStatus: status,
            cleanupConfirmed: proof.0, recoveredByGuardian: proof.1, issue: finalIssue)
        if lock.withLock({ ownsGate }) { owner.finish(result); lock.withLock { ownsGate = false } }
        return result
    }
    private func acceptReply(_ reply: WireMessage, packet: ActionPacket, mustReject: Bool) throws {
        guard reply.runID == configuration.runID, reply.requestID != nil,
              reply.kind == "error" || (reply.kind == "ack" && reply.payload.fields?["admitted"] == .bool(true)),
              !mustReject || reply.kind == "error" else { throw AstraError("control.admissionReply", "The helper returned an inconsistent admission or accepted a packet after disarm.") }
        let receipt = try lock.withLock { () throws -> ExecutionReceipt? in
            guard var value = pending[packet.id] else { throw AstraError("control.receiptIdentity", "The packet disappeared before its admission reply.") }
            guard value.requestID == nil || value.requestID == reply.requestID else { throw AstraError("control.receiptIdentity", "Admission and execution have different request identities.") }
            guard reply.kind != "error" || !value.admittedSeen else { throw AstraError("control.receiptIdentity", "A rejected packet already reported admission.") }
            if let receipt = value.terminal {
                guard reply.kind != "error" || receipt.status == .rejected,
                      reply.kind != "ack" || receipt.status != .rejected else { throw AstraError("control.receiptIdentity", "Admission and terminal execution evidence disagree.") }
            }
            value.requestID = reply.requestID; value.replyAdmitted = reply.kind == "ack"
            if let receipt = value.terminal { pending.removeValue(forKey: packet.id); return receipt }
            pending[packet.id] = value; return nil
        }
        if receipt == nil { scheduleReceiptDeadline(packet) }
        if !mustReject, reply.kind == "error" {
            // Keep the real rejection evidence awaitable, but never continue
            // after native/producer admission counters have diverged.
            let error = AstraError("control.rejected", reply.payload.fields?["message"]?.text ?? "The control helper rejected this packet.")
            let first = lock.withLock { let first = issue == nil; issue = issue ?? error; return first }
            requestStop()
            if first { onFailure(sessionID, error) }
        }
    }
    static func receiptWaitNanos(_ packet: ActionPacket, now: UInt64) -> UInt64 {
        // Called only for validated packets: the complete allowed lookahead
        // plus packet duration must elapse before terminal evidence is overdue.
        let end = packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000
        let remaining = end > now ? end - now : 0
        return min(remaining, ControlLease.maximumLookaheadNanos + 1_000_000_000) + 500_000_000
    }
    private func scheduleReceiptDeadline(_ packet: ActionPacket) {
        let nanos = Self.receiptWaitNanos(packet, now: MonotonicClock.now)
        let task = Task { [weak self] in
            do { try await Task.sleep(nanoseconds: nanos) } catch { return }
            guard let self, lock.withLock({ pending[packet.id] != nil }) else { return }
            fail(.init("control.receiptTimeout", "A control packet did not produce its terminal receipt."))
        }
        lock.withLock { if pending[packet.id] != nil { pending[packet.id]?.timer = task } else { task.cancel() } }
    }
    private func receive(_ message: WireMessage) {
        guard lock.withLock({ callbacksOpen }) else { return }
        var terminal: (NativeReceiptPromise, ExecutionReceipt)?
        do {
            if message.kind == "control.receipt" {
                guard message.runID == configuration.runID, let payload = message.payload.fields?["receipt"] else { throw AstraError("control.receipt", "The helper returned a foreign receipt.") }
                let receipt = try payload.decode(ExecutionReceipt.self)
                try lock.withLock {
                    guard var item = pending[receipt.packetID], receipt.runID == configuration.runID, receipt.sequence == item.packet.sequence,
                          receipt.resultingState.pointer.isFinite, message.requestID != nil,
                          item.requestID == nil || item.requestID == message.requestID, item.terminal == nil else {
                        throw AstraError("control.receipt", "A receipt is duplicated, unknown or belongs to another request.")
                    }
                    item.requestID = message.requestID
                    guard !item.mustReject || receipt.status == .rejected else { throw AstraError("control.disarmedExecution", "The helper reported input admission after acknowledged disarm.") }
                    if let admitted = item.replyAdmitted {
                        guard admitted ? receipt.status != .rejected : receipt.status == .rejected else {
                            throw AstraError("control.receiptIdentity", "Admission and terminal execution evidence disagree.")
                        }
                    }
                    if receipt.status == .executed {
                        try Self.validateExecuted(receipt, packet: item.packet)
                        completedExecutionCount += 1
                    }
                    if receipt.status != .admitted {
                        item.timer?.cancel(); item.terminal = receipt; terminal = (item.promise, receipt)
                        if item.replyAdmitted != nil { pending.removeValue(forKey: receipt.packetID) } else { pending[receipt.packetID] = item }
                    } else {
                        guard !item.admittedSeen else { throw AstraError("control.duplicateReceipt", "The helper repeated an admission receipt.") }
                        item.admittedSeen = true; pending[receipt.packetID] = item
                    }
                }
            } else if message.kind == "control.stopped" {
                guard message.runID == configuration.runID || (message.runID == nil && !lock.withLock({ armed })) else { throw AstraError("control.eventIdentity", "Control stopped for a foreign run.") }
                let cause = message.payload.fields?["cause"]?.text ?? "fault"
                if !lock.withLock({ stopped }) {
                    lock.withLock { issue = issue ?? .init("control." + cause, message.payload.fields?["reason"]?.text ?? "Desktop control stopped.") }
                    requestStop()
                }
            } else if message.kind == "control.error" {
                throw AstraError("control.helper", message.payload.fields?["message"]?.text ?? "The control helper failed.")
            } else { throw AstraError("control.event", "The helper returned an unsupported control event.") }
            try onEvent(.init(sessionID: sessionID, runID: configuration.runID, message: message))
            if let terminal { terminal.0.finish(.success(terminal.1)) }
        } catch {
            terminal?.0.finish(.failure(error))
            fail((error as? AstraError) ?? .init("control.event", error.localizedDescription))
        }
    }
    private func requireOpen() throws {
        try lock.withLock { if let issue { throw issue }; if stopped || closing { throw CancellationError() } }
    }
    private func fail(_ error: AstraError) {
        let result = lock.withLock { () -> (Bool, [Pending]) in
            guard callbacksOpen else { return (false, []) }
            let first = issue == nil; if first { issue = error }
            let entries = Array(pending.values); return (first, entries)
        }
        for value in result.1 { value.timer?.cancel(); value.promise.finish(.failure(error)) }
        requestStop()
        if result.0 { onFailure(sessionID, error) }
    }
    private static func validateExecuted(_ receipt: ExecutionReceipt, packet: ActionPacket) throws {
        guard receipt.commandResults.count == packet.commands.count, Set(receipt.commandResults.map(\.commandIndex)) == Set(packet.commands.indices),
              receipt.commandResults.allSatisfy({ value in
                  guard packet.commands.indices.contains(value.commandIndex) else { return false }
                  let time = packet.executeAtNanos + UInt64(packet.commands[value.commandIndex].offsetMs) * 1_000_000
                  return value.scheduledNanos == time && (value.status == .posted || value.status == .noOp)
                    && (value.status != .posted || value.postedNanos != nil)
                    && (value.postedNanos.map { $0 >= time && $0 <= receipt.observedNanos } ?? true)
              }) else { throw AstraError("control.executionReceipt", "Executed control has incomplete or inconsistent command evidence.") }
    }
    private static func recoveryProof(_ ledger: ControlRecoveryLedger?, attempted: Bool, status: Int32?) async -> (Bool, Bool) {
        guard let ledger else { return (!attempted || status == 0, false) }
        while true {
            do {
                let value = try ledger.snapshot()
                let safe = value.cleanupConfirmed || (!value.everArmed && value.possibleKeys.isEmpty && value.possibleButtons.isEmpty && value.inFlight == 0)
                let guardianAlive = ledger.matchingGuardianIsAlive(value)
                if safe, !guardianAlive { return (true, value.recoveredByGuardian) }
                if !safe, !guardianAlive { return (false, false) }
                try? await Task.sleep(for: .milliseconds(25))
            } catch { return (false, false) }
        }
    }
}
