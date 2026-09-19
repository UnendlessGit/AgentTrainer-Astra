import Foundation
import ApplicationServices
import AppKit
import IOKit.hidsystem
import Darwin
import AstraCore

public struct InputEmission: Sendable {
    public var operation: CommandOperation
    public var keyCode: Int?
    public var button: Int?
    public var location: Point2D
    public var delta: Point2D
    public var heldKeys: Set<Int>
    public var heldButtons: Set<Int>
    public var cleanup = false
}

/// An atomic causal cut through posted input. `lastSequence` acknowledges only
/// history already delivered to this single control owner; state remains valid
/// during silence, but never while posting or cleanup is unsettled.
public struct ControlObservation: Codable, Sendable {
    public var controlState: ControlState
    public var executedEvents: [RawInputEvent]
    public var intervalCovered: Bool
    public var cutoffNanos: UInt64
    public var lastSequence: UInt64?
}

public enum ControlStopCause: String, Codable, Sendable {
    case physicalTakeover, emergencyStop, requested, shutdown, disconnected, sleep, fault
}

public protocol ControlInputBackend: Sendable {
    func prepare(_ request: ArmRequest) throws -> ControlState
    func physicalState() -> ControlState
    func cleanupPhysicalState() -> ControlState
    /// Called on the independent watchdog; read cached proofs only, without OS IPC.
    func checkHealth() throws
    func validate(_ scope: ControlScope, pointer: Point2D?) throws
    func post(_ emission: InputEmission) throws
}

public extension ControlInputBackend {
    func cleanupPhysicalState() -> ControlState { physicalState() }
}

/// One lease for this macOS user, independent of the selected Astra library.
public final class DesktopControlLock: @unchecked Sendable {
    public static var standardURL: URL { URL(fileURLWithPath: "/tmp/com.unendless.astra.desktop-\(getuid()).lock") }
    private let descriptor: Int32
    public init(url: URL = standardURL) throws {
        descriptor = open(url.path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw AstraError("control.lock", "Desktop control ownership could not be opened.") }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_uid == getuid(), info.st_nlink == 1,
              flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            throw AstraError("control.busy", "Another Astra process owns desktop control, or its lock is invalid.")
        }
    }
    /// Duplicates share one flock. Explicit LOCK_UN would also unlock the
    /// surviving guardian's reference; closing our own handle is sufficient.
    deinit { close(descriptor) }
    public init(inheritedDescriptor: Int32) throws {
        descriptor = fcntl(inheritedDescriptor, F_DUPFD_CLOEXEC, 64)
        guard descriptor >= 0 else { throw AstraError("control.lock", "The inherited desktop lease is unavailable.") }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_uid == getuid(), info.st_nlink == 1 else {
            close(descriptor); throw AstraError("control.lock", "The inherited desktop lease is invalid.")
        }
    }
    public func withFileDescriptor<T>(_ body: (Int32) throws -> T) rethrows -> T { try body(descriptor) }
}

/// The scheduler reserves possible owned holds before posting outside its lock.
/// A racing stop releases those holds immediately, then the returning stale post
/// performs cleanup again. A blocked backend therefore cannot block the watchdog
/// from invalidating the generation or revive a hold after cleanup.
public final class InputExecutor: @unchecked Sendable {
    public typealias ReceiptHandler = @Sendable (ExecutionReceipt) -> Void
    private struct Scheduled {
        var time: UInt64; var packet: ActionPacket; var command: TimedCommand?
        var index: Int?; var order: Int
    }
    private struct Pending { var packet: ActionPacket; var results: [CommandResult] = [] }
    private let lock = NSLock()
    private let backend: any ControlInputBackend
    private let clock: @Sendable () -> UInt64
    private let onReceipt: ReceiptHandler
    private let onStop: @Sendable (UUID?, ControlStopCause, String) -> Void
    private let desktopLockURL: URL?
    private let maximumLateness: UInt64
    private var timer: DispatchSourceTimer?
    private var watchdog: DispatchSourceTimer?
    private var lease = ControlLease()
    private var desktopLock: DesktopControlLock?
    private var recovery: ControlRecoveryLedger?
    private var observed = ControlState()
    private var ownedKeys: Set<Int> = [], ownedButtons: Set<Int> = []
    private var scheduled: [Scheduled] = []
    private var pending: [UUID: Pending] = [:]
    private var epoch: UInt64 = 0
    private var driving = false
    private var inFlightPacketID: UUID?
    private var arming = false
    private var armingRunID: UUID?
    private var interruptionRunID: UUID?
    private var cleaning = 0
    private var lastEnd: UInt64 = 0
    private var interruption: String?
    private var interruptionCause: ControlStopCause?
    private var executedEvents: [RawInputEvent] = []
    private var nextEventSequence: UInt64 = 0
    private var acknowledgedEventSequence: UInt64?
    private var deliveredEventSequence: UInt64?
    private var historyCovered = true
    private let historyCapacity: Int
    private let stopQueue = DispatchQueue(label: "astra.control.stop", qos: .userInteractive)

    public init(backend: any ControlInputBackend, clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now },
                automaticScheduling: Bool = true, desktopLockURL: URL? = nil, maximumLatenessNanos: UInt64 = 20_000_000,
                historyCapacity: Int = 2048,
                onReceipt: @escaping ReceiptHandler = { _ in }, onStop: @escaping @Sendable (UUID?, ControlStopCause, String) -> Void = { _, _, _ in }) {
        self.backend = backend; self.clock = clock; self.onReceipt = onReceipt; self.onStop = onStop
        precondition((1...2048).contains(historyCapacity))
        self.desktopLockURL = desktopLockURL; self.maximumLateness = maximumLatenessNanos
        self.historyCapacity = historyCapacity
        if automaticScheduling {
            let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "astra.control.deadlines", qos: .userInteractive))
            timer.schedule(deadline: .now(), repeating: .milliseconds(1), leeway: .microseconds(100))
            timer.setEventHandler { [weak self] in self?.service() }; self.timer = timer; timer.resume()
            let watchdog = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "astra.control.watchdog", qos: .userInteractive))
            watchdog.schedule(deadline: .now(), repeating: .milliseconds(10), leeway: .milliseconds(1))
            watchdog.setEventHandler { [weak self] in self?.checkWatchdog() }; self.watchdog = watchdog; watchdog.resume()
        }
    }
    deinit { timer?.cancel(); watchdog?.cancel(); disarm(reason: "The control executor was released.", cause: .shutdown) }

    public func state() -> ControlState {
        lock.withLock {
            var value = observed
            value.valid = value.valid && lease.request != nil && clock() < lease.expiresAtNanos && inFlightPacketID == nil && !arming && cleaning == 0 && interruption == nil
            return value
        }
    }
    public var currentRunID: UUID? { lock.withLock { lease.request?.runID } }
    public var nextPacketSequence: UInt64 { lock.withLock { lease.nextSequence } }
    /// Disarm invalidates admission immediately. Settlement additionally proves
    /// every possibly posted hold was released or transferred to a physical hold.
    public var cleanupSettled: Bool {
        lock.withLock { lease.request == nil && !arming && !driving && inFlightPacketID == nil && cleaning == 0 && ownedKeys.isEmpty && ownedButtons.isEmpty && interruption == nil }
    }

    public func observation(afterSequence: UInt64? = nil) throws -> ControlObservation {
        try lock.withLock {
            let cutoff = clock()
            if let afterSequence {
                guard let deliveredEventSequence, afterSequence <= deliveredEventSequence else { throw AstraError("control.historyCursor", "The input-history acknowledgement was never produced by this run.") }
                if let acknowledgedEventSequence, afterSequence < acknowledgedEventSequence {
                    throw AstraError("control.historyCursor", "The input-history acknowledgement precedes already released history.")
                }
                acknowledgedEventSequence = afterSequence
                executedEvents.removeAll { $0.sequence <= afterSequence }
            } else if acknowledgedEventSequence != nil {
                throw AstraError("control.historyCursor", "This input history requires its previous acknowledgement cursor.")
            }
            var value = observed
            value.valid = value.valid && lease.request != nil && cutoff < lease.expiresAtNanos && inFlightPacketID == nil && !arming && cleaning == 0 && interruption == nil
            deliveredEventSequence = nextEventSequence == 0 ? nil : nextEventSequence - 1
            return ControlObservation(controlState: value, executedEvents: executedEvents, intervalCovered: historyCovered,
                                      cutoffNanos: cutoff, lastSequence: nextEventSequence == 0 ? nil : nextEventSequence - 1)
        }
    }

    public func arm(_ request: ArmRequest, recovery: ControlRecoveryLedger? = nil, desktopLease: DesktopControlLock? = nil) throws {
        let generation = try lock.withLock { () throws -> UInt64 in
            guard !arming, !driving, cleaning == 0, lease.request == nil, interruption == nil, ownedKeys.isEmpty, ownedButtons.isEmpty else { throw AstraError("control.busy", "Desktop control is still armed or cleaning up.") }
            arming = true; armingRunID = request.runID; return epoch
        }
        defer { lock.withLock { arming = false; armingRunID = nil; settleRecoveryIfQuiescent() } }
        _ = try request.scope.validated(); _ = try request.capabilities.validated()
        if let recovery {
            let snapshot = try recovery.snapshot()
            guard recovery.descriptor.runID == request.runID, recovery.matches(request.capabilities), snapshot.phase == .preparing,
                  snapshot.guardianReady, snapshot.executorPID == getpid(), snapshot.guardianPID > 0,
                  clock() >= snapshot.guardianHeartbeatNanos, clock() - snapshot.guardianHeartbeatNanos < 150_000_000 else {
                throw AstraError("control.guardianUnavailable", "Independent input cleanup protection is not ready for this run.")
            }
        }
        let owner = try desktopLease ?? desktopLockURL.map { try DesktopControlLock(url: $0) }
        let initial = try backend.prepare(request)
        try backend.checkHealth()
        guard initial.valid, initial.pointer.isFinite, initial.keys.isEmpty, initial.buttons.isEmpty else {
            throw AstraError("control.physicalHold", "Release all physical keys and mouse buttons before arming the agent.")
        }
        try lock.withLock {
            guard epoch == generation else { throw AstraError("control.cancelled", "Control was stopped while arming.") }
            self.recovery = recovery
            guard recovery?.arm() ?? true else { throw AstraError("control.guardianStopped", "Input cleanup protection was stopped while arming.") }
            try lease.arm(request, now: clock()); desktopLock = owner
            observed = initial; ownedKeys = []; ownedButtons = []; lastEnd = 0
            executedEvents = []; nextEventSequence = 0; acknowledgedEventSequence = nil; deliveredEventSequence = nil; historyCovered = true
        }
    }

    public func heartbeat(runID: UUID) throws {
        do { try lock.withLock { guard interruption == nil else { throw AstraError("control.stopping", "Desktop control is stopping.") }; try lease.heartbeat(runID: runID, now: clock()) } }
        catch { checkWatchdog(); throw error }
    }

    public func execute(_ packet: ActionPacket) throws {
        do {
            let receipt = try lock.withLock { () throws -> ExecutionReceipt in
                guard interruption == nil else { throw AstraError("control.stopping", "Desktop control is stopping.") }
                guard pending.count < 32, pending[packet.id] == nil, packet.executeAtNanos >= lastEnd else {
                    throw AstraError("control.backpressure", "The command queue is full or packet intervals overlap.")
                }
                var candidate = lease
                try candidate.admit(packet, now: clock()) // Validate every command before changing admission state.
                guard packet.commands.filter({ $0.operation == .pointerRelative }).allSatisfy({
                    $0.dx == $0.dx?.rounded(.toNearestOrEven) && $0.dy == $0.dy?.rounded(.toNearestOrEven)
                }) else { throw AstraError("control.rawCounts", "Relative pointer commands require whole raw motion counts.") }
                let events = Self.expand(packet)
                guard scheduled.count + events.count <= 8192 else { throw AstraError("control.backpressure", "The scheduled motion queue is full.") }
                lease = candidate; lastEnd = packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000
                pending[packet.id] = Pending(packet: packet)
                scheduled.append(contentsOf: events)
                scheduled.sort { ($0.time, $0.packet.sequence, $0.order) < ($1.time, $1.packet.sequence, $1.order) }
                return ExecutionReceipt(packet: packet, status: .admitted, observedNanos: clock(), resultingState: observed)
            }
            onReceipt(receipt)
        } catch {
            onReceipt(ExecutionReceipt(packet: packet, status: .rejected, observedNanos: clock(), resultingState: state()))
            checkWatchdog(); throw error
        }
    }

    public func checkWatchdog() {
        if let generation = lock.withLock({ lease.request == nil ? nil : epoch }) {
            if let recovery = lock.withLock({ recovery }), !recovery.guardianIsFresh(now: clock()) {
                requestDisarm(reason: "The independent cleanup guardian stopped responding.", expectedGeneration: generation); return
            }
            do { try backend.checkHealth() }
            catch { requestDisarm(reason: error.localizedDescription, expectedGeneration: generation); return }
        }
        let status = lock.withLock {
            (lease.request != nil && clock() >= lease.expiresAtNanos,
             lease.request == nil && cleaning == 0 && (!ownedKeys.isEmpty || !ownedButtons.isEmpty), epoch)
        }
        if status.0 { requestDisarm(reason: "The control heartbeat expired.", expectedGeneration: status.2) }
        else if status.1 { requestDisarm(reason: "Retrying owned input cleanup.", expectedGeneration: status.2) }
    }

    /// Event-tap callers invalidate admission immediately and leave potentially
    /// blocking release work to the helper's independent stop queue.
    public func requestDisarm(runID: UUID? = nil, reason: String, cause: ControlStopCause = .fault) {
        requestDisarm(runID: runID, reason: reason, cause: cause, expectedGeneration: nil)
    }
    private func requestDisarm(runID: UUID? = nil, reason: String, cause: ControlStopCause = .fault, expectedGeneration: UInt64?) {
        let generation = lock.withLock { () -> UInt64? in
            if let expectedGeneration, epoch != expectedGeneration { return nil }
            if let runID, lease.request?.runID != runID && armingRunID != runID { return nil }
            guard interruption == nil else { return nil }
            recovery?.stop()
            interruptionRunID = lease.request?.runID ?? armingRunID
            interruption = reason; interruptionCause = cause; epoch &+= 1; return epoch
        }
        if let generation { stopQueue.async { [weak self] in self?.disarm(reason: reason, cause: cause, expectedGeneration: generation) } }
    }

    public func disarm(reason: String = "Desktop control stopped.", cause: ControlStopCause = .requested) {
        disarm(reason: reason, cause: cause, expectedGeneration: nil)
    }
    private func disarm(reason: String, cause: ControlStopCause, expectedGeneration: UInt64?) {
        let captured = lock.withLock { () -> (Set<Int>, Set<Int>, [ExecutionReceipt], Bool, UUID?, ControlStopCause, String)? in
            if let expectedGeneration, epoch != expectedGeneration { return nil }
            epoch &+= 1
            let finalReason = interruption ?? reason, finalCause = interruptionCause ?? cause
            let runID = lease.request?.runID ?? armingRunID ?? interruptionRunID
            let active = lease.request != nil || arming || driving || interruptionRunID != nil
            interruption = nil; interruptionCause = nil; interruptionRunID = nil
            lease.disarm(); scheduled.removeAll(); observed.valid = false
            recovery?.stop()
            let keys = ownedKeys, buttons = ownedButtons
            ownedKeys = []; ownedButtons = []; observed.keys = []; observed.buttons = []
            let receipts = pending.values.filter { $0.packet.id != inFlightPacketID }.sorted { $0.packet.sequence < $1.packet.sequence }.map { value in
                let completed = Set(value.results.map(\.commandIndex))
                let cancelled = value.packet.commands.indices.filter { !completed.contains($0) }.map {
                    CommandResult(commandIndex: $0, scheduledNanos: value.packet.executeAtNanos + UInt64(value.packet.commands[$0].offsetMs) * 1_000_000, status: .cancelled, message: finalReason)
                }
                return ExecutionReceipt(packet: value.packet, status: .cancelled, observedNanos: clock(),
                                        commandResults: (value.results + cancelled).sorted { $0.commandIndex < $1.commandIndex }, resultingState: observed)
            }
            pending = pending.filter { $0.key == inFlightPacketID }; cleaning += 1
            return (keys, buttons, receipts, active, runID, finalCause, finalReason)
        }
        guard let captured else { return }
        cleanup(keys: captured.0, buttons: captured.1)
        lock.withLock { cleaning -= 1; settleRecoveryIfQuiescent(); if !driving, cleaning == 0, ownedKeys.isEmpty, ownedButtons.isEmpty { desktopLock = nil } }
        for var receipt in captured.2 { receipt.resultingState = state(); onReceipt(receipt) }
        if captured.3 { onStop(captured.4, captured.5, captured.6) }
    }

    public func service() {
        guard lock.withLock({ if driving || arming || lease.request == nil || interruption != nil { return false }; driving = true; return true }) else { return }
        defer { lock.withLock { driving = false; settleRecoveryIfQuiescent(); if lease.request == nil, cleaning == 0, ownedKeys.isEmpty, ownedButtons.isEmpty { desktopLock = nil } } }
        while true {
            let item = lock.withLock { () -> (Scheduled, ArmRequest, UInt64)? in
                guard interruption == nil, let request = lease.request, let first = scheduled.first, first.time <= clock() else { return nil }
                scheduled.removeFirst(); return (first, request, epoch)
            }
            guard let (event, request, generation) = item else { return }
            if clock() >= lock.withLock({ lease.expiresAtNanos }) { disarm(reason: "The control heartbeat expired.", cause: .fault); return }
            if clock() - event.time > maximumLateness {
                terminalLate(event.packet); disarm(reason: "An action missed its execution deadline.", cause: .fault); return
            }
            if event.command == nil {
                let receipt = lock.withLock { () -> ExecutionReceipt? in
                    guard epoch == generation, let value = pending.removeValue(forKey: event.packet.id) else { return nil }
                    return ExecutionReceipt(packet: value.packet, status: .executed,
                                            observedNanos: clock(), commandResults: value.results.sorted { $0.commandIndex < $1.commandIndex }, resultingState: observed)
                }
                if let receipt { onReceipt(receipt) }; continue
            }
            let command = event.command!
            do {
                let physical = backend.physicalState()
                guard physical.valid, physical.pointer.isFinite else {
                    throw AstraError("control.physicalState", "Physical input state could not be verified before posting.")
                }
                if request.scope.stopOnPhysicalInput && (!physical.keys.isEmpty || !physical.buttons.isEmpty) {
                    disarm(reason: "Physical input took over desktop control.", cause: .physicalTakeover); return
                }
                // Applications can warp the system cursor without a physical
                // event (for example when using relative mouse input). Resolve
                // the next delta or click from the current native pointer,
                // rather than accumulating against our last synthetic post.
                var location = physical.pointer
                if command.operation == .pointerAbsolute, let surface = request.scope.surfaces.first(where: { $0.id == command.surfaceID }) {
                    location = try surface.globalBounds.globalPoint(normalized: Point2D(x: command.x!, y: command.y!))
                } else if command.operation == .pointerRelative { location.x += command.dx!; location.y += command.dy! }
                try backend.validate(request.scope, pointer: command.operation.isMotion || command.operation == .buttonDown || command.operation == .scroll ? location : nil)
                if clock() - event.time > maximumLateness {
                    terminalLate(event.packet); disarm(reason: "Target verification missed the action deadline.", cause: .fault); return
                }
                let reserved = lock.withLock { () -> (InputEmission?, Set<Int>, Set<Int>)? in
                    guard epoch == generation, lease.request != nil else { return nil }
                    var noOp = false
                    switch command.operation {
                    case .keyDown: noOp = ownedKeys.contains(command.keyCode!) || physical.keys.contains(command.keyCode!); if !noOp { ownedKeys.insert(command.keyCode!) }
                    case .keyUp: noOp = !ownedKeys.contains(command.keyCode!) || physical.keys.contains(command.keyCode!)
                    case .keyRepeat: noOp = !ownedKeys.contains(command.keyCode!) || physical.keys.contains(command.keyCode!)
                    case .buttonDown: noOp = ownedButtons.contains(command.button!) || physical.buttons.contains(command.button!); if !noOp { ownedButtons.insert(command.button!) }
                    case .buttonUp: noOp = !ownedButtons.contains(command.button!) || physical.buttons.contains(command.button!)
                    default: break
                    }
                    if noOp { return (nil, ownedKeys, ownedButtons) }
                    var afterKeys = ownedKeys, afterButtons = ownedButtons
                    if command.operation == .keyUp { afterKeys.remove(command.keyCode!) }
                    if command.operation == .buttonUp { afterButtons.remove(command.button!) }
                    inFlightPacketID = event.packet.id
                    return (InputEmission(operation: command.operation, keyCode: command.keyCode, button: command.button, location: location,
                                          delta: Point2D(x: command.dx ?? 0, y: command.dy ?? 0), heldKeys: afterKeys, heldButtons: afterButtons), ownedKeys, ownedButtons)
                }
                guard let (emission, possibleKeys, possibleButtons) = reserved else { return }
                var postError: (any Error)?
                if let emission {
                    let journal = lock.withLock { recovery }
                    let protected = journal.map { $0.guardianIsFresh(now: clock()) && $0.beginPost(operation: emission.operation, keyCode: emission.keyCode, button: emission.button) } ?? true
                    if protected {
                        do { try backend.post(emission) } catch { postError = error }
                        journal?.endPost(operation: emission.operation, keyCode: emission.keyCode, button: emission.button, success: postError == nil)
                    } else { postError = AstraError("control.guardianStopped", "Posting was cancelled because independent cleanup protection stopped.") }
                }
                let postedAt = clock()
                var historyOverflow = false
                let stale = lock.withLock { () -> Bool in
                    guard epoch == generation else { return true }
                    inFlightPacketID = nil
                    if postError == nil, emission != nil {
                        if command.operation == .keyUp { ownedKeys.remove(command.keyCode!) }
                        if command.operation == .buttonUp { ownedButtons.remove(command.button!) }
                    }
                    observed.keys = ownedKeys; observed.buttons = ownedButtons
                    observed.modifiers = CGEventControlBackend.modifierFlags(keys: ownedKeys).rawValue | physical.modifiers
                    if postError != nil { observed.valid = false }
                    if emission != nil, postError == nil { observed.pointer = location }
                    // Availability is captured under the same lock as observation
                    // snapshots. A racing snapshot cannot omit an event whose
                    // reported availability is already before its cutoff.
                    observed.observedNanos = clock(); observed.revision &+= 1
                    if let emission, postError == nil {
                        if executedEvents.count >= historyCapacity || nextEventSequence == UInt64.max {
                            historyCovered = false; historyOverflow = true; observed.valid = false
                        } else {
                            executedEvents.append(Self.rawEvent(emission, sequence: nextEventSequence, sourceNanos: postedAt,
                                                               observedNanos: observed.observedNanos, modifiers: observed.modifiers))
                            nextEventSequence += 1
                        }
                    }
                    if let index = event.index {
                        pending[event.packet.id]?.results.append(CommandResult(commandIndex: index, scheduledNanos: event.time,
                            postedNanos: emission == nil || postError != nil ? nil : postedAt, status: postError != nil ? .failed : emission == nil ? .noOp : .posted,
                            message: postError?.localizedDescription))
                    }
                    return false
                }
                if stale {
                    cleanup(keys: possibleKeys, buttons: possibleButtons)
                    let receipt = lock.withLock { () -> ExecutionReceipt? in
                        inFlightPacketID = nil
                        guard var value = pending.removeValue(forKey: event.packet.id) else { return nil }
                        if let index = event.index {
                            value.results.append(CommandResult(commandIndex: index, scheduledNanos: event.time,
                                postedNanos: emission == nil || postError != nil ? nil : postedAt,
                                status: postError != nil ? .failed : emission == nil ? .noOp : .posted,
                                message: "Posting returned after stop; owned input was released again."))
                        }
                        let done = Set(value.results.map(\.commandIndex))
                        for index in event.packet.commands.indices where !done.contains(index) {
                            value.results.append(CommandResult(commandIndex: index, scheduledNanos: event.packet.executeAtNanos + UInt64(event.packet.commands[index].offsetMs) * 1_000_000, status: .cancelled))
                        }
                        return ExecutionReceipt(packet: event.packet, status: .cancelled, observedNanos: clock(), commandResults: value.results.sorted { $0.commandIndex < $1.commandIndex }, resultingState: observed)
                    }
                    if let receipt { onReceipt(receipt) }; return
                }
                if historyOverflow { disarm(reason: "Executed input history exceeded its bounded acknowledgement window.", cause: .fault); return }
                if let postError { disarm(reason: "Input posting failed: \(postError.localizedDescription)", cause: .fault); return }
                if postedAt - event.time > maximumLateness {
                    terminalLate(event.packet); disarm(reason: "Input posting exceeded the action deadline.", cause: .fault); return
                }
            } catch { disarm(reason: error.localizedDescription, cause: .fault); return }
        }
    }

    private func terminalLate(_ packet: ActionPacket) {
        let receipt = lock.withLock { () -> ExecutionReceipt? in
            guard let value = pending.removeValue(forKey: packet.id) else { return nil }
            let done = Set(value.results.map(\.commandIndex))
            let cancelled = packet.commands.indices.filter { !done.contains($0) }.map {
                CommandResult(commandIndex: $0, scheduledNanos: packet.executeAtNanos + UInt64(packet.commands[$0].offsetMs) * 1_000_000, status: .cancelled, message: "Execution deadline missed.")
            }
            return ExecutionReceipt(packet: packet, status: .late, observedNanos: clock(), commandResults: value.results + cancelled, resultingState: observed)
        }
        if let receipt { onReceipt(receipt) }
    }

    /// Called under the executor lock only after every local posting and
    /// provisional cleanup path has joined. The guardian does not clear this
    /// conservative journal while the executor can still return from a post.
    private func settleRecoveryIfQuiescent() {
        guard lease.request == nil, !arming, !driving, inFlightPacketID == nil, cleaning == 0,
              ownedKeys.isEmpty, ownedButtons.isEmpty, interruption == nil else { return }
        recovery?.stop(); recovery?.settleLocally(now: clock())
    }

    private func cleanup(keys: Set<Int>, buttons: Set<Int>) {
        guard !keys.isEmpty || !buttons.isEmpty else { return }
        let physical = backend.cleanupPhysicalState()
        guard physical.valid, physical.pointer.isFinite else {
            lock.withLock {
                ownedKeys.formUnion(keys); ownedButtons.formUnion(buttons)
                observed.keys.formUnion(keys); observed.buttons.formUnion(buttons); observed.valid = false
            }
            return
        }
        let location = physical.valid && physical.pointer.isFinite ? physical.pointer : state().pointer
        var remaining = keys
        for key in keys.sorted() {
            remaining.remove(key)
            guard !physical.keys.contains(key) else { continue } // Ownership transfers to the physical hold.
            do { try backend.post(InputEmission(operation: .keyUp, keyCode: key, location: location, delta: .zero, heldKeys: remaining, heldButtons: [], cleanup: true)) }
            catch { lock.withLock { ownedKeys.insert(key); observed.keys.insert(key) } }
        }
        for button in buttons.sorted() where !physical.buttons.contains(button) {
            do { try backend.post(InputEmission(operation: .buttonUp, button: button, location: location, delta: .zero, heldKeys: [], heldButtons: [], cleanup: true)) }
            catch { lock.withLock { ownedButtons.insert(button); observed.buttons.insert(button) } }
        }
    }

    private static func rawEvent(_ emission: InputEmission, sequence: UInt64, sourceNanos: UInt64,
                                 observedNanos: UInt64, modifiers: UInt64) -> RawInputEvent {
        let kind: RawInputKind
        switch emission.operation {
        case .keyDown: kind = .keyDown
        case .keyUp: kind = .keyUp
        case .keyRepeat: kind = .keyRepeat
        case .buttonDown: kind = .buttonDown
        case .buttonUp: kind = .buttonUp
        case .pointerAbsolute, .pointerRelative: kind = .pointer
        case .scroll: kind = .scroll
        }
        return RawInputEvent(sequence: sequence, eventNanos: sourceNanos, observedNanos: observedNanos, origin: .agent,
            kind: kind, keyCode: emission.keyCode, button: emission.button, x: emission.location.x, y: emission.location.y,
            dx: emission.operation.isMotion ? emission.delta.x : nil, dy: emission.operation.isMotion ? emission.delta.y : nil,
            scrollX: kind == .scroll ? emission.delta.x : nil, scrollY: kind == .scroll ? emission.delta.y : nil,
            modifiers: modifiers, detail: kind == .scroll ? "scroll units: points" : nil)
    }

    private static func expand(_ packet: ActionPacket) -> [Scheduled] {
        var events: [Scheduled] = [], previous: TimedCommand?
        for (index, original) in packet.commands.enumerated() {
            var command = original
            if command.operation.isMotion {
                if let start = previous, command.offsetMs > start.offsetMs {
                    let length = command.offsetMs - start.offsetMs
                    var last = Point2D.zero
                    for tick in 1..<length {
                        var sample = command; sample.offsetMs = start.offsetMs + tick
                        let fraction = Double(tick) / Double(length)
                        if command.operation == .pointerAbsolute {
                            // Surface changes are knots, not interpolations through unrelated coordinate systems.
                            guard start.surfaceID == command.surfaceID else { continue }
                            sample.x = start.x! + fraction * (command.x! - start.x!); sample.y = start.y! + fraction * (command.y! - start.y!)
                        } else {
                            let cumulative = Point2D(x: (command.dx! * fraction).rounded(.toNearestOrEven), y: (command.dy! * fraction).rounded(.toNearestOrEven))
                            sample.dx = cumulative.x - last.x; sample.dy = cumulative.y - last.y; last = cumulative
                        }
                        events.append(Scheduled(time: packet.executeAtNanos + UInt64(sample.offsetMs) * 1_000_000, packet: packet, command: sample, index: nil, order: -1))
                    }
                    if command.operation == .pointerRelative { command.dx! -= last.x; command.dy! -= last.y }
                }
                previous = original
            }
            events.append(Scheduled(time: packet.executeAtNanos + UInt64(command.offsetMs) * 1_000_000, packet: packet, command: command, index: index, order: index))
        }
        events.append(Scheduled(time: packet.executeAtNanos + UInt64(packet.durationMs) * 1_000_000, packet: packet, command: nil, index: nil, order: Int.max))
        return events
    }
}

/// Core Graphics event construction is isolated from scheduling and never asks
/// macOS to grant privacy access. The private event source keeps synthetic state
/// separate from the HID state used to recognize a person's held controls.
public final class CGEventControlBackend: ControlInputBackend, @unchecked Sendable {
    private let source: CGEventSource
    private let keyboardTrust: KeyboardObservationTrust
    private let permissionProbe: @Sendable () -> (accessibility: Bool, eventPosting: Bool, inputMonitoring: Bool)
    private let hidProbe: (@Sendable () -> ControlState)?
    private let lock = NSLock()
    private var launchDate: Date?
    private var activeScope: ControlScope?
    private var scopeGeneration = UUID()
    private var scopeCheckedAt: UInt64 = 0
    private var scopeFailure: String?
    private var scopeTimer: DispatchSourceTimer?
    private var permissionCheckedAt: UInt64?
    private var permissionAvailable = false
    private var permissionTimer: DispatchSourceTimer?
    private var keyboardGeneration: UInt64?
    public static let permissionFreshnessNanos: UInt64 = 250_000_000
    public static let scopeFreshnessNanos: UInt64 = 100_000_000
    public init(keyboardTrust: KeyboardObservationTrust = .shared,
                permissionProbe: @escaping @Sendable () -> (accessibility: Bool, eventPosting: Bool, inputMonitoring: Bool) = {
                    (AXIsProcessTrusted(), CGPreflightPostEventAccess(), CGPreflightListenEventAccess())
                },
                hidProbe: (@Sendable () -> ControlState)? = nil) throws {
        guard let source = CGEventSource(stateID: .privateState) else { throw AstraError("control.source", "A private input source could not be created.") }
        self.source = source
        self.keyboardTrust = keyboardTrust; self.permissionProbe = permissionProbe; self.hidProbe = hidProbe
    }
    deinit { permissionTimer?.cancel(); scopeTimer?.cancel() }
    public func prepare(_ request: ArmRequest) throws -> ControlState {
        let keyboard = keyboardTrust.refreshSynchronously(); try keyboard.requireTrusted()
        lock.withLock { keyboardGeneration = keyboard.interruptionGeneration }
        refreshPermissionProof(); try requirePermissionProof()
        lock.withLock {
            if permissionTimer == nil {
                let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "astra.control.permissions", qos: .userInteractive))
                timer.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100), leeway: .milliseconds(5))
                timer.setEventHandler { [weak self] in self?.refreshPermissionProof() }
                permissionTimer = timer; timer.resume()
            }
        }
        guard request.capabilities.keyCodes.isDisjoint(with: [57, 63, 72, 73, 74, 127]) else {
            throw AstraError("control.unsupportedKey", "Caps Lock, Fn, media and power-key semantics are not yet qualified for synthetic control.")
        }
        let launched = request.scope.applicationPID.flatMap { NSRunningApplication(processIdentifier: $0)?.launchDate }
        guard request.scope.applicationPID == nil || launched != nil else {
            throw AstraError("control.applicationIdentity", "The selected application's launch identity could not be verified.")
        }
        lock.withLock { launchDate = launched }
        let scopeBegan = MonotonicClock.now
        try validate(request.scope, pointer: nil)
        lock.withLock {
            activeScope = request.scope; scopeGeneration = UUID(); scopeCheckedAt = scopeBegan; scopeFailure = nil
            if scopeTimer == nil {
                let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "astra.control.scope", qos: .userInteractive))
                timer.schedule(deadline: .now() + .milliseconds(20), repeating: .milliseconds(20), leeway: .milliseconds(1))
                timer.setEventHandler { [weak self] in self?.refreshScopeProof() }
                scopeTimer = timer; timer.resume()
            }
        }
        return physicalState()
    }
    public func physicalState() -> ControlState {
        let before = keyboardTrust.snapshot()
        guard before.trusted, lock.withLock({ keyboardGeneration.map { $0 == before.interruptionGeneration } ?? true }) else { return ControlState() }
        let result = readHIDState()
        let after = keyboardTrust.snapshot()
        guard after.trusted, after.interruptionGeneration == before.interruptionGeneration else { return ControlState() }
        return result
    }
    private func readHIDState() -> ControlState {
        if let hidProbe { return hidProbe() }
        var result = ControlState()
        result.keys = Set((0...127).filter { CGEventSource.keyState(.hidSystemState, key: CGKeyCode($0)) })
        result.buttons = Set((0...31).filter { CGEventSource.buttonState(.hidSystemState, button: CGMouseButton(rawValue: UInt32($0))!) })
        result.modifiers = CGEventSource.flagsState(.hidSystemState).rawValue
        if let point = CGEvent(source: nil)?.location { result.pointer = Point2D(point); result.valid = true }
        result.observedNanos = MonotonicClock.now
        return result
    }
    public func cleanupPhysicalState() -> ControlState {
        // Cleanup may run after ordinary health proof has expired, including in
        // the guardian that never entered normal policy execution. Verify HID
        // access directly before attributing a hold to a person.
        let before = keyboardTrust.refreshSynchronously()
        guard before.trusted, permissionProbe().inputMonitoring else { return ControlState() }
        let state = readHIDState()
        let after = keyboardTrust.refreshSynchronously()
        guard after.trusted, after.interruptionGeneration == before.interruptionGeneration else { return ControlState() }
        return state
    }
    private func refreshPermissionProof() {
        let began = MonotonicClock.now
        let permissions = permissionProbe()
        let available = permissions.accessibility && permissions.eventPosting && permissions.inputMonitoring
        lock.withLock {
            if began >= (permissionCheckedAt ?? 0) { permissionCheckedAt = began; permissionAvailable = available }
        }
    }
    private func requirePermissionProof() throws {
        let valid = lock.withLock {
            permissionAvailable && permissionCheckedAt.map { MonotonicClock.now - $0 < Self.permissionFreshnessNanos } == true
        }
        guard valid else {
            throw AstraError("permission.control", "Control permission is denied or its independent verification expired. Accessibility, input posting, and Input Monitoring must be enabled before arming.")
        }
    }
    public func checkHealth() throws {
        let proof = keyboardTrust.snapshot(); try proof.requireTrusted()
        guard lock.withLock({ keyboardGeneration == proof.interruptionGeneration }) else {
            throw AstraError("input.keyboardInterrupted", "Keyboard observation was interrupted after this control run armed.")
        }
        try requirePermissionProof()
        let failure = lock.withLock { () -> String? in
            if let scopeFailure { return scopeFailure }
            if activeScope == nil || MonotonicClock.now - scopeCheckedAt >= Self.scopeFreshnessNanos {
                return "Independent target verification expired."
            }
            return nil
        }
        if let failure { throw AstraError("control.scopeHealth", failure) }
    }
    private func refreshScopeProof() {
        guard let (scope, generation) = lock.withLock({ activeScope.map { ($0, scopeGeneration) } }) else { return }
        let began = MonotonicClock.now
        var failure: String?
        do { try validate(scope, pointer: nil) } catch { failure = error.localizedDescription }
        lock.withLock {
            guard scopeGeneration == generation, began >= scopeCheckedAt else { return }
            scopeCheckedAt = began
            if scopeFailure == nil { scopeFailure = failure }
        }
    }
    public func validate(_ scope: ControlScope, pointer: Point2D?) throws {
        try requirePermissionProof()
        if let pointer, !scope.surfaces.contains(where: { InputScopeSnapshot.contains($0.globalBounds, pointer) }) {
            throw AstraError("control.pointerScope", "The planned pointer leaves the observed control surfaces.")
        }
        if scope.applicationPID == nil {
            for surface in scope.surfaces {
                guard surface.id.hasPrefix("display:"), let id = UInt32(surface.id.dropFirst(8)),
                      CGDisplayIsActive(id) != 0, Rect2D(CGDisplayBounds(id)) == surface.globalBounds else {
                    throw AstraError("control.display", "The observed display changed or its identity is missing.")
                }
            }
            // A display-scoped pointer is authorized by the observed display,
            // independently of which application's window receives it. Whole
            // desktop keyboard scope likewise needs no window-list lookup.
            if scope.wholeDesktop || pointer != nil { return }
        }
        guard let raw = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] else {
            throw AstraError("control.target", "The current desktop target could not be verified.")
        }
        let windows = raw.compactMap { entry -> (UInt32, Int32, Int, Rect2D)? in
            guard let id = entry[kCGWindowNumber as String] as? NSNumber, let pid = entry[kCGWindowOwnerPID as String] as? NSNumber,
                  let layer = entry[kCGWindowLayer as String] as? NSNumber, let alpha = entry[kCGWindowAlpha as String] as? NSNumber, alpha.doubleValue > 0,
                  let bounds = entry[kCGWindowBounds as String] as? NSDictionary, let rect = CGRect(dictionaryRepresentation: bounds) else { return nil }
            return (id.uint32Value, pid.int32Value, layer.intValue, Rect2D(rect))
        }
        if let pid = scope.applicationPID {
            guard NSWorkspace.shared.frontmostApplication?.processIdentifier == pid,
                  let application = NSRunningApplication(processIdentifier: pid), !application.isTerminated,
                  application.launchDate == lock.withLock({ launchDate }) else {
                throw AstraError("control.focus", "The selected application lost focus or restarted.")
            }
            for surface in scope.surfaces where surface.id.hasPrefix("window:") {
                guard let id = UInt32(surface.id.dropFirst(7)), let window = windows.first(where: { $0.0 == id }),
                      window.1 == pid, window.3 == surface.globalBounds else {
                    throw AstraError("control.geometry", "An observed application window moved, resized, or disappeared.")
                }
            }
            if let windowID = scope.windowID {
                guard let target = windows.first(where: { $0.1 == pid && $0.2 >= 0 && $0.2 < Int(CGWindowLevelForKey(.dockWindow)) }),
                      target.0 == windowID, target.2 == 0,
                      scope.surfaces.contains(where: { $0.globalBounds == target.3 }) else {
                    throw AstraError("control.geometry", "The selected window moved, resized, or lost its foreground position.")
                }
            }
            if let pointer {
                guard let hit = windows.first(where: { $0.2 >= 0 && $0.2 < Int(CGWindowLevelForKey(.cursorWindow)) && InputScopeSnapshot.contains($0.3, pointer) }),
                      hit.1 == pid, scope.windowID == nil || hit.0 == scope.windowID else {
                    throw AstraError("control.pointerTarget", "The planned pointer reaches another window or an unobserved overlay.")
                }
            }
        } else {
            if !scope.wholeDesktop {
                guard let pid = NSWorkspace.shared.frontmostApplication?.processIdentifier,
                      let window = windows.first(where: { $0.1 == pid && $0.2 == 0 }),
                      scope.surfaces.contains(where: { $0.globalBounds.cgRect.intersects(window.3.cgRect) }) else {
                    throw AstraError("control.keyboardScope", "Keyboard focus moved outside the selected display.")
                }
            }
        }
    }
    public func post(_ emission: InputEmission) throws {
        var cleanupKeyboard: KeyboardObservationProof?
        if emission.cleanup {
            let proof = keyboardTrust.refreshSynchronously(); try proof.requireTrusted(); cleanupKeyboard = proof
            let permissions = permissionProbe()
            guard emission.operation == .keyUp || emission.operation == .buttonUp,
                  permissions.eventPosting, permissions.inputMonitoring else {
                throw AstraError("permission.cleanup", "macOS is not permitting owned input release or physical-hold verification. Cleanup remains pending.")
            }
        } else { try checkHealth() }
        let event: CGEvent?
        let flags = Self.modifierFlags(keys: emission.heldKeys).union(CGEventSource.flagsState(.hidSystemState))
        switch emission.operation {
        case .keyDown, .keyUp, .keyRepeat:
            guard let key = emission.keyCode else { throw AstraError("control.key", "A key emission has no key code.") }
            event = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(key), keyDown: emission.operation != .keyUp)
            if [54, 55, 56, 58, 59, 60, 61, 62].contains(key) { event?.type = .flagsChanged }
            event?.setIntegerValueField(.keyboardEventAutorepeat, value: emission.operation == .keyRepeat ? 1 : 0)
        case .buttonDown, .buttonUp:
            guard let number = emission.button, let button = CGMouseButton(rawValue: UInt32(number)) else { throw AstraError("control.button", "A button emission is invalid.") }
            let down = emission.operation == .buttonDown
            let type: CGEventType = number == 0 ? (down ? .leftMouseDown : .leftMouseUp) : number == 1 ? (down ? .rightMouseDown : .rightMouseUp) : (down ? .otherMouseDown : .otherMouseUp)
            event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: emission.location.cgPoint, mouseButton: button)
            event?.setIntegerValueField(.mouseEventClickState, value: 1)
        case .pointerAbsolute, .pointerRelative:
            let held = emission.heldButtons.sorted().first
            let type: CGEventType = held == nil ? .mouseMoved : held == 0 ? .leftMouseDragged : held == 1 ? .rightMouseDragged : .otherMouseDragged
            event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: emission.location.cgPoint, mouseButton: CGMouseButton(rawValue: UInt32(held ?? 0))!)
            event?.setIntegerValueField(.mouseEventDeltaX, value: Int64(emission.delta.x.rounded(.toNearestOrEven)))
            event?.setIntegerValueField(.mouseEventDeltaY, value: Int64(emission.delta.y.rounded(.toNearestOrEven)))
        case .scroll:
            event = CGEvent(scrollWheelEvent2Source: source, units: .pixel, wheelCount: 2,
                            wheel1: Int32(emission.delta.y.rounded(.toNearestOrEven)), wheel2: Int32(emission.delta.x.rounded(.toNearestOrEven)), wheel3: 0)
            event?.setDoubleValueField(.scrollWheelEventFixedPtDeltaAxis1, value: emission.delta.y)
            event?.setDoubleValueField(.scrollWheelEventFixedPtDeltaAxis2, value: emission.delta.x)
            event?.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
        }
        guard let event else { throw AstraError("control.event", "macOS could not allocate an input event.") }
        event.flags = flags; event.setIntegerValueField(.eventSourceUserData, value: AstraSyntheticInput.tag)
        // A long release batch must not rely only on its first HID snapshot.
        // Recheck immediately before submission so a newly physical hold takes
        // ownership instead of receiving an artificial up event.
        if emission.cleanup {
            let proof = keyboardTrust.refreshSynchronously(); try proof.requireTrusted()
            guard proof.interruptionGeneration == cleanupKeyboard?.interruptionGeneration else {
                throw AstraError("input.keyboardInterrupted", "Keyboard trust changed during owned input cleanup.")
            }
            let keyHeld = emission.keyCode.map { CGEventSource.keyState(.hidSystemState, key: CGKeyCode($0)) } ?? false
            let buttonHeld = emission.button.flatMap { CGMouseButton(rawValue: UInt32($0)) }.map {
                CGEventSource.buttonState(.hidSystemState, button: $0)
            } ?? false
            let after = keyboardTrust.refreshSynchronously(); try after.requireTrusted()
            guard after.interruptionGeneration == proof.interruptionGeneration else {
                throw AstraError("input.keyboardInterrupted", "Keyboard trust changed during physical-hold verification.")
            }
            if keyHeld || buttonHeld { return }
        }
        event.post(tap: .cgSessionEventTap)
    }
    public static func modifierFlags(keys: Set<Int>) -> CGEventFlags {
        var bits: UInt64 = 0
        let mapping: [(Int, Int32, CGEventFlags)] = [(54, NX_DEVICERCMDKEYMASK, .maskCommand), (55, NX_DEVICELCMDKEYMASK, .maskCommand),
            (56, NX_DEVICELSHIFTKEYMASK, .maskShift), (60, NX_DEVICERSHIFTKEYMASK, .maskShift), (58, NX_DEVICELALTKEYMASK, .maskAlternate),
            (61, NX_DEVICERALTKEYMASK, .maskAlternate), (59, NX_DEVICELCTLKEYMASK, .maskControl), (62, NX_DEVICERCTLKEYMASK, .maskControl)]
        for (key, side, flag) in mapping where keys.contains(key) { bits |= UInt64(side) | flag.rawValue }
        return CGEventFlags(rawValue: bits)
    }
}
