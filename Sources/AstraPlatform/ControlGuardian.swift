import Foundation
import Darwin
import ApplicationServices
import AstraCore
import CAstraRecovery

private final class GuardianExit: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Int32?
    let completion = AsyncCompletion()
    var status: Int32? { lock.withLock { value } }
    func reap(_ pid: Int32) -> Bool {
        let completed = lock.withLock { () -> Bool in
            if value != nil { return true }
            var status: Int32 = 0
            let result = waitpid(pid, &status, WNOHANG)
            if result == 0 || (result < 0 && errno == EINTR) { return false }
            value = result == pid ? status : -1; return true
        }
        if completed { completion.finish() }
        return completed
    }
    func terminateSettledChild(_ pid: Int32) {
        // Reaping and signalling share this lock: an exited child's PID cannot
        // be reused between the ownership check and this exceptional kill.
        lock.withLock { if value == nil { _ = kill(pid, SIGKILL) } }
    }
}

/// The guardian is a re-exec of the same signed helper identity. It inherits
/// only an existing desktop-lease reference and recovery mapping descriptor.
public final class ControlGuardianPair: @unchecked Sendable {
    public let ledger: ControlRecoveryLedger
    public let desktopLease: DesktopControlLock
    public let guardianPID: Int32
    private let exit = GuardianExit()
    public init(ledger: ControlRecoveryLedger, desktopLease: DesktopControlLock, executable: URL,
                onExit: @escaping @Sendable () -> Void) throws {
        self.ledger = ledger; self.desktopLease = desktopLease
        guard ledger.registerExecutor(pid: getpid()) else { throw AstraError("control.guardianIdentity", "Recovery storage already belongs to an executor.") }
        var pid: pid_t = 0
        let result = desktopLease.withFileDescriptor { lease in ledger.withFileDescriptor { mapping in
            executable.path.withCString { astra_guardian_spawn($0, lease, mapping, getpid(), &pid) }
        } }
        guard result == 0 else { ledger.stop(); ledger.settleLocally(now: MonotonicClock.now); throw AstraError("control.guardianLaunch", "The independent cleanup guardian could not start (\(result)).") }
        guardianPID = pid
        let child = pid
        let exit = exit
        let thread = Thread {
            while !exit.reap(child) { Thread.sleep(forTimeInterval: 0.01) }
            onExit()
        }
        thread.name = "Astra guardian exit"; thread.qualityOfService = .userInteractive; thread.start()
    }

    public func waitUntilReady() async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while ContinuousClock.now < deadline {
            try Task.checkCancellation()
            guard exit.status == nil else { throw AstraError("control.guardianUnavailable", "The cleanup guardian exited before protection was ready. Check the existing control permissions.") }
            let snapshot = try ledger.snapshot()
            if snapshot.guardianReady, snapshot.guardianPID == guardianPID, snapshot.executorPID == getpid(),
               snapshot.phase == .preparing, MonotonicClock.now >= snapshot.guardianHeartbeatNanos,
               MonotonicClock.now - snapshot.guardianHeartbeatNanos < 150_000_000 { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        throw AstraError("control.guardianTimeout", "The independent cleanup guardian did not become ready in time.")
    }

    public func finishAfterLocalSettlement() async {
        // The caller first proves local quiescence. A guardian stuck while
        // synchronizing an already-settled record may then be terminated;
        // this is never permitted while it could still own an input release.
        guard (try? ledger.snapshot().cleanupConfirmed) == true else { return }
        let deadline = ContinuousClock.now.advanced(by: .milliseconds(500))
        while exit.status == nil, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(5)) }
        exit.terminateSettledChild(guardianPID)
        await exit.completion.wait()
    }
}

public enum ControlGuardian {
    /// A virtual backend is supplied only by the non-shipping fault fixture.
    /// Production always verifies already-granted permissions and uses CGEvent.
    public static func run(ownerPID: Int32, backend: any ControlInputBackend,
                           permissionsReady: () -> Bool, onFailure: (any Error) -> Void = { _ in },
                           registerExitWatch: ((Int32, Int32) -> Int32)? = nil) -> Int32 {
        signal(SIGPIPE, SIG_IGN); signal(SIGTERM, SIG_IGN); signal(SIGINT, SIG_IGN); signal(SIGHUP, SIG_IGN)
        do {
            let lease = try DesktopControlLock(inheritedDescriptor: 3)
            let ledger = try ControlRecoveryLedger(inheritedDescriptor: 4)
            close(3); close(4)
            return try withExtendedLifetime(lease) {
                let initial = try ledger.snapshot()
                guard ownerPID > 0, initial.executorPID == ownerPID, initial.phase == .preparing,
                      !initial.everArmed, !initial.guardianReady, initial.guardianPID == 0,
                      initial.possibleKeys.isEmpty, initial.possibleButtons.isEmpty, initial.inFlight == 0 else {
                    throw AstraError("control.guardianParent", "The guardian's initial executor identity or ledger is invalid.")
                }
                var publishedReady = false
                defer {
                    // Before our ready publication no executor can arm. Cancel
                    // that admission atomically and settle the empty journal
                    // even if a failed exit-watch registration races reparenting.
                    // Once ready, only local settlement or NOTE_EXIT can prove
                    // that a potentially armed controller has stopped posting.
                    if !publishedReady { ledger.stop(); ledger.settleLocally(now: MonotonicClock.now) }
                }
                // If reparented before readiness, no posting could have started:
                // arming requires our subsequent ready flag. Never recover an
                // already-armed ledger by guessing from an unverified PID.
                guard getppid() == ownerPID else { ledger.stop(); ledger.settleLocally(now: MonotonicClock.now); return 0 }
                let queue = kqueue()
                guard queue >= 0 else { throw AstraError("control.guardianWatch", "Executor exit monitoring could not start.") }
                defer { close(queue) }
                let watchError: Int32
                if let registerExitWatch { watchError = registerExitWatch(queue, ownerPID) }
                else {
                    var registration = kevent(ident: UInt(ownerPID), filter: Int16(EVFILT_PROC), flags: UInt16(EV_ADD | EV_ENABLE | EV_CLEAR),
                                              fflags: UInt32(NOTE_EXIT), data: 0, udata: nil)
                    watchError = kevent(queue, &registration, 1, nil, 0, nil) == 0 ? 0 : errno
                }
                guard watchError == 0 else {
                    if watchError == ESRCH, getppid() != ownerPID { return 0 }
                    throw AstraError("control.guardianWatch", "The executor exit watch could not be registered (\(watchError)).")
                }
                guard permissionsReady() else { ledger.recordError(1); return 11 }
                // Parent death during permission verification remains queued in
                // kqueue. Ready is published only if still its actual child.
                if getppid() == ownerPID {
                    guard ledger.registerGuardian(pid: getpid(), now: MonotonicClock.now) else {
                        if (try ledger.snapshot()).cleanupConfirmed { return 0 }
                        throw AstraError("control.guardianReadiness", "Recovery readiness was cancelled.")
                    }
                    publishedReady = true
                } else {
                    // No ready flag was published, so the old parent could not
                    // post. Do not wait on a PID potentially reused between the
                    // earlier parent check and kqueue registration.
                    let snapshot = try ledger.snapshot()
                    guard !snapshot.everArmed, snapshot.inFlight == 0, snapshot.possibleKeys.isEmpty, snapshot.possibleButtons.isEmpty else {
                        throw AstraError("control.guardianParent", "An unprotected ledger unexpectedly contains input ownership.")
                    }
                    ledger.stop(); ledger.settleLocally(now: MonotonicClock.now); return 0
                }
                var ownerExited = false
                while true {
                    let snapshot = try ledger.snapshot()
                    if snapshot.cleanupConfirmed { try? ledger.synchronizeTerminal(); return 0 }
                    ledger.heartbeat(now: MonotonicClock.now)
                    var event = kevent(), timeout = timespec(tv_sec: 0, tv_nsec: 10_000_000)
                    let count = kevent(queue, nil, 0, &event, 1, &timeout)
                    if count < 0, errno != EINTR { ledger.recordError(2); ledger.stop() }
                    if count > 0, event.ident == UInt(ownerPID), event.filter == Int16(EVFILT_PROC), event.fflags & UInt32(NOTE_EXIT) != 0 {
                        ownerExited = true
                    }
                    guard ownerExited else { continue }
                    guard ledger.claimAfterExecutorExit() else { continue }
                    recover(ledger, backend: backend)
                }
            }
        } catch { onFailure(error); return 12 }
    }

    private static func recover(_ ledger: ControlRecoveryLedger, backend: any ControlInputBackend) {
        do {
            let snapshot = try ledger.snapshot()
            if snapshot.possibleKeys.isEmpty, snapshot.possibleButtons.isEmpty {
                _ = ledger.settleGuardian(now: MonotonicClock.now); return
            }
            let physical = backend.cleanupPhysicalState()
            guard physical.valid, physical.pointer.isFinite else { ledger.recordError(3); return }
            var remaining = snapshot.possibleKeys
            for key in snapshot.possibleKeys.sorted() {
                remaining.remove(key)
                if physical.keys.contains(key) { ledger.releaseKey(key); continue }
                do {
                    try backend.post(InputEmission(operation: .keyUp, keyCode: key, location: physical.pointer,
                        delta: .zero, heldKeys: remaining, heldButtons: [], cleanup: true))
                    ledger.releaseKey(key)
                } catch { remaining.insert(key); ledger.recordError(4) }
            }
            for button in snapshot.possibleButtons.sorted() {
                if physical.buttons.contains(button) { ledger.releaseButton(button); continue }
                do {
                    try backend.post(InputEmission(operation: .buttonUp, button: button, location: physical.pointer,
                        delta: .zero, heldKeys: remaining, heldButtons: [], cleanup: true))
                    ledger.releaseButton(button)
                } catch { ledger.recordError(4) }
            }
            _ = ledger.settleGuardian(now: MonotonicClock.now)
        } catch { ledger.recordError(5) }
    }

    public static func runNative(ownerPID: Int32) -> Int32 {
        do {
            let backend = try CGEventControlBackend()
            return run(ownerPID: ownerPID, backend: backend, permissionsReady: {
                let permissions = PermissionSnapshot.current()
                return permissions.accessibility && permissions.inputMonitoring && permissions.eventPosting
                    && KeyboardObservationTrust.shared.refreshSynchronously().trusted
            })
        } catch { return 10 }
    }
}
