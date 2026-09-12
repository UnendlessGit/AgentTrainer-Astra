import Foundation
import Darwin
import CAstraRecovery

public struct ControlRecoveryDescriptor: Codable, Hashable, Sendable {
    public var version: Int = 1
    public let runID: UUID
    public let ledgerID: UUID
    public let path: String
    public let device: UInt64
    public let inode: UInt64
}

public enum ControlRecoveryPhase: UInt32, Codable, Sendable { case preparing = 1, armed, stopping, recovering, settled }

public struct ControlRecoverySnapshot: Sendable {
    public let phase: ControlRecoveryPhase
    public let possibleKeys: Set<Int>
    public let possibleButtons: Set<Int>
    public let inFlight: UInt32
    public let executorPID: Int32
    public let guardianPID: Int32
    public let guardianReady: Bool
    public let everArmed: Bool
    public let guardianIdentity: UInt64
    public let guardianHeartbeatNanos: UInt64
    public let completedNanos: UInt64
    public let recoveredByGuardian: Bool
    public let errorCode: UInt32
    public var cleanupConfirmed: Bool { phase == .settled && possibleKeys.isEmpty && possibleButtons.isEmpty && inFlight == 0 }
}

/// A process-crash ledger, not a durable event journal. Every mutable shared
/// word uses lock-free C11 atomics; no Swift reference or process-local lock is
/// placed in mapped storage. A surviving peer owns the dirty mapping and lease.
public final class ControlRecoveryLedger: @unchecked Sendable {
    public static let byteCount = Int(ASTRA_RECOVERY_BYTES)
    public let descriptor: ControlRecoveryDescriptor
    private let file: Int32
    private let address: UnsafeMutableRawPointer
    private let capabilityWords: [UInt64]

    public convenience init(createAt url: URL, runID: UUID, capabilities: ActionCapabilities) throws {
        _ = try capabilities.validated()
        let fd = open(url.path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw AstraError("control.recoveryCreate", "The private control recovery ledger could not be created.") }
        do {
            var reservation = fstore_t(); reservation.fst_flags = UInt32(F_ALLOCATEALL)
            reservation.fst_posmode = F_PEOFPOSMODE; reservation.fst_length = off_t(Self.byteCount)
            guard fcntl(fd, F_PREALLOCATE, &reservation) == 0, reservation.fst_bytesalloc >= Self.byteCount,
                  ftruncate(fd, off_t(Self.byteCount)) == 0 else {
                throw AstraError("control.recoverySpace", "Control recovery storage could not be reserved before arming.")
            }
            try self.init(fd: fd, path: url.path, create: (runID, UUID(), capabilities), expected: nil)
        } catch { close(fd); try? FileManager.default.removeItem(at: url); throw error }
    }

    public convenience init(open descriptor: ControlRecoveryDescriptor) throws {
        guard descriptor.version == 1, descriptor.path.hasPrefix("/") else { throw AstraError("control.recoveryVersion", "The control recovery descriptor is invalid.") }
        let fd = Darwin.open(descriptor.path, O_RDWR | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard fd >= 0 else { throw AstraError("control.recoveryOpen", "The control recovery ledger is missing or linked.") }
        do { try self.init(fd: fd, path: descriptor.path, create: nil, expected: descriptor) }
        catch { close(fd); throw error }
    }

    public convenience init(inheritedDescriptor: Int32) throws {
        let fd = fcntl(inheritedDescriptor, F_DUPFD_CLOEXEC, 64)
        guard fd >= 0 else { throw AstraError("control.recoveryDescriptor", "The inherited recovery descriptor is unavailable.") }
        do { try self.init(fd: fd, path: "", create: nil, expected: nil) }
        catch { close(fd); throw error }
    }

    private init(fd: Int32, path: String, create: (UUID, UUID, ActionCapabilities)?, expected: ControlRecoveryDescriptor?) throws {
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_uid == getuid(), info.st_nlink == 1,
              info.st_mode & 0o077 == 0, info.st_size == Self.byteCount else {
            throw AstraError("control.recoveryFile", "The recovery ledger must be an owned private file of the exact supported size.")
        }
        let device = UInt64(truncatingIfNeeded: info.st_dev), inode = UInt64(info.st_ino)
        if let expected, expected.device != device || expected.inode != inode {
            throw AstraError("control.recoveryIdentity", "The control recovery ledger was replaced.")
        }
        guard let mapping = mmap(nil, Self.byteCount, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0), mapping != MAP_FAILED else {
            throw AstraError("control.recoveryMap", "The recovery ledger could not be mapped.")
        }
        do {
            if let (runID, ledgerID, capabilities) = create {
                var run = runID.uuid, ledger = ledgerID.uuid
                let words = Self.words(capabilities)
                let initialized = withUnsafeBytes(of: &run) { r in withUnsafeBytes(of: &ledger) { l in
                    words.withUnsafeBufferPointer { c in astra_recovery_initialize(mapping, r.baseAddress!.assumingMemoryBound(to: UInt8.self), l.baseAddress!.assumingMemoryBound(to: UInt8.self), c.baseAddress!) }
                } }
                guard initialized, msync(mapping, Self.byteCount, MS_SYNC) == 0, fsync(fd) == 0 else {
                    throw AstraError("control.recoveryInitialize", "Atomic recovery storage could not be initialized.")
                }
            }
            var raw = astra_recovery_snapshot()
            guard astra_recovery_read(mapping, &raw) else { throw AstraError("control.recoveryHeader", "Recovery storage has an invalid or unsupported atomic header.") }
            let runID = UUID(uuid: raw.run), ledgerID = UUID(uuid: raw.ledger)
            if let expected, expected.runID != runID || expected.ledgerID != ledgerID { throw AstraError("control.recoveryIdentity", "Recovery storage belongs to another run.") }
            descriptor = ControlRecoveryDescriptor(runID: runID, ledgerID: ledgerID, path: path, device: device, inode: inode)
            capabilityWords = [raw.capabilities.0, raw.capabilities.1, raw.capabilities.2]
            file = fd; address = mapping
        } catch { munmap(mapping, Self.byteCount); throw error }
    }

    deinit { munmap(address, Self.byteCount); close(file) }
    public func withFileDescriptor<T>(_ body: (Int32) throws -> T) rethrows -> T { try body(file) }
    public func matches(_ capabilities: ActionCapabilities) -> Bool { capabilityWords == Self.words(capabilities) }
    public func snapshot() throws -> ControlRecoverySnapshot {
        var raw = astra_recovery_snapshot()
        var valid = false
        for _ in 0..<4 { if astra_recovery_read(address, &raw) { valid = true; break } }
        guard valid, let phase = ControlRecoveryPhase(rawValue: raw.phase), raw.executor_pid <= Int32.max, raw.guardian_pid <= Int32.max,
              raw.possible.0 & ~capabilityWords[0] == 0, raw.possible.1 & ~capabilityWords[1] == 0,
              raw.possible.2 & ~capabilityWords[2] == 0 else { throw AstraError("control.recoveryCorrupt", "The control recovery ledger is inconsistent.") }
        let keys = Set((0..<128).filter { code in (code < 64 ? raw.possible.0 : raw.possible.1) & (UInt64(1) << (code % 64)) != 0 })
        let buttons = Set((0..<32).filter { raw.possible.2 & (UInt64(1) << $0) != 0 })
        return ControlRecoverySnapshot(phase: phase, possibleKeys: keys, possibleButtons: buttons, inFlight: raw.in_flight,
            executorPID: Int32(raw.executor_pid), guardianPID: Int32(raw.guardian_pid), guardianReady: raw.guardian_ready == 1,
            everArmed: raw.ever_armed == 1, guardianIdentity: raw.guardian_identity,
            guardianHeartbeatNanos: raw.guardian_heartbeat, completedNanos: raw.completed_nanos,
            recoveredByGuardian: raw.performer == 2, errorCode: raw.error_code)
    }
    public func registerExecutor(pid: Int32) -> Bool { pid > 0 && astra_recovery_executor(address, UInt32(pid)) }
    public func registerGuardian(pid: Int32, now: UInt64) -> Bool { pid > 0 && astra_recovery_guardian(address, UInt32(pid), now) }
    public func heartbeat(now: UInt64) { astra_recovery_heartbeat(address, now) }
    public func arm() -> Bool { astra_recovery_arm(address) }
    public func guardianIsFresh(now: UInt64) -> Bool { astra_recovery_guardian_fresh(address, now, 150_000_000) }
    public func stop() { astra_recovery_stop(address) }
    public func beginPost(operation: CommandOperation, keyCode: Int?, button: Int?) -> Bool {
        let kind: UInt32 = keyCode != nil ? 1 : button != nil ? 2 : 0
        guard let code = UInt32(exactly: keyCode ?? button ?? 0) else { return false }
        return astra_recovery_begin_post(address, kind, code, operation == .keyDown || operation == .buttonDown)
    }
    public func endPost(operation: CommandOperation, keyCode: Int?, button: Int?, success: Bool) {
        let kind: UInt32 = keyCode != nil ? 1 : button != nil ? 2 : 0
        astra_recovery_end_post(address, kind, UInt32(keyCode ?? button ?? 0), success && (operation == .keyUp || operation == .buttonUp))
    }
    @discardableResult public func settleLocally(now: UInt64) -> Bool { astra_recovery_local_settle(address, now) }
    /// Only the guardian, after a kernel-confirmed executor exit, may call this.
    public func claimAfterExecutorExit() -> Bool { astra_recovery_claim_after_exit(address) }
    public func releaseKey(_ key: Int) { _ = astra_recovery_release(address, 1, UInt32(key)) }
    public func releaseButton(_ button: Int) { _ = astra_recovery_release(address, 2, UInt32(button)) }
    @discardableResult public func settleGuardian(now: UInt64) -> Bool { astra_recovery_guardian_settle(address, now) }
    public func recordError(_ code: UInt32) { astra_recovery_error(address, code) }
    /// Used after the command process exits, never as permission to post.
    /// Zero means no matching live process; PID reuse cannot impersonate a peer.
    public func matchingGuardianIsAlive(_ snapshot: ControlRecoverySnapshot) -> Bool {
        snapshot.guardianPID > 0 && snapshot.guardianIdentity != 0 && astra_process_identity(snapshot.guardianPID) == snapshot.guardianIdentity
    }
    public func synchronizeTerminal() throws {
        guard msync(address, Self.byteCount, MS_SYNC) == 0, fsync(file) == 0 else {
            throw AstraError("control.recoverySync", "Input cleanup settled but its recovery record could not be synchronized.")
        }
    }
    private static func words(_ capabilities: ActionCapabilities) -> [UInt64] {
        var result: [UInt64] = [0, 0, 0]
        for key in capabilities.keyCodes where (0..<128).contains(key) { result[key / 64] |= UInt64(1) << (key % 64) }
        for button in capabilities.mouseButtons where (0..<32).contains(button) { result[2] |= UInt64(1) << button }
        return result
    }
}
