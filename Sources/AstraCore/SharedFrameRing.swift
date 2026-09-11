import Foundation
import CryptoKit
import Darwin
import Synchronization

public struct SharedFrameAcknowledgement: Codable, Hashable, Sendable {
    public let version: Int
    public let runID: UUID
    public let ringID: UUID
    public let slot: Int
    public let leaseID: UUID
    public let sequence: UInt64

    public init(version: Int = 1, runID: UUID, ringID: UUID, slot: Int, leaseID: UUID, sequence: UInt64) {
        self.version = version; self.runID = runID; self.ringID = ringID
        self.slot = slot; self.leaseID = leaseID; self.sequence = sequence
    }
}

public struct SharedFrameReference: Codable, Hashable, Sendable {
    public let version: Int
    public let runID: UUID
    public let ringID: UUID
    public let slot: Int
    public let leaseID: UUID
    public let sequence: UInt64
    public let offset: Int
    public let size: Int
    public let metadata: FrameMetadata

    public var acknowledgement: SharedFrameAcknowledgement {
        .init(version: version, runID: runID, ringID: ringID, slot: slot, leaseID: leaseID, sequence: sequence)
    }
}

/// One native writer and a pipe-coordinated consumer. A slot is immutable until
/// its exact lease is acknowledged after the consumer's owned MLX copy finishes.
/// This is shared mapped transport with one ingestion copy, not zero-copy MLX.
public final class SharedFrameRing: @unchecked Sendable {
    public static let version = 1
    public static let headerBytes = 128
    public static let slotHeaderBytes = 128
    public static let maximumSlots = 16
    public static let maximumFileBytes = 1_024 * 1_024 * 1_024
    public let url: URL
    public let runID: UUID
    public let ringID: UUID
    public let slotCount: Int
    public let slotCapacity: Int
    public let slotStride: Int
    public let byteCount: Int
    private let lock = NSLock()
    private var descriptor: Int32 = -1
    private var mapping: UnsafeMutableRawPointer?
    private var device: dev_t = 0
    private var inode: ino_t = 0
    private var leases: [SharedFrameAcknowledgement?]
    private var sequence: UInt64 = 0
    private var nextSlot = 0

    public init(url: URL, runID: UUID, slotCount: Int = 4, slotCapacity: Int) throws {
        guard url.isFileURL, (1...Self.maximumSlots).contains(slotCount),
              (1...FrameArchive.maximumFrameBytes).contains(slotCapacity) else {
            throw AstraError("frameRing.configuration", "The frame ring needs a local file and bounded slot dimensions.")
        }
        let stride = ((Self.slotHeaderBytes + slotCapacity + 63) / 64) * 64
        let size = Self.headerBytes + slotCount * stride
        guard size <= Self.maximumFileBytes else {
            throw AstraError("frameRing.capacity", "The frame ring exceeds its total memory budget.")
        }
        self.url = url; self.runID = runID; self.ringID = UUID()
        self.slotCount = slotCount; self.slotCapacity = slotCapacity
        self.slotStride = stride; self.byteCount = size
        self.leases = Array(repeating: nil, count: slotCount)
        descriptor = Darwin.open(url.path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw AstraError("frameRing.create", "The private frame ring could not be created: \(String(cString: strerror(errno))).")
        }
        do {
            var info = stat()
            guard fstat(descriptor, &info) == 0 else { throw Self.systemError("inspect") }
            device = info.st_dev; inode = info.st_ino
            guard (info.st_mode & S_IFMT) == S_IFREG, info.st_uid == getuid(), info.st_nlink == 1,
                  fchmod(descriptor, 0o600) == 0 else {
                throw Self.systemError("prepare")
            }
            // A sparse ftruncate is insufficient: later mapped writes can
            // SIGBUS on a full filesystem. Reserve every byte before mapping;
            // unsupported filesystems fail instead of taking a sparse fallback.
            var reservation = fstore_t()
            reservation.fst_flags = UInt32(F_ALLOCATEALL)
            reservation.fst_posmode = F_PEOFPOSMODE
            reservation.fst_offset = 0
            reservation.fst_length = off_t(size)
            guard fcntl(descriptor, F_PREALLOCATE, &reservation) == 0 else { throw Self.systemError("reserve storage for") }
            guard reservation.fst_bytesalloc >= off_t(size) else {
                throw AstraError("frameRing.reservation", "The filesystem did not reserve the complete frame ring; sparse transport is not supported.")
            }
            guard ftruncate(descriptor, off_t(size)) == 0 else { throw Self.systemError("size") }
            let address = mmap(nil, size, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0)
            guard let address, address != MAP_FAILED else { throw Self.systemError("map") }
            mapping = address
            var header = Data("ASTRAR01".utf8)
            header.ringAppend(UInt32(Self.version)); header.ringAppend(UInt32(Self.headerBytes))
            header.ringAppend(ringID); header.ringAppend(runID)
            header.ringAppend(UInt32(slotCount)); header.ringAppend(UInt32(Self.slotHeaderBytes))
            header.ringAppend(UInt64(slotCapacity)); header.ringAppend(UInt64(stride)); header.ringAppend(UInt64(size))
            header.append(Data(count: Self.headerBytes - header.count))
            header.withUnsafeBytes { address.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            atomicMemoryFence(ordering: .releasing)
        } catch {
            dispose()
            throw error
        }
    }

    deinit {
        // Retirement never changes slot bytes or truncates the inode. Existing
        // consumer mappings survive unlink/unmap; no slot is silently reclaimed.
        dispose()
    }

    public var pendingLeaseCount: Int {
        lock.lock(); defer { lock.unlock() }
        return leases.compactMap { $0 }.count
    }

    public func publish(pixels: Data, metadata: FrameMetadata) throws -> SharedFrameReference {
        try pixels.withUnsafeBytes { try publish(compactPixels: $0, metadata: metadata) }
    }

    /// The buffer is consumed synchronously. It must contain width*height*4
    /// compact BGRA bytes; callers must remove native row padding themselves.
    public func publish(compactPixels pixels: UnsafeRawBufferPointer, metadata original: FrameMetadata) throws -> SharedFrameReference {
        lock.lock(); defer { lock.unlock() }
        guard let mapping else { throw AstraError("frameRing.closed", "The frame ring is closed.") }
        var metadata = try original.validated()
        metadata.codec = "raw"
        guard pixels.count == metadata.byteCount, pixels.count <= slotCapacity, let source = pixels.baseAddress else {
            throw AstraError("frameRing.frameSize", "The compact frame does not fit its metadata or ring slot.")
        }
        guard let slot = (0..<slotCount).map({ (nextSlot + $0) % slotCount }).first(where: { leases[$0] == nil }) else {
            throw AstraError("frameRing.full", "Every frame slot is still owned by the consumer.")
        }
        guard sequence < UInt64.max else { throw AstraError("frameRing.sequence", "The frame ring publication sequence is exhausted.") }
        sequence += 1
        let lease = SharedFrameAcknowledgement(runID: runID, ringID: ringID, slot: slot, leaseID: UUID(), sequence: sequence)
        let start = Self.headerBytes + slot * slotStride
        let payloadOffset = start + Self.slotHeaderBytes
        let slotAddress = mapping.advanced(by: start)
        // Mark unavailable first. Publish the complete header with state=1,
        // then make state=2 visible only after payload and metadata are ready.
        slotAddress.storeBytes(of: UInt32(1).littleEndian, as: UInt32.self)
        mapping.advanced(by: payloadOffset).copyMemory(from: source, byteCount: pixels.count)
        var header = Data()
        header.ringAppend(UInt32(1)); header.ringAppend(UInt32(Self.version))
        header.ringAppend(UInt32(slot)); header.ringAppend(UInt32(0))
        header.ringAppend(lease.leaseID); header.ringAppend(metadata.id)
        header.ringAppend(sequence); header.ringAppend(UInt64(pixels.count))
        header.append(Self.fingerprint(metadata)); header.append(Data(count: 32))
        header.withUnsafeBytes { slotAddress.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        atomicMemoryFence(ordering: .releasing)
        slotAddress.storeBytes(of: UInt32(2).littleEndian, as: UInt32.self)
        atomicMemoryFence(ordering: .releasing)
        leases[slot] = lease
        nextSlot = (slot + 1) % slotCount
        return SharedFrameReference(version: Self.version, runID: runID, ringID: ringID, slot: slot,
                                    leaseID: lease.leaseID, sequence: sequence, offset: payloadOffset,
                                    size: pixels.count, metadata: metadata)
    }

    public func release(_ acknowledgement: SharedFrameAcknowledgement) throws {
        lock.lock(); defer { lock.unlock() }
        guard let mapping else { throw AstraError("frameRing.closed", "The frame ring is closed.") }
        guard acknowledgement.version == Self.version, acknowledgement.runID == runID, acknowledgement.ringID == ringID,
              (0..<slotCount).contains(acknowledgement.slot), leases[acknowledgement.slot] == acknowledgement else {
            throw AstraError("frameRing.staleLease", "The frame acknowledgement does not match a current lease.")
        }
        let address = mapping.advanced(by: Self.headerBytes + acknowledgement.slot * slotStride)
        address.storeBytes(of: UInt32(0), as: UInt32.self)
        atomicMemoryFence(ordering: .releasing)
        leases[acknowledgement.slot] = nil
    }

    public func close() throws {
        lock.lock(); defer { lock.unlock() }
        guard !leases.contains(where: { $0 != nil }) else {
            throw AstraError("frameRing.pendingLeases", "Wait for frame acknowledgements or terminate the consumer before retiring this ring.")
        }
        dispose()
    }

    /// Only for coordinator cleanup after it has joined the consumer process.
    /// A timeout alone is not proof of released ownership. This retires the
    /// entire inode; it never permits outstanding slots to be reused.
    public func closeAfterConsumerExit() {
        lock.lock(); defer { lock.unlock() }
        dispose()
        leases = Array(repeating: nil, count: slotCount)
    }

    private func dispose() {
        if let mapping { munmap(mapping, byteCount); self.mapping = nil }
        if descriptor >= 0 {
            var info = stat()
            if lstat(url.path, &info) == 0, info.st_dev == device, info.st_ino == inode { _ = unlink(url.path) }
            _ = Darwin.close(descriptor); descriptor = -1
        }
    }

    private static func systemError(_ operation: String) -> AstraError {
        AstraError("frameRing.io", "Could not \(operation) the frame ring: \(String(cString: strerror(errno))).")
    }

    private static func fingerprint(_ frame: FrameMetadata) -> Data {
        var data = Data("ASTRAM01".utf8)
        data.ringAppend(frame.id); data.ringAppend(frame.eventNanos); data.ringAppend(frame.observedNanos)
        let identity = Data(frame.surface.id.utf8)
        data.ringAppend(UInt32(identity.count)); data.append(identity)
        data.ringAppend(frame.surface.globalBounds)
        data.ringAppend(UInt32(frame.surface.pixelWidth)); data.ringAppend(UInt32(frame.surface.pixelHeight))
        data.ringAppend(frame.surface.contentBounds)
        data.ringAppend(frame.surface.geometryRevision); data.ringAppend(UInt64(frame.byteCount))
        data.append(Data("bgra8-srgb\0raw\0".utf8))
        return Data(SHA256.hash(data: data))
    }
}

private extension Data {
    mutating func ringAppend<T: FixedWidthInteger>(_ value: T) {
        var bytes = value.littleEndian
        Swift.withUnsafeBytes(of: &bytes) { append(contentsOf: $0) }
    }
    mutating func ringAppend(_ value: UUID) {
        var bytes = value.uuid
        Swift.withUnsafeBytes(of: &bytes) { append(contentsOf: $0) }
    }
    mutating func ringAppend(_ rect: Rect2D) {
        for value in [rect.x, rect.y, rect.width, rect.height] {
            // JSON normalizes signed zero. Its geometry is identical, so use a
            // positive-zero fingerprint in both languages.
            ringAppend((value == 0 ? Double(0) : value).bitPattern)
        }
    }
}
