import Foundation
import Compression
import CryptoKit

public struct FrameMetadata: Codable, Hashable, Sendable {
    public var id: UUID
    public var eventNanos: UInt64
    public var observedNanos: UInt64
    public var surface: SurfaceDescriptor
    public var byteCount: Int
    public var pixelFormat: String
    public var codec: String
    public init(id: UUID = UUID(), eventNanos: UInt64, observedNanos: UInt64, surface: SurfaceDescriptor,
                byteCount: Int, pixelFormat: String = "bgra8-srgb", codec: String = "lzfse") {
        self.id = id; self.eventNanos = eventNanos; self.observedNanos = observedNanos
        self.surface = surface; self.byteCount = byteCount; self.pixelFormat = pixelFormat; self.codec = codec
    }
    public func validated() throws -> Self {
        _ = try surface.validated()
        let pixels = surface.pixelWidth.multipliedReportingOverflow(by: surface.pixelHeight)
        let size = pixels.partialValue.multipliedReportingOverflow(by: 4)
        guard !pixels.overflow, !size.overflow, byteCount == size.partialValue,
              byteCount > 0, byteCount <= FrameArchive.maximumFrameBytes,
              pixelFormat == "bgra8-srgb", ["lzfse", "raw"].contains(codec) else {
            throw AstraError("recording.frame", "The recording frame has an invalid pixel format or byte size.")
        }
        return self
    }
}

public struct FrameBlock: Codable, Hashable, Sendable {
    public var offset: UInt64
    public var length: UInt64
    public var metadata: FrameMetadata
    public var digest: String
}
public struct ArchiveScan: Sendable {
    public var blocks: [FrameBlock]
    public var validBytes: UInt64
    public var incompleteTail: Bool
    public var error: String?
}

public struct PreparedFrame: Sendable {
    public let metadata: FrameMetadata
    public let byteCount: Int
    fileprivate let prefix: Data
    fileprivate let metadataBytes: Data
    fileprivate let payload: Data
    fileprivate let digest: Data
}

/// Independently addressable, checksummed blocks. Scanning is read-only and keeps
/// a corrupt/incomplete tail available for inspection rather than deleting data.
public enum FrameArchive {
    public static let maximumFrameBytes = 256 * 1_024 * 1_024
    private static let magic = Data("ASTRAF01".utf8)
    private static let maximumMetadataBytes = 65_536

    public static func append(pixels: Data, metadata original: FrameMetadata, to handle: FileHandle) throws -> FrameBlock {
        try appendPrepared(prepare(pixels: pixels, metadata: original), to: handle)
    }

    /// Compression may run on bounded parallel workers; publication remains
    /// ordered on the recording writer's single queue.
    public static func prepare(pixels: Data, metadata original: FrameMetadata) throws -> PreparedFrame {
        var metadata = try original.validated()
        guard pixels.count == metadata.byteCount else { throw AstraError("recording.frameBytes", "The captured pixel buffer has the wrong length.") }
        let compressed = try (pixels as NSData).compressed(using: .lzfse) as Data
        let payload: Data
        if compressed.count < pixels.count { payload = compressed; metadata.codec = "lzfse" }
        else { payload = pixels; metadata.codec = "raw" }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let metadataBytes = try encoder.encode(metadata)
        guard metadataBytes.count <= maximumMetadataBytes else { throw AstraError("recording.metadata", "Frame metadata is too large.") }
        var prefix = magic
        prefix.appendLittleEndian(UInt32(metadataBytes.count))
        prefix.appendLittleEndian(UInt64(payload.count))
        var hash = SHA256(); hash.update(data: prefix); hash.update(data: metadataBytes); hash.update(data: payload)
        let digest = Data(hash.finalize())
        return PreparedFrame(metadata: metadata, byteCount: prefix.count + metadataBytes.count + payload.count + digest.count,
                             prefix: prefix, metadataBytes: metadataBytes, payload: payload, digest: digest)
    }

    public static func appendPrepared(_ frame: PreparedFrame, to handle: FileHandle) throws -> FrameBlock {
        let offset = try handle.offset()
        try handle.write(contentsOf: frame.prefix)
        try handle.write(contentsOf: frame.metadataBytes)
        try handle.write(contentsOf: frame.payload)
        try handle.write(contentsOf: frame.digest)
        return FrameBlock(offset: offset, length: UInt64(frame.byteCount), metadata: frame.metadata, digest: frame.digest.hex)
    }

    public static func read(from handle: FileHandle, offset: UInt64) throws -> (FrameBlock, Data)? {
        try handle.seek(toOffset: offset)
        let prefix = try handle.read(upToCount: 20) ?? Data()
        if prefix.isEmpty { return nil }
        guard prefix.count == 20, prefix.prefix(8) == magic else {
            throw AstraError("recording.blockHeader", "A frame block is truncated or has an invalid header.")
        }
        let metadataCount = Int(prefix.littleEndian(UInt32.self, at: 8))
        let payloadCount = prefix.littleEndian(UInt64.self, at: 12)
        guard (1...maximumMetadataBytes).contains(metadataCount), payloadCount > 0,
              payloadCount <= UInt64(maximumFrameBytes) else {
            throw AstraError("recording.blockSize", "A frame block declares an unsupported length.")
        }
        let metadataBytes = try exactRead(handle, count: metadataCount)
        let metadata = try JSONDecoder().decode(FrameMetadata.self, from: metadataBytes).validated()
        let payload = try exactRead(handle, count: Int(payloadCount))
        let expected = try exactRead(handle, count: 32)
        var hash = SHA256(); hash.update(data: prefix); hash.update(data: metadataBytes); hash.update(data: payload)
        guard Data(hash.finalize()) == expected else { throw AstraError("recording.checksum", "A recording frame failed its integrity check.") }
        let pixels: Data
        if metadata.codec == "raw" { pixels = payload }
        else {
            var output = Data(count: metadata.byteCount)
            let count = output.withUnsafeMutableBytes { destination in
                payload.withUnsafeBytes { source in
                    compression_decode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                                              source.bindMemory(to: UInt8.self).baseAddress!, source.count, nil, COMPRESSION_LZFSE)
                }
            }
            guard count == metadata.byteCount else { throw AstraError("recording.decompression", "A frame cannot be decoded to its declared size.") }
            pixels = output
        }
        guard pixels.count == metadata.byteCount else { throw AstraError("recording.frameBytes", "A recording frame has the wrong decoded length.") }
        return (FrameBlock(offset: offset, length: UInt64(20 + metadataCount + Int(payloadCount) + 32),
                           metadata: metadata, digest: expected.hex), pixels)
    }

    public static func scan(url: URL) throws -> ArchiveScan {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var blocks: [FrameBlock] = []
        var offset: UInt64 = 0
        do {
            while let (block, _) = try read(from: handle, offset: offset) {
                blocks.append(block); offset += block.length
            }
            return ArchiveScan(blocks: blocks, validBytes: offset, incompleteTail: false, error: nil)
        } catch {
            return ArchiveScan(blocks: blocks, validBytes: offset, incompleteTail: true, error: error.localizedDescription)
        }
    }

    private static func exactRead(_ handle: FileHandle, count: Int) throws -> Data {
        var result = Data()
        while result.count < count {
            guard let part = try handle.read(upToCount: count - result.count), !part.isEmpty else {
                throw AstraError("recording.truncated", "The recording ends inside an unfinished frame block.")
            }
            result.append(part)
        }
        return result
    }
}

extension Data {
    fileprivate mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
    fileprivate func littleEndian<T: FixedWidthInteger>(_ type: T.Type, at offset: Int) -> T {
        withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: T.self)) }
    }
    public var hex: String { map { String(format: "%02x", $0) }.joined() }
}
