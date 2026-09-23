import Foundation
import CryptoKit
import Darwin

public struct CorrectionReference: Codable, Hashable, Sendable {
    public var schemaVersion = 1
    public let path: String
    public let sha256: String
    public func validated() throws -> Self {
        guard schemaVersion == 1, path == "correction/prelude.json", sha256.count == 64,
              sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw AstraError("correction.reference", "The correction provenance reference is invalid.")
        }
        return self
    }
}

public struct CorrectionSourceFrame: Sendable {
    public let metadata: FrameMetadata
    public let pixels: Data
    public let coverage: CaptureFrameCoverage?
    public init(metadata: FrameMetadata, pixels: Data, coverage: CaptureFrameCoverage?) {
        self.metadata = metadata; self.pixels = pixels; self.coverage = coverage
    }
}
public struct CorrectionSourceObservation: Sendable {
    public let actorInput: JSONValue
    public let frames: [CorrectionSourceFrame]
    public init(actorInput: JSONValue, frames: [CorrectionSourceFrame]) { self.actorInput = actorInput; self.frames = frames }
}
public struct CorrectionRecordingSeed: Sendable {
    public static let maximumDurationNanos: UInt64 = 2_000_000_000
    public static let maximumBytes = 256 * 1024 * 1024
    public let sourceRunID: UUID
    public let sourceCheckpointID: UUID
    public let sourcePolicySignature: String
    public let contextIDs: [Int]
    public let requestedAtNanos: UInt64
    public let controlJoinedAtNanos: UInt64
    public let observations: [CorrectionSourceObservation]
    public init(sourceRunID: UUID, sourceCheckpointID: UUID, sourcePolicySignature: String, contextIDs: [Int],
                requestedAtNanos: UInt64, controlJoinedAtNanos: UInt64, observations: [CorrectionSourceObservation]) {
        self.sourceRunID = sourceRunID; self.sourceCheckpointID = sourceCheckpointID; self.sourcePolicySignature = sourcePolicySignature
        self.contextIDs = contextIDs; self.requestedAtNanos = requestedAtNanos; self.controlJoinedAtNanos = controlJoinedAtNanos
        self.observations = observations
    }
}

/// The disconnected lead-up is inspectable evidence, never expert supervision.
/// The physical recording keeps its own original input sequence and clock.
public struct CorrectionPrelude: Codable, Sendable {
    public struct Frame: Codable, Sendable {
        public let shard: String
        public let block: FrameBlock
        public let coverage: CaptureFrameCoverage?
    }
    public struct Observation: Codable, Sendable {
        public let actorInput: JSONValue
        public let frames: [Frame]
    }
    public let schemaVersion: Int
    public let sourceRunID: UUID
    public let sourceCheckpointID: UUID
    public let sourcePolicySignature: String
    public let contextIDs: [Int]
    public let requestedAtNanos: UInt64
    public let controlJoinedAtNanos: UInt64
    public let supervisionStartNanos: UInt64
    public let maximumDurationNanos: UInt64
    public let maximumBytes: Int
    public let continuityProven: Bool
    public let observations: [Observation]

    public func validated() throws -> Self {
        guard schemaVersion == 1, !continuityProven,
              sourcePolicySignature.count == 64,
              sourcePolicySignature.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              contextIDs.count <= 32, contextIDs.allSatisfy({ (0..<65_536).contains($0) }),
              supervisionStartNanos >= max(requestedAtNanos, controlJoinedAtNanos), supervisionStartNanos <= UInt64(Int64.max),
              maximumDurationNanos == CorrectionRecordingSeed.maximumDurationNanos,
              maximumBytes == CorrectionRecordingSeed.maximumBytes, observations.count <= 256 else {
            throw AstraError("correction.provenance", "The correction's source identity, handoff or retention limits are invalid.")
        }
        var previous: UInt64?, first: UInt64?, bytes = 0
        var roles: [String]?
        var observationIDs = Set<UUID>()
        for observation in observations {
            let cutoff = try observation.actorInput.required("cutoffNanos").decode(UInt64.self)
            let id = try observation.actorInput.requiredUUID("observationID")
            guard cutoff <= min(requestedAtNanos, controlJoinedAtNanos), previous.map({ cutoff > $0 }) ?? true,
                  observationIDs.insert(id).inserted, (1...16).contains(observation.frames.count),
                  try observation.actorInput.required("contextIDs").decode([Int].self) == contextIDs else {
                throw AstraError("correction.observation", "Correction history must preserve distinct, causal observations from the selected run.")
            }
            let ids = observation.frames.map(\.block.metadata.surface.id)
            guard Set(ids).count == ids.count, roles == nil || roles == ids else {
                throw AstraError("correction.sources", "Correction history changed its ordered source roles.")
            }
            roles = ids; first = first ?? cutoff; previous = cutoff
            guard cutoff - first! <= maximumDurationNanos else { throw AstraError("correction.duration", "The correction history exceeds its time limit.") }
            for frame in observation.frames {
                let metadata = try frame.block.metadata.validated()
                guard frame.shard == "correction/frames.astraframes", metadata.eventNanos <= metadata.observedNanos,
                      metadata.observedNanos <= cutoff, frame.block.length > 0, frame.block.offset <= UInt64(Int64.max),
                      frame.block.length <= UInt64(FrameArchive.maximumFrameBytes + 65_588) else {
                    throw AstraError("correction.frame", "A correction source frame has invalid timing or archive bounds.")
                }
                if let coverage = frame.coverage { try coverage.validated(frame: metadata, cutoffNanos: cutoff, maximumAgeMS: 250) }
                else if cutoff - metadata.eventNanos > 250_000_000 { throw AstraError("correction.coverage", "A retained correction source has no recent coverage.") }
                bytes += metadata.byteCount
                guard bytes <= maximumBytes else { throw AstraError("correction.memory", "The correction history exceeds its pixel budget.") }
            }
        }
        return self
    }

    public static func load(in directory: URL, reference: CorrectionReference) throws -> Self {
        _ = try reference.validated()
        let bytes = try readFile(in: directory, name: "prelude.json", limit: 16 * 1024 * 1024)
        guard SHA256.hash(data: bytes).map({ String(format: "%02x", $0) }).joined() == reference.sha256 else {
            throw AstraError("correction.integrity", "The correction provenance failed its integrity check.")
        }
        return try JSONDecoder().decode(Self.self, from: bytes).validated()
    }

    public func pixels(for frame: Frame, in directory: URL) throws -> Data {
        guard frame.shard == "correction/frames.astraframes" else { throw AstraError("correction.path", "The correction frame path is invalid.") }
        let descriptor = try Self.openCorrectionFile(in: directory, name: "frames.astraframes")
        guard descriptor >= 0 else { throw AstraError("correction.archive", "The correction archive is missing or linked.") }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true); defer { try? file.close() }
        guard let (block, pixels) = try FrameArchive.read(from: file, offset: frame.block.offset), block == frame.block else {
            throw AstraError("correction.frameIntegrity", "The correction frame differs from its original archive reference.")
        }
        return pixels
    }

    public static func write(_ seed: CorrectionRecordingSeed, supervisionStartNanos: UInt64, in directory: URL) throws -> CorrectionReference {
        guard seed.observations.count <= 256 else { throw AstraError("correction.limit", "Too many correction observations were retained.") }
        let folder = directory.appendingPathComponent("correction", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let archiveURL = folder.appendingPathComponent("frames.astraframes")
        let descriptor = open(archiveURL.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw AstraError("correction.archive", "The correction archive could not be created.") }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true); defer { try? file.close() }
        var blocks: [UUID: FrameBlock] = [:], digests: [UUID: SHA256.Digest] = [:], retained = 0
        let observations = try seed.observations.map { observation -> Observation in
            let frames = try observation.frames.map { frame -> Frame in
                retained += frame.pixels.count
                guard retained <= CorrectionRecordingSeed.maximumBytes else { throw AstraError("correction.memory", "The correction exceeds its pixel budget.") }
                let block: FrameBlock
                if let existing = blocks[frame.metadata.id] {
                    var normalized = frame.metadata; normalized.codec = existing.metadata.codec
                    guard normalized == existing.metadata, digests[frame.metadata.id] == SHA256.hash(data: frame.pixels) else { throw AstraError("correction.frameIdentity", "A retained frame changed identity.") }
                    block = existing
                } else {
                    block = try FrameArchive.append(pixels: frame.pixels, metadata: frame.metadata, to: file)
                    blocks[frame.metadata.id] = block; digests[frame.metadata.id] = SHA256.hash(data: frame.pixels)
                }
                return Frame(shard: "correction/frames.astraframes", block: block, coverage: frame.coverage)
            }
            return Observation(actorInput: observation.actorInput, frames: frames)
        }
        let prelude = try Self(schemaVersion: 1, sourceRunID: seed.sourceRunID, sourceCheckpointID: seed.sourceCheckpointID,
            sourcePolicySignature: seed.sourcePolicySignature, contextIDs: seed.contextIDs, requestedAtNanos: seed.requestedAtNanos,
            controlJoinedAtNanos: seed.controlJoinedAtNanos, supervisionStartNanos: supervisionStartNanos,
            maximumDurationNanos: CorrectionRecordingSeed.maximumDurationNanos, maximumBytes: CorrectionRecordingSeed.maximumBytes,
            continuityProven: false, observations: observations).validated()
        try file.synchronize()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let bytes = try encoder.encode(prelude)
        guard bytes.count <= 16 * 1024 * 1024 else { throw AstraError("correction.limit", "The correction provenance exceeds its size limit.") }
        let destination = folder.appendingPathComponent("prelude.json")
        try bytes.write(to: destination, options: .withoutOverwriting)
        let saved = try FileHandle(forWritingTo: destination); try saved.synchronize(); try saved.close()
        let fd = open(folder.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard fd >= 0 else { throw AstraError("correction.sync", "The correction directory could not be synchronized.") }
        defer { close(fd) }; guard fsync(fd) == 0 else { throw AstraError("correction.sync", "The correction directory could not be synchronized.") }
        return .init(path: "correction/prelude.json", sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
    }

    private static func openCorrectionFile(in directory: URL, name: String) throws -> Int32 {
        let root = open(directory.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard root >= 0 else { throw AstraError("correction.path", "The correction recording directory is missing or linked.") }
        defer { close(root) }
        let folder = openat(root, "correction", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard folder >= 0 else { throw AstraError("correction.path", "The correction history directory is missing or linked.") }
        defer { close(folder) }
        let fd = openat(folder, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else { throw AstraError("correction.file", "The correction source is missing or linked.") }
        return fd
    }
    private static func readFile(in directory: URL, name: String, limit: Int) throws -> Data {
        let fd = try openCorrectionFile(in: directory, name: name)
        let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? file.close() }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size >= 0, info.st_size <= limit,
              let bytes = try file.read(upToCount: limit + 1), bytes.count == info.st_size else {
            throw AstraError("correction.file", "The correction provenance is not a bounded regular file.")
        }
        return bytes
    }
}
