import Foundation
import CryptoKit
import Darwin
import AstraCore

/// Reads only the immutable package authenticated by feedback.inspect. Source
/// and index hashes bind frame identities; each selected pixel range is checked
/// again before it can be displayed. No pixels travel through the job protocol.
final class FeedbackSourceReader: Sendable {
    private struct ImageReference: Decodable, Sendable {
        let offset: Int64, bytes: Int, shape: [Int], checksum: String
        let metadata: FrameMetadata
    }
    private struct ObservationReference: Decodable, Sendable, Equatable {
        let id: UUID, cutoffNanos: UInt64
        let images: [ImageReference]
        static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.id == rhs.id && lhs.cutoffNanos == rhs.cutoffNanos
                && lhs.images.map(\.metadata) == rhs.images.map(\.metadata)
                && lhs.images.map(\.offset) == rhs.images.map(\.offset)
                && lhs.images.map(\.checksum) == rhs.images.map(\.checksum)
        }
    }
    private struct Row: Decodable { let packetID: UUID; let observation: ObservationReference; let endpoint: ObservationReference }
    let source: VerifiedFeedbackSource
    let directory: URL
    let manifestSHA256: String
    private let observations: [UUID: ObservationReference]
    private let frameBytes: Int64

    private init(source: VerifiedFeedbackSource, directory: URL, manifestSHA256: String,
                 observations: [UUID: ObservationReference], frameBytes: Int64) {
        self.source = source; self.directory = directory; self.manifestSHA256 = manifestSHA256
        self.observations = observations; self.frameBytes = frameBytes
    }

    static func open(inspection: JSONValue) async throws -> FeedbackSourceReader {
        try await Task.detached {
            let directory = URL(fileURLWithPath: try inspection.required("sourcePath").decode(String.self), isDirectory: true)
            let manifestHash = try inspection.required("manifestSHA256").decode(String.self)
            let manifestData = try read(directory.appendingPathComponent("manifest.json"), maximum: 1_048_576)
            guard digest(manifestData) == manifestHash else { throw AstraError("feedback.sourceChanged", "The review source changed after inspection.") }
            let manifest = try JSONDecoder().decode(JSONValue.self, from: manifestData)
            let artifacts = try manifest.required("artifacts")
            func member(_ name: String, maximum: Int) throws -> Data {
                let entry = try artifacts.required(name)
                let bytes = try read(directory.appendingPathComponent(name), maximum: maximum)
                guard entry.fields?["bytes"]?.int == bytes.count, entry.fields?["sha256"]?.text == digest(bytes) else {
                    throw AstraError("feedback.sourceChanged", "The original \(name) artifact failed its integrity check.")
                }
                return bytes
            }
            let metadata = try member("source.json", maximum: FeedbackLimits.maximumBytes)
            let program = try member("program.json", maximum: 1_048_576)
            let source = try VerifiedFeedbackSource(metadataBytes: metadata,
                expectedSHA256: try inspection.required("sourceSHA256").decode(String.self), programBytes: program)
            let index = try member("review-frames.ndjson", maximum: 512 * 1024 * 1024)
            let expected = Dictionary(uniqueKeysWithValues: source.metadata.intervals.map { ($0.target.packetID, $0) })
            var seen: Set<UUID> = [], observations: [UUID: ObservationReference] = [:]
            let extent = try artifacts.required("frames.bgra").required("bytes").decode(Int64.self)
            guard extent >= 0 else { throw AstraError("feedback.frameExtent", "Invalid original pixel extent.") }
            for line in index.split(separator: 10) {
                try Task.checkCancellation()
                guard line.count <= 1_048_576, seen.count < FeedbackLimits.maximumIntervals else {
                    throw AstraError("feedback.frameIndex", "The original observation index exceeds its limits.")
                }
                let row = try JSONDecoder().decode(Row.self, from: Data(line))
                guard let interval = expected[row.packetID], seen.insert(row.packetID).inserted,
                      row.observation.id == interval.target.observationID, row.observation.cutoffNanos == interval.target.startNanos,
                      row.endpoint.id == interval.endpointObservationID, row.endpoint.cutoffNanos == interval.target.endNanos else {
                    throw AstraError("feedback.frameIndex", "The source index does not match the exact reviewed interval.")
                }
                for observation in [row.observation, row.endpoint] {
                    guard (1...16).contains(observation.images.count), Set(observation.images.map(\.metadata.surface.id)).count == observation.images.count else {
                        throw AstraError("feedback.frameIndex", "The source observation has invalid surfaces.")
                    }
                    for image in observation.images {
                        _ = try image.metadata.validated()
                        guard image.offset >= 0, image.bytes == image.metadata.byteCount,
                              image.shape == [image.metadata.surface.pixelHeight, image.metadata.surface.pixelWidth, 4],
                              image.offset <= extent, Int64(image.bytes) <= extent - image.offset else {
                            throw AstraError("feedback.frameExtent", "The original frame range is outside its immutable source.")
                        }
                    }
                    if let previous = observations[observation.id], previous != observation {
                        throw AstraError("feedback.frameIndex", "One original observation has conflicting source ranges.")
                    }
                    observations[observation.id] = observation
                }
            }
            guard seen.count == expected.count else { throw AstraError("feedback.frameIndex", "The source is missing original review observations.") }
            return Self(source: source, directory: directory, manifestSHA256: manifestHash, observations: observations, frameBytes: extent)
        }.value
    }

    func load(_ request: FeedbackObservationRequest) async throws -> FeedbackReviewObservation {
        try await Task.detached { [self] in
            guard request.sourceSHA256 == source.sha256, request.trajectorySHA256 == source.metadata.trajectorySHA256,
                  let reference = observations[request.observationID], reference.cutoffNanos == request.cutoffNanos,
                  source.metadata.intervals.contains(where: { $0.target.episodeID == request.episodeID &&
                      ($0.target.observationID == request.observationID || $0.endpointObservationID == request.observationID) }),
                  reference.images.count <= request.maximumFrames,
                  reference.images.reduce(0, { $0 + $1.bytes }) <= request.maximumBytes else {
                throw AstraError("feedback.observation", "The preview request differs from its original source or byte budget.")
            }
            let fd = Self.openFile(directory.appendingPathComponent("frames.bgra"), maximum: frameBytes)
            let handle = try fd.get(); defer { try? handle.close() }
            var frames: [FeedbackReviewObservation.Frame] = []
            for item in reference.images {
                try Task.checkCancellation()
                try handle.seek(toOffset: UInt64(item.offset))
                guard let bytes = try handle.read(upToCount: item.bytes), bytes.count == item.bytes,
                      Self.digest(bytes) == item.checksum else { throw AstraError("feedback.frameChanged", "The original pixels failed their integrity check.") }
                frames.append(.init(metadata: item.metadata, pixels: bytes, pixelSHA256: item.checksum))
            }
            return FeedbackReviewObservation(sourceSHA256: source.sha256, trajectorySHA256: source.metadata.trajectorySHA256,
                episodeID: request.episodeID, observationID: request.observationID, cutoffNanos: request.cutoffNanos, frames: frames)
        }.value
    }

    private static func openFile(_ url: URL, maximum: Int64) -> Result<FileHandle, any Error> {
        let fd = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { return .failure(AstraError("feedback.sourceFile", "An original source file is unavailable.")) }
        var value = stat()
        guard fstat(fd, &value) == 0, value.st_mode & S_IFMT == S_IFREG, value.st_size >= 0, value.st_size <= maximum else {
            close(fd); return .failure(AstraError("feedback.sourceFile", "An original source file is linked, invalid or oversized."))
        }
        return .success(FileHandle(fileDescriptor: fd, closeOnDealloc: true))
    }
    private static func read(_ url: URL, maximum: Int) throws -> Data {
        let handle = try openFile(url, maximum: Int64(maximum)).get(); defer { try? handle.close() }
        let bytes = try handle.read(upToCount: maximum + 1) ?? Data()
        guard bytes.count <= maximum else { throw AstraError("feedback.sourceFile", "An original source artifact exceeds its byte limit.") }
        return bytes
    }
    private static func digest(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
}
