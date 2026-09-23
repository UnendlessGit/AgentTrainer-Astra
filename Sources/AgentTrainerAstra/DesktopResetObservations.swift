import Foundation
import AstraCore
import AstraPlatform

/// Bounded reset observation work on one analysis queue. This owns detector
/// caching and immutable pixels, never capture or physical control. The reset
/// driver joins each operation before it can confirm readiness or release.
final class DesktopResetObservations: @unchecked Sendable {
    typealias ReadFrames = @Sendable () throws -> [InferenceImage]
    typealias ReadManual = @Sendable (ResetContext, UInt64) throws -> [SignalReading]
    private let program: RewardProgram
    private let scope: ControlScope
    private let templates: [String: Data]
    private let frames: ReadFrames
    private let manual: ReadManual
    private let detector: RewardAnalysisQueue.Detector
    private let clock: @Sendable () -> UInt64
    private let queue = DispatchQueue(label: "astra.desktop.reset-observation", qos: .userInitiated)
    private let lock = NSLock()
    private var reservations = 0
    // Only the analysis queue touches the detector cache.
    private var cachedEpisode: UUID?
    private var cachedFrames: [FrameMetadata] = []
    private var cachedReadings: [SignalReading] = []

    private init(program: RewardProgram, scope: ControlScope, templates: [String: Data], frames: @escaping ReadFrames,
                 manual: @escaping ReadManual, detector: @escaping RewardAnalysisQueue.Detector,
                 clock: @escaping @Sendable () -> UInt64) {
        self.program = program; self.scope = scope; self.templates = templates; self.frames = frames
        self.manual = manual; self.detector = detector; self.clock = clock
    }

    static func prepare(program: RewardProgram, scope: ControlScope, assetRoot: URL, frames: @escaping ReadFrames,
                        manual: @escaping ReadManual = { _, _ in [] },
                        detector: @escaping RewardAnalysisQueue.Detector = { try VisualRewardDetector.read(signals: $0, frames: $1, episodeID: $2, templates: $3) },
                        clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now }) async throws -> DesktopResetObservations {
        let program = try program.validated(), scope = try scope.validated()
        let prepared = try await Task.detached {
            var templates: [String: Data] = [:]
            for digest in Set(program.signals.compactMap(\.templateDigest)) { templates[digest] = try RewardAssets.read(digest, root: assetRoot) }
            let value = Self(program: program, scope: scope, templates: templates, frames: frames, manual: manual, detector: detector, clock: clock)
            let warm = try value.snapshot()
            // Warming does not populate actual episode readings or readiness.
            _ = try detector(program.signals, warm.frames, UUID(), templates)
            return value
        }.value
        try Task.checkCancellation()
        return prepared
    }

    func observe(context: ResetContext) async throws -> ResetObservationSnapshot {
        guard context.scope == scope else { throw AstraError("reset.observationScope", "The reset observation source is bound to different geometry.") }
        try Task.checkCancellation()
        try lock.withLock {
            guard reservations < 4 else { throw AstraError("reset.observationBackpressure", "The reset observation queue is full.") }
            reservations += 1
        }
        return try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                defer { lock.withLock { reservations -= 1 } }
                do {
                    let value = try snapshot()
                    let metadata = value.frames.map(\.metadata)
                    if cachedEpisode != context.nextEpisodeID || cachedFrames != metadata {
                        cachedReadings = try detector(program.signals, value.frames, context.nextEpisodeID, templates)
                        cachedFrames = metadata; cachedEpisode = context.nextEpisodeID
                    }
                    let manualReadings = try manual(context, value.cutoff)
                    let manualIDs = Set(program.signals.filter { $0.kind == .manual }.map(\.id))
                    guard manualReadings.allSatisfy({ manualIDs.contains($0.signalID) }) else {
                        throw AstraError("reset.manualSource", "Manual readings cannot replace a visual or elapsed-time signal.")
                    }
                    continuation.resume(returning: ResetObservationSnapshot(context: context, observedNanos: value.cutoff,
                        sourceCoverage: value.coverage, readings: cachedReadings + manualReadings))
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func snapshot() throws -> (frames: [RewardImageFrame], coverage: [RewardSourceCoverage], cutoff: UInt64) {
        let selected = try frames(), cutoff = clock()
        guard selected.count == scope.surfaces.count, selected.count <= 16 else {
            throw AstraError("reset.observationSource", "The reset is missing a source or exceeds its surface limit.")
        }
        for image in selected { _ = try image.metadata.validated() }
        guard Set(selected.map(\.metadata.surface.id)).count == selected.count,
              selected.allSatisfy({ scope.surfaces.contains($0.metadata.surface) }),
              selected.reduce(0, { $0 + $1.metadata.byteCount }) <= 256 * 1024 * 1024 else {
            throw AstraError("reset.observationSource", "The reset is missing a source or its geometry or memory bound changed.")
        }
        var images: [RewardImageFrame] = [], coverage: [RewardSourceCoverage] = []
        for image in selected {
            let frame = try image.metadata.validated()
            if let proof = image.coverage { try proof.validated(frame: frame, cutoffNanos: cutoff, maximumAgeMS: 250) }
            else {
                guard frame.eventNanos <= frame.observedNanos, frame.observedNanos <= cutoff, cutoff - frame.eventNanos <= 250_000_000 else {
                    throw AstraError("reset.observationStale", "The reset has no current screen observation.")
                }
            }
            let proof = image.coverage
            let source = RewardSourceCoverage(sourceObservationID: frame.id, surface: frame.surface,
                eventNanos: frame.eventNanos, observedNanos: frame.observedNanos,
                throughNanos: proof?.kind == .unchanged ? proof!.throughNanos : frame.eventNanos,
                verifiedAtNanos: proof?.verifiedAtNanos ?? frame.observedNanos, kind: proof?.kind == .unchanged ? .unchanged : .frame)
            _ = try source.validated(scope: scope, cutoffNanos: cutoff)
            let pixels = try image.pixels()
            guard pixels.count == frame.byteCount else { throw AstraError("reset.observationBytes", "The reset's owned pixels do not match their source metadata.") }
            images.append(.init(metadata: frame, pixels: pixels)); coverage.append(source)
        }
        return (images, coverage, cutoff)
    }
}
