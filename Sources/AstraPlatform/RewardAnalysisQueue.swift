import Foundation
import AstraCore

public struct RewardAnalysisObservation: Sendable {
    public let id: UUID
    public let episodeID: UUID
    public let cutoffNanos: UInt64
    public let frames: [RewardImageFrame]
    public let coverage: [CaptureFrameCoverage]
    public let manualReadings: [SignalReading]
    public let markers: [ManualRewardMarker]
    public let markerCoverage: ManualRewardCoverage?
    public init(id: UUID, episodeID: UUID, cutoffNanos: UInt64, frames: [RewardImageFrame], coverage: [CaptureFrameCoverage] = [],
                manualReadings: [SignalReading] = [], markers: [ManualRewardMarker] = [], markerCoverage: ManualRewardCoverage? = nil) {
        self.id = id; self.episodeID = episodeID; self.cutoffNanos = cutoffNanos; self.frames = frames
        self.coverage = coverage; self.manualReadings = manualReadings; self.markers = markers; self.markerCoverage = markerCoverage
    }
}

public enum RewardAnalysisResult: Sendable {
    /// The first actual actor cutoff initializes the reward interval. Readiness
    /// is rechecked against these actual pixels; no pre-actor interval is scored.
    case baseline(observationID: UUID, snapshot: ResolvedRewardSnapshot)
    case interval(observationID: UUID, evaluation: RewardEvaluation)
}

/// One physical episode's bounded, ordered CPU analysis owner. Actor decisions
/// never await Vision. Output offers must also be bounded and nonblocking. This
/// type neither samples/posts actions nor invents manual-feedback coverage.
public final class RewardAnalysisQueue: @unchecked Sendable {
    public typealias Detector = @Sendable ([RewardSignal], [RewardImageFrame], UUID, [String: Data]) throws -> [SignalReading]
    public typealias ResultHandler = @Sendable (RewardAnalysisResult) throws -> Void
    private let queue = DispatchQueue(label: "astra.reward.analysis", qos: .userInitiated)
    private let lock = NSLock()
    private let program: RewardProgram, scope: ControlScope
    private let episodeID: UUID
    private let templates: [String: Data]
    private let detector: Detector
    private let onResult: ResultHandler
    private let onFault: @Sendable (AstraError) -> Void
    private let maximumBytes: Int, maximumItems: Int
    private var reservedBytes = 0, reservedItems = 0
    private var closed = false
    private var storedFault: AstraError?
    private var readiness: (resetID: UUID, readyNanos: UInt64)?
    private let completion = AsyncCompletion()
    // The analysis queue alone owns the following episode state.
    private var evaluator: RewardEvaluator
    private var lastCutoff: UInt64?
    private var terminal = false
    private var detectedFrames: [FrameMetadata] = []
    private var detectedReadings: [SignalReading] = []

    private init(program: RewardProgram, scope: ControlScope, episodeID: UUID, templates: [String: Data],
                 maximumBytes: Int, maximumItems: Int, detector: @escaping Detector,
                 onResult: @escaping ResultHandler, onFault: @escaping @Sendable (AstraError) -> Void) throws {
        self.program = try program.validated(); self.scope = try scope.validated(); self.episodeID = episodeID
        self.templates = templates; self.maximumBytes = maximumBytes; self.maximumItems = maximumItems
        self.detector = detector; self.onResult = onResult; self.onFault = onFault
        evaluator = try RewardEvaluator(program: program)
    }

    /// Complete detector/template loading before physical control is armed.
    /// Warmup results do not initialize an episode or fabricate fresh readings.
    public static func prepare(program: RewardProgram, scope: ControlScope, episodeID: UUID, assetRoot: URL,
                               warmupFrames: [RewardImageFrame], maximumQueuedBytes: Int = 256 * 1024 * 1024,
                               maximumQueuedItems: Int = 128,
                               detector: @escaping Detector = { try VisualRewardDetector.read(signals: $0, frames: $1, episodeID: $2, templates: $3) },
                               onResult: @escaping ResultHandler, onFault: @escaping @Sendable (AstraError) -> Void) async throws -> RewardAnalysisQueue {
        guard (1024 * 1024 + 4...1024 * 1024 * 1024).contains(maximumQueuedBytes), (1...128).contains(maximumQueuedItems),
              warmupFrames.count <= 16, warmupFrames.reduce(0, { $0 + $1.pixels.count }) <= maximumQueuedBytes else {
            throw AstraError("reward.analysisLimits", "Reward analysis requires bounded source pixels and queue limits.")
        }
        let validated = try program.validated(), validatedScope = try scope.validated()
        let surfaces = Set(validatedScope.surfaces.map(\.id))
        let warmSurfaces = Set(warmupFrames.map { $0.metadata.surface.id })
        guard validated.signals.compactMap(\.surfaceID).allSatisfy(surfaces.contains),
              validated.signals.compactMap(\.surfaceID).allSatisfy(warmSurfaces.contains),
              warmSurfaces.count == warmupFrames.count,
              warmupFrames.allSatisfy({ validatedScope.surfaces.contains($0.metadata.surface) }) else {
            throw AstraError("reward.sourceBinding", "Bind every visual reward signal to the selected environment before starting.")
        }
        let prepared = try await Task.detached {
            var templates: [String: Data] = [:]
            for digest in Set(validated.signals.compactMap(\.templateDigest)) { templates[digest] = try RewardAssets.read(digest, root: assetRoot) }
            _ = try detector(validated.signals, warmupFrames, episodeID, templates)
            return try Self(program: validated, scope: validatedScope, episodeID: episodeID, templates: templates,
                maximumBytes: maximumQueuedBytes, maximumItems: maximumQueuedItems, detector: detector, onResult: onResult, onFault: onFault)
        }.value
        try Task.checkCancellation()
        return prepared
    }

    /// The reset/control owner supplies this only after joining prior actions
    /// and confirming physical readiness. It is not inferred from an empty
    /// event list, detector warmup or the collector's process acknowledgement.
    public func confirmReady(resetID: UUID, readyNanos: UInt64, controlsReleased: Bool, pendingPackets: Int) throws {
        try lock.withLock {
            guard !closed, storedFault == nil, readiness == nil, reservedItems == 0,
                  controlsReleased, pendingPackets == 0 else {
                throw AstraError("reward.readinessProof", "Reward analysis requires one confirmed reset with released controls and no pending packets.")
            }
            readiness = (resetID, readyNanos)
        }
    }

    public func offer(_ observation: RewardAnalysisObservation) throws {
        do {
            guard observation.episodeID == episodeID, observation.frames.count <= 16, observation.coverage.count <= 16,
                  observation.manualReadings.count <= 32, observation.markers.count <= 4096 else {
                throw AstraError("reward.analysisObservation", "Reward evidence has an invalid episode or exceeds its bounded schema.")
            }
            // Inputs already own immutable Data. The queue retains them only
            // after reserving their complete pixel + bounded metadata budget.
            let bytes = observation.frames.reduce(0) { $0 + $1.pixels.count } + 1024 * 1024
            try lock.withLock {
                guard let readiness, observation.cutoffNanos >= readiness.readyNanos else {
                    throw AstraError("reward.readinessProof", "Confirm the physical episode boundary before submitting actor observations.")
                }
                guard !closed, storedFault == nil, reservedItems < maximumItems, bytes <= maximumBytes - reservedBytes else {
                    throw AstraError("reward.analysisBackpressure", "Reward analysis cannot retain more observations within its queue limits.")
                }
                reservedItems += 1; reservedBytes += bytes
                // Enqueue under the admission lock so finish cannot overtake an
                // accepted observation between reservation and queue submission.
                queue.async { [self] in
                    defer { lock.withLock { reservedItems -= 1; reservedBytes -= bytes } }
                    guard lock.withLock({ storedFault == nil }), !terminal else { return }
                    do { try analyze(observation) }
                    catch { fail((error as? AstraError) ?? AstraError("reward.analysis", error.localizedDescription)) }
                }
            }
        } catch {
            let failure = (error as? AstraError) ?? AstraError("reward.analysis", error.localizedDescription)
            fail(failure); throw failure
        }
    }

    public func finish() async throws {
        lock.withLock {
            if !closed { closed = true; queue.async { [self] in completion.finish() } }
        }
        await completion.wait()
        if let error = lock.withLock({ storedFault }) { throw error }
    }

    private func fail(_ error: AstraError) {
        let notify = lock.withLock { () -> Bool in
            if storedFault != nil { return false }
            storedFault = error; return true
        }
        if notify { onFault(error) }
    }

    private func analyze(_ observation: RewardAnalysisObservation) throws {
        guard lastCutoff.map({ observation.cutoffNanos > $0 }) ?? true,
              Set(observation.frames.map { $0.metadata.surface.id }).count == observation.frames.count,
              Set(observation.coverage.map(\.frameID)).count == observation.coverage.count else {
            throw AstraError("reward.analysisOrder", "Reward observations must be unique, causal snapshots in decision order.")
        }
        let byID = Dictionary(uniqueKeysWithValues: observation.coverage.map { ($0.frameID, $0) })
        guard Set(byID.keys).isSubset(of: Set(observation.frames.map { $0.metadata.id })) else {
            throw AstraError("reward.analysisCoverage", "Reward coverage references pixels outside this observation.")
        }
        var sources: [String: RewardSourceCoverage] = [:]
        for frame in observation.frames {
            _ = try frame.metadata.validated()
            guard scope.surfaces.contains(frame.metadata.surface), frame.pixels.count == frame.metadata.byteCount else {
                throw AstraError("reward.analysisSource", "Reward pixels changed source geometry or byte size within the episode.")
            }
            let evidence = byID[frame.metadata.id]
            if let evidence { try evidence.validated(frame: frame.metadata, cutoffNanos: observation.cutoffNanos) }
            // Reward source age starts at the actual display event. Collector
            // transport's frame-availability clock has a different meaning.
            let source = RewardSourceCoverage(sourceObservationID: frame.metadata.id, surface: frame.metadata.surface,
                eventNanos: frame.metadata.eventNanos, observedNanos: frame.metadata.observedNanos,
                throughNanos: evidence?.kind == .unchanged ? evidence!.throughNanos : frame.metadata.eventNanos,
                verifiedAtNanos: evidence?.verifiedAtNanos ?? frame.metadata.observedNanos,
                kind: evidence?.kind == .unchanged ? .unchanged : .frame)
            _ = try source.validated(scope: scope, cutoffNanos: observation.cutoffNanos)
            sources[frame.metadata.surface.id] = source
        }
        let manualIDs = Set(program.signals.filter { $0.kind == .manual }.map(\.id))
        guard observation.manualReadings.allSatisfy({ manualIDs.contains($0.signalID) }) else {
            throw AstraError("reward.manualSource", "Manual observations cannot impersonate a visual or elapsed-time signal.")
        }
        let metadata = observation.frames.map(\.metadata)
        if metadata != detectedFrames {
            detectedReadings = try detector(program.signals, observation.frames, episodeID, templates)
            detectedFrames = metadata
        }
        let readings = detectedReadings + observation.manualReadings
        let readIDs = Set(readings.map(\.signalID))
        let coverage = program.signals.compactMap { signal -> RewardReadingCoverage? in
            guard readIDs.contains(signal.id), let surface = signal.surfaceID, let source = sources[surface] else { return nil }
            return RewardReadingCoverage(signalID: signal.id, source: source)
        }
        if lastCutoff == nil {
            guard observation.markers.isEmpty, observation.markerCoverage == nil else {
                throw AstraError("reward.preActorFeedback", "Feedback before the first actor cutoff cannot be assigned to a policy decision.")
            }
            guard let readyNanos = lock.withLock({ readiness?.readyNanos }) else {
                throw AstraError("reward.readinessProof", "The episode readiness proof is unavailable.")
            }
            let snapshot = try evaluator.resolveSnapshot(episodeID: episodeID, cutoffNanos: observation.cutoffNanos,
                readings: readings, coverage: coverage, scope: scope, minimumEvidenceNanos: readyNanos)
            // Readiness/control release must already have been confirmed by the
            // coordinator before it starts this episode's actor observations.
            try evaluator.reset(snapshot: snapshot, controlsReleased: true)
            try onResult(.baseline(observationID: observation.id, snapshot: snapshot))
        } else {
            let result = try evaluator.evaluate(endNanos: observation.cutoffNanos, readings: readings,
                markers: observation.markers, markerCoverage: observation.markerCoverage, coverage: coverage, scope: scope)
            try onResult(.interval(observationID: observation.id, evaluation: result))
            terminal = [.succeeded, .failed, .truncated].contains(result.outcome)
        }
        lastCutoff = observation.cutoffNanos
    }
}
