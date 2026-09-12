import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

private final class RewardResults: @unchecked Sendable {
    let lock = NSLock()
    var values: [RewardAnalysisResult] = []
    var failures: [String] = []
    var results: [RewardAnalysisResult] { lock.withLock { values } }
    func receive(_ result: RewardAnalysisResult) { lock.withLock { values.append(result) } }
    func fail(_ error: AstraError) { lock.withLock { failures.append(error.code) } }
}
private func analysisFrame(time: UInt64, value: UInt8 = 10) -> RewardImageFrame {
    let surface = SurfaceDescriptor(id: "source", globalBounds: .init(x: 0, y: 0, width: 32, height: 32), pixelWidth: 32, pixelHeight: 32)
    return .init(metadata: .init(eventNanos: time, observedNanos: time, surface: surface, byteCount: 4096, codec: "raw"),
                 pixels: Data(repeating: value, count: 4096))
}
private let analysisDetector: RewardAnalysisQueue.Detector = { signals, frames, episode, _ in
    signals.filter { $0.kind.isVisual }.compactMap { signal in
        guard let frame = frames.first(where: { $0.metadata.surface.id == signal.surfaceID }) else { return nil }
        return SignalReading(signalID: signal.id, episodeID: episode, eventNanos: frame.metadata.eventNanos,
            observedNanos: frame.metadata.observedNanos, confidence: 1, value: .number(Double(frame.pixels.first!)),
            sourceObservationID: frame.metadata.id)
    }
}

@Test func rewardAnalysisPreservesDurationAndManualUnknownWithoutCoverage() async throws {
    let episode = UUID(), first = analysisFrame(time: 100_000_000), results = RewardResults()
    let manual = RewardRule(name: "Good", kind: .manualMarker, amount: 1)
    let program = RewardProgram(name: "Feedback", rules: [.init(name: "Time", kind: .ratePerSecond, amount: 2), manual])
    let queue = try await RewardAnalysisQueue.prepare(program: program, scope: .init(surfaces: [first.metadata.surface]),
        episodeID: episode, assetRoot: FileManager.default.temporaryDirectory, warmupFrames: [first],
        detector: analysisDetector, onResult: results.receive, onFault: results.fail)
    try queue.confirmReady(resetID: UUID(), readyNanos: 90_000_000, controlsReleased: true, pendingPackets: 0)
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 100_000_000, frames: [first]))
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 225_000_000, frames: [analysisFrame(time: 225_000_000)]))
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 425_000_000, frames: [analysisFrame(time: 425_000_000)],
        markerCoverage: .init(episodeID: episode, startNanos: 225_000_000, endNanos: 425_000_000, lastSequence: nil)))
    try await queue.finish()
    #expect(results.results.count == 3)
    guard case .interval(_, let missing) = results.results[1], case .interval(_, let covered) = results.results[2] else {
        Issue.record("Expected ordered interval results"); return
    }
    #expect(missing.value == nil)
    #expect(missing.unknownRules == [manual.id])
    #expect(missing.components[program.rules[0].id] == 0.25)
    #expect(covered.value == 0.4)
    #expect(covered.startNanos == 225_000_000)
}

@Test func rewardAnalysisUsesBoundUnchangedPixelsWithoutRetimestamping() async throws {
    let episode = UUID(), first = analysisFrame(time: 100_000_000), results = RewardResults()
    var signal = RewardSignal(name: "Score", kind: .ocrNumber, surfaceID: "source", region: .init(x: 0, y: 0, width: 1, height: 1))
    signal.maximumAgeMS = 100
    let program = RewardProgram(name: "Score", signals: [signal], rules: [.init(name: "Change", kind: .scoreDelta, amount: 1, signalID: signal.id)])
    let queue = try await RewardAnalysisQueue.prepare(program: program, scope: .init(surfaces: [first.metadata.surface]),
        episodeID: episode, assetRoot: FileManager.default.temporaryDirectory, warmupFrames: [first],
        detector: analysisDetector, onResult: results.receive, onFault: results.fail)
    try queue.confirmReady(resetID: UUID(), readyNanos: 100_000_000, controlsReleased: true, pendingPackets: 0)
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 100_000_000, frames: [first]))
    let initial = try CaptureFrameCoverage(streamID: UUID(), frame: first.metadata)
    let unchanged = try initial.verifyingUnchanged(streamID: initial.streamID, surface: first.metadata.surface,
        throughNanos: 390_000_000, verifiedAtNanos: 400_000_000)
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 400_000_000, frames: [first], coverage: [unchanged]))
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 600_000_000, frames: [first], coverage: [unchanged]))
    try await queue.finish()
    guard case .baseline(_, let baseline) = results.results[0], case .interval(_, let covered) = results.results[1],
          case .interval(_, let stale) = results.results[2] else { Issue.record("Missing signal intervals"); return }
    #expect(baseline.readings[0].eventNanos == 100_000_000)
    #expect(covered.value == 0)
    #expect(stale.value == nil)
}

@Test func rewardAnalysisRequiresExplicitPhysicalReadiness() async throws {
    let episode = UUID(), first = analysisFrame(time: 100), results = RewardResults()
    let queue = try await RewardAnalysisQueue.prepare(program: RewardProgram(name: "Time", rules: [.init(name: "Time", kind: .ratePerSecond, amount: 1)]),
        scope: .init(surfaces: [first.metadata.surface]), episodeID: episode, assetRoot: FileManager.default.temporaryDirectory,
        warmupFrames: [first], detector: analysisDetector, onResult: results.receive, onFault: results.fail)
    #expect(throws: AstraError.self) { try queue.confirmReady(resetID: UUID(), readyNanos: 90, controlsReleased: false, pendingPackets: 0) }
    #expect(throws: AstraError.self) { try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 100, frames: [first])) }
    await #expect(throws: AstraError.self) { try await queue.finish() }
    #expect(results.results.isEmpty)
}

@Test func rewardAnalysisDoesNotInitializeScoreFromPreReadyPixels() async throws {
    let episode = UUID(), first = analysisFrame(time: 90_000_000), results = RewardResults()
    let signal = RewardSignal(name: "Score", kind: .ocrNumber, surfaceID: "source", region: .init(x: 0, y: 0, width: 1, height: 1))
    let program = RewardProgram(name: "Score", signals: [signal], rules: [.init(name: "Change", kind: .scoreDelta, amount: 1, signalID: signal.id)])
    let queue = try await RewardAnalysisQueue.prepare(program: program, scope: .init(surfaces: [first.metadata.surface]),
        episodeID: episode, assetRoot: FileManager.default.temporaryDirectory, warmupFrames: [first],
        detector: analysisDetector, onResult: results.receive, onFault: results.fail)
    try queue.confirmReady(resetID: UUID(), readyNanos: 100_000_000, controlsReleased: true, pendingPackets: 0)
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 110_000_000, frames: [first]))
    try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: 150_000_000, frames: [analysisFrame(time: 150_000_000, value: 20)]))
    try await queue.finish()
    guard case .baseline(_, let baseline) = results.results[0], case .interval(_, let result) = results.results[1] else {
        Issue.record("Missing baseline and interval"); return
    }
    guard case .unknown = baseline.values[signal.id] else { Issue.record("Pre-ready score became a baseline"); return }
    #expect(result.value == nil)
}

@Test func rewardAnalysisEndsAtFirstTerminalObservation() async throws {
    let episode = UUID(), first = analysisFrame(time: 100_000_000), results = RewardResults()
    var program = RewardProgram(name: "Timed", rules: [.init(name: "Time", kind: .ratePerSecond, amount: 1)])
    program.maximumEpisodeMS = 100
    let queue = try await RewardAnalysisQueue.prepare(program: program, scope: .init(surfaces: [first.metadata.surface]),
        episodeID: episode, assetRoot: FileManager.default.temporaryDirectory, warmupFrames: [first],
        detector: analysisDetector, onResult: results.receive, onFault: results.fail)
    try queue.confirmReady(resetID: UUID(), readyNanos: 100_000_000, controlsReleased: true, pendingPackets: 0)
    for time: UInt64 in [100_000_000, 200_000_000, 300_000_000] {
        try queue.offer(.init(id: UUID(), episodeID: episode, cutoffNanos: time, frames: [analysisFrame(time: time)]))
    }
    try await queue.finish()
    #expect(results.results.count == 2)
    guard case .interval(_, let terminal) = results.results[1] else { Issue.record("Missing terminal"); return }
    #expect(terminal.outcome == .truncated)
    #expect(terminal.endNanos == 200_000_000)
}
