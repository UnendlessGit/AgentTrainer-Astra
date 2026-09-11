import Foundation
import AstraCore

public struct RewardRehearsalReport: Codable, Sendable {
    public let schemaVersion: Int
    public let id: UUID
    public let recordingID: UUID
    public let program: RewardProgram
    public let periodMS: Int
    public let evaluations: [RewardEvaluation]
    public let inspectedDecisions: Int
    public let readinessReached: Bool
}

/// Replays source observations through the live reward evaluator. This does not
/// run a policy, invent human markers, or automatically reset a finished world.
public enum RewardRehearsal {
    public static func save(_ report: RewardRehearsalReport, root: URL) throws -> URL {
        let folder = root.appendingPathComponent("RewardRehearsals", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let properties = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard properties.isDirectory == true, properties.isSymbolicLink != true else {
            throw AstraError("reward.reportFolder", "Reward reports require a regular local directory.")
        }
        let url = folder.appendingPathComponent(report.id.uuidString.lowercased() + ".json")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let bytes = try encoder.encode(report)
        let staging = folder.appendingPathComponent(".rehearsal-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: staging) }
        try bytes.write(to: staging, options: .withoutOverwriting)
        let handle = try FileHandle(forWritingTo: staging)
        defer { try? handle.close() }
        try handle.synchronize()
        try FileManager.default.linkItem(at: staging, to: url)
        return url
    }
    public static func run(program: RewardProgram, directory: URL, assetRoot: URL,
                           startSeconds: Double = 0, durationSeconds: Double = 10, periodMS: Int = 100,
                           cancelled: @Sendable () -> Bool = { false },
                           progress: @Sendable (Int, Int) -> Void = { _, _ in }) throws -> RewardRehearsalReport {
        let program = try program.validated()
        guard startSeconds.isFinite, startSeconds >= 0, durationSeconds.isFinite, durationSeconds > 0,
              durationSeconds <= 300, [50, 100, 200, 500, 1000].contains(periodMS) else {
            throw AstraError("reward.rehearsalRange", "Rehearse up to five minutes at a supported decision interval.")
        }
        let reader = try RecordingReader(directory: directory), inspection = try reader.inspect()
        guard let first = inspection.firstFrameNanos, let last = inspection.lastFrameNanos,
              startSeconds <= Double(last - first) / 1e9 else { throw AstraError("reward.rehearsalFrames", "Choose a range containing complete recorded frames.") }
        let start = first + UInt64(startSeconds * 1e9)
        let available = Double(last - start) / 1e9
        let count = Int(min(durationSeconds, available) * 1000 / Double(periodMS))
        guard count > 0, count <= 6000 else { throw AstraError("reward.rehearsalFrames", "The selected range is shorter than one decision interval.") }
        var templates: [String: Data] = [:]
        for digest in Set(program.signals.compactMap(\.templateDigest)) { templates[digest] = try RewardAssets.read(digest, root: assetRoot) }
        let episodeID = UUID()
        var evaluator = try RewardEvaluator(program: program), ready = false, evaluated: [RewardEvaluation] = []
        var inspected = 0
        for index in 0...count {
            if cancelled() { throw CancellationError() }
            let time = start + UInt64(index * periodMS) * 1_000_000
            guard let preview = try reader.preview(at: time, eventRadiusNanos: 0) else {
                throw AstraError("reward.rehearsalFrame", "The selected reward interval has no available observation.")
            }
            let readings = try VisualRewardDetector.read(signals: program.signals,
                frames: [.init(metadata: preview.frame, pixels: preview.pixels)], episodeID: episodeID, templates: templates)
            if cancelled() { throw CancellationError() }
            inspected += 1
            if !ready {
                if try evaluator.readiness(episodeID: episodeID, cutoffNanos: time, readings: readings) == .yes {
                    // Rehearsal owns no OS controls. This is a virtual boundary,
                    // not proof that a live reset routine is ready or harmless.
                    try evaluator.reset(episodeID: episodeID, readyAtNanos: time, readings: readings, controlsReleased: true)
                    ready = true
                }
            } else {
                let value = try evaluator.evaluate(endNanos: time, readings: readings)
                evaluated.append(value)
                if [.succeeded, .failed, .truncated].contains(value.outcome) { progress(index, count); break }
            }
            progress(index, count)
        }
        return RewardRehearsalReport(schemaVersion: 1, id: UUID(), recordingID: inspection.manifest.id,
            program: program, periodMS: periodMS, evaluations: evaluated, inspectedDecisions: inspected, readinessReached: ready)
    }
}
