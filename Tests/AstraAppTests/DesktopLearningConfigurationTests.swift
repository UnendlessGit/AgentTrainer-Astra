import Foundation
import Testing
import AstraCore
import AstraPlatform
@testable import AgentTrainerAstra

@Suite struct DesktopLearningConfigurationTests {
    @Test func resumeRetainsSavedLearningAndContextContractAcrossWindowBinding() throws {
        let agent = UUID(), run = UUID(), checkpointID = UUID()
        let old = SurfaceDescriptor(id: "old-window", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
        let fresh = SurfaceDescriptor(id: "new-window", globalBounds: .init(x: 20, y: 20, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
        let scope = ControlScope(surfaces: [fresh])
        let signal = RewardSignal(name: "Ready", kind: .ocrText, surfaceID: old.id, region: .init(x: 0, y: 0, width: 1, height: 1))
        let program = RewardProgram(name: "Task", signals: [signal])
        let binding = try RewardProgramBinding.singleSource(program, surface: fresh, scope: scope)
        let actions = ActionCapabilities(keyCodes: [13])
        let model: JSONValue = .object(["period_ms": .integer(100), "lead_ms": .integer(100), "packet_capacity": .integer(16), "context_sizes": .array([.integer(3)])])
        let document = CheckpointDocument(id: checkpointID, agentID: agent, runID: run, name: "Saved", kind: "reinforcement",
            trainingStep: 1, policySignature: String(repeating: "a", count: 64), parameterCount: 10)
        var options = DesktopLearningOptions(); options.contextIDs = [2]
        var manifest: [String: JSONValue] = ["model": model, "actions": try actions.policyVocabulary(scrollUnitsPerPoint: 4)]
        var prepared = PreparedDesktopPolicy(checkpoint: .init(document: document, directory: URL(fileURLWithPath: "/tmp/model")), manifest: .object(manifest))
        let first = try DesktopLearningConfiguration(prepared: prepared, reward: binding, scope: scope, options: options)
        manifest["trainingConfig"] = first.training
        manifest["metrics"] = .object(["sourceKind": .string("external_rollout"), "requiresEnvironmentReset": .bool(true)])
        prepared = .init(checkpoint: prepared.checkpoint, manifest: .object(manifest))
        let saved: JSONValue = .object(["operation": .string("train.reinforcement.external"), "runID": .string(run.uuidString),
            "agentID": .string(agent.uuidString), "destinationCheckpointID": .string(checkpointID.uuidString),
            "environment": first.environment, "model": model, "contextIDs": .array([.integer(2)])])
        options.resume = true; options.initialCheckpointID = checkpointID; options.contextIDs = [99]
        options.training.learningRate = .nan; options.training.rolloutDecisions = -1
        let resumed = try DesktopLearningConfiguration(prepared: prepared, reward: binding, scope: scope, options: options, resumedConfiguration: saved)
        #expect(resumed.environment == first.environment && resumed.training == first.training && resumed.contextIDs == [2])
        #expect(resumed.environment.fields?["action_vocabulary"]?.fields?["scrollUnitsPerPoint"]?.int == 4)
        #expect(resumed.program.signals[0].surfaceID == fresh.id && resumed.reward.definition.signals[0].surfaceID == old.id)
        var changed = program; changed.maximumEpisodeMS = 1000
        let changedBinding = try RewardProgramBinding.singleSource(changed, surface: fresh, scope: scope)
        #expect(throws: AstraError.self) {
            try DesktopLearningConfiguration(prepared: prepared, reward: changedBinding, scope: scope, options: options, resumedConfiguration: saved)
        }
    }

    @Test func resetObservationCacheRetainsRealSourceTimeButRebindsEpisodeReadings() async throws {
        let fixture = ResetObservationFixture()
        let source = try await DesktopResetObservations.prepare(program: fixture.program, scope: fixture.scope,
            assetRoot: URL(fileURLWithPath: "/tmp/unused-assets"), frames: { [fixture.image] }, detector: { fixture.detect($0, $1, $2, $3) }, clock: { 110 })
        let first = try ResetContext(nextEpisodeID: UUID(), environmentID: UUID(), scope: fixture.scope)
        let one = try await source.observe(context: first), repeated = try await source.observe(context: first)
        #expect(one.readings.first?.episodeID == first.nextEpisodeID && repeated.sourceCoverage.first?.eventNanos == 100)
        #expect(one.observedNanos == 110 && one.sourceCoverage.first?.throughNanos == 100)
        #expect(fixture.calls == 2) // One warmup, one real read; repeated frame uses the cache.
        let next = try ResetContext(nextEpisodeID: UUID(), environmentID: first.environmentID, scope: fixture.scope)
        let two = try await source.observe(context: next)
        #expect(two.readings.first?.episodeID == next.nextEpisodeID && fixture.calls == 3)
    }

    @Test func resetRejectsStaleEvidenceBeforeCopyOrDetectorWork() async throws {
        let fixture = ResetObservationFixture()
        await #expect(throws: AstraError.self) {
            try await DesktopResetObservations.prepare(program: fixture.program, scope: fixture.scope,
                assetRoot: URL(fileURLWithPath: "/tmp/unused-assets"), frames: { [fixture.image] },
                detector: { fixture.detect($0, $1, $2, $3) }, clock: { 500_000_000 })
        }
        #expect(fixture.calls == 0 && fixture.copies == 0)
    }
}

private final class ResetObservationFixture: @unchecked Sendable {
    private let lock = NSLock()
    private var reads = 0, pixelReads = 0
    var calls: Int { lock.withLock { reads } }
    var copies: Int { lock.withLock { pixelReads } }
    let scope: ControlScope
    let program: RewardProgram
    let metadata: FrameMetadata
    init() {
        let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
        scope = .init(surfaces: [surface])
        metadata = .init(eventNanos: 100, observedNanos: 105, surface: surface, byteCount: 16, codec: "raw")
        program = .init(name: "Fixture", signals: [.init(name: "Text", kind: .ocrText, surfaceID: surface.id, region: .init(x: 0, y: 0, width: 1, height: 1))])
    }
    var image: InferenceImage { .init(metadata: metadata, pixels: { [self] in lock.withLock { pixelReads += 1 }; return Data(repeating: 0, count: 16) }) }
    func detect(_ signals: [RewardSignal], _ images: [RewardImageFrame], _ episode: UUID, _ templates: [String: Data]) -> [SignalReading] {
        lock.withLock { reads += 1 }
        return [.init(signalID: signals[0].id, episodeID: episode, eventNanos: metadata.eventNanos, observedNanos: metadata.observedNanos,
            value: .text("ready"), sourceObservationID: metadata.id)]
    }
}
