import Foundation
import Testing
import AstraCore
@testable import AgentTrainerAstra

/// Explicit bundle qualification, separate from quick protocol fixture tests.
/// Invoke via scripts/check_native_learning.py after assembling the runtime.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["ASTRA_VERIFY_BUNDLE"] != nil,
                            "Requires an assembled bundle and explicit qualification request"))
@MainActor struct BundledLearningTests {
    @Test func bundledBehavioralTrainingPublishesAndEvaluatesItsRealCheckpoint() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["ASTRA_VERIFY_BUNDLE"])
        let output = try #require(ProcessInfo.processInfo.environment["ASTRA_NATIVE_VERIFY_ROOT"])
        let root = URL(fileURLWithPath: output, isDirectory: true)
        let bundle = try #require(Bundle(url: URL(fileURLWithPath: path)))
        let lease = try LibraryLease(root: root)
        defer { withExtendedLifetime(lease) {} }
        let store = try LibraryStore(root: root)
        let agent = AgentDocument(name: "Bundled learning verification")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: root, bundle: bundle, changed: {})
        var options = BehaviorOptions()
        options.source = .practice; options.practiceEpisodes = 3; options.epochs = 3
        // Keep production sensory/model dimensions and contiguous defaults.
        // Short pointing episodes exercise common tail-padding removal.
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await wait(coordinator)
        #expect(coordinator.failure == nil)
        let snapshot = try await store.snapshot()
        let run = try #require(snapshot.learningRuns.first)
        #expect(run.status == .completed && run.kind == .behavioral)
        let id = try #require(run.checkpointID)
        let checkpoint = try #require(snapshot.checkpoints.first { $0.id == id })
        #expect(checkpoint.parameterCount == 34_638_639 && checkpoint.trainingStep > 0)
        #expect(snapshot.agents.first?.selectedCheckpointID == id)
        try coordinator.evaluate(checkpoint: checkpoint, agentID: agent.id, split: "validation")
        try await wait(coordinator)
        let evaluation = try #require(coordinator.evaluation)
        #expect(coordinator.failure == nil && evaluation.available && evaluation.decisions > 0)
        #expect(evaluation.meanNLL?.isFinite == true)
        let report: [String: Any] = ["bundle": path, "runID": run.id.uuidString, "checkpointID": id.uuidString,
                                    "parameterCount": checkpoint.parameterCount, "trainingUpdates": checkpoint.trainingStep,
                                    "validationDecisions": evaluation.decisions, "validationMeanNLL": evaluation.meanNLL ?? -1,
                                    "privacyPermissionsUsed": false, "provenance": "practice_oracle",
                                    "completed": run.status == .completed]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("native-learning-report.json"), options: .atomic)
    }

    private func wait(_ coordinator: LearningCoordinator) async throws {
        let deadline = ContinuousClock.now + .seconds(180)
        while coordinator.isBusy {
            if ContinuousClock.now >= deadline {
                await coordinator.stopAndWait()
                throw AstraError("verification.timeout", "Bundled native learning did not finish within its qualification bound.")
            }
            try await Task.sleep(for: .milliseconds(50))
        }
    }
}
