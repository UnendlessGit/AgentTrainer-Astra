import Foundation
import Testing
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

/// Run with scripts/qualify_desktop_host.py. This is an entire native host
/// workflow with real MLX child processes, not a mocked learning response.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["ASTRA_HOST_QUALIFICATION_ROOT"] != nil,
                            "Requires explicit real-process desktop qualification"))
@MainActor struct DesktopHostQualificationTests {
    @Test func actualHostCollectsLearnsPublishesResumesAndPreservesAStoppedCursor() async throws {
        let environment = ProcessInfo.processInfo.environment
        let output = URL(fileURLWithPath: try #require(environment["ASTRA_HOST_QUALIFICATION_ROOT"]), isDirectory: true)
        let root = output.appendingPathComponent("Library", isDirectory: true)
        let bundlePath = try #require(environment["ASTRA_HOST_QUALIFICATION_BUNDLE"])
        let bundle = try #require(Bundle(url: URL(fileURLWithPath: bundlePath)))
        let initialInfo = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: output.appendingPathComponent("initial.json")))
        let initialID = try initialInfo.requiredUUID("id")
        let lease = try LibraryLease(root: root); defer { withExtendedLifetime(lease) {} }
        let store = try LibraryStore(root: root), agent = AgentDocument(name: "Generated desktop qualification")
        try await store.save(agent)
        let initial = CheckpointDocument(id: initialID, agentID: agent.id, runID: nil, name: "Real numerical-test policy", kind: "initial",
            trainingStep: 0, policySignature: try initialInfo.required("policySignature").decode(String.self),
            parameterCount: try initialInfo.required("parameterCount").decode(Int.self))
        try await store.saveCheckpoint(initial)
        let multiple = environment["ASTRA_HOST_QUALIFICATION_MULTI_SURFACE"] == "1"
        let liveValues = environment["ASTRA_HOST_QUALIFICATION_LIVE_VALUES"] == "1"
        let queuedControl = environment["ASTRA_HOST_QUALIFICATION_QUEUED_CONTROL"] == "1"
        let capture = HostQualificationCapture(multiSurface: multiple), owner = NativeControlOwner()
        let controls = HostQualificationControls(surfaces: capture.surfaces)
        let processes = HostQualificationProcesses(executable: bundle.bundleURL.appendingPathComponent("Contents/Helpers/AstraCompute.app/Contents/MacOS/AstraCompute"))
        let dependencies = InferenceDependencies(runtime: processes.actor, capture: { _ in capture.runtime }, activate: { _ in },
            countdownSeconds: 0, protectsPhysicalInputs: false, controlOwner: owner)
        let learner = LearningCoordinator(store: store, root: root, bundle: bundle, changed: {})
        let host = DesktopLearningHost(store: store, root: root, learner: learner, bundle: bundle,
            dependencies: dependencies, controlFactory: controls.factory, collectorFactory: processes.collector,
            scopeVerifier: { source, scope in
                guard source.id == capture.source.id, scope.surfaces == capture.surfaces else {
                    throw AstraError("qualification.scope", "The host changed its generated environment.")
                }
                return MonotonicClock.now
            }, detector: { signals, _, _, _ in
                guard signals.allSatisfy({ $0.kind == .elapsedSeconds || (liveValues && $0.kind == .manual) }) else {
                    throw AstraError("qualification.detector", "This fixture never reads a real or visual detector.")
                }
                return []
            }, changed: {})
        let elapsed = RewardSignal(name: "Episode time", kind: .elapsedSeconds)
        var score = RewardSignal(name: "Generated live score", kind: .manual); score.maximumAgeMS = 250
        let telemetry = liveValues ? HostQualificationTelemetry(signalID: score.id) : nil
        var program = RewardProgram(name: "Generated timed episodes", signals: [elapsed],
            rules: [.init(name: "Time reward", kind: .ratePerSecond, amount: 1)])
        if liveValues {
            let available = RewardPredicate(conditions: [.init(signalID: score.id, comparison: .atLeast, number: 0)])
            program.signals.append(score); program.ready = available
            program.rules = [.init(name: "Generated score delta", kind: .scoreDelta, amount: 0.25, signalID: score.id),
                .init(name: "Known live state rate", kind: .ratePerSecond, amount: 2, predicate: available)]
        }
        program.maximumEpisodeMS = 1000
        program.success = .init(conditions: [.init(signalID: elapsed.id, comparison: .atLeast, number: 0.35)])
        var options = DesktopLearningOptions(); options.initialCheckpointID = initial.id; options.iterations = 1
        options.connectLiveSignals = liveValues
        options.training.rolloutDecisions = 6; options.training.epochs = 1
        options.training.sequenceLength = 2; options.training.burnIn = 1; options.training.effectiveBatchDecisions = 4
        var phases: [String] = [], readyConfirmations = 0
        var learned: CheckpointDocument?, stopped: CheckpointDocument?
        var failure: String?
        do {
            try host.start(agent: agent, source: capture.source, program: program, options: options)
            try await wait(host, phases: &phases, confirmations: &readyConfirmations, telemetry: telemetry)
            try #require(host.failure == nil, "Host: \(host.failure ?? "") / learner: \(learner.failure ?? "")")
            let saved = try #require(host.checkpoint); learned = saved
            try #require(saved.id != initial.id && saved.trainingStep > 0, "No real PPO optimizer update was published")
            let snapshot = try await store.snapshot()
            try #require(snapshot.checkpoints.contains { $0.matchesIdentity(of: saved) })
            try #require(snapshot.agents.first?.selectedCheckpointID == saved.id)
            try #require(snapshot.learningRuns.contains { $0.checkpointID == saved.id && $0.status == .completed })
            try #require(readyConfirmations >= 2 && owner.priorCleanupJoined && capture.hasJoined)
            if !multiple && !liveValues && !queuedControl {
                let previousChildren = controls.all.count
                options.initialCheckpointID = saved.id; options.resume = true; options.iterations = 2
                try host.start(agent: agent, source: capture.source, program: program, options: options)
                try await wait(host, phases: &phases, confirmations: &readyConfirmations, shouldStop: {
                    controls.all.dropFirst(previousChildren).contains { !$0.receipts.isEmpty }
                })
                try #require(host.failure == nil, "Resume: \(host.failure ?? "") / learner: \(learner.failure ?? "")")
                let preserved = try #require(host.checkpoint); stopped = preserved
                try #require(preserved.id != saved.id && preserved.trainingStep == saved.trainingStep)
            }
            try #require(owner.priorCleanupJoined && controls.all.allSatisfy { $0.hasJoined && $0.backend.clean })
            try #require(capture.hasJoined && capture.produced > 0)
            let ended = await processes.statuses()
            let expectedWorkers = multiple || liveValues || queuedControl ? 1 : 2
            try #require(ended["actor"]?.count == expectedWorkers && ended["collector"]?.count == expectedWorkers)
            try #require(ended.values.flatMap { $0 }.allSatisfy { $0 == 0 })
            let executed = controls.all.flatMap(\.receipts).filter { $0.status == .executed }
            try #require(!executed.isEmpty && controls.all.reduce(0) { $0 + $1.backend.posted } > 0)
            if queuedControl {
                let feedback = controls.all.flatMap(\.feedback)
                try #require(!feedback.isEmpty && feedback.contains { !$0.packets.isEmpty })
                try #require(Set(feedback.map(\.controlEpochID)).count >= 2)
                try #require(controls.all.flatMap(\.packets).allSatisfy { $0.durationMs == 100 })
            }
            if let telemetry {
                try #require(telemetry.issue == nil, "Generated telemetry: \(telemetry.issue ?? "")")
                try #require(Set(telemetry.receipts.compactMap { $0.fields?["bindingID"]?.uuid }).count >= 2)
                try #require(host.liveSignalEndpoint == nil && host.liveSignalBinding == nil)
            }
        } catch {
            failure = host.failure ?? learner.failure ?? error.localizedDescription
            await host.stopAndWait(); await learner.stopAndWait()
        }
        await telemetry?.stopAndJoin()
        if failure == nil, let issue = telemetry?.issue { failure = issue }
        let statuses = await processes.statuses()
        let sourceFrames = try capture.metadata.map { try JSONSerialization.jsonObject(with: JSONEncoder().encode($0)) }
        let sourceSurfaces = try capture.surfaces.map { try JSONSerialization.jsonObject(with: JSONEncoder().encode($0)) }
        let report: [String: Any] = ["completed": failure == nil, "issue": failure ?? NSNull(),
            "multiSurface": multiple, "liveValues": liveValues, "queuedControl": queuedControl,
            "controlFeedbackSnapshots": try JSONSerialization.jsonObject(with: JSONEncoder().encode(controls.all.flatMap(\.feedback))),
            "submittedPackets": try JSONSerialization.jsonObject(with: JSONEncoder().encode(controls.all.flatMap(\.packets))),
            "sourceFrames": sourceFrames, "sourceSurfaces": sourceSurfaces,
            "liveSignalID": liveValues ? score.id.uuidString.lowercased() : NSNull(),
            "telemetryReceipts": try JSONSerialization.jsonObject(with: JSONEncoder().encode(telemetry?.receipts ?? [])),
            "telemetryJoined": telemetry?.hasJoined ?? true,
            "telemetryIssue": telemetry?.issue ?? NSNull(),
            "privacyPermissionsUsed": false, "physicalInputPosted": false, "personalPixelsRead": false,
            "initialCheckpointID": initial.id.uuidString.lowercased(), "learnedCheckpointID": learned?.id.uuidString.lowercased() ?? NSNull(),
            "stoppedCheckpointID": stopped?.id.uuidString.lowercased() ?? NSNull(), "trainingUpdates": learned?.trainingStep ?? 0,
            "readyConfirmations": readyConfirmations, "captureFrames": capture.produced, "workerExitStatuses": statuses,
            "controlChildren": controls.all.count, "controlReceipts": controls.all.map { $0.receipts.count },
            "virtualInputEffects": controls.all.reduce(0) { $0 + $1.backend.posted }, "phases": phases,
            "controlCleanupConfirmed": owner.priorCleanupJoined, "captureJoined": capture.hasJoined,
            "hostFailure": host.failure ?? NSNull(), "learnerFailure": learner.failure ?? NSNull()]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("desktop-host-report.json"), options: .atomic)
        if let failure { throw AstraError("qualification.host", failure) }
    }

    private func wait(_ host: DesktopLearningHost, phases: inout [String], confirmations: inout Int,
                      telemetry: HostQualificationTelemetry? = nil,
                      shouldStop: () -> Bool = { false }) async throws {
        let deadline = ContinuousClock.now + .seconds(90)
        var confirmed: Set<UUID> = [], requested = false
        while host.isBusy {
            if phases.last != host.phase { phases.append(host.phase) }
            if let endpoint = host.liveSignalEndpoint { telemetry?.startIfNeeded(endpoint) }
            if let issue = telemetry?.issue { throw AstraError("qualification.telemetry", issue) }
            if let reset = host.awaitingReady, confirmed.insert(reset).inserted {
                confirmations += 1; host.confirmReady()
            }
            if !requested && shouldStop() { requested = true; host.requestStop() }
            guard ContinuousClock.now < deadline else {
                await host.stopAndWait()
                throw AstraError("qualification.timeout", "The real desktop host did not join within 90 seconds.")
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        phases.append(host.phase)
    }
}
