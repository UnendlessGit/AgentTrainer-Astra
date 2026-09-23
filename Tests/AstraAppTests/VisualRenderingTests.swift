import AppKit
import SwiftUI
import Testing
import AstraCore
@testable import AgentTrainerAstra

/// Opt-in visual evidence, deliberately excluded from ordinary test runs. Every
/// image is drawn from this process's own NSHostingView, never from a screen,
/// another application, or an accessibility snapshot.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_UI_RENDER_DIR"] != nil))
@MainActor func renderActualApplicationViews() async throws {
    let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_UI_RENDER_DIR"]), isDirectory: true)
    let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_WORKSPACE_ROOT"]), isDirectory: true)
    #expect(root.deletingLastPathComponent().standardizedFileURL == output.standardizedFileURL)
    NSApplication.shared.setActivationPolicy(.prohibited)
    let store = try LibraryStore(root: root)
    let fixtureDate = Date(timeIntervalSince1970: 1_800_000_000)
    var agent = AgentDocument(name: "Desktop skills · precise pointing and delayed visual memory", createdAt: fixtureDate)
    let environment = EnvironmentDocument(name: "Practice environment · generated visual targets", kind: .practice)
    agent.environmentID = environment.id
    try await store.save(environment)
    try await store.save(agent)
    var checkpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: nil,
        name: "Pointing · after 24 epochs · independently evaluated", kind: "behavioral", trainingStep: 2400,
        policySignature: String(repeating: "a", count: 64), parameterCount: 34_640_000)
    checkpoint.createdAt = fixtureDate
    try await store.saveCheckpoint(checkpoint)
    agent.selectedCheckpointID = checkpoint.id; try await store.save(agent)
    var pausedRun = LearningRunDocument(agentID: agent.id, kind: .behavioral, name: "Paused experiment · saved progress", sourceKind: "practice_oracle")
    var pausedCheckpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: pausedRun.id,
        name: "Paused · epoch 8 of 24", kind: "behavioral", trainingStep: 800,
        policySignature: String(repeating: "b", count: 64), parameterCount: 34_640_000)
    pausedCheckpoint.createdAt = fixtureDate
    pausedRun.status = .cancelled; pausedRun.epoch = 8; pausedRun.updates = 800; pausedRun.decisions = 40_960
    pausedRun.checkpointID = pausedCheckpoint.id; pausedRun.createdAt = fixtureDate; pausedRun.modifiedAt = fixtureDate
    try await store.saveCheckpoint(pausedCheckpoint); try await store.saveLearningRun(pausedRun)
    let checkpointDirectory = root.appendingPathComponent("Models/" + checkpoint.id.uuidString.lowercased())
    try FileManager.default.createDirectory(at: checkpointDirectory, withIntermediateDirectories: true)
    try Data(#"{"model":{"context_sizes":[]}}"#.utf8).write(to: checkpointDirectory.appendingPathComponent("manifest.json"))
    var run = LearningRunDocument(agentID: agent.id, kind: .behavioral,
        name: "Demonstrations · multiple layouts and carefully timed controls", sourceKind: "practice_oracle")
    run.status = .completed; run.epoch = 24; run.updates = 2400; run.decisions = 132_840; run.meanNLL = 0.382
    run.checkpointID = checkpoint.id; run.createdAt = fixtureDate; run.modifiedAt = fixtureDate
    try await store.saveLearningRun(run)
    let recording = try makeRenderRecording(root: root, environment: environment)
    try await store.saveRecording(recording)
    let selectionAgent = AgentDocument(name: "Selected demonstrations · reusable source")
    try await store.save(selectionAgent)
    try await store.linkRecordings([recording.id], to: selectionAgent.id)
    let ranges = RecordingTrainingSelection(ranges: [
        .init(startNanos: 1_000_010_000, endNanos: 1_100_000_000),
        .init(startNanos: 1_200_000_000, endNanos: 1_350_000_000)])
    try await store.saveRecordingSelection(ranges, recordingID: recording.id, agentID: selectionAgent.id)
    let score = RewardSignal(name: "Visible score", kind: .ocrNumber, surfaceID: "generated-render-fixture", region: .init(x: 0.1, y: 0.1, width: 0.3, height: 0.2))
    var rewards = RewardProgram(name: "Desktop feedback · recorded visual signals", signals: [score], rules: [
        .init(name: "Score change", kind: .scoreDelta, amount: 0.1, signalID: score.id),
        .init(name: "Positive feedback", kind: .manualMarker, amount: 1)])
    rewards.success = .init(conditions: [.init(signalID: score.id, comparison: .atLeast, number: 100)])
    try await store.saveRewardProgram(rewards, for: agent.id); agent.rewardProgramID = rewards.id
    var resetAgent = AgentDocument(name: "Authored environment reset")
    var resetRewards = rewards; resetRewards.id = UUID(); resetRewards.name = "Return to a verified starting state"
    resetRewards.ready = .init(conditions: [.init(signalID: score.id, comparison: .atMost, number: 0)])
    resetRewards.resetPlan = .init(steps: [
        .init(name: "Confirm the restart dialog", packet: ResetPacketPresets.keyPress(key: 36, modifiers: [55], holdMS: 80)),
        .init(name: "Click the restart control", packet: ResetPacketPresets.click(surfaceID: "generated-render-fixture")),
        .init(name: "Allow the environment to settle", pauseMS: 500),
        .init(name: "Wait for score zero", condition: resetRewards.ready!, timeoutMS: 10_000)])
    try await store.save(resetAgent); try await store.saveRewardProgram(resetRewards, for: resetAgent.id)
    resetAgent.rewardProgramID = resetRewards.id
    let interruptedRoot = root.appendingPathComponent("InspectorOnly")
    let interrupted = try makeRenderRecording(root: interruptedRoot, environment: environment,
        issue: "The selected environment changed size while this demonstration was recording. Captured frames and input events were preserved. Review the final interval before including this recording in a training dataset.")
    var evaluationAgent = AgentDocument(name: "Checkpoint comparison")
    let evaluationRun = LearningRunDocument(agentID: evaluationAgent.id, kind: .behavioral, name: "Saved shared demonstrations", sourceKind: "practice_oracle")
    let evaluationFirst = CheckpointDocument(id: UUID(), agentID: evaluationAgent.id, runID: evaluationRun.id,
        name: "Pointing · after 12 epochs", kind: "behavioral", trainingStep: 120, policySignature: String(repeating: "a", count: 64), parameterCount: 34_638_639)
    let evaluationSecond = CheckpointDocument(id: UUID(), agentID: evaluationAgent.id, runID: nil,
        name: "Pointing · refined with reinforcement learning", kind: "reinforcement", trainingStep: 240, policySignature: String(repeating: "a", count: 64), parameterCount: 34_638_639)
    evaluationAgent.selectedCheckpointID = evaluationFirst.id
    try await store.save(evaluationAgent); try await store.saveLearningRun(evaluationRun)
    try await store.saveCheckpoint(evaluationFirst); try await store.saveCheckpoint(evaluationSecond)
    let evaluationProtocol = try EvaluationProtocol(sourceRunID: evaluationRun.id, sourceCheckpointID: evaluationFirst.id,
        sourceName: evaluationFirst.name, dataset: .object(["kind": .string("practice_oracle")]),
        identity: .object(["fixture": .bool(true)]), expectedDatasetID: nil, provenance: "practice_oracle",
        policySignature: evaluationFirst.policySignature, split: "validation", verificationMode: true)
    let evaluationDatasetID = UUID()
    var evaluationRows: [EvaluationDocument] = []
    for (index, checkpoint) in [evaluationFirst, evaluationSecond].enumerated() {
        var row = EvaluationDocument(agentID: evaluationAgent.id, checkpoint: checkpoint, protocolDefinition: evaluationProtocol, createdAt: fixtureDate)
        row.status = .completed; row.datasetID = evaluationDatasetID; row.finishedAt = fixtureDate + 10
        row.decisions = 1024; row.meanNLL = index == 0 ? 0.7842 : 0.4913
        try await store.saveEvaluation(row); evaluationRows.append(row)
    }
    let model = WorkspaceModel()
    await model.start()
    #expect(!model.loading && model.errorMessage == nil)
    var practice = BehaviorOptions(); practice.source = .practice
    var freshDesktopAgent = agent; freshDesktopAgent.selectedCheckpointID = nil
    var resume = BehaviorOptions(); resume.initialCheckpointID = pausedCheckpoint.id; resume.resume = true
    let metrics = (1...12).map { EpochMetric(epoch: $0, nll: 2.8 / Double($0) + 0.15, decisions: $0 * 10_240) }
    var reinforcement: [ReinforcementMetric] = []
    for index in 1...12 {
        let count = Double(index)
        let fields: [String: JSONValue] = ["iteration": .integer(Int64(index)), "elapsed_seconds": .number(count * 36.5),
            "mean_reward": .number(-0.008 + count * 0.001), "mean_value_loss": .number(0.12 / count),
            "mean_policy_loss": .number(-0.001), "maximum_sampled_kl": .number(0.002 + count * 0.0001),
            "maximum_accepted_kl": .number(0.002 + count * 0.0001), "maximum_candidate_kl": .number(0.032),
            "backtrack_count": .integer(2), "rejected_optimizer_steps": .integer(0), "minimum_step_scale": .number(0.25),
            "clip_fraction": .number(0.05), "optimizer_updates": .integer(Int64(index * 16))]
        reinforcement.append(try #require(ReinforcementMetric(fields)))
    }
    var evidence: [[String: Any]] = []
    let requestedNames = Set((ProcessInfo.processInfo.environment["ASTRA_UI_RENDER_NAMES"] ?? "").split(separator: ",").map(String.init))
    func includes(_ name: String) -> Bool { requestedNames.isEmpty || requestedNames.contains(name) }
    for size in [NSSize(width: 1120, height: 760), NSSize(width: 860, height: 580)] {
        for scheme in [ColorScheme.light, .dark] {
            let variants: [(String, AnyView)] = [
                ("evaluation-comparison", AnyView(EvaluationWorkspaceView(agent: evaluationAgent, model: model, comparing: true,
                    sourceID: evaluationFirst.id, candidates: [evaluationFirst.id, evaluationSecond.id],
                    selectedResults: Set(evaluationRows.map(\.id))).padding(28))),
                ("behavior-empty", AnyView(BehaviorTrainingView(agent: agent, model: model).padding(28))),
                ("behavior-practice", AnyView(BehaviorTrainingView(agent: agent, model: model, options: practice).padding(28))),
                ("behavior-resume", AnyView(BehaviorTrainingView(agent: agent, model: model, options: resume).padding(28))),
                ("reinforcement", AnyView(ReinforcementTrainingView(agent: agent, model: model).padding(28))),
                ("reinforcement-practice", AnyView(PracticeReinforcementTrainingView(agent: agent, model: model).padding(28))),
                ("desktop-learning-advanced", AnyView(DesktopTrainingView(agent: freshDesktopAgent, model: model, showAdvanced: true).padding(28))),
                ("run-empty", AnyView(RunView(coordinator: nil, checkpoints: [], sources: [], refreshingSources: false,
                    sourceIssue: nil, unavailableReason: nil, refreshSources: {}, selectCheckpoint: { _ in }, start: { _, _, _ in }).padding(28))),
                ("run-checkpoint", AnyView(RunView(coordinator: nil, checkpoints: [checkpoint], sources: [], refreshingSources: false,
                    sourceIssue: "Choose an environment after enabling Screen Recording for Astra.", unavailableReason: nil,
                    selectedCheckpointID: checkpoint.id, refreshSources: {}, selectCheckpoint: { _ in }, start: { _, _, _ in }).padding(28))),
                ("behavior-metrics", AnyView(LearningProgressContent(metrics: metrics, phase: "Training epoch 12", updates: 2400,
                    decisionsPerSecond: 128.6, peakMemoryBytes: 3_200_291_796).padding(28))),
                ("reinforcement-metrics", AnyView(ReinforcementProgressContent(metrics: reinforcement, phase: "Finishing the current episode",
                    rolloutTarget: 512, rolloutDecisions: 576, updates: 192, elapsedSeconds: 438, peakMemoryBytes: 4_182_662_144).padding(28))),
                ("recording-inspector", AnyView(RecordingInspector(recording: recording, directory: root.appendingPathComponent("Recordings/" + recording.id.uuidString + ".astrarecord")))),
                ("recording-interrupted", AnyView(RecordingInspector(recording: interrupted, directory: interruptedRoot.appendingPathComponent("Recordings/" + interrupted.id.uuidString + ".astrarecord")))),
                ("recording-selections", AnyView(RecordingInspector(recording: recording, directory: root.appendingPathComponent("Recordings/" + recording.id.uuidString + ".astrarecord"), workspace: model, agentID: selectionAgent.id))),
                ("recording-link", AnyView(RecordingLinkSheet(model: model, request: .init(agentID: agent.id)))),
                ("reward-signals", AnyView(RewardEditor(agent: agent, model: model, referenceRecordingID: recording.id))),
                ("reward-rules", AnyView(RewardEditor(agent: agent, model: model, initialPage: "Rewards"))),
                ("reward-episode", AnyView(RewardEditor(agent: agent, model: model, initialPage: "Episode"))),
                ("reset-episode", AnyView(RewardEditor(agent: resetAgent, model: model, initialPage: "Episode", referenceRecordingID: recording.id))),
                ("reset-click", AnyView(ScrollView {
                    ResetPlanEditor(plan: .constant(.init(steps: [.init(name: "Restart control", packet: ResetPacketPresets.click(surfaceID: "generated-render-fixture"))])),
                        signals: resetRewards.signals, referenceSurface: SurfaceDescriptor(id: "generated-render-fixture",
                            globalBounds: .init(x: 0, y: 0, width: 640, height: 360), pixelWidth: 640, pixelHeight: 360)).padding(24)
                })),
                ("reward-rehearse", AnyView(RewardEditor(agent: agent, model: model, initialPage: "Rehearse", referenceRecordingID: recording.id))),
            ]
            for (name, view) in variants where includes(name) {
                evidence.append(try await renderOwnedView(view, name: name, size: size, scheme: scheme, output: output))
            }
            model.destination = .library
            if includes("workspace-library") { evidence.append(try await renderOwnedView(AnyView(WorkspaceView(model: model)), name: "workspace-library", size: size, scheme: scheme, output: output)) }
            model.destination = .agent(agent.id); model.section = .training
            if includes("workspace-training") { evidence.append(try await renderOwnedView(AnyView(WorkspaceView(model: model)), name: "workspace-training", size: size, scheme: scheme, output: output)) }
            model.destination = .agent(selectionAgent.id)
            model.section = .demonstrations
            if includes("workspace-demonstrations") { evidence.append(try await renderOwnedView(AnyView(WorkspaceView(model: model)), name: "workspace-demonstrations", size: size, scheme: scheme, output: output)) }
            model.destination = .agent(agent.id)
            model.section = .run
            if includes("workspace-run") { evidence.append(try await renderOwnedView(AnyView(WorkspaceView(model: model)), name: "workspace-run", size: size, scheme: scheme, output: output)) }
        }
    }
    await model.prepareForTermination()
    let manifest: [String: Any] = ["fixtureDataOnly": true, "screenCapture": false, "accessibilityReads": false,
                                   "renderer": "AppKit NSHostingView cacheDisplay", "images": evidence]
    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("manifest.json"), options: .atomic)
}

private func makeRenderRecording(root: URL, environment: EnvironmentDocument, issue: String? = nil) throws -> RecordingManifest {
    let manifest = RecordingManifest(name: "Generated fixture · capture and precisely timed input", environment: environment)
    let directory = root.appendingPathComponent("Recordings/" + manifest.id.uuidString + ".astrarecord")
    let writer = try RecordingWriter(directory: directory, manifest: manifest)
    let surface = SurfaceDescriptor(id: "generated-render-fixture", globalBounds: .init(x: 0, y: 0, width: 640, height: 360), pixelWidth: 640, pixelHeight: 360)
    // Generated pixel content only: this is deliberately not a user recording
    // or an imitation of a product screenshot. The inspector remains the real UI.
    var pixels = Data(count: 640 * 360 * 4)
    pixels.withUnsafeMutableBytes { raw in
        let bytes = raw.bindMemory(to: UInt8.self)
        for y in 0..<360 { for x in 0..<640 {
            let offset = (y * 640 + x) * 4
            let target = (250..<390).contains(x) && (110..<250).contains(y)
            bytes[offset] = target ? 220 : 38; bytes[offset + 1] = target ? 160 : 38
            bytes[offset + 2] = target ? 45 : 38; bytes[offset + 3] = 255
        } }
    }
    for index in 0..<4 {
        let time = UInt64(1_000_000_000 + index * 100_000_000)
        try writer.append(FrameArchive.prepare(pixels: pixels, metadata: .init(eventNanos: time, observedNanos: time + 10_000, surface: surface, byteCount: pixels.count)))
    }
    try writer.append(events: [
        RawInputEvent(sequence: 0, eventNanos: 1_000_000_000, observedNanos: 1_000_010_000, origin: .physical, kind: .pointer, x: 320, y: 180),
        RawInputEvent(sequence: 1, eventNanos: 1_020_000_000, observedNanos: 1_020_010_000, origin: .physical, kind: .buttonDown, button: 0),
        RawInputEvent(sequence: 2, eventNanos: 1_080_000_000, observedNanos: 1_080_010_000, origin: .physical, kind: .buttonUp, button: 0),
    ])
    return try writer.finish(at: 1_400_000_000, status: issue == nil ? .complete : .interrupted, issue: issue)
}

@MainActor func renderOwnedView(_ view: AnyView, name: String, size: NSSize, scheme: ColorScheme, output: URL) async throws -> [String: Any] {
    let frame = NSRect(origin: .zero, size: size)
    let hosting = NSHostingView(rootView: view.environment(\.colorScheme, scheme).preferredColorScheme(scheme)
        .frame(width: size.width, height: size.height).background(Color(nsColor: .windowBackgroundColor)))
    hosting.sizingOptions = []
    let window = NSWindow(contentRect: NSRect(x: -20_000, y: -20_000, width: size.width, height: size.height),
                          styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    let appearance = try #require(NSAppearance(named: scheme == .dark ? .darkAqua : .aqua))
    window.appearance = appearance; hosting.appearance = appearance
    window.contentView = hosting; hosting.frame = frame; window.setContentSize(size)
    window.orderFront(nil)
    defer { window.orderOut(nil); window.contentView = nil; window.close() }
    // Give SwiftUI's normal tasks, AppKit tables, and layout two bounded run-loop
    // opportunities to settle; never block the main actor with Thread.sleep.
    try await Task.sleep(for: .milliseconds(160))
    hosting.layoutSubtreeIfNeeded(); hosting.displayIfNeeded()
    try await Task.sleep(for: .milliseconds(80))
    hosting.layoutSubtreeIfNeeded()
    let bitmap = try #require(hosting.bitmapImageRepForCachingDisplay(in: frame))
    hosting.cacheDisplay(in: frame, to: bitmap)
    let image = try #require(bitmap.representation(using: .png, properties: [:]))
    #expect(image.count > 10_000)
    let filename = "\(name)-\(scheme == .dark ? "dark" : "light")-\(Int(size.width))x\(Int(size.height)).png"
    try image.write(to: output.appendingPathComponent(filename), options: .atomic)
    var details: [String: Any] = ["file": filename, "widthPoints": size.width, "heightPoints": size.height,
            "widthPixels": bitmap.pixelsWide, "heightPixels": bitmap.pixelsHigh,
            "appearance": scheme == .dark ? "dark" : "light"]
    // Exercise a real scroll view owned by this test process. This does not send
    // an input event and does not read the OS accessibility hierarchy.
    if let scroll = ownedScrollViews(hosting).max(by: { scrollExtent($0) < scrollExtent($1) }), scrollExtent(scroll) > 1,
       let document = scroll.documentView {
        let y = document.isFlipped ? document.bounds.maxY - scroll.contentView.bounds.height : document.bounds.minY
        scroll.contentView.scroll(to: NSPoint(x: 0, y: y)); scroll.reflectScrolledClipView(scroll.contentView)
        try await Task.sleep(for: .milliseconds(80)); hosting.layoutSubtreeIfNeeded()
        let bottom = try #require(hosting.bitmapImageRepForCachingDisplay(in: frame))
        hosting.cacheDisplay(in: frame, to: bottom)
        let bottomFile = filename.replacingOccurrences(of: ".png", with: "-bottom.png")
        try #require(bottom.representation(using: .png, properties: [:])).write(to: output.appendingPathComponent(bottomFile), options: .atomic)
        details["scrolledFile"] = bottomFile
    }
    return details
}

@MainActor private func ownedScrollViews(_ view: NSView) -> [NSScrollView] {
    (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap { ownedScrollViews($0) }
}
@MainActor private func scrollExtent(_ scroll: NSScrollView) -> CGFloat {
    (scroll.documentView?.bounds.height ?? 0) - scroll.contentView.bounds.height
}
