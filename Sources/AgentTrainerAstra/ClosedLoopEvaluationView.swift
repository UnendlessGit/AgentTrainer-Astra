import AppKit
import SwiftUI
import AstraCore

private enum PracticeEvaluationLayout: String, CaseIterable, Identifiable {
    case standard, compact, retina
    var id: String { rawValue }
    var title: String {
        switch self { case .standard: "1280 × 720"; case .compact: "960 × 600 · negative origin"; case .retina: "1920 × 1200 · 2× scale" }
    }
    func trial(seed: Int) -> ClosedLoopTrial {
        switch self {
        case .standard: .init(seed: seed)
        case .compact: .init(seed: seed, pixelWidth: 960, pixelHeight: 600, logicalBounds: [-960, 0, 960, 600])
        case .retina: .init(seed: seed, pixelWidth: 1920, pixelHeight: 1200, logicalBounds: [0, 0, 960, 600])
        }
    }
}

struct ClosedLoopEvaluationView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var candidates: Set<UUID> = []
    @State private var sourceID: UUID?
    @State private var protocolDraft = ClosedLoopProtocol(trials: [])
    @State private var firstSeed = 100_000
    @State private var seedCount = 8
    @State private var layouts: Set<PracticeEvaluationLayout> = [.standard, .compact]
    @State private var selectedResults: Set<UUID> = []
    @State private var sourceLoading = false
    @State private var sourceReady = false
    @State private var issue: String?
    private var checkpoints: [CheckpointDocument] { model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true } }
    private var history: [ClosedLoopEvaluationDocument] { model.closedLoopEvaluations.filter { $0.agentID == agent.id } }
    private var chosenHistory: [ClosedLoopEvaluationDocument] { history.filter { selectedResults.contains($0.id) } }
    private var busy: Bool { model.isClosing || model.isRunningAgent || model.isLearning || model.isRecording || model.learning == nil }
    private var definition: ClosedLoopProtocol? {
        guard (0...1_000_000_000).contains(firstSeed), (1...64).contains(seedCount),
              firstSeed <= 1_000_000_000 - seedCount + 1, !layouts.isEmpty else { return nil }
        var result = protocolDraft
        result.trials = PracticeEvaluationLayout.allCases.filter(layouts.contains).flatMap { layout in
            (firstSeed..<(firstSeed + seedCount)).map { layout.trial(seed: $0) }
        }
        return try? result.validated()
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Practice outcomes").font(.title2.weight(.semibold))
                Text("Run each checkpoint in the same local visual tasks. These simulated trials use the agent’s actions and require no screen or input permissions.")
                    .foregroundStyle(.secondary)
                GroupBox("Checkpoints") {
                    VStack(alignment: .leading, spacing: 12) {
                        Picker("Timing and contexts from", selection: $sourceID) {
                            Text("Choose a checkpoint").tag(nil as UUID?)
                            ForEach(checkpoints) { Text($0.name).tag(Optional($0.id)) }
                        }
                        Text("The protocol keeps these timing and context meanings for every candidate. Model architectures may differ; controls must fit the selected practice task.")
                            .font(.caption).foregroundStyle(.secondary)
                        ForEach(checkpoints) { checkpoint in
                            Toggle(checkpoint.name, isOn: Binding(get: { candidates.contains(checkpoint.id) }, set: {
                                if $0 { candidates.insert(checkpoint.id) } else { candidates.remove(checkpoint.id) }
                            })).toggleStyle(.checkbox)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                }.disabled(busy)
                protocolSettings.disabled(busy || sourceLoading)
                if let issue { AttentionLabel(message: issue) }
                if let failure = model.learning?.failure, model.learning?.resultAgentID == agent.id { AttentionLabel(message: failure) }
                HStack {
                    Button("Evaluate Selected Checkpoints", systemImage: "play.fill") {
                        guard let definition else { return }
                        model.evaluateClosedLoop(checkpoints: checkpoints.filter { candidates.contains($0.id) }, agentID: agent.id, definition: definition)
                    }.buttonStyle(.borderedProminent)
                        .disabled(busy || !sourceReady || definition == nil || candidates.intersection(Set(checkpoints.map(\.id))).isEmpty)
                    if sourceLoading { ProgressView().controlSize(.small) }
                    if let learning = model.learning, learning.isBusy, learning.activeAgentID == agent.id, learning.activeRun == nil {
                        ProgressView().controlSize(.small)
                        Text(learning.phase).font(.caption).foregroundStyle(.secondary)
                        Button("Stop") { Task { await learning.requestStop() } }.disabled(learning.isStopping)
                    }
                }
                Text("Seeds determine the task layouts. A separate seed range does not prove a checkpoint never saw them during training. Results measure these practice tasks, not general desktop success or live control timing.")
                    .font(.caption).foregroundStyle(.secondary)
                if !history.isEmpty {
                    Text("Saved practice evaluations").font(.headline)
                    Table(history.sorted { $0.createdAt > $1.createdAt }, selection: $selectedResults) {
                        TableColumn("Checkpoint") { Text($0.checkpointName) }.width(min: 150, ideal: 220)
                        TableColumn("Successes") { row in Text(row.result.map { "\($0.successes) / \($0.trials.count)" } ?? "—").monospacedDigit() }.width(95)
                        TableColumn("Timeouts") { row in Text(row.result.map { "\($0.timeouts)" } ?? "—") }.width(80)
                        TableColumn("Faults") { row in Text(row.result.map { "\($0.faults)" } ?? "—") }.width(60)
                        TableColumn("Status") { Text($0.status.rawValue.capitalized) }.width(95)
                        TableColumn("Date") { Text($0.createdAt.formatted(date: .abbreviated, time: .shortened)) }.width(min: 145, ideal: 165)
                    }.frame(height: 230)
                    ClosedLoopResultDetails(documents: chosenHistory, root: model.supportRoot,
                        repeatDisabled: busy || candidates.intersection(Set(checkpoints.map(\.id))).isEmpty,
                        repeatProtocol: { saved in
                            model.evaluateClosedLoop(checkpoints: checkpoints.filter { candidates.contains($0.id) }, agentID: agent.id, definition: saved)
                        })
                }
            }.padding(.trailing, 8).padding(.bottom, 24)
        }
        .onAppear {
            if sourceID == nil { sourceID = agent.selectedCheckpointID ?? checkpoints.first?.id }
            if candidates.isEmpty, let id = sourceID { candidates = [id] }
        }
        .task(id: sourceID) { await readSource() }
    }

    private var protocolSettings: some View {
        GroupBox("Fixed evaluation protocol") {
            VStack(alignment: .leading, spacing: 14) {
                Picker("Practice task", selection: $protocolDraft.task) {
                    Text("Pointing").tag("pointing"); Text("Delayed visual memory").tag("delayed_memory")
                }
                if protocolDraft.task == "delayed_memory" {
                    Picker("Memory delay", selection: $protocolDraft.delayMS) {
                        Text("2 seconds").tag(2000); Text("8 seconds").tag(8000); Text("30 seconds").tag(30_000)
                    }.onChange(of: protocolDraft.delayMS) { _, delay in protocolDraft.timeLimitMS = max(protocolDraft.timeLimitMS, delay + protocolDraft.cueMS + 5000) }
                }
                LabeledContent("Decision timing", value: "\(protocolDraft.periodMS) ms cadence · \(protocolDraft.leadMS) ms lead")
                Stepper("Time limit per trial: \(protocolDraft.timeLimitMS / 1000) seconds", value: $protocolDraft.timeLimitMS, in: 1000...600_000, step: 1000)
                HStack {
                    LabeledContent("First environment seed") {
                        TextField("First environment seed", value: $firstSeed, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder).labelsHidden().frame(maxWidth: 120)
                    }
                    Stepper("Seeds per layout: \(seedCount)", value: $seedCount, in: 1...64)
                }
                ForEach(PracticeEvaluationLayout.allCases) { layout in
                    Toggle(layout.title, isOn: Binding(get: { layouts.contains(layout) }, set: {
                        if $0 { layouts.insert(layout) } else { layouts.remove(layout) }
                    })).toggleStyle(.checkbox)
                }
                Picker("Action selection", selection: $protocolDraft.deterministic) {
                    Text("Most likely").tag(true); Text("Sample the policy").tag(false)
                }
                if !protocolDraft.deterministic {
                    LabeledContent("Policy random seed") {
                        TextField("Policy random seed", value: $protocolDraft.policySeed, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder).labelsHidden().frame(maxWidth: 180)
                    }
                    Text("Trial i starts a fresh random stream with policy seed + i. Recurrence resets before every trial.").font(.caption).foregroundStyle(.secondary)
                }
                ContextValuePickers(vocabulary: protocolDraft.contextVocabulary, sizes: protocolDraft.contextSizes, indices: $protocolDraft.contextIDs)
                Text("\(definition?.trials.count ?? 0) fixed trials per checkpoint · no reward shaping").font(.caption).foregroundStyle(.secondary)
                if sourceReady && definition == nil { AttentionLabel(message: "Use valid seeds and contexts, choose a layout, and allow time after the memory delay. Reduce the trial count or time limit if the decision budget is exceeded.").font(.caption) }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
        }
    }
    private func readSource() async {
        sourceReady = false; sourceLoading = true; issue = nil
        defer { if !Task.isCancelled { sourceLoading = false } }
        guard let id = sourceID, let source = checkpoints.first(where: { $0.id == id }) else { return }
        do {
            let manifest = try await LearningFiles.read(model.checkpointDirectory(id).appendingPathComponent("manifest.json"))
            guard manifest.fields?["id"]?.uuid == id, manifest.fields?["policySignature"]?.text == source.policySignature else {
                throw AstraError("evaluation.source", "The selected checkpoint changed identity.")
            }
            let config = try manifest.required("model")
            let sizes = try config.required("context_sizes").decode([Int].self)
            guard let period = config.fields?["period_ms"]?.int, let lead = config.fields?["lead_ms"]?.int else {
                throw AstraError("evaluation.timing", "The checkpoint has no valid decision timing.")
            }
            try Task.checkCancellation()
            protocolDraft.periodMS = period; protocolDraft.leadMS = lead; protocolDraft.contextSizes = sizes
            protocolDraft.contextVocabulary = try ContextVocabulary.from(model: config); protocolDraft.contextIDs = sizes.map { _ in 0 }
            if let keys = try? manifest.required("actions").required("keyCodes").decode([Int].self), keys.contains(123) && keys.contains(124) { protocolDraft.task = "delayed_memory" }
            sourceReady = true
        } catch is CancellationError {} catch { if !Task.isCancelled { issue = error.localizedDescription } }
    }
}

private struct ClosedLoopResultDetails: View {
    let documents: [ClosedLoopEvaluationDocument]
    let root: URL
    let repeatDisabled: Bool
    let repeatProtocol: (ClosedLoopProtocol) -> Void
    var body: some View {
        if let first = documents.first {
            GroupBox("Selected results") {
                VStack(alignment: .leading, spacing: 12) {
                    if documents.count > 1 {
                        if let mismatch = documents.dropFirst().compactMap({ first.comparisonIssue(with: $0) }).first { AttentionLabel(message: mismatch) }
                        else { Text("Same fixed task, layouts, seeds, timing, action selection and contexts.").font(.caption).foregroundStyle(.secondary) }
                    }
                    ForEach(documents) { row in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(row.checkpointName).font(.headline)
                            Text("\(row.protocolDefinition.task == "pointing" ? "Pointing" : "Delayed visual memory") · \(row.protocolDefinition.periodMS) ms cadence · \(row.protocolDefinition.leadMS) ms lead · \(row.protocolDefinition.deterministic ? "Most likely actions" : "Sampled actions")")
                                .font(.caption).foregroundStyle(.secondary)
                            if let result = row.result {
                                Text("\(result.successes) / \(result.trials.count) successes · \(result.timeouts) timeouts · \(result.wrongChoices) wrong choices · \(result.faults) faults")
                                Text("Mean return: \(result.meanReturn.formatted(.number.precision(.fractionLength(3))))").font(.callout).monospacedDigit()
                                DisclosureGroup("Per-trial outcomes") {
                                    Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 5) {
                                        GridRow { Text("Seed"); Text("Layout"); Text("Outcome"); Text("Decisions"); Text("Virtual time") }.fontWeight(.medium)
                                        ForEach(result.trials) { trial in
                                            let layout = row.protocolDefinition.trials[trial.index]
                                            GridRow {
                                                Text("\(trial.seed)"); Text("\(layout.pixelWidth) × \(layout.pixelHeight)")
                                                Text(trial.success ? "Success" : trial.outcome.replacingOccurrences(of: "_", with: " "))
                                                Text("\(trial.decisions)"); Text("\(Double(trial.virtualDurationMS) / 1000, specifier: "%.2f") s")
                                            }.font(.caption).monospacedDigit()
                                            if let fault = trial.fault { GridRow { Text(fault).foregroundStyle(.secondary).gridCellColumns(5) } }
                                        }
                                    }.padding(.top, 8)
                                }
                            } else { Text(row.status.rawValue.capitalized).foregroundStyle(.secondary) }
                            if let issue = row.issue { Text(issue).foregroundStyle(.secondary) }
                            Button("Evaluate Selected with This Protocol") { repeatProtocol(row.protocolDefinition) }
                                .disabled(repeatDisabled)
                            Button("Show Saved Result") {
                                NSWorkspace.shared.activateFileViewerSelecting([root.appendingPathComponent("Jobs/\(row.id.uuidString.lowercased())")])
                            }.controlSize(.small).disabled(row.status == .running)
                        }
                    }
                }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
            }
        }
    }
}
