import AppKit
import SwiftUI
import AstraCore

struct EvaluationWorkspaceView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var comparing = false
    @State private var split = "validation"
    @State private var sourceID: UUID?
    @State private var candidates: Set<UUID> = []
    @State private var selectedResults: Set<UUID> = []

    init(agent: AgentDocument, model: WorkspaceModel, comparing: Bool = false, sourceID: UUID? = nil,
         candidates: Set<UUID> = [], selectedResults: Set<UUID> = []) {
        self.agent = agent; self.model = model
        _comparing = State(initialValue: comparing); _sourceID = State(initialValue: sourceID)
        _candidates = State(initialValue: candidates); _selectedResults = State(initialValue: selectedResults)
    }

    private var checkpoints: [CheckpointDocument] {
        model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true }
    }
    private var checkpoint: CheckpointDocument? { checkpoints.first { $0.id == agent.selectedCheckpointID } }
    private var sources: [CheckpointDocument] {
        var runs: Set<UUID> = []
        return checkpoints.sorted { $0.createdAt > $1.createdAt }.filter {
            guard let runID = $0.runID, model.learningRuns.contains(where: { $0.id == runID && $0.kind == .behavioral }) else { return false }
            return runs.insert(runID).inserted
        }
    }
    private var source: CheckpointDocument? { sources.first { $0.id == sourceID } }
    private var history: [EvaluationDocument] { model.evaluations.filter { $0.agentID == agent.id } }
    private var disabled: Bool {
        model.isClosing || model.isRunningAgent || model.learning == nil || model.learning?.isBusy == true
    }
    private var chosenResults: [EvaluationDocument] { history.filter { selectedResults.contains($0.id) } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Evaluation").font(.title2.weight(.semibold))
                Text("Measure how well a checkpoint predicts recorded demonstrations. Lower mean negative log likelihood (NLL) is better on the same dataset and scoring protocol.")
                    .foregroundStyle(.secondary)
                Picker("Evaluation mode", selection: $comparing) {
                    Text("Selected checkpoint").tag(false)
                    Text("Compare checkpoints").tag(true)
                }.pickerStyle(.segmented).frame(maxWidth: 420).disabled(disabled)
                GroupBox {
                    VStack(alignment: .leading, spacing: 14) {
                        if comparing { comparisonSetup.disabled(disabled) }
                        else { singleSetup }
                        Picker("Demonstrations", selection: $split) {
                            Text("Validation").tag("validation")
                            Text("Test").tag("test")
                            Text("Training (diagnostic)").tag("train")
                        }.frame(maxWidth: 440).disabled(disabled)
                        if split == "train" {
                            Text("Training loss is a diagnostic; these demonstrations were available during learning.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        HStack {
                            Button(comparing ? "Evaluate \(candidates.intersection(Set(checkpoints.map(\.id))).count) Checkpoints" : "Evaluate", systemImage: "checkmark.seal") {
                                if comparing, let source {
                                    model.evaluateCheckpoints(checkpoints.filter { candidates.contains($0.id) }, agentID: agent.id,
                                                              datasetCheckpoint: source, split: split)
                                } else if let checkpoint { model.evaluateCheckpoint(checkpoint, agentID: agent.id, split: split) }
                            }.buttonStyle(.borderedProminent)
                                .disabled(disabled || (comparing ? (source == nil || candidates.isDisjoint(with: Set(checkpoints.map(\.id)))) : !hasDataset(checkpoint)))
                            if let learning = model.learning, learning.isBusy, learning.activeAgentID == agent.id, learning.activeRun == nil {
                                ProgressView().controlSize(.small)
                                Text(learning.phase).font(.caption).foregroundStyle(.secondary)
                                Button("Stop") { Task { await learning.requestStop() } }.disabled(learning.isStopping)
                            }
                        }
                        Text("Checkpoint jobs run one at a time. Demonstration likelihood does not measure success while controlling an environment.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                }
                if let failure = model.learning?.failure, model.learning?.resultAgentID == agent.id {
                    AttentionLabel(message: failure)
                }
                if !history.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Evaluation history").font(.headline)
                        Text("Select results to compare. Hold Command to select several rows.").font(.caption).foregroundStyle(.secondary)
                        EvaluationHistoryTable(documents: history, selection: $selectedResults)
                            .frame(height: min(440, max(170, CGFloat(history.count) * 33 + 35)))
                        if !chosenResults.isEmpty { EvaluationResultDetails(documents: chosenResults, root: model.supportRoot) }
                    }
                } else {
                    ContentUnavailableView("No evaluations yet", systemImage: "checkmark.seal",
                                           description: Text("Scores and their exact dataset, split and checkpoint identities are saved here after evaluation."))
                }
            }.padding(.trailing, 8)
        }
    }

    @ViewBuilder private var singleSetup: some View {
        if let checkpoint {
            Text(checkpoint.name).font(.headline)
            Text("\(checkpoint.trainingStep.formatted()) updates · \(checkpoint.parameterCount.formatted()) parameters")
                .font(.caption).foregroundStyle(.secondary)
            if !hasDataset(checkpoint) {
                Text("This checkpoint has no saved demonstration dataset. Use Compare checkpoints to score it against another saved, compatible dataset.")
                    .font(.callout).foregroundStyle(.secondary)
            } else { Text("Uses this checkpoint’s saved demonstration dataset.").font(.callout).foregroundStyle(.secondary) }
        } else {
            Text("Choose a checkpoint in Training, or compare any checkpoints linked to this agent.").foregroundStyle(.secondary)
            Button("Open Training") { model.section = .training }
        }
    }

    @ViewBuilder private var comparisonSetup: some View {
        Text("One shared dataset").font(.headline)
        if sources.isEmpty {
            Text("Train from demonstrations first to save a dataset for comparison.").foregroundStyle(.secondary)
        } else {
            Picker("Saved dataset from", selection: $sourceID) {
                Text("Choose a saved dataset…").tag(nil as UUID?)
                ForEach(sources) { item in
                    Text("\(item.name) · \(item.createdAt.formatted(date: .abbreviated, time: .shortened))").tag(Optional(item.id))
                }
            }
            if let source {
                let generated = model.learningRuns.first { $0.id == source.runID }?.sourceKind == "practice_oracle"
                Text(generated ? "Generated practice demonstrations with saved seeds and task settings." : "A saved recording revision with its original session splits and intervals.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("The split belongs to this dataset. Other checkpoints may have seen these demonstrations during training; this score alone does not establish held-out generalization.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Checkpoints").font(.subheadline.weight(.medium))
            ForEach(checkpoints) { item in
                HStack(alignment: .top) {
                    Toggle(isOn: Binding(get: { candidates.contains(item.id) }, set: { enabled in
                        if enabled { candidates.insert(item.id) } else { candidates.remove(item.id) }
                    })) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(item.name)
                            Text("\(item.trainingStep.formatted()) updates").font(.caption).foregroundStyle(.secondary)
                            if let source, source.policySignature != item.policySignature {
                                Text("Incompatible model, controls or timing").font(.caption).foregroundStyle(.orange)
                            }
                        }
                    }.toggleStyle(.checkbox)
                    Spacer(minLength: 0)
                }
            }
        }
    }

    private func hasDataset(_ checkpoint: CheckpointDocument?) -> Bool {
        guard let runID = checkpoint?.runID else { return false }
        return model.learningRuns.contains { $0.id == runID && $0.kind == .behavioral }
    }
}

private struct EvaluationHistoryTable: View {
    let documents: [EvaluationDocument]
    @Binding var selection: Set<UUID>
    var body: some View {
        Table(documents, selection: $selection) {
            TableColumn("Checkpoint") { Text($0.checkpointName).lineLimit(1) }.width(min: 150, ideal: 220)
            TableColumn("Mean NLL") { row in
                if let value = row.meanNLL { Text(value.formatted(.number.precision(.fractionLength(4)))).monospacedDigit() }
                else { Text("—").foregroundStyle(.secondary).accessibilityLabel("No score") }
            }.width(85)
            TableColumn("Decisions") { row in Text(row.decisions?.formatted() ?? "—").monospacedDigit() }.width(85)
            TableColumn("Split") { Text($0.protocolDefinition.split.capitalized) }.width(85)
            TableColumn("Status") { Text($0.status.displayName).foregroundStyle($0.status == .failed ? Color.orange : Color.secondary) }.width(110)
            TableColumn("Date") { Text($0.createdAt.formatted(date: .abbreviated, time: .shortened)) }.width(min: 150, ideal: 170)
        }.tableStyle(.inset)
    }
}

private struct EvaluationResultDetails: View {
    let documents: [EvaluationDocument]
    let root: URL
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                if let first = documents.first {
                    if documents.count > 1 {
                        let issue = documents.dropFirst().compactMap { first.comparisonIssue(with: $0) }.first
                        if let issue {
                            AttentionLabel(message: "These results cannot be compared: \(issue)")
                        } else {
                            let ordered = documents.sorted { ($0.meanNLL ?? .infinity) < ($1.meanNLL ?? .infinity) }
                            Text("Same dataset and protocol · \(first.decisions?.formatted() ?? "—") decisions each").font(.subheadline.weight(.medium))
                            ForEach(ordered) { row in
                                if let value = row.meanNLL, let best = ordered.first?.meanNLL {
                                    LabeledContent(row.checkpointName) {
                                        Text("\(value.formatted(.number.precision(.fractionLength(4))))\(row.id == ordered.first?.id ? " · lowest NLL" : " · +" + (value - best).formatted(.number.precision(.fractionLength(4))))")
                                            .monospacedDigit().textSelection(.enabled)
                                    }
                                }
                            }
                        }
                    }
                    ForEach(documents) { row in
                        VStack(alignment: .leading, spacing: 5) {
                            Text(row.checkpointName).font(.subheadline.weight(.medium))
                            Text("\(row.protocolDefinition.sourceName) · \(row.protocolDefinition.provenance == "practice_oracle" ? "Generated practice" : "Recorded demonstrations") · \(row.protocolDefinition.split.capitalized)")
                                .font(.caption).foregroundStyle(.secondary)
                            if let issue = row.issue { Text(issue).font(.callout).foregroundStyle(.secondary).textSelection(.enabled) }
                            if let dataset = row.datasetID { Text("Dataset \(dataset.uuidString.lowercased())").font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
                            Text("\(row.status.displayName) · \((row.finishedAt ?? row.createdAt).formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption).foregroundStyle(.secondary)
                            if row.status != .running, FileManager.default.fileExists(atPath: root.appendingPathComponent("Jobs/\(row.id.uuidString.lowercased())").path) {
                                Button("Show Saved Result") {
                                    NSWorkspace.shared.activateFileViewerSelecting([root.appendingPathComponent("Jobs/\(row.id.uuidString.lowercased())")])
                                }.controlSize(.small)
                            }
                        }
                    }
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
        }
    }
}

private extension EvaluationStatus {
    var displayName: String {
        switch self {
        case .running: "Running"
        case .completed: "Completed"
        case .unavailable: "Unavailable"
        case .failed: "Needs attention"
        case .cancelled: "Stopped"
        case .interrupted: "Interrupted"
        }
    }
}
