import SwiftUI
import AstraCore
import AstraPlatform

/// Selection stays in the native workspace; this view owns only the draft run
/// settings. Capture discovery and checkpoint inspection remain explicit inputs.
struct RunView: View {
    let coordinator: InferenceCoordinator?
    let checkpoints: [CheckpointDocument]
    let sources: [CaptureSource]
    let refreshingSources: Bool
    let sourceIssue: String?
    let unavailableReason: String?
    var contextSizes: [Int] = []
    var selectedCheckpointID: UUID? = nil
    let refreshSources: () -> Void
    let selectCheckpoint: (UUID?) -> Void
    let start: (CheckpointDocument, CaptureSource, InferenceOptions) -> Void

    @State private var checkpointID: UUID?
    @State private var sourceID: String?
    @State private var options = InferenceOptions()

    private var checkpoint: CheckpointDocument? { checkpoints.first { $0.id == checkpointID } }
    private var source: CaptureSource? { sources.first { $0.id == sourceID } }
    private var busy: Bool { coordinator?.isBusy == true }
    private var validOptions: Bool {
        (0...1_000_000_000).contains(options.seed) && options.contextIDs.count == contextSizes.count
            && zip(options.contextIDs, contextSizes).allSatisfy { $0 >= 0 && $0 < $1 }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Run your agent").font(.title2.weight(.semibold))
                    Text("Choose a checkpoint and an environment. Astra checks the policy’s timing before it starts using the selected controls.")
                        .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                if checkpoints.isEmpty {
                    ContentUnavailableView("No checkpoint yet", systemImage: "cpu",
                                           description: Text("Train from demonstrations or reinforcement learning to create a local policy."))
                } else {
                    configuration
                    if let unavailableReason { Label(unavailableReason, systemImage: "info.circle").foregroundStyle(.secondary) }
                    HStack {
                        Button {
                            if let checkpoint, let source { start(checkpoint, source, options) }
                        } label: { Label("Start Agent", systemImage: "play.fill") }
                            .buttonStyle(.borderedProminent)
                            .disabled(busy || checkpoint == nil || source == nil || !validOptions || unavailableReason != nil
                                      || coordinator?.requiresManualControlCleanupAcknowledgement == true)
                        if busy, let coordinator {
                            Button("Stop Agent", role: .destructive) { Task { await coordinator.stopAndWait() } }
                                .disabled(coordinator.isStopping)
                        }
                    }
                    Text("Move the pointer or press a key to take over. Control–Option–Command–Escape is the emergency stop. Astra releases the controls when the run ends.")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                if let coordinator, !coordinator.phase.isEmpty { InferenceStatus(coordinator: coordinator) }
            }.padding(.trailing, 8).padding(.bottom, 20)
                .frame(maxWidth: 850, alignment: .leading).frame(maxWidth: .infinity, alignment: .leading)
        }
        .task {
            chooseCheckpoint(selectedCheckpointID ?? checkpoints.first?.id)
            resetContexts()
        }
        .onChange(of: selectedCheckpointID) { _, value in
            if value != checkpointID { chooseCheckpoint(value) }
        }
        .onChange(of: checkpoints.map(\.id)) { _, ids in
            if checkpointID.map({ !ids.contains($0) }) ?? true { chooseCheckpoint(ids.first) }
        }
        .onChange(of: contextSizes) { _, _ in resetContexts() }
        .onChange(of: sources.map(\.id)) { _, ids in
            if sourceID.map({ !ids.contains($0) }) == true { sourceID = nil }
        }
    }

    private var configuration: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 18) {
                Picker("Checkpoint", selection: Binding(get: { checkpointID }, set: { chooseCheckpoint($0) })) {
                    Text("Choose a checkpoint").tag(nil as UUID?)
                    ForEach(checkpoints) { checkpoint in
                        Text("\(checkpoint.name) · \(checkpoint.createdAt.formatted(date: .abbreviated, time: .shortened))")
                            .tag(Optional(checkpoint.id))
                    }
                }
                Picker("Environment", selection: $sourceID) {
                    Text("Choose a window or display").tag(nil as String?)
                    ForEach(sources.filter { $0.kind == .window || $0.kind == .display }) { source in
                        Text(source.name).tag(Optional(source.id))
                    }
                }
                HStack {
                    Button("Refresh Environments", action: refreshSources).disabled(refreshingSources || busy)
                    if refreshingSources { ProgressView().controlSize(.small).accessibilityLabel("Finding environments") }
                }
                if let sourceIssue {
                    AttentionLabel(message: sourceIssue, symbol: "exclamationmark.circle").font(.callout)
                }
                if let source {
                    Text("\(source.pixelWidth) × \(source.pixelHeight) source pixels. The agent stops if the selected environment changes size or position.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                DisclosureGroup("Run settings") {
                    VStack(alignment: .leading, spacing: 14) {
                        Picker("Action selection", selection: $options.deterministic) {
                            Text("Most likely").tag(true)
                            Text("Sample from the learned policy").tag(false)
                        }
                        TextField("Random seed", value: $options.seed, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder)
                            .help("A whole number from 0 to 1,000,000,000. Reusing a seed repeats random choices when observations are identical.")
                        if !contextSizes.isEmpty {
                            Text("Context values stay fixed throughout this run.").font(.caption).foregroundStyle(.secondary)
                            ForEach(contextSizes.indices, id: \.self) { index in
                                let selection = Binding(get: { options.contextIDs.indices.contains(index) ? options.contextIDs[index] : 0 },
                                                        set: { value in if options.contextIDs.indices.contains(index) { options.contextIDs[index] = value } })
                                TextField("Context \(index + 1) · 0–\(contextSizes[index] - 1)", value: selection, format: .number.grouping(.never))
                                    .textFieldStyle(.roundedBorder)
                            }
                        }
                    }.padding(.top, 12)
                }
            }.padding(10).disabled(busy)
        } label: { Text("Environment and policy").font(.headline) }
    }

    private func chooseCheckpoint(_ value: UUID?) { checkpointID = value; selectCheckpoint(value) }
    private func resetContexts() { options.contextIDs = contextSizes.map { _ in 0 } }
}

struct InferenceBanner: View {
    let coordinator: InferenceCoordinator
    var body: some View {
        HStack(spacing: 12) {
            if coordinator.isStopping { ProgressView().controlSize(.small) }
            else { Image(systemName: "cpu").foregroundStyle(.tint) }
            Text(coordinator.phase).font(.callout).lineLimit(2)
            Spacer(minLength: 12)
            Button("Stop Agent", role: .destructive) { Task { await coordinator.stopAndWait() } }
                .disabled(coordinator.isStopping)
        }.padding(.horizontal, 20).padding(.vertical, 12).background(.quaternary)
    }
}

private struct InferenceStatus: View {
    let coordinator: InferenceCoordinator
    @State private var cleanupIssue: String?
    @State private var acknowledgedCleanupRun: UUID?
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Text(coordinator.phase).font(.headline)
                if let failure = coordinator.failure {
                    AttentionLabel(message: failure)
                } else if let reason = coordinator.stopReason {
                    Label(reason, systemImage: "hand.raised").foregroundStyle(.secondary)
                }
                Grid(alignment: .leading, horizontalSpacing: 36, verticalSpacing: 8) {
                    GridRow { Text("Decisions").foregroundStyle(.secondary); Text(coordinator.decisions.formatted()).monospacedDigit() }
                    GridRow { Text("Executed packets").foregroundStyle(.secondary); Text(coordinator.executedPackets.formatted()).monospacedDigit() }
                    if let latency = coordinator.lastLatencyMS {
                        GridRow { Text("Latest prediction").foregroundStyle(.secondary); Text("\(latency, format: .number.precision(.fractionLength(1))) ms").monospacedDigit() }
                    }
                    if let maximum = coordinator.maximumLatencyMS {
                        GridRow { Text("Longest prediction").foregroundStyle(.secondary); Text("\(maximum, format: .number.precision(.fractionLength(1))) ms").monospacedDigit() }
                    }
                    if let policy = coordinator.policy {
                        GridRow { Text("Checkpoint timing").foregroundStyle(.secondary); Text("\(policy.periodMS) ms cadence · \(policy.leadMS) ms lead").monospacedDigit() }
                    }
                }.font(.callout)
                if coordinator.requiresManualControlCleanupAcknowledgement {
                    Button("I’ve released the held controls") {
                        do {
                            try coordinator.acknowledgeManualControlCleanup()
                            acknowledgedCleanupRun = coordinator.runID; cleanupIssue = nil
                        } catch { cleanupIssue = error.localizedDescription }
                    }
                    .help("Confirm only after releasing any held keys or mouse buttons. The previous run keeps its unconfirmed cleanup result.")
                } else if acknowledgedCleanupRun == coordinator.runID, acknowledgedCleanupRun != nil {
                    Text("Manual release acknowledged. You can start another run.").font(.callout).foregroundStyle(.secondary)
                }
                if let cleanupIssue { AttentionLabel(message: cleanupIssue) }
                if let url = coordinator.resultsURL { ShareLink("Export Run Summary", item: url) }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(10)
        } label: { Text("Run feedback").font(.headline) }
    }
}
