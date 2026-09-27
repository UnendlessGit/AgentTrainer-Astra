import SwiftUI
import AstraCore

struct QueuedActionMemoryOption: View {
    @Binding var enabled: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle("Queued action memory", isOn: $enabled).toggleStyle(.checkbox)
            Text("Adds a learned input for commands the agent has already issued, including pending commands and their progress. Human recordings alone do not teach this input; use practice demonstrations or reinforcement experience.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if enabled {
                Text("Long execution leads and large command packets can fill the retained-control budget. Astra stops if that history cannot be preserved.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

struct QueuedActionCheckpointDescription: View {
    let sourceName: String
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Create a queued-action copy").font(.title2.weight(.semibold))
            Text(sourceName).font(.headline).fixedSize(horizontal: false, vertical: true)
            Text("Copies the learned weights into a separate checkpoint with queued action memory. The added input starts with no effect on the policy’s output.")
                .fixedSize(horizontal: false, vertical: true)
            Text("The copy starts new learning state: optimizer progress, recurrent state and random sampling state are not resumed. Choose the new checkpoint for a fresh training run to teach the added input.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text("The original checkpoint stays available.").font(.callout).foregroundStyle(.secondary)
        }
    }
}

struct QueuedActionCheckpointSheet: View {
    let agent: AgentDocument
    let checkpoint: CheckpointDocument
    let model: WorkspaceModel
    let created: (CheckpointDocument) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var checking = true
    @State private var eligible = false
    @State private var working = false
    @State private var stopRequested = false
    @State private var issue: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            QueuedActionCheckpointDescription(sourceName: checkpoint.name)
            if checking { ProgressView("Checking the saved checkpoint…").controlSize(.small) }
            if let issue { AttentionLabel(message: issue) }
            if let unavailable = model.checkpointManagementUnavailableReason, !working { AttentionLabel(message: unavailable) }
            if working {
                ProgressView(stopRequested || model.learning?.isStopping == true ? "Stopping after the current checkpoint operation…" : "Creating the separate checkpoint…")
                    .controlSize(.small)
            }
            HStack {
                Spacer()
                if working {
                    Button(stopRequested || model.learning?.isStopping == true ? "Stopping…" : "Stop Copy", role: .destructive) {
                        stopRequested = true
                        Task { await model.learning?.requestStop() }
                    }.keyboardShortcut(.cancelAction).disabled(stopRequested || model.learning?.isStopping == true)
                } else { Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction) }
                Button("Create Copy") {
                    working = true; stopRequested = false; issue = nil
                    Task {
                        defer { working = false }
                        do {
                            if stopRequested { throw CancellationError() }
                            let copy = try await model.createQueuedActionCheckpoint(agent: agent, checkpoint: checkpoint)
                            created(copy); dismiss()
                        } catch is CancellationError { issue = "Checkpoint copy stopped." }
                        catch { issue = error.localizedDescription }
                    }
                }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(checking || !eligible || working || model.checkpointManagementUnavailableReason != nil)
            }
        }.padding(24).frame(width: 500).interactiveDismissDisabled(working)
            .task(id: checkpoint.id) {
                checking = true; eligible = false; issue = nil
                defer { if !Task.isCancelled { checking = false } }
                do {
                    let manifest = try await LearningFiles.read(model.checkpointDirectory(checkpoint.id).appendingPathComponent("manifest.json"))
                    guard manifest.fields?["id"]?.uuid == checkpoint.id,
                          manifest.fields?["policySignature"]?.text == checkpoint.policySignature else {
                        throw AstraError("checkpoint.identity", "The checkpoint no longer matches its catalog identity.")
                    }
                    try Task.checkCancellation()
                    switch manifest.fields?["model"]?.fields?["schema_version"]?.int {
                    case 2: eligible = true
                    case 3: issue = "This checkpoint already has queued action memory. Use it directly as a starting point."
                    default: issue = "This checkpoint cannot be copied into the current queued-action model."
                    }
                } catch is CancellationError {} catch { if !Task.isCancelled { issue = error.localizedDescription } }
            }
    }
}
