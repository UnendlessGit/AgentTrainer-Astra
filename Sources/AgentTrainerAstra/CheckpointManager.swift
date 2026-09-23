import SwiftUI
import AstraCore

struct CheckpointManager: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @Environment(\.dismiss) private var dismiss
    @State private var keepNewest = 5
    @State private var preview: CheckpointRetentionPreview?
    @State private var issue: String?
    @State private var result: String?
    @State private var working = false
    private var checkpoints: [CheckpointDocument] {
        model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true }
            .sorted { $0.isPinned != $1.isPinned ? $0.isPinned : $0.createdAt > $1.createdAt }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Manage checkpoints").font(.title2.weight(.semibold))
                    Text(agent.name).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction).disabled(working)
            }
            if let issue { AttentionLabel(message: issue) }
            if let result { Text(result).foregroundStyle(.secondary) }
            if let unavailable = model.checkpointManagementUnavailableReason, !working { AttentionLabel(message: unavailable) }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    if checkpoints.isEmpty { Text("No checkpoints are linked to this agent.").foregroundStyle(.secondary) }
                    ForEach(checkpoints) { checkpoint in
                        CheckpointPresentationRow(checkpoint: checkpoint,
                            selected: model.agents.first { $0.id == agent.id }?.selectedCheckpointID == checkpoint.id) { name, pinned in
                            working = true; issue = nil; result = nil; preview = nil
                            defer { working = false }
                            do { try await model.updateCheckpoint(checkpoint, name: name, pinned: pinned) }
                            catch { issue = error.localizedDescription }
                        }.disabled(working || model.checkpointManagementUnavailableReason != nil)
                        Divider()
                    }
                }
            }.frame(minHeight: 180, idealHeight: 250, maxHeight: 320)
            Text("Renaming pins a checkpoint so cleanup keeps it; you can unpin it afterward. Names and pins are shared across agents using the same model.")
                .font(.caption).foregroundStyle(.secondary)
            GroupBox("Clean up older checkpoints") {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Stepper("Keep newest \(keepNewest) unpinned", value: $keepNewest, in: 0...100)
                            .onChange(of: keepNewest) { _, _ in preview = nil }
                        Spacer()
                        Button("Preview Cleanup") {
                            Task {
                                working = true; issue = nil; result = nil
                                defer { working = false }
                                do { preview = try await model.previewCheckpointCleanup(agentID: agent.id, keepNewest: keepNewest) }
                                catch { issue = error.localizedDescription }
                            }
                        }.disabled(working || model.checkpointManagementUnavailableReason != nil)
                    }
                    Text("Pinned, selected and actively referenced checkpoints are kept. Shared models stay available to their other agents.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let preview {
                        if preview.items.isEmpty { Text("Nothing can be removed with these settings.").foregroundStyle(.secondary) }
                        else {
                            ScrollView {
                                VStack(alignment: .leading, spacing: 8) {
                                    ForEach(preview.items) { item in
                                        HStack(alignment: .top) {
                                            Text(item.checkpoint.name).lineLimit(2)
                                            Spacer()
                                            if item.disposition == .unlinkShared {
                                                Text("Unlink here · kept by \(item.retainedByAgents.joined(separator: ", "))")
                                                    .foregroundStyle(.secondary).multilineTextAlignment(.trailing)
                                            } else {
                                                Text("Delete local files · \(ByteCountFormatter.string(fromByteCount: Int64(clamping: item.bytes), countStyle: .file))")
                                                    .foregroundStyle(.secondary)
                                            }
                                        }.font(.caption)
                                    }
                                }
                            }.frame(maxHeight: 160)
                            Text("This permanently deletes \(ByteCountFormatter.string(fromByteCount: Int64(clamping: preview.bytesToDelete), countStyle: .file)) of unowned local model files. Historical results and source recordings are retained.")
                                .font(.caption)
                            if preview.remainingCandidates > 0 {
                                Text("\(preview.remainingCandidates) more candidates remain. Preview again after this batch.").font(.caption).foregroundStyle(.secondary)
                            }
                            HStack {
                                Button("Cancel Preview") { self.preview = nil }.disabled(working)
                                Spacer()
                                Button("Remove \(preview.items.count) Checkpoints", role: .destructive) {
                                    Task {
                                        working = true; issue = nil; result = nil
                                        defer { working = false }
                                        do {
                                            let outcome = try await model.applyCheckpointCleanup(preview)
                                            self.preview = nil
                                            result = "Deleted \(outcome.deleted) local models and unlinked \(outcome.unlinked) shared checkpoints."
                                            if !outcome.issues.isEmpty { issue = outcome.issues.joined(separator: "\n") }
                                        } catch { self.preview = nil; issue = error.localizedDescription }
                                    }
                                }.disabled(working || model.checkpointManagementUnavailableReason != nil)
                            }
                        }
                    }
                }.padding(8)
            }
            if working { ProgressView().controlSize(.small) }
        }.padding(24).frame(minWidth: 620, idealWidth: 740, maxWidth: 880)
            .onChange(of: model.checkpoints) { _, _ in preview = nil }
            .interactiveDismissDisabled(working)
    }
}

private struct CheckpointPresentationRow: View {
    let checkpoint: CheckpointDocument
    let selected: Bool
    let save: (String, Bool) async -> Void
    @State private var name: String
    init(checkpoint: CheckpointDocument, selected: Bool, save: @escaping (String, Bool) async -> Void) {
        self.checkpoint = checkpoint; self.selected = selected; self.save = save
        _name = State(initialValue: checkpoint.name)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                TextField("Checkpoint name", text: $name).textFieldStyle(.roundedBorder)
                    .onSubmit { if canRename { Task { await save(name, checkpoint.isPinned) } } }
                Button("Rename") { Task { await save(name, checkpoint.isPinned) } }.disabled(!canRename)
                Button { Task { await save(checkpoint.name, !checkpoint.isPinned) } } label: {
                    Label(checkpoint.isPinned ? "Unpin" : "Pin", systemImage: checkpoint.isPinned ? "pin.fill" : "pin")
                }.help(checkpoint.isPinned ? "Allow this checkpoint to become a cleanup candidate." : "Keep this checkpoint during cleanup.")
            }
            Text("\(checkpoint.kind.capitalized) · step \(checkpoint.trainingStep.formatted()) · \(checkpoint.createdAt.formatted(date: .abbreviated, time: .shortened))\(selected ? " · Selected" : "")")
                .font(.caption).foregroundStyle(.secondary)
        }.onChange(of: checkpoint.name) { _, value in name = value }
    }
    private var canRename: Bool { (try? DocumentNames.validated(name)).map { $0 != checkpoint.name } ?? false }
}
