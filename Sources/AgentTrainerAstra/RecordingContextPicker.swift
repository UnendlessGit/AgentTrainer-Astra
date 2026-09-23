import SwiftUI
import AstraCore

/// A historical checkpoint can supply the exact values needed for additional
/// demonstrations, even if today's reusable context catalog has changed.
struct RecordingContextPicker: View {
    let agent: AgentDocument
    let workspace: WorkspaceModel
    @Binding var vocabulary: ContextVocabulary?
    @Binding var loading: Bool
    @State private var checkpointID: UUID?
    @State private var issue: String?
    private var checkpoints: [CheckpointDocument] {
        workspace.checkpoints.filter { workspace.checkpointLinks[agent.id]?.contains($0.id) == true }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Context vocabulary", selection: $checkpointID) {
                Text("Current agent fields").tag(nil as UUID?)
                ForEach(checkpoints) { Text("Checkpoint · \($0.name)").tag(Optional($0.id)) }
            }
            if loading { ProgressView("Reading saved contexts…").controlSize(.small) }
            if let issue { AttentionLabel(message: issue).font(.callout) }
        }.task(id: checkpointID) {
            loading = true; issue = nil; vocabulary = nil
            defer { if !Task.isCancelled { loading = false } }
            do {
                let resolved: ContextVocabulary
                if let checkpointID {
                    guard let checkpoint = checkpoints.first(where: { $0.id == checkpointID }) else {
                        throw AstraError("context.checkpoint", "The selected checkpoint is no longer linked to this agent.")
                    }
                    let manifest = try await LearningFiles.read(workspace.checkpointDirectory(checkpointID).appendingPathComponent("manifest.json"))
                    guard manifest.fields?["id"]?.uuid == checkpointID,
                          manifest.fields?["policySignature"]?.text == checkpoint.policySignature else {
                        throw AstraError("context.checkpoint", "The saved context vocabulary does not match this checkpoint.")
                    }
                    let model = try manifest.required("model")
                    if let named = try ContextVocabulary.from(model: model) { resolved = named }
                    else {
                        guard (try model.required("context_sizes").decode([Int].self)).isEmpty else {
                            throw AstraError("context.unnamed", "This older checkpoint has unnamed numeric contexts. It cannot supply named demonstration values.")
                        }
                        resolved = .empty
                    }
                } else { resolved = try workspace.contextVocabulary(for: agent) }
                try Task.checkCancellation(); vocabulary = resolved
            } catch is CancellationError {} catch { if !Task.isCancelled { issue = error.localizedDescription } }
        }
    }
}
