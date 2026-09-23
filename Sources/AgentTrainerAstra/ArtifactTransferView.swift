import SwiftUI
import AppKit
import AstraCore

enum ArtifactTransferMode: String, Identifiable {
    case importArchive, exportArchive
    var id: String { rawValue }
}

struct ArtifactTransferView: View {
    let model: WorkspaceModel
    let mode: ArtifactTransferMode
    @Environment(\.dismiss) private var dismiss
    @State private var agentID: UUID?
    @State private var recordingIDs: Set<UUID> = []
    @State private var checkpointIDs: Set<UUID> = []
    @State private var archiveURL: URL?
    @State private var exportPlan: ArtifactExportPlan?
    @State private var importPlan: ArtifactImportPlan?
    @State private var importResult: ArtifactImportResult?
    @State private var exportedURL: URL?
    @State private var issue: String?
    @State private var localBusy = false
    @State private var cancelRequested = false
    @State private var transferStarted = false
    @State private var work: Task<Void, Never>?
    @State private var didInitialize = false

    private var isImport: Bool { mode == .importArchive }
    private var busy: Bool { localBusy || model.artifactTransferBusy || cancelRequested }
    private var complete: Bool { isImport ? importResult != nil : exportedURL != nil }
    private var hasPreview: Bool { isImport ? importPlan != nil : exportPlan != nil }
    private var items: [ArtifactTransferItem] { isImport ? (importPlan?.items ?? []) : (exportPlan?.items ?? []) }
    private var notices: [String] { importResult?.notices ?? (isImport ? (importPlan?.notices ?? []) : (exportPlan?.notices ?? [])) }
    private var totalBytes: UInt64 { isImport ? (importPlan?.totalBytes ?? 0) : (exportPlan?.totalBytes ?? 0) }
    private var agentName: String { model.agents.first { $0.id == agentID }?.name ?? "AgentTrainer Astra" }
    private var recordings: [RecordingManifest] {
        guard let agentID else { return [] }
        return model.recordings.filter { model.recordingLinks[agentID]?.contains($0.id) == true && $0.status != .recording }
            .sorted { $0.createdAt > $1.createdAt }
    }
    private var checkpoints: [CheckpointDocument] {
        guard let agentID else { return [] }
        return model.checkpoints.filter { model.checkpointLinks[agentID]?.contains($0.id) == true }
            .sorted { $0.createdAt > $1.createdAt }
    }
    private var canPreview: Bool {
        isImport ? archiveURL != nil : (agentID != nil && (!recordingIDs.isEmpty || !checkpointIDs.isEmpty))
    }

    var body: some View {
        VStack(spacing: 0) {
            header.padding(24)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if complete { completion }
                    else if hasPreview { preview }
                    else if isImport { importSelection }
                    else { exportSelection }
                    if !notices.isEmpty {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            ForEach(Array(notices.enumerated()), id: \.offset) { _, notice in
                                Text(notice).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                    if let issue { AttentionLabel(message: issue).textSelection(.enabled) }
                    if !busy, !complete, let reason = model.artifactUnavailableReason {
                        AttentionLabel(message: reason).font(.callout)
                    }
                    if busy {
                        if cancelRequested {
                            ProgressView("Stopping after the current file operation…")
                        } else if model.artifactTransferBusy {
                            ArtifactTransferProgressView(progress: model.artifactTransferProgress)
                        } else {
                            ProgressView(hasPreview ? "Choosing the archive destination…" : "Checking the archive contents…")
                        }
                    }
                }.padding(24).frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            footer.padding(18)
        }
        .frame(minWidth: 640, idealWidth: 780, maxWidth: 1000, minHeight: 520, idealHeight: 690)
        .interactiveDismissDisabled(busy)
        .onAppear {
            guard !didInitialize else { return }; didInitialize = true
            if !isImport {
                agentID = model.selectedAgent?.id ?? model.agents.first?.id
                selectAllForAgent()
            }
        }
        .onDisappear { if !transferStarted { work?.cancel() } }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(isImport ? "Import Astra Archive" : "Export Astra Archive").font(.title2.weight(.semibold))
            Text(isImport ? "Preview the original recordings, models and learning history before adding them to your library."
                 : "Choose what to include, then review the complete archive before exporting.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var importSelection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(archiveURL?.lastPathComponent ?? "Choose an archive").font(.headline).lineLimit(2)
                    if let archiveURL { Text(archiveURL.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
                    else { Text("AgentTrainer Astra archives use the .astraarchive extension.").font(.callout).foregroundStyle(.secondary) }
                }
                Spacer()
                Button(archiveURL == nil ? "Choose Archive…" : "Choose Another…", action: chooseArchive)
            }
            Picker("Link imported items to", selection: $agentID) {
                Text("Keep the archive’s agents").tag(Optional<UUID>.none)
                ForEach(model.agents) { agent in Text(agent.name).tag(Optional(agent.id)) }
            }
            Text("Choosing an existing agent adds links to its library while preserving the artifacts’ original ownership and history.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.disabled(busy)
    }

    private var exportSelection: some View {
        VStack(alignment: .leading, spacing: 16) {
            if model.agents.isEmpty {
                ContentUnavailableView("No agents to export", systemImage: "square.and.arrow.up",
                    description: Text("Create an agent and add demonstrations or a model before exporting."))
            } else {
                Picker("Agent", selection: Binding(get: { agentID }, set: { agentID = $0; selectAllForAgent() })) {
                    Text("Choose an agent").tag(Optional<UUID>.none)
                    ForEach(model.agents) { agent in Text(agent.name).tag(Optional(agent.id)) }
                }
                Text("Models can require additional datasets, recordings and training settings for resume. The preview lists every included artifact.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        selectionHeader("Demonstrations", selected: recordingIDs.count, total: recordings.count,
                            select: { recordingIDs = Set(recordings.map(\.id)) }, clear: { recordingIDs = [] })
                        if recordings.isEmpty { Text("No demonstrations are linked to this agent.").font(.callout).foregroundStyle(.secondary) }
                        else {
                            LazyVStack(alignment: .leading, spacing: 9) {
                                ForEach(recordings) { recording in
                                    Toggle(isOn: membership(recording.id, in: $recordingIDs)) {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(recording.name).lineLimit(2)
                                            Text("\(recording.frameCount.formatted()) frames · \(bytes(recording.storedBytes))")
                                                .font(.caption).foregroundStyle(.secondary)
                                        }
                                    }.toggleStyle(.checkbox)
                                }
                            }
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(6)
                }
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        selectionHeader("Models", selected: checkpointIDs.count, total: checkpoints.count,
                            select: { checkpointIDs = Set(checkpoints.map(\.id)) }, clear: { checkpointIDs = [] })
                        if checkpoints.isEmpty { Text("No model checkpoints are linked to this agent.").font(.callout).foregroundStyle(.secondary) }
                        else {
                            LazyVStack(alignment: .leading, spacing: 9) {
                                ForEach(checkpoints) { checkpoint in
                                    Toggle(isOn: membership(checkpoint.id, in: $checkpointIDs)) {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(checkpoint.name).lineLimit(2)
                                            Text("\(checkpoint.kind.capitalized) · \(checkpoint.trainingStep.formatted()) updates")
                                                .font(.caption).foregroundStyle(.secondary)
                                        }
                                    }.toggleStyle(.checkbox)
                                }
                            }
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(6)
                }
            }
        }.disabled(busy)
    }

    private var preview: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text(isImport ? "Archive contents" : "Ready to export").font(.headline)
                Spacer()
                Text("\(items.count.formatted()) artifacts · \(bytes(totalBytes))").foregroundStyle(.secondary).monospacedDigit()
            }
            if isImport, let source = importPlan?.source {
                Text(source.lastPathComponent).font(.callout).textSelection(.enabled)
                if let target = importPlan?.linkToAgentID, let agent = model.agents.first(where: { $0.id == target }) {
                    Text("Also link imported items to \(agent.name).").font(.callout).foregroundStyle(.secondary)
                }
            }
            Table(items) {
                TableColumn("Name") { item in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.name).lineLimit(1).truncationMode(.middle)
                        if !isImport, isDependency(item) { Text("Required dependency").font(.caption).foregroundStyle(.secondary) }
                    }.help(item.name + "\n" + item.identity)
                }
                TableColumn("Kind") { item in Text(kindName(item.kind)) }.width(min: 100, ideal: 125, max: 150)
                TableColumn("Size") { item in Text(bytes(item.bytes)).monospacedDigit() }.width(min: 70, ideal: 95, max: 120)
            }.frame(minHeight: 190, idealHeight: 280, maxHeight: 320)
                .accessibilityLabel("Exact archive contents including required dependencies")
            if !isImport {
                Text("The archive includes the items above and their catalog descriptions. Check this list before choosing where to save it.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var completion: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(isImport ? "Import complete" : "Archive exported", systemImage: "checkmark.circle").font(.title3.weight(.semibold))
            if let result = importResult {
                Text("\(result.importedCount.formatted()) artifacts imported · \(result.reusedCount.formatted()) already present")
                Text("\(result.recordingIDs.count.formatted()) demonstrations and \(result.checkpointIDs.count.formatted()) model checkpoints are available in the library.")
                    .foregroundStyle(.secondary)
            }
            if let exportedURL {
                Text(exportedURL.path).font(.callout).textSelection(.enabled)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([exportedURL]) }
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var footer: some View {
        HStack {
            if hasPreview, !complete {
                Button("Back") { exportPlan = nil; importPlan = nil; issue = nil }.disabled(busy)
            }
            Spacer()
            if busy {
                Button(cancelRequested ? "Stopping…" : "Cancel", action: cancel).disabled(cancelRequested).keyboardShortcut(.cancelAction)
            } else if complete {
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            } else {
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                if hasPreview {
                    Button(isImport ? "Import Archive" : "Export Archive…", action: transfer)
                        .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                        .disabled(model.artifactUnavailableReason != nil)
                } else {
                    Button("Preview Contents", action: prepare)
                        .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                        .disabled(!canPreview || model.artifactUnavailableReason != nil)
                }
            }
        }
    }

    private func selectAllForAgent() {
        recordingIDs = Set(recordings.map(\.id)); checkpointIDs = Set(checkpoints.map(\.id))
        exportPlan = nil; issue = nil
    }
    private func membership(_ id: UUID, in values: Binding<Set<UUID>>) -> Binding<Bool> {
        Binding(get: { values.wrappedValue.contains(id) }, set: { included in
            if included { values.wrappedValue.insert(id) } else { values.wrappedValue.remove(id) }
        })
    }
    private func selectionHeader(_ title: String, selected: Int, total: Int,
                                 select: @escaping () -> Void, clear: @escaping () -> Void) -> some View {
        HStack {
            Text(title).font(.headline)
            Text("\(selected) of \(total)").font(.caption).foregroundStyle(.secondary)
            Spacer()
            Button("Select All", action: select).disabled(total == 0 || selected == total)
            Button("Clear", action: clear).disabled(selected == 0)
        }.buttonStyle(.borderless).controlSize(.small)
    }
    private func chooseArchive() {
        guard !busy else { return }
        localBusy = true; issue = nil
        work = Task {
            defer { localBusy = false; work = nil }
            if let value = await NativeArtifactPanels.chooseArchive(), !Task.isCancelled {
                archiveURL = value; importPlan = nil
            }
        }
    }
    private func prepare() {
        guard !busy, canPreview else { return }
        localBusy = true; issue = nil
        let selectedAgent = agentID, selectedRecordings = recordingIDs, selectedCheckpoints = checkpointIDs, archive = archiveURL
        work = Task {
            defer { localBusy = false; work = nil }
            do {
                try Task.checkCancellation()
                if isImport, let archive {
                    let plan = try await model.previewArtifactImport(from: archive, linkToAgentID: selectedAgent)
                    try Task.checkCancellation(); importPlan = plan
                } else if let selectedAgent {
                    let plan = try await model.previewArtifactExport(agentID: selectedAgent, recordingIDs: selectedRecordings, checkpointIDs: selectedCheckpoints)
                    try Task.checkCancellation(); exportPlan = plan
                }
            } catch { issue = error is CancellationError ? "Preview stopped." : error.localizedDescription }
        }
    }
    private func transfer() {
        guard !busy, hasPreview else { return }
        localBusy = true; issue = nil
        work = Task {
            defer { transferStarted = false; localBusy = false; work = nil }
            do {
                try Task.checkCancellation()
                if let plan = importPlan, isImport {
                    transferStarted = true
                    importResult = try await model.importArtifactArchive(plan)
                } else if !isImport, let plan = exportPlan {
                    guard let destination = await NativeArtifactPanels.saveArchive(suggestedName: agentName), !Task.isCancelled else { return }
                    transferStarted = true
                    try await model.exportArtifactArchive(plan, to: destination)
                    exportedURL = destination
                }
            } catch { issue = error is CancellationError ? "Transfer stopped. Any interrupted transfer can be reviewed in Settings." : error.localizedDescription }
        }
    }
    private func cancel() {
        guard busy, !cancelRequested else { return }
        cancelRequested = true
        let pending = work
        Task {
            if transferStarted { await model.cancelArtifactTransfer() }
            else { pending?.cancel() }
            await pending?.value
            cancelRequested = false
        }
    }
    private func isDependency(_ item: ArtifactTransferItem) -> Bool {
        guard let id = UUID(uuidString: item.identity) else { return true }
        switch item.kind {
        case .recording: return !recordingIDs.contains(id)
        case .checkpoint: return !checkpointIDs.contains(id)
        default: return true
        }
    }
    private func kindName(_ kind: ArtifactTransferKind) -> String {
        switch kind {
        case .recording: "Demonstration"
        case .checkpoint: "Model"
        case .dataset: "Dataset"
        case .runConfiguration: "Training settings"
        case .desktopConfiguration: "Desktop settings"
        case .rewardAsset: "Reward reference"
        }
    }
    private func bytes(_ value: UInt64) -> String { ByteCountFormatter.string(fromByteCount: Int64(clamping: value), countStyle: .file) }
}
