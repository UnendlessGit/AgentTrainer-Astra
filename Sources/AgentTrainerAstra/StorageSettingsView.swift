import SwiftUI
import AppKit
import AstraCore

struct StorageSettingsView: View {
    @Bindable var model: WorkspaceModel
    @State private var plan: StorageMigrationPlan?
    @State private var preparing = false
    @State private var previewWork: Task<Void, Never>?
    @State private var issue: String?
    @State private var result: String?
    @State private var destinationName = "Astra Recordings"
    @State private var kind = ArtifactStorageKind.recordings
    @State private var parent: URL?

    var body: some View {
        Form {
            if preparing {
                Section("Checking storage") {
                    ProgressView("Preparing a verified file inventory…")
                    Button("Cancel Preview") { previewWork?.cancel() }
                }
            }
            if model.artifactTransferBusy {
                Section("Transferring artifacts") {
                    ArtifactTransferProgressView(progress: model.artifactTransferProgress)
                    Button("Cancel Transfer") { Task { await model.cancelArtifactTransfer() } }
                }
            }
            Section("Catalog and run history") {
                Text(model.supportRoot.path).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([model.supportRoot]) }
            }
            Section("Recordings and models") {
                location("Recordings", url: model.storageLayout.recordingsRoot)
                location("Models", url: model.storageLayout.modelsRoot)
                Button(model.storageRoutingNeedsReload ? "Reload Storage Routing" : "Check Connected Drives") { Task { await model.refreshStorageAvailability() } }
                    .disabled(model.artifactTransferBusy || model.saving)
                Text("Move recordings and models independently, including to an external drive. Each workflow uses the verified location selected here.")
                    .font(.caption).foregroundStyle(.secondary)
                Picker("Move", selection: $kind) {
                    Text("Recordings").tag(ArtifactStorageKind.recordings)
                    Text("Models").tag(ArtifactStorageKind.models)
                }.onChange(of: kind) { _, value in
                    destinationName = value == .recordings ? "Astra Recordings" : "Astra Models"; plan = nil
                }
                HStack {
                    Button("Choose Destination Folder…") {
                        Task { if let selected = await NativeArtifactPanels.chooseFolder() { parent = selected; plan = nil } }
                    }
                    if let parent { Text(parent.lastPathComponent).foregroundStyle(.secondary).lineLimit(1).help(parent.path) }
                }
                TextField("New folder name", text: $destinationName).onChange(of: destinationName) { _, _ in plan = nil }
                Button(preparing ? "Preparing Preview…" : "Preview Move") { preview() }
                    .disabled(parent == nil || !validName || preparing || model.artifactUnavailableReason != nil)
                if let plan {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("\(plan.files.count.formatted()) files · \(ByteCountFormatter.string(fromByteCount: Int64(clamping: plan.totalBytes), countStyle: .file))")
                            .fontWeight(.medium)
                        Text(plan.destination.path).font(.caption).textSelection(.enabled)
                        Text("Astra copies and verifies the files, then switches to the new folder. The original copy remains available until you remove it.")
                            .font(.caption).foregroundStyle(.secondary)
                        Button("Copy, Verify and Use New Folder") {
                            Task {
                                issue = nil; result = nil
                                do {
                                    try await model.applyStorageMigration(plan)
                                    self.plan = nil; parent = nil
                                    result = "The new storage folder is active. The original copy was retained."
                                } catch is CancellationError { issue = "Transfer cancelled. Any recoverable transfer is listed below." }
                                catch { issue = error.localizedDescription }
                            }
                        }.buttonStyle(.borderedProminent).disabled(model.artifactUnavailableReason != nil)
                    }.padding(.vertical, 8)
                }
                if let reason = model.artifactUnavailableReason, !model.artifactTransferBusy { Text(reason).font(.caption).foregroundStyle(.secondary) }
            }.disabled(model.artifactTransferBusy || preparing)
            if !model.pendingArtifactTransfers.isEmpty {
                Section("Interrupted transfers") {
                    ForEach(model.pendingArtifactTransfers) { record in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(record.operation.capitalized).fontWeight(.medium)
                            if let migration = record.migration { Text(migration.destination.path).font(.caption).textSelection(.enabled) }
                            if let importing = record.importPlan { Text(importing.source.path).font(.caption).textSelection(.enabled) }
                            if let exporting = record.exportDestination { Text(exporting.path).font(.caption).textSelection(.enabled) }
                            if let problem = record.issue { Text(problem).font(.caption).foregroundStyle(.secondary) }
                            Button("Retry Verified Transfer") {
                                Task {
                                    issue = nil
                                    do { try await model.retryArtifactTransfer(record); result = "The transfer completed." }
                                    catch { issue = error.localizedDescription }
                                }
                            }.disabled(model.artifactUnavailableReason != nil)
                        }.padding(.vertical, 4)
                    }
                }
            }
            if let issue { Section { AttentionLabel(message: issue) } }
            if let result { Section { Text(result).foregroundStyle(.secondary) } }
            Section("About") {
                LabeledContent("Application", value: "AgentTrainer Astra")
                LabeledContent("Development build", value: "0.1.0")
                Text("Recordings, training, and inference stay on your Mac.").foregroundStyle(.secondary)
            }
        }.formStyle(.grouped).frame(width: 640, height: 620)
            .onDisappear { previewWork?.cancel() }
    }

    private var validName: Bool {
        !destinationName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && destinationName.utf8.count <= 200
            && ![".", ".."].contains(destinationName) && !destinationName.contains("/") && !destinationName.contains(":")
            && !destinationName.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains)
    }
    private func preview() {
        guard let parent, validName else { return }
        let destination = parent.appendingPathComponent(destinationName, isDirectory: true), chosen = kind
        preparing = true; issue = nil; result = nil; plan = nil
        previewWork = Task {
            defer { preparing = false; previewWork = nil }
            do {
                let prepared = try await model.previewStorageMigration(kind: chosen, destination: destination)
                try Task.checkCancellation(); plan = prepared
            } catch is CancellationError { issue = nil }
            catch { issue = error.localizedDescription }
        }
    }
    private func location(_ title: String, url: URL) -> some View {
        LabeledContent(title) {
            Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                Text(url.path).font(.caption).lineLimit(2).multilineTextAlignment(.trailing)
            }.buttonStyle(.plain).help("Show storage folder in Finder")
        }
    }
}

struct ArtifactTransferProgressView: View {
    let progress: ArtifactTransferProgress?
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let progress {
                Text(progress.phase).fontWeight(.medium)
                ProgressView(value: Double(progress.completedBytes), total: Double(max(1, progress.totalBytes)))
                Text(progress.name).font(.caption).lineLimit(1).truncationMode(.middle)
                Text("\(ByteCountFormatter.string(fromByteCount: Int64(clamping: progress.completedBytes), countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: Int64(clamping: progress.totalBytes), countStyle: .file))")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            } else { ProgressView("Preparing transfer…") }
        }.accessibilityElement(children: .combine)
    }
}

struct ArtifactTransferBanner: View {
    @Bindable var model: WorkspaceModel
    var body: some View {
        HStack(spacing: 12) {
            ProgressView().controlSize(.small)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.artifactTransferProgress?.phase ?? "Transferring artifacts…").font(.callout.weight(.medium))
                if let progress = model.artifactTransferProgress { Text(progress.name).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
            }
            Spacer()
            Button("Cancel Transfer") { Task { await model.cancelArtifactTransfer() } }
        }.padding(.horizontal, 20).padding(.vertical, 12).background(.quaternary)
    }
}
