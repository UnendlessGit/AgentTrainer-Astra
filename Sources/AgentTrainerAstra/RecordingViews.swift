import SwiftUI
import AstraCore
import AstraPlatform

struct RecordingBanner: View {
    @Bindable var model: WorkspaceModel
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "record.circle.fill").foregroundStyle(.red)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.recordingStopping ? "Saving recording…" : model.recordingCountdown.map { "Recording begins in \($0)…" }
                     ?? (model.recordingStarting ? "Starting recording…" : "Recording demonstration"))
                    .fontWeight(.medium)
                if let progress = model.recordingProgress {
                    Text("\(progress.elapsedSeconds, specifier: "%.1f") s · \(progress.frames) frames · \(ByteCountFormatter.string(fromByteCount: Int64(progress.bytes), countStyle: .file))")
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
            }
            Spacer()
            Button("Stop Recording", systemImage: "stop.fill") { Task { await model.stopRecording() } }
                .disabled(model.recordingStopping)
        }.padding(14).background(.red.opacity(0.08))
        .accessibilityElement(children: .contain)
    }
}

struct RecordingList: View {
    let recordings: [RecordingManifest]
    var onOpen: (RecordingManifest) -> Void = { _ in }
    @State private var selection: UUID?
    var body: some View {
        if recordings.isEmpty {
            ContentUnavailableView("No demonstrations yet", systemImage: "record.circle",
                                   description: Text("Record a session in an environment to teach an agent how to act."))
        } else {
            VStack(alignment: .trailing, spacing: 8) {
            Table(recordings, selection: $selection) {
                TableColumn("Recording") { recording in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(recording.name).fontWeight(.medium)
                        Text(recording.environment.name).font(.caption).foregroundStyle(.secondary)
                    }.padding(.vertical, 4)
                }
                TableColumn("Duration") { Text($0.durationSeconds.formatted(.number.precision(.fractionLength(1))) + " s").monospacedDigit() }
                    .width(min: 70, ideal: 85, max: 120)
                TableColumn("Frames") { Text($0.frameCount.formatted()).monospacedDigit() }
                    .width(min: 60, ideal: 75, max: 100)
                TableColumn("Status") { recording in
                    Text(status(recording.status)).foregroundStyle(recording.status == .complete ? Color.secondary : Color.orange)
                        .help(recording.issue ?? status(recording.status))
                }.width(min: 90, ideal: 110, max: 160)
            }
            .contextMenu(forSelectionType: UUID.self) { identifiers in
                if let identifier = identifiers.first, let recording = recordings.first(where: { $0.id == identifier }) {
                    Button("Review Recording") { onOpen(recording) }.disabled(recording.status == .recording)
                }
            } primaryAction: { identifiers in
                if let identifier = identifiers.first, let recording = recordings.first(where: { $0.id == identifier }), recording.status != .recording { onOpen(recording) }
            }
            Button("Review Recording", systemImage: "play.rectangle") {
                if let recording = recordings.first(where: { $0.id == selection }) { onOpen(recording) }
            }.disabled(!recordings.contains { $0.id == selection && $0.status != .recording })
            }
        }
    }
    private func status(_ value: RecordingStatus) -> String {
        switch value {
        case .recording: "Recording"
        case .complete: "Saved"
        case .interrupted: "Interrupted"
        case .failed: "Needs recovery"
        }
    }
}

struct RecorderSheet: View {
    @Bindable var model: WorkspaceModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = "Demonstration"
    @State private var selectedSourceID: String?
    @State private var fps = 30
    @State private var refreshing = true

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Record a demonstration").font(.title2.weight(.semibold))
            Text("Choose what the agent can observe. Screen frames and your controls stay together in a lossless recording.")
                .foregroundStyle(.secondary)
            if !model.permissions.screenRecording || !model.permissions.inputMonitoring {
                VStack(alignment: .leading, spacing: 12) {
                    permissionRow("Screen Recording", allowed: model.permissions.screenRecording, pane: .screenRecording)
                    permissionRow("Input Monitoring", allowed: model.permissions.inputMonitoring, pane: .inputMonitoring)
                    Button("Refresh Permissions") { Task { await refresh() } }
                    Text("After changing access, macOS may ask you to relaunch Astra.").font(.caption).foregroundStyle(.secondary)
                }.padding(14).background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
            }
            Form {
                TextField("Name", text: $name)
                Picker("Environment", selection: $selectedSourceID) {
                    Text(refreshing ? "Loading sources…" : "Choose a display or window").tag(nil as String?)
                    ForEach(model.sources) { source in
                        Text("\(source.name) · \(source.pixelWidth) × \(source.pixelHeight)").tag(Optional(source.id))
                    }
                }.disabled(refreshing || !model.permissions.screenRecording)
                Picker("Capture rate", selection: $fps) {
                    ForEach([15, 30, 60], id: \.self) { Text("\($0) fps").tag($0) }
                }
            }
            if let source {
                let rawRate = Int64(source.pixelWidth) * Int64(source.pixelHeight) * 4 * Int64(fps)
                Text("Native resolution · Lossless color. Before compression, this source produces \(ByteCountFormatter.string(fromByteCount: rawRate, countStyle: .file))/s. Actual storage depends on the content.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("Refresh Sources") { Task { await refresh() } }.disabled(refreshing)
                Spacer()
                Button("Cancel") {
                    Task {
                        if model.recordingStarting { await model.stopRecording(issue: "Recording cancelled during startup.") }
                        dismiss()
                    }
                }.keyboardShortcut(.cancelAction)
                Button("Start Recording") {
                    guard let source else { return }
                    Task { await model.startRecording(source: source, name: name, fps: fps) }
                }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(source == nil || !model.permissions.screenRecording || !model.permissions.inputMonitoring
                              || model.recordingStarting || (try? DocumentNames.validated(name)) == nil)
            }
        }.padding(24).frame(width: 590)
            .interactiveDismissDisabled(model.recordingStarting)
            .task { await refresh() }
    }

    private var source: CaptureSource? { model.sources.first { $0.id == selectedSourceID } }
    private func refresh() async {
        refreshing = true
        await model.refreshPermissionsAndSources()
        if !model.sources.contains(where: { $0.id == selectedSourceID }) { selectedSourceID = nil }
        refreshing = false
    }
    private func permissionRow(_ title: String, allowed: Bool, pane: PrivacyPane) -> some View {
        HStack {
            Image(systemName: allowed ? "checkmark.circle.fill" : "lock.circle").foregroundStyle(allowed ? .green : .secondary)
            Text(title)
            Spacer()
            if allowed { Text("Allowed").foregroundStyle(.secondary) }
            else { Button("Open Settings") { pane.open() } }
        }
    }
}
