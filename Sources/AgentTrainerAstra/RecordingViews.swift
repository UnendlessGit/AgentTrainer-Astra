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
    var onLink: ((RecordingManifest) -> Void)?
    var onRemove: ((RecordingManifest) -> Void)?
    var trainingSummaries: [UUID: String]?
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
                if trainingSummaries == nil {
                    TableColumn("Frames") { Text($0.frameCount.formatted()).monospacedDigit() }
                        .width(min: 60, ideal: 75, max: 100)
                }
                if let trainingSummaries {
                    TableColumn("Training") { Text(trainingSummaries[$0.id] ?? "Review selection").font(.callout) }
                        .width(min: 110, ideal: 155, max: 180)
                }
                TableColumn("Status") { recording in
                    Text(status(recording.status)).foregroundStyle(recording.status == .complete ? Color.secondary : Color.orange)
                        .help(recording.issue ?? status(recording.status))
                }.width(min: 90, ideal: 110, max: 160)
            }
            .contextMenu(forSelectionType: UUID.self) { identifiers in
                if let identifier = identifiers.first, let recording = recordings.first(where: { $0.id == identifier }) {
                    Button("Review Recording") { onOpen(recording) }.disabled(recording.status == .recording)
                    if let onLink { Button("Use with Agent…") { onLink(recording) }.disabled((try? RecordingTrainingSelection.whole.resolved(for: recording)) == nil) }
                    if let onRemove { Button("Remove from Agent") { onRemove(recording) }.disabled(recording.status == .recording) }
                }
            } primaryAction: { identifiers in
                if let identifier = identifiers.first, let recording = recordings.first(where: { $0.id == identifier }), recording.status != .recording { onOpen(recording) }
            }
            HStack {
            if let onLink {
                Button("Use with Agent…", systemImage: "link") { if let recording = selected { onLink(recording) } }
                    .disabled(selected.flatMap { try? RecordingTrainingSelection.whole.resolved(for: $0) } == nil)
            }
            if let onRemove {
                Button("Remove from Agent", systemImage: "minus.circle") { if let recording = selected { onRemove(recording) } }
                    .disabled(selected == nil || selected?.status == .recording)
            }
            Spacer()
            Button("Review Recording", systemImage: "play.rectangle") {
                if let recording = recordings.first(where: { $0.id == selection }) { onOpen(recording) }
            }.disabled(!recordings.contains { $0.id == selection && $0.status != .recording })
            }
            }
        }
    }
    private var selected: RecordingManifest? { recordings.first { $0.id == selection } }
    private func status(_ value: RecordingStatus) -> String {
        switch value {
        case .recording: "Recording"
        case .complete: "Saved"
        case .interrupted: "Interrupted"
        case .failed: "Needs recovery"
        }
    }
}

struct RecordingLinkSheet: View {
    @Bindable var model: WorkspaceModel
    let request: RecordingLinkRequest
    @Environment(\.dismiss) private var dismiss
    @State private var selectedIDs: Set<UUID> = []
    @State private var selectedAgentID: UUID?
    @State private var search = ""
    @State private var linking = false

    private var agentID: UUID? { request.agentID ?? selectedAgentID }
    private var candidates: [RecordingManifest] {
        model.recordings.filter {
            (request.recordingID == nil || request.recordingID == $0.id) &&
            (try? RecordingTrainingSelection.whole.resolved(for: $0)) != nil &&
            (search.isEmpty || $0.name.localizedCaseInsensitiveContains(search))
        }
    }
    private var chosen: Set<UUID> {
        guard let agentID else { return [] }
        let requested = request.recordingID.map { Set([$0]) } ?? selectedIDs
        return requested.intersection(Set(candidates.map(\.id))).subtracting(model.recordingLinks[agentID] ?? [])
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(request.agentID == nil ? "Use recording with an agent" : "Add from Library").font(.title2.weight(.semibold))
            Text("Agents share the original recording and keep their own training intervals.").foregroundStyle(.secondary)
            if request.agentID == nil {
                Picker("Agent", selection: $selectedAgentID) {
                    Text("Choose an agent").tag(nil as UUID?)
                    ForEach(model.agents) { Text($0.name).tag(Optional($0.id)) }
                }
            } else {
                TextField("Find recordings", text: $search).textFieldStyle(.roundedBorder)
            }
            if candidates.isEmpty {
                ContentUnavailableView("No usable recordings", systemImage: "rectangle.stack", description: Text("Saved demonstrations with verified frames appear here."))
            } else {
                List(candidates) { recording in
                    let linked = agentID.map { model.recordingLinks[$0]?.contains(recording.id) == true } ?? false
                    HStack {
                        if request.recordingID == nil {
                            Toggle(recording.name, isOn: Binding(get: { linked || selectedIDs.contains(recording.id) }, set: { value in
                                if value { selectedIDs.insert(recording.id) } else { selectedIDs.remove(recording.id) }
                            })).toggleStyle(.checkbox).disabled(linked)
                        } else { Text(recording.name).fontWeight(.medium) }
                        Spacer()
                        Text(linked ? "Already linked" : "\(recording.durationSeconds.formatted(.number.precision(.fractionLength(1)))) s")
                            .font(.callout).foregroundStyle(.secondary)
                    }.padding(.vertical, 4)
                }.clipShape(RoundedRectangle(cornerRadius: 8))
            }
            HStack {
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Spacer()
                Button(linking ? "Adding…" : "Add \(chosen.count == 1 ? "Recording" : "\(chosen.count) Recordings")") {
                    guard let agentID else { return }
                    let identifiers = chosen
                    linking = true
                    Task { let saved = await model.linkRecordings(identifiers, to: agentID); linking = false; if saved { dismiss() } }
                }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(chosen.isEmpty || model.saving)
            }
        }.padding(24).frame(width: 600, height: 450).disabled(linking).interactiveDismissDisabled(linking || model.saving)
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
                    Text(refreshing ? "Loading sources…" : "Choose an environment").tag(nil as String?)
                    ForEach(model.sources) { source in
                        Text(source.bindings.map { "\(source.name) · \($0.count) surfaces" } ?? "\(source.name) · \(source.pixelWidth) × \(source.pixelHeight)").tag(Optional(source.id))
                    }
                }.disabled(refreshing || !model.permissions.screenRecording)
                Picker("Capture rate", selection: $fps) {
                    ForEach([15, 30, 60], id: \.self) { Text("\($0) fps").tag($0) }
                }
            }
            if let source {
                let rawRate = (source.bindings ?? [source]).reduce(Int64(0)) { $0 + Int64($1.pixelWidth) * Int64($1.pixelHeight) * 4 * Int64(fps) }
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
