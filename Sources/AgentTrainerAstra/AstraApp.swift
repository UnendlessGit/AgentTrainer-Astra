import SwiftUI
import AppKit
import AstraCore
import AstraPlatform

@main struct AgentTrainerAstraApp: App {
    @State private var model = WorkspaceModel()
    @NSApplicationDelegateAdaptor(AstraAppDelegate.self) private var appDelegate

    init() { NSApplication.shared.setActivationPolicy(.regular) }

    var body: some Scene {
        Window("AgentTrainer Astra", id: "workspace") {
            WorkspaceView(model: model)
                .frame(minWidth: 860, minHeight: 580)
                .task { appDelegate.model = model; await model.start() }
        }
        .defaultSize(width: 1120, height: 760)
        .commands {
            CommandGroup(after: .newItem) {
                Button("New Agent…") { model.showingNewAgent = true }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
                    .disabled(model.loading)
                Button("Duplicate Agent") { Task { await model.duplicateSelectedAgent() } }
                    .disabled(model.selectedAgent == nil)
            }
        }
        Settings { SettingsView(root: model.supportRoot) }
        MenuBarExtra("AgentTrainer Astra", systemImage: model.isRecording ? "record.circle.fill" : "cpu") {
            StatusMenu(model: model)
        }
    }
}

@MainActor private final class AstraAppDelegate: NSObject, NSApplicationDelegate {
    weak var model: WorkspaceModel?
    private var terminating = false
    private var terminationSignal: DispatchSourceSignal?
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Development relaunch and normal process termination must run the
        // same save/disarm path as Quit, rather than interrupting a checkpoint.
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        source.setEventHandler { NSApplication.shared.terminate(nil) }
        terminationSignal = source; source.resume()
    }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model, model.isRecording || model.recordingStarting || model.recordingStopping || model.isLearning || model.isRunningAgent else { return .terminateNow }
        guard !terminating else { return .terminateLater }
        terminating = true
        Task {
            let allowed = await model.prepareForTermination()
            if !allowed { terminating = false }
            sender.reply(toApplicationShouldTerminate: allowed)
        }
        return .terminateLater
    }
}

private struct StatusMenu: View {
    @Bindable var model: WorkspaceModel
    @Environment(\.openWindow) private var openWindow
    var body: some View {
        Text(model.isRecording ? "Recording demonstration" : "No active recording")
        if model.isRecording || model.recordingStarting {
            Button("Stop Recording") { Task { await model.stopRecording() } }.disabled(model.recordingStopping)
        }
        if let desktop = model.desktopLearning, desktop.isBusy {
            Text(desktop.phase)
            Button("Stop Desktop Learning") { desktop.requestStop() }.disabled(desktop.isStopping)
        } else if let learning = model.learning, learning.isBusy {
            Text(learning.phase)
            Button("Stop Learning") { Task { await learning.requestStop() } }.disabled(learning.isStopping)
        }
        if let inference = model.inference, inference.isBusy {
            Text(inference.phase)
            Button("Stop Agent") { Task { await inference.stopAndWait() } }.disabled(inference.isStopping)
        }
        Divider()
        Button("Show AgentTrainer Astra") { openWindow(id: "workspace"); NSApp.activate(ignoringOtherApps: true) }
        Button("Quit AgentTrainer Astra") { NSApp.terminate(nil) }
    }
}

struct WorkspaceView: View {
    @Bindable var model: WorkspaceModel

    var body: some View {
        NavigationSplitView {
            List(selection: $model.destination) {
                Section("Agents") {
                    ForEach(model.agents) { agent in
                        Label(agent.name, systemImage: "cpu")
                            .tag(WorkspaceDestination.agent(agent.id))
                            .help(agent.name)
                    }
                }
                Section {
                    Label("Library", systemImage: "square.stack.3d.up")
                        .tag(WorkspaceDestination.library)
                    Label("Activity", systemImage: "clock.arrow.circlepath")
                        .tag(WorkspaceDestination.activity)
                }
            }
            .navigationSplitViewColumnWidth(min: 190, ideal: 230, max: 320)
            .safeAreaInset(edge: .bottom) {
                Button { model.showingNewAgent = true } label: {
                    Label("New Agent", systemImage: "plus")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain).padding(16).disabled(model.loading)
            }
        } detail: {
            if model.loading {
                ProgressView("Opening your workspace…").frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    if model.isRecording || model.recordingStarting { RecordingBanner(model: model) }
                    if let desktop = model.desktopLearning, desktop.isBusy { DesktopLearningBanner(host: desktop) }
                    else if let learning = model.learning, learning.isBusy { LearningBanner(learning: learning) }
                    if let inference = model.inference, inference.isBusy { InferenceBanner(coordinator: inference) }
                    if !model.issues.isEmpty { recoveryBanner }
                    switch model.destination {
                    case .agent:
                        if let agent = model.selectedAgent { AgentWorkspace(agent: agent, model: model) }
                        else { welcome }
                    case .library: LibraryOverview(model: model)
                    case .activity:
                        ScrollView { LearningRunList(runs: model.learningRuns, model: model).padding(28) }
                    case nil: welcome
                    }
                }
            }
        }
        .navigationTitle(model.selectedAgent?.name ?? "AgentTrainer Astra")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button { model.showingNewAgent = true } label: { Label("New Agent", systemImage: "plus") }
                    .help("Create an agent").disabled(model.loading)
            }
        }
        .sheet(isPresented: $model.showingNewAgent) { NewAgentSheet(model: model) }
        .sheet(isPresented: $model.showingRecorder) { RecorderSheet(model: model) }
        .sheet(item: $model.recordingLinkRequest) { RecordingLinkSheet(model: model, request: $0) }
        .sheet(item: $model.recordingToInspect) { recording in
            RecordingInspector(recording: recording, directory: model.supportRoot.appendingPathComponent("Recordings")
                .appendingPathComponent(recording.id.uuidString + ".astrarecord"), workspace: model, agentID: model.recordingInspectionAgentID)
        }
        .sheet(item: Binding(get: { model.desktopLearning?.reviewPresentation }, set: { value in
            if value == nil { model.desktopLearning?.cancelFeedbackReview() }
        })) { review in
            FeedbackReviewView(model: review.model) { review.finish($0) }
                .frame(idealWidth: 960, idealHeight: 760)
                .onDisappear { Task { await review.cancelAndJoin() } }
        }
        .alert("Workspace needs attention", isPresented: Binding(
            get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } }
        )) { Button("OK") { model.errorMessage = nil } } message: { Text(model.errorMessage ?? "") }
    }

    private var welcome: some View {
        ContentUnavailableView {
            Label("Your agents, learning locally", systemImage: "cpu")
        } description: { Text("Create an agent to organize demonstrations, training, and evaluation.") }
        actions: { Button("Create Agent…") { model.showingNewAgent = true }.buttonStyle(.borderedProminent) }
    }

    private var recoveryBanner: some View {
        let review = model.pendingControlHistory.first
        return HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
            VStack(alignment: .leading, spacing: 5) {
                Text(review == nil ? "\(model.issues.count) workspace items need attention" : "Review previous control cleanup")
                    .fontWeight(.medium)
                if let review { Text(review.title).font(.caption).foregroundStyle(.secondary) }
                Text(review?.message ?? model.issues.first?.message ?? "")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                HStack {
                    if let review {
                        Button("I’ve released the held controls") {
                            Task {
                                do { try await model.acknowledgeControlHistory(review) }
                                catch { model.errorMessage = error.localizedDescription }
                            }
                        }
                        .disabled(model.historyReviewUnavailableReason(review) != nil)
                        .help(model.historyReviewUnavailableReason(review) ?? "Acknowledge this exact history without changing its native cleanup result.")
                    }
                    Button("Refresh History") { Task { await model.refreshControlHistory() } }
                        .disabled(model.isRunningAgent || model.isLearning || model.isRecording)
                    if model.controlHistoryBusy { ProgressView().controlSize(.small) }
                }
                if model.pendingControlHistory.count > 1 {
                    Text("\(model.pendingControlHistory.count - 1) other control histories also need review.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 8)
        }
        .padding(12).background(.yellow.opacity(0.12))
        .accessibilityElement(children: .contain)
    }

}

private struct AgentWorkspace: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(agent.name).font(.largeTitle.weight(.semibold)).textSelection(.enabled).lineLimit(2).help(agent.name)
                    Text(environmentName).foregroundStyle(.secondary)
                    Picker("Checkpoint", selection: Binding(get: { agent.selectedCheckpointID }, set: { selected in
                        Task { await model.selectCheckpoint(selected, agentID: agent.id) }
                    })) {
                        Text("No checkpoint selected").tag(nil as UUID?)
                        ForEach(model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true }) { checkpoint in
                            Text(checkpoint.name).tag(Optional(checkpoint.id))
                        }
                    }.frame(maxWidth: 400, alignment: .leading).disabled(model.checkpointLinks[agent.id]?.isEmpty != false)
                }
                Spacer()
                Menu {
                    Button("Duplicate Agent") { Task { await model.duplicateSelectedAgent() } }
                } label: { Image(systemName: "ellipsis.circle").imageScale(.large) }
                .menuStyle(.borderlessButton).fixedSize().help("Agent actions")
            }
            Picker("Agent workspace", selection: $model.section) {
                ForEach(AgentSection.allCases) { section in Text(section.rawValue).tag(section) }
            }.pickerStyle(.segmented).labelsHidden().accessibilityLabel("Agent workspace")
            Group {
                switch model.section {
                case .demonstrations:
                    VStack(alignment: .leading, spacing: 16) {
                        HStack {
                            Text("Teach by demonstration").font(.headline)
                            Spacer()
                            Button("Add from Library…", systemImage: "link") { model.recordingLinkRequest = .init(agentID: agent.id) }
                                .disabled(model.saving)
                            Button("Record…", systemImage: "record.circle") { model.showingRecorder = true }
                                .buttonStyle(.borderedProminent)
                                .disabled(model.isRecording || model.recordingStarting || model.recordingStopping)
                        }
                        RecordingList(recordings: model.recordings.filter { model.recordingLinks[agent.id]?.contains($0.id) == true },
                            onOpen: { model.inspectRecording($0, for: agent.id) },
                            onRemove: { recording in Task { await model.unlinkRecording(recording.id, from: agent.id) } },
                            trainingSummaries: Dictionary(uniqueKeysWithValues: model.recordings.map { ($0.id, model.trainingSelectionSummary(recording: $0, agentID: agent.id)) }))
                    }
                case .training:
                    LearningTrainingView(agent: agent, model: model).id(agent.id)
                case .evaluation:
                    LearningEvaluationView(agent: agent, model: model).id(agent.id)
                case .run:
                    RunView(coordinator: model.inference,
                            checkpoints: model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true },
                            sources: model.sources, refreshingSources: model.refreshingSources, sourceIssue: model.sourceIssue,
                            unavailableReason: model.inferenceUnavailableReason, contextSizes: model.checkpointContextSizes,
                            selectedCheckpointID: agent.selectedCheckpointID,
                            refreshSources: { Task { await model.refreshPermissionsAndSources() } },
                            selectCheckpoint: { id in Task { await model.selectCheckpoint(id, agentID: agent.id) } },
                            start: { checkpoint, source, options in model.startInference(agent: agent, checkpoint: checkpoint, source: source, options: options) },
                            acknowledgeCleanup: { try await model.acknowledgeInferenceCleanup() })
                        .task(id: agent.selectedCheckpointID) { await model.inspectCheckpointContexts(agent.selectedCheckpointID, agentID: agent.id) }
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(28)
    }

    private var environmentName: String {
        model.environments.first(where: { $0.id == agent.environmentID })?.name ?? "Environment not selected"
    }
}

private struct LibraryOverview: View {
    @Bindable var model: WorkspaceModel
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Library").font(.largeTitle.weight(.semibold))
            Text("Recordings and environments can be shared by your agents.").foregroundStyle(.secondary)
            if model.agents.isEmpty {
                ContentUnavailableView {
                    Label("Start with an agent", systemImage: "cpu")
                } description: { Text("Keep each experiment's demonstrations, checkpoints, and results together.") }
                actions: { Button("Create Agent…") { model.showingNewAgent = true }.buttonStyle(.borderedProminent) }
            } else {
                HStack {
                    Spacer()
                    Button("Record…", systemImage: "record.circle") { model.showingRecorder = true }
                        .disabled(model.isRecording || model.recordingStarting)
                }
                RecordingList(recordings: model.recordings, onOpen: { model.inspectRecording($0) },
                              onLink: { model.recordingLinkRequest = .init(recordingID: $0.id) })
            }
        }.padding(28).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

private struct NewAgentSheet: View {
    @Bindable var model: WorkspaceModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @FocusState private var focused: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("New Agent").font(.title2.weight(.semibold))
            Text("Give this learning project a name.").foregroundStyle(.secondary)
            TextField("Agent name", text: $name).textFieldStyle(.roundedBorder)
                .focused($focused).onSubmit { create() }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Create Agent") { create() }.keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(model.saving || (try? DocumentNames.validated(name)) == nil)
            }
        }.padding(24).frame(width: 390).onAppear { focused = true }
    }
    private func create() {
        guard !model.saving, (try? DocumentNames.validated(name)) != nil else { return }
        Task { await model.createAgent(name: name) }
    }
}

private struct SettingsView: View {
    let root: URL
    var body: some View {
        Form {
            Section("Local workspace") {
                Text(root.path).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([root]) }
            }
            Section("About") {
                LabeledContent("Application", value: "AgentTrainer Astra")
                LabeledContent("Development build", value: "0.1.0")
                Text("Recordings, training, and inference stay on your Mac.").foregroundStyle(.secondary)
            }
        }.formStyle(.grouped).frame(width: 520, height: 300)
    }
}
