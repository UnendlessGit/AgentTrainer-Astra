import AppKit
import AstraCore
import Observation
import AstraPlatform

enum WorkspaceDestination: Hashable { case agent(UUID), library, activity }
enum AgentSection: String, CaseIterable, Identifiable {
    case demonstrations = "Demonstrations", training = "Training", evaluation = "Evaluation", run = "Run"
    var id: String { rawValue }
}

@MainActor @Observable final class WorkspaceModel {
    private(set) var agents: [AgentDocument] = []
    private(set) var environments: [EnvironmentDocument] = []
    private(set) var issues: [LibraryIssue] = []
    private(set) var recordings: [RecordingManifest] = []
    private(set) var sources: [CaptureSource] = []
    private(set) var permissions = PermissionSnapshot.current()
    private(set) var recordingProgress: RecordingProgress?
    private(set) var recordingStarting = false
    private(set) var recordingCountdown: Int?
    private(set) var recordingStopping = false
    var showingRecorder = false
    private(set) var loading = true
    private(set) var saving = false
    var destination: WorkspaceDestination? = .library
    var section: AgentSection = .demonstrations
    var errorMessage: String?
    var showingNewAgent = false
    private var store: LibraryStore?
    private var started = false
    private var recorder: RecordingSession?
    private var recordingAgentID: UUID?
    private var activeRecordingID: UUID?
    private(set) var recordingLinks: [UUID: Set<UUID>] = [:]

    var selectedAgent: AgentDocument? {
        guard case .agent(let id) = destination else { return nil }
        return agents.first { $0.id == id }
    }

    var supportRoot: URL {
        if let path = ProcessInfo.processInfo.environment["ASTRA_WORKSPACE_ROOT"], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AgentTrainer Astra", isDirectory: true)
    }

    func start() async {
        guard !started else { return }
        started = true
        do {
            let root = supportRoot
            store = try await Task.detached { try LibraryStore(root: root) }.value
            _ = try await store?.recoverInterruptedRecordings()
            try await refresh()
            if let first = agents.first { destination = .agent(first.id) }
        } catch { errorMessage = error.localizedDescription }
        loading = false
    }

    func createAgent(name: String) async {
        guard let store else { return }
        saving = true
        defer { saving = false }
        do {
            let agent = try AgentDocument(name: name).validated()
            try await store.save(agent)
            try await refresh()
            destination = .agent(agent.id); section = .demonstrations
            showingNewAgent = false
        } catch { errorMessage = error.localizedDescription }
    }

    func saveAgent(_ document: AgentDocument) async {
        guard let store else { return }
        saving = true
        defer { saving = false }
        do {
            var value = document; value.modifiedAt = Date()
            try await store.save(value)
            try await refresh()
        } catch { errorMessage = error.localizedDescription }
    }

    func refreshPermissionsAndSources() async {
        permissions = PermissionSnapshot.current()
        guard permissions.screenRecording else { sources = []; return }
        do { sources = try await CaptureDiscovery.sources() }
        catch { errorMessage = error.localizedDescription }
    }

    func startRecording(source: CaptureSource, name: String, fps: Int) async {
        guard let store, !recordingStarting, !recordingStopping, recorder == nil else { return }
        permissions = PermissionSnapshot.current()
        guard permissions.screenRecording && permissions.inputMonitoring else {
            errorMessage = "Screen Recording and Input Monitoring permissions are required to record a demonstration."
            return
        }
        recordingStarting = true
        defer { recordingStarting = false; recordingCountdown = nil }
        let agentID = selectedAgent?.id
        let environment = EnvironmentDocument(name: String(source.name.prefix(160)), kind: source.kind, displayID: source.displayID,
                                              windowID: source.windowID, applicationBundleID: source.applicationBundleID, captureFPS: fps)
        let manifest = RecordingManifest(name: name, environment: environment, recordedForAgentID: agentID)
        activeRecordingID = manifest.id
        do {
            let directory = store.recordingDirectory(id: manifest.id)
            let progressHandler: @Sendable (RecordingProgress) -> Void = { [weak self] progress in
                Task { @MainActor [weak self] in
                    guard self?.activeRecordingID == progress.id else { return }
                    self?.recordingProgress = progress
                }
            }
            let faultHandler: @Sendable (String) -> Void = { [weak self] message in
                Task { @MainActor [weak self] in
                    guard self?.activeRecordingID == manifest.id else { return }
                    await self?.stopRecording(issue: message)
                }
            }
            let created = try await Task.detached {
                try RecordingSession(directory: directory, manifest: manifest, source: source,
                                     onProgress: progressHandler, onFault: faultHandler)
            }.value
            guard activeRecordingID == manifest.id else {
                let cancelled = try await created.stop(issue: "Cancelled before capture started.")
                try await store.saveRecording(cancelled, linkTo: agentID)
                try await refresh()
                return
            }
            recorder = created; recordingAgentID = agentID
            try await store.save(environment)
            if let agentID, var agent = agents.first(where: { $0.id == agentID }) {
                agent.environmentID = environment.id; agent.modifiedAt = Date()
                try await store.save(agent)
            }
            try await store.saveRecording(manifest, linkTo: agentID)
            try await refresh()
            showingRecorder = false
            // Dismiss the operator sheet before enabling input observation.
            // The visible countdown lets a window settle and provides a stop
            // path while the selected environment is brought to the front.
            for remaining in (1...3).reversed() {
                guard activeRecordingID == manifest.id, !recordingStopping else { return }
                recordingCountdown = remaining
                if remaining == 2, let pid = source.applicationPID {
                    guard let application = NSRunningApplication(processIdentifier: pid),
                          !application.isTerminated,
                          source.applicationLaunchDate == nil || source.applicationLaunchDate == application.launchDate else {
                        throw AstraError("recording.targetChanged", "The selected application closed or restarted. Choose it again.")
                    }
                    application.activate()
                }
                try await Task.sleep(for: .seconds(1))
            }
            recordingCountdown = nil
            guard activeRecordingID == manifest.id, !recordingStopping else { return }
            try await created.start()
        } catch {
            errorMessage = error.localizedDescription
            if recorder != nil { await stopRecording(issue: error.localizedDescription) }
        }
    }

    func stopRecording(issue: String? = nil) async {
        guard let store, !recordingStopping else { return }
        guard let recorder else { activeRecordingID = nil; return }
        recordingStopping = true
        defer { recordingStopping = false }
        do {
            let manifest = try await recorder.stop(issue: issue)
            try await store.saveRecording(manifest, linkTo: recordingAgentID)
            self.recorder = nil; recordingProgress = nil; recordingAgentID = nil; activeRecordingID = nil
            try await refresh()
            if let issue { errorMessage = issue }
        } catch {
            errorMessage = "Recording data was preserved, but finalization needs recovery: \(error.localizedDescription)"
            self.recorder = nil; recordingProgress = nil; recordingAgentID = nil; activeRecordingID = nil
            try? await refresh()
        }
    }

    var isRecording: Bool { recorder != nil }

    func duplicateSelectedAgent() async {
        guard let store, let selectedAgent else { return }
        do {
            let copy = try await store.duplicateAgent(selectedAgent)
            try await refresh(); destination = .agent(copy.id)
        } catch { errorMessage = error.localizedDescription }
    }

    private func refresh() async throws {
        guard let store else { return }
        let snapshot = try await store.snapshot()
        agents = snapshot.agents; environments = snapshot.environments; recordings = snapshot.recordings; issues = snapshot.issues
        var links: [UUID: Set<UUID>] = [:]
        for agent in agents { links[agent.id] = try await store.recordingIDs(for: agent.id) }
        recordingLinks = links
    }
}
