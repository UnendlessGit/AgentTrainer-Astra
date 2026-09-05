import AppKit
import AstraCore
import Observation

enum WorkspaceDestination: Hashable { case agent(UUID), library, activity }
enum AgentSection: String, CaseIterable, Identifiable {
    case demonstrations = "Demonstrations", training = "Training", evaluation = "Evaluation", run = "Run"
    var id: String { rawValue }
}

@MainActor @Observable final class WorkspaceModel {
    private(set) var agents: [AgentDocument] = []
    private(set) var environments: [EnvironmentDocument] = []
    private(set) var issues: [LibraryIssue] = []
    private(set) var loading = true
    private(set) var saving = false
    var destination: WorkspaceDestination? = .library
    var section: AgentSection = .demonstrations
    var errorMessage: String?
    var showingNewAgent = false
    private var store: LibraryStore?
    private var started = false

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
        agents = snapshot.agents; environments = snapshot.environments; issues = snapshot.issues
    }
}
