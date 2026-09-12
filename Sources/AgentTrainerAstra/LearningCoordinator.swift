import Foundation
import Observation
import Darwin
import AstraCore
import AstraPlatform

enum DemonstrationSource: String, CaseIterable, Identifiable, Sendable {
    case recordings, practice
    var id: String { rawValue }
    var title: String { self == .recordings ? "My recordings" : "Practice demonstrations (generated)" }
}

struct BehaviorOptions: Sendable {
    var source: DemonstrationSource = .recordings
    var recordingIDs: Set<UUID> = []
    var initialCheckpointID: UUID?
    var resume = false
    var epochs = 20
    var learningRate = 0.0003
    var periodMS = 100
    var leadMS = 100
    var packetCapacity = 16
    var sequenceLength = 64
    var lanes = 2
    var seed = 0
    var keys: Set<Int> = []
    var buttons: Set<Int> = []
    var pointerMode = "absolute"
    var pointerEnabled = true
    var scrollEnabled = false
    var practiceTask = "pointing"
    var practiceDelayMS = 2000
    var practiceEpisodes = 12

    func validated() throws -> Self {
        if resume {
            guard initialCheckpointID != nil else { throw AstraError("behavioral.resume", "Choose a saved behavioral checkpoint to resume.") }
            return self
        }
        guard (1...100_000).contains(epochs), learningRate.isFinite, learningRate > 0, learningRate <= 1,
              [50, 100].contains(periodMS), (0...2000).contains(leadMS), [16, 32, 64].contains(packetCapacity),
              (1...512).contains(sequenceLength), (1...32).contains(lanes), (0...1_000_000_000).contains(seed),
              ["absolute", "relative"].contains(pointerMode), ["pointing", "delayed_memory"].contains(practiceTask),
              [2000, 8000, 30000].contains(practiceDelayMS), (1...1000).contains(practiceEpisodes),
              keys.allSatisfy({ (0...127).contains($0) }), buttons.allSatisfy({ (0...31).contains($0) }),
              (!resume || initialCheckpointID != nil), resume || source != .recordings || !recordingIDs.isEmpty else {
            throw AstraError("learning.options", "Choose demonstrations and valid training settings before starting.")
        }
        return self
    }

    var model: JSONValue {
        .object(["period_ms": .integer(Int64(periodMS)), "lead_ms": .integer(Int64(leadMS)),
                 "packet_capacity": .integer(Int64(packetCapacity))])
    }
    var training: JSONValue {
        .object(["epochs": .integer(Int64(epochs)), "learning_rate": .number(learningRate),
                 "sequence_length": .integer(Int64(sequenceLength)), "lanes": .integer(Int64(lanes)), "seed": .integer(Int64(seed))])
    }
    var actions: JSONValue {
        let generated = source == .practice
        return .object(["keyCodes": .array((generated ? (practiceTask == "delayed_memory" ? [123, 124] : []) : keys.sorted()).map { .integer(Int64($0)) }),
                        "mouseButtons": .array((generated ? [0] : buttons.sorted()).map { .integer(Int64($0)) }),
                        "absolutePointer": .bool(generated || (pointerEnabled && pointerMode == "absolute")),
                        "relativePointer": .bool(!generated && pointerEnabled && pointerMode == "relative"),
                        "scroll": .bool(!generated && scrollEnabled), "scrollUnitsPerPoint": .integer(8)])
    }
}

struct EpochMetric: Identifiable, Sendable {
    var epoch: Int
    var nll: Double
    var decisions: Int
    var id: Int { epoch }
}

struct BehaviorEvaluation: Sendable {
    var checkpointID: UUID
    var split: String
    var available: Bool
    var decisions: Int
    var meanNLL: Double?
    var reason: String?
    var savedAt: URL
}

/// The app owns lifecycle and catalog publication; the child owns model state.
@MainActor @Observable final class LearningCoordinator {
    private(set) var isBusy = false
    private(set) var isStopping = false
    private(set) var activeAgentID: UUID?
    private(set) var resultAgentID: UUID?
    private(set) var activeRun: LearningRunDocument?
    private(set) var phase = ""
    private(set) var metrics: [EpochMetric] = []
    private(set) var reinforcementMetrics: [ReinforcementMetric] = []
    private(set) var rolloutDecisions: Int?
    private(set) var rolloutTarget: Int?
    private(set) var elapsedSeconds: Double?
    private(set) var decisionsPerSecond: Double?
    private(set) var peakMemoryBytes: Int?
    private(set) var evaluation: BehaviorEvaluation?
    private(set) var failure: String?
    private let store: LibraryStore
    private let root: URL
    private let executable: URL
    private let weights: URL
    private let changed: @MainActor () async -> Void
    private var process: ComputeProcess?
    private var work: Task<Void, Never>?
    private var persistTask: Task<Void, Never>?
    private var lastPersist = Date.distantPast
    private var cancelRequested = false
    private var jobRunID: UUID?
    private var jobID: String?
    private var jobRequestID: UUID?
    private var earlyProgress: WireMessage?
    private var completion: CheckedContinuation<WireMessage, any Error>?
    private var earlyCompletion: WireMessage?
    private var processFailure: AstraError?
    private var processGeneration: UUID?

    init(store: LibraryStore, root: URL, bundle: Bundle = .main, changed: @escaping @MainActor () async -> Void) {
        self.store = store; self.root = root; self.changed = changed
        self.executable = bundle.bundleURL.appendingPathComponent("Contents/Helpers/AstraCompute.app/Contents/MacOS/AstraCompute")
        self.weights = (bundle.resourceURL ?? bundle.bundleURL).appendingPathComponent("Weights/convnext_tiny.safetensors")
    }

    func start(agent: AgentDocument, options rawOptions: BehaviorOptions, recordings: [RecordingManifest],
               selections: [UUID: RecordingTrainingSelection]? = nil) throws {
        guard !isBusy else { throw AstraError("learning.busy", "Finish or stop the current learning job first.") }
        let options = try rawOptions.validated()
        let selected = options.source == .recordings && !options.resume ? recordings.filter { options.recordingIDs.contains($0.id) } : []
        guard options.resume || options.source != .recordings || selected.count == options.recordingIDs.count,
              selected.allSatisfy({ $0.status != .recording && $0.frameCount > 0 }) else {
            throw AstraError("learning.sources", "A selected recording is unavailable or has not finished saving.")
        }
        let chosen = try selected.sorted { $0.id.uuidString < $1.id.uuidString }.map { recording -> JSONValue in
            guard let selection = selections == nil ? .whole : selections?[recording.id] else {
                throw AstraError("learning.selection", "Review the saved training intervals for \(recording.name) before training.")
            }
            _ = try selection.resolved(for: recording)
            return try selection.payload(recordingID: recording.id)
        }
        guard chosen.count <= 4096, chosen.reduce(0, { total, value in
            if case .array(let ranges) = value.fields?["ranges"] { return total + ranges.count }
            return total + 1
        }) <= 100_000 else { throw AstraError("learning.selectionLimit", "The selection exceeds the supported recording or interval count.") }
        // Check the selection envelope before starting a job. The final request
        // and saved configuration retain their full transport/artifact limits.
        _ = try WireMessage(kind: "dataset.prepare", sequence: 0, payload: .object(["selections": .array(chosen)])).framed()
        begin(agentID: agent.id)
        let run = LearningRunDocument(agentID: agent.id, kind: .behavioral,
                                      name: "\(String(agent.name.prefix(125))) · Imitation", sourceKind: options.source == .practice ? "practice_oracle" : "recordings")
        activeRun = run
        work = Task { await performTraining(agent: agent, options: options, recordings: selected, selections: chosen, runID: run.id) }
    }

    func startReinforcement(agent: AgentDocument, options rawOptions: ReinforcementOptions) throws {
        guard !isBusy else { throw AstraError("learning.busy", "Finish or stop the current learning job first.") }
        let options = try rawOptions.validated()
        begin(agentID: agent.id)
        let run = LearningRunDocument(agentID: agent.id, kind: .reinforcement,
                                      name: "\(String(agent.name.prefix(120))) · Reinforcement", sourceKind: "practice_rollout")
        activeRun = run
        work = Task { await performReinforcement(agent: agent, options: options, runID: run.id) }
    }

    private func performReinforcement(agent: AgentDocument, options: ReinforcementOptions, runID: UUID) async {
        let directory = artifact("Jobs", runID)
        let initialID = options.initialCheckpointID ?? UUID(), finalID = UUID()
        var model = options.model, actions = options.actions, environment = options.environment, training = options.training
        var contextIDs: JSONValue = .array([])
        do {
            activeRun?.initialCheckpointID = initialID
            try await saveRun()
            var selected: CheckpointDocument?
            if options.initialCheckpointID != nil {
                guard try await store.checkpointIDs(for: agent.id).contains(initialID) else {
                    throw AstraError("learning.checkpointOwner", "Choose a starting checkpoint linked to this agent.")
                }
                selected = try await store.snapshot().checkpoints.first { $0.id == initialID }
            }
            try await openProcess()
            if options.initialCheckpointID != nil {
                phase = "Checking starting checkpoint…"
                let inspected = try await job("checkpoint.inspect", .object(["path": .string(artifact("Models", initialID).path)]))
                try inspected.requireComplete()
                let manifest = try inspected.result.required("manifest")
                model = try manifest.required("model"); actions = try manifest.required("actions")
                if options.resume {
                    guard let selected, selected.kind == "reinforcement", let sourceRunID = selected.runID else {
                        throw AstraError("reinforcement.resume", "Resume requires a reinforcement checkpoint with its saved run configuration.")
                    }
                    let source = try await LearningFiles.read(artifact("Jobs", sourceRunID).appendingPathComponent("configuration.json"))
                    guard source.fields?["operation"] == .string("train.reinforcement"),
                          source.fields?["runID"]?.text.flatMap(UUID.init(uuidString:)) == sourceRunID,
                          source.fields?["agentID"]?.text.flatMap(UUID.init(uuidString:)) == selected.agentID else {
                        throw AstraError("reinforcement.resumeIdentity", "The saved reinforcement configuration belongs to a different run.")
                    }
                    environment = try source.required("environment"); training = try source.required("training")
                    contextIDs = source.fields?["contextIDs"] ?? .array([])
                } else {
                    guard actions == options.actions else {
                        throw AstraError("reinforcement.controls", "This checkpoint's controls do not match the selected practice task. Choose a compatible checkpoint or start a new agent.")
                    }
                    if var fields = environment.fields {
                        fields["period_ms"] = model.fields?["period_ms"] ?? .integer(100)
                        fields["lead_ms"] = model.fields?["lead_ms"] ?? .integer(100)
                        environment = .object(fields)
                    }
                    if case .array(let contexts) = model.fields?["context_sizes"] {
                        contextIDs = .array(contexts.map { _ in .integer(0) })
                    }
                }
            }
            let configuration: JSONValue = .object(["schemaVersion": .integer(1), "runID": .string(runID.uuidString.lowercased()),
                "agentID": .string(agent.id.uuidString.lowercased()), "operation": .string("train.reinforcement"),
                "sourceKind": .string("practice_rollout"), "environment": environment, "training": training,
                "iterations": .integer(Int64(options.iterations)), "resume": .bool(options.resume), "contextIDs": contextIDs,
                "model": model, "actions": actions, "initialCheckpointID": .string(initialID.uuidString.lowercased()),
                "destinationCheckpointID": .string(finalID.uuidString.lowercased())])
            try await LearningFiles.write(configuration, to: directory.appendingPathComponent("configuration.json"), exclusive: true)
            if options.initialCheckpointID == nil {
                guard FileManager.default.fileExists(atPath: weights.path) else {
                    throw AstraError("learning.weights", "The bundled visual weights are missing. Rebuild or reinstall AgentTrainer Astra.")
                }
                phase = "Preparing the model…"
                let initial = try await job("checkpoint.create", .object(["destination": .string(artifact("Models", initialID).path),
                    "model": model, "actions": actions, "seed": .integer(Int64(options.seed)), "pretrainedPath": .string(weights.path)]))
                try initial.requireComplete()
                try await publishCheckpoint(initial.result, agent: agent, runID: runID, expectedID: initialID, name: "Starting weights")
            }
            phase = "Collecting practice experience…"
            activeRun?.status = .running; try await saveRun()
            let trained = try await job("train.reinforcement", .object(["checkpointPath": .string(artifact("Models", initialID).path),
                "environment": environment, "training": training, "iterations": .integer(Int64(options.iterations)),
                "destination": .string(artifact("Models", finalID).path), "resume": .bool(options.resume), "contextIDs": contextIDs]))
            if case .array(let history) = trained.result.fields?["iterationMetrics"] {
                for item in history { if let fields = item.fields { recordReinforcementMetric(fields) } }
            }
            if trained.result.fields?["checkpointPublished"] == .bool(true) {
                try await publishCheckpoint(trained.result, agent: agent, runID: runID, expectedID: finalID,
                    name: trained.cancelled ? "Reinforcement · Stopped" : "Reinforcement · \(options.iterations) iterations")
                activeRun?.checkpointID = finalID
                if let fields = trained.result.fields?["manifest"]?.fields?["metrics"]?.fields {
                    activeRun?.epoch = fields["iteration"]?.int ?? activeRun?.epoch ?? 0
                    activeRun?.updates = fields["optimizer_updates"]?.int ?? activeRun?.updates ?? 0
                    activeRun?.decisions = fields["decisions"]?.int ?? activeRun?.decisions ?? 0
                    elapsedSeconds = fields["elapsed_seconds"]?.double
                }
            }
            try await LearningFiles.write(trained.result, to: directory.appendingPathComponent("results.json"), exclusive: true)
            activeRun?.status = trained.cancelled ? .cancelled : .completed
            phase = trained.cancelled
                ? (activeRun?.checkpointID == nil ? "Training stopped before a checkpoint was saved" : "Training stopped · checkpoint saved; resume starts a fresh episode")
                : "Reinforcement training complete"
            try await saveRun()
        } catch is CancellationError {
            activeRun?.status = .cancelled; phase = "Training stopped"
            do { try await saveRun() } catch { failure = error.localizedDescription }
        } catch {
            activeRun?.status = .failed; activeRun?.issue = error.localizedDescription
            failure = error.localizedDescription; phase = "Training needs attention"
            do { try await saveRun() } catch { failure = "\(failure ?? "")\nRun metadata could not be saved: \(error.localizedDescription)" }
        }
        await finish()
    }

    private func recordReinforcementMetric(_ fields: [String: JSONValue]) {
        guard let point = ReinforcementMetric(fields) else { return }
        if let index = reinforcementMetrics.firstIndex(where: { $0.iteration == point.iteration }) { reinforcementMetrics[index] = point }
        else {
            reinforcementMetrics.append(point); reinforcementMetrics.sort { $0.iteration < $1.iteration }
            if reinforcementMetrics.count > 1000 { reinforcementMetrics.removeFirst() }
        }
        if let memory = fields["peak_memory_bytes"]?.int { peakMemoryBytes = memory }
    }

    func evaluate(checkpoint: CheckpointDocument, agentID: UUID? = nil, split: String) throws {
        guard !isBusy, ["validation", "test", "train"].contains(split), let sourceRunID = checkpoint.runID else {
            throw AstraError("evaluation.source", "Choose a checkpoint with a saved demonstration dataset and finish any active job.")
        }
        let owner = agentID ?? checkpoint.agentID
        begin(agentID: owner); activeRun = nil
        work = Task { await performEvaluation(checkpoint: checkpoint, agentID: owner, sourceRunID: sourceRunID, split: split) }
    }

    func requestStop() async {
        guard isBusy else { return }
        cancelRequested = true; isStopping = true; phase = "Stopping at a safe boundary…"
        if activeRun != nil { activeRun?.status = .cancelling }
        if let process, let jobID, let jobRunID {
            let generation = processGeneration
            do { _ = try await process.request(kind: "cancel", payload: .object(["jobID": .string(jobID)]), runID: jobRunID) }
            catch {
                guard processGeneration == generation, isBusy else { return }
                failure = "Stop could not be acknowledged: \(error.localizedDescription)"
            }
        }
    }

    func stopAndWait() async {
        // Capture the work before yielding: a completed stop may allow a new
        // user job to begin while its old child's cancellation reply drains.
        let running = work
        await requestStop()
        await running?.value
    }

    private func begin(agentID: UUID) {
        isBusy = true; isStopping = false; cancelRequested = false; failure = nil; activeAgentID = agentID
        resultAgentID = agentID; evaluation = nil
        metrics = []; reinforcementMetrics = []; rolloutDecisions = nil; rolloutTarget = nil; elapsedSeconds = nil
        decisionsPerSecond = nil; peakMemoryBytes = nil; phase = "Preparing training…"
        processFailure = nil; earlyCompletion = nil; earlyProgress = nil; completion = nil; jobID = nil; jobRequestID = nil; jobRunID = nil
    }

    private func openProcess() async throws {
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw AstraError("learning.runtime", "The bundled learning runtime is missing. Rebuild or reinstall AgentTrainer Astra.")
        }
        let generation = UUID(); processGeneration = generation
        let child = ComputeProcess(executable: executable, environment: ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin"], onEvent: { [weak self] message in
            Task { @MainActor [weak self] in
                guard let self, processGeneration == generation else { return }
                receive(message)
            }
        }, onFailure: { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self, processGeneration == generation else { return }
                processFailed(error)
            }
        })
        process = child
        _ = try await child.start()
    }

    private func performTraining(agent: AgentDocument, options: BehaviorOptions, recordings: [RecordingManifest], selections: [JSONValue], runID: UUID) async {
        let directory = artifact("Jobs", runID)
        let datasetID = UUID(), initialID = options.initialCheckpointID ?? UUID(), finalID = UUID()
        var model = options.model, actions = options.actions, training = options.training
        var verificationMode = options.source == .practice
        var recordingIDs = recordings.map(\.id).sorted { $0.uuidString < $1.uuidString }
        var selectionProvenance: JSONValue? = options.source == .recordings && !options.resume ? .array(selections) : nil
        var dataset: JSONValue = .object([:])
        if !options.resume && options.source == .practice {
            let heldout = max(1, options.practiceEpisodes / 4)
            dataset = .object(["kind": .string("practice_oracle"), "environment": .object([
                "task": .string(options.practiceTask), "seed": .integer(Int64(options.seed)),
                "period_ms": .integer(Int64(options.periodMS)), "lead_ms": .integer(Int64(options.leadMS)),
                "delay_ms": .integer(Int64(options.practiceDelayMS))]), "seedsBySplit": .object([
                    "train": .array((0..<options.practiceEpisodes).map { .integer(Int64(options.seed + $0)) }),
                    "validation": .array((0..<heldout).map { .integer(Int64(options.seed + 10000 + $0)) }),
                    "test": .array((0..<heldout).map { .integer(Int64(options.seed + 20000 + $0)) })])])
        } else if !options.resume {
            dataset = .object(["kind": .string("recordings"), "path": .string(artifact("Datasets", datasetID).path),
                               "recordingRoot": .string(root.appendingPathComponent("Recordings").path)])
        }
        do {
            activeRun?.sourceRecordingIDs = recordingIDs
            activeRun?.initialCheckpointID = initialID
            try await saveRun()
            if options.initialCheckpointID != nil {
                guard try await store.checkpointIDs(for: agent.id).contains(initialID) else {
                    throw AstraError("learning.checkpointOwner", "Choose a starting checkpoint linked to this agent.")
                }
            }
            try await openProcess()
            if options.initialCheckpointID != nil {
                phase = "Checking starting checkpoint…"
                let inspected = try await job("checkpoint.inspect", .object(["path": .string(artifact("Models", initialID).path)]))
                try inspected.requireComplete()
                model = try inspected.result.required("manifest").required("model")
                actions = try inspected.result.required("manifest").required("actions")
                if options.resume {
                    let manifest = try inspected.result.required("manifest")
                    guard let checkpoint = try await store.snapshot().checkpoints.first(where: { $0.id == initialID }),
                          checkpoint.kind == "behavioral", let originalRunID = checkpoint.runID else {
                        throw AstraError("behavioral.resume", "Resume requires a behavioral checkpoint with its saved run configuration.")
                    }
                    let original = try await LearningFiles.read(artifact("Jobs", originalRunID).appendingPathComponent("configuration.json"))
                    guard original.fields?["operation"] == .string("train.behavioral"),
                          original.fields?["runID"]?.text.flatMap(UUID.init(uuidString:)) == originalRunID,
                          original.fields?["agentID"]?.text.flatMap(UUID.init(uuidString:)) == checkpoint.agentID else {
                        throw AstraError("behavioral.resumeIdentity", "The saved demonstration configuration belongs to another run.")
                    }
                    dataset = try original.required("dataset")
                    training = try manifest.required("trainingConfig")
                    guard case .object = training, let target = training.fields?["epochs"]?.int, target > 0,
                          case .bool(let verified) = original.fields?["verificationMode"] else {
                        throw AstraError("behavioral.resumeConfiguration", "The saved optimizer configuration or demonstration provenance is incomplete.")
                    }
                    verificationMode = verified
                    recordingIDs = try original.required("sourceRecordingIDs").decode([UUID].self)
                    selectionProvenance = original.fields?["sourceSelections"]
                    activeRun?.sourceRecordingIDs = recordingIDs
                    activeRun?.sourceKind = verified ? "practice_oracle" : "recordings"
                    if let saved = manifest.fields?["datasetID"]?.text.flatMap(UUID.init(uuidString:)) { activeRun?.datasetID = saved }
                    try await saveRun()
                } else if options.source == .practice, var fields = dataset.fields, var environment = fields["environment"]?.fields {
                    environment["period_ms"] = model.fields?["period_ms"]
                    environment["lead_ms"] = model.fields?["lead_ms"]
                    fields["environment"] = .object(environment); dataset = .object(fields)
                }
            }
            var configurationFields: [String: JSONValue] = ["schemaVersion": .integer(1), "runID": .string(runID.uuidString.lowercased()),
                "agentID": .string(agent.id.uuidString.lowercased()), "operation": .string("train.behavioral"),
                "dataset": dataset, "training": training, "model": model, "actions": actions, "resume": .bool(options.resume),
                "verificationMode": .bool(verificationMode), "sourceRecordingIDs": .array(recordingIDs.map { .string($0.uuidString.lowercased()) }),
                "initialCheckpointID": .string(initialID.uuidString.lowercased()), "destinationCheckpointID": .string(finalID.uuidString.lowercased())]
            if let selectionProvenance { configurationFields["sourceSelections"] = selectionProvenance }
            let configuration: JSONValue = .object(configurationFields)
            try await LearningFiles.write(configuration, to: directory.appendingPathComponent("configuration.json"), exclusive: true)
            if options.source == .recordings && !options.resume {
                phase = "Preparing demonstrations…"
                let prepared = try await job("dataset.prepare", .object([
                    "destination": .string(artifact("Datasets", datasetID).path), "recordingRoot": .string(root.appendingPathComponent("Recordings").path),
                    "selections": .array(selections),
                    "model": model, "actions": actions,
                    "pointerMode": .string(actions.fields?["relativePointer"] == .bool(true) && actions.fields?["absolutePointer"] != .bool(true) ? "relative" : options.pointerMode),
                    "splitSeed": .integer(Int64(options.seed))]))
                try prepared.requireComplete()
                model = try prepared.result.required("manifest").required("model")
                activeRun?.datasetID = datasetID; try await saveRun()
            }
            if options.initialCheckpointID == nil {
                guard FileManager.default.fileExists(atPath: weights.path) else {
                    throw AstraError("learning.weights", "The bundled visual weights are missing. Rebuild or reinstall AgentTrainer Astra.")
                }
                phase = "Preparing the model…"
                let initial = try await job("checkpoint.create", .object(["destination": .string(artifact("Models", initialID).path),
                    "model": model, "actions": actions, "seed": .integer(Int64(options.seed)), "pretrainedPath": .string(weights.path)]))
                try initial.requireComplete()
                try await publishCheckpoint(initial.result, agent: agent, runID: runID, expectedID: initialID, name: "Starting weights")
            }
            phase = "Learning from demonstrations…"
            activeRun?.status = .running; try await saveRun()
            let trained = try await job("train.behavioral", .object(["checkpointPath": .string(artifact("Models", initialID).path),
                "dataset": dataset, "training": training, "destination": .string(artifact("Models", finalID).path),
                "resume": .bool(options.resume), "verificationMode": .bool(verificationMode)]))
            if trained.result.fields?["checkpointPublished"] == .bool(true) {
                try await publishCheckpoint(trained.result, agent: agent, runID: runID, expectedID: finalID,
                                            name: trained.cancelled ? "Behavioral · Stopped" : "Behavioral · \(training.fields?["epochs"]?.int ?? options.epochs) epochs")
                activeRun?.checkpointID = finalID
                if let dataID = trained.result.fields?["manifest"]?.fields?["datasetID"]?.text.flatMap(UUID.init(uuidString:)) { activeRun?.datasetID = dataID }
                if let metricFields = trained.result.fields?["manifest"]?.fields?["metrics"]?.fields {
                    activeRun?.epoch = metricFields["epoch"]?.int ?? activeRun?.epoch ?? 0
                    activeRun?.updates = metricFields["updates"]?.int ?? activeRun?.updates ?? 0
                    activeRun?.decisions = metricFields["decisions"]?.int ?? activeRun?.decisions ?? 0
                }
            }
            var savedResult = trained.result.fields ?? [:]
            savedResult["epochMetrics"] = .array(metrics.map { .object(["epoch": .integer(Int64($0.epoch)),
                "meanNLL": .number($0.nll), "decisions": .integer(Int64($0.decisions))]) })
            try await LearningFiles.write(.object(savedResult), to: directory.appendingPathComponent("results.json"), exclusive: true)
            activeRun?.status = trained.cancelled ? .cancelled : .completed
            phase = trained.cancelled
                ? (activeRun?.checkpointID == nil ? "Training stopped before a checkpoint was saved" : "Training stopped · checkpoint saved")
                : "Training complete"
            try await saveRun()
        } catch is CancellationError {
            activeRun?.status = .cancelled; phase = "Training stopped"
            do { try await saveRun() } catch { failure = error.localizedDescription }
        } catch {
            activeRun?.status = .failed; activeRun?.issue = error.localizedDescription
            failure = error.localizedDescription; phase = "Training needs attention"
            do { try await saveRun() } catch { failure = "\(failure ?? "")\nRun metadata could not be saved: \(error.localizedDescription)" }
        }
        await finish()
    }

    private func performEvaluation(checkpoint: CheckpointDocument, agentID: UUID, sourceRunID: UUID, split: String) async {
        let destination = artifact("Jobs", UUID())
        do {
            guard try await store.snapshot().checkpoints.contains(where: { $0.matchesIdentity(of: checkpoint) }) else {
                throw AstraError("evaluation.checkpoint", "This checkpoint is no longer available in the workspace catalog.")
            }
            guard try await store.checkpointIDs(for: agentID).contains(checkpoint.id) else {
                throw AstraError("evaluation.checkpointOwner", "This checkpoint is not linked to the selected agent.")
            }
            let source = try await LearningFiles.read(artifact("Jobs", sourceRunID).appendingPathComponent("configuration.json"))
            guard source.fields?["agentID"]?.text.flatMap(UUID.init(uuidString:)) == checkpoint.agentID,
                  source.fields?["runID"]?.text.flatMap(UUID.init(uuidString:)) == sourceRunID else {
                throw AstraError("evaluation.sourceIdentity", "The saved demonstration configuration belongs to a different agent or run.")
            }
            guard source.fields?["operation"] == .string("train.behavioral"), source.fields?["dataset"] != nil else {
                throw AstraError("evaluation.dataset", "This checkpoint has no saved demonstration dataset to evaluate.")
            }
            let payload: JSONValue = .object(["checkpointPath": .string(artifact("Models", checkpoint.id).path),
                "dataset": try source.required("dataset"), "verificationMode": source.fields?["verificationMode"] ?? .bool(false),
                "split": .string(split), "sequenceLength": .integer(64)])
            try await LearningFiles.write(payload, to: destination.appendingPathComponent("configuration.json"), exclusive: true)
            phase = "Evaluating \(checkpoint.name)…"
            try await openProcess()
            let response = try await job("evaluate.behavioral", payload)
            try response.requireComplete()
            let result = try response.result.required("evaluation")
            let path = destination.appendingPathComponent("results.json")
            try await LearningFiles.write(response.result, to: path, exclusive: true)
            evaluation = BehaviorEvaluation(checkpointID: checkpoint.id, split: split,
                available: result.fields?["available"] == .bool(true), decisions: result.fields?["decisions"]?.int ?? 0,
                meanNLL: result.fields?["meanNLL"]?.double, reason: result.fields?["reason"]?.text, savedAt: path)
            phase = "Evaluation complete"
        } catch is CancellationError { phase = "Evaluation stopped" }
        catch { failure = error.localizedDescription; phase = "Evaluation needs attention" }
        await finish()
    }

    private struct JobResult {
        let result: JSONValue
        let cancelled: Bool
        func requireComplete() throws { if cancelled { throw CancellationError() } }
    }

    private func job(_ kind: String, _ payload: JSONValue) async throws -> JobResult {
        guard !cancelRequested else { throw CancellationError() }
        guard let process else { throw AstraError("learning.runtime", "The learning runtime is not running.") }
        if let processFailure { throw processFailure }
        let identifier = UUID(); jobRunID = identifier; jobID = nil; jobRequestID = nil; earlyCompletion = nil; earlyProgress = nil
        defer { jobRunID = nil; jobID = nil; jobRequestID = nil; completion = nil; earlyCompletion = nil; earlyProgress = nil }
        let accepted = try await process.request(kind: kind, payload: payload, runID: identifier)
        guard let acceptedID = accepted.payload.fields?["jobID"]?.text, UUID(uuidString: acceptedID) != nil,
              accepted.requestID != nil, accepted.runID == identifier else {
            throw AstraError("learning.reply", "The runtime did not identify the accepted job and request.")
        }
        jobID = acceptedID; jobRequestID = accepted.requestID
        if let earlyProgress { self.earlyProgress = nil; receive(earlyProgress) }
        if let processFailure { throw processFailure }
        if cancelRequested { await requestStop() }
        let ended: WireMessage
        if let earlyCompletion { ended = earlyCompletion }
        else {
            if let processFailure { throw processFailure }
            ended = try await withCheckedThrowingContinuation { completion = $0 }
        }
        guard ended.payload.fields?["jobID"]?.text == acceptedID, ended.requestID == accepted.requestID else {
            throw AstraError("learning.jobIdentity", "The runtime completed a different job than the one it acknowledged.")
        }
        if ended.kind == "job.failed" {
            throw AstraError("learning.jobFailed", ended.payload.fields?["error"]?.fields?["message"]?.text ?? "The learning job failed.")
        }
        return JobResult(result: try ended.payload.required("result"), cancelled: ended.kind == "job.cancelled")
    }

    private func receive(_ message: WireMessage) {
        guard message.runID == jobRunID else { return }
        if message.kind != "job.progress" {
            if let waiting = completion { completion = nil; waiting.resume(returning: message) }
            else { earlyCompletion = message }
            return
        }
        // A child can emit progress before the acknowledgement continuation
        // resumes. Keep only its latest sample until all three identities are known.
        guard let jobID, let jobRequestID else { earlyProgress = message; return }
        guard message.requestID == jobRequestID, message.payload.fields?["jobID"]?.text == jobID else {
            processFailed(AstraError("learning.progressIdentity", "The runtime reported progress for a different job or request."))
            return
        }
        guard let fields = message.payload.fields else { return }
        if fields["phase"]?.text == "checkpointing", !isStopping { phase = "Saving checkpoint…" }
        if fields["sourceKind"] == .string("practice_rollout") {
            if !isStopping {
                switch fields["phase"]?.text {
                case "collecting": phase = "Collecting practice experience…"
                case "finishing_episode": phase = "Collecting through the end of this episode…"
                case "updating": phase = "Updating the learner…"
                case "validating_update": phase = "Checking the proposed policy update…"
                case "waiting_for_reset": phase = "Finishing the current episode before switching policies…"
                case "iteration": phase = "Iteration complete · next policy waits for reset"
                case "checkpointing": phase = "Saving reinforcement checkpoint…"
                default: break
                }
            }
            activeRun?.epoch = fields["iteration"]?.int ?? activeRun?.epoch ?? 0
            activeRun?.updates = fields["optimizer_updates"]?.int ?? activeRun?.updates ?? 0
            activeRun?.decisions = fields["decisions"]?.int ?? activeRun?.decisions ?? 0
            elapsedSeconds = fields["elapsed_seconds"]?.double
            rolloutDecisions = fields["rollout_decisions"]?.int
            rolloutTarget = fields["rollout_target"]?.int
            if let last = fields["last_iteration"]?.fields { recordReinforcementMetric(last) }
            recordReinforcementMetric(fields)
            decisionsPerSecond = fields["decisions_per_second"]?.double
        } else {
            guard let epoch = fields["epoch"]?.int, let nll = fields["mean_nll"]?.double, nll.isFinite else { return }
            let sample = EpochMetric(epoch: epoch + 1, nll: nll, decisions: fields["decisions"]?.int ?? 0)
            if let index = metrics.firstIndex(where: { $0.epoch == sample.epoch }) { metrics[index] = sample }
            else { metrics.append(sample); if metrics.count > 1000 { metrics.removeFirst() } }
            activeRun?.epoch = epoch; activeRun?.meanNLL = nll
            activeRun?.updates = fields["updates"]?.int ?? activeRun?.updates ?? 0
            activeRun?.decisions = sample.decisions
            decisionsPerSecond = fields["decisions_per_second"]?.double
            peakMemoryBytes = fields["peak_memory_bytes"]?.int
        }
        if persistTask == nil, Date().timeIntervalSince(lastPersist) >= 1 {
            lastPersist = Date()
            persistTask = Task { [weak self] in
                guard let self else { return }
                do {
                    if var run = activeRun { run.modifiedAt = Date(); try await store.saveLearningRun(run) }
                    await changed()
                } catch { failure = "Training continues, but progress could not be saved: \(error.localizedDescription)" }
                persistTask = nil
            }
        }
    }

    private func processFailed(_ error: AstraError) {
        processFailure = error
        if let waiting = completion { completion = nil; waiting.resume(throwing: error) }
    }

    private func saveRun() async throws {
        await persistTask?.value
        guard var run = activeRun else { return }
        run.modifiedAt = Date(); activeRun = run
        try await store.saveLearningRun(run)
        await changed()
    }

    private func publishCheckpoint(_ value: JSONValue, agent: AgentDocument, runID: UUID, expectedID: UUID, name: String) async throws {
        let manifest = try value.required("manifest")
        guard let id = manifest.fields?["id"]?.text.flatMap(UUID.init(uuidString:)),
              let kind = manifest.fields?["kind"]?.text, let signature = manifest.fields?["policySignature"]?.text,
              let step = manifest.fields?["step"]?.int, let count = value.fields?["parameterCount"]?.int,
              id == expectedID, value.fields?["checkpointPath"]?.text == artifact("Models", expectedID).path,
              FileManager.default.fileExists(atPath: artifact("Models", expectedID).appendingPathComponent("manifest.json").path),
              FileManager.default.fileExists(atPath: artifact("Models", expectedID).appendingPathComponent("policy.safetensors").path) else {
            throw AstraError("learning.checkpoint", "The runtime's checkpoint result is incomplete. Its files were preserved.")
        }
        let document = CheckpointDocument(id: id, agentID: agent.id, runID: runID, name: name, kind: kind,
                                          trainingStep: step, policySignature: signature, parameterCount: count)
        try await store.saveCheckpoint(document)
        if kind != "initial" {
            let latest = try await store.snapshot().agents.first { $0.id == agent.id }
            if var current = latest { current.selectedCheckpointID = id; current.modifiedAt = Date(); try await store.save(current) }
        }
        await changed()
    }

    private func finish() async {
        await persistTask?.value
        processGeneration = nil // Closing-child callbacks cannot affect the next session.
        await process?.shutdown(); process = nil
        isBusy = false; isStopping = false; activeAgentID = nil; work = nil
        await changed()
    }

    private func artifact(_ folder: String, _ id: UUID) -> URL {
        root.appendingPathComponent(folder, isDirectory: true).appendingPathComponent(id.uuidString.lowercased(), isDirectory: true)
    }
}

enum LearningFiles {
    static func write(_ value: JSONValue, to url: URL, exclusive: Bool) async throws {
        try await Task.detached {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(value)
            guard data.count <= 8 * 1024 * 1024 else { throw AstraError("learning.artifact", "Learning metadata exceeds its supported size.") }
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let staging = url.deletingLastPathComponent().appendingPathComponent(".learning-\(UUID().uuidString).json")
            defer { try? FileManager.default.removeItem(at: staging) }
            try data.write(to: staging, options: .withoutOverwriting)
            let handle = try FileHandle(forWritingTo: staging)
            do { try handle.synchronize(); try handle.close() }
            catch { try? handle.close(); throw error }
            // Same-directory hard-link publication is an atomic no-clobber
            // operation. A check followed by Foundation atomic replacement
            // could overwrite a competing writer's immutable configuration.
            let published = exclusive ? link(staging.path, url.path) : rename(staging.path, url.path)
            guard published == 0 else {
                throw AstraError(exclusive && errno == EEXIST ? "learning.immutable" : "learning.publish",
                                 exclusive && errno == EEXIST ? "A saved run configuration cannot be overwritten." : "The learning artifact could not be published.")
            }
            let parent = open(url.deletingLastPathComponent().path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
            guard parent >= 0 else { throw AstraError("learning.artifactSync", "The learning artifact directory could not be synchronized.") }
            defer { close(parent) }
            guard fsync(parent) == 0 else { throw AstraError("learning.artifactSync", "The learning artifact directory could not be synchronized.") }
        }.value
    }
    static func read(_ url: URL) async throws -> JSONValue {
        try await Task.detached {
            let limit = 8 * 1024 * 1024
            let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
            guard descriptor >= 0 else {
                throw AstraError("learning.artifact", "Saved learning metadata is missing or invalid.")
            }
            let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
            defer { try? handle.close() }
            var metadata = stat()
            guard fstat(descriptor, &metadata) == 0, metadata.st_mode & S_IFMT == S_IFREG,
                  metadata.st_size >= 0, metadata.st_size <= limit else {
                throw AstraError("learning.artifact", "Saved learning metadata is missing or invalid.")
            }
            var data = Data()
            while let chunk = try handle.read(upToCount: min(65_536, limit + 1 - data.count)), !chunk.isEmpty {
                data.append(chunk)
                guard data.count <= limit else { throw AstraError("learning.artifact", "Saved learning metadata exceeds its supported size.") }
            }
            return try JSONDecoder().decode(JSONValue.self, from: data)
        }.value
    }
    static func recordedCapabilities(_ recordings: [RecordingManifest], root: URL,
                                     selections: [UUID: RecordingTrainingSelection]? = nil) async throws -> ActionCapabilities {
        let operation = Task.detached {
            var result = ActionCapabilities()
            let decoder = JSONDecoder()
            for recording in recordings {
                try Task.checkCancellation()
                guard recording.status != .recording else { throw AstraError("learning.recordingActive", "Finish recording before choosing its controls.") }
                let ranges: [RecordingTimeRange]?
                if let selections {
                    guard let selection = selections[recording.id] else { throw AstraError("learning.selection", "Review the saved intervals for \(recording.name) before choosing its controls.") }
                    ranges = try selection.resolved(for: recording)
                } else { ranges = nil }
                let directory = root.appendingPathComponent("Recordings").appendingPathComponent(recording.id.uuidString + ".astrarecord")
                let descriptor = open(directory.appendingPathComponent(".writer.lock").path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
                guard descriptor >= 0 else { throw AstraError("learning.recordingLock", "The recording's lock is unavailable.") }
                defer { close(descriptor) }
                guard flock(descriptor, LOCK_SH | LOCK_NB) == 0 else { throw AstraError("learning.recordingActive", "The recording is still being saved or recovered.") }
                defer { flock(descriptor, LOCK_UN) }
                let database = try SQLiteDatabase(url: recording.indexURL(in: directory), readOnly: true)
                defer { try? database.close() }
                var previous: Int64 = -1
                while true {
                    try Task.checkCancellation()
                    let rows = try database.query("SELECT sequence,observed,source_time,CASE WHEN length(event)<=262144 THEN event ELSE NULL END AS event FROM events WHERE sequence>? ORDER BY sequence LIMIT 256", [.integer(previous)])
                    if rows.isEmpty { break }
                    for row in rows {
                        guard let sequence = row["sequence"]?.integer, sequence >= 0,
                              let observed = row["observed"]?.integer, observed >= 0,
                              let source = row["source_time"]?.integer, source >= 0,
                              let data = row["event"]?.data else { throw AstraError("learning.inputMetadata", "Recorded input metadata is oversized or invalid.") }
                        previous = sequence
                        let event = try decoder.decode(RawInputEvent.self, from: data)
                        guard event.sequence == UInt64(sequence), event.observedNanos == UInt64(observed), event.eventNanos == UInt64(source) else {
                            throw AstraError("learning.inputIdentity", "Recorded input does not agree with its saved index.")
                        }
                        guard event.origin == .physical || event.origin == .reconciliation else { continue }
                        if let ranges {
                            guard event.origin == .physical else { continue }
                            // Source times may arrive late relative to sequence;
                            // binary search preserves that order without R scans.
                            var lower = 0, upper = ranges.count
                            while lower < upper {
                                let middle = lower + (upper - lower) / 2
                                if ranges[middle].endNanos <= event.eventNanos { lower = middle + 1 } else { upper = middle }
                            }
                            guard lower < ranges.count, ranges[lower].startNanos <= event.eventNanos else { continue }
                        }
                        if let key = event.keyCode { result.keyCodes.insert(key) }
                        if let button = event.button { result.mouseButtons.insert(button) }
                        if event.kind == .pointer || ((event.kind == .buttonDown || event.kind == .buttonUp) && event.x != nil && event.y != nil) {
                            result.absolutePointer = true; result.relativePointer = true
                        }
                        if event.kind == .scroll { result.scroll = true }
                    }
                }
            }
            return try result.validated()
        }
        return try await withTaskCancellationHandler { try await operation.value } onCancel: { operation.cancel() }
    }
}

extension JSONValue {
    var fields: [String: JSONValue]? { if case .object(let value) = self { value } else { nil } }
    var text: String? { if case .string(let value) = self { value } else { nil } }
    var int: Int? {
        switch self { case .integer(let value): Int(exactly: value); case .unsigned(let value): Int(exactly: value); default: nil }
    }
    var double: Double? {
        switch self { case .number(let value): value; case .integer(let value): Double(value); case .unsigned(let value): Double(value); default: nil }
    }
    func required(_ key: String) throws -> JSONValue {
        guard let value = fields?[key], value != .null else { throw AstraError("learning.metadata", "Learning metadata is missing \(key).") }
        return value
    }
}
