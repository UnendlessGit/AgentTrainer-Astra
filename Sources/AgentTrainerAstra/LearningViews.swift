import SwiftUI
import Charts
import AstraCore

/// Semantic warning color belongs to the symbol; body text needs readable
/// contrast on both Aqua and dark Aqua surfaces.
struct AttentionLabel: View {
    let message: String
    var symbol = "exclamationmark.triangle"
    var body: some View {
        Label { Text(message).foregroundStyle(.primary) } icon: { Image(systemName: symbol).foregroundStyle(.orange) }
            .textSelection(.enabled)
    }
}

struct LearningBanner: View {
    @Bindable var learning: LearningCoordinator
    var body: some View {
        HStack(spacing: 12) {
            ProgressView().controlSize(.small)
            VStack(alignment: .leading, spacing: 3) {
                Text(learning.phase).fontWeight(.medium)
                if let run = learning.activeRun {
                    Text(run.kind == .reinforcement ? "\(run.name) · \(run.epoch.formatted()) iterations completed"
                         : "\(run.name) · \(run.decisions.formatted()) decisions")
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
            }
            Spacer()
            Button(learning.isStopping ? "Stopping…" : "Stop", systemImage: "stop.fill") {
                Task { await learning.requestStop() }
            }.disabled(learning.isStopping)
        }.padding(14).background(.tint.opacity(0.08)).accessibilityElement(children: .contain)
    }
}

struct BehaviorTrainingView: View {
    private struct ControlSelectionKey: Hashable {
        let recordings: Set<UUID>
        let selections: [UUID: RecordingTrainingSelection]
    }
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var options = BehaviorOptions()
    @State private var detected = ActionCapabilities()
    @State private var readingControls = false
    @State private var controlsError: String?
    @State private var initialized = false

    init(agent: AgentDocument, model: WorkspaceModel, options: BehaviorOptions = .init()) {
        self.agent = agent; self.model = model; _options = State(initialValue: options)
    }

    private var recordings: [RecordingManifest] {
        model.recordings.filter { model.recordingLinks[agent.id]?.contains($0.id) == true && $0.status != .recording && $0.frameCount > 0 }
    }
    private var checkpoints: [CheckpointDocument] { model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true } }
    private var controlSelectionKey: ControlSelectionKey { .init(recordings: options.recordingIDs, selections: model.recordingSelections[agent.id] ?? [:]) }
    private var ownsActiveRun: Bool { model.learning?.activeRun?.agentID == agent.id && model.learning?.activeRun?.kind == .behavioral }
    private var startingCheckpoint: CheckpointDocument? { checkpoints.first { $0.id == options.initialCheckpointID } }
    private var completedStartingRun: Bool {
        guard let checkpoint = startingCheckpoint else { return false }
        return model.learningRuns.first { $0.id == checkpoint.runID || $0.checkpointID == checkpoint.id }?.status == .completed
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Imitation learning").font(.title2.weight(.semibold))
                    Text("Learn from recorded examples, then evaluate a saved checkpoint on independent demonstrations.")
                        .foregroundStyle(.secondary)
                }
                if ownsActiveRun, let learning = model.learning {
                    LearningProgressView(learning: learning)
                }
                if let failure = model.learning?.failure, model.learning?.activeRun?.agentID == agent.id {
                    AttentionLabel(message: failure)
                }
                GroupBox {
                    VStack(alignment: .leading, spacing: 16) {
                        if options.resume {
                            Text("Resume uses the original saved dataset, including its training, validation, and test sessions.")
                                .foregroundStyle(.secondary)
                        } else {
                            Picker("Demonstrations", selection: $options.source) {
                                ForEach(DemonstrationSource.allCases) { Text($0.title).tag($0) }
                            }
                            if options.source == .recordings {
                                if recordings.isEmpty {
                                    Text("Record a demonstration in this agent's Demonstrations tab to train from your own behavior.")
                                        .foregroundStyle(.secondary)
                                } else {
                                    VStack(alignment: .leading, spacing: 10) {
                                        ForEach(recordings) { recording in
                                            Toggle(isOn: Binding(get: { options.recordingIDs.contains(recording.id) }, set: { chosen in
                                                if chosen { options.recordingIDs.insert(recording.id) } else { options.recordingIDs.remove(recording.id) }
                                            })) {
                                                HStack {
                                                    Text(recording.name)
                                                    Spacer()
                                                    Text(model.trainingSelectionSummary(recording: recording, agentID: agent.id)).foregroundStyle(.secondary).monospacedDigit()
                                                }
                                            }.toggleStyle(.checkbox)
                                        }
                                        Text("Each recording stays in one training, validation, or test set. Separate intervals reset the model’s memory. Edit intervals in Demonstrations → Review Recording.")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                }
                            } else {
                                Text("A local practice environment generates labeled examples. These are generated demonstrations, not recordings of your behavior. No screen or input permissions are needed.")
                                    .font(.callout).foregroundStyle(.secondary)
                                Picker("Task", selection: $options.practiceTask) {
                                    Text("Pointing").tag("pointing")
                                    Text("Delayed visual memory").tag("delayed_memory")
                                }
                                if options.practiceTask == "delayed_memory" {
                                    Picker("Memory delay", selection: $options.practiceDelayMS) {
                                        Text("2 seconds").tag(2000); Text("8 seconds").tag(8000); Text("30 seconds").tag(30000)
                                    }
                                }
                                Stepper("Training episodes: \(options.practiceEpisodes)", value: $options.practiceEpisodes, in: 1...1000)
                                Text("\(max(1, options.practiceEpisodes / 4)) additional episodes each for validation and test, with independent layouts.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }.padding(8)
                } label: { Text("Demonstrations").font(.headline) }
                GroupBox {
                    VStack(alignment: .leading, spacing: 14) {
                        Picker("Starting point", selection: $options.initialCheckpointID) {
                            Text("New agent · pretrained vision").tag(nil as UUID?)
                            ForEach(checkpoints) { checkpoint in
                                Text("\(checkpoint.name) · \(checkpoint.createdAt.formatted(date: .abbreviated, time: .shortened))").tag(Optional(checkpoint.id))
                            }
                        }.onChange(of: options.initialCheckpointID) { _, _ in options.resume = false }
                        if startingCheckpoint?.kind == "behavioral" {
                            Toggle("Resume saved optimizer and progress", isOn: $options.resume)
                                .disabled(completedStartingRun)
                            if completedStartingRun {
                                Text("This run already completed its epoch target. Continue from its checkpoint to begin another training run.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        Text(options.initialCheckpointID == nil
                             ? "Start with transferable visual features and learn the controls from these demonstrations."
                             : "Continue learning from this checkpoint. Its observation, timing, and control settings remain fixed.")
                            .font(.caption).foregroundStyle(.secondary)
                        if options.resume {
                            Text("Restores the saved optimizer and training progress with the original settings and epoch target. Start another run from this checkpoint to choose a different target or dataset.")
                                .font(.callout).foregroundStyle(.secondary)
                        } else {
                            Stepper("Epochs: \(options.epochs)", value: $options.epochs, in: 1...100_000)
                            Text("An epoch visits every training demonstration once.").font(.caption).foregroundStyle(.secondary)
                            DisclosureGroup("Advanced") {
                                VStack(alignment: .leading, spacing: 14) {
                                    TextField("Learning rate", value: $options.learningRate, format: .number.precision(.fractionLength(1...6)))
                                        .textFieldStyle(.roundedBorder)
                                    Stepper("Contiguous sequence: \(options.sequenceLength) decisions", value: $options.sequenceLength, in: 1...512)
                                    Stepper("Parallel episode lanes: \(options.lanes)", value: $options.lanes, in: 1...32)
                                    TextField("Experiment seed", value: $options.seed, format: .number.grouping(.never))
                                        .textFieldStyle(.roundedBorder)
                                    if options.initialCheckpointID == nil {
                                        Picker("Decision rate", selection: $options.periodMS) {
                                            Text("10 Hz").tag(100); Text("20 Hz").tag(50)
                                        }
                                        Stepper("Execution lead: \(options.leadMS) ms", value: $options.leadMS, in: 0...2000, step: 5)
                                        Picker("Commands per decision", selection: $options.packetCapacity) {
                                            ForEach([16, 32, 64], id: \.self) { Text("\($0)").tag($0) }
                                        }
                                        Text("Timing stays fixed for this model. Demonstrations must fit the command budget without losing input events.")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                }.padding(.top, 12)
                            }
                        }
                    }.padding(8)
                } label: { Text("Training").font(.headline) }
                if options.source == .recordings && options.initialCheckpointID == nil {
                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            if readingControls { ProgressView("Reading recorded controls…") }
                            if let controlsError { AttentionLabel(message: controlsError) }
                            if !detected.keyCodes.isEmpty {
                                Text("Keyboard").font(.subheadline.weight(.medium))
                                LazyVGrid(columns: [GridItem(.adaptive(minimum: 110), alignment: .leading)], alignment: .leading) {
                                    ForEach(detected.keyCodes.sorted(), id: \.self) { key in
                                        Toggle(KeyNames.name(key), isOn: Binding(get: { options.keys.contains(key) }, set: { chosen in
                                            if chosen { options.keys.insert(key) } else { options.keys.remove(key) }
                                        })).toggleStyle(.checkbox)
                                    }
                                }
                            }
                            if !detected.mouseButtons.isEmpty {
                                LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), alignment: .leading)], alignment: .leading) {
                                    ForEach(detected.mouseButtons.sorted(), id: \.self) { button in
                                        Toggle(KeyNames.button(button), isOn: Binding(get: { options.buttons.contains(button) }, set: { chosen in
                                            if chosen { options.buttons.insert(button) } else { options.buttons.remove(button) }
                                        })).toggleStyle(.checkbox)
                                    }
                                }
                            }
                            Toggle("Pointer movement", isOn: $options.pointerEnabled).toggleStyle(.checkbox)
                            if options.pointerEnabled {
                                Picker("Pointer representation", selection: $options.pointerMode) {
                                    Text("Screen position").tag("absolute"); Text("Relative motion").tag("relative")
                                }
                            }
                            Toggle("Scrolling", isOn: $options.scrollEnabled).toggleStyle(.checkbox)
                            Text("Controls are detected inside the selected training intervals. Every demonstration must fit the enabled controls; incompatible input is reported before training.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.padding(8)
                    } label: { Text("Controls the agent can learn").font(.headline) }
                }
                HStack {
                    Text("Training and checkpoints stay on this Mac.").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(options.resume ? "Resume Training" : "Start Training", systemImage: "play.fill") { model.startBehaviorTraining(agent: agent, options: options) }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isClosing || model.isRunningAgent || model.learning == nil || model.learning?.isBusy == true
                                  || (options.resume && (startingCheckpoint?.kind != "behavioral" || completedStartingRun))
                                  || (options.source == .recordings && !options.resume && (readingControls || controlsError != nil)) || (try? options.validated()) == nil)
                }
                LearningRunList(runs: model.learningRuns.filter { $0.agentID == agent.id }, model: model, compact: true)
            }.padding(.trailing, 8).padding(.bottom, 20)
        }
        .onAppear {
            guard !initialized else { return }
            initialized = true; options.recordingIDs = Set(recordings.map(\.id))
        }
        .task(id: controlSelectionKey) { await readControls() }
    }

    private func readControls() async {
        let selected = options.recordingIDs
        let key = controlSelectionKey
        readingControls = true
        defer { if key == controlSelectionKey { readingControls = false } }
        do {
            let result = try await LearningFiles.recordedCapabilities(recordings.filter { selected.contains($0.id) }, root: model.supportRoot, selections: key.selections)
            guard !Task.isCancelled, key == controlSelectionKey else { return }
            detected = result; options.keys = result.keyCodes; options.buttons = result.mouseButtons
            options.pointerEnabled = result.absolutePointer; options.scrollEnabled = result.scroll; controlsError = nil
        } catch is CancellationError { }
        catch { if key == controlSelectionKey { controlsError = error.localizedDescription } }
    }
}

struct LearningProgressView: View {
    @Bindable var learning: LearningCoordinator
    var body: some View {
        LearningProgressContent(metrics: learning.metrics, phase: learning.phase, updates: learning.activeRun?.updates,
                                decisionsPerSecond: learning.decisionsPerSecond, peakMemoryBytes: learning.peakMemoryBytes)
    }
}

struct LearningProgressContent: View {
    let metrics: [EpochMetric]
    let phase: String
    let updates: Int?
    let decisionsPerSecond: Double?
    let peakMemoryBytes: Int?
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), alignment: .leading)], alignment: .leading, spacing: 16) {
                    if let point = metrics.last {
                        metric("Epoch", value: point.epoch.formatted())
                        metric("Imitation loss", value: point.nll.formatted(.number.precision(.fractionLength(3))))
                    }
                    if let updates { metric("Updates", value: updates.formatted()) }
                    if let rate = decisionsPerSecond { metric("Decisions / second", value: rate.formatted(.number.precision(.fractionLength(1)))) }
                }.monospacedDigit()
                if !metrics.isEmpty {
                    Chart(metrics) { point in
                        LineMark(x: .value("Epoch", point.epoch), y: .value("Imitation loss", point.nll))
                            .interpolationMethod(.linear)
                        PointMark(x: .value("Epoch", point.epoch), y: .value("Imitation loss", point.nll))
                            .symbolSize(18)
                    }.chartXAxisLabel("Epoch").chartYAxisLabel("Imitation loss").frame(height: 170)
                        .accessibilityLabel("Imitation loss by epoch")
                    DisclosureGroup("Metrics table") {
                        Grid(alignment: .leading, horizontalSpacing: 30, verticalSpacing: 6) {
                            GridRow { Text("Epoch"); Text("Loss"); Text("Decisions") }.fontWeight(.medium)
                            ForEach(metrics.suffix(20)) { point in
                                GridRow { Text(point.epoch.formatted()); Text(point.nll.formatted(.number.precision(.fractionLength(3)))); Text(point.decisions.formatted()) }
                            }
                        }.font(.caption).monospacedDigit().padding(.top, 8)
                    }
                } else { Text(phase).foregroundStyle(.secondary) }
                if let bytes = peakMemoryBytes {
                    Text("Peak model memory: \(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory))")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
        } label: { Text("Training progress").font(.headline) }
    }
    private func metric(_ title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption).foregroundStyle(.secondary); Text(value).font(.title3.weight(.medium)) }
    }
}

struct LearningEvaluationView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    var body: some View { EvaluationWorkspaceView(agent: agent, model: model) }
}

struct LearningRunList: View {
    let runs: [LearningRunDocument]
    @Bindable var model: WorkspaceModel
    var compact = false
    var body: some View {
        if runs.isEmpty {
            if !compact {
                ContentUnavailableView("No activity yet", systemImage: "clock.arrow.circlepath",
                                       description: Text("Saved training runs and their checkpoints will appear here."))
            }
        } else {
            VStack(alignment: .leading, spacing: 12) {
                Text(compact ? "Training history" : "Activity").font(compact ? .headline : .largeTitle.weight(.semibold))
                ForEach(runs) { run in
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: run.status == .completed ? "checkmark.circle" : (run.status == .failed ? "exclamationmark.triangle" : "chart.xyaxis.line"))
                            .foregroundStyle(run.status == .failed ? Color.orange : Color.accentColor)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(run.name).fontWeight(.medium)
                            Text("\(run.status.displayName) · \(run.createdAt.formatted(date: .abbreviated, time: .shortened)) · \(run.updates.formatted()) updates")
                                .font(.caption).foregroundStyle(.secondary)
                            if let loss = run.meanNLL {
                                Text("Loss \(loss, specifier: "%.3f") · \(run.decisions.formatted()) decisions")
                                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                            }
                            if run.sourceKind == "practice_oracle" { Text("Generated practice demonstrations").font(.caption).foregroundStyle(.secondary) }
                            if run.sourceKind == "practice_rollout" { Text("Self-collected practice experience · \(run.decisions.formatted()) decisions").font(.caption).foregroundStyle(.secondary) }
                            if let issue = run.issue { AttentionLabel(message: issue).font(.caption) }
                        }
                        Spacer()
                        if let checkpointID = run.checkpointID {
                            let selected = model.agents.first(where: { $0.id == run.agentID })?.selectedCheckpointID == checkpointID
                            Button(selected ? "Selected" : "Select Checkpoint") { Task { await model.selectCheckpoint(checkpointID, agentID: run.agentID) } }
                                .disabled(selected)
                        }
                    }.padding(12).background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                        .accessibilityElement(children: .contain)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

extension LearningStatus {
    var displayName: String {
        switch self {
        case .preparing: "Preparing"; case .running: "Training"; case .cancelling: "Stopping"
        case .completed: "Completed"; case .cancelled: "Stopped"; case .failed: "Needs attention"; case .interrupted: "Interrupted"
        }
    }
}

enum KeyNames {
    static func button(_ code: Int) -> String { code == 0 ? "Left mouse" : code == 1 ? "Right mouse" : code == 2 ? "Middle mouse" : "Mouse \(code + 1)" }
    static func name(_ code: Int) -> String {
        let names = [0:"A", 1:"S", 2:"D", 3:"F", 4:"H", 5:"G", 6:"Z", 7:"X", 8:"C", 9:"V", 11:"B", 12:"Q", 13:"W", 14:"E", 15:"R", 16:"Y", 17:"T",
                     18:"1", 19:"2", 20:"3", 21:"4", 22:"6", 23:"5", 25:"9", 26:"7", 28:"8", 29:"0", 31:"O", 32:"U", 34:"I", 35:"P", 37:"L", 38:"J", 40:"K", 45:"N", 46:"M",
                     36:"Return", 48:"Tab", 49:"Space", 51:"Delete", 53:"Escape", 54:"Right Command", 55:"Command", 56:"Shift", 57:"Caps Lock", 58:"Option", 59:"Control", 60:"Right Shift", 61:"Right Option", 62:"Right Control", 63:"Fn",
                     123:"Left Arrow", 124:"Right Arrow", 125:"Down Arrow", 126:"Up Arrow"]
        return names[code] ?? "Key \(code)"
    }
}
