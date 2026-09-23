import SwiftUI
import AstraCore
import AstraPlatform

struct DesktopLearningBanner: View {
    @Bindable var host: DesktopLearningHost
    var body: some View {
        HStack(spacing: 12) {
            if host.awaitingReady == nil { ProgressView().controlSize(.small) }
            else { Image(systemName: "pause.circle").foregroundStyle(.secondary) }
            VStack(alignment: .leading, spacing: 3) {
                Text(host.phase).fontWeight(.medium)
                if let progress = host.progress {
                    Text("\(progress.completedUpdates.formatted()) updates · \(progress.completedEpisodes.formatted()) episodes · \(progress.learningDecisions.formatted()) rollout decisions")
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
            }
            Spacer()
            if host.awaitingReady != nil {
                Button(host.countdown.map { "Starting in \($0)…" } ?? "Environment Ready") { host.confirmReady() }
                    .buttonStyle(.borderedProminent).disabled(host.isStopping || host.countdown != nil)
            }
            Button(host.isStopping ? "Stopping…" : "Stop", systemImage: "stop.fill") { host.requestStop() }.disabled(host.isStopping)
        }.padding(14).background(.tint.opacity(0.08)).accessibilityElement(children: .contain)
    }
}

struct DesktopTrainingView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var options = DesktopLearningOptions()
    @State private var sourceID: String?
    @State private var programID: UUID?
    @State private var showingRewards = false
    @State private var issue: String?
    @State private var initialized = false
    @State private var advancedExpanded = false
    @State private var pendingSurfaceBindings: [UUID: [String: String]] = [:]
    init(agent: AgentDocument, model: WorkspaceModel, showAdvanced: Bool = false) {
        self.agent = agent; self.model = model; _advancedExpanded = State(initialValue: showAdvanced)
    }
    private var source: CaptureSource? { model.sources.first { $0.id == sourceID } }
    private var program: RewardProgram? { model.rewardPrograms.first { $0.id == programID } }
    private var checkpoints: [CheckpointDocument] { model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true } }
    private var checkpoint: CheckpointDocument? { checkpoints.first { $0.id == options.initialCheckpointID } }
    private var host: DesktopLearningHost? { model.desktopLearning?.agentID == agent.id ? model.desktopLearning : nil }
    private var busy: Bool { model.isLearning || model.isRunningAgent || model.isRecording || model.recordingStarting || model.recordingStopping || model.isClosing }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Learn in a desktop environment").font(.title2.weight(.semibold))
                        Spacer()
                        Button("Rewards & Episodes", systemImage: "slider.horizontal.3") { showingRewards = true }.disabled(busy)
                    }
                    Text("Collect experience in a selected application or display. Learning updates run between episodes with controls released.")
                        .foregroundStyle(.secondary)
                }
                if let host, let progress = host.progress {
                    GroupBox("Session") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(host.phase)
                            Text("\(progress.completedEpisodes.formatted()) episodes · \(progress.completedUpdates.formatted()) updates")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                    }
                }
                if let message = issue ?? host?.failure { AttentionLabel(message: message) }
                let pending = model.pendingFeedback.filter { $0.agentID == agent.id && ![.completed, .discarded].contains($0.status) }
                if !pending.isEmpty {
                    GroupBox("Saved experience waiting for feedback") {
                        VStack(alignment: .leading, spacing: 14) {
                            ForEach(pending) { item in
                                VStack(alignment: .leading, spacing: 10) {
                                HStack(alignment: .top) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(item.checkpoint.name).fontWeight(.medium)
                                        Text("\(item.collectedDecisions.formatted()) decisions · \(item.fragments.count) saved segments · \(item.modifiedAt.formatted(date: .abbreviated, time: .shortened))")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    if item.collectedDecisions < (item.configuration.fields?["training"]?.fields?["rollout_decisions"]?.int ?? 0) {
                                        Button("Collect More") {
                                            guard let source else { return }
                                            do { try model.reopenFeedback(item, source: source, surfaceBindings: pendingSurfaceBindings[item.id] ?? [:]); issue = nil } catch { issue = error.localizedDescription }
                                        }.disabled(source == nil || busy || !canBind(savedProgram(item), source: source, choices: pendingSurfaceBindings[item.id] ?? [:]))
                                            .help("Choose the current environment and bind the saved reward sources; the policy and task stay unchanged.")
                                    }
                                    Button("Review / Learn") {
                                        do { try model.reopenFeedback(item); issue = nil } catch { issue = error.localizedDescription }
                                    }.disabled(busy)
                                }
                                if item.collectedDecisions < (item.configuration.fields?["training"]?.fields?["rollout_decisions"]?.int ?? 0), let source {
                                    if let savedProgram = savedProgram(item) {
                                        RewardSourceBindingsView(program: savedProgram, source: source, choices: Binding(
                                            get: { pendingSurfaceBindings[item.id] ?? [:] },
                                            set: { pendingSurfaceBindings[item.id] = $0 }))
                                            .disabled(busy)
                                    } else {
                                        AttentionLabel(message: "The saved reward definition is unavailable. Review the saved experience before collecting more.")
                                    }
                                }
                                }
                            }
                            Text("Review uses saved observations with controls released. Explicitly review zero feedback; unreviewed intervals remain unknown.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                    }
                }
                if let host, let warning = host.cleanupWarning {
                    GroupBox {
                        VStack(alignment: .leading, spacing: 10) {
                            AttentionLabel(message: warning.issue?.message ?? "Control cleanup could not be confirmed. Release any held controls manually.")
                            Button("I’ve Released the Held Controls") {
                                Task {
                                    do { try await model.acknowledgeDesktopCleanup(); issue = nil }
                                    catch { issue = error.localizedDescription }
                                }
                            }.disabled(host.isBusy || model.controlHistoryBusy)
                        }.padding(8)
                    }
                }
                GroupBox("Environment & feedback") {
                    VStack(alignment: .leading, spacing: 14) {
                        Picker("Environment", selection: $sourceID) {
                            Text("Choose an application, window, display or desktop").tag(nil as String?)
                            ForEach(model.sources) { Text($0.name).tag(Optional($0.id)) }
                        }
                        HStack {
                            Button("Refresh Environments") { Task { await model.refreshPermissionsAndSources() } }.disabled(model.refreshingSources)
                            if model.refreshingSources { ProgressView().controlSize(.small) }
                        }
                        if let message = model.sourceIssue { AttentionLabel(message: message) }
                        Picker("Reward definition", selection: $programID) {
                            Text("Choose rewards and episode boundaries").tag(nil as UUID?)
                            ForEach(model.rewardPrograms) { Text($0.name).tag(Optional($0.id)) }
                        }
                        if let program {
                            if let source { RewardSourceBindingsView(program: program, source: source, choices: $options.surfaceBindings) }
                            Text(program.resetPlan == nil ? "You confirm Ready after resetting each episode. Physical keyboard or pointer input takes over immediately."
                                 : "The saved reset runs between episodes. Physical keyboard or pointer input takes over immediately.")
                                .font(.caption).foregroundStyle(.secondary)
                            Text("Episode time limit: \(Double(program.maximumEpisodeMS) / 1000, specifier: "%.1f") seconds.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                }.disabled(busy)
                GroupBox("Policy & learning") {
                    VStack(alignment: .leading, spacing: 14) {
                        Picker("Starting point", selection: $options.initialCheckpointID) {
                            Text("New agent · pretrained vision").tag(nil as UUID?)
                            ForEach(checkpoints) { Text($0.name).tag(Optional($0.id)) }
                        }.onChange(of: options.initialCheckpointID) { _, _ in options.resume = false }
                        if checkpoint?.kind == "reinforcement", model.learningRuns.first(where: { $0.id == checkpoint?.runID })?.sourceKind == "desktop_rollout" {
                            Toggle("Resume saved desktop learning state", isOn: $options.resume)
                                .onChange(of: options.resume) { _, resuming in
                                    if resuming, let saved = model.learningRuns.first(where: { $0.id == checkpoint?.runID }) {
                                        options.iterations = min(100_000, max(options.iterations, saved.epoch + 1))
                                    }
                                }
                        }
                        Stepper(options.resume ? "Total update target: \(options.iterations)" : "Learning updates: \(options.iterations)", value: $options.iterations, in: 1...100_000)
                        Text(options.resume ? "Resume keeps the saved reward task, contexts, optimizer and random stream. A fresh physical reset is required."
                             : "Each update collects at least \(options.training.rolloutDecisions.formatted()) decisions and finishes the current episode.")
                            .font(.caption).foregroundStyle(.secondary)
                        if options.initialCheckpointID == nil { controls }
                        if !options.resume {
                            ContextValuePickers(vocabulary: activeVocabulary, sizes: activeContextSizes, indices: $options.contextIDs)
                        }
                        DisclosureGroup("Advanced learning settings", isExpanded: $advancedExpanded) { advanced.padding(.top, 10) }.disabled(options.resume)
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                }.disabled(busy)
                if let host, host.progress != nil {
                    ReinforcementProgressContent(metrics: host.metrics,
                        phase: model.learning?.isBusy == true ? (model.learning?.phase ?? host.phase) : host.phase,
                        rolloutTarget: options.resume ? nil : options.training.rolloutDecisions,
                        rolloutDecisions: host.progress?.learningDecisions, updates: host.metrics.last?.updates,
                        elapsedSeconds: model.learning?.elapsedSeconds, peakMemoryBytes: model.learning?.peakMemoryBytes)
                }
                HStack {
                    Text("Control stops on takeover, source changes or missed action deadlines.").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(options.resume ? "Resume Desktop Training" : "Start Desktop Training", systemImage: "play.fill") {
                        guard let source, let program else { return }
                        do { try model.startDesktopTraining(agent: agent, source: source, program: program, options: options); issue = nil }
                        catch { issue = error.localizedDescription }
                    }.buttonStyle(.borderedProminent).disabled(busy || source == nil || program == nil || (try? options.validated()) == nil
                        || !canBind(program, source: source, choices: options.surfaceBindings))
                }
            }.padding(.bottom, 24).padding(.trailing, 5)
        }.sheet(isPresented: $showingRewards) { RewardEditor(agent: agent, model: model) }
            .onAppear {
                guard !initialized else { return }; initialized = true
                options.initialCheckpointID = agent.selectedCheckpointID; programID = agent.rewardProgramID
            }.onChange(of: agent.rewardProgramID) { _, value in programID = value }
            .onChange(of: activeContextSizes) { _, sizes in options.contextIDs = sizes.map { _ in 0 } }
            .onChange(of: activeVocabulary) { _, _ in options.contextIDs = activeContextSizes.map { _ in 0 } }
            .task(id: options.initialCheckpointID) {
                await model.inspectCheckpointContexts(options.initialCheckpointID, agentID: agent.id)
                options.contextIDs = activeContextSizes.map { _ in 0 }
            }
    }

    private var activeVocabulary: ContextVocabulary? {
        options.initialCheckpointID == nil ? (try? model.contextVocabulary(for: agent)) : model.checkpointContextVocabulary
    }
    private func savedProgram(_ document: PendingFeedbackDocument) -> RewardProgram? {
        (try? document.configuration.required("rewardBinding").decode(RewardProgramBinding.self))?.definition
    }
    private func canBind(_ program: RewardProgram?, source: CaptureSource?, choices: [String: String]) -> Bool {
        guard let program, let source, let leaves = try? source.captureBindings() else { return false }
        return (try? RewardProgramBinding.bound(program, sourceIDs: choices,
            scope: ControlScope(surfaces: leaves.map { $0.surfaceDescriptor() }))) != nil
    }
    private var activeContextSizes: [Int] {
        options.initialCheckpointID == nil ? (activeVocabulary?.sizes ?? []) : model.checkpointContextSizes
    }

    private var controls: some View {
        DisclosureGroup("Controls the new agent can use") {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Pointer", selection: Binding(get: { options.actions.absolutePointer ? "absolute" : options.actions.relativePointer ? "relative" : "none" }, set: {
                    options.actions.absolutePointer = $0 == "absolute"; options.actions.relativePointer = $0 == "relative"
                })) { Text("Absolute position").tag("absolute"); Text("Relative movement").tag("relative"); Text("Disabled").tag("none") }
                Toggle("Scrolling", isOn: $options.actions.scroll)
                HStack {
                    Menu("Add Key") { ForEach(0..<128, id: \.self) { code in Button(KeyNames.name(code)) { options.actions.keyCodes.insert(code) }.disabled(options.actions.keyCodes.contains(code)) } }
                    Menu("Add Mouse Button") { ForEach(0..<32, id: \.self) { code in Button(KeyNames.button(code)) { options.actions.mouseButtons.insert(code) }.disabled(options.actions.mouseButtons.contains(code)) } }
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 145), alignment: .leading)], alignment: .leading, spacing: 8) {
                    ForEach(options.actions.keyCodes.sorted(), id: \.self) { code in
                        Button { options.actions.keyCodes.remove(code) } label: { Label(KeyNames.name(code), systemImage: "xmark.circle") }.help("Remove \(KeyNames.name(code))")
                    }
                    ForEach(options.actions.mouseButtons.sorted(), id: \.self) { code in
                        Button { options.actions.mouseButtons.remove(code) } label: { Label(KeyNames.button(code), systemImage: "xmark.circle") }.help("Remove \(KeyNames.button(code))")
                    }
                }
                Text("Saved checkpoints retain their trained controls and timing.").font(.caption).foregroundStyle(.secondary)
            }.padding(.top, 10)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private var advanced: some View {
        VStack(alignment: .leading, spacing: 12) {
            Stepper("Minimum rollout: \(options.training.rolloutDecisions) decisions", value: $options.training.rolloutDecisions, in: 1...65_536)
            Stepper("PPO epochs: \(options.training.epochs)", value: $options.training.epochs, in: 1...100)
            Stepper("Contiguous sequence: \(options.training.sequenceLength)", value: $options.training.sequenceLength, in: 1...512)
            Stepper("Recurrent burn-in: \(options.training.burnIn)", value: $options.training.burnIn, in: 0...4096)
            Stepper("Effective batch: \(options.training.effectiveBatchDecisions) decisions", value: $options.training.effectiveBatchDecisions, in: 1...65_536)
            numberField("Learning rate", value: $options.training.learningRate)
            numberField("Pretrained vision learning rate", value: $options.training.pretrainedLearningRate)
            numberField("PPO clipping", value: $options.training.clipRatio)
            numberField("KL guard", value: $options.training.targetKL)
            numberField("Entropy coefficient", value: $options.training.entropyCoefficient)
            numberField("Reward discount half-life (seconds)", value: $options.training.discountHalfLifeSeconds)
            LabeledContent("Experiment seed") {
                TextField("Experiment seed", value: $options.training.seed, format: .number.grouping(.never)).textFieldStyle(.roundedBorder).labelsHidden().frame(width: 140)
            }
            if options.initialCheckpointID == nil {
                Picker("Decision rate", selection: $options.periodMS) { Text("10 Hz").tag(100); Text("20 Hz").tag(50) }
                Stepper("Execution lead: \(options.leadMS) ms", value: $options.leadMS, in: 1...2000, step: 5)
                Picker("Commands per decision", selection: $options.packetCapacity) { ForEach([16, 32, 64], id: \.self) { Text("\($0)").tag($0) } }
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private func numberField(_ label: String, value: Binding<Double>) -> some View {
        LabeledContent(label) {
            TextField(label, value: value, format: .number.precision(.significantDigits(1...8)))
                .textFieldStyle(.roundedBorder).labelsHidden().frame(width: 140)
        }
    }
}

/// The same explicit source selection is used for a fresh run and for saved
/// feedback collection. Changing the source never remaps by title or position.
private struct RewardSourceBindingsView: View {
    let program: RewardProgram
    let source: CaptureSource
    @Binding var choices: [String: String]
    private var references: [String] { RewardProgramBinding.referencedSurfaces(in: program).sorted() }
    private var leaves: [CaptureSource] { (try? source.captureBindings()) ?? [] }
    private var single: Bool { references.count <= 1 && leaves.count == 1 }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !references.isEmpty {
                Text("Reward & reset sources").font(.subheadline.weight(.medium))
                if single, let leaf = leaves.first {
                    Text(leaf.name).foregroundStyle(.secondary)
                } else {
                    ForEach(references, id: \.self) { reference in
                        Picker(label(reference), selection: Binding(get: { choices[reference] }, set: { choices[reference] = $0 })) {
                            Text("Choose a captured surface").tag(nil as String?)
                            ForEach(leaves) { leaf in Text(leaf.name).tag(Optional(leaf.id)) }
                        }.help("Reference surface: \(reference). Used by the saved reward definition and reset actions.")
                    }
                    Text("Match each saved source to the surface it observes now. These choices also apply to reset actions.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .onAppear(perform: normalize)
        .onChange(of: source) { _, _ in normalize() }
        .onChange(of: program) { _, _ in normalize() }
    }
    private func label(_ reference: String) -> String {
        let names = program.signals.filter { $0.surfaceID == reference }.map(\.name)
        return names.isEmpty ? "Reset source · \(reference)" : names.joined(separator: ", ")
    }
    private func normalize() {
        let available = Set(leaves.map(\.id))
        var next: [String: String] = [:]
        for reference in references {
            if single, let id = leaves.first?.id { next[reference] = id }
            else if let selected = choices[reference], available.contains(selected) { next[reference] = selected }
            else if available.contains(reference) { next[reference] = reference }
        }
        if next != choices { choices = next }
    }
}
