import SwiftUI
import Charts
import AstraCore

struct ReinforcementOptions: Sendable {
    var initialCheckpointID: UUID?
    var resume = false
    var iterations = 20
    var task = "pointing"
    var delayMS = 2000
    var seed = 0
    var periodMS = 100
    var leadMS = 100
    var packetCapacity = 16
    var rolloutDecisions = 512
    var epochs = 4
    var sequenceLength = 64
    var burnIn = 32
    var effectiveBatchDecisions = 256
    var learningRate = 0.0001
    var pretrainedLearningRate = 0.00001
    var clipRatio = 0.2
    var targetKL = 0.02
    var entropyCoefficient = 0.01
    var shapingScale = 0.1

    var settings: ReinforcementSettings {
        var value = ReinforcementSettings()
        value.rolloutDecisions = rolloutDecisions; value.epochs = epochs; value.sequenceLength = sequenceLength
        value.burnIn = burnIn; value.effectiveBatchDecisions = effectiveBatchDecisions
        value.learningRate = learningRate; value.pretrainedLearningRate = pretrainedLearningRate
        value.clipRatio = clipRatio; value.targetKL = targetKL; value.entropyCoefficient = entropyCoefficient; value.seed = seed
        return value
    }

    func validated() throws -> Self {
        if resume {
            guard initialCheckpointID != nil, (1...100_000).contains(iterations) else {
                throw AstraError("reinforcement.resume", "Choose a saved reinforcement checkpoint and a valid total iteration target.")
            }
            return self
        }
        _ = try settings.validated()
        guard (1...100_000).contains(iterations), ["pointing", "delayed_memory"].contains(task),
              [2000, 8000, 30000].contains(delayMS), (0...1_000_000_000).contains(seed),
              [50, 100].contains(periodMS), (0...2000).contains(leadMS), [16, 32, 64].contains(packetCapacity),
              shapingScale.isFinite, shapingScale >= 0, shapingScale <= 1,
              !resume || initialCheckpointID != nil else {
            throw AstraError("reinforcement.options", "Choose a practice environment and valid reinforcement settings before starting.")
        }
        return self
    }

    var model: JSONValue {
        .object(["period_ms": .integer(Int64(periodMS)), "lead_ms": .integer(Int64(leadMS)),
                 "packet_capacity": .integer(Int64(packetCapacity))])
    }
    var actions: JSONValue {
        .object(["keyCodes": .array((task == "delayed_memory" ? [123, 124] : []).map { .integer(Int64($0)) }),
                 "mouseButtons": .array([.integer(0)]), "absolutePointer": .bool(true), "relativePointer": .bool(false),
                 "scroll": .bool(false), "scrollUnitsPerPoint": .integer(8)])
    }
    var environment: JSONValue {
        .object(["task": .string(task), "seed": .integer(Int64(seed)), "period_ms": .integer(Int64(periodMS)),
                 "lead_ms": .integer(Int64(leadMS)), "delay_ms": .integer(Int64(delayMS)),
                 "shaping_scale": .number(task == "pointing" ? shapingScale : 0), "discount_half_life_ms": .number(30000)])
    }
    var training: JSONValue {
        settings.payload
    }
}

struct ReinforcementMetric: Identifiable, Sendable {
    let iteration: Int
    let elapsedSeconds: Double
    let reward: Double
    let valueLoss: Double
    let policyLoss: Double
    let kl: Double
    let klIsAccepted: Bool
    let candidateKL: Double?
    let hasCandidateKL: Bool
    let backtrackCount: Int?
    let rejectedUpdates: Int?
    let minimumStepScale: Double?
    let clipFraction: Double
    let updates: Int
    var id: Int { iteration }

    init?(_ fields: [String: JSONValue]) {
        guard let iteration = fields["iteration"]?.int, iteration > 0,
              let elapsed = fields["elapsed_seconds"]?.double, elapsed.isFinite, elapsed >= 0,
              let reward = fields["mean_reward"]?.double, reward.isFinite,
              let value = fields["mean_value_loss"]?.double, value.isFinite, value >= 0,
              let policy = fields["mean_policy_loss"]?.double, policy.isFinite,
              let kl = (fields["maximum_accepted_kl"] ?? fields["maximum_sampled_kl"])?.double, kl.isFinite, kl >= 0,
              let clip = fields["clip_fraction"]?.double, clip.isFinite, (0...1).contains(clip),
              let updates = fields["optimizer_updates"]?.int, updates >= 0,
              fields["maximum_candidate_kl"].map({ $0 == .null || $0.double.map({ $0.isFinite && $0 >= 0 }) == true }) ?? true,
              fields["backtrack_count"].map({ $0.int.map({ $0 >= 0 }) == true }) ?? true,
              fields["rejected_optimizer_steps"].map({ $0.int.map({ $0 >= 0 }) == true }) ?? true,
              fields["minimum_step_scale"].map({ $0.double.map({ $0.isFinite && $0 > 0 && $0 <= 1 }) == true }) ?? true else { return nil }
        self.iteration = iteration; elapsedSeconds = elapsed; self.reward = reward; valueLoss = value
        policyLoss = policy; self.kl = kl; clipFraction = clip; self.updates = updates
        klIsAccepted = fields["maximum_accepted_kl"] != nil
        candidateKL = fields["maximum_candidate_kl"]?.double; hasCandidateKL = fields["maximum_candidate_kl"] != nil
        backtrackCount = fields["backtrack_count"]?.int; rejectedUpdates = fields["rejected_optimizer_steps"]?.int
        minimumStepScale = fields["minimum_step_scale"]?.double
    }
}

struct LearningTrainingView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var mode: LearningKind = .behavioral
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Picker("Training method", selection: $mode) {
                Text("Behavioral learning").tag(LearningKind.behavioral)
                Text("Reinforcement learning").tag(LearningKind.reinforcement)
            }.pickerStyle(.segmented).frame(maxWidth: 560)
            if mode == .behavioral { BehaviorTrainingView(agent: agent, model: model) }
            else { ReinforcementTrainingView(agent: agent, model: model) }
        }.onAppear {
            if model.desktopLearning?.agentID == agent.id, model.desktopLearning?.isBusy == true { mode = .reinforcement }
            else if model.learning?.activeRun?.agentID == agent.id, let kind = model.learning?.activeRun?.kind { mode = kind }
        }
    }
}

struct ReinforcementTrainingView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var desktop = true
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Picker("Learning environment", selection: $desktop) {
                Text("Desktop environment").tag(true)
                Text("Practice environment").tag(false)
            }.pickerStyle(.segmented).labelsHidden().frame(maxWidth: 420)
            if desktop { DesktopTrainingView(agent: agent, model: model) }
            else { PracticeReinforcementTrainingView(agent: agent, model: model) }
        }.onAppear {
            if model.learning?.activeRun?.agentID == agent.id, model.learning?.activeRun?.sourceKind == "practice_rollout" { desktop = false }
        }
    }
}

struct PracticeReinforcementTrainingView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var options = ReinforcementOptions()
    @State private var failure: String?
    @State private var showingRewards = false
    private var checkpoints: [CheckpointDocument] {
        model.checkpoints.filter { model.checkpointLinks[agent.id]?.contains($0.id) == true }
    }
    private var resumableSelection: Bool {
        guard let selected = checkpoints.first(where: { $0.id == options.initialCheckpointID }), selected.kind == "reinforcement" else { return false }
        return model.learningRuns.first { $0.id == selected.runID }?.sourceKind == "practice_rollout"
    }
    private var ownsRun: Bool { model.learning?.activeRun?.agentID == agent.id && model.learning?.activeRun?.kind == .reinforcement }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Reinforcement learning").font(.title2.weight(.semibold))
                        Spacer()
                        Button("Rewards & Episodes", systemImage: "slider.horizontal.3") { showingRewards = true }
                    }
                    Text("Let the agent collect its own experience and improve from rewards. The actor keeps a fixed policy during each episode.")
                        .foregroundStyle(.secondary)
                }
                if ownsRun, let learning = model.learning { ReinforcementProgressView(learning: learning) }
                if let message = failure ?? (ownsRun ? model.learning?.failure : nil) {
                    AttentionLabel(message: message)
                }
                GroupBox {
                    VStack(alignment: .leading, spacing: 14) {
                        if options.resume {
                            Text("The saved checkpoint supplies its original practice environment and reward settings.").foregroundStyle(.secondary)
                        } else {
                        Picker("Environment", selection: $options.task) {
                            Text("Practice · Pointing").tag("pointing")
                            Text("Practice · Delayed visual memory").tag("delayed_memory")
                        }.disabled(options.resume)
                        Text("These environments run locally without screen or input permissions. Experience comes from the agent's actions, not generated demonstrations.")
                            .font(.callout).foregroundStyle(.secondary)
                        if options.task == "delayed_memory" {
                            Picker("Memory delay", selection: $options.delayMS) {
                                Text("2 seconds").tag(2000); Text("8 seconds").tag(8000); Text("30 seconds").tag(30000)
                            }.disabled(options.resume)
                            Text("Remember the visible cue, then choose the corresponding target after the delay. Rewards do not reveal the hidden answer.")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text("Reach and click targets. Optional distance shaping adds a potential-based reward while preserving the task's reward objective.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        }
                    }.padding(8)
                } label: { Text("Environment").font(.headline) }
                GroupBox {
                    VStack(alignment: .leading, spacing: 14) {
                        Picker("Starting point", selection: $options.initialCheckpointID) {
                            Text("New agent · pretrained vision").tag(nil as UUID?)
                            ForEach(checkpoints) { Text($0.name).tag(Optional($0.id)) }
                        }.onChange(of: options.initialCheckpointID) { _, _ in options.resume = false }
                        if resumableSelection {
                            Toggle("Resume optimizer and iteration state", isOn: $options.resume)
                        }
                        Text(options.resume
                             ? "Resume with the checkpoint's saved environment and training settings. The iteration target includes already completed iterations; the practice world starts with a confirmed reset."
                             : "A new run learns all model layers. Starting checkpoints must use controls compatible with the selected practice task.")
                            .font(.caption).foregroundStyle(.secondary)
                        Stepper(options.resume ? "Total iteration target: \(options.iterations)" : "Iterations: \(options.iterations)", value: $options.iterations, in: 1...100_000)
                        Text(options.resume ? "The saved configuration determines rollout size, recurrent sequences and PPO epochs."
                             : "Each iteration collects at least \(options.rolloutDecisions.formatted()) decisions and finishes the current episode, then trains on contiguous sequences for up to \(options.epochs) PPO epochs.")
                            .font(.caption).foregroundStyle(.secondary)
                        DisclosureGroup("Advanced") {
                            VStack(alignment: .leading, spacing: 14) {
                                Stepper("Minimum rollout: \(options.rolloutDecisions) decisions", value: $options.rolloutDecisions, in: 1...65536)
                                Stepper("PPO epochs: \(options.epochs)", value: $options.epochs, in: 1...100)
                                Stepper("Contiguous sequence: \(options.sequenceLength)", value: $options.sequenceLength, in: 1...512)
                                Stepper("Recurrent burn-in: \(options.burnIn)", value: $options.burnIn, in: 0...4096)
                                Stepper("Effective batch: \(options.effectiveBatchDecisions) decisions", value: $options.effectiveBatchDecisions, in: 1...65536)
                                numberField("Learning rate", value: $options.learningRate)
                                numberField("Pretrained vision learning rate", value: $options.pretrainedLearningRate)
                                numberField("PPO clipping", value: $options.clipRatio)
                                numberField("KL guard", value: $options.targetKL)
                                numberField("Entropy coefficient", value: $options.entropyCoefficient)
                                if options.task == "pointing" { numberField("Distance shaping", value: $options.shapingScale) }
                                TextField("Experiment seed", value: $options.seed, format: .number.grouping(.never)).textFieldStyle(.roundedBorder)
                                if options.initialCheckpointID == nil {
                                    Picker("Decision rate", selection: $options.periodMS) { Text("10 Hz").tag(100); Text("20 Hz").tag(50) }
                                    Stepper("Execution lead: \(options.leadMS) ms", value: $options.leadMS, in: 0...2000, step: 5)
                                    Picker("Commands per decision", selection: $options.packetCapacity) {
                                        ForEach([16, 32, 64], id: \.self) { Text("\($0)").tag($0) }
                                    }
                                }
                            }.disabled(options.resume).padding(.top, 12)
                        }
                    }.padding(8)
                } label: { Text("Learning").font(.headline) }
                HStack {
                    Text("Updates activate only after an episode reset.").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(options.resume ? "Resume Training" : "Start Training", systemImage: "play.fill") {
                        do {
                            try model.startReinforcementTraining(agent: agent, options: options); failure = nil
                        }
                        catch { failure = error.localizedDescription }
                    }.buttonStyle(.borderedProminent)
                        .disabled(model.isClosing || model.isRunningAgent || model.learning == nil || model.learning?.isBusy == true || (try? options.validated()) == nil)
                }
                LearningRunList(runs: model.learningRuns.filter { $0.agentID == agent.id }, model: model, compact: true)
            }.padding(.trailing, 8).padding(.bottom, 20)
        }.sheet(isPresented: $showingRewards) { RewardEditor(agent: agent, model: model) }
    }
    private func numberField(_ title: String, value: Binding<Double>) -> some View {
        TextField(title, value: value, format: .number.precision(.fractionLength(1...6))).textFieldStyle(.roundedBorder)
    }
}

struct ReinforcementProgressView: View {
    @Bindable var learning: LearningCoordinator
    var body: some View {
        ReinforcementProgressContent(metrics: learning.reinforcementMetrics, phase: learning.phase,
            rolloutTarget: learning.rolloutTarget, rolloutDecisions: learning.rolloutDecisions,
            updates: learning.activeRun?.updates, elapsedSeconds: learning.elapsedSeconds, peakMemoryBytes: learning.peakMemoryBytes)
    }
}

struct ReinforcementProgressContent: View {
    let metrics: [ReinforcementMetric]
    let phase: String
    let rolloutTarget: Int?
    let rolloutDecisions: Int?
    let updates: Int?
    let elapsedSeconds: Double?
    let peakMemoryBytes: Int?
    @State private var chartMetric = "reward"
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Text(phase).foregroundStyle(.secondary)
                if let target = rolloutTarget, let completed = rolloutDecisions {
                    ProgressView(completed >= target ? "\(completed.formatted()) decisions · finishing episode"
                                 : "Rollout \(completed.formatted()) / \(target.formatted()) minimum decisions",
                                 value: Double(min(completed, target)), total: Double(target))
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), alignment: .leading)], alignment: .leading, spacing: 14) {
                    if let point = metrics.last {
                        metric("Iteration", point.iteration.formatted())
                        metric("Mean reward / decision", point.reward.formatted(.number.precision(.fractionLength(4))))
                        metric("Value loss", point.valueLoss.formatted(.number.precision(.fractionLength(4))))
                        metric(point.klIsAccepted ? "Maximum accepted KL" : "Maximum sampled KL", point.kl.formatted(.number.precision(.fractionLength(4))))
                    }
                    if let updates { metric("Optimizer updates", updates.formatted()) }
                    if let elapsed = elapsedSeconds { metric("Elapsed", Duration.seconds(elapsed).formatted(.time(pattern: .hourMinuteSecond))) }
                }.monospacedDigit()
                if !metrics.isEmpty {
                    Picker("Chart", selection: $chartMetric) {
                        Text("Mean reward").tag("reward"); Text("Value loss").tag("value"); Text("Policy KL").tag("kl")
                    }.pickerStyle(.segmented)
                    Chart(metrics) { point in
                        let value = chartMetric == "reward" ? point.reward : chartMetric == "value" ? point.valueLoss : point.kl
                        LineMark(x: .value("Elapsed seconds", point.elapsedSeconds), y: .value("Value", value))
                        PointMark(x: .value("Elapsed seconds", point.elapsedSeconds), y: .value("Value", value)).symbolSize(18)
                    }.chartXAxisLabel("Elapsed seconds").frame(height: 170)
                        .accessibilityLabel("Reinforcement \(chartMetric) across completed iterations")
                    DisclosureGroup("Metrics table · recent iterations") {
                        ScrollView(.horizontal) {
                            Grid(alignment: .leading, horizontalSpacing: 22, verticalSpacing: 7) {
                                GridRow { Text("Iteration"); Text("Seconds"); Text("Reward"); Text("Value loss"); Text("KL"); Text("Clipped") }.fontWeight(.medium)
                                ForEach(metrics.suffix(20)) { point in
                                    GridRow {
                                        Text(point.iteration.formatted()); Text(point.elapsedSeconds.formatted(.number.precision(.fractionLength(1))))
                                        Text(point.reward.formatted(.number.precision(.fractionLength(4))))
                                        Text(point.valueLoss.formatted(.number.precision(.fractionLength(4))))
                                        Text(point.kl.formatted(.number.precision(.fractionLength(4))))
                                        Text(point.clipFraction.formatted(.percent.precision(.fractionLength(1))))
                                    }
                                }
                            }.font(.caption).monospacedDigit().padding(.top, 8)
                        }
                    }
                    if let point = metrics.last, point.hasCandidateKL || point.backtrackCount != nil || point.rejectedUpdates != nil {
                        DisclosureGroup("Policy update details") {
                            VStack(alignment: .leading, spacing: 8) {
                                if point.hasCandidateKL {
                                    LabeledContent("Maximum candidate KL", value: point.candidateKL.map { $0.formatted(.number.precision(.fractionLength(4))) } ?? "Non-finite candidate")
                                }
                                if let count = point.backtrackCount { LabeledContent("Backtracking attempts", value: count.formatted()) }
                                if let count = point.rejectedUpdates { LabeledContent("Rejected optimizer updates", value: count.formatted()) }
                                if point.updates > 0, let scale = point.minimumStepScale { LabeledContent("Smallest accepted step scale", value: scale.formatted(.percent.precision(.fractionLength(1)))) }
                            }.font(.caption).monospacedDigit().padding(.top, 8)
                        }
                    }
                }
                if let bytes = peakMemoryBytes {
                    Text("Peak model memory: \(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory))").font(.caption).foregroundStyle(.secondary)
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
        } label: { Text("Reinforcement progress").font(.headline) }
    }
    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption).foregroundStyle(.secondary); Text(value).font(.title3.weight(.medium)) }
    }
}
