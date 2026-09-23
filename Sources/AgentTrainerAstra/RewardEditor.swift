import SwiftUI
import AppKit
import AstraCore
import AstraPlatform

struct RewardEditor: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @Environment(\.dismiss) private var dismiss
    @State private var program: RewardProgram
    @State private var page = "Signals"
    @State private var recordingID: UUID?
    @State private var selectedSignal: UUID?
    @State private var preview = RecordingPreviewModel()
    @State private var readings: [SignalReading] = []
    @State private var testing = false
    @State private var saving = false
    @State private var issue: String?
    @State private var rehearsalReport: RewardRehearsalReport?
    @State private var rehearsalReportURL: URL?
    @State private var rehearsalSeconds: Double = 10
    @State private var rehearsalProgress: Double = 0
    @State private var rehearsalWork: Task<Void, Never>?
    @State private var rehearsalGeneration = UUID()

    init(agent: AgentDocument, model: WorkspaceModel, initialPage: String = "Signals", referenceRecordingID: UUID? = nil) {
        self.agent = agent; self.model = model
        _page = State(initialValue: initialPage); _recordingID = State(initialValue: referenceRecordingID)
        let selected = model.rewardPrograms.first { $0.id == agent.rewardProgramID }
        _program = State(initialValue: selected ?? RewardProgram(name: String(agent.name.prefix(145)) + " rewards", rules: [
            .init(name: "Positive feedback", kind: .manualMarker, amount: 1), .init(name: "Negative feedback", kind: .manualMarker, amount: -1)]))
        _selectedSignal = State(initialValue: selected?.signals.first?.id)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Rewards & episodes").font(.title2.weight(.semibold))
                    Text("Define feedback and starting conditions, then check visual signals against a recording.").foregroundStyle(.secondary)
                }
                Spacer()
                if !model.rewardPrograms.isEmpty {
                    Menu("Use Saved Definition") {
                        ForEach(model.rewardPrograms) { saved in Button(saved.name) { program = saved; selectedSignal = saved.signals.first?.id; readings = []; issue = nil } }
                    }.disabled(testing || saving)
                }
            }
            LabeledContent("Definition") { TextField("Definition name", text: $program.name).labelsHidden() }
            Picker("Editor section", selection: $page) {
                ForEach(["Signals", "Rewards", "Episode", "Rehearse"], id: \.self) { Text($0).tag($0) }
            }.pickerStyle(.segmented).disabled(testing || saving)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    switch page { case "Signals": signalsEditor; case "Rewards": rulesEditor; case "Episode": episodeEditor; default: rehearsal }
                }.padding(.vertical, 6).padding(.trailing, 4)
            }.disabled(saving)
            if let issue { AttentionLabel(message: issue).textSelection(.enabled) }
            Divider()
            HStack {
                Text("Saving keeps earlier definitions intact.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(saving)
                Button(saving ? "Saving…" : "Save Definition") { save() }.buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction).disabled(saving || testing)
            }
        }.padding(24).frame(minWidth: 760, idealWidth: 900, minHeight: 560, idealHeight: 720)
            .interactiveDismissDisabled(saving).onDisappear { preview.close(); rehearsalGeneration = UUID(); rehearsalWork?.cancel() }
            .task(id: recordingID) {
                readings = []
                guard let recordingID else { preview.close(); return }
                await preview.open(model.recordingDirectory(recordingID))
            }
    }
    private var sourceChooser: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Reference recording", selection: $recordingID) {
                Text("Choose a saved recording").tag(nil as UUID?)
                ForEach(model.recordings.filter { $0.status != .recording && $0.frameCount > 0 }) { Text($0.name).tag(Optional($0.id)) }
            }.disabled(testing)
            if let image = preview.image, let frame = preview.preview {
                RewardRegionPreview(image: image, surface: frame.frame.surface, region: page == "Signals" ? selectedRegion : nil)
                if selectedRegion != nil && page == "Signals" { Text("Drag over the part of the image this signal should read.").font(.caption).foregroundStyle(.secondary) }
                HStack {
                    Slider(value: Binding(get: { preview.seconds }, set: { preview.seek($0); readings = [] }), in: 0...max(preview.maximumSeconds, 0.001))
                        .accessibilityLabel("Reference time").disabled(testing)
                    Text("\(preview.seconds, specifier: "%.2f") s").monospacedDigit().frame(width: 80, alignment: .trailing)
                }
            }
            if preview.loading { ProgressView("Reading frame…") }
            if let issue = preview.issue { AttentionLabel(message: issue) }
        }
    }
    private var selectedRegion: Binding<Rect2D?>? {
        guard let id = selectedSignal, let signal = program.signals.first(where: { $0.id == id }),
              signal.kind.isVisual, signal.surfaceID == preview.preview?.frame.surface.id else { return nil }
        return Binding(get: { program.signals.first(where: { $0.id == id })?.region }, set: { value in
            if let index = program.signals.firstIndex(where: { $0.id == id }) { program.signals[index].region = value; readings = [] }
        })
    }
    private var signalsEditor: some View {
        VStack(alignment: .leading, spacing: 16) {
            sourceChooser
            HStack {
                Picker("Signal", selection: $selectedSignal) {
                    Text("Choose a signal").tag(nil as UUID?)
                    ForEach(program.signals) { Text($0.name).tag(Optional($0.id)) }
                }
                Menu("Add Signal", systemImage: "plus") { ForEach(RewardSignalKind.allCases, id: \.self) { kind in Button(kind.title) { addSignal(kind) } } }
                    .disabled(program.signals.count >= 32)
            }
            ForEach($program.signals) { $signal in
                if signal.id == selectedSignal {
                    RewardSignalEditor(signal: $signal, frame: preview.preview, root: model.supportRoot,
                        onIssue: { issue = $0 }, onTemplate: { id, digest in
                            if let index = program.signals.firstIndex(where: { $0.id == id }) { program.signals[index].templateDigest = digest }
                        })
                    Button("Remove Signal", role: .destructive) {
                        let id = signal.id
                        guard !isReferenced(id) else { issue = "Remove this signal's reward and episode conditions first."; return }
                        program.signals.removeAll { $0.id == id }; selectedSignal = program.signals.first?.id
                    }
                }
            }
            if program.signals.isEmpty { Text("Manual feedback works without visual signals. Add a signal to read text, a score, an image or elapsed time.").foregroundStyle(.secondary) }
        }
    }
    private var rulesEditor: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Positive rewards encourage behavior; negative rewards discourage it.").foregroundStyle(.secondary)
            ForEach($program.rules) { $rule in
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack { TextField("Rule name", text: $rule.name); Spacer(); Text(rule.kind.title).foregroundStyle(.secondary) }
                        RewardNumberField(title: rule.kind == .ratePerSecond ? "Reward per second" : rule.kind == .scoreDelta ? "Reward per score unit" : "Reward", value: $rule.amount)
                        if rule.kind == .scoreDelta {
                            Picker("Score signal", selection: $rule.signalID) {
                                Text("Choose a numeric signal").tag(nil as UUID?)
                                ForEach(program.signals.filter { [.manual, .ocrNumber].contains($0.kind) }) { Text($0.name).tag(Optional($0.id)) }
                            }
                            RewardNumberField(title: "Largest valid score change", value: $rule.maximumDelta)
                        }
                        if rule.kind == .risingEdge || rule.kind == .ratePerSecond {
                            Toggle("Only when a condition matches", isOn: Binding(get: { rule.predicate != nil }, set: { rule.predicate = $0 ? defaultPredicate() : nil }))
                                .disabled(rule.kind == .risingEdge)
                            if let binding = Binding($rule.predicate) { RewardPredicateEditor(predicate: binding, signals: program.signals) }
                        }
                        if rule.kind == .manualMarker { Text("Each explicit press of this feedback control contributes once.").font(.caption).foregroundStyle(.secondary) }
                        Button("Remove Rule", role: .destructive) { let id = rule.id; program.rules.removeAll { $0.id == id } }
                    }.padding(6)
                }
            }
            Menu("Add Reward Rule", systemImage: "plus") {
                ForEach(RewardRuleKind.allCases, id: \.self) { kind in Button(kind.title) {
                    program.rules.append(.init(name: kind.title, kind: kind, amount: kind == .ratePerSecond ? -0.01 : 1,
                        predicate: kind == .risingEdge ? defaultPredicate() : nil,
                        signalID: kind == .scoreDelta ? program.signals.first(where: { [.manual, .ocrNumber].contains($0.kind) })?.id : nil))
                } }
            }.disabled(program.rules.count >= 64)
        }
    }
    private var episodeEditor: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Readiness establishes fresh reward baselines. Reset actions happen outside the policy episode, and all owned controls must be released first.").foregroundStyle(.secondary)
            Stepper("Episode time limit: \(program.maximumEpisodeMS / 1000) seconds", value: $program.maximumEpisodeMS, in: 1000...3_600_000, step: 1000)
            if program.resetPlan != nil {
                DisclosureGroup("Reference recording for pointer actions") { sourceChooser }
            }
            ResetPlanEditor(plan: $program.resetPlan, signals: program.signals, referenceSurface: preview.preview?.frame.surface)
                .id(program.id)
            if program.resetPlan != nil {
                Text("Authored resets require a starting condition below. This editor saves the plan; it does not execute reset actions.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            predicateSection("Starting condition", value: $program.ready)
            predicateSection("Success", value: $program.success)
            predicateSection("Failure", value: $program.failure)
            Text("The time limit truncates an episode. Unreadable conditions remain unknown; matching success and failure together is an error.").font(.caption).foregroundStyle(.secondary)
        }
    }
    private func predicateSection(_ title: String, value: Binding<RewardPredicate?>) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Toggle(title, isOn: Binding(get: { value.wrappedValue != nil }, set: { value.wrappedValue = $0 ? defaultPredicate() : nil }))
                if let binding = Binding(value) { RewardPredicateEditor(predicate: binding, signals: program.signals) }
            }.padding(6)
        }
    }
    private var rehearsal: some View {
        VStack(alignment: .leading, spacing: 16) {
            sourceChooser
            Text("Check regions and confidence with the exact visual detectors, without controlling the computer.").foregroundStyle(.secondary)
            Button(testing ? "Reading Signals…" : "Read Signals from This Frame", systemImage: "viewfinder") { testFrame() }
                .buttonStyle(.borderedProminent).disabled(testing || preview.loading || preview.preview == nil)
            ForEach(program.signals) { signal in
                HStack(alignment: .top) {
                    Text(signal.name).fontWeight(.medium).frame(width: 170, alignment: .leading)
                    if let reading = readings.first(where: { $0.signalID == signal.id }) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(reading.value.displayText).textSelection(.enabled)
                            Text("Confidence: \(reading.confidence, format: .percent.precision(.fractionLength(0)))").font(.caption).foregroundStyle(.secondary)
                        }
                    } else { Text(signal.kind.isVisual ? "Not read in this frame" : "Supplied during the episode").foregroundStyle(.secondary) }
                    Spacer()
                }
            }
            Divider()
            HStack {
                Stepper("Rehearse \(Int(rehearsalSeconds)) seconds from this point", value: $rehearsalSeconds, in: 1...300).disabled(testing)
                Spacer()
                if rehearsalWork != nil { Button("Cancel Rehearsal") { rehearsalWork?.cancel() } }
                else { Button("Rehearse Sequence", systemImage: "play.rectangle") { rehearseSequence() }.disabled(testing || preview.loading || recordingID == nil) }
            }
            if rehearsalWork != nil { ProgressView("Evaluating recorded observations…", value: rehearsalProgress) }
            if let report = rehearsalReport {
                Text(report.readinessReached ? "\(report.evaluations.count) intervals · \(report.evaluations.filter { $0.value == nil }.count) with unknown reward"
                     : "The starting condition was not confirmed in this range.").fontWeight(.medium)
                Table(report.evaluations) {
                    TableColumn("Time") { value in Text(Double(value.endNanos - (report.evaluations.first?.startNanos ?? value.endNanos)) / 1e9, format: .number.precision(.fractionLength(2))).monospacedDigit() }
                    TableColumn("Reward") { value in Text(value.value.map { $0.formatted() } ?? "Unknown").monospacedDigit() }
                    TableColumn("Outcome") { value in Text(value.outcome.rawValue.capitalized) }
                }.frame(height: 160)
                if let rehearsalReportURL { Button("Show Saved Report", systemImage: "doc.text") { NSWorkspace.shared.activateFileViewerSelecting([rehearsalReportURL]) } }
            }
            Text("Sequence rehearsal uses 100 ms decision intervals and stops at the first terminal condition. Manual values and feedback are not invented from recordings; affected rewards remain unknown.").font(.caption).foregroundStyle(.secondary)
        }
    }
    private func addSignal(_ kind: RewardSignalKind) {
        let signal = RewardSignal(name: kind.title, kind: kind, surfaceID: kind.isVisual ? preview.preview?.frame.surface.id : nil,
            region: kind.isVisual ? .init(x: 0, y: 0, width: 1, height: 1) : nil)
        program.signals.append(signal); selectedSignal = signal.id
    }
    private func defaultPredicate() -> RewardPredicate { .init(conditions: program.signals.first.map { [RewardPredicateEditor.condition(for: $0)] } ?? []) }
    private func isReferenced(_ id: UUID) -> Bool {
        program.rules.contains { $0.signalID == id || $0.predicate?.conditions.contains(where: { $0.signalID == id }) == true }
        || [program.ready, program.success, program.failure].compactMap({ $0 }).contains { $0.conditions.contains { $0.signalID == id } }
        || program.resetPlan?.steps.contains { $0.condition?.conditions.contains { $0.signalID == id } == true } == true
    }
    private func save() {
        saving = true; issue = nil
        var frozen = program; frozen.id = UUID()
        Task { defer { saving = false }; do { try await model.saveRewardProgram(frozen, for: agent.id); dismiss() } catch { issue = error.localizedDescription } }
    }
    private func testFrame() {
        guard let frame = preview.preview else { return }
        testing = true; issue = nil; readings = []
        let signals = program.signals, root = model.supportRoot
        Task {
            defer { testing = false }
            do {
                readings = try await Task.detached {
                    var templates: [String: Data] = [:]
                    for digest in Set(signals.compactMap(\.templateDigest)) { templates[digest] = try RewardAssets.read(digest, root: root) }
                    return try VisualRewardDetector.read(signals: signals, frames: [.init(metadata: frame.frame, pixels: frame.pixels)], episodeID: UUID(), templates: templates)
                }.value
            } catch { issue = error.localizedDescription }
        }
    }
    private func rehearseSequence() {
        guard let recordingID else { return }
        testing = true; issue = nil; rehearsalReport = nil; rehearsalReportURL = nil; rehearsalProgress = 0
        let generation = UUID(); rehearsalGeneration = generation
        let frozen = program, start = preview.seconds, duration = rehearsalSeconds, root = model.supportRoot
        let recordingDirectory = model.recordingDirectory(recordingID)
        rehearsalWork = Task {
            defer { testing = false; rehearsalWork = nil }
            do {
                let work = Task.detached {
                    try RewardRehearsal.run(program: frozen, directory: recordingDirectory,
                        assetRoot: root, startSeconds: start, durationSeconds: duration, cancelled: { Task.isCancelled },
                        progress: { completed, total in
                            if completed % 10 == 0 || completed == total {
                                Task { @MainActor in if rehearsalGeneration == generation { rehearsalProgress = Double(completed) / Double(max(total, 1)) } }
                            }
                        })
                }
                let result = try await withTaskCancellationHandler(operation: { try await work.value }, onCancel: { work.cancel() })
                try Task.checkCancellation(); rehearsalReport = result
                rehearsalReportURL = try await Task.detached { try RewardRehearsal.save(result, root: root) }.value
            } catch is CancellationError { issue = "Rehearsal cancelled. The recording and definition are unchanged." }
            catch { issue = error.localizedDescription }
        }
    }
}
