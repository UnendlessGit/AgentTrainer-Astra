import SwiftUI
import ImageIO
import UniformTypeIdentifiers
import AstraCore
import AstraPlatform

struct RewardSignalEditor: View {
    @Binding var signal: RewardSignal
    let frame: RecordingPreview?
    let root: URL
    let onIssue: (String?) -> Void
    let onTemplate: (UUID, String) -> Void
    @State private var savingTemplate = false
    @State private var languages = ["en-US"]
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            LabeledContent("Name") { TextField("Signal name", text: $signal.name).labelsHidden() }
            LabeledContent("Source", value: signal.kind.title)
            if signal.kind.isVisual {
                Button(signal.surfaceID == frame?.frame.surface.id && frame != nil ? "Using This Recorded Surface" : "Use This Recorded Surface") {
                    signal.surfaceID = frame?.frame.surface.id
                }.disabled(frame == nil)
                if let region = Binding($signal.region) {
                    DisclosureGroup("Precise region") {
                        Text("Fractions of the visible source, from 0 to 1.").font(.caption).foregroundStyle(.secondary)
                        HStack { RewardNumberField(title: "Left", value: region.x); RewardNumberField(title: "Top", value: region.y) }
                        HStack { RewardNumberField(title: "Width", value: region.width); RewardNumberField(title: "Height", value: region.height) }
                    }
                }
                Slider(value: $signal.minimumConfidence, in: 0...1) { Text("Minimum confidence") }
                Text("Minimum confidence: \(signal.minimumConfidence, format: .percent.precision(.fractionLength(0)))").font(.caption)
                Stepper("Maximum signal age: \(signal.maximumAgeMS) ms", value: $signal.maximumAgeMS, in: 50...60_000, step: 50)
                if signal.kind == .ocrText || signal.kind == .ocrNumber {
                    Picker("Text language", selection: $signal.language) {
                        ForEach(languages, id: \.self) { code in Text(Locale.current.localizedString(forIdentifier: code) ?? code).tag(code) }
                        if !languages.contains(signal.language) { Text("\(signal.language) · unavailable").tag(signal.language) }
                    }
                }
                if signal.kind == .ocrNumber {
                    Picker("Decimal separator", selection: $signal.decimalSeparator) { Text("Period · 1,234.5").tag("."); Text("Comma · 1.234,5").tag(",") }
                }
                if signal.kind == .imageMatch {
                    Button(savingTemplate ? "Saving Template…" : "Use Selected Region as Image Template") { saveTemplate() }.disabled(frame == nil || savingTemplate)
                    Text(signal.templateDigest == nil ? "An image template is required." : "Image template saved and verified.").font(.caption).foregroundStyle(.secondary)
                }
            }
        }.task {
            do { languages = try await Task.detached { try VisualRewardDetector.supportedLanguages() }.value }
            catch { onIssue(error.localizedDescription) }
        }
    }
    private func saveTemplate() {
        guard let frame, let region = signal.region else { return }
        savingTemplate = true; onIssue(nil)
        let id = signal.id, root = root
        Task {
            defer { savingTemplate = false }
            do {
                let digest = try await Task.detached {
                    let image = try VisualRewardDetector.image(.init(metadata: frame.frame, pixels: frame.pixels))
                    let crop = try VisualRewardDetector.crop(image, region: region, content: frame.frame.surface.contentBounds)
                    guard crop.width <= 1024, crop.height <= 1024 else { throw AstraError("reward.templateSize", "Choose a template region no larger than 1024 × 1024 pixels.") }
                    let data = NSMutableData()
                    guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { throw AstraError("reward.templateSave", "The image template could not be encoded.") }
                    CGImageDestinationAddImage(destination, crop, nil)
                    guard CGImageDestinationFinalize(destination) else { throw AstraError("reward.templateSave", "The image template could not be saved.") }
                    return try RewardAssets.save(data as Data, root: root)
                }.value
                onTemplate(id, digest)
            } catch { onIssue(error.localizedDescription) }
        }
    }
}

struct RewardNumberField: View {
    let title: String
    @Binding var value: Double
    var body: some View {
        LabeledContent(title) {
            TextField(title, value: $value, format: .number).labelsHidden().frame(minWidth: 80, idealWidth: 150, maxWidth: 220)
        }
    }
}

struct RewardPredicateEditor: View {
    @Binding var predicate: RewardPredicate
    let signals: [RewardSignal]
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker("Match", selection: $predicate.logic) { Text("All conditions").tag(RewardLogic.all); Text("Any condition").tag(RewardLogic.any) }
            ForEach($predicate.conditions) { $condition in
                HStack {
                    Picker("Signal", selection: $condition.signalID) { ForEach(signals) { Text($0.name).tag($0.id) } }
                        .onChange(of: condition.signalID) { _, id in
                            if let signal = signals.first(where: { $0.id == id }) { var next = Self.condition(for: signal); next.id = condition.id; condition = next }
                        }
                    Picker("Comparison", selection: $condition.comparison) {
                        ForEach(comparisons(for: condition.signalID), id: \.self) { Text($0.title).tag($0) }
                    }.onChange(of: condition.comparison) { _, comparison in
                        condition.number = [.atLeast, .atMost].contains(comparison) ? (condition.number ?? 0) : nil
                        condition.text = [.equalText, .containsText].contains(comparison) ? (condition.text ?? "") : nil
                    }
                    if let value = Binding($condition.number) { TextField("Threshold", value: value, format: .number).frame(width: 100) }
                    if let text = Binding($condition.text) { TextField("Text", text: text) }
                    Button { predicate.conditions.removeAll { $0.id == condition.id } } label: { Image(systemName: "minus.circle") }
                        .accessibilityLabel("Remove condition")
                }
            }
            Button("Add Condition", systemImage: "plus") { if let first = signals.first { predicate.conditions.append(Self.condition(for: first)) } }
                .disabled(signals.isEmpty || predicate.conditions.count >= 32)
            if signals.isEmpty { Text("Add a signal before defining a condition.").foregroundStyle(.secondary) }
        }
    }
    private func comparisons(for id: UUID) -> [RewardComparison] {
        guard let signal = signals.first(where: { $0.id == id }) else { return [] }
        switch signal.kind {
        case .ocrText: return [.equalText, .containsText]
        case .manual: return RewardComparison.allCases
        default: return [.atLeast, .atMost]
        }
    }
    static func condition(for signal: RewardSignal) -> RewardCondition {
        signal.kind == .ocrText ? .init(signalID: signal.id, comparison: .containsText, text: "") :
        signal.kind == .manual ? .init(signalID: signal.id, comparison: .isTrue) : .init(signalID: signal.id, comparison: .atLeast, number: 0)
    }
}

extension RewardSignalKind {
    var title: String { switch self { case .manual: "Manual value"; case .ocrText: "Visible text"; case .ocrNumber: "Visible score"; case .imageMatch: "Image similarity"; case .elapsedSeconds: "Elapsed time" } }
}
extension RewardRuleKind {
    var title: String { switch self { case .risingEdge: "New event"; case .ratePerSecond: "Time-based reward"; case .scoreDelta: "Score change"; case .manualMarker: "Manual feedback" } }
}
extension RewardComparison {
    var title: String { switch self { case .atLeast: "is at least"; case .atMost: "is at most"; case .equalText: "equals"; case .containsText: "contains"; case .isTrue: "is true"; case .isFalse: "is false" } }
}
extension SignalValue {
    var displayText: String { switch self { case .number(let value): value.formatted(); case .text(let value): value; case .flag(let value): value ? "True" : "False"; case .unknown(let reason): "Unknown · " + reason } }
}
