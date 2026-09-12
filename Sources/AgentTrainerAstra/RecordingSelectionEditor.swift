import SwiftUI
import AstraCore

struct RecordingRangeDraft: Identifiable, Equatable {
    var id: UUID
    var start: Double
    var end: Double
    private var original: RecordingTimeRange?
    private var originalOrigin: UInt64?
    init(_ range: RecordingTimeRange, origin: UInt64) {
        id = range.id
        start = range.startNanos >= origin ? Double(range.startNanos - origin) / 1e9 : -Double(origin - range.startNanos) / 1e9
        end = range.endNanos >= origin ? Double(range.endNanos - origin) / 1e9 : -Double(origin - range.endNanos) / 1e9
        original = range; originalOrigin = origin
    }
    init(start: Double, end: Double) { id = UUID(); self.start = start; self.end = end; original = nil; originalOrigin = nil }
    func range(origin: UInt64) throws -> RecordingTimeRange {
        func timestamp(_ seconds: Double) throws -> UInt64 {
            guard seconds.isFinite, seconds >= 0, let offset = UInt64(exactly: (seconds * 1e9).rounded()),
                  offset <= UInt64(Int64.max), origin <= UInt64(Int64.max) - offset else {
                throw AstraError("selection.time", "Enter finite, non-negative times within the recording.")
            }
            return origin + offset
        }
        if let original, originalOrigin == origin {
            let initial = RecordingRangeDraft(original, origin: origin)
            return try .init(id: id, startNanos: start == initial.start ? original.startNanos : timestamp(start),
                             endNanos: end == initial.end ? original.endNanos : timestamp(end))
        }
        return try .init(id: id, startNanos: timestamp(start), endNanos: timestamp(end))
    }
}

struct RecordingSelectionEditor: View {
    let recording: RecordingManifest
    let agentName: String
    let playheadSeconds: Double
    let onSave: (RecordingTrainingSelection) async -> String?
    var onDirty: (Bool) -> Void = { _ in }
    var onSaving: (Bool) -> Void = { _ in }
    @State private var whole: Bool
    @State private var draft: [RecordingRangeDraft]
    @State private var saved: RecordingTrainingSelection?
    @State private var saving = false
    @State private var saveError: String?

    init(recording: RecordingManifest, agentName: String, selection: RecordingTrainingSelection?, playheadSeconds: Double,
         onSave: @escaping (RecordingTrainingSelection) async -> String?, onDirty: @escaping (Bool) -> Void = { _ in },
         onSaving: @escaping (Bool) -> Void = { _ in }) {
        self.recording = recording; self.agentName = agentName; self.playheadSeconds = playheadSeconds
        self.onSave = onSave; self.onDirty = onDirty
        self.onSaving = onSaving
        _saved = State(initialValue: selection); _whole = State(initialValue: selection?.ranges == nil)
        let ranges = selection?.ranges ?? ((try? RecordingTrainingSelection.whole.resolved(for: recording)) ?? [])
        _draft = State(initialValue: ranges.map { .init($0, origin: recording.firstObservedNanos ?? 0) })
    }

    private var chosen: Result<RecordingTrainingSelection, any Error> {
        Result {
            let selection = whole ? .whole : RecordingTrainingSelection(ranges: try draft.map { try $0.range(origin: recording.firstObservedNanos ?? 0) }
                .sorted { ($0.startNanos, $0.endNanos) < ($1.startNanos, $1.endNanos) })
            _ = try selection.resolved(for: recording)
            return selection
        }
    }
    private var dirty: Bool { (try? chosen.get()) != saved }
    private var nextRange: RecordingRangeDraft? {
        guard draft.count < 256, let all = try? RecordingTrainingSelection.whole.resolved(for: recording), let end = all.first?.durationSeconds,
              draft.allSatisfy({ $0.start.isFinite && $0.end.isFinite && $0.start >= 0 && $0.end > $0.start && $0.end <= end }) else { return nil }
        var gaps: [(Double, Double)] = [], cursor = 0.0
        for range in draft.sorted(by: { $0.start < $1.start }) {
            guard range.start >= cursor else { return nil }
            if range.start > cursor { gaps.append((cursor, range.start)) }
            cursor = range.end
        }
        if cursor < end { gaps.append((cursor, end)) }
        guard let gap = gaps.first(where: { $0.0 <= playheadSeconds && playheadSeconds < $0.1 }) ?? gaps.first else { return nil }
        let start = gap.0 <= playheadSeconds && playheadSeconds < gap.1 ? playheadSeconds : gap.0
        return .init(start: start, end: min(start + 1, gap.1))
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Text("Training intervals for \(agentName)").font(.headline)
                Toggle("Use the whole usable recording", isOn: $whole).toggleStyle(.checkbox)
                if !whole {
                    Text("Enter seconds or set a boundary at the preview time. Separate intervals reset the model’s memory during training.")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(Array(draft.enumerated()), id: \.element.id) { index, row in
                        HStack(spacing: 8) {
                            Text("\(index + 1)").foregroundStyle(.secondary).frame(width: 20)
                            timeField("Start", id: row.id, start: true)
                            Text("to").foregroundStyle(.secondary)
                            timeField("End", id: row.id, start: false)
                            Text("s").foregroundStyle(.secondary)
                            Button("Set Start") { update(row.id, start: true, value: playheadSeconds) }
                                .help("Use the current preview time as this interval's start")
                            Button("Set End") { update(row.id, start: false, value: playheadSeconds) }
                                .help("Use the current preview time as this interval's exclusive end")
                            Button { draft.removeAll { $0.id == row.id } } label: { Image(systemName: "trash") }
                                .accessibilityLabel("Remove interval \(index + 1)").help("Remove this training interval")
                        }
                    }
                    Button("Add Interval", systemImage: "plus") {
                        if let nextRange { draft.append(nextRange); draft.sort { $0.start < $1.start } }
                    }.disabled(nextRange == nil)
                        .help(nextRange == nil ? "Shorten an existing interval to leave time for another one." : "Add an interval in unselected time.")
                }
                if case .failure(let error) = chosen { AttentionLabel(message: error.localizedDescription).font(.callout) }
                if let saveError { AttentionLabel(message: saveError).font(.callout) }
                HStack {
                    Text(dirty ? "Unsaved selection · Original recording stays unchanged" : "Saved for this agent · Original recording stays unchanged")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(saving ? "Saving…" : "Save Selection") {
                        guard let selection = try? chosen.get() else { return }
                        saving = true; onSaving(true); saveError = nil
                        Task {
                            let failure = await onSave(selection)
                            saving = false; onSaving(false); saveError = failure
                            if failure == nil {
                                saved = selection
                                if let ranges = selection.ranges { draft = ranges.map { .init($0, origin: recording.firstObservedNanos ?? 0) } }
                            }
                        }
                    }.buttonStyle(.borderedProminent).disabled(!dirty || saving || (try? chosen.get()) == nil)
                }
            }.padding(8).disabled(saving)
        }
        .onAppear { onDirty(dirty) }
        .onChange(of: dirty) { _, value in onDirty(value) }
    }

    private func update(_ id: UUID, start: Bool, value: Double) {
        guard let index = draft.firstIndex(where: { $0.id == id }) else { return }
        if start { draft[index].start = value } else { draft[index].end = value }
    }
    private func timeField(_ title: String, id: UUID, start: Bool) -> some View {
        TextField(title, value: Binding(get: {
            guard let row = draft.first(where: { $0.id == id }) else { return 0 }
            return start ? row.start : row.end
        }, set: { update(id, start: start, value: $0) }), format: .number.precision(.fractionLength(0...6)))
            .textFieldStyle(.roundedBorder).frame(minWidth: 75, idealWidth: 95, maxWidth: 150)
            .accessibilityLabel("Interval \((draft.firstIndex(where: { $0.id == id }) ?? 0) + 1) \(title.lowercased()) in seconds")
    }
}
