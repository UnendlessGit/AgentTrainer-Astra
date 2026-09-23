import SwiftUI
import AstraCore

/// A parent-owned sheet/detail. The parent holds the control/actor boundary
/// until the one-shot outcome returns; this view creates no runtime or shortcut.
struct FeedbackReviewView: View {
    let model: FeedbackReviewModel
    let onFinish: @MainActor (FeedbackReviewOutcome) -> Void
    @Environment(\.scenePhase) private var scenePhase
    @State private var previewVisible = false

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                header.padding(20)
                Divider()
                if geometry.size.width >= 780 {
                    HStack(spacing: 0) {
                        intervals.frame(width: 224)
                        Divider()
                        detail(showHeading: true)
                    }
                } else {
                    VStack(spacing: 0) { compactNavigation.padding(12); Divider(); detail(showHeading: false) }
                }
                Divider()
                footer.padding(16)
            }
        }
        .frame(minWidth: 420, minHeight: 500)
        .task { await model.open() }
        .interactiveDismissDisabled(model.phase != .finished)
        .onChange(of: scenePhase) { _, phase in model.setPresentationActive(phase == .active && previewVisible) }
        .onDisappear { model.setPresentationActive(false) }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Review feedback").font(.title2.weight(.semibold))
                Spacer()
                Text("\(model.reviewedPairs) / \(model.totalPairs) reviewed").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(model.source.program.name).foregroundStyle(.secondary).lineLimit(2)
            ProgressView(value: Double(model.reviewedPairs), total: Double(max(1, model.totalPairs)))
                .accessibilityLabel("Feedback review progress")
                .accessibilityValue("\(model.reviewedPairs) of \(model.totalPairs) judgments reviewed")
        }
    }

    private var intervals: some View {
        List(selection: Binding<UUID?>(get: { model.selected.target.packetID }, set: { id in
            if let index = model.intervals.firstIndex(where: { $0.target.packetID == id }) { model.select(index) }
        })) {
            ForEach(model.intervals.indices, id: \.self) { index in
                let interval = model.intervals[index]
                let selected = index == model.selectedIndex
                Label {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Interval \(index + 1)").fontWeight(.medium)
                            .foregroundStyle(selected ? Color(nsColor: .alternateSelectedControlTextColor) : Color.primary)
                        Text(model.relativeTime(interval)).font(.caption)
                            .foregroundStyle(selected ? Color(nsColor: .alternateSelectedControlTextColor) : Color.secondary)
                    }
                } icon: {
                    Image(systemName: model.intervalReviewed(interval) ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(selected ? Color(nsColor: .alternateSelectedControlTextColor) : (model.intervalReviewed(interval) ? Color.accentColor : Color.secondary))
                }
                .tag(interval.target.packetID)
                .accessibilityLabel("Interval \(index + 1), \(model.relativeTime(interval)), \(model.intervalReviewed(interval) ? "reviewed" : "not fully reviewed")")
            }
        }.listStyle(.sidebar).disabled(!model.canEdit)
    }

    private var compactNavigation: some View {
        HStack {
            Button { model.select(model.selectedIndex - 1) } label: { Image(systemName: "chevron.left") }
                .accessibilityLabel("Previous interval").disabled(!model.canEdit || model.selectedIndex == 0)
            VStack(spacing: 3) {
                Text("Interval \(model.selectedIndex + 1) of \(model.intervals.count)").font(.headline)
                Text(model.relativeTime(model.selected)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity)
            Button { model.select(model.selectedIndex + 1) } label: { Image(systemName: "chevron.right") }
                .accessibilityLabel("Next interval").disabled(!model.canEdit || model.selectedIndex + 1 == model.intervals.count)
        }
    }

    private func detail(showHeading: Bool) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let issue = model.issue {
                    Label(issue, systemImage: "exclamationmark.triangle").foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
                }
                if model.phase == .checking { ProgressView("Checking the released-control boundary…") }
                if showHeading { HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Interval \(model.selectedIndex + 1)").font(.headline)
                        Text(model.relativeTime(model.selected)).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(outcomeLabel).font(.callout).foregroundStyle(.secondary)
                } } else { Text(outcomeLabel).font(.callout).foregroundStyle(.secondary) }
                preview
                playbackControls
                VStack(alignment: .leading, spacing: 12) {
                    Text("Your judgments").font(.headline)
                    Text("Set counts, then explicitly review the intervals you watched. Unreviewed rewards stay unknown.")
                        .font(.callout).foregroundStyle(.secondary)
                    ForEach(model.rules) { rule in ruleEditor(rule) }
                    Button(model.selectedReviewed ? "Interval reviewed" : "Mark interval reviewed") {
                        Task { await model.markReviewed() }
                    }.disabled(!model.canConfirmSelected || model.selectedReviewed)
                    if !model.canConfirmSelected && !model.selectedReviewed {
                        Text("View both Before and After on every surface to review this interval.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                rangeReview
                DisclosureGroup("Interval details") {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Episode \(model.selected.target.episodeID.uuidString)")
                        Text("Observation \(model.selected.target.observationID.uuidString)")
                        Text("Packet \(model.selected.target.packetID.uuidString)")
                        Text("Original cutoff: \(model.selected.target.startNanos)–\(model.selected.target.endNanos) ns")
                    }.font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }.padding(20)
        }
    }

    private var outcomeLabel: String {
        switch model.selected.outcome {
        case "terminated": "Episode finished"
        case "truncated": "Episode stopped at this boundary"
        default: "Episode continues"
        }
    }

    private var preview: some View {
        VStack(spacing: 10) {
            Picker("Original observation", selection: Binding(get: { model.point }, set: model.selectPoint)) {
                Text("Before").tag(FeedbackObservationPoint.before)
                Text("After").tag(FeedbackObservationPoint.after)
            }.pickerStyle(.segmented).disabled(!model.canEdit)
            if model.previewSurfaceNames.count > 1 {
                Picker("Surface", selection: Binding(get: { model.selectedSurface }, set: model.selectSurface)) {
                    ForEach(Array(model.previewSurfaceNames.enumerated()), id: \.offset) { index, name in Text(name).tag(index) }
                }.disabled(!model.canEdit)
            }
            ZStack {
                RoundedRectangle(cornerRadius: 8).fill(.quaternary)
                if let image = model.image {
                    Image(decorative: image, scale: 1).resizable().interpolation(.none).scaledToFit().padding(1)
                        .accessibilityLabel("Original \(model.point.rawValue) observation for interval \(model.selectedIndex + 1)")
                } else if model.loadingFrames {
                    ProgressView("Loading original observation…")
                } else if let issue = model.frameIssue {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.badge.exclamationmark").font(.title2)
                        Text("Original observation unavailable").font(.headline)
                        Text(issue).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        Button("Retry") { model.retryPreview() }.disabled(!model.canEdit)
                    }.padding(20)
                } else {
                    Text(model.phase == .blocked ? "Review paused" : "Original observation")
                        .foregroundStyle(.secondary)
                }
            }.frame(height: 240).clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay {
                    FeedbackPreviewVisibility(identity: model.previewIdentity) { visible, identity in
                        previewVisible = visible
                        model.setPresentationActive(visible && scenePhase == .active)
                        if let identity, visible { model.didDisplay(identity) }
                    }.allowsHitTesting(false)
                }
        }
    }

    private var playbackControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Button {
                    if model.isPlaying { model.pausePlayback() } else { model.play() }
                } label: {
                    Label(model.isPlaying ? "Pause" : "Play episode", systemImage: model.isPlaying ? "pause.fill" : "play.fill")
                }.disabled(!model.isPlaying && !model.canPlay)
                Picker("Speed", selection: Binding(get: { model.playbackSpeed }, set: model.setPlaybackSpeed)) {
                    ForEach([0.25, 0.5, 1.0, 2.0, 4.0], id: \.self) { speed in Text("\(speed.formatted())×").tag(speed) }
                }.frame(maxWidth: 150).disabled(!model.canEdit)
                Spacer(minLength: 0)
            }
            Text("Original timing · pauses when this preview is inactive. Playback does not review rewards automatically.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var rangeReview: some View {
        DisclosureGroup("Review several intervals") {
            VStack(alignment: .leading, spacing: 12) {
                Text("Watched in this episode: \(model.watchedRanges)")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 16) {
                    Stepper("From: \(model.rangeStart + 1)", value: Binding(get: { model.rangeStart }, set: model.setRangeStart),
                            in: 0...(model.intervals.count - 1))
                    Stepper("Through: \(model.rangeEnd + 1)", value: Binding(get: { model.rangeEnd }, set: model.setRangeEnd),
                            in: 0...(model.intervals.count - 1))
                }.monospacedDigit().disabled(!model.canEdit)
                Button("Mark range reviewed") { Task { await model.markRangeReviewed() } }
                    .disabled(!model.canReviewRange)
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) { watchedReviewButton; episodeReviewButton }
                    VStack(alignment: .leading, spacing: 8) { watchedReviewButton; episodeReviewButton }
                }
                Text("Reviewing retains your counts and confirms zero only for the covered judgments. Every original observation must be viewed first.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(.top, 8)
        }
    }
    private var watchedReviewButton: some View {
        Button("Mark watched reviewed (\(model.watchedEpisodeIndices.count))") { Task { await model.markWatchedReviewed() } }
            .disabled(!model.canEdit || model.watchedEpisodeIndices.isEmpty)
    }
    private var episodeReviewButton: some View {
        Button("Mark episode reviewed") { Task { await model.markEpisodeReviewed() } }
            .disabled(!model.canReviewEpisode)
    }

    private func ruleEditor(_ rule: RewardRule) -> some View {
        let packet = model.selected.target.packetID, cell = model.cell(packetID: model.selected.target.packetID, ruleID: rule.id)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Label(rule.name, systemImage: rule.amount < 0 ? "hand.thumbsdown" : "hand.thumbsup")
                    .fontWeight(.medium).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Text("\(rule.amount > 0 ? "+" : "")\(rule.amount.formatted()) each")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            HStack {
                Stepper(value: Binding(get: { model.cell(packetID: packet, ruleID: rule.id).count }, set: { count in
                    Task { await model.setCount(packetID: packet, ruleID: rule.id, count: count) }
                }), in: 0...FeedbackLimits.maximumCount) {
                    Text("Count: \(cell.count)").monospacedDigit()
                }.accessibilityLabel("\(rule.name) count").accessibilityValue(String(cell.count)).disabled(!model.canEdit)
                Spacer(minLength: 20)
                Toggle("Reviewed", isOn: Binding(get: { model.cell(packetID: packet, ruleID: rule.id).reviewed }, set: { reviewed in
                    Task { await model.markReviewed(ruleID: rule.id, reviewed: reviewed) }
                })).accessibilityLabel("\(rule.name) reviewed")
                    .disabled(!model.canEdit || (!cell.reviewed && !model.canConfirmSelected))
            }
        }.padding(12).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if model.phase == .saving || model.phase == .closing { ProgressView().controlSize(.small) }
            VStack(alignment: .leading, spacing: 3) {
                Text(model.complete ? "All judgments reviewed" : "Unfinished judgments remain a draft")
                    .font(.callout)
                Text("Saving creates a new feedback revision.").font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button("Cancel") { Task { await model.cancel(); deliverOutcome() } }.disabled(!model.canCancel)
            Button(model.complete ? "Save feedback" : "Save reviewed feedback") {
                Task { await model.save(); deliverOutcome() }
            }.buttonStyle(.borderedProminent).disabled(!model.canSave)
        }
    }
    private func deliverOutcome() { if let outcome = model.takeOutcome() { onFinish(outcome) } }
}

/// This bridge reports an actual, sufficiently visible preview in the active
/// review window. It observes only our view/window; it does not monitor input.
private struct FeedbackPreviewVisibility: NSViewRepresentable {
    let identity: FeedbackReviewModel.PreviewIdentity?
    let report: @MainActor (Bool, FeedbackReviewModel.PreviewIdentity?) -> Void
    func makeNSView(context: Context) -> FeedbackPreviewVisibilityView { FeedbackPreviewVisibilityView() }
    func updateNSView(_ view: FeedbackPreviewVisibilityView, context: Context) {
        view.identity = identity; view.report = report; view.needsDisplay = true
    }
    static func dismantleNSView(_ view: FeedbackPreviewVisibilityView, coordinator: ()) { view.stop() }
}

@MainActor private final class FeedbackPreviewVisibilityView: NSView {
    var identity: FeedbackReviewModel.PreviewIdentity?
    var report: (@MainActor (Bool, FeedbackReviewModel.PreviewIdentity?) -> Void)?
    private var notifications: [NSObjectProtocol] = []
    private var pendingReport: Task<Void, Never>?
    override var isOpaque: Bool { false }
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        notifications.forEach(NotificationCenter.default.removeObserver); notifications.removeAll()
        guard window != nil else { publish(false); return }
        for name in [NSWindow.didBecomeKeyNotification, NSWindow.didResignKeyNotification,
                     NSWindow.didChangeOcclusionStateNotification, NSWindow.didMiniaturizeNotification,
                     NSApplication.didResignActiveNotification, NSApplication.didBecomeActiveNotification,
                     NSView.boundsDidChangeNotification] {
            notifications.append(NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.visibilityChanged() }
            })
        }
        needsDisplay = true
    }
    override func layout() { super.layout(); visibilityChanged() }
    override func draw(_ dirtyRect: NSRect) { publish(isReviewVisible) }
    private var isReviewVisible: Bool {
        guard let window, window.isKeyWindow, window.isVisible, !window.isMiniaturized,
              window.occlusionState.contains(.visible), NSApp.isActive, !isHiddenOrHasHiddenAncestor,
              bounds.width > 0, bounds.height > 0 else { return false }
        let visible = visibleRect.intersection(bounds)
        return visible.width * visible.height >= bounds.width * bounds.height * 0.95
    }
    private func visibilityChanged() {
        if isReviewVisible { needsDisplay = true } else { publish(false) }
    }
    private func publish(_ visible: Bool) {
        pendingReport?.cancel()
        let identity = identity
        pendingReport = Task { [weak self] in
            await Task.yield()
            guard let self, !Task.isCancelled else { return }
            self.report?(visible && self.isReviewVisible, identity)
        }
    }
    func stop() {
        pendingReport?.cancel(); pendingReport = nil
        notifications.forEach(NotificationCenter.default.removeObserver); notifications.removeAll()
        report?(false, nil); report = nil
    }
}
