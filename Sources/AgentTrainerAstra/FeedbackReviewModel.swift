import AppKit
import Observation
import AstraCore

@MainActor @Observable final class FeedbackReviewModel {
    enum Phase { case inactive, checking, reviewing, saving, closing, blocked, finished }
    let source: VerifiedFeedbackSource
    let parent: LoadedFeedbackRevision?
    let artifactDirectory: URL
    let rules: [RewardRule]
    private let dependencies: FeedbackReviewDependencies
    private let episodeOrigins: [UUID: UInt64]
    private(set) var phase: Phase = .inactive
    private(set) var selectedIndex = 0
    private(set) var point: FeedbackObservationPoint = .before
    private(set) var selectedSurface = 0
    private(set) var image: CGImage?
    private(set) var observation: FeedbackReviewObservation?
    private(set) var frameIssue: String?
    private(set) var issue: String?
    private(set) var loadingFrames = false
    private(set) var editing = false
    private(set) var draftReference: FeedbackReviewDraftReference?
    private(set) var unadmittedRevision: LoadedFeedbackRevision?
    private var cells: [FeedbackReviewPair: FeedbackReviewDraft.Cell] = [:]
    private var seenObservations: Set<UUID> = []
    private var displayedSurfaces: [UUID: Set<String>] = [:]
    private(set) var presentationActive = false
    private(set) var isPlaying = false
    private(set) var playbackSpeed = 1.0
    private(set) var rangeStart = 0
    private(set) var rangeEnd = 0
    private var playbackWork: Task<Void, Never>?
    private var playbackGeneration = UUID()
    private var frameGeneration = UUID()
    private var frameWork: Task<Void, Never>?
    private var saveWork: Task<Void, Never>?
    private var cancelWork: Task<Void, Never>?
    private var cancelRequested = false
    private var outcome: FeedbackReviewOutcome?

    init(source: VerifiedFeedbackSource, parent: LoadedFeedbackRevision? = nil, artifactDirectory: URL,
         dependencies: FeedbackReviewDependencies, draft: FeedbackReviewDraft? = nil) throws {
        guard artifactDirectory.isFileURL, (4...FrameArchive.maximumFrameBytes).contains(dependencies.maximumPreviewBytes),
              (parent?.document.sourceSHA256 ?? source.sha256) == source.sha256 else {
            throw AstraError("feedback.reviewConfiguration", "Feedback review requires matching immutable source and bounded local storage.")
        }
        self.source = source; self.parent = parent; self.artifactDirectory = artifactDirectory; self.dependencies = dependencies
        var origins: [UUID: UInt64] = [:]
        for interval in source.metadata.intervals where origins[interval.target.episodeID] == nil { origins[interval.target.episodeID] = interval.target.startNanos }
        episodeOrigins = origins
        rules = source.program.rules.filter { $0.kind == .manualMarker }.sorted { $0.id.uuidString < $1.id.uuidString }
        if let parent {
            for row in parent.document.resolved {
                for component in row.components where component.reviewed {
                    let cell = FeedbackReviewDraft.Cell(packetID: row.packetID, ruleID: component.ruleID, count: component.count ?? 0, reviewed: true)
                    cells[cell.pair] = cell
                }
            }
        }
        if let draft {
            _ = try draft.validated(source: source, parent: parent)
            cells = Dictionary(uniqueKeysWithValues: draft.cells.map { ($0.pair, $0) })
            selectedIndex = source.metadata.intervals.firstIndex(where: { $0.target.packetID == draft.selectedPacketID })!
            point = draft.point
            rangeStart = selectedIndex; rangeEnd = selectedIndex
        }
    }

    var intervals: [FeedbackEligibleInterval] { source.metadata.intervals }
    var selected: FeedbackEligibleInterval { intervals[selectedIndex] }
    var reviewedPairs: Int { cells.values.filter(\.reviewed).count }
    var totalPairs: Int { intervals.count * rules.count }
    var complete: Bool { reviewedPairs == totalPairs }
    var canEdit: Bool { phase == .reviewing && !editing }
    var canConfirmSelected: Bool {
        canEdit && !loadingFrames && frameIssue == nil && intervalReviewable(selected)
    }
    var canSave: Bool { canEdit && !loadingFrames && frameIssue == nil && reviewedPairs > 0 }
    var canPlay: Bool { canEdit && presentationActive && frameIssue == nil }
    var canCancel: Bool { phase != .finished && phase != .closing && !cancelRequested }
    var selectedReviewed: Bool { rules.allSatisfy { cell(packetID: selected.target.packetID, ruleID: $0.id).reviewed } }
    var previewSurfaceNames: [String] { observation?.frames.map(\.metadata.surface.id) ?? [] }

    func cell(packetID: UUID, ruleID: UUID) -> FeedbackReviewDraft.Cell {
        cells[.init(packetID: packetID, ruleID: ruleID)] ?? .init(packetID: packetID, ruleID: ruleID, count: 0, reviewed: false)
    }
    func intervalReviewed(_ interval: FeedbackEligibleInterval) -> Bool {
        rules.allSatisfy { cell(packetID: interval.target.packetID, ruleID: $0.id).reviewed }
    }
    func relativeTime(_ interval: FeedbackEligibleInterval) -> String {
        let origin = episodeOrigins[interval.target.episodeID]!
        let start = Double(interval.target.startNanos - origin) / 1e9, end = Double(interval.target.endNanos - origin) / 1e9
        return String(format: "%.3f–%.3f s", start, end)
    }

    struct PreviewIdentity: Hashable {
        let generation: UUID
        let observationID: UUID
        let surfaceID: String
    }
    var previewIdentity: PreviewIdentity? {
        guard image != nil, let observation, observation.frames.indices.contains(selectedSurface) else { return nil }
        return .init(generation: frameGeneration, observationID: observation.observationID,
                     surfaceID: observation.frames[selectedSurface].metadata.surface.id)
    }
    /// Only the visible preview acknowledges presentation. Loading pixels alone
    /// never qualifies an interval for explicit zero-reward review.
    func didDisplay(_ identity: PreviewIdentity) {
        guard presentationActive, phase == .reviewing, identity == previewIdentity, lastDisplayedPreview != identity, let observation else { return }
        lastDisplayedPreview = identity
        displayedSurfaces[identity.observationID, default: []].insert(identity.surfaceID)
        if Set(observation.frames.map(\.metadata.surface.id)).isSubset(of: displayedSurfaces[identity.observationID] ?? []) {
            seenObservations.insert(identity.observationID)
        }
    }
    func setPresentationActive(_ active: Bool) {
        guard presentationActive != active else { return }
        presentationActive = active
        if !active { pausePlayback() }
    }
    func intervalReviewable(_ interval: FeedbackEligibleInterval) -> Bool {
        seenObservations.contains(interval.target.observationID) && seenObservations.contains(interval.endpointObservationID)
    }
    var episodeIndices: [Int] { intervals.indices.filter { intervals[$0].target.episodeID == selected.target.episodeID } }
    var watchedEpisodeIndices: [Int] { episodeIndices.filter { intervalReviewable(intervals[$0]) } }
    var canReviewRange: Bool { canEdit && (rangeStart...rangeEnd).allSatisfy { intervalReviewable(intervals[$0]) } }
    var canReviewEpisode: Bool { canEdit && episodeIndices.allSatisfy { intervalReviewable(intervals[$0]) } }
    var watchedRanges: String {
        var ranges: [String] = [], first: Int?, last: Int?
        for index in watchedEpisodeIndices {
            if let previous = last, index != previous + 1, let start = first {
                ranges.append(start == previous ? "\(start + 1)" : "\(start + 1)–\(previous + 1)"); first = nil
            }
            if first == nil { first = index }
            last = index
        }
        if let first, let last { ranges.append(first == last ? "\(first + 1)" : "\(first + 1)–\(last + 1)") }
        return ranges.isEmpty ? "None yet" : ranges.joined(separator: ", ")
    }
    func setRangeStart(_ index: Int) {
        guard canEdit, intervals.indices.contains(index) else { return }
        pausePlayback(); rangeStart = index; rangeEnd = max(index, rangeEnd)
    }
    func setRangeEnd(_ index: Int) {
        guard canEdit, intervals.indices.contains(index) else { return }
        pausePlayback(); rangeEnd = index; rangeStart = min(index, rangeStart)
    }
    func markRangeReviewed() async { await markIndicesReviewed(Array(rangeStart...rangeEnd)) }
    func markWatchedReviewed() async { await markIndicesReviewed(watchedEpisodeIndices) }
    func markEpisodeReviewed() async { await markIndicesReviewed(episodeIndices) }
    private func markIndicesReviewed(_ indices: [Int]) async {
        guard canEdit, !indices.isEmpty, indices.allSatisfy({ intervals.indices.contains($0) && intervalReviewable(intervals[$0]) }) else { return }
        pausePlayback(); editing = true; defer { editing = false }
        do {
            try await dependencies.validateBoundary()
            guard phase == .reviewing else { return }
            for index in indices { for rule in rules {
                var value = cell(packetID: intervals[index].target.packetID, ruleID: rule.id)
                value.reviewed = true; cells[value.pair] = value
            } }
            issue = nil
        } catch { await block(error) }
    }
    func setPlaybackSpeed(_ value: Double) {
        guard [0.25, 0.5, 1, 2, 4].contains(value) else { return }
        pausePlayback(); playbackSpeed = value
    }
    func pausePlayback() {
        playbackGeneration = UUID(); isPlaying = false; playbackWork?.cancel()
    }
    func play() {
        guard canPlay, !isPlaying else { return }
        let prior = playbackWork, token = UUID(), first = selectedIndex, episode = selected.target.episodeID
        playbackGeneration = token; isPlaying = true
        playbackWork = Task { [weak self] in
            await prior?.value
            guard let self else { return }
            defer { if self.playbackGeneration == token { self.isPlaying = false; self.playbackWork = nil } }
            for index in first..<self.intervals.count {
                guard self.playbackCurrent(token), self.intervals[index].target.episodeID == episode else { return }
                guard await self.showPlayback(index: index, point: .before, token: token) else { return }
                let interval = self.intervals[index]
                // Preserve source duration. Slow I/O stretches playback instead of
                // dropping observations or turning an unseen interval into zero.
                let delay = Double(interval.target.endNanos - interval.target.startNanos) / self.playbackSpeed
                do { try await Task.sleep(for: .seconds(delay / 1e9)) }
                catch { return }
                guard await self.showPlayback(index: index, point: .after, token: token) else { return }
            }
        }
    }
    private func playbackCurrent(_ token: UUID) -> Bool {
        playbackGeneration == token && isPlaying && presentationActive && phase == .reviewing && !Task.isCancelled
    }
    private func showPlayback(index: Int, point: FeedbackObservationPoint, token: UUID) async -> Bool {
        guard playbackCurrent(token) else { return false }
        let interval = intervals[index]
        let observationID = point == .before ? interval.target.observationID : interval.endpointObservationID
        let cutoff = point == .before ? interval.target.startNanos : interval.target.endNanos
        selectedIndex = index; self.point = point
        if observation?.observationID != observationID || observation?.cutoffNanos != cutoff {
            invalidatePreview(); startPreview()
        }
        await frameWork?.value
        guard playbackCurrent(token), image != nil, frameIssue == nil else { return false }
        // Wait for this presentation, even if this observation was seen before.
        // No background/offscreen playback can create a watched range.
        let identity = previewIdentity
        while playbackCurrent(token), lastDisplayedPreview != identity {
            do { try await Task.sleep(for: .milliseconds(16)) } catch { return false }
        }
        return playbackCurrent(token)
    }
    private var lastDisplayedPreview: PreviewIdentity?

    func open() async {
        guard phase == .inactive else { return }
        phase = .checking
        do {
            try await dependencies.validateBoundary()
            guard phase == .checking else { return }
            phase = .reviewing; startPreview()
        } catch { await block(error) }
    }

    func select(_ index: Int) {
        guard canEdit, intervals.indices.contains(index), index != selectedIndex else { return }
        pausePlayback(); rangeStart = index; rangeEnd = index
        selectedIndex = index; point = .before; invalidatePreview(); startPreview()
    }
    func selectPoint(_ point: FeedbackObservationPoint) {
        guard canEdit, self.point != point else { return }
        pausePlayback(); self.point = point; invalidatePreview(); startPreview()
    }
    func selectSurface(_ index: Int) {
        guard canEdit, let observation, observation.frames.indices.contains(index) else { return }
        pausePlayback()
        selectedSurface = index; image = Self.image(observation.frames[index])
    }
    func retryPreview() { guard canEdit else { return }; pausePlayback(); invalidatePreview(); startPreview() }

    func setCount(packetID: UUID, ruleID: UUID, count: Int) async {
        guard canEdit else { return }
        pausePlayback()
        editing = true; defer { editing = false }
        guard intervals.contains(where: { $0.target.packetID == packetID }), rules.contains(where: { $0.id == ruleID }),
              (0...FeedbackLimits.maximumCount).contains(count) else { issue = "Choose a count from 0 to \(FeedbackLimits.maximumCount)."; return }
        do {
            try await dependencies.validateBoundary()
            guard phase == .reviewing else { return }
            var value = cell(packetID: packetID, ruleID: ruleID)
            guard value.count != count else { return }
            value.count = count; value.reviewed = false; cells[value.pair] = value; issue = nil
        } catch { await block(error) }
    }

    func markReviewed(ruleID: UUID? = nil, reviewed: Bool = true) async {
        guard reviewed ? canConfirmSelected : canEdit else { return }
        let packet = selected.target.packetID
        let selectedRules = ruleID.map { id in rules.filter { $0.id == id } } ?? rules
        pausePlayback()
        editing = true; defer { editing = false }
        do {
            try await dependencies.validateBoundary()
            guard phase == .reviewing, selected.target.packetID == packet else { return }
            for rule in selectedRules {
                var value = cell(packetID: packet, ruleID: rule.id); value.reviewed = reviewed; cells[value.pair] = value
            }
            issue = nil
        } catch { await block(error) }
    }

    func save() async {
        guard canSave, saveWork == nil else { return }
        pausePlayback(); phase = .saving; cancelRequested = false; issue = nil
        let work = Task { await self.publishReview() }
        saveWork = work; await work.value; saveWork = nil
    }

    /// Every caller joins the same teardown, including a host Stop racing the
    /// sheet's Cancel button. It is safe to release the source only after return.
    func cancel() async {
        if let cancelWork { await cancelWork.value; return }
        guard phase != .finished else { return }
        let work = Task { await self.performCancel() }
        cancelWork = work; await work.value; cancelWork = nil
    }
    private func performCancel() async {
        guard phase != .finished, !cancelRequested else { return }
        cancelRequested = true
        if let saveWork { await saveWork.value }
        if phase == .finished { return }
        phase = .closing
        await joinPlaybackAndPreview()
        do {
            let reference = try await preserveDraft()
            outcome = .cancelled(reference, retainedRevision: unadmittedRevision?.reference); phase = .finished
        } catch {
            cancelRequested = false; phase = .blocked; issue = "Your edits are still in this window. The review draft could not be saved: \(error.localizedDescription)"
        }
    }

    func takeOutcome() -> FeedbackReviewOutcome? { defer { outcome = nil }; return outcome }

    private func publishReview() async {
        await joinPlaybackAndPreview()
        do {
            let draft = makeDraft()
            let reference = try await FeedbackReviewDraftStore.save(draft, source: source, parent: parent, directory: artifactDirectory)
            draftReference = reference
            if cancelRequested { finishCancelled(reference); return }
            try await dependencies.validateBoundary()
            if cancelRequested { finishCancelled(reference); return }
            let authored = dependencies.authoredNow(), source = source, parent = parent, directory = artifactDirectory
            // Save commits the currently reviewed judgments. New record IDs use
            // this actual commit time; no historical action/marker is backdated.
            let loaded = try await Task.detached {
                let byPacket = Dictionary(uniqueKeysWithValues: source.metadata.intervals.map { ($0.target.packetID, $0.target) })
                var annotations: [FeedbackAnnotation] = [], pairs: [FeedbackReviewPair] = []
                for cell in draft.cells where cell.reviewed {
                    pairs.append(cell.pair)
                    if cell.count > 0 {
                        annotations.append(.init(sequence: annotations.count, target: byPacket[cell.packetID]!, ruleID: cell.ruleID,
                                                 count: cell.count, authored: authored))
                    }
                }
                let reviews = pairs.isEmpty ? [] : [FeedbackReviewCompletion(sequence: 0, pairs: pairs, authored: authored)]
                let revision = try FeedbackRewardRevision.create(source: source, authored: authored, annotations: annotations, reviews: reviews, parent: parent)
                return try FeedbackArtifactStore.publish(revision, source: source, directory: directory, parent: parent)
            }.value
            unadmittedRevision = loaded
            if cancelRequested { finishCancelled(reference); return }
            try await dependencies.validateBoundary()
            if cancelRequested { finishCancelled(reference); return }
            let complete = loaded.document.resolved.allSatisfy { $0.manualReward != nil }
            outcome = .saved(loaded, complete: complete, draft: reference); unadmittedRevision = nil; phase = .finished
        } catch {
            phase = .blocked
            issue = unadmittedRevision == nil
                ? "Feedback was not submitted. Your draft is preserved: \(error.localizedDescription)"
                : "The revision was saved, but the review boundary changed. It has not been admitted for learning. Your draft is preserved."
            if draftReference == nil {
                do { _ = try await preserveDraft() }
                catch { issue = "Your edits are still in this window, but the draft could not be saved: \(error.localizedDescription)" }
            }
        }
    }

    private func finishCancelled(_ reference: FeedbackReviewDraftReference) {
        outcome = .cancelled(reference, retainedRevision: unadmittedRevision?.reference); phase = .finished
    }
    private func makeDraft() -> FeedbackReviewDraft {
        .init(schemaVersion: 1, id: UUID(), sourceSHA256: source.sha256, parent: parent?.reference,
              selectedPacketID: selected.target.packetID, point: point,
              cells: cells.values.sorted { ($0.packetID.uuidString, $0.ruleID.uuidString) < ($1.packetID.uuidString, $1.ruleID.uuidString) })
    }
    private func preserveDraft() async throws -> FeedbackReviewDraftReference {
        let reference = try await FeedbackReviewDraftStore.save(makeDraft(), source: source, parent: parent, directory: artifactDirectory)
        draftReference = reference; return reference
    }
    private func block(_ error: any Error) async {
        guard phase != .finished, phase != .closing else { return }
        pausePlayback(); phase = .blocked; invalidatePreview(); issue = error.localizedDescription
        do { _ = try await preserveDraft() }
        catch { issue = "Review is paused. Your edits remain in this window, but the draft could not be saved: \(error.localizedDescription)" }
    }
    private func invalidatePreview() {
        frameGeneration = UUID(); image = nil; observation = nil; selectedSurface = 0; frameIssue = nil
    }
    private func joinPlaybackAndPreview() async {
        pausePlayback()
        let playback = playbackWork
        await playback?.value
        playbackWork = nil
        await joinPreview()
    }
    private func joinPreview() async {
        frameGeneration = UUID(); let work = frameWork; work?.cancel(); await work?.value
        frameWork = nil; loadingFrames = false; image = nil; observation = nil
    }
    private func startPreview() {
        guard phase == .reviewing, frameWork == nil else { return }
        loadingFrames = true
        frameWork = Task { [weak self] in
            guard let self else { return }
            defer { self.frameWork = nil; self.loadingFrames = false }
            while self.phase == .reviewing, !Task.isCancelled {
                let token = self.frameGeneration, selected = self.selected
                let request = FeedbackObservationRequest(sourceSHA256: self.source.sha256,
                    trajectorySHA256: self.source.metadata.trajectorySHA256, episodeID: selected.target.episodeID,
                    observationID: self.point == .before ? selected.target.observationID : selected.endpointObservationID,
                    cutoffNanos: self.point == .before ? selected.target.startNanos : selected.target.endNanos,
                    maximumBytes: self.dependencies.maximumPreviewBytes, maximumFrames: 16)
                do { try await self.dependencies.validateBoundary() }
                catch { await self.block(error); return }
                guard self.phase == .reviewing, !Task.isCancelled else { return }
                if token != self.frameGeneration { continue }
                do {
                    let loaded = try await self.dependencies.loadObservation(request)
                    guard self.phase == .reviewing, !Task.isCancelled else { return }
                    if token != self.frameGeneration { continue }
                    try await Task.detached { try Self.validate(loaded, request: request) }.value
                    guard self.phase == .reviewing, !Task.isCancelled else { return }
                    if token != self.frameGeneration { continue }
                    do { try await self.dependencies.validateBoundary() }
                    catch { await self.block(error); return }
                    guard self.phase == .reviewing, !Task.isCancelled else { return }
                    if token != self.frameGeneration { continue }
                    self.observation = loaded; self.selectedSurface = 0; self.image = Self.image(loaded.frames[0])
                    guard self.image != nil else { throw AstraError("feedback.preview", "The original pixels could not be displayed.") }
                    self.frameIssue = nil; return
                } catch {
                    guard self.phase == .reviewing, !Task.isCancelled else { return }
                    if token != self.frameGeneration { continue }
                    self.seenObservations.remove(request.observationID)
                    self.displayedSurfaces.removeValue(forKey: request.observationID)
                    self.frameIssue = error.localizedDescription; self.image = nil; self.observation = nil; return
                }
            }
        }
    }

    nonisolated private static func validate(_ value: FeedbackReviewObservation, request: FeedbackObservationRequest) throws {
        guard value.sourceSHA256 == request.sourceSHA256, value.trajectorySHA256 == request.trajectorySHA256,
              value.episodeID == request.episodeID, value.observationID == request.observationID, value.cutoffNanos == request.cutoffNanos,
              (1...request.maximumFrames).contains(value.frames.count), Set(value.frames.map(\.metadata.id)).count == value.frames.count,
              Set(value.frames.map(\.metadata.surface.id)).count == value.frames.count else {
            throw AstraError("feedback.observation", "The preview does not match the original observation and frozen source.")
        }
        var remaining = request.maximumBytes
        for frame in value.frames {
            _ = try frame.metadata.validated()
            guard frame.pixels.count == frame.metadata.byteCount, frame.pixels.count <= remaining,
                  frame.metadata.eventNanos <= frame.metadata.observedNanos, frame.metadata.observedNanos <= request.cutoffNanos,
                  FeedbackArtifactStore.digest(frame.pixels) == frame.pixelSHA256 else {
                throw AstraError("feedback.frame", "The original frame is missing, corrupt, oversized or unavailable at this cutoff.")
            }
            remaining -= frame.pixels.count
        }
    }
    private static func image(_ frame: FeedbackReviewObservation.Frame) -> CGImage? {
        let surface = frame.metadata.surface
        guard let provider = CGDataProvider(data: frame.pixels as CFData), let color = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: surface.pixelWidth, height: surface.pixelHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: surface.pixelWidth * 4, space: color,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).union(.byteOrder32Little),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return image
    }
}
