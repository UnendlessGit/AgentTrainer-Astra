import Foundation
import Testing
import AstraCore
@testable import AgentTrainerAstra

actor FeedbackReviewHarness {
    var checks = 0
    var rejectBoundary = false
    var failAtCheck: Int?
    var holdAtCheck: Int?
    var boundaryWaiting = false
    var boundaryContinuation: CheckedContinuation<Void, Never>?
    var requests: [FeedbackObservationRequest] = []
    var activeLoads = 0, peakLoads = 0
    var suspended: CheckedContinuation<Void, Never>?
    var holdFirst = false
    var invalidFrame: String?

    func validateBoundary() async throws {
        checks += 1
        if holdAtCheck == checks { boundaryWaiting = true; await withCheckedContinuation { boundaryContinuation = $0 } }
        if rejectBoundary || failAtCheck == checks { throw AstraError("fixture.boundary", "The actor pause or joined-control proof is no longer valid.") }
    }
    func failNextBoundary(after successfulChecks: Int = 0) { failAtCheck = checks + successfulChecks + 1 }
    func reject() { rejectBoundary = true }
    func holdNextBoundary(after successfulChecks: Int) { holdAtCheck = checks + successfulChecks + 1 }
    func releaseBoundary() { boundaryContinuation?.resume(); boundaryContinuation = nil; boundaryWaiting = false }
    func blockFirst() { holdFirst = true }
    func invalid(_ mode: String) { invalidFrame = mode }
    func release() { suspended?.resume(); suspended = nil }
    func load(_ request: FeedbackObservationRequest) async throws -> FeedbackReviewObservation {
        requests.append(request); activeLoads += 1; peakLoads = max(peakLoads, activeLoads)
        defer { activeLoads -= 1 }
        if holdFirst, requests.count == 1 { await withCheckedContinuation { suspended = $0 } }
        if invalidFrame == "missing" { throw AstraError("fixture.missing", "The original frame could not be read.") }
        let surface = SurfaceDescriptor(id: "recorded-window", globalBounds: .init(x: 0, y: 0, width: 320, height: 180),
            pixelWidth: 160, pixelHeight: 90)
        var pixels = Data(repeating: 255, count: 160 * 90 * 4)
        pixels.withUnsafeMutableBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for y in 0..<90 { for x in 0..<160 {
                let i = (y * 160 + x) * 4
                let card = (12..<148).contains(x) && (12..<78).contains(y)
                bytes[i] = card ? 214 : 48; bytes[i + 1] = card ? 202 : 42; bytes[i + 2] = card ? 185 : 38
                if (28..<132).contains(x) && (35..<55).contains(y) { bytes[i] = 190; bytes[i + 1] = 117; bytes[i + 2] = 39 }
            } }
        }
        let metadata = FrameMetadata(eventNanos: request.cutoffNanos - 1, observedNanos: request.cutoffNanos,
                                     surface: surface, byteCount: pixels.count, codec: "raw")
        return .init(sourceSHA256: invalidFrame == "source" ? String(repeating: "c", count: 64) : request.sourceSHA256,
            trajectorySHA256: request.trajectorySHA256, episodeID: request.episodeID,
            observationID: invalidFrame == "identity" ? UUID() : request.observationID, cutoffNanos: request.cutoffNanos,
            frames: [.init(metadata: metadata, pixels: pixels, pixelSHA256: invalidFrame == "checksum" ? String(repeating: "0", count: 64) : FeedbackArtifactStore.digest(pixels))])
    }
}

struct FeedbackReviewFixture {
    let directory: URL
    let source: VerifiedFeedbackSource
    let positive: UUID, negative: UUID
    let harness = FeedbackReviewHarness()
    let authorship: FeedbackAuthorship
    init(longNames: Bool = false) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("astra-feedback-review-" + UUID().uuidString)
        positive = UUID(); negative = UUID()
        let program = RewardProgram(name: longNames ? "Desktop interaction · precise and patient choices across unfamiliar layouts" : "Desktop interaction", rules: [
            .init(id: positive, name: longNames ? "Useful progress toward the intended task" : "Helpful action", kind: .manualMarker, amount: 1),
            .init(id: negative, name: longNames ? "Avoidable action that interrupted progress" : "Unhelpful action", kind: .manualMarker, amount: -1)])
        let programBytes = try FeedbackArtifactStore.encoded(program), episode = UUID(), policy = UUID(), clock = UUID(), session = UUID()
        let observations = (0..<4).map { _ in UUID() }
        let intervals: [FeedbackEligibleInterval] = (0..<3).map { (index: Int) -> FeedbackEligibleInterval in
            let target = FeedbackIntervalIdentity(episodeID: episode, observationID: observations[index], packetID: UUID(),
                startNanos: UInt64(index + 1) * 100_000_000, endNanos: UInt64(index + 2) * 100_000_000)
            return FeedbackEligibleInterval(target: target,
                packetSequence: UInt64(index), drawIndex: UInt64(index), endpointObservationID: observations[index + 1],
                execution: "executed", outcome: index == 2 ? "terminated" : "continuing", eligible: true)
        }
        let metadata = try FeedbackTrajectoryMetadata(collectionID: UUID(), runID: UUID(), clockID: clock, sourceSessionID: session,
            policyID: policy, program: program, trajectorySHA256: String(repeating: "a", count: 64), policySignature: String(repeating: "b", count: 64),
            programSHA256: FeedbackArtifactStore.digest(programBytes), status: "awaiting_manual_review", controlClosureKnown: true,
            closedNanos: 500_000_000, closedWallTimeMS: 1000, intervals: intervals)
        let bytes = try FeedbackArtifactStore.encoded(metadata)
        source = try VerifiedFeedbackSource(metadataBytes: bytes, expectedSHA256: FeedbackArtifactStore.digest(bytes), programBytes: programBytes)
        authorship = .init(sessionID: session, clockID: clock, observedNanos: 1_000_000_000, wallTimeMS: 2000)
    }
    @MainActor func model(draft: FeedbackReviewDraft? = nil, maximumBytes: Int = FrameArchive.maximumFrameBytes) throws -> FeedbackReviewModel {
        try .init(source: source, artifactDirectory: directory, dependencies: .init(
            validateBoundary: { try await harness.validateBoundary() }, loadObservation: { try await harness.load($0) },
            authoredNow: { authorship }, maximumPreviewBytes: maximumBytes), draft: draft)
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
}

@MainActor func feedbackEventually(_ predicate: () async -> Bool) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(3))
    while !(await predicate()) {
        guard ContinuousClock.now < end else { throw AstraError("fixture.timeout", "Feedback review did not reach the expected boundary.") }
        try await Task.sleep(for: .milliseconds(2))
    }
}
@MainActor func viewFeedbackInterval(_ model: FeedbackReviewModel, index: Int) async throws {
    model.select(index)
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    model.setPresentationActive(true)
    model.didDisplay(try #require(model.previewIdentity))
    model.selectPoint(.after)
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    model.didDisplay(try #require(model.previewIdentity))
    #expect(model.canConfirmSelected)
}

@Suite @MainActor struct FeedbackReviewTests {
@Test func playbackShowsOriginalTimingAndRequiresExplicitRangeReview() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    await model.setCount(packetID: model.intervals[1].target.packetID, ruleID: fixture.negative, count: 2)
    model.setPresentationActive(true); model.setPlaybackSpeed(4)
    let began = ContinuousClock.now
    model.play()
    try await feedbackEventually {
        if let identity = model.previewIdentity { model.didDisplay(identity) }
        return !model.isPlaying
    }
    #expect(began.duration(to: .now) >= .milliseconds(75))
    #expect(model.watchedEpisodeIndices == [0, 1, 2] && model.reviewedPairs == 0)
    #expect(await fixture.harness.requests.count == 4)
    #expect(await fixture.harness.peakLoads == 1)
    model.setRangeStart(0); model.setRangeEnd(1)
    await model.markRangeReviewed()
    #expect(model.reviewedPairs == 4 && !model.complete)
    #expect(model.cell(packetID: model.intervals[1].target.packetID, ruleID: fixture.negative).count == 2)
    #expect(model.cell(packetID: model.intervals[0].target.packetID, ruleID: fixture.positive).count == 0)
    #expect(!model.intervalReviewed(model.intervals[2]))
    await model.markEpisodeReviewed(); #expect(model.complete)
    await model.cancel()
}

@Test func loadedButUnpresentedOrInactiveFramesNeverQualifyForBulkReview() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    model.didDisplay(try #require(model.previewIdentity)) // inactive window
    model.selectPoint(.after)
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    model.setPresentationActive(true); model.didDisplay(try #require(model.previewIdentity))
    #expect(model.watchedEpisodeIndices.isEmpty && !model.canConfirmSelected)
    await model.markWatchedReviewed(); await model.markEpisodeReviewed()
    #expect(model.reviewedPairs == 0)
    model.play(); model.setPresentationActive(false)
    #expect(!model.isPlaying)
    await model.cancel()
}

@Test func playbackCancellationJoinsOneLoaderForConcurrentCallers() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    await fixture.harness.blockFirst()
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { await fixture.harness.activeLoads == 1 }
    model.setPresentationActive(true); model.play()
    let first = Task { await model.cancel() }
    try await feedbackEventually { model.phase == .closing }
    let second = Task { await model.cancel() }
    #expect(model.takeOutcome() == nil)
    await fixture.harness.release(); await first.value; await second.value
    #expect(!model.isPlaying && !model.loadingFrames)
    #expect(await fixture.harness.activeLoads == 0)
    #expect(await fixture.harness.peakLoads == 1)
    guard case .cancelled? = model.takeOutcome() else { Issue.record("Teardown lost the durable draft"); return }
}

@Test func reviewRequiresBothOriginalObservationsAndExplicitZero() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    #expect(!model.canConfirmSelected && model.reviewedPairs == 0 && !model.canSave)
    await model.markReviewed()
    #expect(model.reviewedPairs == 0)
    try await viewFeedbackInterval(model, index: 0)
    await model.markReviewed()
    #expect(model.reviewedPairs == 2 && !model.complete)
    await model.save()
    guard case .saved(let revision, let complete, let draft)? = model.takeOutcome() else { Issue.record("Missing saved partial revision"); return }
    #expect(!complete && revision.document.resolved[0].manualReward == 0)
    #expect(revision.document.resolved[1].manualReward == nil && revision.document.resolved[2].manualReward == nil)
    #expect(FileManager.default.fileExists(atPath: draft.url.path))
    #expect(model.takeOutcome() == nil)
}

@Test func countChangesInvalidateReviewAndPartialSavePreservesUnreviewedEdits() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open(); try await viewFeedbackInterval(model, index: 0)
    await model.markReviewed()
    let packet = model.selected.target.packetID
    await model.setCount(packetID: packet, ruleID: fixture.positive, count: 3)
    #expect(!model.cell(packetID: packet, ruleID: fixture.positive).reviewed)
    #expect(model.cell(packetID: packet, ruleID: fixture.negative).reviewed)
    await model.save()
    guard case .saved(let revision, let complete, let reference)? = model.takeOutcome() else { Issue.record("Missing partial save"); return }
    #expect(!complete && revision.document.annotations.isEmpty && revision.document.resolved[0].manualReward == nil)
    let draft = try await FeedbackReviewDraftStore.load(reference, source: fixture.source, parent: nil)
    #expect(draft.cells.contains { $0.ruleID == fixture.positive && $0.count == 3 && !$0.reviewed })
}

@Test func cancelPersistsDraftWithoutPublishingOrClaimingReviewCompletion() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    await model.setCount(packetID: model.selected.target.packetID, ruleID: fixture.negative, count: 2)
    await model.cancel()
    guard case .cancelled(let reference, let retained)? = model.takeOutcome() else { Issue.record("Missing durable cancellation"); return }
    #expect(retained == nil)
    let draft = try await FeedbackReviewDraftStore.load(reference, source: fixture.source, parent: nil)
    #expect(draft.cells.count == 1 && draft.cells[0].count == 2 && !draft.cells[0].reviewed)
    #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.directory.path) == ["Drafts"])
    let resumed = try fixture.model(draft: draft)
    #expect(resumed.reviewedPairs == 0 && resumed.cell(packetID: model.selected.target.packetID, ruleID: fixture.negative).count == 2)
}

@Test func completeReviewPublishesRealCountsWithCommitTimeAndExactOriginalTargets() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open()
    for index in 0..<3 {
        try await viewFeedbackInterval(model, index: index)
        await model.setCount(packetID: model.selected.target.packetID, ruleID: fixture.positive, count: index + 1)
        await model.markReviewed()
    }
    #expect(model.complete)
    await model.save()
    guard case .saved(let revision, let complete, _)? = model.takeOutcome() else { Issue.record("Missing completed revision"); return }
    #expect(complete && revision.document.annotations.count == 3)
    #expect(revision.document.annotations.allSatisfy { $0.authored == fixture.authorship && fixture.source.metadata.intervals.map(\.target).contains($0.target) })
    #expect(try FeedbackArtifactStore.load(revision.reference, source: fixture.source, directory: fixture.directory).document == revision.document)
}

@Test(arguments: ["missing", "identity", "source", "checksum", "oversized"])
func invalidPreviewNeverManufacturesPixelsOrAllowsCompletion(mode: String) async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    await fixture.harness.invalid(mode)
    let model = try fixture.model(maximumBytes: mode == "oversized" ? 16 : FrameArchive.maximumFrameBytes)
    await model.open(); try await feedbackEventually { !model.loadingFrames }
    #expect(model.image == nil && model.frameIssue != nil && !model.canConfirmSelected && !model.canSave)
    await model.cancel()
}

@Test func rapidSelectionKeepsOneLoadAndDiscardsStalePixels() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    await fixture.harness.blockFirst()
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { await fixture.harness.activeLoads == 1 }
    model.select(1); model.select(2); model.selectPoint(.after)
    #expect(await fixture.harness.requests.count == 1)
    await fixture.harness.release()
    try await feedbackEventually { !model.loadingFrames && model.image != nil }
    #expect(model.observation?.observationID == fixture.source.metadata.intervals[2].endpointObservationID)
    #expect(await fixture.harness.requests.count == 2)
    #expect(await fixture.harness.peakLoads == 1)
    await model.cancel()
}

@Test func cancellationWaitsForOwnedFrameLoaderBeforeReturning() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    await fixture.harness.blockFirst()
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { await fixture.harness.activeLoads == 1 }
    let cancelling = Task { await model.cancel() }
    try await feedbackEventually { model.phase == .closing }
    #expect(model.takeOutcome() == nil)
    await fixture.harness.release(); await cancelling.value
    #expect(await fixture.harness.activeLoads == 0)
    guard case .cancelled? = model.takeOutcome() else { Issue.record("Cancellation did not join/preserve the draft"); return }
}

@Test func boundaryLossBeforeOpeningPreventsFrameAccessAndPreservesDraft() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    await fixture.harness.reject()
    let model = try fixture.model(); await model.open()
    #expect(model.phase == .blocked && model.image == nil && model.draftReference != nil)
    #expect(await fixture.harness.requests.isEmpty)
    await model.cancel()
}

@Test func postPublicationBoundaryLossKeepsRevisionUnadmittedAndDraftRecoverable() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open(); try await viewFeedbackInterval(model, index: 0); await model.markReviewed()
    await fixture.harness.failNextBoundary(after: 1)
    await model.save()
    #expect(model.phase == .blocked && model.takeOutcome() == nil && model.unadmittedRevision != nil && model.draftReference != nil)
    let revision = try #require(model.unadmittedRevision)
    #expect(try FeedbackArtifactStore.load(revision.reference, source: fixture.source, directory: fixture.directory).document == revision.document)
    await model.cancel()
    guard case .cancelled(let reference, let retained)? = model.takeOutcome() else { Issue.record("Missing blocked cancellation result"); return }
    #expect(retained == revision.reference)
    _ = try await FeedbackReviewDraftStore.load(reference, source: fixture.source, parent: nil)
}

@Test func failedDraftPublicationKeepsEditsInMemoryAndCanBeRetried() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open(); try await viewFeedbackInterval(model, index: 0)
    await model.setCount(packetID: model.selected.target.packetID, ruleID: fixture.positive, count: 2); await model.markReviewed()
    try Data("occupied".utf8).write(to: fixture.directory)
    await model.save()
    #expect(model.phase == .blocked && model.takeOutcome() == nil && model.draftReference == nil)
    #expect(model.cell(packetID: model.selected.target.packetID, ruleID: fixture.positive).count == 2)
    try FileManager.default.removeItem(at: fixture.directory)
    await model.cancel()
    guard case .cancelled(let reference, _)? = model.takeOutcome() else { Issue.record("Draft retry failed"); return }
    let draft = try await FeedbackReviewDraftStore.load(reference, source: fixture.source, parent: nil)
    #expect(draft.cells.contains { $0.count == 2 && $0.reviewed })
}
@Test(arguments: [false, true])
func cancellationDuringPublicationNeverReportsLearningReadiness(afterPublication: Bool) async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open(); try await viewFeedbackInterval(model, index: 0); await model.markReviewed()
    await fixture.harness.holdNextBoundary(after: afterPublication ? 1 : 0)
    let saving = Task { await model.save() }
    try await feedbackEventually { await fixture.harness.boundaryWaiting }
    let cancelling = Task { await model.cancel() }
    try await feedbackEventually { !model.canCancel }
    #expect(model.takeOutcome() == nil)
    await fixture.harness.releaseBoundary(); await saving.value; await cancelling.value
    guard case .cancelled(let draft, let retained)? = model.takeOutcome() else { Issue.record("Cancelled publication claimed a saved/ready result"); return }
    #expect((retained != nil) == afterPublication)
    _ = try await FeedbackReviewDraftStore.load(draft, source: fixture.source, parent: nil)
    if !afterPublication { #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.directory.path) == ["Drafts"]) }
}

@Test func boundaryLostDuringLoadingRejectsThePreviewAndPreservesExistingDraft() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    await fixture.harness.blockFirst()
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { await fixture.harness.activeLoads == 1 }
    await fixture.harness.reject(); await fixture.harness.release()
    try await feedbackEventually { model.phase == .blocked && model.draftReference != nil }
    #expect(model.image == nil && !model.canEdit && !model.canSave)
    await model.cancel()
}

@Test func tamperedDraftCannotRestoreAReviewedJudgment() async throws {
    let fixture = try FeedbackReviewFixture(); defer { fixture.remove() }
    let model = try fixture.model(); await model.open()
    try await feedbackEventually { !model.loadingFrames }
    await model.setCount(packetID: model.selected.target.packetID, ruleID: fixture.positive, count: 2)
    await model.cancel()
    guard case .cancelled(let reference, _)? = model.takeOutcome() else { Issue.record("Missing draft"); return }
    var data = try Data(contentsOf: reference.url); data.append(Data(" ".utf8)); try data.write(to: reference.url)
    await #expect(throws: (any Error).self) { try await FeedbackReviewDraftStore.load(reference, source: fixture.source, parent: nil) }
}

}
