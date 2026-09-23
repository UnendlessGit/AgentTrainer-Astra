import Foundation
import Testing
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

/// Explicit generated-source qualification. Real worker processes and the
/// production review model; no screen capture, event tap or OS input posting.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["ASTRA_HOST_QUALIFICATION_FEEDBACK"] == "1",
                            "Requires explicit real-process feedback qualification"))
@MainActor struct DesktopFeedbackQualificationTests {
    @Test func actualHostReviewsOriginalExperienceAndLearns() async throws {
        let environment = ProcessInfo.processInfo.environment
        let output = URL(fileURLWithPath: try #require(environment["ASTRA_HOST_QUALIFICATION_ROOT"]), isDirectory: true)
        let root = output.appendingPathComponent("Library", isDirectory: true)
        let bundlePath = try #require(environment["ASTRA_HOST_QUALIFICATION_BUNDLE"])
        let bundle = try #require(Bundle(url: URL(fileURLWithPath: bundlePath)))
        let initialInfo = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: output.appendingPathComponent("initial.json")))
        let lease = try LibraryLease(root: root); defer { withExtendedLifetime(lease) {} }
        let store = try LibraryStore(root: root), agent = AgentDocument(name: "Generated feedback qualification")
        try await store.save(agent)
        let initial = CheckpointDocument(id: try initialInfo.requiredUUID("id"), agentID: agent.id, runID: nil,
            name: "Real numerical-test policy", kind: "initial", trainingStep: 0,
            policySignature: try initialInfo.required("policySignature").decode(String.self),
            parameterCount: try initialInfo.required("parameterCount").decode(Int.self))
        try await store.saveCheckpoint(initial)
        let controls = HostQualificationControls(), capture = HostQualificationCapture(), owner = NativeControlOwner()
        let processes = HostQualificationProcesses(executable: bundle.bundleURL.appendingPathComponent("Contents/Helpers/AstraCompute.app/Contents/MacOS/AstraCompute"))
        let dependencies = InferenceDependencies(runtime: processes.actor, capture: { _ in capture.runtime }, activate: { _ in },
            countdownSeconds: 0, protectsPhysicalInputs: false, controlOwner: owner)
        let learner = LearningCoordinator(store: store, root: root, bundle: bundle, changed: {})
        let host = DesktopLearningHost(store: store, root: root, learner: learner, bundle: bundle,
            dependencies: dependencies, controlFactory: controls.factory, collectorFactory: processes.collector,
            scopeVerifier: { source, scope in
                guard source.id == capture.surface.id, scope.surfaces == [capture.surface] else {
                    throw AstraError("qualification.scope", "The host changed its generated environment.")
                }
                return MonotonicClock.now
            }, detector: { signals, _, _, _ in
                guard signals.allSatisfy({ $0.kind == .elapsedSeconds }) else {
                    throw AstraError("qualification.detector", "This fixture does not use visual detectors.")
                }
                return []
            }, changed: {})
        let elapsed = RewardSignal(name: "Episode time", kind: .elapsedSeconds)
        let feedback = RewardRule(name: "Good behavior", kind: .manualMarker, amount: 2)
        var program = RewardProgram(name: "Generated feedback episodes", signals: [elapsed],
            rules: [.init(name: "Time reward", kind: .ratePerSecond, amount: 1), feedback])
        program.maximumEpisodeMS = 1000
        program.success = .init(conditions: [.init(signalID: elapsed.id, comparison: .atLeast, number: 0.35)])
        var options = DesktopLearningOptions(); options.initialCheckpointID = initial.id; options.iterations = 1
        options.training.rolloutDecisions = 6; options.training.epochs = 1
        options.training.sequenceLength = 2; options.training.burnIn = 1; options.training.effectiveBatchDecisions = 4
        var phases: [String] = [], confirmations = Set<UUID>(), presentations = Set<UUID>()
        var reviewedIntervals = 0, displayedObservations = 0
        var learned: CheckpointDocument?, reopened: CheckpointDocument?, continued: CheckpointDocument?, failure: String?
        var continuationBatchID: UUID?, prefixDecisions = 0
        var continuedReviewIdentities: [[String: String]] = []
        do {
            try host.start(agent: agent, source: capture.source, program: program, options: options)
            let deadline = ContinuousClock.now + .seconds(120)
            while host.isBusy {
                if phases.last != host.phase { phases.append(host.phase) }
                if let reset = host.awaitingReady, confirmations.insert(reset).inserted { host.confirmReady() }
                if let review = host.reviewPresentation, presentations.insert(review.id).inserted {
                    try #require(owner.priorCleanupJoined && controls.all.allSatisfy { $0.hasJoined && $0.backend.clean })
                    let count = try await reviewOriginalExperience(review, marker: feedback.id)
                    reviewedIntervals += count; displayedObservations += count * 2
                }
                guard ContinuousClock.now < deadline else { throw AstraError("qualification.timeout", "The feedback host did not finish within 120 seconds.") }
                try await Task.sleep(for: .milliseconds(5))
            }
            phases.append(host.phase)
            try #require(host.failure == nil, "Host: \(host.failure ?? "") / learner: \(learner.failure ?? "")")
            let saved = try #require(host.checkpoint); learned = saved
            try #require(saved.id != initial.id && saved.trainingStep > 0)
            let snapshot = try await store.snapshot()
            try #require(snapshot.checkpoints.contains { $0.matchesIdentity(of: saved) })
            try #require(snapshot.agents.first?.selectedCheckpointID == saved.id)
            try #require(snapshot.learningRuns.contains { $0.checkpointID == saved.id && $0.status == .completed })
            try #require(snapshot.pendingFeedback.count == 1 && snapshot.pendingFeedback[0].status == .completed)
            try #require(!presentations.isEmpty && reviewedIntervals >= 6)
            try #require(owner.priorCleanupJoined && capture.hasJoined && controls.all.allSatisfy { $0.hasJoined && $0.backend.clean })

            // Suspend one subsequent behavior batch at its original review
            // boundary, then reopen it without capture or actor processes.
            options.initialCheckpointID = saved.id; options.resume = true; options.iterations = 2
            try host.start(agent: agent, source: capture.source, program: program, options: options)
            let suspendDeadline = ContinuousClock.now + .seconds(90)
            while host.isBusy && host.reviewPresentation == nil {
                if phases.last != host.phase { phases.append(host.phase) }
                if let reset = host.awaitingReady, confirmations.insert(reset).inserted { host.confirmReady() }
                guard ContinuousClock.now < suspendDeadline else { throw AstraError("qualification.timeout", "The next feedback batch did not reach review.") }
                try await Task.sleep(for: .milliseconds(5))
            }
            try #require(host.reviewPresentation != nil, "\(host.failure ?? learner.failure ?? host.phase)")
            await host.stopAndWait()
            try #require(host.failure == nil, "\(host.failure ?? "")")
            let pending = try #require(try await store.snapshot().pendingFeedback.first { $0.status == .awaitingReview })
            try #require(pending.checkpoint.matchesIdentity(of: saved) && pending.fragments.count == 1)
            try #require(pending.fragments[0].draft != nil)
            let beforeReopen = await processes.statuses(), captureCount = capture.produced, controlsCount = controls.all.count
            try host.resumeFeedback(pending, agent: agent)
            let reopenDeadline = ContinuousClock.now + .seconds(90)
            while host.isBusy {
                if phases.last != host.phase { phases.append(host.phase) }
                if let review = host.reviewPresentation, presentations.insert(review.id).inserted {
                    let count = try await reviewOriginalExperience(review, marker: feedback.id)
                    reviewedIntervals += count; displayedObservations += count * 2
                }
                guard ContinuousClock.now < reopenDeadline else { throw AstraError("qualification.timeout", "The saved feedback batch did not finish.") }
                try await Task.sleep(for: .milliseconds(5))
            }
            try #require(host.failure == nil, "\(host.failure ?? learner.failure ?? host.phase)")
            let resumed = try #require(host.checkpoint); reopened = resumed
            try #require(resumed.id != saved.id && resumed.trainingStep > saved.trainingStep)
            let afterReopen = try await store.snapshot()
            try #require(afterReopen.pendingFeedback.count == 2 && afterReopen.pendingFeedback.allSatisfy { $0.status == .completed })
            try #require(afterReopen.agents.first?.selectedCheckpointID == resumed.id)
            try #require(afterReopen.learningRuns.contains { $0.checkpointID == resumed.id && $0.status == .completed })
            try #require(await processes.statuses() == beforeReopen && capture.produced == captureCount && controls.all.count == controlsCount)

            // Stop at the second Ready prompt: the first complete episode is a
            // valid immutable prefix, but remains below the original minimum.
            options.initialCheckpointID = resumed.id; options.resume = true; options.iterations = 3
            try host.start(agent: agent, source: capture.source, program: program, options: options)
            var prefixReady = Set<UUID>()
            let prefixDeadline = ContinuousClock.now + .seconds(90)
            while host.isBusy {
                if phases.last != host.phase { phases.append(host.phase) }
                if let reset = host.awaitingReady, prefixReady.insert(reset).inserted {
                    if prefixReady.count == 1 { confirmations.insert(reset); host.confirmReady() }
                    else { await host.stopAndWait(); break }
                }
                guard ContinuousClock.now < prefixDeadline else { throw AstraError("qualification.timeout", "The fragment did not reach its next reset boundary.") }
                try await Task.sleep(for: .milliseconds(5))
            }
            try #require(prefixReady.count == 2 && host.failure == nil, "\(host.failure ?? learner.failure ?? host.phase)")
            let prefix = try #require(try await store.snapshot().pendingFeedback.first { $0.status == .awaitingReview })
            prefixDecisions = prefix.collectedDecisions; continuationBatchID = prefix.id
            try #require(prefix.fragments.count == 1 && prefixDecisions > 0 && prefixDecisions < options.training.rolloutDecisions)
            let original = prefix.fragments[0]
            let originalIdentity = try original.boundary.decode(PendingReviewBoundary.self).identity
            let originalDirectory = URL(fileURLWithPath: original.sourceDirectory, isDirectory: true)
            let originalFiles = try ["manifest.json", "source.json", "program.json", "decisions.ndjson", "review-frames.ndjson", "frames.bgra", "journal.ndjson", "audit.json"]
                .map { ($0, try Data(contentsOf: originalDirectory.appendingPathComponent($0))) }
            try host.continueFeedback(prefix, agent: agent, source: capture.source)
            let continueDeadline = ContinuousClock.now + .seconds(90)
            while host.isBusy {
                if phases.last != host.phase { phases.append(host.phase) }
                if let reset = host.awaitingReady, confirmations.insert(reset).inserted { host.confirmReady() }
                if let review = host.reviewPresentation, presentations.insert(review.id).inserted {
                    let source = review.model.source.metadata
                    continuedReviewIdentities.append(["runID": source.runID.uuidString.lowercased(), "clockID": source.clockID.uuidString.lowercased()])
                    let count = try await reviewOriginalExperience(review, marker: feedback.id)
                    reviewedIntervals += count; displayedObservations += count * 2
                }
                guard ContinuousClock.now < continueDeadline else { throw AstraError("qualification.timeout", "The continued feedback fragments did not finish learning.") }
                try await Task.sleep(for: .milliseconds(5))
            }
            try #require(host.failure == nil, "\(host.failure ?? learner.failure ?? host.phase)")
            let combined = try #require(host.checkpoint); continued = combined
            try #require(combined.id != resumed.id && combined.trainingStep > resumed.trainingStep)
            let afterContinuation = try await store.snapshot()
            let completedPrefix = try #require(afterContinuation.pendingFeedback.first { $0.id == prefix.id })
            try #require(completedPrefix.status == .completed && completedPrefix.fragments.count == 2)
            try #require(completedPrefix.fragments[0].boundary == original.boundary)
            try #require(afterContinuation.agents.first?.selectedCheckpointID == combined.id)
            try #require(afterContinuation.learningRuns.contains { $0.checkpointID == combined.id && $0.status == .completed })
            for (name, bytes) in originalFiles {
                try #require(try Data(contentsOf: originalDirectory.appendingPathComponent(name)) == bytes, "Original source \(name) was rewritten")
            }
            let nextIdentity = try completedPrefix.fragments[1].boundary.decode(PendingReviewBoundary.self).identity
            try #require(nextIdentity.runID != originalIdentity.runID && nextIdentity.clockID != originalIdentity.clockID)
            try #require(continuedReviewIdentities == [originalIdentity, nextIdentity].map {
                ["runID": $0.runID.uuidString.lowercased(), "clockID": $0.clockID.uuidString.lowercased()]
            })
            try #require(owner.priorCleanupJoined && capture.hasJoined && controls.all.allSatisfy { $0.hasJoined && $0.backend.clean })
            let statuses = await processes.statuses()
            try #require(statuses.values.flatMap { $0 }.allSatisfy { $0 == 0 })
        } catch {
            failure = host.failure ?? learner.failure ?? error.localizedDescription
            await host.stopAndWait(); await learner.stopAndWait()
        }
        let report: [String: Any] = ["completed": failure == nil, "issue": failure ?? NSNull(),
            "privacyPermissionsUsed": false, "physicalInputPosted": false, "personalPixelsRead": false,
            "initialCheckpointID": initial.id.uuidString.lowercased(), "learnedCheckpointID": learned?.id.uuidString.lowercased() ?? NSNull(),
            "reopenedCheckpointID": reopened?.id.uuidString.lowercased() ?? NSNull(),
            "continuedCheckpointID": continued?.id.uuidString.lowercased() ?? NSNull(),
            "continuationBatchID": continuationBatchID?.uuidString.lowercased() ?? NSNull(), "prefixDecisions": prefixDecisions,
            "continuedReviewIdentities": continuedReviewIdentities,
            "trainingUpdates": learned?.trainingStep ?? 0, "readyConfirmations": confirmations.count,
            "reviewPresentations": presentations.count, "reviewedIntervals": reviewedIntervals,
            "displayedObservations": displayedObservations, "workerExitStatuses": await processes.statuses(),
            "controlCleanupConfirmed": owner.priorCleanupJoined, "captureJoined": capture.hasJoined,
            "controlReceipts": controls.all.map { $0.receipts.count }, "phases": phases,
            "hostFailure": host.failure ?? NSNull(), "learnerFailure": learner.failure ?? NSNull()]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("desktop-host-report.json"), options: .atomic)
        if let failure { throw AstraError("qualification.feedback", failure) }
    }

    private func reviewOriginalExperience(_ review: FeedbackReviewPresentation, marker: UUID) async throws -> Int {
        let model = review.model
        await model.open(); model.setPresentationActive(true)
        try #require(model.rules.map(\.id) == [marker])
        for index in model.intervals.indices {
            model.select(index); model.selectPoint(.before); try await preview(model)
            model.didDisplay(try #require(model.previewIdentity))
            model.selectPoint(.after); try await preview(model)
            model.didDisplay(try #require(model.previewIdentity))
            try #require(model.canConfirmSelected)
            // Nonzero annotations and explicit reviewed zeros use the same
            // public UI-model operations as a visible review sheet.
            await model.setCount(packetID: model.selected.target.packetID, ruleID: marker, count: index == 0 ? 1 : 0)
            await model.markReviewed()
        }
        try #require(model.complete && model.canSave)
        await model.save()
        try #require(model.phase == .finished, "\(model.issue ?? "Review did not finish")")
        review.finish(try #require(model.takeOutcome()))
        return model.intervals.count
    }

    private func preview(_ model: FeedbackReviewModel) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while model.loadingFrames {
            guard ContinuousClock.now < deadline else { throw AstraError("qualification.preview", "Original feedback pixels did not load.") }
            try await Task.sleep(for: .milliseconds(2))
        }
        try #require(model.image != nil && model.frameIssue == nil, "\(model.frameIssue ?? model.issue ?? "No original feedback image")")
    }
}
