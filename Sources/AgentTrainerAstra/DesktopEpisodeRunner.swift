import Foundation
import AstraCore
import AstraPlatform

struct DesktopManualObservation: Sendable {
    let generationID: UUID
    let episodeID: UUID
    let observationID: UUID
    let cutoffNanos: UInt64
}

/// A bound manual source owns its event time/coverage policy. Closing must
/// actually join its callbacks; a timeout or empty marker list is not a seal.
struct DesktopEpisodeManualProducer: Sendable {
    /// A bounded synchronous offer; queue work rather than waiting for user input.
    let observe: @Sendable (DesktopManualObservation, @escaping @Sendable (DesktopManualSeal) throws -> Void) throws -> Void
    let closeAndJoin: @Sendable () async throws -> Void
}

enum DesktopEpisodePhase: String, Sendable { case preparing, arming, running, drainingPrediction, releasingControl, joiningEvidence, finished }

struct DesktopEpisodeRunResult: Sendable {
    let generationID: UUID
    let episodeID: UUID
    let stop: DesktopEpisodeStop
    let terminal: DesktopTerminalEvidence?
    let producedDecisions: Int
    let admittedPackets: Int
    let executedPackets: Int
    let lastProducedSequence: UInt64?
    let actorState: PolicyActorState
    let control: NativeControlCompletion?
    let cleanupConfirmed: Bool
    let evidence: DesktopEpisodeCompletion?
    let join: DesktopEpisodeJoin?
    let issue: AstraError?
}

/// One physical policy episode, with persistent actor/capture/collector owners
/// supplied by the parent. Only the fresh control child and this evidence bridge
/// end here. The actor's original sampled result is drained before closure.
final class DesktopEpisodeRunner: @unchecked Sendable {
    typealias CaptureRead = @Sendable () throws -> InferenceImage?
    typealias ScopeVerification = @Sendable () async throws -> UInt64
    let generationID = UUID()
    let episodeID: UUID
    private let actor: PolicyActorSession
    private let checkpoint: CheckpointDocument
    private let ready: ResetResult
    private let scope: ControlScope
    private let identity: DesktopEvidenceIdentity
    private let collector: CollectorSession
    private let sequence: DesktopEnvironmentSequence
    private let program: RewardProgram
    private let assetRoot: URL
    private let captureRead: CaptureRead
    private let verifyScope: ScopeVerification
    private let factory: NativeControlRuntimeFactory
    private let controlOwner: NativeControlOwner
    private let recoveryDirectory: URL
    private let manual: DesktopEpisodeManualProducer?
    private let deferManualFeedback: Bool
    private let detector: RewardAnalysisQueue.Detector
    private let limits: DesktopEpisodeEvidence.Limits
    private let clock: @Sendable () -> UInt64
    private let onPhase: @Sendable (DesktopEpisodePhase) -> Void
    private let lock = NSLock()
    private var work: Task<DesktopEpisodeRunResult, Never>?
    private var bridge: DesktopEpisodeEvidence?
    private var control: NativeControlSession?
    private var ending: DesktopEpisodeStop?
    private var terminal: DesktopTerminalEvidence?
    private var issue: AstraError?
    private var finished = false
    private var manualClosed = false
    private var manualOffers = 0
    private var predictionDeadline: UInt64?
    private var phaseValue: DesktopEpisodePhase = .preparing

    init(actor: PolicyActorSession, checkpoint: CheckpointDocument, reset: ResetResult,
         identity: DesktopEvidenceIdentity, collector: CollectorSession, sequence: DesktopEnvironmentSequence,
         program: RewardProgram, assetRoot: URL, captureRead: @escaping CaptureRead,
         verifyScope: @escaping ScopeVerification, controlFactory: NativeControlRuntimeFactory,
         controlOwner: NativeControlOwner = .shared, recoveryDirectory: URL,
         manualProducer: DesktopEpisodeManualProducer? = nil, deferManualFeedback: Bool = false,
         evidenceLimits: DesktopEpisodeEvidence.Limits = .init(),
         detector: @escaping RewardAnalysisQueue.Detector = { try VisualRewardDetector.read(signals: $0, frames: $1, episodeID: $2, templates: $3) },
         clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now },
         onPhase: @escaping @Sendable (DesktopEpisodePhase) -> Void = { _ in }) {
        self.actor = actor; self.checkpoint = checkpoint; ready = reset; scope = reset.context.scope
        episodeID = reset.context.nextEpisodeID; self.identity = identity; self.collector = collector; self.sequence = sequence
        self.program = program; self.assetRoot = assetRoot; self.captureRead = captureRead; self.verifyScope = verifyScope
        factory = controlFactory; self.controlOwner = controlOwner; self.recoveryDirectory = recoveryDirectory
        manual = manualProducer; limits = evidenceLimits; self.detector = detector; self.clock = clock; self.onPhase = onPhase
        self.deferManualFeedback = deferManualFeedback
    }

    var phase: DesktopEpisodePhase { lock.withLock { phaseValue } }

    func run() async -> DesktopEpisodeRunResult {
        let task = lock.withLock { () -> Task<DesktopEpisodeRunResult, Never> in
            if let work { return work }
            let task = Task { await self.perform() }; work = task; return task
        }
        return await withTaskCancellationHandler { await task.value } onCancel: { self.requestStop() }
    }

    func requestStop(reason: String = "The operator stopped the episode.") {
        end(.operatorAbort, error: nil, auditReason: reason)
    }

    func sourceFailed(_ error: AstraError, generation: UUID) {
        guard generation == generationID else { return }
        end(.failure(error.message), error: error, auditReason: error.message)
    }

    func offerManualSeal(_ seal: DesktopManualSeal, generation: UUID) throws {
        let target = try lock.withLock { () throws -> DesktopEpisodeEvidence in
            guard generation == generationID, seal.episodeID == episodeID, !finished, !manualClosed, let bridge else {
                throw AstraError("desktop.manualGeneration", "Manual evidence belongs to an inactive episode producer.")
            }
            manualOffers += 1; return bridge
        }
        defer { lock.withLock { manualOffers -= 1 } }
        do { try target.offerManualSeal(seal, generation: generationID) }
        catch { sourceFailed(asFailure(error), generation: generationID); throw error }
    }

    private func end(_ stop: DesktopEpisodeStop, error: AstraError?, auditReason: String?) {
        let evidence = lock.withLock { () -> DesktopEpisodeEvidence? in
            guard !finished else { return nil }
            // Native Stop is synchronous admission closure and never calls
            // back under its lock. Close that gate before publishing our stop.
            control?.requestStop()
            if let error { issue = issue ?? error; ending = .failure(issue!.message) }
            else if issue == nil {
                switch stop {
                case .operatorAbort: ending = .operatorAbort
                case .semanticBoundary: if ending == nil { ending = stop }
                case .failure: ending = stop
                }
            }
            return bridge
        }
        if let auditReason {
            let reason = auditReason.isEmpty ? "Episode stopped." : String(auditReason.prefix(512))
            try? evidence?.requestAuditAbort(reason: reason, generation: generationID)
        }
    }
    private func receivedTerminal(_ value: DesktopTerminalEvidence) {
        guard value.generationID == generationID, value.episodeID == episodeID else { return }
        lock.withLock { if terminal == nil { terminal = value } }
        end(.semanticBoundary(value.cutoffNanos), error: nil, auditReason: nil)
    }
    private func receivedControl(_ event: NativeControlEvent, bridge: DesktopEpisodeEvidence) throws {
        try bridge.offer(.control(event.message), generation: generationID)
        guard event.message.kind == "control.stopped" else { return }
        let cause = event.message.payload.fields?["cause"]?.text ?? "fault"
        let reason = event.message.payload.fields?["reason"]?.text ?? "Desktop control stopped."
        if ["physicalTakeover", "emergencyStop"].contains(cause) { requestStop(reason: reason) }
        else if cause != "requested" || lock.withLock({ ending == nil }) {
            sourceFailed(AstraError("desktop.controlStop", reason), generation: generationID)
        }
    }
    private func controlFailed(_ id: UUID, _ error: AstraError) {
        let relevant = lock.withLock { control?.sessionID == id && !finished }
        guard relevant else { return }
        if ["control.physicalTakeover", "control.emergencyStop"].contains(error.code) { requestStop(reason: error.message) }
        else if error.code != "control.rejected" || lock.withLock({ ending == nil }) { sourceFailed(error, generation: generationID) }
    }
    private func setPhase(_ value: DesktopEpisodePhase) {
        lock.withLock { phaseValue = value }; onPhase(value)
    }
    private func checkAdmission() throws {
        if lock.withLock({ ending != nil }) { throw CancellationError() }
    }
    private func asFailure(_ error: any Error) -> AstraError { (error as? AstraError) ?? AstraError("desktop.episode", error.localizedDescription) }

    private func perform() async -> DesktopEpisodeRunResult {
        var produced = 0, admitted = 0
        var lastProduced: UInt64?
        var watchdog: Task<Void, Never>?
        var prediction: PolicyActorPrediction?
        do {
            setPhase(.preparing)
            _ = try checkpoint.validated(); _ = try scope.validated(); _ = try program.validated()
            let requiresManual = program.signals.contains { $0.kind == .manual } || (!deferManualFeedback && program.rules.contains { $0.kind == .manualMarker })
            guard !requiresManual || manual != nil, scope.surfaces.count == 1, ready.status == .ready, ready.cleanupConfirmed,
                  ready.context.environmentID == identity.environmentID, collector.runID == identity.runID,
                  sequence.identity == identity, controlOwner.priorCleanupJoined,
                  let binding = await actor.binding, binding.collecting, binding.isAvailable,
                  binding.checkpoint.matchesIdentity(of: checkpoint), binding.state.episodeID == episodeID,
                  binding.state.episodeStep == 0, binding.state.stateID != nil, binding.state.sampledProgressKnown,
                  !binding.state.stopped, !binding.state.joined, binding.state.nextPacketSequence == binding.state.nextDrawIndex,
                  binding.runID == identity.runID else {
                throw AstraError("desktop.episodeBinding", "The episode needs a ready reset, unchanged categorical actor, joined controls and bound evidence producers.")
            }
            try checkAdmission()
            try await verifyCurrentScope()
            let image = try await firstFreshFrame()
            let warm = try await Task.detached { RewardImageFrame(metadata: image.metadata, pixels: try image.pixels()) }.value
            guard warm.pixels.count == warm.metadata.byteCount else { throw AstraError("desktop.captureBytes", "The captured image does not match its declared byte size.") }
            let evidence = try await DesktopEpisodeEvidence.prepare(identity: identity, episodeID: episodeID, generationID: generationID,
                policyID: checkpoint.id, policySignature: checkpoint.policySignature, collector: collector, sequence: sequence,
                program: program, scope: scope, assetRoot: assetRoot, warmupFrames: [warm], limits: limits, deferManualFeedback: deferManualFeedback, detector: detector,
                onTerminal: { [weak self] in self?.receivedTerminal($0) },
                onFault: { [weak self] in self?.sourceFailed($0.error, generation: $0.generationID) })
            lock.withLock { bridge = evidence }
            try evidence.offer(.prepared(runID: identity.runID, actor: .object([
                "checkpointID": .string(binding.checkpoint.id.uuidString.lowercased()),
                "policySignature": .string(binding.checkpoint.policySignature), "collectionVersion": .integer(1),
                "deterministic": .bool(false), "nextPacketSequence": .unsigned(binding.state.nextPacketSequence)])), generation: generationID)
            try await evidence.confirmReady(ready)
            try checkAdmission()
            try await verifyCurrentScope()
            let configured = try NativeControlConfiguration(runID: identity.runID, scope: scope,
                capabilities: binding.policy.capabilities, packetCapacity: binding.policy.capacity,
                initialPacketSequence: binding.state.nextPacketSequence, recoveryDirectory: recoveryDirectory)
            let native = NativeControlSession(configuration: configured, owner: controlOwner, runtimeFactory: factory,
                onEvent: { [weak self, evidence] event in
                    guard let self else { throw AstraError("desktop.episodeOwner", "The episode owner was retired before control joined.") }
                    try self.receivedControl(event, bridge: evidence)
                }, onFailure: { [weak self] in self?.controlFailed($0, $1) })
            setPhase(.arming)
            try lock.withLock {
                guard ending == nil, !finished else { throw CancellationError() }
                control = native
            }
            _ = try await native.start()
            try checkAdmission(); try await verifyCurrentScope()
            _ = try currentFrame()
            watchdog = startWatchdog(native)
            setPhase(.running)
            var nextCutoff = clock(), cursor: UInt64?
            while lock.withLock({ ending == nil }) {
                try await sleepUntil(nextCutoff); try checkAdmission()
                let frame = try currentFrame()
                let controls = try await settledObservation(native, after: cursor)
                try checkAdmission()
                let cutoff = controls.cutoffNanos
                let deadline = cutoff.addingReportingOverflow(UInt64(binding.policy.leadMS) * 1_000_000)
                guard !deadline.overflow else { throw AstraError("desktop.clock", "The episode decision clock is exhausted.") }
                lock.withLock { predictionDeadline = deadline.partialValue }
                let ticket = try await actor.beginPrediction(PolicyActorSnapshot(frame: frame, controls: controls), onObservation: { [weak self, evidence] owned in
                    guard let self else { throw AstraError("desktop.episodeOwner", "The episode owner was retired before observation retention.") }
                    let observation = try owned.singleSource()
                    try evidence.offer(.observation(observation), generation: self.generationID)
                    if let manual = self.manual {
                        let context = DesktopManualObservation(generationID: self.generationID, episodeID: self.episodeID,
                            observationID: try owned.actorInput.requiredUUID("observationID"), cutoffNanos: cutoff)
                        try manual.observe(context, { [weak self] seal in
                            guard let self else { throw AstraError("desktop.manualOwner", "The manual episode producer was retired.") }
                            try self.offerManualSeal(seal, generation: context.generationID)
                        })
                    }
                })
                prediction = ticket
                let result = try await ticket.value()
                prediction = nil
                lock.withLock { predictionDeadline = nil }
                produced += 1; lastProduced = result.packet.sequence
                guard result.packet.runID == identity.runID, result.observation.actorInput.fields?["episodeID"]?.uuid == episodeID,
                      result.response.payload.fields?["checkpointID"]?.uuid == checkpoint.id,
                      result.response.payload.fields?["policySignature"]?.text == checkpoint.policySignature else {
                    throw AstraError("desktop.changedActor", "The actor changed episode or policy while producing its result.")
                }
                do { try evidence.offer(.decision(result.response.payload), generation: generationID) }
                catch { sourceFailed(asFailure(error), generation: generationID) }
                if clock() >= result.packet.executeAtNanos, lock.withLock({ ending == nil }) {
                    sourceFailed(AstraError("desktop.policyDeadline", "The actor missed its immutable execution lead."), generation: generationID)
                }
                if lock.withLock({ ending != nil }) { setPhase(.drainingPrediction) }
                let submission = try await native.submitPreservingStoppedPacket(result.packet)
                if submission.admitted { admitted += 1 }
                else if lock.withLock({ ending == nil }) { throw AstraError("desktop.admission", "The helper rejected the policy packet.") }
                cursor = controls.lastSequence
                let next = cutoff.addingReportingOverflow(UInt64(binding.policy.periodMS) * 1_000_000)
                guard !next.overflow else { throw AstraError("desktop.clock", "The episode cadence clock is exhausted.") }
                nextCutoff = next.partialValue
            }
        } catch is CancellationError {
            if lock.withLock({ ending == nil }) { sourceFailed(AstraError("desktop.actorCancelled", "The actor operation ended without an episode stop request."), generation: generationID) }
        } catch {
            let error = asFailure(error)
            if ["control.physicalTakeover", "control.emergencyStop"].contains(error.code) { requestStop(reason: error.message) }
            else if !["control.rejected", "control.inactive", "control.closed"].contains(error.code) || lock.withLock({ ending == nil }) {
                sourceFailed(error, generation: generationID)
            }
        }
        lock.withLock { predictionDeadline = nil }
        // A ticket usually joined in the loop; this also covers a future error
        // path between reservation and its value without cancelling the draw.
        if let prediction { _ = try? await prediction.value() }
        watchdog?.cancel(); await watchdog?.value
        let native = lock.withLock { control }
        native?.requestStop(); setPhase(.releasingControl)
        let completion = await native?.shutdown()
        let cleanup = completion?.cleanupConfirmed ?? (ready.cleanupConfirmed && controlOwner.priorCleanupJoined)
        if let error = completion?.issue, error.code != "control.rejected" {
            if ["control.physicalTakeover", "control.emergencyStop"].contains(error.code) { requestStop(reason: error.message) }
            else { sourceFailed(error, generation: generationID) }
        }
        if !cleanup { sourceFailed(AstraError("desktop.cleanupUnconfirmed", "Control cleanup remains unconfirmed."), generation: generationID) }
        setPhase(.joiningEvidence)
        var manualJoined = true
        if let manual {
            do { try await manual.closeAndJoin() }
            catch { manualJoined = false; sourceFailed(asFailure(error), generation: generationID) }
        }
        lock.withLock { manualClosed = true }
        while lock.withLock({ manualOffers > 0 }) { try? await Task.sleep(for: .milliseconds(1)) }
        let state = await actor.state
        if !state.sampledProgressKnown { sourceFailed(AstraError("desktop.unknownActorProgress", "The sampled actor result did not provide a trustworthy progress boundary."), generation: generationID) }
        let finalStop = lock.withLock { ending ?? .operatorAbort }
        var evidenceResult: DesktopEpisodeCompletion?
        var join: DesktopEpisodeJoin?
        if let bridge = lock.withLock({ bridge }) {
            do {
                let proof = DesktopEpisodeJoin(runID: identity.runID, episodeID: episodeID, generationID: generationID,
                    actorJoined: true, controlJoined: true, manualProducerJoined: manualJoined,
                    predictionResolved: state.sampledProgressKnown, cleanupConfirmed: cleanup,
                    stoppedNanos: clock(), lastProducedSequence: lastProduced, stop: finalStop)
                join = proof
                evidenceResult = try await bridge.finish(joined: proof)
            } catch { sourceFailed(asFailure(error), generation: generationID) }
        } else if produced == 0, state.sampledProgressKnown { evidenceResult = .empty }
        lock.withLock { finished = true }
        setPhase(.finished)
        return DesktopEpisodeRunResult(generationID: generationID, episodeID: episodeID, stop: lock.withLock { ending ?? .operatorAbort },
            terminal: lock.withLock { terminal }, producedDecisions: produced, admittedPackets: admitted,
            executedPackets: native?.executedPacketCount ?? 0, lastProducedSequence: lastProduced, actorState: state,
            control: completion, cleanupConfirmed: cleanup, evidence: evidenceResult, join: join, issue: lock.withLock { issue })
    }

    private func verifyCurrentScope() async throws {
        let checked = try await verifyScope(), now = clock()
        guard checked <= now, now - checked < NativeResetDriver.scopeFreshnessNanos else {
            throw AstraError("desktop.scopeProof", "Scope verification returned stale or future evidence.")
        }
        try checkAdmission()
    }
    private func currentFrame() throws -> InferenceImage {
        guard let image = try captureRead(), image.metadata.surface == scope.surfaces[0] else {
            throw AstraError("desktop.capture", "The bound capture is unavailable or changed geometry.")
        }
        _ = try image.metadata.validated()
        let now = clock()
        if let coverage = image.coverage { try coverage.validated(frame: image.metadata, cutoffNanos: now, maximumAgeMS: 250) }
        else {
            guard image.metadata.eventNanos <= image.metadata.observedNanos, image.metadata.observedNanos <= now,
                  now - image.metadata.eventNanos <= 250_000_000 else { throw AstraError("desktop.captureAge", "No recent source evidence is available.") }
        }
        return image
    }
    private func firstFreshFrame() async throws -> InferenceImage {
        let deadline = clock().addingReportingOverflow(5_000_000_000)
        guard !deadline.overflow else { throw AstraError("desktop.clock", "The source wait clock is exhausted.") }
        let barrier = ready.readyObservation?.observedNanos ?? UInt64.max
        while true {
            try checkAdmission()
            let image = try currentFrame()
            if (image.coverage?.throughNanos ?? image.metadata.eventNanos) >= barrier { return image }
            guard clock() < deadline.partialValue else { throw AstraError("desktop.startingFrame", "No source evidence arrived after Ready.") }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
    private func settledObservation(_ native: NativeControlSession, after cursor: UInt64?) async throws -> ControlObservation {
        let deadline = clock().addingReportingOverflow(20_000_000)
        guard !deadline.overflow else { throw AstraError("desktop.clock", "The control wait clock is exhausted.") }
        while true {
            let observed = try await native.observation(afterSequence: cursor)
            guard observed.intervalCovered, observed.cutoffNanos <= clock() else { throw AstraError("desktop.controlCoverage", "The control producer cannot prove observation coverage.") }
            if observed.controlState.valid {
                guard observed.controlCoverageNanos == observed.cutoffNanos else { throw AstraError("desktop.controlCoverage", "The control producer did not verify unchanged input through this cutoff.") }
                return observed
            }
            try checkAdmission()
            guard clock() < deadline.partialValue else { throw AstraError("desktop.controlBusy", "The control state did not settle for observation.") }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
    private func sleepUntil(_ cutoff: UInt64) async throws {
        while true {
            try checkAdmission()
            let now = clock()
            guard now < cutoff else { return }
            try await Task.sleep(nanoseconds: min(cutoff - now, 5_000_000))
        }
    }
    private func startWatchdog(_ native: NativeControlSession) -> Task<Void, Never> {
        Task { [weak self] in
            while !Task.isCancelled {
                guard let self, lock.withLock({ ending == nil }) else { return }
                do {
                    try native.checkHealth(); _ = try currentFrame()
                    if let deadline = lock.withLock({ predictionDeadline }), clock() >= deadline {
                        throw AstraError("desktop.policyDeadline", "The actor missed its immutable execution lead; waiting for its original result.")
                    }
                    try await Task.sleep(for: .milliseconds(5))
                } catch {
                    if !Task.isCancelled { controlFailed(native.sessionID, asFailure(error)) }
                    return
                }
            }
        }
    }
}
