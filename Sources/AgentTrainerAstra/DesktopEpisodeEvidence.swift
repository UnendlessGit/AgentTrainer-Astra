import Foundation
import AstraCore
import AstraPlatform

/// One physical episode with immutable producer, generation and collector
/// bindings. CPU pairing never waits for Vision, disk or learner execution.
final class DesktopEpisodeEvidence: @unchecked Sendable {
    struct Limits: Sendable {
        var maximumBytes = 256 * 1024 * 1024
        var maximumItems = 128
        var maximumEpisodeDecisions = 65_536
        var maximumPendingSeconds: Double = 30
    }
    let generationID: UUID
    let episodeID: UUID
    let identity: DesktopEvidenceIdentity
    private let policyID: UUID, policySignature: String
    private let collector: CollectorSession
    private let sequence: DesktopEnvironmentSequence
    private let program: RewardProgram, scope: ControlScope
    private let limits: Limits
    private let deferManualFeedback: Bool
    private let onTerminal: @Sendable (DesktopTerminalEvidence) -> Void
    private let onFault: @Sendable (DesktopEpisodeFault) -> Void
    private let owner = DispatchQueue(label: "astra.desktop.episodeEvidence", qos: .userInitiated)
    private let lock = NSLock()
    private var queuedBytes = 0, queuedItems = 0, controlItems = 0, completionItems = 0
    private var sourceClosed = false, allClosed = false, finishing = false
    private var storedFault: AstraError?
    private var analysis: RewardAnalysisQueue?
    private var timer: DispatchSourceTimer?
    // Only owner touches episode metadata below.
    private var ready: (resetID: UUID, nanos: UInt64)?
    private var began = false, auditOnly = false
    private var deferredAbortReason: String?
    private var seenEvaluationStarts: Set<UInt64> = []
    private var order: [UUID] = []
    private var entries: [UUID: Entry] = [:]
    private var packets: [UUID: PacketIdentity] = [:]
    private var byCutoff: [UInt64: UUID] = [:]
    private var pendingReceipts: [(JSONValue, ExecutionReceipt)] = []
    private var cancelledReceipts: [(JSONValue, ExecutionReceipt)] = []
    private var finalReceipts: Set<UUID> = [], admissions: Set<UUID> = []
    private var evaluations: [UInt64: RewardEvaluation] = [:]
    private var lastObservationCutoff: UInt64?, baseline: UInt64?, terminal: DesktopTerminalEvidence?
    private var lastProduced: UInt64?, lastStateID: UUID?, actorNextSequence: UInt64?
    private var produced = 0, nextActor = 0, nextAnalysis = 0
    private var controlStopCause: String?
    private var lastControlEventSequence: UInt64?

    private struct Reservation { let bytes: Int; let kind: Int }
    private struct PacketIdentity { let observationID: UUID; let sequence: UInt64; let cutoff: UInt64 }
    private struct Entry {
        var observation: InferenceCollectedObservation?
        var response: JSONValue?
        var manual: DesktopManualSeal?
        var observedReservation: Reservation?
        var responseReservation: Reservation?
        var manualReservation: Reservation?
        var analysisOffered = false
        var actorOffered = false
        var packetID: UUID?
        var cutoff: UInt64?
        var observationIndex: Int?
        let arrived = ContinuousClock.now
    }
    private enum Input: Sendable {
        case event(InferenceCollectionEvent)
        case manual(DesktopManualSeal)
        case result(RewardAnalysisResult)
        case abort(String)
    }
    private init(identity: DesktopEvidenceIdentity, episodeID: UUID, generationID: UUID, policyID: UUID, policySignature: String,
                 collector: CollectorSession, sequence: DesktopEnvironmentSequence, program: RewardProgram, scope: ControlScope,
                 limits: Limits, deferManualFeedback: Bool, onTerminal: @escaping @Sendable (DesktopTerminalEvidence) -> Void,
                 onFault: @escaping @Sendable (DesktopEpisodeFault) -> Void) throws {
        guard collector.runID == identity.runID, sequence.identity == identity, policySignature.utf8.count == 64,
              (2 * AstraVersion.maximumMessageBytes...1024 * 1024 * 1024).contains(limits.maximumBytes),
              (1...128).contains(limits.maximumItems), (1...65_536).contains(limits.maximumEpisodeDecisions),
              limits.maximumPendingSeconds.isFinite, (0.01...3600).contains(limits.maximumPendingSeconds) else {
            throw AstraError("desktop.evidenceConfiguration", "Episode evidence requires consistent identities and bounded queues.")
        }
        self.identity = identity; self.episodeID = episodeID; self.generationID = generationID
        self.policyID = policyID; self.policySignature = policySignature; self.collector = collector; self.sequence = sequence
        self.program = try program.validated(); self.scope = try scope.validated(); self.limits = limits
        self.deferManualFeedback = deferManualFeedback
        self.onTerminal = onTerminal; self.onFault = onFault
    }
    static func prepare(identity: DesktopEvidenceIdentity, episodeID: UUID, generationID: UUID = UUID(),
                        policyID: UUID, policySignature: String, collector: CollectorSession, sequence: DesktopEnvironmentSequence,
                        program: RewardProgram, scope: ControlScope, assetRoot: URL, warmupFrames: [RewardImageFrame],
                        limits: Limits = Limits(), deferManualFeedback: Bool = false, detector: @escaping RewardAnalysisQueue.Detector = {
                            try VisualRewardDetector.read(signals: $0, frames: $1, episodeID: $2, templates: $3)
                        }, onTerminal: @escaping @Sendable (DesktopTerminalEvidence) -> Void,
                        onFault: @escaping @Sendable (DesktopEpisodeFault) -> Void) async throws -> DesktopEpisodeEvidence {
        let result = try Self(identity: identity, episodeID: episodeID, generationID: generationID, policyID: policyID,
            policySignature: policySignature, collector: collector, sequence: sequence, program: program, scope: scope,
            limits: limits, deferManualFeedback: deferManualFeedback, onTerminal: onTerminal, onFault: onFault)
        result.analysis = try await RewardAnalysisQueue.prepare(program: program, scope: scope, episodeID: episodeID,
            assetRoot: assetRoot, warmupFrames: warmupFrames, detector: detector,
            onResult: { [weak result] value in
                guard let result else { throw AstraError("desktop.retiredEpisode", "The reward episode was retired.") }
                try result.submit(.result(value), generation: generationID)
            }, onFault: { [weak result] error in result?.fail(error) })
        let timer = DispatchSource.makeTimerSource(queue: result.owner)
        timer.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
        timer.setEventHandler { [weak result] in result?.checkLag() }; result.timer = timer; timer.resume()
        return result
    }
    deinit { timer?.cancel() }

    func confirmReady(_ result: ResetResult) async throws {
        try await perform {
            guard self.ready == nil, !self.began, self.order.isEmpty, result.status == .ready, result.cleanupConfirmed,
                  result.context.nextEpisodeID == self.episodeID, result.context.environmentID == self.identity.environmentID,
                  result.context.scope == self.scope, let observed = result.readyObservation, observed.context == result.context,
                  let signals = result.readySignals, signals.episodeID == self.episodeID,
                  signals.cutoffNanos == observed.observedNanos, result.cleanup.observedNanos <= observed.observedNanos,
                  try RewardEvaluator(program: self.program).readiness(snapshot: signals) == .yes else {
                throw AstraError("desktop.resetProof", "The episode requires the matching successful reset and post-cleanup readiness snapshot.")
            }
            guard observed.sourceCoverage.count == self.scope.surfaces.count,
                  Set(observed.sourceCoverage.map { $0.surface.id }).count == self.scope.surfaces.count,
                  Set(observed.sourceCoverage.map(\.sourceObservationID)).count == observed.sourceCoverage.count else {
                throw AstraError("desktop.resetCoverage", "Readiness must identify every bound source after cleanup.")
            }
            for source in observed.sourceCoverage {
                _ = try source.validated(scope: self.scope, cutoffNanos: observed.observedNanos)
                guard source.throughNanos >= result.cleanup.observedNanos else {
                    throw AstraError("desktop.resetCoverage", "Readiness pixels precede the reset cleanup boundary.")
                }
            }
            try self.analysis!.confirmReady(resetID: result.context.resetID, readyNanos: observed.observedNanos,
                                            controlsReleased: true, pendingPackets: 0)
            self.ready = (result.context.resetID, observed.observedNanos)
        }
    }

    func offer(_ event: InferenceCollectionEvent, generation: UUID) throws {
        try requireGeneration(generation)
        if case .control(let message) = event {
            // Independent final-control headroom survives pairing/transport
            // failures. This immutable bridge can never redirect to a new sink.
            try collector.offer(.controlAudit(message))
        }
        try submit(.event(event), generation: generation)
    }
    func offerManualSeal(_ seal: DesktopManualSeal, generation: UUID) throws {
        try requireGeneration(generation)
        do {
            guard seal.readings.count <= 32, seal.markers.count <= 4096 else {
                throw AstraError("desktop.manualCapacity", "Manual evidence exceeds its bounded schema.")
            }
            // Validate bounded scalar/text payloads before retaining callbacks.
            for reading in seal.readings { _ = try reading.value.validated() }
            try submit(.manual(seal), generation: generation)
        } catch {
            let failure = (error as? AstraError) ?? AstraError("desktop.manualEvidence", error.localizedDescription)
            fail(failure); throw failure
        }
    }
    func requestAuditAbort(reason: String, generation: UUID) throws {
        try requireGeneration(generation); try submit(.abort(reason), generation: generation)
    }
    private func requireGeneration(_ generation: UUID) throws {
        // A stale callback is neither routed to this collector nor allowed to
        // fault a different active episode that happens to share the run ID.
        guard generation == generationID else { throw AstraError("desktop.retiredGeneration", "Evidence belongs to a different episode generation.") }
    }

    private func submit(_ input: Input, generation: UUID) throws {
        let kind: Int, bytes: Int
        switch input {
        case .event(.observation(let value)): kind = 0; bytes = value.pixelByteCount + 2 * AstraVersion.maximumMessageBytes
        case .event(.control): kind = 1; bytes = 0
        case .manual, .result, .abort: kind = 2; bytes = 0
        default: kind = 0; bytes = 2 * AstraVersion.maximumMessageBytes
        }
        do {
            try lock.withLock {
                guard generation == generationID, !allClosed, (!sourceClosed || { if case .result = input { return true }; return false }()),
                      storedFault == nil || kind == 1 else { throw AstraError("desktop.retiredEpisode", "This evidence belongs to a closed or failed episode generation.") }
                if kind == 0 {
                    guard queuedItems < limits.maximumItems, bytes <= limits.maximumBytes - queuedBytes else { throw AstraError("desktop.evidenceBackpressure", "Episode evidence exceeded its retained-observation budget.") }
                    queuedItems += 1; queuedBytes += bytes
                } else if kind == 1 {
                    guard controlItems < 72 else { throw AstraError("desktop.controlBackpressure", "Final control evidence exceeded its reserved capacity.") }; controlItems += 1
                } else {
                    guard completionItems < 256 else { throw AstraError("desktop.completionBackpressure", "Delayed reward evidence exceeded its reserved capacity.") }; completionItems += 1
                }
                let reservation = Reservation(bytes: bytes, kind: kind)
                owner.async { [self] in
                    var retained = false
                    defer { if !retained { release(reservation) } }
                    do { try process(input, reservation: reservation, retained: &retained) }
                    catch { fail((error as? AstraError) ?? AstraError("desktop.evidence", error.localizedDescription)) }
                }
            }
        } catch { let value = (error as? AstraError) ?? AstraError("desktop.evidence", error.localizedDescription); fail(value); throw value }
    }
    private func release(_ item: Reservation) {
        lock.withLock {
            if item.kind == 0 { queuedItems -= 1; queuedBytes -= item.bytes }
            else if item.kind == 1 { controlItems -= 1 }
            else { completionItems -= 1 }
        }
    }
    private func fail(_ error: AstraError) {
        let notify = lock.withLock { if storedFault != nil { return false }; storedFault = error; return true }
        if notify { onFault(DesktopEpisodeFault(generationID: generationID, episodeID: episodeID, error: error)) }
    }
    private func perform<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            owner.async { do { continuation.resume(returning: try body()) } catch { continuation.resume(throwing: error) } }
        }
    }

    private var requiresManualSeal: Bool { program.signals.contains { $0.kind == .manual } || (!deferManualFeedback && program.rules.contains { $0.kind == .manualMarker }) }
    private func entry(_ id: UUID) throws -> Entry {
        if let value = entries[id] { return value }
        guard entries.count < limits.maximumEpisodeDecisions else { throw AstraError("desktop.episodeCapacity", "Episode identity history reached its configured capacity.") }
        return Entry()
    }
    private func process(_ input: Input, reservation: Reservation, retained: inout Bool) throws {
        switch input {
        case .event(.prepared(let run, let actor)):
            guard run == identity.runID, actor.fields?["checkpointID"]?.uuid == policyID,
                  actor.fields?["policySignature"]?.text == policySignature,
                  actor.fields?["collectionVersion"]?.int == 1, actor.fields?["deterministic"] == .bool(false),
                  produced == 0, actorNextSequence == nil else { throw AstraError("desktop.actorPreparation", "The prepared actor differs from this episode's categorical policy.") }
            actorNextSequence = try actor.required("nextPacketSequence").decode(UInt64.self)
        case .event(.observation(let observed)):
            guard ready != nil, observed.runID == identity.runID,
                  try observed.actorInput.required("episodeID").decode(UUID.self) == episodeID,
                  try observed.actorInput.required("geometryRevision").decode(UInt64.self) == scope.geometryRevision,
                  observed.frames.map({ $0.metadata.surface }) == scope.surfaces,
                  Set(observed.frames.map { $0.metadata.id }).count == observed.frames.count,
                  observed.frames.allSatisfy({ $0.pixels.count == $0.metadata.byteCount }) else {
                throw AstraError("desktop.observationIdentity", "Actor observations require the ready episode and its unchanged source.")
            }
            let id = try observed.actorInput.required("observationID").decode(UUID.self)
            let cutoff = try observed.actorInput.required("cutoffNanos").decode(UInt64.self)
            guard cutoff >= ready!.nanos, lastObservationCutoff.map({ cutoff > $0 }) ?? true else {
                throw AstraError("desktop.observationOrder", "Actor observation cutoffs must advance from actual physical readiness.")
            }
            var value = try entry(id)
            guard value.observation == nil, value.cutoff == nil || value.cutoff == cutoff, !value.actorOffered else {
                throw AstraError("desktop.duplicateObservation", "The source observation was repeated or changed its cutoff.")
            }
            value.observation = observed; value.observedReservation = reservation; value.cutoff = cutoff
            value.observationIndex = order.count
            entries[id] = value; order.append(id); lastObservationCutoff = cutoff; retained = true
        case .event(.decision(let response)):
            let record = try response.required("collectionRecord"), packet = try response.required("packet").decode(ActionPacket.self)
            let id = try record.required("observationID").decode(UUID.self)
            guard packet.runID == identity.runID, packet.observationID == id,
                  try record.required("episodeID").decode(UUID.self) == episodeID,
                  try record.required("checkpointID").decode(UUID.self) == policyID,
                  try record.required("policySignature").decode(String.self) == policySignature,
                  record.fields?["schemaVersion"]?.int == 1 else { throw AstraError("desktop.actorIdentity", "The actor result belongs to another episode, observation or policy.") }
            let cutoff = try record.required("cutoffNanos").decode(UInt64.self)
            var value = try entry(id)
            guard value.response == nil, !value.actorOffered, value.cutoff == nil || value.cutoff == cutoff else {
                throw AstraError("desktop.duplicateDecision", "The actor result was repeated or changed its observation cutoff.")
            }
            value.response = response; value.responseReservation = reservation; value.cutoff = cutoff
            entries[id] = value; retained = true
        case .event(.control(let message)):
            guard message.runID == identity.runID, lastControlEventSequence.map({ message.sequence > $0 }) ?? true else {
                throw AstraError("desktop.controlIdentity", "Control evidence changed run or reversed its sender sequence.")
            }
            lastControlEventSequence = message.sequence
            if message.kind == "control.receipt" {
                let raw = try message.payload.required("receipt"), receipt = try raw.decode(ExecutionReceipt.self)
                guard receipt.runID == identity.runID else { throw AstraError("desktop.receiptIdentity", "A native receipt belongs to another actor run.") }
                guard pendingReceipts.count + cancelledReceipts.count < 72 else { throw AstraError("desktop.receiptCapacity", "Final control evidence exceeded its bounded pending capacity.") }
                pendingReceipts.append((raw, receipt))
            } else if message.kind == "control.stopped" {
                let cause = try message.payload.required("cause").decode(String.self)
                guard controlStopCause == nil || controlStopCause == cause else { throw AstraError("desktop.stopCause", "Native control reported conflicting stop causes.") }
                controlStopCause = cause
            } else if message.kind == "control.error" {
                throw AstraError("desktop.controlFailure", "The native control producer reported a transport or execution failure.")
            }
        case .manual(let seal):
            guard seal.episodeID == episodeID, seal.readings.count <= 32, seal.markers.count <= 4096 else {
                throw AstraError("desktop.manualIdentity", "Manual coverage belongs to another episode or exceeds its bounds.")
            }
            // Manual callbacks can finish after a terminal detector or an
            // operator abort. These intervals are explicitly outside learning;
            // no label is fabricated and the original actor/control audit stays.
            if auditOnly { return }
            if let terminal, seal.cutoffNanos >= terminal.cutoffNanos,
               entries[seal.observationID]?.analysisOffered == true { return }
            var value = try entry(seal.observationID)
            guard value.manual == nil, !value.analysisOffered, value.cutoff == nil || value.cutoff == seal.cutoffNanos else {
                throw AstraError("desktop.manualRevision", "Manual producer coverage cannot revise a submitted observation.")
            }
            value.manual = seal; value.manualReservation = reservation; value.cutoff = seal.cutoffNanos
            entries[seal.observationID] = value; retained = true
        case .result(let result):
            if auditOnly { return }
            switch result {
            case .baseline(let id, let snapshot):
                guard baseline == nil, order.first == id, entries[id]?.cutoff == snapshot.cutoffNanos,
                      snapshot.episodeID == episodeID else { throw AstraError("desktop.rewardBaseline", "Reward baseline does not match the first actual actor cutoff.") }
                baseline = snapshot.cutoffNanos
            case .interval(let id, let evaluation):
                guard entries[id]?.cutoff == evaluation.endNanos, evaluation.episodeID == episodeID,
                      evaluation.endNanos > evaluation.startNanos, !seenEvaluationStarts.contains(evaluation.startNanos),
                      let index = entries[id]?.observationIndex, index > 0,
                      entries[order[index - 1]]?.cutoff == evaluation.startNanos, baseline != nil else { throw AstraError("desktop.rewardInterval", "Reward labels do not match the observed episode interval.") }
                seenEvaluationStarts.insert(evaluation.startNanos)
                evaluations[evaluation.startNanos] = evaluation
                if [.succeeded, .failed, .truncated].contains(evaluation.outcome) {
                    guard terminal == nil else { throw AstraError("desktop.duplicateTerminal", "The reward producer repeated the terminal boundary.") }
                    let value = DesktopTerminalEvidence(generationID: generationID, episodeID: episodeID,
                        cutoffNanos: evaluation.endNanos, outcome: evaluation.outcome)
                    terminal = value; onTerminal(value)
                }
            }
        case .abort(let reason): try enterAudit(reason)
        }
        try forwardActors(); try submitAnalysis(); try forwardEvaluations(); try forwardReceipts()
    }

    private func forwardActors() throws {
        while nextActor < order.count {
            let id = order[nextActor]
            guard var value = entries[id], let observation = value.observation, let response = value.response else { return }
            let record = try response.required("collectionRecord"), packet = try response.required("packet").decode(ActionPacket.self)
            let sampler = try record.required("sampler")
            let draw = try sampler.required("drawIndex").decode(UInt64.self)
            guard packet.sequence == draw, packet.sequence < UInt64.max,
                  actorNextSequence.map({ packet.sequence == $0 }) ?? true,
                  try record.required("episodeStep").decode(Int.self) == produced,
                  record.fields?["recurrentReset"] == .bool(produced == 0),
                  sampler.fields?["kind"] == .string("categorical"), sampler.fields?["temperature"]?.double == 1,
                  sampler.fields?["mixture"] == .string("none"), sampler.fields?["version"]?.int == 1,
                  try record.required("frameIDs").decode([UUID].self) == observation.frames.map({ $0.metadata.id }),
                  try record.required("geometryRevision").decode(UInt64.self) == scope.geometryRevision,
                  packet.geometryRevision == scope.geometryRevision else {
                throw AstraError("desktop.actorSequence", "Actor sampling, packet counters or retained frames are inconsistent.")
            }
            let before = try record.required("previousStateID").decode(UUID.self)
            let after = try record.required("nextStateID").decode(UUID.self)
            guard before != after, lastStateID.map({ $0 == before }) ?? true, packets[packet.id] == nil,
                  byCutoff[value.cutoff!] == nil else { throw AstraError("desktop.actorState", "Actor state, packet or observation identity was repeated.") }
            if !began {
                guard let ready else { throw AstraError("desktop.resetProof", "The actor started before a physical reset was confirmed.") }
                try collector.offer(.request("collector.begin", .object([
                    "sourceID": .string(identity.environmentSourceID.uuidString.lowercased()),
                    "episodeID": .string(episodeID.uuidString.lowercased()), "resetID": .string(ready.resetID.uuidString.lowercased()),
                    "readyNanos": .unsigned(ready.nanos), "controlsReleased": .bool(true), "pendingPackets": .integer(0)])))
                began = true
                if let reason = deferredAbortReason {
                    try collector.offer(.request("collector.abort", .object(["reason": .string(reason)])))
                    deferredAbortReason = nil
                }
            }
            try collector.offer(.actor(sourceID: identity.actorSourceID, response: response, observation: observation))
            packets[packet.id] = PacketIdentity(observationID: id, sequence: packet.sequence, cutoff: value.cutoff!)
            byCutoff[value.cutoff!] = packet.id; value.packetID = packet.id
            value.response = nil; value.actorOffered = true
            if let held = value.responseReservation { value.responseReservation = nil; release(held) }
            entries[id] = value; releaseObservationIfTransferred(id)
            produced += 1; nextActor += 1; lastProduced = packet.sequence; actorNextSequence = packet.sequence + 1; lastStateID = after
        }
    }
    private func submitAnalysis() throws {
        while nextAnalysis < order.count {
            let id = order[nextAnalysis]
            guard var value = entries[id], let observed = value.observation else { return }
            if auditOnly || terminal.map({ value.cutoff! >= $0.cutoffNanos }) == true {
                value.analysisOffered = true
            } else {
                if requiresManualSeal, value.manual == nil { return }
                let feedback = value.manual
                guard nextAnalysis != 0 || ((feedback?.markers.isEmpty ?? true) && feedback?.coverage == nil) else {
                    throw AstraError("desktop.preActorFeedback", "The baseline has no policy interval for manual markers or coverage.")
                }
                try analysis!.offer(RewardAnalysisObservation(id: id, episodeID: episodeID, cutoffNanos: value.cutoff!,
                    frames: observed.frames.map { .init(metadata: $0.metadata, pixels: $0.pixels) }, coverage: observed.frames.compactMap(\.coverage),
                    manualReadings: feedback?.readings ?? [], markers: feedback?.markers ?? [], markerCoverage: feedback?.coverage))
                value.analysisOffered = true
            }
            value.manual = nil
            if let held = value.manualReservation { value.manualReservation = nil; release(held) }
            entries[id] = value; releaseObservationIfTransferred(id); nextAnalysis += 1
        }
    }
    private func releaseObservationIfTransferred(_ id: UUID) {
        guard var value = entries[id], value.actorOffered, value.analysisOffered else { return }
        value.observation = nil
        if let held = value.observedReservation { value.observedReservation = nil; release(held) }
        entries[id] = value
    }
    private func forwardEvaluations() throws {
        guard began, !auditOnly else { return }
        for start in evaluations.keys.sorted() {
            guard let packet = byCutoff[start], let evaluation = evaluations[start] else { return }
            try sequence.interval(evaluation, packetID: packet, collector: collector, deferredProgram: deferManualFeedback ? program : nil)
            evaluations[start] = nil
        }
    }
    private func forwardReceipts() throws {
        var waiting: [(JSONValue, ExecutionReceipt)] = []
        for (raw, receipt) in pendingReceipts {
            guard let packet = packets[receipt.packetID] else { waiting.append((raw, receipt)); continue }
            guard packet.sequence == receipt.sequence else { throw AstraError("desktop.receiptSequence", "The receipt does not match the original produced packet sequence.") }
            if receipt.status == .admitted {
                guard admissions.insert(receipt.packetID).inserted else { throw AstraError("desktop.duplicateAdmission", "Packet admission was repeated.") }
            } else {
                guard finalReceipts.insert(receipt.packetID).inserted else { throw AstraError("desktop.duplicateReceipt", "The final packet receipt was repeated.") }
            }
            if receipt.status == .cancelled { cancelledReceipts.append((raw, receipt)) }
            else { try sequence.receipt(raw, packetID: receipt.packetID, episodeID: episodeID, cause: nil, collector: collector) }
        }
        pendingReceipts = waiting
    }
    private func enterAudit(_ reason: String) throws {
        guard !auditOnly else { return }
        guard !reason.isEmpty, reason.utf8.count <= 2048 else { throw AstraError("desktop.abortReason", "The audit stop requires a bounded reason.") }
        if began { try collector.offer(.request("collector.abort", .object(["reason": .string(reason)]))) }
        else { deferredAbortReason = reason }
        auditOnly = true; evaluations.removeAll()
    }
    private func checkLag() {
        guard !lock.withLock({ allClosed || storedFault != nil }) else { return }
        let now = ContinuousClock.now
        let duration = Duration.seconds(limits.maximumPendingSeconds)
        if entries.values.contains(where: { (!$0.actorOffered || !$0.analysisOffered) && $0.arrived.duration(to: now) > duration }) {
            fail(AstraError("desktop.evidenceLag", "Actor pairing or manual coverage exceeded its bounded lag allowance."))
        }
    }

    /// Closes this physical episode only. The parent decides when the shared
    /// collector has enough complete episodes and owns its finish/abandon call.
    func finish(joined: DesktopEpisodeJoin) async throws -> DesktopEpisodeCompletion {
        let start = lock.withLock { () -> Bool in
            if finishing { return false }; finishing = true; sourceClosed = true; return true
        }
        guard start else { throw AstraError("desktop.duplicateFinish", "Episode evidence can be joined only once.") }
        do {
            try await perform {
                guard joined.runID == self.identity.runID, joined.episodeID == self.episodeID,
                      joined.generationID == self.generationID, joined.actorJoined, joined.controlJoined,
                      joined.manualProducerJoined, joined.predictionResolved, joined.cleanupConfirmed else {
                    throw AstraError("desktop.joinProof", "Episode closure requires joined producers, resolved actor work and genuine control cleanup.")
                }
                if case .operatorAbort = joined.stop { try self.enterAudit("The operator stopped episode learning."); try self.submitAnalysis() }
            }
            try await analysis!.finish()
            return try await perform {
                if let fault = self.lock.withLock({ self.storedFault }) { throw fault }
                guard joined.lastProducedSequence == self.lastProduced else {
                    throw AstraError("desktop.producedWatermark", "Joined actor progress differs from the last produced packet, including rejected packets.")
                }
                if self.produced == 0 {
                    guard !self.began, self.entries.values.allSatisfy({ $0.response == nil }),
                          self.pendingReceipts.isEmpty, self.cancelledReceipts.isEmpty else {
                        throw AstraError("desktop.emptyDeclaration", "An unpaired produced result or receipt cannot become an empty episode.")
                    }
                    self.retire(); return .empty
                }
                guard self.nextActor == self.order.count, self.entries.values.allSatisfy({ $0.actorOffered }),
                      self.pendingReceipts.isEmpty, self.finalReceipts.count == self.produced,
                      self.lastObservationCutoff.map({ joined.stoppedNanos >= $0 }) ?? false else {
                    throw AstraError("desktop.incompleteEvidence", "Not every produced actor result, original observation and final receipt was retained before stop.")
                }
                var cancellationCause = "requested"
                switch joined.stop {
                case .semanticBoundary(let cutoff):
                    guard self.terminal?.cutoffNanos == cutoff, cutoff <= joined.stoppedNanos,
                          self.cancelledReceipts.isEmpty || self.controlStopCause == "requested" else {
                        throw AstraError("desktop.terminalStop", "Boundary cancellation lacks the matching semantic outcome and native requested stop.")
                    }
                    cancellationCause = "episodeBoundary"
                case .operatorAbort: break
                case .failure(let reason): throw AstraError("desktop.actorFailure", reason)
                }
                guard self.auditOnly || (self.terminal != nil && self.evaluations.isEmpty && self.baseline != nil) else {
                    throw AstraError("desktop.unsealedReward", "The learning episode has no sealed terminal reward boundary.")
                }
                for (raw, receipt) in self.cancelledReceipts {
                    try self.sequence.receipt(raw, packetID: receipt.packetID, episodeID: self.episodeID,
                        cause: cancellationCause, collector: self.collector)
                }
                try self.collector.offer(.request("collector.end", .object([
                    "sourceID": .string(self.identity.environmentSourceID.uuidString.lowercased()),
                    "episodeID": .string(self.episodeID.uuidString.lowercased()), "stoppedNanos": .unsigned(joined.stoppedNanos),
                    "lastActorSequence": .unsigned(self.lastProduced!), "controlsReleased": .bool(true), "pendingPackets": .integer(0)])))
                let learning = self.auditOnly ? 0 : self.packets.values.filter { $0.cutoff < self.terminal!.cutoffNanos }.count
                let result = DesktopEpisodeCompletion.ended(producedDecisions: self.produced, learningDecisions: learning,
                    lastProducedSequence: self.lastProduced!, terminal: self.terminal, auditOnly: self.auditOnly)
                self.retire(); return result
            }
        } catch {
            // Drain already accepted detector work before retiring its captured
            // generation. The shared collector remains the parent's property.
            _ = try? await analysis!.finish()
            _ = try? await perform { self.retire() }
            throw error
        }
    }
    private func retire() {
        timer?.cancel(); timer = nil
        for value in entries.values {
            for held in [value.observedReservation, value.responseReservation, value.manualReservation].compactMap({ $0 }) { release(held) }
        }
        entries.removeAll(); pendingReceipts.removeAll(); cancelledReceipts.removeAll(); evaluations.removeAll()
        order.removeAll(); packets.removeAll(); byCutoff.removeAll(); admissions.removeAll(); finalReceipts.removeAll()
        seenEvaluationStarts.removeAll()
        lock.withLock { allClosed = true }
    }
}
