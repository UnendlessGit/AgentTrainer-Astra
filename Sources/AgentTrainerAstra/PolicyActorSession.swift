import Foundation
import AstraCore
import AstraPlatform

struct PolicyActorCheckpoint: Sendable {
    let document: CheckpointDocument
    let directory: URL
}

enum PolicyActorPreparation: Sendable {
    case fresh(seed: UInt64)
    /// The worker authenticates progress embedded in the immutable checkpoint.
    /// This optional comparison is a caller expectation, never an RNG input.
    case resume(expectedProgress: JSONValue? = nil)
    case resumeCollection(sourcePath: URL, manifestSHA256: String, expectedProgress: JSONValue)
}

struct PolicyActorSnapshot: Sendable {
    let observationID: UUID
    let frames: [InferenceImage]
    let controls: ControlObservation
    let geometryRevision: UInt64

    init(observationID: UUID = UUID(), frame: InferenceImage, controls: ControlObservation) {
        self.init(observationID: observationID, frames: [frame], geometryRevision: frame.metadata.surface.geometryRevision, controls: controls)
    }
    init(observationID: UUID = UUID(), frames: [InferenceImage], geometryRevision: UInt64, controls: ControlObservation) {
        self.observationID = observationID; self.frames = frames; self.geometryRevision = geometryRevision; self.controls = controls
    }
}

struct PolicyActorOwnedObservation: Sendable {
    struct Frame: Sendable {
        let metadata: FrameMetadata
        let pixels: Data
        let coverage: CaptureFrameCoverage?
    }
    let runID: UUID
    let actorInput: JSONValue
    let frames: [Frame]

    func collected() -> InferenceCollectedObservation {
        .init(runID: runID, actorInput: actorInput, frames: frames)
    }

    /// Legacy single-source adapters must explicitly reject a source group.
    func singleSource() throws -> InferenceCollectedObservation {
        guard frames.count == 1, let frame = frames.first else {
            throw AstraError("inference.collectorSurfaces", "This collector requires exactly one observed surface.")
        }
        return .init(runID: runID, actorInput: actorInput, frame: frame.metadata, pixels: frame.pixels, coverage: frame.coverage)
    }
}

struct PolicyActorResult: Sendable {
    let response: WireMessage
    let packet: ActionPacket
    let observation: PolicyActorOwnedObservation
    /// Last-real-result watermark. Warmup and administrative acknowledgements
    /// never manufacture collection progress.
    let actorProgress: JSONValue?
}

struct PolicyActorPrediction: Sendable {
    fileprivate let task: Task<PolicyActorResult, any Error>
    /// Awaiting cancellation does not cancel the sampled operation. The caller
    /// must retain/drain this ticket and reconcile its original packet.
    func value() async throws -> PolicyActorResult { try await task.value }
}

struct PolicyActorState: Codable, Sendable, Equatable {
    let nextPacketSequence: UInt64
    let nextDrawIndex: UInt64
    let actorResetGeneration: UInt64
    let episodeStep: UInt64
    let episodeID: UUID?
    let stateID: UUID?
    let actorProgress: JSONValue?
    let sampledProgressKnown: Bool
    let stopped: Bool
    let joined: Bool
}

/// One atomic read of the identity and state admitted by this actor owner.
struct PolicyActorBinding: Sendable {
    let runID: UUID
    let checkpoint: CheckpointDocument
    let policy: InferencePolicyDetails
    let collecting: Bool
    let state: PolicyActorState
    /// This read is not a reservation. Parent episode/learner ownership must
    /// still prevent a new pause from racing after control startup admission.
    let isAvailable: Bool
    /// Physical readiness may be established before the first recurrent reset.
    /// It still requires an idle, unpaused actor with known sampling progress.
    let canReset: Bool
}

/// An exclusive idle-actor hold. The owner may carry this across collector
/// finalization and learner publication, but only this session can release it.
/// No-draw pauses are allowed; they never invent actorProgress. A learning batch
/// must independently require the sealed real-result progress it intends to use.
struct PolicyActorLearningPause: Sendable {
    let binding: PolicyActorBinding
    fileprivate let sessionID: UUID
    fileprivate let pauseID: UUID
    fileprivate init(binding: PolicyActorBinding, sessionID: UUID, pauseID: UUID) {
        self.binding = binding; self.sessionID = sessionID; self.pauseID = pauseID
    }
}

/// Persistent compute/lease ownership, independent of control helpers, reset
/// actions, capture production, reward analysis and learner scheduling.
///
/// Each request runs in an owned unstructured task. Stop closes admission while
/// preserving the in-flight result; force interruption explicitly forfeits any
/// unresolved sampled progress. Frame transport retires only after process exit.
actor PolicyActorSession {
    let runID: UUID
    private let sessionID = UUID()
    private let ringURL: URL
    private let slotCapacities: [Int]
    private let runtime: InferenceRuntime
    private let clock: @Sendable () -> UInt64
    private var ring: ObservationFrameRings?
    private var checkpoint: PolicyActorCheckpoint?
    private(set) var policy: InferencePolicyDetails?
    private var collecting = false
    private var rngStreamID: UUID?
    private var expectedRNG: [UInt32]?
    private var nextSequence: UInt64 = 0
    private var nextDraw: UInt64 = 0
    private var resetGeneration: UInt64 = 0
    private var episodeStep: UInt64 = 0
    private var episodeID: UUID?
    private var stateID: UUID?
    private var contexts: [Int] = []
    private var lastCutoff: UInt64?
    private var lastEventSequence: UInt64?
    private var episodeSurfaces: [SurfaceDescriptor]?
    private var episodeGeometryRevision: UInt64?
    private var lastProgress: JSONValue?
    private var progressKnown = true
    private var stopped = false
    private var joined = false
    private var activeID: UUID?
    private var active: Task<Void, Never>?
    private var prediction: PolicyActorPrediction?
    private var shutdownTask: Task<Void, Never>?
    private var processExit: Task<Void, Never>?
    private var forceInterrupted = false
    private var learningPauseID: UUID?

    init(runID: UUID, ringURL: URL, slotCapacity: Int, slotCapacities: [Int]? = nil, runtime: InferenceRuntime,
         clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now }) {
        self.runID = runID; self.ringURL = ringURL; self.slotCapacities = slotCapacities ?? [slotCapacity]
        self.runtime = runtime; self.clock = clock
    }

    var state: PolicyActorState {
        .init(nextPacketSequence: nextSequence, nextDrawIndex: nextDraw, actorResetGeneration: resetGeneration,
              episodeStep: episodeStep, episodeID: episodeID, stateID: stateID, actorProgress: lastProgress,
              sampledProgressKnown: progressKnown, stopped: stopped, joined: joined)
    }

    var binding: PolicyActorBinding? {
        guard let checkpoint, let policy else { return nil }
        return .init(runID: runID, checkpoint: checkpoint.document, policy: policy, collecting: collecting, state: state,
                     isAvailable: activeID == nil && learningPauseID == nil && progressKnown && episodeID != nil && stateID != nil && !stopped && !joined,
                     canReset: activeID == nil && learningPauseID == nil && progressKnown && !stopped && !joined)
    }

    func acquireLearningPause() throws -> PolicyActorLearningPause {
        try requireIdle()
        guard let binding, binding.state.episodeID != nil, binding.state.stateID != nil,
              binding.state.sampledProgressKnown else {
            throw AstraError("inference.learningBoundary", "A learning pause requires a confirmed, idle actor with known sampled progress.")
        }
        let id = UUID(); learningPauseID = id
        return .init(binding: binding, sessionID: sessionID, pauseID: id)
    }

    func validateLearningPause(_ token: PolicyActorLearningPause) throws {
        guard token.sessionID == sessionID, learningPauseID == token.pauseID, !stopped, !joined,
              activeID == nil, progressKnown, let binding, binding.runID == token.binding.runID,
              binding.checkpoint.matchesIdentity(of: token.binding.checkpoint), binding.collecting == token.binding.collecting,
              binding.state == token.binding.state else {
            throw AstraError("inference.learningPause", "The exclusive actor learning boundary was released, stopped or belongs to another session.")
        }
    }

    func releaseLearningPause(_ token: PolicyActorLearningPause) throws {
        try validateLearningPause(token)
        learningPauseID = nil
    }

    func prepare(checkpoint: PolicyActorCheckpoint, collection: Bool, deterministic: Bool = false,
                 mode: PolicyActorPreparation = .fresh(seed: 0)) async throws -> WireMessage {
        try await operation { try await self.prepareWork(checkpoint, collection: collection, deterministic: deterministic, mode: mode) }
    }

    /// Call only after actual environment readiness and prior control cleanup.
    /// A new ID is required even if the same checkpoint remains active.
    func reset(confirmedEpisodeID: UUID, contextIDs: [Int], activating replacement: PolicyActorCheckpoint? = nil,
               timeout: Duration = .seconds(120)) async throws -> WireMessage {
        try await operation { try await self.resetWork(confirmedEpisodeID, contexts: contextIDs, replacement: replacement, timeout: timeout) }
    }

    func warmup(_ snapshot: PolicyActorSnapshot) async throws -> PolicyActorResult {
        try await operation { try await self.predictWork(snapshot, warmup: true) }
    }

    func beginPrediction(_ snapshot: PolicyActorSnapshot,
                         onObservation: (@Sendable (PolicyActorOwnedObservation) throws -> Void)? = nil) throws -> PolicyActorPrediction {
        try requireIdle()
        let id = UUID(); activeID = id
        let task = Task { () throws -> PolicyActorResult in
            do {
                let result = try await self.predictWork(snapshot, warmup: false, onObservation: onObservation)
                self.complete(id); return result
            } catch { self.stopped = true; self.complete(id); throw error }
        }
        let ticket = PolicyActorPrediction(task: task)
        prediction = ticket; active = Task { _ = try? await task.value }
        return ticket
    }

    func drainPrediction() async throws -> PolicyActorResult? { try await prediction?.value() }
    func requestStop() { stopped = true; learningPauseID = nil }

    /// Normal shutdown drains the exact result before joining. Interrupt only
    /// when the caller must abandon an unresponsive actor; unknown progress is
    /// then an explicit non-resumable outcome, never guessed from old counters.
    func shutdown(interruptPending: Bool = false) async {
        stopped = true; learningPauseID = nil
        if interruptPending, !joined {
            forceInterrupted = true
            _ = startRuntimeShutdown()
        }
        if let shutdownTask { await shutdownTask.value; return }
        let pending = active
        let task = Task {
            await pending?.value
            await self.startRuntimeShutdown().value
            self.ring?.closeAfterConsumerExit(); self.ring = nil; self.joined = true
        }
        shutdownTask = task
        await task.value
    }

    private func startRuntimeShutdown() -> Task<Void, Never> {
        if let processExit { return processExit }
        let task = Task { _ = await self.runtime.shutdown() }
        processExit = task
        return task
    }

    private func requireIdle() throws {
        guard !stopped, !joined else { throw AstraError("inference.closed", "This actor session no longer accepts work.") }
        guard activeID == nil else { throw AstraError("inference.busy", "Drain the current actor operation before starting another.") }
        guard learningPauseID == nil else { throw AstraError("inference.learningPaused", "Release the exclusive learning pause before using the actor.") }
    }

    private func operation<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) async throws -> T {
        try requireIdle()
        let id = UUID(); activeID = id
        let task = Task { () throws -> T in
            do { let result = try await body(); self.complete(id); return result }
            catch { self.stopped = true; self.complete(id); throw error }
        }
        active = Task { _ = try? await task.value }
        return try await task.value
    }

    private func complete(_ id: UUID) { if activeID == id { activeID = nil } }

    private func prepareWork(_ selected: PolicyActorCheckpoint, collection: Bool, deterministic: Bool,
                             mode: PolicyActorPreparation) async throws -> WireMessage {
        guard ring == nil, checkpoint == nil, !collection || !deterministic else {
            throw AstraError("inference.preparation", "Prepare a new actor once; collection requires categorical sampling.")
        }
        _ = try selected.document.validated()
        guard selected.directory.isFileURL, ringURL.isFileURL else { throw AstraError("inference.path", "Actor artifacts require local paths.") }
        let ring = try await Task.detached { [ringURL, runID, slotCapacities] in
            try ObservationFrameRings(url: ringURL, runID: runID, capacities: slotCapacities)
        }.value
        self.ring = ring
        if forceInterrupted { throw CancellationError() }
        try PolicyActorValidation.hello(try await runtime.start())
        if forceInterrupted { throw CancellationError() }
        var fields: [String: JSONValue] = ["checkpointPath": .string(selected.directory.path),
            "ring": ring.descriptors[0],
            "collection": .bool(collection), "deterministic": .bool(deterministic)]
        if ring.rings.count > 1 { fields["additionalRings"] = .array(Array(ring.descriptors.dropFirst())) }
        var expected: JSONValue?
        let resuming: Bool
        switch mode {
        case .fresh(let seed): fields["seed"] = .unsigned(seed); resuming = false
        case .resume(let progress):
            guard collection else { throw AstraError("inference.resumeMode", "Resuming an actor requires collection mode.") }
            if let progress { try PolicyActorValidation.progress(progress); expected = progress }
            fields["resumeActor"] = .bool(true); resuming = true
        case .resumeCollection(let source, let digest, let progress):
            guard collection, source.isFileURL, digest.count == 64,
                  digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw AstraError("inference.resumeSource", "Collection resume needs an authenticated local source and categorical sampling.")
            }
            try PolicyActorValidation.progress(progress); expected = progress
            fields["resumeCollection"] = .object(["sourcePath": .string(source.path), "manifestSHA256": .string(digest)])
            resuming = true
        }
        let reply = try await runtime.request("inference.prepare", .object(fields), runID, .seconds(120), false)
        try PolicyActorValidation.success(reply, runID: runID)
        if case .resumeCollection = mode, reply.payload.fields?["resumedCollection"] != .bool(true) {
            throw AstraError("inference.resumeSource", "The actor did not authenticate its suspended collection cursor.")
        }
        let details = try InferencePolicyDetails(reply.payload, checkpoint: selected.document, runID: runID, ringID: ring.primary.ringID)
        if ring.rings.count > 1 {
            guard try reply.payload.required("ringIDs").decode([UUID].self) == ring.rings.map(\.ringID) else {
                throw AstraError("inference.frameTransport", "The actor did not open the complete ordered source transport.")
            }
        }
        let counters = try PolicyActorValidation.counters(reply.payload)
        guard reply.payload.fields?["needsReset"] == .bool(true), reply.payload.fields?["resumedActor"] == .bool(resuming),
              reply.payload.fields?["collection"] == .bool(collection), reply.payload.fields?["deterministic"] == .bool(deterministic),
              resuming || (counters.next == 0 && counters.generation == 0) else {
            throw AstraError("inference.preparation", "The actor prepared an inconsistent stream or reset state.")
        }
        if collection {
            guard reply.payload.fields?["collectionVersion"] == .integer(1), let stream = reply.payload.fields?["rngStreamID"]?.uuid else {
                throw AstraError("inference.collectionVersion", "The actor cannot supply categorical collection evidence.")
            }
            rngStreamID = stream
        }
        if let expected {
            guard expected.fields?["rngStreamID"]?.uuid == rngStreamID,
                  let draw = expected.fields?["drawIndex"]?.uint64, draw < UInt64.max - 1,
                  counters.next == draw + 1, counters.generation == expected.fields?["actorResetGeneration"]?.uint64 else {
                throw AstraError("inference.resumeIdentity", "The checkpoint's authenticated actor stream differs from the expected boundary.")
            }
            expectedRNG = try expected.required("rngState").decode([UInt32].self)
        }
        checkpoint = selected; policy = details; collecting = collection
        nextSequence = counters.next; nextDraw = counters.next; resetGeneration = counters.generation
        return reply
    }

    private func resetWork(_ episode: UUID, contexts: [Int], replacement: PolicyActorCheckpoint?, timeout: Duration) async throws -> WireMessage {
        guard let checkpoint, let policy, let oldRing = ring, episode != episodeID,
              contexts.count == policy.contextSizes.count,
              zip(contexts, policy.contextSizes).allSatisfy({ $0 >= 0 && $0 < $1 }), resetGeneration < UInt64.max else {
            throw AstraError("inference.reset", "A prepared actor requires a new episode and valid checkpoint contexts.")
        }
        let selected = replacement ?? checkpoint
        _ = try selected.document.validated()
        guard selected.document.policySignature == checkpoint.document.policySignature, selected.directory.isFileURL else {
            throw AstraError("inference.policyMismatch", "A different policy or action signature requires a new actor session.")
        }
        var fields: [String: JSONValue] = ["confirmed": .bool(true), "episodeID": .string(episode.uuidString.lowercased()),
            "contextIDs": .array(contexts.map { .integer(Int64($0)) })]
        if replacement != nil { fields["checkpointPath"] = .string(selected.directory.path) }
        let reply = try await runtime.request("inference.reset", .object(fields), runID, timeout, false)
        try PolicyActorValidation.success(reply, runID: runID)
        let counters = try PolicyActorValidation.counters(reply.payload)
        guard reply.payload.fields?["runID"]?.uuid == runID, reply.payload.fields?["episodeID"]?.uuid == episode,
              reply.payload.fields?["checkpointID"]?.uuid == selected.document.id,
              reply.payload.fields?["policySignature"]?.text == checkpoint.document.policySignature,
              reply.payload.fields?["needsReset"] == .bool(false), let state = reply.payload.fields?["stateID"]?.uuid,
              state != stateID, counters.next == nextSequence, nextSequence == nextDraw,
              counters.generation == resetGeneration + 1, self.ring === oldRing else {
            throw AstraError("inference.reset", "The actor did not confirm the expected checkpoint, recurrent reset and persistent counters.")
        }
        self.checkpoint = selected; episodeID = episode; stateID = state; self.contexts = contexts
        resetGeneration = counters.generation; episodeStep = 0; lastEventSequence = nil; episodeSurfaces = nil; episodeGeometryRevision = nil
        prediction = nil
        return reply
    }

    private func predictWork(_ snapshot: PolicyActorSnapshot, warmup: Bool,
                             onObservation: (@Sendable (PolicyActorOwnedObservation) throws -> Void)? = nil) async throws -> PolicyActorResult {
        guard nextSequence < UInt64.max, nextSequence == nextDraw else {
            throw AstraError("inference.counterExhausted", "The actor has no packet and random-draw index available for admission.")
        }
        guard let checkpoint, let policy, let ring, let episodeID, let stateID,
              !warmup || episodeStep == 0 else {
            throw AstraError("inference.resetRequired", "Reset the prepared actor before predicting or warming its policy.")
        }
        let surfaces = try PolicyActorValidation.snapshot(snapshot, now: clock(), previousCutoff: lastCutoff,
            periodMS: policy.periodMS, episodeStep: episodeStep, previousEvent: lastEventSequence, collecting: collecting)
        guard (episodeSurfaces == nil || episodeSurfaces == surfaces),
              episodeGeometryRevision == nil || episodeGeometryRevision == snapshot.geometryRevision else {
            throw AstraError("inference.geometryChanged", "Surface geometry may change only at a confirmed environment reset.")
        }
        let published = try await Task.detached {
            let frames = try snapshot.frames.map { image in
                PolicyActorOwnedObservation.Frame(metadata: image.metadata, pixels: try image.pixels(), coverage: image.coverage)
            }
            let references = try ring.publish(frames)
            let owned = zip(references, frames).map { reference, frame in
                PolicyActorOwnedObservation.Frame(metadata: reference.metadata, pixels: frame.pixels, coverage: frame.coverage)
            }
            return (references, owned)
        }.value
        let references = published.0
        var input: [String: JSONValue] = ["observationID": .string(snapshot.observationID.uuidString.lowercased()),
            "episodeID": .string(episodeID.uuidString.lowercased()), "previousStateID": .string(stateID.uuidString.lowercased()),
            "cutoffNanos": .unsigned(snapshot.controls.cutoffNanos), "geometryRevision": .unsigned(snapshot.geometryRevision),
            "controlState": try .encode(snapshot.controls.controlState), "executedEvents": try .encode(snapshot.controls.executedEvents),
            "intervalCovered": .bool(true), "contextIDs": try .encode(contexts)]
        if let coverage = snapshot.controls.controlCoverageNanos { input["controlCoverageNanos"] = .unsigned(coverage) }
        let owned = PolicyActorOwnedObservation(runID: runID, actorInput: .object(input), frames: published.1)
        input["frames"] = try .encode(references)
        if forceInterrupted { throw CancellationError() }
        do { try onObservation?(owned) }
        catch {
            // These leases have never been sent to the worker. Returning their
            // writer-owned slots is not an acknowledgement of consumed pixels.
            for reference in references { try ring.release(reference.acknowledgement) }
            throw error
        }
        if !warmup { progressKnown = false }
        let reply = try await runtime.request(warmup ? "inference.warmup" : "inference.step", .object(input), runID, .seconds(120), true)
        guard reply.runID == runID else { throw AstraError("inference.run", "The actor replied for another run.") }
        let released = try reply.payload.required("releasedFrames").decode([SharedFrameAcknowledgement].self)
        let expectedLeases = references.map(\.acknowledgement)
        guard released == expectedLeases || (reply.kind == "error" && released == Array(expectedLeases.prefix(released.count))) else {
            throw AstraError("inference.frameAcknowledgement", "The actor acknowledged a foreign, missing or repeated frame lease.")
        }
        for acknowledgement in released { try ring.release(acknowledgement) }
        try PolicyActorValidation.success(reply, runID: runID)
        let packet = try PolicyActorValidation.result(reply.payload, checkpoint: checkpoint.document, runID: runID,
            episodeID: episodeID, previousState: stateID, observationID: snapshot.observationID,
            cutoff: snapshot.controls.cutoffNanos, sequence: nextSequence, surfaces: surfaces, geometryRevision: snapshot.geometryRevision, policy: policy)
        var progress: JSONValue?
        if warmup {
            guard reply.payload.fields?["warmup"] == .bool(true), reply.payload.fields?["collectionRecord"] == nil else {
                throw AstraError("inference.warmup", "Warmup did not isolate collection state.")
            }
        } else {
            if collecting {
                progress = try PolicyActorValidation.collection(reply.payload, packet: packet, checkpoint: checkpoint.document,
                    observation: owned, episodeStep: episodeStep, resetGeneration: resetGeneration,
                    rngStreamID: rngStreamID, expectedRNG: expectedRNG)
                expectedRNG = try progress?.required("rngState").decode([UInt32].self)
            }
            self.stateID = try reply.payload.requiredUUID("stateID")
            nextSequence += 1; nextDraw += 1; self.episodeStep += 1
            lastCutoff = snapshot.controls.cutoffNanos; lastEventSequence = snapshot.controls.lastSequence
            episodeSurfaces = surfaces; episodeGeometryRevision = snapshot.geometryRevision; lastProgress = progress; progressKnown = true
        }
        return .init(response: reply, packet: packet, observation: owned, actorProgress: progress)
    }
}
