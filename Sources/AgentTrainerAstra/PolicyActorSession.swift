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
}

struct PolicyActorSnapshot: Sendable {
    let observationID: UUID
    let frames: [InferenceImage]
    let controls: ControlObservation

    init(observationID: UUID = UUID(), frame: InferenceImage, controls: ControlObservation) {
        self.observationID = observationID; frames = [frame]; self.controls = controls
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

    /// The current native collector is single-source. Keep the session's owned
    /// frames explicit so a later multi-surface collector cannot silently drop one.
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

struct PolicyActorState: Sendable {
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

/// Persistent compute/lease ownership, independent of control helpers, reset
/// actions, capture production, reward analysis and learner scheduling.
///
/// Each request runs in an owned unstructured task. Stop closes admission while
/// preserving the in-flight result; force interruption explicitly forfeits any
/// unresolved sampled progress. Frame transport retires only after process exit.
actor PolicyActorSession {
    let runID: UUID
    private let ringURL: URL
    private let slotCapacity: Int
    private let runtime: InferenceRuntime
    private let clock: @Sendable () -> UInt64
    private var ring: SharedFrameRing?
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

    init(runID: UUID, ringURL: URL, slotCapacity: Int, runtime: InferenceRuntime,
         clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now }) {
        self.runID = runID; self.ringURL = ringURL; self.slotCapacity = slotCapacity
        self.runtime = runtime; self.clock = clock
    }

    var state: PolicyActorState {
        .init(nextPacketSequence: nextSequence, nextDrawIndex: nextDraw, actorResetGeneration: resetGeneration,
              episodeStep: episodeStep, episodeID: episodeID, stateID: stateID, actorProgress: lastProgress,
              sampledProgressKnown: progressKnown, stopped: stopped, joined: joined)
    }

    func prepare(checkpoint: PolicyActorCheckpoint, collection: Bool, deterministic: Bool = false,
                 mode: PolicyActorPreparation = .fresh(seed: 0)) async throws -> WireMessage {
        try await operation { try await self.prepareWork(checkpoint, collection: collection, deterministic: deterministic, mode: mode) }
    }

    /// Call only after actual environment readiness and prior control cleanup.
    /// A new ID is required even if the same checkpoint remains active.
    func reset(confirmedEpisodeID: UUID, contextIDs: [Int], activating replacement: PolicyActorCheckpoint? = nil) async throws -> WireMessage {
        try await operation { try await self.resetWork(confirmedEpisodeID, contexts: contextIDs, replacement: replacement) }
    }

    func warmup(_ snapshot: PolicyActorSnapshot) async throws -> PolicyActorResult {
        try await operation { try await self.predictWork(snapshot, warmup: true) }
    }

    func beginPrediction(_ snapshot: PolicyActorSnapshot) throws -> PolicyActorPrediction {
        try requireIdle()
        let id = UUID(); activeID = id
        let task = Task { () throws -> PolicyActorResult in
            do {
                let result = try await self.predictWork(snapshot, warmup: false)
                self.complete(id); return result
            } catch { self.stopped = true; self.complete(id); throw error }
        }
        let ticket = PolicyActorPrediction(task: task)
        prediction = ticket; active = Task { _ = try? await task.value }
        return ticket
    }

    func drainPrediction() async throws -> PolicyActorResult? { try await prediction?.value() }
    func requestStop() { stopped = true }

    /// Normal shutdown drains the exact result before joining. Interrupt only
    /// when the caller must abandon an unresponsive actor; unknown progress is
    /// then an explicit non-resumable outcome, never guessed from old counters.
    func shutdown(interruptPending: Bool = false) async {
        stopped = true
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
        let ring = try await Task.detached { [ringURL, runID, slotCapacity] in
            try SharedFrameRing(url: ringURL, runID: runID, slotCount: 2, slotCapacity: slotCapacity)
        }.value
        self.ring = ring
        if forceInterrupted { throw CancellationError() }
        try PolicyActorValidation.hello(try await runtime.start())
        if forceInterrupted { throw CancellationError() }
        var fields: [String: JSONValue] = ["checkpointPath": .string(selected.directory.path),
            "ring": .object(["path": .string(ring.url.path), "ringID": .string(ring.ringID.uuidString.lowercased())]),
            "collection": .bool(collection), "deterministic": .bool(deterministic)]
        var expected: JSONValue?
        let resuming: Bool
        switch mode {
        case .fresh(let seed): fields["seed"] = .unsigned(seed); resuming = false
        case .resume(let progress):
            guard collection else { throw AstraError("inference.resumeMode", "Resuming an actor requires collection mode.") }
            if let progress { try PolicyActorValidation.progress(progress); expected = progress }
            fields["resumeActor"] = .bool(true); resuming = true
        }
        let reply = try await runtime.request("inference.prepare", .object(fields), runID, .seconds(120), false)
        try PolicyActorValidation.success(reply, runID: runID)
        let details = try InferencePolicyDetails(reply.payload, checkpoint: selected.document, runID: runID, ringID: ring.ringID)
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

    private func resetWork(_ episode: UUID, contexts: [Int], replacement: PolicyActorCheckpoint?) async throws -> WireMessage {
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
        let reply = try await runtime.request("inference.reset", .object(fields), runID, .seconds(120), false)
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
        resetGeneration = counters.generation; episodeStep = 0; lastEventSequence = nil; episodeSurfaces = nil
        prediction = nil
        return reply
    }

    private func predictWork(_ snapshot: PolicyActorSnapshot, warmup: Bool) async throws -> PolicyActorResult {
        guard nextSequence < UInt64.max, nextSequence == nextDraw else {
            throw AstraError("inference.counterExhausted", "The actor has no packet and random-draw index available for admission.")
        }
        guard let checkpoint, let policy, let ring, let episodeID, let stateID,
              !warmup || episodeStep == 0 else {
            throw AstraError("inference.resetRequired", "Reset the prepared actor before predicting or warming its policy.")
        }
        let surfaces = try PolicyActorValidation.snapshot(snapshot, now: clock(), previousCutoff: lastCutoff,
            periodMS: policy.periodMS, episodeStep: episodeStep, previousEvent: lastEventSequence, collecting: collecting)
        guard episodeSurfaces == nil || episodeSurfaces == surfaces else {
            throw AstraError("inference.geometryChanged", "Surface geometry may change only at a confirmed environment reset.")
        }
        let published = try await Task.detached {
            try snapshot.frames.map { image -> (SharedFrameReference, PolicyActorOwnedObservation.Frame) in
                let pixels = try image.pixels()
                let reference = try ring.publish(pixels: pixels, metadata: image.metadata)
                return (reference, .init(metadata: reference.metadata, pixels: pixels, coverage: image.coverage))
            }
        }.value
        let references = published.map(\.0)
        var input: [String: JSONValue] = ["observationID": .string(snapshot.observationID.uuidString.lowercased()),
            "episodeID": .string(episodeID.uuidString.lowercased()), "previousStateID": .string(stateID.uuidString.lowercased()),
            "cutoffNanos": .unsigned(snapshot.controls.cutoffNanos), "geometryRevision": .unsigned(surfaces[0].geometryRevision),
            "controlState": try .encode(snapshot.controls.controlState), "executedEvents": try .encode(snapshot.controls.executedEvents),
            "intervalCovered": .bool(true), "contextIDs": try .encode(contexts)]
        let owned = PolicyActorOwnedObservation(runID: runID, actorInput: .object(input), frames: published.map(\.1))
        input["frames"] = try .encode(references)
        if forceInterrupted { throw CancellationError() }
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
            cutoff: snapshot.controls.cutoffNanos, sequence: nextSequence, surfaces: surfaces, policy: policy)
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
            episodeSurfaces = surfaces; lastProgress = progress; progressKnown = true
        }
        return .init(response: reply, packet: packet, observation: owned, actorProgress: progress)
    }
}
