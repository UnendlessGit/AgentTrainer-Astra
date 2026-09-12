import Foundation
import Darwin
import AstraCore
import AstraPlatform

struct CollectorRuntime: Sendable {
    let start: @Sendable () async throws -> WireMessage
    let request: @Sendable (String, JSONValue, UUID) async throws -> WireMessage
    let shutdown: @Sendable () async -> Void
    typealias Factory = @Sendable (@escaping ComputeProcess.EventHandler, @escaping ComputeProcess.FailureHandler) -> Self

    static func live(bundle: Bundle) -> Factory {
        let executable = bundle.bundleURL.appendingPathComponent("Contents/Helpers/AstraCompute.app/Contents/MacOS/AstraCompute")
        return { events, failure in
            let process = ComputeProcess(executable: executable, arguments: ["--role", "collector"], expectedRole: "collector",
                allowedEvents: ["collector.framesConsumed", "collector.applied", "collector.fault", "collector.sealed", "collector.audited"],
                onEvent: events, onFailure: failure)
            return Self(start: { try await process.start() }, request: { kind, payload, run in
                try await process.request(kind: kind, payload: payload, runID: run, timeout: .seconds(30))
            }, shutdown: { await process.shutdown() })
        }
    }
}

/// A CPU-only, single-owner bridge. Offers reserve a bounded amount of owned
/// memory synchronously; JSON, file I/O and ring publication run off the UI and
/// actor paths. Transport failure still drains accepted inputs to the native
/// journal before joining the process and retiring its private ring.
final class CollectorSession: @unchecked Sendable {
    enum Input: Sendable {
        case request(String, JSONValue)
        case actor(sourceID: UUID, response: JSONValue, observation: InferenceCollectedObservation)
        case controlAudit(WireMessage)

        var reservedBytes: Int {
            if case .actor(_, _, let observation) = self { return AstraVersion.maximumMessageBytes * 2 + observation.pixels.count }
            return AstraVersion.maximumMessageBytes * 2
        }
        var isControlAudit: Bool { if case .controlAudit = self { return true }; return false }
    }

    let runID: UUID
    let collectionID: UUID
    let journalURL: URL
    private let ring: SharedFrameRing
    private let queue: CollectorQueue
    private let state: CollectorState
    private let runtime: CollectorRuntime
    private var task: Task<Void, Never>?
    private let completion = AsyncCompletion()
    private let finishLock = NSLock()
    private var finishing = false

    private init(runID: UUID, collectionID: UUID, journalURL: URL, ring: SharedFrameRing,
                 queue: CollectorQueue, state: CollectorState, runtime: CollectorRuntime) {
        self.runID = runID; self.collectionID = collectionID; self.journalURL = journalURL
        self.ring = ring; self.queue = queue; self.state = state; self.runtime = runtime
    }

    static func start(runID: UUID, configuration: JSONValue, journalURL: URL, ringURL: URL, slotCapacity: Int,
                      maximumQueuedBytes: Int = 256 * 1024 * 1024, maximumQueuedItems: Int = 128,
                      factory: CollectorRuntime.Factory,
                      onEvent: @escaping @Sendable (WireMessage) -> Void = { _ in },
                      onFault: @escaping @Sendable (AstraError) -> Void) async throws -> CollectorSession {
        guard var fields = configuration.fields, let destination = fields["destination"]?.text,
              let id = UUID(uuidString: URL(fileURLWithPath: destination).lastPathComponent),
              URL(fileURLWithPath: destination).lastPathComponent == id.uuidString.lowercased(),
              journalURL.isFileURL, ringURL.isFileURL,
              (4...FrameArchive.maximumFrameBytes).contains(slotCapacity) else {
            throw AstraError("collector.configuration", "The collector requires a new identified destination and bounded local frame transport.")
        }
        let queue = try CollectorQueue(maximumBytes: maximumQueuedBytes, maximumItems: maximumQueuedItems)
        let ring = try await Task.detached {
            try SharedFrameRing(url: ringURL, runID: runID, slotCount: max(1, min(4, (256 * 1024 * 1024) / slotCapacity)), slotCapacity: slotCapacity)
        }.value
        let state = CollectorState(runID: runID, collectionID: id, ring: ring, onEvent: onEvent, onFault: onFault)
        let runtime = factory({ state.receive($0) }, { state.fail($0) })
        let session = CollectorSession(runID: runID, collectionID: id, journalURL: journalURL, ring: ring,
                                       queue: queue, state: state, runtime: runtime)
        var journal: CollectorJournal?
        do {
            let opened = try await Task.detached { try CollectorJournal(url: journalURL) }.value
            journal = opened
            let hello = try await runtime.start()
            guard hello.kind == "hello", hello.payload.fields?["role"] == .string("collector"),
                  hello.payload.fields?["protocolVersion"] == .integer(1) else {
                throw AstraError("collector.handshake", "The local collector did not negotiate its protocol.")
            }
            fields["rings"] = .array([.object(["path": .string(ring.url.path), "ringID": .string(ring.ringID.uuidString.lowercased())])])
            let prepare = JSONValue.object(fields)
            try await Task.detached { try opened.append(kind: "collector.prepare", payload: prepare, sequence: 0) }.value
            let acknowledgement = try await runtime.request("collector.prepare", prepare, runID)
            guard acknowledgement.kind == "ack", acknowledgement.runID == runID,
                  acknowledgement.payload.fields?["collectionID"]?.uuid == id,
                  acknowledgement.payload.fields?["collectionVersion"] == .integer(1),
                  acknowledgement.payload.fields?["status"] == .string("ready") else {
                throw AstraError("collector.prepare", "The collector prepared a different collection or incompatible transport.")
            }
            if let fault = state.fault { throw fault }
            try Task.checkCancellation()
            session.task = Task.detached { await session.drain(journal: opened) }
            return session
        } catch {
            await runtime.shutdown()
            ring.closeAfterConsumerExit()
            if let journal { await Task.detached { journal.close() }.value }
            throw error
        }
    }

    func offer(_ input: Input) throws {
        do {
            if case .request(let kind, _) = input {
                guard ["collector.begin", "collector.evidence", "collector.bootstrap", "collector.end", "collector.abort"].contains(kind) else {
                    throw AstraError("collector.operation", "This collection operation is not an ordered source input.")
                }
            }
            try queue.offer(input)
        } catch {
            let fault = (error as? AstraError) ?? AstraError("collector.offer", error.localizedDescription)
            state.fail(fault); throw fault
        }
    }

    /// Call only after the actor, reward producer and native control I/O have
    /// joined. This closes ingress once, drains it and awaits immutable output.
    /// It never interprets a collector acknowledgement as physical cleanup.
    func finish() async throws -> WireMessage {
        if finishLock.withLock({ if finishing { return false }; finishing = true; return true }) { queue.close() }
        await completion.wait()
        if let fault = state.fault { throw fault }
        guard let result = state.result else { throw AstraError("collector.noResult", "The collector joined without a completed audit.") }
        return result
    }

    private func drain(journal: CollectorJournal) async {
        var sequence: UInt64 = 1
        while let input = await queue.next() {
            defer { queue.release(input) }
            do {
                let (auditKind, auditPayload) = try auditEntry(input)
                try journal.append(kind: auditKind, payload: auditPayload, sequence: sequence)
                // Even after a transport fault, retain the accepted native
                // evidence. No subsequent wire request may hide a sequence gap.
                if auditKind != "native.control", state.fault == nil {
                    let (kind, payload) = try materialize(input)
                    let response = try await runtime.request(kind, payload, runID)
                    try validateQueued(response)
                    sequence += 1
                }
            } catch { state.fail((error as? AstraError) ?? AstraError("collector.write", error.localizedDescription)) }
        }
        if state.fault == nil {
            do {
                let payload: JSONValue = .object(["throughSequence": .unsigned(sequence - 1)])
                try journal.append(kind: "collector.finish", payload: payload, sequence: sequence)
                state.beginFinish()
                try validateQueued(try await runtime.request("collector.finish", payload, runID))
                try await state.waitForResult(timeout: .seconds(35))
            } catch { state.fail((error as? AstraError) ?? AstraError("collector.finish", error.localizedDescription)) }
        }
        await runtime.shutdown()
        // The ring can have outstanding leases on failure. Joined process
        // ownership permits retirement of the inode, never slot reuse.
        ring.closeAfterConsumerExit()
        journal.close()
        completion.finish()
    }

    private func validateQueued(_ response: WireMessage) throws {
        guard response.kind == "ack", response.runID == runID,
              response.payload.fields?["collectionID"]?.uuid == collectionID,
              response.payload.fields?["status"] == .string("queued") else {
            throw AstraError("collector.ack", "The collector acknowledged a different collection or invalid operation.")
        }
    }

    private func auditEntry(_ input: Input) throws -> (String, JSONValue) {
        switch input {
        case .request(let kind, let payload): return (kind, payload)
        case .controlAudit(let message):
            guard message.runID == runID else { throw AstraError("collector.controlRun", "Native control evidence belongs to another run.") }
            return ("native.control", try .encode(message))
        case .actor(let source, let response, let observation):
            // The control audit survives a failed pixel publication or child
            // transport. It is not a substitute for the collector's pixel spool.
            return ("native.actor", .object(["sourceID": .string(source.uuidString.lowercased()),
                "runID": .string(observation.runID.uuidString.lowercased()), "response": response,
                "actorInput": observation.actorInput, "frame": try .encode(observation.frame),
                "pixelsPersistedByNativeAudit": .bool(false)]))
        }
    }

    private func materialize(_ input: Input) throws -> (String, JSONValue) {
        switch input {
        case .request(let kind, let payload): return (kind, payload)
        case .controlAudit(let message):
            guard message.runID == runID else { throw AstraError("collector.controlRun", "Native control evidence belongs to another run.") }
            return ("native.control", try .encode(message))
        case .actor(let source, let response, let observation):
            guard observation.runID == runID, let fields = observation.actorInput.fields,
                  let id = fields["observationID"]?.uuid, fields["episodeID"]?.uuid != nil,
                  observation.pixels.count == observation.frame.byteCount else {
                throw AstraError("collector.observation", "The retained observation does not belong to this collector.")
            }
            let cutoff = try observation.actorInput.required("cutoffNanos").decode(UInt64.self)
            if let coverage = observation.coverage { try coverage.validated(frame: observation.frame, cutoffNanos: cutoff, maximumAgeMS: 250) }
            else {
                guard observation.frame.eventNanos <= observation.frame.observedNanos, observation.frame.observedNanos <= cutoff,
                      cutoff - observation.frame.eventNanos <= 250_000_000 else { throw AstraError("collector.staleFrame", "The retained collection frame is stale.") }
            }
            let published = try ring.publish(pixels: observation.pixels, metadata: observation.frame)
            try state.register(observation: id, lease: published.acknowledgement)
            let reference = try JSONValue.encode(published)
            let frame: JSONValue = .object(["metadata": try .encode(published.metadata), "reference": reference,
                "coverageNanos": .unsigned(observation.coverage?.throughNanos ?? observation.frame.observedNanos),
                "coverageKind": .string(observation.coverage?.kind.rawValue ?? "frame")])
            let snapshot: JSONValue = .object(["id": .string(id.uuidString.lowercased()),
                "episodeID": try observation.actorInput.required("episodeID"), "cutoffNanos": .unsigned(cutoff),
                "geometryRevision": try observation.actorInput.required("geometryRevision"), "frames": .array([frame]),
                "controlState": try observation.actorInput.required("controlState"),
                "events": try observation.actorInput.required("executedEvents")])
            return ("collector.actor", .object(["sourceID": .string(source.uuidString.lowercased()), "response": response, "observation": snapshot]))
        }
    }
}

/// Reservations include the item currently awaited by the process writer.
private final class CollectorQueue: @unchecked Sendable {
    private let lock = NSLock()
    private let maximumBytes: Int, maximumItems: Int
    private var bytes = 0, items = 0
    // Separate reserved headroom keeps final control receipts admissible after
    // actor-image backpressure. Control currently bounds pending packets at 32.
    private var controlItems = 0
    private let maximumControlItems = 72
    private var values: [CollectorSession.Input] = []
    private var waiter: CheckedContinuation<CollectorSession.Input?, Never>?
    private var closed = false
    init(maximumBytes: Int, maximumItems: Int) throws {
        guard (2 * AstraVersion.maximumMessageBytes...1024 * 1024 * 1024).contains(maximumBytes), (1...128).contains(maximumItems) else {
            throw AstraError("collector.queueConfiguration", "Choose bounded collector queue limits.")
        }
        self.maximumBytes = maximumBytes; self.maximumItems = maximumItems
    }
    func offer(_ input: CollectorSession.Input) throws {
        let target = try lock.withLock { () throws -> CheckedContinuation<CollectorSession.Input?, Never>? in
            guard !closed, input.isControlAudit ? controlItems < maximumControlItems
                : (items < maximumItems && input.reservedBytes <= maximumBytes - bytes) else {
                throw AstraError("collector.backpressure", "The collector could not retain more evidence within its queue limits.")
            }
            if input.isControlAudit { controlItems += 1 }
            else { items += 1; bytes += input.reservedBytes }
            if let waiter { self.waiter = nil; return waiter }
            values.append(input); return nil
        }
        target?.resume(returning: input)
    }
    func next() async -> CollectorSession.Input? {
        await withCheckedContinuation { continuation in
            let value = lock.withLock { () -> (Bool, CollectorSession.Input?) in
                if !values.isEmpty { return (true, values.removeFirst()) }
                if closed { return (true, nil) }
                waiter = continuation; return (false, nil)
            }
            if value.0 { continuation.resume(returning: value.1) }
        }
    }
    func release(_ input: CollectorSession.Input) {
        lock.withLock {
            if input.isControlAudit { controlItems -= 1 }
            else { items -= 1; bytes -= input.reservedBytes }
        }
    }
    func close() {
        let target = lock.withLock { () -> CheckedContinuation<CollectorSession.Input?, Never>? in
            closed = true; let target = waiter; waiter = nil; return target
        }
        target?.resume(returning: nil)
    }
}

private final class CollectorJournal: @unchecked Sendable {
    private let file: FileHandle
    private var ordinal: UInt64 = 0
    init(url: URL) throws {
        let descriptor = Darwin.open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { throw AstraError("collector.journal", "The private native control journal could not be created.") }
        file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    }
    func append(kind: String, payload: JSONValue, sequence: UInt64) throws {
        let value: JSONValue = .object(["ordinal": .unsigned(ordinal), "requestSequence": .unsigned(sequence),
                                        "kind": .string(kind), "payload": payload])
        var bytes = try JSONEncoder().encode(value)
        guard bytes.count < 2 * AstraVersion.maximumMessageBytes else { throw AstraError("collector.journalLimit", "Native collection evidence exceeds the metadata limit.") }
        bytes.append(10); try file.write(contentsOf: bytes); try file.synchronize(); ordinal += 1
    }
    func close() { try? file.close() }
}

private final class CollectorState: @unchecked Sendable {
    private let lock = NSLock()
    private let runID: UUID, collectionID: UUID
    private let ring: SharedFrameRing
    private let onEvent: @Sendable (WireMessage) -> Void
    private let onFault: @Sendable (AstraError) -> Void
    private var storedFault: AstraError?
    private var storedResult: WireMessage?
    private var finishRequested = false
    private var leases: [UUID: Set<SharedFrameAcknowledgement>] = [:]
    private var waiting: CheckedContinuation<Void, any Error>?
    var fault: AstraError? { lock.withLock { storedFault } }
    var result: WireMessage? { lock.withLock { storedResult } }
    init(runID: UUID, collectionID: UUID, ring: SharedFrameRing,
         onEvent: @escaping @Sendable (WireMessage) -> Void, onFault: @escaping @Sendable (AstraError) -> Void) {
        self.runID = runID; self.collectionID = collectionID; self.ring = ring; self.onEvent = onEvent; self.onFault = onFault
    }
    func register(observation: UUID, lease: SharedFrameAcknowledgement) throws {
        try lock.withLock {
            guard leases[observation] == nil else { throw AstraError("collector.duplicateObservation", "This observation was already sent to the collector.") }
            leases[observation] = [lease]
        }
    }
    func beginFinish() { lock.withLock { finishRequested = true } }
    func fail(_ fault: AstraError) {
        let notify = lock.withLock { () -> (Bool, CheckedContinuation<Void, any Error>?) in
            guard storedFault == nil else { return (false, nil) }
            storedFault = fault; let target = waiting; waiting = nil; return (true, target)
        }
        notify.1?.resume(throwing: fault)
        if notify.0 { onFault(fault) }
    }
    func receive(_ event: WireMessage) {
        do {
            guard event.runID == runID, event.payload.fields?["collectionID"]?.uuid == collectionID else {
                throw AstraError("collector.eventIdentity", "The collector emitted evidence for a different run or collection.")
            }
            switch event.kind {
            case "collector.framesConsumed":
                let observation = try event.payload.required("observationID").decode(UUID.self)
                let acknowledgements = try event.payload.required("acknowledgements").decode([SharedFrameAcknowledgement].self)
                try lock.withLock {
                    guard !acknowledgements.isEmpty, Set(acknowledgements).count == acknowledgements.count,
                          let expected = leases[observation], Set(acknowledgements) == expected else {
                        throw AstraError("collector.frameRelease", "The collector released an unknown, partial or repeated frame lease.")
                    }
                    for acknowledgement in acknowledgements { try ring.release(acknowledgement) }
                    leases[observation] = nil
                }
            case "collector.fault":
                fail(AstraError(event.payload.fields?["code"]?.text ?? "collector.rejected",
                                event.payload.fields?["message"]?.text ?? "The collector rejected this experience."))
            case "collector.sealed", "collector.audited":
                let target = try lock.withLock { () throws -> CheckedContinuation<Void, any Error>? in
                    guard finishRequested, storedResult == nil, leases.isEmpty else { throw AstraError("collector.earlyResult", "The collector completed before joined input/frame ownership or repeated its result.") }
                    storedResult = event; let target = waiting; waiting = nil; return target
                }
                target?.resume()
            case "collector.applied": break
            default: throw AstraError("collector.event", "The collector emitted an unknown event.")
            }
            onEvent(event)
        } catch { fail((error as? AstraError) ?? AstraError("collector.event", error.localizedDescription)) }
    }
    func waitForResult(timeout: Duration) async throws {
        let timer = Task { [weak self] in
            do { try await Task.sleep(for: timeout) } catch { return }
            self?.fail(AstraError("collector.finalizeTimeout", "The collector did not complete its joined audit in time."))
        }
        defer { timer.cancel() }
        try await withCheckedThrowingContinuation { continuation in
            let status = lock.withLock { () -> (Bool, AstraError?) in
                if let storedFault { return (true, storedFault) }
                if storedResult != nil { return (true, nil) }
                waiting = continuation; return (false, nil)
            }
            if status.0 { if let error = status.1 { continuation.resume(throwing: error) } else { continuation.resume() } }
        }
    }
}
