import Foundation
import Darwin
import AstraCore

private final class ChannelStop: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    var isStopped: Bool { lock.withLock { stopped } }
    func stop() { lock.withLock { stopped = true } }
}

/// One inherited-pipe connection and one child lifetime. Protocol state belongs
/// to eventQueue; bounded nonblocking I/O has separate threads/queues. Shutdown
/// joins the child and every pipe owner before callers may retire shared memory.
public final class ComputeProcess: @unchecked Sendable {
    public typealias EventHandler = @Sendable (WireMessage) -> Void
    public typealias FailureHandler = @Sendable (AstraError) -> Void
    private struct Pending {
        var runID: UUID?
        var acceptingError: Bool
        var continuation: CheckedContinuation<WireMessage, any Error>?
    }
    private let eventQueue = DispatchQueue(label: "astra.compute.protocol", qos: .userInitiated)
    private let writeQueue = DispatchQueue(label: "astra.compute.write", qos: .userInitiated)
    private let executable: URL
    private let arguments: [String]
    private let environment: [String: String]?
    private let expectedRole: String
    private let allowedEvents: Set<String>
    private let onEvent: EventHandler
    private let onFailure: FailureHandler
    private let readStop = ChannelStop(), writeStop = ChannelStop()
    // eventQueue owns all following mutable state.
    private var process: Process?
    private var input: FileHandle?
    private var framer = MessageFramer()
    private var sendingSequence: UInt64 = 0
    private var receivedSequence: UInt64?
    private var ready: WireMessage?
    private var starting: CheckedContinuation<WireMessage, any Error>?
    private var pending: [UUID: Pending] = [:]
    private var outboundBytes = 0
    private var failed = false
    private var closing = false
    private var launched = false
    private var exitStatus: Int32?
    private var readers = 0
    private var writerClosed = true
    private var shutdownWaiters: [CheckedContinuation<Void, Never>] = []

    public init(executable: URL, arguments: [String] = [], environment: [String: String]? = nil,
                expectedRole: String = "compute",
                allowedEvents: Set<String> = ["job.progress", "job.completed", "job.failed", "job.cancelled"],
                onEvent: @escaping EventHandler = { _ in }, onFailure: @escaping FailureHandler = { _ in }) {
        self.executable = executable; self.arguments = arguments; self.environment = environment
        self.expectedRole = expectedRole; self.allowedEvents = allowedEvents
        self.onEvent = onEvent; self.onFailure = onFailure
    }

    deinit {
        readStop.stop(); writeStop.stop()
        if let process, process.isRunning { process.terminate() }
    }

    public func start(timeout: Duration = .seconds(20)) async throws -> WireMessage {
        let seconds = Self.seconds(timeout)
        guard seconds > 0, seconds <= 120 else { throw AstraError("compute.timeout", "Invalid compute startup timeout.") }
        return try await withCheckedThrowingContinuation { continuation in
            eventQueue.async { [self] in
                guard !launched, starting == nil, !failed, !closing else {
                    continuation.resume(throwing: AstraError("compute.state", "This compute process has already started or closed.")); return
                }
                launched = true; starting = continuation
                let task = Process()
                let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
                task.executableURL = executable; task.arguments = arguments; task.environment = environment
                task.standardInput = stdin; task.standardOutput = stdout; task.standardError = stderr
                task.terminationHandler = { [weak self] process in
                    let status = process.terminationStatus
                    self?.eventQueue.async { [weak self] in self?.terminated(status: status) }
                }
                do {
                    try Self.nonblocking(stdin.fileHandleForWriting)
                    // A dead helper must become an EPIPE error, never SIGPIPE in
                    // the UI process. This flag applies only to this pipe.
                    guard fcntl(stdin.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1) == 0 else {
                        throw AstraError("compute.pipe", "Could not configure the local runtime input pipe.")
                    }
                    try Self.nonblocking(stdout.fileHandleForReading)
                    try Self.nonblocking(stderr.fileHandleForReading)
                    try task.run()
                    process = task; input = stdin.fileHandleForWriting; writerClosed = false
                    try? stdin.fileHandleForReading.close()
                    try? stdout.fileHandleForWriting.close()
                    try? stderr.fileHandleForWriting.close()
                    readers = 2
                    read(stdout.fileHandleForReading, diagnostics: false)
                    read(stderr.fileHandleForReading, diagnostics: true)
                    eventQueue.asyncAfter(deadline: .now() + seconds) { [weak self] in
                        guard let self, starting != nil else { return }
                        finishFailure(AstraError("compute.startTimeout", "The local compute runtime did not become ready in time."))
                    }
                } catch {
                    // No reader or writer owns these handles after launch fails.
                    for file in [stdin.fileHandleForWriting, stdin.fileHandleForReading,
                                 stdout.fileHandleForWriting, stdout.fileHandleForReading,
                                 stderr.fileHandleForWriting, stderr.fileHandleForReading] { try? file.close() }
                    finishFailure(AstraError("compute.launch", "The local compute runtime could not start: \(error.localizedDescription)"))
                }
            }
        }
    }

    /// acceptingError preserves the entire rejection envelope, including exact
    /// frame-release acknowledgements. Transport faults always throw.
    public func request(kind: String, payload: JSONValue = .object([:]), runID: UUID? = nil,
                        timeout: Duration = .seconds(20), acceptingError: Bool = false) async throws -> WireMessage {
        let requestID = UUID(), cancellation = ChannelStop()
        let seconds = Self.seconds(timeout)
        guard seconds > 0, seconds <= 300 else { throw AstraError("compute.timeout", "Invalid compute request timeout.") }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                eventQueue.async { [self] in
                    guard !cancellation.isStopped else { continuation.resume(throwing: CancellationError()); return }
                    do {
                        try enqueue(kind: kind, payload: payload, runID: runID, requestID: requestID,
                                    acceptingError: acceptingError, continuation: continuation)
                        eventQueue.asyncAfter(deadline: .now() + seconds) { [weak self] in
                            guard let self, pending[requestID] != nil else { return }
                            finishFailure(AstraError("compute.requestTimeout", "The compute runtime did not acknowledge a request in time."))
                        }
                    } catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: {
            cancellation.stop()
            self.eventQueue.async { [weak self] in
                guard let self, pending[requestID] != nil else { return }
                // A dispatched request may already have side effects. Closing
                // this child prevents orphaned jobs after caller cancellation.
                finishFailure(AstraError("compute.requestCancelled", "The compute request was cancelled."))
            }
        }
    }

    /// Coalesced, uncancellable teardown. Compute/actor children use bounded
    /// TERM/KILL escalation. A control helper may own posted keys or buttons:
    /// only cooperative TERM is allowed, and teardown waits for its actual exit
    /// even when release is slow or temporarily denied by macOS.
    public func shutdown(grace: Duration = .seconds(2)) async {
        let seconds = min(10, max(0.01, Self.seconds(grace)))
        await withCheckedContinuation { continuation in
            eventQueue.async { [self] in
                shutdownWaiters.append(continuation)
                guard !closing else { completeShutdownIfJoined(); return }
                closing = true
                if process == nil { closeWriter(); completeShutdownIfJoined(); return }
                if !failed, ready != nil {
                    do {
                        try enqueue(kind: "shutdown", payload: .object([:]), runID: nil,
                                    requestID: UUID(), acceptingError: true, continuation: nil, internalShutdown: true)
                    } catch { beginTermination() }
                } else { beginTermination() }
                eventQueue.asyncAfter(deadline: .now() + seconds) { [weak self] in self?.beginTermination() }
            }
        }
    }

    /// Inspect only after shutdown has joined the child. Nil means this runtime
    /// never reached a launched process; a nonzero result includes signal exits.
    public func terminationStatus() async -> Int32? {
        await withCheckedContinuation { continuation in
            eventQueue.async { [self] in continuation.resume(returning: exitStatus) }
        }
    }

    private func enqueue(kind: String, payload: JSONValue, runID: UUID?, requestID: UUID,
                         acceptingError: Bool, continuation: CheckedContinuation<WireMessage, any Error>?,
                         internalShutdown: Bool = false) throws {
        guard ready != nil, !failed, (!closing || internalShutdown), pending.count < 64,
              sendingSequence < UInt64.max, let input, !writeStop.isStopped else {
            throw AstraError("compute.unavailable", "The compute runtime is unavailable or its request queue is full.")
        }
        let data = try WireMessage(kind: kind, sequence: sendingSequence, requestID: requestID, runID: runID, payload: payload).framed()
        guard outboundBytes + data.count <= 4 * AstraVersion.maximumMessageBytes else {
            throw AstraError("compute.backpressure", "The compute request queue could not keep up.")
        }
        pending[requestID] = Pending(runID: runID, acceptingError: acceptingError, continuation: continuation)
        sendingSequence += 1; outboundBytes += data.count
        let stop = writeStop
        writeQueue.async { [self] in
            var writeFailed = false
            data.withUnsafeBytes { raw in
                var offset = 0
                while offset < data.count, !stop.isStopped {
                    let count = Darwin.write(input.fileDescriptor, raw.baseAddress!.advanced(by: offset), data.count - offset)
                    if count > 0 { offset += count }
                    else if count < 0, errno == EINTR { continue }
                    else if count < 0, errno == EAGAIN || errno == EWOULDBLOCK {
                        var descriptor = pollfd(fd: input.fileDescriptor, events: Int16(POLLOUT), revents: 0)
                        _ = Darwin.poll(&descriptor, 1, 50)
                    } else { writeFailed = true; break }
                }
            }
            let didFail = writeFailed
            eventQueue.async { [self] in
                outboundBytes -= data.count
                if didFail, !closing { finishFailure(AstraError("compute.write", "The compute request channel closed unexpectedly.")) }
            }
        }
    }

    private func read(_ file: FileHandle, diagnostics: Bool) {
        let stop = readStop
        let thread = Thread { [self] in
            var buffer = [UInt8](repeating: 0, count: 16_384)
            var readFailed = false
            while !stop.isStopped {
                let count = Darwin.read(file.fileDescriptor, &buffer, buffer.count)
                if count == 0 { break }
                if count < 0 {
                    if errno == EINTR { continue }
                    if errno == EAGAIN || errno == EWOULDBLOCK {
                        var descriptor = pollfd(fd: file.fileDescriptor, events: Int16(POLLIN), revents: 0)
                        _ = Darwin.poll(&descriptor, 1, 50); continue
                    }
                    readFailed = true; break
                }
                if !diagnostics {
                    let data = Data(buffer.prefix(count))
                    eventQueue.sync { if !failed { receive(data) } }
                }
                // stderr is drained, never interpreted or copied into UI errors.
            }
            try? file.close()
            let didFail = readFailed
            eventQueue.async { [self] in
                readers -= 1
                if !diagnostics, !failed {
                    do { try framer.finish() }
                    catch { finishFailure(AstraError("compute.truncated", "The compute runtime closed inside a protocol message.")) }
                }
                if didFail, !closing { finishFailure(AstraError("compute.read", "The compute output channel could not be read.")) }
                finalizeExitIfDrained()
            }
        }
        thread.name = diagnostics ? "Astra compute diagnostics" : "Astra compute messages"
        thread.qualityOfService = .userInitiated; thread.start()
    }

    private func receive(_ data: Data) {
        do {
            for message in try framer.append(data) {
                guard receivedSequence != UInt64.max, message.sequence == (receivedSequence.map { $0 + 1 } ?? 0) else {
                    throw AstraError("compute.sequence", "The compute runtime reordered or repeated a message.")
                }
                receivedSequence = message.sequence
                if ready == nil {
                    guard message.kind == "hello", message.requestID == nil, message.runID == nil,
                          case .object(let fields) = message.payload,
                          fields["role"] == .string(expectedRole), fields["protocolVersion"] == .integer(1) else {
                        throw AstraError("compute.handshake", "The local runtime sent an incompatible handshake.")
                    }
                    ready = message; starting?.resume(returning: message); starting = nil
                } else if message.kind == "ack" || message.kind == "error" {
                    guard let identifier = message.requestID, let request = pending[identifier], request.runID == message.runID else {
                        throw AstraError("compute.reply", "The local runtime replied to an unknown request or a different run.")
                    }
                    pending.removeValue(forKey: identifier)
                    if let continuation = request.continuation {
                        if message.kind == "error", !request.acceptingError {
                            let failure = (try? message.payload.decode(AstraError.self)) ?? AstraError("compute.rejected", "The local runtime rejected the request.")
                            continuation.resume(throwing: failure)
                        } else { continuation.resume(returning: message) }
                    } else {
                        closeWriter()
                        // An acknowledgement is not process-exit evidence. Give
                        // helper finalizers time; the grace timer escalates later.
                    }
                } else {
                    guard allowedEvents.contains(message.kind), message.runID != nil else {
                        throw AstraError("compute.event", "The local runtime emitted an unknown or unscoped event.")
                    }
                    onEvent(message)
                }
            }
        } catch { finishFailure((error as? AstraError) ?? AstraError("compute.protocol", "The compute runtime sent invalid protocol data.")) }
    }

    private func terminated(status: Int32) {
        exitStatus = status
        closeWriter()
        // A descendant must not retain a dead child's pipes indefinitely. Allow
        // pending final output to drain before interrupting the nonblocking loops.
        eventQueue.asyncAfter(deadline: .now() + 0.25) { [weak self] in self?.readStop.stop() }
        finalizeExitIfDrained()
    }

    private func finalizeExitIfDrained() {
        guard let status = exitStatus, readers == 0 else { return }
        if !closing, !failed { finishFailure(AstraError("compute.exited", "The compute runtime exited unexpectedly (status \(status)).")) }
        failContinuations(AstraError("compute.closed", "The compute runtime closed."))
        process = nil
        completeShutdownIfJoined()
    }

    private func beginTermination() {
        closeWriter()
        guard let process, process.isRunning, exitStatus == nil else { completeShutdownIfJoined(); return }
        process.terminate()
        // An unsettled control helper retains its desktop lock and retries
        // owned input cleanup independently. Killing it here would discard that
        // ledger and let the UI falsely report that controls were released.
        guard expectedRole != "control" else { return }
        eventQueue.asyncAfter(deadline: .now() + 1) { [weak self, weak process] in
            guard let self, let process, self.process === process, exitStatus == nil, process.isRunning else { return }
            _ = Darwin.kill(process.processIdentifier, SIGKILL)
        }
    }

    private func closeWriter() {
        guard let input else { return }
        self.input = nil; writeStop.stop()
        // Closing belongs to the writer queue, after every in-flight write. No
        // concurrent close/reused descriptor race can redirect bytes elsewhere.
        writeQueue.async { [self] in
            try? input.close()
            eventQueue.async { [self] in writerClosed = true; completeShutdownIfJoined() }
        }
    }

    private func finishFailure(_ error: AstraError) {
        guard !failed else { return }
        failed = true; failContinuations(error); beginTermination()
        if !closing { onFailure(error) }
    }

    private func failContinuations(_ error: AstraError) {
        starting?.resume(throwing: error); starting = nil
        for request in pending.values { request.continuation?.resume(throwing: error) }
        pending.removeAll()
    }

    private func completeShutdownIfJoined() {
        guard process == nil, readers == 0, writerClosed else { return }
        failContinuations(AstraError("compute.closed", "The compute runtime closed."))
        let waiters = shutdownWaiters; shutdownWaiters.removeAll()
        for continuation in waiters { continuation.resume() }
    }

    private static func nonblocking(_ file: FileHandle) throws {
        let flags = fcntl(file.fileDescriptor, F_GETFL)
        guard flags >= 0, fcntl(file.fileDescriptor, F_SETFL, flags | O_NONBLOCK) == 0 else {
            throw AstraError("compute.pipe", "Could not configure a bounded local runtime pipe.")
        }
    }
    private static func seconds(_ value: Duration) -> Double {
        let components = value.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
