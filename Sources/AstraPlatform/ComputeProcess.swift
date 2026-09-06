import Foundation
import Darwin
import AstraCore

/// A bounded inherited-pipe client. Blocking reads/writes have dedicated queues;
/// protocol state and request completion belong to a single event queue.
public final class ComputeProcess: @unchecked Sendable {
    public typealias EventHandler = @Sendable (WireMessage) -> Void
    public typealias FailureHandler = @Sendable (AstraError) -> Void
    private let eventQueue = DispatchQueue(label: "astra.compute.protocol", qos: .userInitiated)
    private let writeQueue = DispatchQueue(label: "astra.compute.write", qos: .userInitiated)
    private let executable: URL
    private let arguments: [String]
    private let environment: [String: String]?
    private let onEvent: EventHandler
    private let onFailure: FailureHandler
    // eventQueue owns all following mutable state.
    private var process: Process?
    private var input: FileHandle?
    private var framer = MessageFramer()
    private var sendingSequence: UInt64 = 0
    private var receivedSequence: UInt64?
    private var ready: WireMessage?
    private var starting: CheckedContinuation<WireMessage, any Error>?
    private var pending: [UUID: CheckedContinuation<WireMessage, any Error>] = [:]
    private var outboundBytes = 0
    private var failed = false
    private var closing = false
    private var generation = UUID()

    public init(executable: URL, arguments: [String] = [], environment: [String: String]? = nil,
                onEvent: @escaping EventHandler = { _ in }, onFailure: @escaping FailureHandler = { _ in }) {
        self.executable = executable; self.arguments = arguments; self.environment = environment
        self.onEvent = onEvent; self.onFailure = onFailure
    }

    deinit {
        try? input?.close()
        if let process, process.isRunning { process.terminate() }
    }

    public func start(timeout: Duration = .seconds(20)) async throws -> WireMessage {
        let seconds = Self.seconds(timeout)
        guard seconds > 0, seconds <= 120 else { throw AstraError("compute.timeout", "Invalid compute startup timeout.") }
        return try await withCheckedThrowingContinuation { continuation in
            eventQueue.async { [self] in
                guard process == nil, starting == nil, !failed else {
                    continuation.resume(throwing: AstraError("compute.state", "This compute process has already started or closed.")); return
                }
                starting = continuation
                let task = Process()
                let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
                task.executableURL = executable; task.arguments = arguments
                task.environment = environment
                task.standardInput = stdin; task.standardOutput = stdout; task.standardError = stderr
                let token = generation
                task.terminationHandler = { [weak self] process in
                    let status = process.terminationStatus
                    self?.eventQueue.async { [weak self] in self?.terminated(status: status, generation: token) }
                }
                do {
                    try task.run()
                    process = task; input = stdin.fileHandleForWriting
                    // Parent closes the unused pipe ends so EOF has a precise
                    // lifecycle meaning and a helper cannot stay alive by leak.
                    try stdin.fileHandleForReading.close()
                    try stdout.fileHandleForWriting.close()
                    try stderr.fileHandleForWriting.close()
                    read(stdout.fileHandleForReading, generation: token, diagnostics: false)
                    read(stderr.fileHandleForReading, generation: token, diagnostics: true)
                    eventQueue.asyncAfter(deadline: .now() + seconds) { [weak self] in
                        guard let self, generation == token, starting != nil else { return }
                        finishFailure(AstraError("compute.startTimeout", "The local compute runtime did not become ready in time."))
                    }
                } catch {
                    finishFailure(AstraError("compute.launch", "The local compute runtime could not start: \(error.localizedDescription)"))
                }
            }
        }
    }

    public func request(kind: String, payload: JSONValue = .object([:]), runID: UUID? = nil,
                        timeout: Duration = .seconds(20)) async throws -> WireMessage {
        let requestID = UUID()
        let seconds = Self.seconds(timeout)
        guard seconds > 0, seconds <= 300 else { throw AstraError("compute.timeout", "Invalid compute request timeout.") }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                eventQueue.async { [self] in
                    guard ready != nil, !failed, (!closing || kind == "shutdown"), pending.count < 64,
                          sendingSequence < UInt64.max else {
                        continuation.resume(throwing: AstraError("compute.unavailable", "The compute runtime is unavailable or its request queue is full.")); return
                    }
                    do {
                        let message = WireMessage(kind: kind, sequence: sendingSequence, requestID: requestID, runID: runID, payload: payload)
                        let data = try message.framed()
                        guard outboundBytes + data.count <= 4 * AstraVersion.maximumMessageBytes, let input else {
                            throw AstraError("compute.backpressure", "The compute request queue could not keep up.")
                        }
                        pending[requestID] = continuation
                        sendingSequence += 1; outboundBytes += data.count
                        writeQueue.async { [weak self] in
                            do { try input.write(contentsOf: data) }
                            catch {
                                self?.eventQueue.async { [weak self] in
                                    self?.finishFailure(AstraError("compute.write", "The compute request channel closed unexpectedly."))
                                }
                            }
                            self?.eventQueue.async { [weak self] in self?.outboundBytes -= data.count }
                        }
                        eventQueue.asyncAfter(deadline: .now() + seconds) { [weak self] in
                            guard let self, pending[requestID] != nil else { return }
                            finishFailure(AstraError("compute.requestTimeout", "The compute runtime did not acknowledge a request in time."))
                        }
                    } catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: {
            // Cancellation cannot infer that a dispatched job never started.
            // Close this child so callers cannot accidentally orphan its work.
            self.eventQueue.async { [weak self] in
                guard let self, pending[requestID] != nil else { return }
                finishFailure(AstraError("compute.requestCancelled", "The compute request was cancelled."))
            }
        }
    }

    public func shutdown() async {
        await withCheckedContinuation { continuation in
            eventQueue.async { [self] in closing = true; continuation.resume() }
        }
        _ = try? await request(kind: "shutdown", timeout: .seconds(10))
        await withCheckedContinuation { continuation in
            eventQueue.async { [self] in
                closing = true
                try? input?.close(); input = nil
                if let process, process.isRunning { process.terminate() }
                failContinuations(AstraError("compute.closed", "The compute runtime closed."))
                continuation.resume()
            }
        }
    }

    private func read(_ file: FileHandle, generation token: UUID, diagnostics: Bool) {
        let thread = Thread { [weak self] in
            defer { try? file.close() }
            do {
                var buffer = [UInt8](repeating: 0, count: 16_384)
                while true {
                    let count = Darwin.read(file.fileDescriptor, &buffer, buffer.count)
                    if count == 0 { break }
                    if count < 0 {
                        if errno == EINTR { continue }
                        throw AstraError("compute.pipeRead", "The compute pipe could not be read.")
                    }
                    let data = Data(buffer.prefix(count))
                    guard let self else { return }
                    // Synchronous delivery bounds raw transport memory even if
                    // a malformed child writes faster than parsing can proceed.
                    let keepReading = eventQueue.sync { () -> Bool in
                        guard generation == token, !failed else { return false }
                        if !diagnostics { receive(data) }
                        // stderr is drained, never interpreted as protocol or
                        // copied into user-facing errors with recorded content.
                        return !failed
                    }
                    if !keepReading { return }
                }
                self?.eventQueue.async { [weak self] in
                    guard let self, generation == token, !diagnostics else { return }
                    do { try framer.finish() }
                    catch { finishFailure(AstraError("compute.truncated", "The compute runtime closed inside a protocol message.")) }
                }
            } catch {
                self?.eventQueue.async { [weak self] in
                    guard let self, generation == token, !closing else { return }
                    finishFailure(AstraError("compute.read", "The compute output channel could not be read."))
                }
            }
        }
        thread.name = diagnostics ? "Astra compute diagnostics" : "Astra compute messages"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    private func receive(_ data: Data) {
        do {
            for message in try framer.append(data) {
                guard receivedSequence != UInt64.max, message.sequence == (receivedSequence.map { $0 + 1 } ?? 0) else {
                    throw AstraError("compute.sequence", "The compute runtime reordered or repeated a message.")
                }
                receivedSequence = message.sequence
                if ready == nil {
                    guard message.kind == "hello", case .object(let fields) = message.payload,
                          fields["role"] == .string("compute"), fields["protocolVersion"] == .integer(1) else {
                        throw AstraError("compute.handshake", "The local runtime sent an incompatible handshake.")
                    }
                    ready = message; starting?.resume(returning: message); starting = nil
                } else if message.kind == "ack" || message.kind == "error" {
                    guard let identifier = message.requestID, let continuation = pending.removeValue(forKey: identifier) else {
                        throw AstraError("compute.reply", "The local runtime replied to an unknown request.")
                    }
                    if message.kind == "error" {
                        let failure = (try? message.payload.decode(AstraError.self)) ?? AstraError("compute.rejected", "The local runtime rejected the request.")
                        continuation.resume(throwing: failure)
                    } else { continuation.resume(returning: message) }
                } else {
                    guard ["job.progress", "job.completed", "job.failed", "job.cancelled"].contains(message.kind), message.runID != nil else {
                        throw AstraError("compute.event", "The local runtime emitted an unknown job event.")
                    }
                    onEvent(message)
                }
            }
        } catch { finishFailure((error as? AstraError) ?? AstraError("compute.protocol", "The compute runtime sent invalid protocol data.")) }
    }

    private func terminated(status: Int32, generation token: UUID) {
        guard generation == token else { return }
        if !closing { finishFailure(AstraError("compute.exited", "The compute runtime exited unexpectedly (status \(status)).")) }
        else { failContinuations(AstraError("compute.closed", "The compute runtime closed.")) }
        process = nil; try? input?.close(); input = nil
    }

    private func finishFailure(_ error: AstraError) {
        guard !failed else { return }
        failed = true
        failContinuations(error)
        try? input?.close(); input = nil
        if let process, process.isRunning { process.terminate() }
        onFailure(error)
    }

    private func failContinuations(_ error: AstraError) {
        starting?.resume(throwing: error); starting = nil
        for continuation in pending.values { continuation.resume(throwing: error) }
        pending.removeAll()
    }

    private static func seconds(_ value: Duration) -> Double {
        let components = value.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
