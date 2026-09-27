import Foundation
import Darwin
import Security
import AstraCore

public struct LiveSignalEndpoint: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let host: String
    public let port: UInt16
    public let token: String
    public let sessionID: UUID
    public var address: String { "\(host):\(port)" }
    public var description: String { "Local state source at \(address)" }
    public var debugDescription: String { "\(description), session \(sessionID), token <redacted>" }
}
public struct LiveSignalServerStatus: Sendable {
    public let listening: Bool
    public let connected: Bool
    public let rejectedRequests: Int
    public let issue: String?
}

/// Local telemetry only. A single socket owner closes descriptors after its
/// bounded poll loop exits; cancellation never closes an FD another task uses.
public final class LoopbackLiveSignalServer: @unchecked Sendable {
    public let endpoint: LiveSignalEndpoint
    private let worker: LiveSignalSocketWorker
    private init(store: LiveSignalStore) throws {
        var secret = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, secret.count, &secret) == errSecSuccess else {
            throw AstraError("liveSignal.random", "A private local source token could not be created.")
        }
        let token = secret.map { String(format: "%02x", $0) }.joined()
        let listener = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard listener >= 0 else { throw Self.socketError("create the local listener") }
        do {
            guard fcntl(listener, F_SETFD, FD_CLOEXEC) == 0, fcntl(listener, F_SETFL, O_NONBLOCK) == 0 else {
                throw Self.socketError("configure the local listener")
            }
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size); address.sin_family = sa_family_t(AF_INET)
            address.sin_port = 0; address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
            let bound = withUnsafePointer(to: &address) { pointer in
                pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bound == 0, Darwin.listen(listener, 4) == 0 else { throw Self.socketError("bind the local listener") }
            var length = socklen_t(MemoryLayout<sockaddr_in>.size)
            let found = withUnsafeMutablePointer(to: &address) { pointer in
                pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(listener, $0, &length) }
            }
            guard found == 0, address.sin_family == sa_family_t(AF_INET), address.sin_addr.s_addr == inet_addr("127.0.0.1"), address.sin_port != 0 else {
                throw Self.socketError("identify the bound local listener")
            }
            endpoint = .init(host: "127.0.0.1", port: UInt16(bigEndian: address.sin_port), token: token, sessionID: store.sessionID)
            worker = LiveSignalSocketWorker(listener: listener, store: store, token: token)
        } catch { Darwin.close(listener); throw error }
        let owner = worker
        let thread = Thread { owner.run() }
        thread.name = "Astra local state source"; thread.qualityOfService = .userInitiated; thread.start()
    }
    public static func start(store: LiveSignalStore) async throws -> LoopbackLiveSignalServer {
        try Task.checkCancellation()
        let server = try Self(store: store)
        if Task.isCancelled { await server.stopAndJoin(); throw CancellationError() }
        return server
    }
    public var status: LiveSignalServerStatus { worker.status }
    public func stopAndJoin() async { worker.requestStop(); await worker.finished.wait() }
    deinit { worker.requestStop() }
    private static func socketError(_ action: String) -> AstraError {
        .init("liveSignal.socket", "Could not \(action): \(String(cString: strerror(errno))).")
    }
}

private final class LiveSignalSocketWorker: @unchecked Sendable {
    static let maximumMessageBytes = 256 * 1024
    let finished = AsyncCompletion()
    private let listener: Int32
    private let store: LiveSignalStore
    private let token: [UInt8]
    private let lock = NSLock()
    private var stopping = false, listening = true, connected = false
    private var rejected = 0
    private var issue: String?
    init(listener: Int32, store: LiveSignalStore, token: String) {
        self.listener = listener; self.store = store; self.token = Array(token.utf8)
    }
    var status: LiveSignalServerStatus { lock.withLock { .init(listening: listening, connected: connected, rejectedRequests: rejected, issue: issue) } }
    func requestStop() { lock.withLock { stopping = true } }
    private var shouldStop: Bool { lock.withLock { stopping } }
    private func reject(_ message: String) { lock.withLock { rejected = min(Int.max - 1, rejected) + 1; issue = message } }

    func run() {
        var client: Int32 = -1
        var input = Data(), output = Data(), sent = 0
        var authenticated = false
        var lastActivity = ContinuousClock.now
        var authenticationDeadline = ContinuousClock.now.advanced(by: .seconds(2))
        func closeClient() {
            if client >= 0 { _ = shutdown(client, SHUT_RDWR); Darwin.close(client); client = -1 }
            input.removeAll(keepingCapacity: true); output.removeAll(keepingCapacity: true); sent = 0; authenticated = false
            lock.withLock { connected = false }
        }
        defer {
            closeClient(); Darwin.close(listener); store.close()
            lock.withLock { listening = false }; finished.finish()
        }
        while !shouldStop {
            // Buffered pipelined requests still execute one at a time. A slow
            // reader cannot grow an application-side response queue.
            if client >= 0, output.isEmpty, let newline = input.firstIndex(of: 10) {
                let line = Data(input.prefix(upTo: newline)); input.removeSubrange(...newline)
                do {
                    let result = try respond(line)
                    authenticated = true
                    lock.withLock { connected = true; issue = nil }
                    output = try JSONEncoder().encode(result); output.append(10); sent = 0
                    guard output.count <= Self.maximumMessageBytes else { throw AstraError("liveSignal.response", "The local source response exceeds its size limit.") }
                    lastActivity = .now
                } catch {
                    let failure = (error as? AstraError) ?? .init("liveSignal.request", "The local source request is malformed.")
                    reject(failure.message)
                    if failure.code == "liveSignal.authentication" { closeClient(); continue }
                    let reply: JSONValue = .object(["ok": .bool(false), "code": .string(failure.code), "message": .string(failure.message)])
                    output = (try? JSONEncoder().encode(reply)) ?? Data(); output.append(10); sent = 0
                }
            }
            if client >= 0, (!authenticated && ContinuousClock.now > authenticationDeadline)
                || (authenticated && lastActivity.duration(to: .now) > .seconds(30)) { closeClient() }
            var polls = [pollfd(fd: listener, events: Int16(POLLIN), revents: 0)]
            if client >= 0 { polls.append(pollfd(fd: client, events: Int16(output.isEmpty ? POLLIN : POLLOUT), revents: 0)) }
            let ready = Darwin.poll(&polls, nfds_t(polls.count), 50)
            if ready < 0 {
                if errno == EINTR { continue }
                lock.withLock { issue = "The local state listener stopped unexpectedly." }; return
            }
            if polls[0].revents & Int16(POLLERR | POLLHUP | POLLNVAL) != 0 {
                lock.withLock { issue = "The local state listener is unavailable." }; return
            }
            if polls[0].revents & Int16(POLLIN) != 0 {
                let accepted = Darwin.accept(listener, nil, nil)
                if accepted >= 0 {
                    if client >= 0 { Darwin.close(accepted) }
                    else {
                        var one: Int32 = 1
                        guard fcntl(accepted, F_SETFD, FD_CLOEXEC) == 0, fcntl(accepted, F_SETFL, O_NONBLOCK) == 0,
                              setsockopt(accepted, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size)) == 0 else {
                            Darwin.close(accepted); continue
                        }
                        client = accepted; lastActivity = .now; authenticationDeadline = lastActivity.advanced(by: .seconds(2))
                    }
                }
            }
            guard polls.count == 2, client >= 0 else { continue }
            if polls[1].revents & Int16(POLLERR | POLLHUP | POLLNVAL) != 0 { closeClient(); continue }
            if polls[1].revents & Int16(POLLIN) != 0 {
                var buffer = [UInt8](repeating: 0, count: 16 * 1024)
                let count = Darwin.recv(client, &buffer, buffer.count, 0)
                if count > 0 {
                    guard input.count + count <= Self.maximumMessageBytes else { reject("The local source exceeded its message limit."); closeClient(); continue }
                    input.append(contentsOf: buffer.prefix(count)); lastActivity = .now
                } else if count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { closeClient() }
            }
            if client >= 0, polls[1].revents & Int16(POLLOUT) != 0, !output.isEmpty {
                let count = output.withUnsafeBytes { bytes in Darwin.send(client, bytes.baseAddress!.advanced(by: sent), bytes.count - sent, 0) }
                if count > 0 {
                    sent += count; lastActivity = .now
                    if sent == output.count { output.removeAll(keepingCapacity: true); sent = 0 }
                } else if count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { closeClient() }
            }
        }
    }

    private func respond(_ data: Data) throws -> JSONValue {
        guard !data.isEmpty, data.count < Self.maximumMessageBytes,
              let fields = try JSONDecoder().decode(JSONValue.self, from: data).fields else {
            throw AstraError("liveSignal.request", "Send one bounded JSON object per line.")
        }
        guard let candidate = fields["token"]?.text, constantTimeToken(candidate),
              fields["sessionID"]?.uuid == store.sessionID else {
            throw AstraError("liveSignal.authentication", "The local source credentials do not match this session.")
        }
        guard fields["version"] == .integer(1), let operation = fields["op"]?.text else {
            throw AstraError("liveSignal.version", "Use local state protocol version one.")
        }
        switch operation {
        case "binding.get":
            guard Set(fields.keys) == ["version", "op", "token", "sessionID"] else { throw AstraError("liveSignal.fields", "The binding request contains unsupported fields.") }
            let status = store.status
            guard !status.closed else { throw AstraError("liveSignal.closed", "The state source has closed.") }
            return .object(["ok": .bool(true), "sessionID": .string(store.sessionID.uuidString.lowercased()),
                "binding": try status.binding.map(JSONValue.encode) ?? .null, "nextSequence": .unsigned(status.nextSequence)])
        case "values.put":
            guard Set(fields.keys) == ["version", "op", "token", "sessionID", "bindingID", "episodeID", "sequence", "values"],
                  let binding = fields["bindingID"]?.uuid, let episode = fields["episodeID"]?.uuid,
                  let sequence = fields["sequence"]?.uint64, case .array(let values) = fields["values"], (1...32).contains(values.count) else {
                throw AstraError("liveSignal.fields", "Publish the current binding, episode, sequence and a bounded value list; sender timestamps are not accepted.")
            }
            let updates = try values.map { value -> LiveSignalValueUpdate in
                guard let item = value.fields, Set(item.keys).isSubset(of: ["signalID", "value", "confidence"]),
                      let signal = item["signalID"]?.uuid, let raw = item["value"],
                      let confidence = item["confidence"].map({ $0.double }) ?? 1 else {
                    throw AstraError("liveSignal.value", "Each update requires a signal identity and scalar value.")
                }
                let signalValue: SignalValue
                switch raw {
                case .number(let value): signalValue = .number(value)
                case .integer(let value):
                    guard let exact = Double(exactly: value) else { throw AstraError("liveSignal.number", "The integer cannot be represented exactly as a numeric state value; send exact identifiers as text.") }
                    signalValue = .number(exact)
                case .unsigned(let value):
                    guard let exact = Double(exactly: value) else { throw AstraError("liveSignal.number", "The integer cannot be represented exactly as a numeric state value; send exact identifiers as text.") }
                    signalValue = .number(exact)
                case .bool(let value): signalValue = .flag(value)
                case .string(let value): signalValue = .text(value)
                case .null: signalValue = .unknown("The source reported that its value is unavailable.")
                default: throw AstraError("liveSignal.value", "A state value must be a finite number, text, Boolean or null.")
                }
                return .init(signalID: signal, value: signalValue, confidence: confidence)
            }
            let receipt = try store.accept(sessionID: store.sessionID, bindingID: binding, episodeID: episode, sequence: sequence, values: updates)
            return .object(["ok": .bool(true), "receipt": try .encode(receipt)])
        default: throw AstraError("liveSignal.operation", "The local source operation is unsupported.")
        }
    }
    private func constantTimeToken(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        guard bytes.count == token.count else { return false }
        return zip(bytes, token).reduce(UInt8(0)) { $0 | ($1.0 ^ $1.1) } == 0
    }
}
