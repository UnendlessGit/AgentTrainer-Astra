import Foundation
import Darwin
import Testing
import AstraCore
@testable import AstraPlatform

private final class LiveSignalClock: @unchecked Sendable {
    private let lock = NSLock()
    private var value: UInt64 = 100
    func now() -> UInt64 { lock.withLock { value } }
    func set(_ value: UInt64) { lock.withLock { self.value = value } }
}
private final class StateSocketClient: @unchecked Sendable {
    private var descriptor: Int32
    private let endpoint: LiveSignalEndpoint
    init(_ endpoint: LiveSignalEndpoint) throws {
        self.endpoint = endpoint; descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw AstraError("test.socket", "Could not create the test socket") }
        var timeout = timeval(tv_sec: 2, tv_usec: 0), one: Int32 = 1
        _ = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        _ = setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        _ = setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in(); address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET); address.sin_port = endpoint.port.bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard result == 0 else { Darwin.close(descriptor); descriptor = -1; throw AstraError("test.connect", "Could not connect to the test source") }
    }
    deinit { close() }
    func close() { if descriptor >= 0 { Darwin.close(descriptor); descriptor = -1 } }
    func request(_ operation: String, fields: [String: JSONValue] = [:], token: String? = nil) throws -> JSONValue {
        var value = fields
        value["version"] = .integer(1); value["op"] = .string(operation)
        value["token"] = .string(token ?? endpoint.token); value["sessionID"] = .string(endpoint.sessionID.uuidString.lowercased())
        var data = try JSONEncoder().encode(JSONValue.object(value)); data.append(10)
        let count = data.withUnsafeBytes { Darwin.send(descriptor, $0.baseAddress, $0.count, 0) }
        guard count == data.count else { throw AstraError("test.write", "Test request could not be sent") }
        var response = Data(), byte: UInt8 = 0
        while response.count < 256 * 1024 {
            guard Darwin.recv(descriptor, &byte, 1, 0) == 1 else { throw AstraError("test.read", "Test source closed or timed out") }
            if byte == 10 { return try JSONDecoder().decode(JSONValue.self, from: response) }
            response.append(byte)
        }
        throw AstraError("test.size", "Oversized test response")
    }
}

@Test func localStateSocketSealsCausalValuesAndReconnectsWithoutReplayingHistory() async throws {
    let clock = LiveSignalClock()
    let score = RewardSignal(name: "Score", kind: .manual), state = RewardSignal(name: "State", kind: .manual)
    let store = try LiveSignalStore(sessionID: UUID(), signals: [score, state], clock: { clock.now() })
    let server = try await LoopbackLiveSignalServer.start(store: store)
    var client: StateSocketClient?
    do {
        let binding = try store.bind(episodeID: UUID(), resetID: UUID())
        client = try StateSocketClient(server.endpoint)
        let connected = try #require(client)
        let announced = try connected.request("binding.get")
        #expect(announced.fields?["ok"] == .bool(true) && announced.fields?["nextSequence"]?.uint64 == 0)
        func fields(_ sequence: UInt64, _ number: Int) -> [String: JSONValue] {
            ["bindingID": .string(binding.bindingID.uuidString.lowercased()), "episodeID": .string(binding.episodeID.uuidString.lowercased()),
             "sequence": .unsigned(sequence), "values": .array([
                .object(["signalID": .string(score.id.uuidString.lowercased()), "value": .integer(Int64(number))]),
                .object(["signalID": .string(state.id.uuidString.lowercased()), "value": .bool(true)])])]
        }
        clock.set(110)
        let accepted = try connected.request("values.put", fields: fields(0, 5))
        #expect(accepted.fields?["ok"] == .bool(true) && accepted.fields?["receipt"]?.fields?["receivedAtNanos"]?.uint64 == 110)
        #expect(try store.readings(binding: binding, cutoffNanos: 105).isEmpty)
        clock.set(120)
        let sealed = try store.seal(binding: binding, observationID: UUID(), cutoffNanos: 120)
        #expect(sealed.first(where: { $0.signalID == score.id })?.value == .number(5))
        let retroactive = try connected.request("values.put", fields: fields(1, 6))
        #expect(retroactive.fields?["code"] == .string("liveSignal.causality"))
        clock.set(140)
        #expect(try connected.request("values.put", fields: fields(1, 7)).fields?["ok"] == .bool(true))
        let earlier = try store.seal(binding: binding, observationID: UUID(), cutoffNanos: 130)
        #expect(earlier.first(where: { $0.signalID == score.id })?.value == .number(5))
        #expect(sealed.first(where: { $0.signalID == score.id })?.value == .number(5))
        connected.close(); client = nil
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while server.status.connected && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        client = try StateSocketClient(server.endpoint)
        let reconnected = try #require(client)
        #expect(try reconnected.request("binding.get").fields?["nextSequence"]?.uint64 == 2)
        clock.set(160)
        #expect(try reconnected.request("values.put", fields: fields(2, 11)).fields?["ok"] == .bool(true))
        clock.set(180)
        #expect(try reconnected.request("values.put", fields: fields(5, 9)).fields?["code"] == .string("liveSignal.sequence"))
        let gap = try store.readings(binding: binding, cutoffNanos: 180)
        #expect(gap.count == 2 && gap.allSatisfy { if case .unknown = $0.value { true } else { false } })
        clock.set(200)
        #expect(try reconnected.request("values.put", fields: fields(3, 15)).fields?["code"] == .string("liveSignal.discontinuity"))
        clock.set(1_000_000_000)
        let readings = try store.readings(binding: binding, cutoffNanos: clock.now())
        let resolved = try RewardEvaluator(program: .init(name: "Socket fixture", signals: [score, state]))
            .resolveSnapshot(episodeID: binding.episodeID, cutoffNanos: clock.now(), readings: readings)
        #expect(resolved.values.values.allSatisfy { if case .unknown = $0 { true } else { false } })
        let next = try store.bind(episodeID: UUID(), resetID: UUID())
        #expect(next.bindingID != binding.bindingID && store.status.nextSequence == 0)
        #expect(try reconnected.request("values.put", fields: fields(3, 15)).fields?["code"] == .string("liveSignal.binding"))
        client?.close(); await server.stopAndJoin()
        #expect(store.status.closed && !server.status.listening && !server.status.connected)
    } catch {
        client?.close(); await server.stopAndJoin(); throw error
    }
}
