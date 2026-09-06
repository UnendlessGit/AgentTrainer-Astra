import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

private func pythonURL() -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent(".venv/bin/python")
}

private let wireFixture = #"""
import json, sys, time
sequence = 0
def send(kind, payload, request=None):
    global sequence
    value = dict(version=1, kind=kind, sequence=sequence, payload=payload)
    if request:
        for field in ('requestID', 'runID'):
            if field in request: value[field] = request[field]
    print(json.dumps(value), flush=True)
    sequence += 1
send('hello', dict(role='compute', protocolVersion=1, capabilities=['ping','shutdown']))
for line in sys.stdin:
    request = json.loads(line)
    if request['kind'] == 'hang':
        time.sleep(5)
    elif request['kind'] == 'malformed':
        print('{broken', flush=True)
    else:
        send('ack', dict(echo=request['payload']), request)
        if request['kind'] == 'shutdown': break
"""#

@Test func computeClientCorrelatesConcurrentRequestsAndClosesItsChild() async throws {
    let client = ComputeProcess(executable: pythonURL(), arguments: ["-u", "-c", wireFixture])
    let hello = try await client.start()
    #expect(hello.kind == "hello")
    try await withThrowingTaskGroup(of: (Int, WireMessage).self) { group in
        for number in 0..<12 {
            group.addTask {
                (number, try await client.request(kind: "ping", payload: .object(["number": .integer(Int64(number))])))
            }
        }
        for try await (number, reply) in group {
            #expect(reply.payload == .object(["echo": .object(["number": .integer(Int64(number))])]))
            #expect(reply.requestID != nil)
        }
    }
    await client.shutdown()
    await #expect(throws: AstraError.self) { try await client.request(kind: "ping") }
}

@Test func computeClientTimeoutAndMalformedMessagesFailOutstandingRequests() async throws {
    let hung = ComputeProcess(executable: pythonURL(), arguments: ["-u", "-c", wireFixture])
    _ = try await hung.start()
    await #expect(throws: AstraError.self) { try await hung.request(kind: "hang", timeout: .milliseconds(100)) }
    await hung.shutdown()
    let malformed = ComputeProcess(executable: pythonURL(), arguments: ["-u", "-c", wireFixture])
    _ = try await malformed.start()
    await #expect(throws: AstraError.self) { try await malformed.request(kind: "malformed") }
    await malformed.shutdown()
}

@Test func computeClientRejectsWrongHandshakeAndMissingExecutable() async throws {
    let wrong = ComputeProcess(executable: pythonURL(), arguments: ["-u", "-c", "print('{\"version\":1,\"kind\":\"hello\",\"sequence\":0,\"payload\":{\"role\":\"wrong\",\"protocolVersion\":1}}',flush=True)"])
    await #expect(throws: AstraError.self) { try await wrong.start() }
    await wrong.shutdown()
    let missing = ComputeProcess(executable: URL(fileURLWithPath: "/astra-no-such-runtime-\(UUID().uuidString)"))
    await #expect(throws: AstraError.self) { try await missing.start() }
    await missing.shutdown()
}
