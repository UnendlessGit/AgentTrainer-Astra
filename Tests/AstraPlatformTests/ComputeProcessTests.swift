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

private let lifetimeFixture = #"""
import json, sys, time, os, signal
sequence=0
mode=sys.argv[1]
def send(kind,payload,request=None):
    global sequence
    value=dict(version=1,kind=kind,sequence=sequence,payload=payload)
    if request:
        for field in ('requestID','runID'):
            if field in request: value[field]=request[field]
    print(json.dumps(value),flush=True); sequence+=1
if mode in ('ignore','blocked'): signal.signal(signal.SIGTERM, signal.SIG_IGN)
send('hello',dict(role='actor',protocolVersion=1,pid=os.getpid()))
if mode=='blocked':
    while True: time.sleep(.1)
for line in sys.stdin:
    request=json.loads(line)
    if request['kind']=='reject':
        send('error',dict(code='actor.invalid',message='Rejected',recoverable=True,
             releasedFrames=[dict(slotIndex=1,lease=9223372036854775859)]),request)
    elif request['kind']=='wrongRun':
        request['runID']='22222222-2222-2222-2222-222222222222'
        send('ack',{},request)
    else:
        send('ack',{},request)
    if request['kind']=='shutdown':
        if mode=='ignore':
            while True: time.sleep(.1)
        time.sleep(.2)
        break
"""#

private func lifetimeClient(_ mode: String) -> ComputeProcess {
    ComputeProcess(executable: pythonURL(), arguments: ["-u", "-c", lifetimeFixture, mode], expectedRole: "actor", allowedEvents: [])
}

private func childPID(_ hello: WireMessage) throws -> Int32 {
    struct Identity: Decodable { let pid: Int32 }
    return try hello.payload.decode(Identity.self).pid
}

@Test func computeClientPreservesRawRejectionsAndChecksReplyRunIdentity() async throws {
    let client = lifetimeClient("normal")
    let pid = try childPID(await client.start())
    let runID = UUID()
    let reply = try await client.request(kind: "reject", runID: runID, acceptingError: true)
    #expect(reply.kind == "error" && reply.runID == runID)
    guard case .object(let payload) = reply.payload else { Issue.record("Missing error payload"); return }
    #expect(payload["releasedFrames"] == .array([.object(["slotIndex": .integer(1), "lease": .unsigned(9_223_372_036_854_775_859)])]))
    await #expect(throws: AstraError.self) { try await client.request(kind: "reject", runID: runID) }
    _ = try await client.request(kind: "ping")
    await #expect(throws: AstraError.self) { try await client.request(kind: "wrongRun", runID: runID, acceptingError: true) }
    await client.shutdown()
    #expect(kill(pid, 0) == -1 && errno == ESRCH)
}

@Test func computeClientShutdownJoinsFinalizersAndCoalescesConcurrentStops() async throws {
    let client = lifetimeClient("normal")
    let pid = try childPID(await client.start())
    let start = ContinuousClock.now
    async let one: Void = client.shutdown()
    async let two: Void = client.shutdown()
    _ = await (one, two)
    #expect(start.duration(to: .now) >= .milliseconds(180))
    #expect(kill(pid, 0) == -1 && errno == ESRCH)
    await #expect(throws: AstraError.self) { try await client.start() }
    await client.shutdown()
}

@Test func computeClientKillsUnresponsiveChildOnlyAfterShutdownGrace() async throws {
    let client = lifetimeClient("ignore")
    let pid = try childPID(await client.start())
    let start = ContinuousClock.now
    await client.shutdown(grace: .milliseconds(20))
    #expect(start.duration(to: .now) >= .seconds(1))
    #expect(start.duration(to: .now) < .seconds(5))
    #expect(kill(pid, 0) == -1 && errno == ESRCH)
}

@Test func computeClientCanJoinWithAFullInputPipeAndCancelledRequest() async throws {
    let client = lifetimeClient("blocked")
    let pid = try childPID(await client.start())
    let request = Task {
        try await client.request(kind: "large", payload: .object(["text": .string(String(repeating: "x", count: 700_000))]))
    }
    try await Task.sleep(for: .milliseconds(80))
    request.cancel()
    await #expect(throws: (any Error).self) { try await request.value }
    await client.shutdown(grace: .milliseconds(20))
    #expect(kill(pid, 0) == -1 && errno == ESRCH)
}

@Test func computeClientShutdownBeforeStartPermanentlyClosesAdmission() async throws {
    let client = lifetimeClient("normal")
    await client.shutdown()
    await #expect(throws: AstraError.self) { try await client.start() }
}

private let unsettledControlFixture = #"""
import fcntl, json, os, pathlib, signal, sys, time
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
lease=(root/'desktop.lock').open('w'); fcntl.flock(lease, fcntl.LOCK_EX)
def stopping(number, frame):
    (root/'term-observed').write_text('cooperative')
signal.signal(signal.SIGTERM, stopping)
print(json.dumps(dict(version=1,kind='hello',sequence=0,payload=dict(role='control',protocolVersion=1,pid=os.getpid()))),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    (root/'request-entered').write_text(request['kind'])
    if request['kind']=='shutdown':
        response=dict(version=1,kind='error',sequence=1,requestID=request['requestID'],
            payload=dict(code='control.cleanupPending',message='Owned fixture hold is still pending',recoverable=True))
        if 'runID' in request:response['runID']=request['runID']
        print(json.dumps(response),flush=True)
    break
# Fail-safe bounds a broken test, not the production helper's cleanup policy.
deadline=time.monotonic()+8
while not (root/'release').exists() and time.monotonic()<deadline:time.sleep(.01)
(root/'settled').write_text('released')
fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
"""#

private actor ControlShutdownProbe {
    var finished = false
    func finish() { finished = true }
}

@Test(arguments: ["shutdown", "cancel", "timeout"])
func controlClientPreservesUnsettledHelperAndLeasePastComputeKillDeadline(mode: String) async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraControlLifetime-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? Data().write(to: root.appendingPathComponent("release")) }
    let client = ComputeProcess(executable: pythonURL(), arguments: ["-u", "-c", unsettledControlFixture, root.path, mode],
                                expectedRole: "control", allowedEvents: [])
    let pid = try childPID(await client.start())
    if mode != "shutdown" {
        let request = Task { try await client.request(kind: "execute", timeout: mode == "timeout" ? .milliseconds(100) : .seconds(10)) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !FileManager.default.fileExists(atPath: root.appendingPathComponent("request-entered").path), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("request-entered").path))
        if mode == "cancel" { request.cancel() }
        await #expect(throws: (any Error).self) { try await request.value }
    }
    let probe = ControlShutdownProbe()
    let closing = Task { await client.shutdown(grace: .milliseconds(20)); await probe.finish() }
    try await Task.sleep(for: .milliseconds(1300))
    #expect(kill(pid, 0) == 0)
    #expect(await !probe.finished)
    #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("term-observed").path))
    #expect(throws: AstraError.self) { _ = try DesktopControlLock(url: root.appendingPathComponent("desktop.lock")) }
    try Data().write(to: root.appendingPathComponent("release"))
    await closing.value
    #expect(await probe.finished)
    #expect(kill(pid, 0) == -1 && errno == ESRCH)
    #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("settled").path))
    let acquired = try DesktopControlLock(url: root.appendingPathComponent("desktop.lock"))
    withExtendedLifetime(acquired) {}
    try FileManager.default.removeItem(at: root)
}
