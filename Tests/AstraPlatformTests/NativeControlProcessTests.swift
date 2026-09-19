import Foundation
import Testing
import Darwin
import AstraCore
@testable import AstraPlatform

private func nativeFixturePython() -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent(".venv/bin/python")
}
private final class NativeChildProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Int32 = 0
    var pid: Int32 { lock.withLock { value } }
    func store(_ pid: Int32) { lock.withLock { value = pid } }
}
private let nativeTransportFixture = #"""
import json, os, pathlib, signal, sys, threading, time
root=pathlib.Path(sys.argv[1]); sequence=0; lock=threading.Lock(); alive=False; heartbeat=None
def send(kind,payload,request=None):
    global sequence
    with lock:
        value=dict(version=1,kind=kind,sequence=sequence,payload=payload); sequence+=1
        if request:
            for key in ('requestID','runID'):
                if key in request:value[key]=request[key]
        print(json.dumps(value),flush=True)
def interrupted(signum,frame):
    (root/'term').write_text('unexpected transport cancellation'); sys.exit(7)
signal.signal(signal.SIGTERM,interrupted)
def beat(request):
    (root/'heartbeat').touch(); deadline=time.monotonic()+.8
    while not (root/'release-heartbeat').exists() and time.monotonic()<deadline:time.sleep(.005)
    send('ack',{},request)
send('hello',dict(role='control',protocolVersion=1,initialPacketSequenceVersion=1,pid=os.getpid()))
for line in sys.stdin:
    request=json.loads(line); kind=request['kind']
    if kind=='arm':
        alive=True; send('ack',dict(armed=True,nextPacketSequence=request['payload'].get('initialPacketSequence',0)),request)
    elif kind=='heartbeat':
        heartbeat=threading.Thread(target=beat,args=(request,)); heartbeat.start()
    elif kind=='disarm':
        alive=False; send('ack',dict(stopped=True,cleanupSettled=True),request)
    elif kind=='execute':
        assert not alive, 'fixture only accepts the late rejection test path'
        packet=request['payload']; now=time.monotonic_ns()
        state=dict(keys=[],buttons=[],modifiers=0,pointer=dict(x=0,y=0),observedNanos=now,revision=0,valid=True)
        receipt=dict(packetID=packet['id'],runID=packet['runID'],sequence=packet['sequence'],status='rejected',observedNanos=now,commandResults=[],resultingState=state)
        send('control.receipt',dict(receipt=receipt),request)
        send('error',dict(code='control.session',message='Disarmed',recoverable=True),request)
    elif kind=='shutdown':
        send('ack',{},request); break
if heartbeat:heartbeat.join()
"""#

@Test func nativeControlRealTransportDrainsHeartbeatAndJoinsAfterDisarmedLateRejection() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraNativeControl-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let probe = NativeChildProbe(), owner = NativeControlOwner()
    let factory = NativeControlRuntimeFactory(protectsPhysicalInputs: false) { events, failure in
        let child = ComputeProcess(executable: nativeFixturePython(), arguments: ["-u", "-c", nativeTransportFixture, root.path],
            expectedRole: "control", allowedEvents: ["control.receipt", "control.stopped", "control.error"], onEvent: events, onFailure: failure)
        return .init(start: {
            let reply = try await child.start()
            struct Identity: Decodable { let pid: Int32 }
            probe.store(try reply.payload.decode(Identity.self).pid)
            return reply
        }, request: { kind, payload, run, timeout in
            try await child.request(kind: kind, payload: payload, runID: run, timeout: timeout, acceptingError: true)
        }, shutdown: { await child.shutdown(); return await child.terminationStatus() })
    }
    let config = try controlConfiguration(root: root, origin: 312)
    let session = NativeControlSession(configuration: config, owner: owner, runtimeFactory: factory)
    do {
        _ = try await session.start()
        try await controlWait { FileManager.default.fileExists(atPath: root.appendingPathComponent("heartbeat").path) }
        _ = try await session.disarm()
        #expect(kill(probe.pid, 0) == 0 && !owner.priorCleanupJoined)
        let result = try await session.rejectLatePacket(controlPacket(config))
        #expect(!result.admitted && result.reply.kind == "error")
        #expect(try await result.terminalReceipt().status == .rejected)
        let done = ControlTestFlag()
        let joining = Task { defer { done.set() }; return await session.shutdown() }
        try await Task.sleep(for: .milliseconds(15))
        #expect(!done.value && kill(probe.pid, 0) == 0)
        try Data().write(to: root.appendingPathComponent("release-heartbeat"))
        #expect(await joining.value.cleanupConfirmed && owner.priorCleanupJoined)
        #expect(kill(probe.pid, 0) == -1 && errno == ESRCH)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("term").path))
    } catch {
        try? Data().write(to: root.appendingPathComponent("release-heartbeat"))
        _ = await session.shutdown(); throw error
    }
}

private final class MappedControlFixture: @unchecked Sendable {
    let root: URL
    let guardian = Process()
    let closed = ControlTestFlag()
    private let lock = NSLock()
    private var ledger: ControlRecoveryLedger?
    init(root: URL) throws {
        self.root = root
        guardian.executableURL = nativeFixturePython()
        guardian.arguments = ["-u", "-c", #"""
import fcntl,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); lease=(root/'lease').open('w'); fcntl.flock(lease,fcntl.LOCK_EX)
(root/'guardian-ready').touch(); deadline=time.monotonic()+8
while not (root/'guardian-exit').exists() and time.monotonic()<deadline:time.sleep(.005)
lease.close()
"""#, root.path]
        guardian.standardInput = FileHandle.nullDevice; guardian.standardOutput = FileHandle.nullDevice; guardian.standardError = FileHandle.nullDevice
        try guardian.run()
    }
    var factory: NativeControlRuntimeFactory {
        .init(protectsPhysicalInputs: true) { _, _ in
            .init(start: {
                .init(kind: "hello", sequence: 0, payload: .object(["role": .string("control"), "protocolVersion": .integer(1), "recoveryVersion": .integer(1)]))
            }, request: { kind, payload, run, _ in
                var fields: [String: JSONValue] = [:]
                if kind == "arm" {
                    let arm = try payload.decode(ArmRequest.self)
                    let ledger = try ControlRecoveryLedger(open: #require(arm.recovery))
                    #expect(ledger.registerExecutor(pid: getpid()))
                    #expect(ledger.registerGuardian(pid: self.guardian.processIdentifier, now: MonotonicClock.now))
                    #expect(ledger.arm())
                    self.lock.withLock { self.ledger = ledger }
                    fields = ["armed": .bool(true), "nextPacketSequence": .integer(0),
                              "recoveryLedgerID": .string(ledger.descriptor.ledgerID.uuidString), "guardianPID": .integer(Int64(self.guardian.processIdentifier))]
                } else if kind == "disarm" {
                    if let ledger = self.lock.withLock({ self.ledger }), try !ledger.snapshot().cleanupConfirmed { ledger.stop(); #expect(ledger.settleLocally(now: MonotonicClock.now)) }
                    fields = ["stopped": .bool(true), "cleanupSettled": .bool(true)]
                }
                return .init(kind: "ack", sequence: 0, requestID: UUID(), runID: run, payload: .object(fields))
            }, shutdown: { self.closed.set(); return 0 })
        }
    }
    func finish() async {
        try? Data().write(to: root.appendingPathComponent("guardian-exit"))
        while guardian.isRunning { try? await Task.sleep(for: .milliseconds(5)) }
    }
}

@Test func nativeControlTerminalLedgerDoesNotReleaseOwnerWhileMatchingGuardianStillHoldsLease() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraNativeGuardian-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try MappedControlFixture(root: root), owner = NativeControlOwner()
    let session = NativeControlSession(configuration: try controlConfiguration(root: root), owner: owner, runtimeFactory: fixture.factory)
    do {
        try await controlWait { FileManager.default.fileExists(atPath: root.appendingPathComponent("guardian-ready").path) }
        _ = try await session.start()
        let done = ControlTestFlag()
        let joining = Task { defer { done.set() }; return await session.shutdown() }
        try await controlWait { fixture.closed.value }
        #expect(!done.value && !owner.priorCleanupJoined)
        #expect(throws: AstraError.self) { _ = try DesktopControlLock(url: root.appendingPathComponent("lease")) }
        await fixture.finish()
        #expect(await joining.value.cleanupConfirmed && owner.priorCleanupJoined)
        let lease = try DesktopControlLock(url: root.appendingPathComponent("lease"))
        withExtendedLifetime(lease) {}
    } catch { await fixture.finish(); _ = await session.shutdown(); throw error }
}
