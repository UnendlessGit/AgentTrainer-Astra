import Foundation
import Testing
@testable import AstraCore

@Test func expiredLeaseCannotBeRevivedByLateHeartbeatOrQueuedAction() throws {
    var lease = ControlLease()
    let surface = SurfaceDescriptor(id: "target", globalBounds: .init(x: 0, y: 0, width: 100, height: 100),
                                    pixelWidth: 100, pixelHeight: 100)
    let request = ArmRequest(runID: UUID(), scope: .init(surfaces: [surface]), capabilities: .init(keyCodes: [13]))
    try lease.arm(request, now: 100)
    try lease.heartbeat(runID: request.runID, now: 200)
    #expect(throws: AstraError.self) {
        try lease.heartbeat(runID: request.runID, now: 200 + ControlLease.durationNanos)
    }
    let packet = ActionPacket(runID: request.runID, sequence: 0, observationID: UUID(), geometryRevision: 0,
                              executeAtNanos: 900_000_000, durationMs: 100,
                              commands: [.init(offsetMs: 0, operation: .keyDown, keyCode: 13)])
    #expect(throws: AstraError.self) { try lease.admit(packet, now: 800_000_000) }
    lease.disarm()
    try lease.arm(request, now: 900_000_000)
    #expect(lease.nextSequence == 0)
}

@Test func packetAdmissionRejectsDuplicatesAndOldSessionIdentity() throws {
    var lease = ControlLease()
    let surface = SurfaceDescriptor(id: "target", globalBounds: .init(x: 0, y: 0, width: 100, height: 100),
                                    pixelWidth: 100, pixelHeight: 100)
    let request = ArmRequest(runID: UUID(), scope: .init(surfaces: [surface]), capabilities: .init(keyCodes: [13]))
    try lease.arm(request, now: 0)
    var packet = ActionPacket(runID: request.runID, sequence: 0, observationID: UUID(), geometryRevision: 0,
                              executeAtNanos: 10_000, durationMs: 100, commands: [])
    try lease.admit(packet, now: 1)
    #expect(throws: AstraError.self) { try lease.admit(packet, now: 2) }
    packet.sequence = 1; packet.runID = UUID()
    #expect(throws: AstraError.self) { try lease.admit(packet, now: 3) }
    #expect(lease.nextSequence == 1)
}

@Test func protocolHandlesMultipleFramesAndSizeBoundary() throws {
    let first = WireMessage(kind: "ping", sequence: 0)
    let second = WireMessage(kind: "ping", sequence: 1)
    var framer = MessageFramer()
    #expect(try framer.append(first.framed() + second.framed()) == [first, second])
    var over = MessageFramer()
    #expect(throws: AstraError.self) {
        try over.append(Data(repeating: 120, count: AstraVersion.maximumMessageBytes))
    }
}
