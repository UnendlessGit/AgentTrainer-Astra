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

@Test func freshControlLeaseAdmitsPersistentActorSequenceWithoutRewritingPacket() throws {
    let surface = SurfaceDescriptor(id: "target", globalBounds: .init(x: 0, y: 0, width: 100, height: 100), pixelWidth: 100, pixelHeight: 100)
    let request = ArmRequest(runID: UUID(), scope: .init(surfaces: [surface]), capabilities: .init(keyCodes: [13]), initialPacketSequence: 917)
    var lease = ControlLease(); try lease.arm(request, now: 1)
    let packet = ActionPacket(runID: request.runID, sequence: 917, observationID: UUID(), geometryRevision: 0,
                              executeAtNanos: 10_000, durationMs: 100, commands: [.init(offsetMs: 0, operation: .keyDown, keyCode: 13)])
    var stale = packet; stale.sequence = 0
    #expect(throws: AstraError.self) { try lease.admit(stale, now: 2) }
    try lease.admit(packet, now: 2)
    #expect(lease.nextSequence == 918)
    #expect(packet.sequence == 917)
    var exhausted = request; exhausted.initialPacketSequence = UInt64.max
    lease.disarm()
    #expect(throws: AstraError.self) { try lease.arm(exhausted, now: 3) }
    var old = try JSONEncoder().encode(request)
    var json = try #require(try JSONSerialization.jsonObject(with: old) as? [String: Any])
    json.removeValue(forKey: "initialPacketSequence"); old = try JSONSerialization.data(withJSONObject: json)
    let decoded = try JSONDecoder().decode(ArmRequest.self, from: old)
    try lease.arm(decoded, now: 3)
    #expect(lease.nextSequence == 0)
}
