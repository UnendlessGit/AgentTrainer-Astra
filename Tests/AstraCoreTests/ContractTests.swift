import Foundation
import Testing
@testable import AstraCore

@Test func protocolPreservesClockIntegersAcrossFragments() throws {
    let clock = UInt64.max - 17
    let message = WireMessage(kind: "heartbeat", sequence: 9,
                              payload: .object(["clock": .unsigned(clock)]))
    let encoded = try message.framed()
    var framer = MessageFramer()
    let middle = encoded.count / 2
    #expect(try framer.append(encoded.prefix(middle)).isEmpty)
    let decoded = try framer.append(encoded.suffix(from: middle))
    #expect(decoded == [message])
    try framer.finish()
}

@Test func malformedTransportCannotRecoverIntoAnAction() throws {
    var framer = MessageFramer()
    #expect(throws: (any Error).self) { try framer.append(Data("broken\n".utf8)) }
    let message = WireMessage(kind: "execute", sequence: 0)
    #expect(throws: (any Error).self) { try framer.append(message.framed()) }
    var truncated = MessageFramer()
    #expect(try truncated.append(Data("{".utf8)).isEmpty)
    #expect(throws: (any Error).self) { try truncated.finish() }
}

@Test func geometryMapsMixedScaleNegativeOriginsWithoutAssumingRetina() throws {
    let surface = SurfaceDescriptor(id: "secondary", globalBounds: Rect2D(x: -1500, y: 100, width: 1500, height: 900),
                                    pixelWidth: 3000, pixelHeight: 1800,
                                    contentBounds: Rect2D(x: 20, y: 10, width: 2960, height: 1780))
    let global = try surface.globalBounds.globalPoint(normalized: Point2D(x: 0.25, y: 0.75))
    #expect(global == Point2D(x: -1125, y: 775))
    #expect(try surface.pixelPoint(global: global) == Point2D(x: 760, y: 1345))
    #expect(throws: AstraError.self) { try surface.pixelPoint(global: Point2D(x: 1, y: 200)) }
}

private func surface() -> SurfaceDescriptor {
    SurfaceDescriptor(id: "primary", globalBounds: Rect2D(x: 0, y: 0, width: 1000, height: 600),
                      pixelWidth: 2000, pixelHeight: 1200)
}
private func packet(_ commands: [TimedCommand]) -> ActionPacket {
    ActionPacket(runID: UUID(), sequence: 1, observationID: UUID(), geometryRevision: 0,
                 executeAtNanos: 1000, durationMs: 100, commands: commands)
}

@Test func timedPacketsPreserveTapsAndEnforceCapabilitiesBeforeAdmission() throws {
    let actions = ActionCapabilities(keyCodes: [13], mouseButtons: [0], absolutePointer: true)
    let tap = packet([.init(offsetMs: 1, operation: .keyDown, keyCode: 13),
                      .init(offsetMs: 3, operation: .keyUp, keyCode: 13),
                      .init(offsetMs: 3, operation: .keyDown, keyCode: 13),
                      .init(offsetMs: 8, operation: .keyUp, keyCode: 13)])
    #expect(try tap.validated(capabilities: actions, surfaces: [surface()], capacity: 16) == tap)
    #expect(throws: AstraError.self) {
        try tap.validated(capabilities: .init(keyCodes: [12]), surfaces: [surface()], capacity: 16)
    }
    var reversed = tap
    reversed.commands.reverse()
    #expect(throws: AstraError.self) {
        try reversed.validated(capabilities: actions, surfaces: [surface()], capacity: 16)
    }
    let ambiguous = packet([.init(offsetMs: 0, operation: .keyDown, keyCode: 13, x: 0.5)])
    #expect(throws: AstraError.self) {
        try ambiguous.validated(capabilities: actions, surfaces: [surface()], capacity: 16)
    }
}

@Test func motionEndpointAndGeometryRevisionAreExplicit() throws {
    let actions = ActionCapabilities(absolutePointer: true, relativePointer: true)
    let path = packet([.init(offsetMs: 0, operation: .pointerAbsolute, surfaceID: "primary", x: 0, y: 0),
                       .init(offsetMs: 100, operation: .pointerAbsolute, surfaceID: "primary", x: 0.9, y: 0.9)])
    #expect(try path.validated(capabilities: actions, surfaces: [surface()], capacity: 16) == path)
    let outside = packet([.init(offsetMs: 100, operation: .pointerAbsolute, surfaceID: "primary", x: 1, y: 1)])
    #expect(throws: AstraError.self) { try outside.validated(capabilities: actions, surfaces: [surface()], capacity: 16) }
    var changed = surface(); changed.geometryRevision = 1
    #expect(throws: AstraError.self) { try path.validated(capabilities: actions, surfaces: [changed], capacity: 16) }
    let mixed = packet([.init(offsetMs: 0, operation: .pointerAbsolute, surfaceID: "primary", x: 0, y: 0),
                        .init(offsetMs: 1, operation: .pointerRelative, dx: 1, dy: 1)])
    #expect(throws: AstraError.self) { try mixed.validated(capabilities: actions, surfaces: [surface()], capacity: 16) }
}
