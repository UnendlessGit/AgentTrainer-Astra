import Foundation
import Testing
import AstraCore
@testable import AgentTrainerAstra

@Test func resetShortcutPresetReleasesEveryModifierInReverseOrder() throws {
    let packet = ResetPacketPresets.keyPress(key: 15, modifiers: [55, 56], holdMS: 80)
    _ = try packet.validated()
    #expect(packet.commands.map(\.keyCode) == [55, 56, 15, 15, 56, 55])
    #expect(packet.commands.map(\.offsetMs) == [0, 0, 0, 80, 80, 80])
    let details = try #require(ResetPacketPresets.keyPress(packet))
    #expect(details.key == 15 && details.modifiers == [55, 56] && details.holdMS == 80)
    #expect(throws: AstraError.self) { _ = try ResetPacketPresets.keyPress(holdMS: Int.max).validated() }
}

@Test func resetClickPresetBindsOnlyItsAuthoredSurfaceAndRejectsTheFarEdge() throws {
    var packet = ResetPacketPresets.click(surfaceID: "authored")
    #expect(ResetPacketPresets.isClick(packet))
    _ = try packet.validated()
    let surface = SurfaceDescriptor(id: "different", globalBounds: .init(x: 0, y: 0, width: 10, height: 10), pixelWidth: 10, pixelHeight: 10)
    let context = try ResetContext(nextEpisodeID: UUID(), environmentID: UUID(), scope: .init(surfaces: [surface], wholeDesktop: true))
    #expect(throws: AstraError.self) { _ = try packet.packet(context: context, observationID: UUID(), sequence: 0, executeAtNanos: 0) }
    packet.commands[0].x = 1
    #expect(throws: AstraError.self) { _ = try packet.validated() }
}
