import Foundation
import Testing
import AstraCore
import AstraPlatform
@testable import AgentTrainerAstra

@Test func inferenceRetainsCoverageWithoutReplacingSourcePixels() throws {
    let frame = FrameMetadata(eventNanos: 100, observedNanos: 110,
        surface: SurfaceDescriptor(id: "surface", globalBounds: .init(x: 0, y: 0, width: 32, height: 32),
                                   pixelWidth: 32, pixelHeight: 32), byteCount: 4096, codec: "raw")
    let stream = UUID(), initial = try CaptureFrameCoverage(streamID: stream, frame: frame)
    let inbox = InferenceFrameInbox()
    inbox.receive(.init(metadata: frame, pixels: { Data(repeating: 7, count: 4096) }, coverage: initial))
    inbox.health(.idle)
    #expect(try inbox.read()?.coverage?.throughNanos == 110)
    let advanced = try initial.verifyingUnchanged(streamID: stream, surface: frame.surface, throughNanos: 200, verifiedAtNanos: 210)
    inbox.health(.coverage(advanced))
    let selected = try #require(try inbox.read())
    #expect(selected.coverage == advanced)
    #expect(selected.metadata == frame)
    #expect(try selected.pixels() == Data(repeating: 7, count: 4096))
    inbox.health(.unavailable("source suspended"))
    inbox.health(.coverage(advanced))
    #expect(throws: AstraError.self) { try inbox.read() }
}

@Test func inferenceRejectsCoverageForReplacedSourceFrame() throws {
    let frame = FrameMetadata(eventNanos: 100, observedNanos: 110,
        surface: SurfaceDescriptor(id: "surface", globalBounds: .init(x: 0, y: 0, width: 32, height: 32),
                                   pixelWidth: 32, pixelHeight: 32), byteCount: 4096, codec: "raw")
    let initial = try CaptureFrameCoverage(streamID: UUID(), frame: frame)
    let newer = FrameMetadata(eventNanos: 200, observedNanos: 210, surface: frame.surface, byteCount: 4096, codec: "raw")
    let inbox = InferenceFrameInbox()
    inbox.receive(.init(metadata: newer, pixels: { Data(repeating: 7, count: 4096) }))
    inbox.health(.coverage(initial))
    #expect(throws: AstraError.self) { try inbox.read() }
}

@Test func sourceGroupWaitsForAllRolesAndAdvancesCoverageIndependently() throws {
    let inbox = InferenceFrameInbox(sourceIDs: ["left", "right"])
    func image(_ id: String, observed: UInt64, value: UInt8) throws -> InferenceImage {
        let metadata = FrameMetadata(eventNanos: observed - 1, observedNanos: observed,
            surface: SurfaceDescriptor(id: id, globalBounds: .init(x: 0, y: 0, width: 2, height: 2),
                pixelWidth: 2, pixelHeight: 2), byteCount: 16, codec: "raw")
        return .init(metadata: metadata, pixels: { Data(repeating: value, count: 16) },
                     coverage: try CaptureFrameCoverage(streamID: UUID(), frame: metadata))
    }
    let left = try image("left", observed: 110, value: 1), right = try image("right", observed: 100, value: 2)
    inbox.receive(left)
    #expect(try inbox.readAll() == nil)
    inbox.receive(right) // Independent arrival clocks must not be compared across sources.
    let before = try #require(try inbox.readAll())
    #expect(before.map(\.metadata.id) == [left.metadata.id, right.metadata.id])
    #expect(throws: AstraError.self) { try inbox.read() }
    let initial = try #require(right.coverage)
    let continued = try initial.verifyingUnchanged(streamID: initial.streamID, surface: right.metadata.surface,
        throughNanos: 200, verifiedAtNanos: 210)
    inbox.health(.coverage(continued))
    let after = try #require(try inbox.readAll())
    #expect(after[0].coverage == left.coverage)
    #expect(after[1].coverage == continued)
    #expect(after.map(\.metadata.id) == before.map(\.metadata.id))
    #expect(try after[1].pixels() == Data(repeating: 2, count: 16))
    inbox.receive(try image("foreign", observed: 220, value: 3))
    #expect(throws: AstraError.self) { try inbox.readAll() }
}
