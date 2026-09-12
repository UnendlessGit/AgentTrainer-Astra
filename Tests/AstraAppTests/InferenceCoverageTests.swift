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
