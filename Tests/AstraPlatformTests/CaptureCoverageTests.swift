import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

private func coverageFrame(event: UInt64 = 100_000_000, observed: UInt64 = 110_000_000) -> FrameMetadata {
    FrameMetadata(eventNanos: event, observedNanos: observed,
        surface: SurfaceDescriptor(id: "window:7", globalBounds: .init(x: 0, y: 0, width: 32, height: 32),
                                   pixelWidth: 32, pixelHeight: 32), byteCount: 4096, codec: "raw")
}

@Test func unchangedCoveragePreservesOriginalTimesAndExpiresFromSourceTime() throws {
    let frame = coverageFrame(), stream = UUID()
    let initial = try CaptureFrameCoverage(streamID: stream, frame: frame)
    #expect(throws: AstraError.self) { try initial.validated(frame: frame, cutoffNanos: 1_000_000_000, maximumAgeMS: 250) }
    let unchanged = try initial.verifyingUnchanged(streamID: stream, surface: frame.surface,
        throughNanos: 900_000_000, verifiedAtNanos: 1_000_000_000)
    #expect(unchanged.frameID == frame.id)
    #expect(unchanged.eventNanos == frame.eventNanos)
    #expect(unchanged.observedNanos == frame.observedNanos)
    try unchanged.validated(frame: frame, cutoffNanos: 1_100_000_000, maximumAgeMS: 250)
    // The callback arrived late: age is measured from the source event, not
    // from its more recent callback availability.
    #expect(throws: AstraError.self) { try unchanged.validated(frame: frame, cutoffNanos: 1_200_000_000, maximumAgeMS: 250) }
    #expect(throws: AstraError.self) { try unchanged.validated(frame: frame, cutoffNanos: 950_000_000) }
}

@Test func unchangedCoverageCannotCrossFrameStreamOrGeometry() throws {
    let frame = coverageFrame(), stream = UUID()
    let initial = try CaptureFrameCoverage(streamID: stream, frame: frame)
    #expect(throws: AstraError.self) {
        try initial.verifyingUnchanged(streamID: UUID(), surface: frame.surface, throughNanos: 200_000_000, verifiedAtNanos: 210_000_000)
    }
    var moved = frame.surface; moved.globalBounds.x += 1; moved.geometryRevision += 1
    #expect(throws: AstraError.self) {
        try initial.verifyingUnchanged(streamID: stream, surface: moved, throughNanos: 200_000_000, verifiedAtNanos: 210_000_000)
    }
    let next = coverageFrame(event: 200_000_000, observed: 210_000_000)
    #expect(throws: AstraError.self) { try initial.validated(frame: next, cutoffNanos: 300_000_000) }
}

@Test func unchangedCoverageRejectsUnavailableAndRegressingEvidence() throws {
    let frame = coverageFrame(), stream = UUID()
    let initial = try CaptureFrameCoverage(streamID: stream, frame: frame)
    for (source, available): (UInt64, UInt64) in [(109_000_000, 200_000_000), (200_000_000, 199_000_000)] {
        #expect(throws: AstraError.self) {
            try initial.verifyingUnchanged(streamID: stream, surface: frame.surface, throughNanos: source, verifiedAtNanos: available)
        }
    }
    let advanced = try initial.verifyingUnchanged(streamID: stream, surface: frame.surface,
        throughNanos: 200_000_000, verifiedAtNanos: 210_000_000)
    #expect(throws: AstraError.self) {
        try advanced.verifyingUnchanged(streamID: stream, surface: frame.surface, throughNanos: 200_000_000, verifiedAtNanos: 209_000_000)
    }
}
