import Foundation
import Testing
@testable import AstraCore

@Test func recordingCoverageCommitsAfterOriginalFrameWithoutRewritingPixelsOrTime() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("astra-coverage-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: directory) }
    let manifest = RecordingManifest(name: "Coverage", environment: .init(name: "Display", kind: .display), surfaceIDs: ["display:1"])
    let writer = try RecordingWriter(directory: directory, manifest: manifest)
    let surface = SurfaceDescriptor(id: "display:1", globalBounds: .init(x: 0, y: 0, width: 1, height: 1),
        pixelWidth: 1, pixelHeight: 1, geometryRevision: 7, nativeDisplayID: 1)
    let frame = FrameMetadata(eventNanos: 90, observedNanos: 100, surface: surface, byteCount: 4, codec: "raw")
    let base = try CaptureFrameCoverage(streamID: UUID(), frame: frame)
    let proof = try base.verifyingUnchanged(streamID: base.streamID, surface: surface, throughNanos: 300, verifiedAtNanos: 310)
    try writer.append(coverage: proof)
    try writer.flush()
    let reader = try SQLiteDatabase(url: directory.appendingPathComponent("index.sqlite"), readOnly: true)
    #expect(try reader.query("SELECT proof FROM coverage").isEmpty)
    try writer.append(FrameArchive.prepare(pixels: Data([1, 2, 3, 4]), metadata: frame))
    try writer.flush()
    let rows = try reader.query("SELECT observed,frame_id,proof FROM coverage")
    let row = try #require(rows.first)
    #expect(row["observed"]?.integer == 310)
    let restored = try JSONDecoder().decode(CaptureFrameCoverage.self, from: #require(row["proof"]?.data))
    #expect(restored == proof)
    #expect(restored.eventNanos == 90 && restored.observedNanos == 100 && restored.surface.geometryRevision == 7)
    try reader.close()
    let sealed = try writer.finish(at: 400, status: .complete)
    #expect(sealed.frameCount == 1 && sealed.firstObservedNanos == 100 && sealed.surfaceIDs == ["display:1"])
}
