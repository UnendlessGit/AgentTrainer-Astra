import Foundation
import Testing
@testable import AstraCore

@Test func recordingPreviewUsesSealedIdentityChecksAndCausalFrameSelection() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let manifest = RecordingManifest(name: "Inspectable", environment: .init(name: "Fixture", kind: .practice))
    let directory = root.appendingPathComponent(manifest.id.uuidString + ".astrarecord")
    let writer = try RecordingWriter(directory: directory, manifest: manifest)
    #expect(throws: AstraError.self) { try RecordingReader(directory: directory) }
    let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: -10, y: 0, width: 16, height: 16), pixelWidth: 32, pixelHeight: 32)
    let pixels = Data(repeating: 128, count: 32 * 32 * 4)
    let frame = try FrameArchive.prepare(pixels: pixels, metadata: .init(eventNanos: 90, observedNanos: 100, surface: surface, byteCount: pixels.count))
    try writer.append(frame)
    let event = RawInputEvent(sequence: 0, eventNanos: 95, observedNanos: 110, origin: .physical, kind: .keyDown, keyCode: 13)
    try writer.append(events: [event])
    _ = try writer.finish(at: 200, status: .complete)
    let reader = try RecordingReader(directory: directory)
    #expect(try reader.inspect().firstFrameNanos == 100)
    #expect(try reader.preview(at: 99) == nil)
    let preview = try #require(try reader.preview(at: 110, eventRadiusNanos: 10))
    #expect(preview.pixels == pixels && preview.frame.id == frame.metadata.id)
    #expect(preview.events == [event] && !preview.moreEvents)
    // A corrupted payload cannot be displayed as a successful frame.
    let file = try FileHandle(forUpdating: directory.appendingPathComponent("frames-00000.astraframes"))
    try file.seek(toOffset: 0); try file.write(contentsOf: Data([0])); try file.close()
    #expect(throws: AstraError.self) { try reader.preview(at: 110) }
}
