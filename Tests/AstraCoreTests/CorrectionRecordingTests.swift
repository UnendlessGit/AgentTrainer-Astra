import Foundation
import Testing
@testable import AstraCore

@Test func correctionArchiveSeparatesOriginalAgentHistoryFromExpertRecording() throws {
    let exported = ProcessInfo.processInfo.environment["ASTRA_CORRECTION_FIXTURE_ROOT"]
    let root = exported.map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { if exported == nil { try? FileManager.default.removeItem(at: root) } }
    let manifest = RecordingManifest(name: "Explicit correction", environment: .init(name: "Generated source", kind: .practice))
    let directory = root.appendingPathComponent(manifest.id.uuidString + ".astrarecord")
    let writer = try RecordingWriter(directory: directory, manifest: manifest)
    let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 16, height: 16), pixelWidth: 16, pixelHeight: 16)
    let pixels = Data(repeating: 18, count: 1024)
    let before = FrameMetadata(eventNanos: 999_990_000, observedNanos: 1_000_000_000, surface: surface, byteCount: pixels.count, codec: "raw")
    var controls = ControlState(); controls.valid = true; controls.observedNanos = 1_000_000_000
    let actorInput: JSONValue = .object(["observationID": .string(UUID().uuidString.lowercased()),
        "episodeID": .string(UUID().uuidString.lowercased()), "previousStateID": .string(UUID().uuidString.lowercased()),
        "cutoffNanos": .unsigned(1_000_000_000), "geometryRevision": .unsigned(0),
        "controlState": try .encode(controls), "executedEvents": .array([]), "intervalCovered": .bool(true), "contextIDs": .array([])])
    let seed = CorrectionRecordingSeed(sourceRunID: UUID(), sourceCheckpointID: UUID(), sourcePolicySignature: String(repeating: "a", count: 64),
        contextIDs: [], requestedAtNanos: 1_400_000_000, controlJoinedAtNanos: 1_500_000_000,
        observations: [.init(actorInput: actorInput, frames: [.init(metadata: before, pixels: pixels, coverage: nil)])])
    try writer.attachCorrection(seed, supervisionStartNanos: 1_600_000_000)
    for time in stride(from: UInt64(2_000_000_000), through: 4_000_000_000, by: 100_000_000) {
        try writer.append(FrameArchive.prepare(pixels: pixels, metadata: .init(eventNanos: time - 10_000,
            observedNanos: time, surface: surface, byteCount: pixels.count)))
    }
    _ = try writer.finish(at: 4_100_000_000, status: .complete)
    let reader = try RecordingReader(directory: directory), reference = try #require(reader.manifest.correction)
    let prelude = try CorrectionPrelude.load(in: directory, reference: reference)
    #expect(!prelude.continuityProven && prelude.sourceCheckpointID == seed.sourceCheckpointID)
    #expect(prelude.observations[0].actorInput == (try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(actorInput))))
    #expect(prelude.observations[0].frames[0].block.metadata.id == before.id)
    #expect(try prelude.pixels(for: prelude.observations[0].frames[0], in: directory) == pixels)
    #expect(try reader.inspect().firstFrameNanos == 2_000_000_000)
    #expect(try reader.preview(at: 1_500_000_000) == nil)
    #expect(reader.manifest.frameCount == 21 && reader.manifest.eventCount == 0)
    if exported == nil {
        try Data("{}".utf8).write(to: directory.appendingPathComponent(reference.path))
        #expect(throws: AstraError.self) { try CorrectionPrelude.load(in: directory, reference: reference) }
    }
}
