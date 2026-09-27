import Foundation
import Testing
@testable import AstraCore

@Test func recordingControlExclusionRequiresAJoinedBoundedIntervalAndRecoveryDoesNotInventOne() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let initial = RecordingManifest(name: "Exclusive recording", environment: .init(name: "Generated", kind: .practice))
    let directory = root.appendingPathComponent(initial.id.uuidString + ".astrarecord")
    let writer = try RecordingWriter(directory: directory, manifest: initial)
    let proof = RecordingControlExclusion(recordingID: initial.id, startedNanos: 100)
    try writer.beginControlExclusion(proof)
    #expect(!writer.snapshot.controlExclusion!.covers(150))
    let surface = SurfaceDescriptor(id: "generated", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
    try writer.append(FrameArchive.prepare(pixels: Data(repeating: 3, count: 16),
        metadata: .init(eventNanos: 105, observedNanos: 110, surface: surface, byteCount: 16)))
    #expect(throws: AstraError.self) { try writer.sealControlExclusion(ownershipID: UUID(), through: 200, producersJoined: 210) }
    try writer.sealControlExclusion(ownershipID: proof.ownershipID, through: 200, producersJoined: 210)
    let completed = try writer.finish(at: 200, status: .complete)
    let saved = try #require(try RecordingReader(directory: directory).manifest.controlExclusion)
    #expect(saved == completed.controlExclusion)
    #expect(saved.covers(100) && saved.covers(200) && !saved.covers(99) && !saved.covers(201))
    // Model a crash after the writer's intermediate flush but before its
    // terminal manifest publication. Recovery must not upgrade that tail.
    var interrupted = completed; interrupted.status = .recording; interrupted.stoppedNanos = nil
    let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .millisecondsSince1970
    try encoder.encode(interrupted).write(to: directory.appendingPathComponent("manifest.json"))
    let recovered = try RecordingRecovery.recover(directory: directory, expectedID: initial.id, fallback: nil)
    #expect(recovered.recovered)
    #expect(recovered.manifest.controlExclusion?.ownershipID == proof.ownershipID)
    #expect(recovered.manifest.controlExclusion?.throughNanos == nil)
    #expect(recovered.manifest.controlExclusion?.covers(150) == false)
    var inconsistent = saved; inconsistent.producersJoinedNanos = 199
    #expect(throws: AstraError.self) { try inconsistent.validated(recordingID: initial.id) }
}

@Test(arguments: [false, true])
func recordingWriterRejectsPrepopulatedExclusionBeforeCreatingPackage(sealed: Bool) throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    var manifest = RecordingManifest(name: "Fresh recording", environment: .init(name: "Generated", kind: .practice))
    var proof = RecordingControlExclusion(recordingID: manifest.id, startedNanos: 100)
    if sealed { proof.throughNanos = 200; proof.producersJoinedNanos = 210 }
    manifest.controlExclusion = proof
    _ = try manifest.validated() // A readable recovery manifest is not a new writer's admission proof.
    let directory = root.appendingPathComponent(manifest.id.uuidString + ".astrarecord")
    #expect(throws: AstraError.self) { try RecordingWriter(directory: directory, manifest: manifest) }
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}
