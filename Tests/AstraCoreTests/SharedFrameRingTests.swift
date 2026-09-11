import Foundation
import Darwin
import Testing
@testable import AstraCore

private func ringTestDirectory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}
private func ringMetadata() -> FrameMetadata {
    .init(eventNanos: 17, observedNanos: 31,
          surface: .init(id: "window", globalBounds: .init(x: -20, y: 30, width: 4, height: 4), pixelWidth: 8, pixelHeight: 8),
          byteCount: 256)
}

@Test func frameRingRequiresExactLeaseBeforeReuseAndClose() throws {
    let root = try ringTestDirectory(); defer { try? FileManager.default.removeItem(at: root) }
    let ring = try SharedFrameRing(url: root.appendingPathComponent("frames"), runID: UUID(), slotCount: 1, slotCapacity: 256)
    let first = try ring.publish(pixels: Data(repeating: 7, count: 256), metadata: ringMetadata())
    #expect(first.metadata.codec == "raw")
    #expect(ring.pendingLeaseCount == 1)
    #expect(throws: AstraError.self) { try ring.publish(pixels: Data(repeating: 19, count: 256), metadata: ringMetadata()) }
    #expect(throws: AstraError.self) { try ring.close() }
    let wrong: [SharedFrameAcknowledgement] = [
        .init(version: 2, runID: first.runID, ringID: first.ringID, slot: first.slot, leaseID: first.leaseID, sequence: first.sequence),
        .init(runID: UUID(), ringID: first.ringID, slot: first.slot, leaseID: first.leaseID, sequence: first.sequence),
        .init(runID: first.runID, ringID: UUID(), slot: first.slot, leaseID: first.leaseID, sequence: first.sequence),
        .init(runID: first.runID, ringID: first.ringID, slot: 1, leaseID: first.leaseID, sequence: first.sequence),
        .init(runID: first.runID, ringID: first.ringID, slot: first.slot, leaseID: UUID(), sequence: first.sequence),
        .init(runID: first.runID, ringID: first.ringID, slot: first.slot, leaseID: first.leaseID, sequence: first.sequence + 1)
    ]
    for acknowledgement in wrong { #expect(throws: AstraError.self) { try ring.release(acknowledgement) } }
    #expect(ring.pendingLeaseCount == 1)
    #expect(try Data(contentsOf: ring.url).subdata(in: first.offset..<(first.offset + first.size)) == Data(repeating: 7, count: 256))
    try ring.release(first.acknowledgement)
    let second = try ring.publish(pixels: Data(repeating: 19, count: 256), metadata: ringMetadata())
    #expect(second.slot == first.slot && second.leaseID != first.leaseID && second.sequence > first.sequence)
    #expect(throws: AstraError.self) { try ring.release(first.acknowledgement) }
    #expect(ring.pendingLeaseCount == 1)
    try ring.release(second.acknowledgement)
    #expect(throws: AstraError.self) { try ring.release(second.acknowledgement) }
    try ring.close(); try ring.close()
    #expect(!FileManager.default.fileExists(atPath: ring.url.path))
    #expect(throws: AstraError.self) { try ring.publish(pixels: Data(repeating: 1, count: 256), metadata: ringMetadata()) }
}

@Test func frameRingRejectsOversizeBuffersAndNeverOverwritesAnExistingFile() throws {
    let root = try ringTestDirectory(); defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("frames")
    let ring = try SharedFrameRing(url: url, runID: UUID(), slotCount: 2, slotCapacity: 128)
    #expect(throws: AstraError.self) { try ring.publish(pixels: Data(repeating: 1, count: 256), metadata: ringMetadata()) }
    #expect(ring.pendingLeaseCount == 0)
    #expect(throws: AstraError.self) { try SharedFrameRing(url: url, runID: UUID(), slotCapacity: 128) }
    let linked = root.appendingPathComponent("linked")
    try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: url)
    #expect(throws: AstraError.self) { try SharedFrameRing(url: linked, runID: UUID(), slotCapacity: 128) }
    let permissions = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? NSNumber
    #expect(permissions?.intValue == 0o600)
    var information = stat()
    #expect(stat(url.path, &information) == 0)
    #expect(information.st_size == off_t(ring.byteCount))
    // The small fixture must already occupy real storage before publication;
    // sparse ftruncate alone can pass size checks but fails this requirement.
    #expect(information.st_blocks * 512 >= Int64(ring.byteCount))
    #expect(throws: AstraError.self) { try SharedFrameRing(url: root.appendingPathComponent("huge"), runID: UUID(), slotCount: 16, slotCapacity: FrameArchive.maximumFrameBytes) }
    try ring.close()
}

@Test func frameRingConsumerExitRetiresTheInodeWithoutMutatingOutstandingData() throws {
    let root = try ringTestDirectory(); defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("frames")
    let ring = try SharedFrameRing(url: url, runID: UUID(), slotCount: 1, slotCapacity: 256)
    let reference = try ring.publish(pixels: Data(repeating: 79, count: 256), metadata: ringMetadata())
    let retained = try FileHandle(forReadingFrom: url)
    defer { try? retained.close() }
    let original = try retained.readToEnd()
    ring.closeAfterConsumerExit()
    #expect(!FileManager.default.fileExists(atPath: url.path))
    #expect(ring.pendingLeaseCount == 0)
    #expect(throws: AstraError.self) { try ring.release(reference.acknowledgement) }
    #expect(throws: AstraError.self) { try ring.publish(pixels: Data(repeating: 3, count: 256), metadata: ringMetadata()) }

    let replacement = try SharedFrameRing(url: url, runID: UUID(), slotCount: 1, slotCapacity: 256)
    let latest = try replacement.publish(pixels: Data(repeating: 3, count: 256), metadata: ringMetadata())
    try retained.seek(toOffset: 0)
    #expect(try retained.readToEnd() == original)
    #expect(replacement.ringID != ring.ringID)
    #expect(throws: AstraError.self) { try replacement.release(reference.acknowledgement) }
    try replacement.release(latest.acknowledgement)
    try replacement.close()
}

@Test func concurrentFramePublishersCannotClaimTheSameSlot() async throws {
    let root = try ringTestDirectory(); defer { try? FileManager.default.removeItem(at: root) }
    let ring = try SharedFrameRing(url: root.appendingPathComponent("frames"), runID: UUID(), slotCount: 4, slotCapacity: 256)
    let references = await withTaskGroup(of: SharedFrameReference?.self) { group in
        for index in 0..<24 {
            group.addTask { try? ring.publish(pixels: Data(repeating: UInt8(index), count: 256), metadata: ringMetadata()) }
        }
        var values: [SharedFrameReference] = []
        for await value in group { if let value { values.append(value) } }
        return values
    }
    #expect(references.count == 4)
    #expect(Set(references.map(\.slot)).count == 4)
    #expect(Set(references.map(\.sequence)).count == 4)
    for reference in references { try ring.release(reference.acknowledgement) }
    try ring.close()
}

@Test func frameRingRetirementNeverReusesOutstandingSlotsOrDeletesAReplacementPath() throws {
    let root = try ringTestDirectory(); defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("frames")
    let ring = try SharedFrameRing(url: url, runID: UUID(), slotCount: 1, slotCapacity: 256)
    let reference = try ring.publish(pixels: Data(repeating: 31, count: 256), metadata: ringMetadata())
    try FileManager.default.moveItem(at: url, to: root.appendingPathComponent("moved-ring"))
    let replacement = Data("An unrelated replacement file".utf8)
    try replacement.write(to: url)
    ring.closeAfterConsumerExit()
    #expect(try Data(contentsOf: url) == replacement)
    #expect(throws: AstraError.self) { try ring.release(reference.acknowledgement) }
    #expect(throws: AstraError.self) { try ring.publish(pixels: Data(repeating: 2, count: 256), metadata: ringMetadata()) }
}
