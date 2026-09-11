import Foundation
import CryptoKit
import AstraCore

private struct RingFixtureReport: Encodable {
    let path: String
    let reference: SharedFrameReference
    let pixelSHA256: String
    let pendingCloseRefused: Bool
}
private struct RingFixtureRelease: Decodable { let release: SharedFrameAcknowledgement }

private func emit<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    try FileHandle.standardOutput.write(contentsOf: encoder.encode(value) + Data([10]))
}

private func readRingRelease() throws -> SharedFrameAcknowledgement {
    var bytes = Data()
    while bytes.count <= 4096 {
        guard let byte = try FileHandle.standardInput.read(upToCount: 1), !byte.isEmpty else {
            throw AstraError("fixture.eof", "Frame fixture ended before its lease acknowledgement.")
        }
        if byte[0] == 10 { return try JSONDecoder().decode(RingFixtureRelease.self, from: bytes).release }
        bytes.append(byte)
    }
    throw AstraError("fixture.oversize", "Frame fixture acknowledgement exceeds its bound.")
}

private func runRingFixture(parent: URL) throws {
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let ring = try SharedFrameRing(url: parent.appendingPathComponent(UUID().uuidString + ".astraring"),
                                   runID: UUID(), slotCount: 1, slotCapacity: 512)
    for iteration in 0..<2 {
        let pixels = Data((0..<140).map { UInt8(truncatingIfNeeded: $0 * 17) ^ (iteration == 0 ? 0 : 0xA5) })
        let surface = SurfaceDescriptor(id: "display:α", globalBounds: .init(x: -1920.5, y: -0.0, width: 5.25, height: 3.75),
                                        pixelWidth: 7, pixelHeight: 5,
                                        contentBounds: .init(x: 0.25, y: 0.125, width: 6.5, height: 4.75), geometryRevision: 17)
        let reference = try ring.publish(pixels: pixels, metadata: .init(eventNanos: UInt64(Int64.max) + 5,
                                        observedNanos: UInt64(Int64.max) + 13, surface: surface, byteCount: pixels.count))
        var refused = false
        do { try ring.close() }
        catch let error as AstraError where error.code == "frameRing.pendingLeases" { refused = true }
        guard refused else { throw AstraError("fixture.lease", "A live frame lease was closed unexpectedly.") }
        try emit(RingFixtureReport(path: ring.url.path, reference: reference, pixelSHA256: Data(SHA256.hash(data: pixels)).hex,
                                   pendingCloseRefused: refused))
        try ring.release(readRingRelease())
    }
    try ring.close()
    try emit(["closed": true, "pathExists": FileManager.default.fileExists(atPath: ring.url.path)])
}

// Development-only interoperability fixture. No capture/input permissions,
// user content, or recorded events are executed by this tool.
do {
    if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--frame-ring" {
        try runRingFixture(parent: URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true))
        exit(0)
    }
    guard CommandLine.arguments.count == 2 else {
        throw AstraError("fixture.arguments", "Usage: AstraFixture <output-parent-directory>")
    }
    let parent = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let identifier = UUID()
    let directory = parent.appendingPathComponent(identifier.uuidString + ".astrarecord", isDirectory: true)
    let manifest = RecordingManifest(id: identifier, name: "Cross-language fixture",
                                     environment: .init(name: "Deterministic surface", kind: .practice))
    let writer = try RecordingWriter(directory: directory, manifest: manifest)
    var checks: [[String: Any]] = []
    for index in 0..<3 {
        let size = index == 0 ? 4 : 64
        let pixels: Data
        if index == 0 { pixels = Data((0..<64).map { UInt8($0) }) }
        else {
            let color: [UInt8] = index == 1 ? [32, 64, 128, 255] : [64, 128, 32, 255]
            pixels = Data((0..<(size * size * 4)).map { color[$0 % 4] })
        }
        let surface = SurfaceDescriptor(id: "surface:0", globalBounds: .init(x: -200, y: 100, width: 32, height: 32),
                                        pixelWidth: size, pixelHeight: size)
        let sourceTime = UInt64(1_000_000_000 + index * 33_333_333)
        let prepared = try FrameArchive.prepare(pixels: pixels, metadata: .init(eventNanos: sourceTime,
                                                observedNanos: sourceTime + 2_000_000, surface: surface, byteCount: pixels.count))
        try writer.append(prepared)
        checks.append(["id": prepared.metadata.id.uuidString, "codec": prepared.metadata.codec,
                       "pixelSHA256": Data(SHA256.hash(data: pixels)).hex,
                       "observedNanos": prepared.metadata.observedNanos])
    }
    let events: [RawInputEvent] = [
        .init(sequence: 0, eventNanos: 910_000_000, observedNanos: 910_000_000, origin: .reconciliation,
              kind: .pointer, x: -180, y: 120, modifiers: 0, detail: "initial physical state"),
        .init(sequence: 1, eventNanos: 1_008_000_000, observedNanos: 1_011_000_000, origin: .physical,
              kind: .keyDown, keyCode: 13, isDown: true),
        .init(sequence: 2, eventNanos: 1_012_000_000, observedNanos: 1_014_000_000, origin: .physical,
              kind: .keyRepeat, keyCode: 13, isDown: true),
        .init(sequence: 3, eventNanos: 1_025_000_000, observedNanos: 1_027_000_000, origin: .physical,
              kind: .keyUp, keyCode: 13, isDown: false),
        .init(sequence: 4, eventNanos: 1_020_000_000, observedNanos: 1_030_000_000, origin: .physical,
              kind: .pointer, x: -170, y: 124, dx: 1, dy: -1),
        .init(sequence: 5, eventNanos: 1_040_000_000, observedNanos: 1_042_000_000, origin: .physical,
              kind: .scroll, x: -170, y: 124, scrollX: 0.5, scrollY: -1.25,
              detail: "scroll units: points", rawPlatformData: Data([0, 1, 2, 255]))
    ]
    try writer.append(events: events)
    let sealed = try writer.finish(at: 1_100_000_000, status: .complete)
    let report: [String: Any] = ["directory": directory.path, "frames": checks, "eventCount": sealed.eventCount]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data([10]))
} catch {
    FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
    exit(1)
}
