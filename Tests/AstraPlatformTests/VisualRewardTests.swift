import Foundation
import CoreGraphics
import CoreText
import CryptoKit
import ImageIO
import UniformTypeIdentifiers
import Testing
import AstraCore
@testable import AstraPlatform

@Test func rewardSequenceRehearsalReadsSealedFramesAndPaysAnEdgeOnce() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 8, height: 8), pixelWidth: 8, pixelHeight: 8)
    var red = Data(repeating: 0, count: 8 * 8 * 4), black = red
    for offset in stride(from: 0, to: red.count, by: 4) { red[offset + 2] = 255; red[offset + 3] = 255; black[offset + 3] = 255 }
    let image = try VisualRewardDetector.image(.init(metadata: .init(eventNanos: 0, observedNanos: 0, surface: surface, byteCount: red.count), pixels: red))
    let png = NSMutableData()
    let destination = try #require(CGImageDestinationCreateWithData(png, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil); #expect(CGImageDestinationFinalize(destination))
    var signal = RewardSignal(name: "Red", kind: .imageMatch, surfaceID: "fixture", region: .init(x: 0, y: 0, width: 1, height: 1))
    signal.templateDigest = try RewardAssets.save(png as Data, root: root)
    let condition = RewardPredicate(conditions: [.init(signalID: signal.id, comparison: .atLeast, number: 0.99)])
    let program = RewardProgram(name: "Edge", signals: [signal], rules: [.init(name: "Became red", kind: .risingEdge, amount: 1, predicate: condition)])
    let manifest = RecordingManifest(name: "Reward rehearsal fixture", environment: .init(name: "Generated", kind: .practice))
    let directory = root.appendingPathComponent("Recordings/\(manifest.id.uuidString).astrarecord")
    let writer = try RecordingWriter(directory: directory, manifest: manifest)
    for index in 0..<3 {
        let time = UInt64(index + 1) * 100_000_000, pixels = index == 0 ? black : red
        try writer.append(FrameArchive.prepare(pixels: pixels, metadata: .init(eventNanos: time, observedNanos: time, surface: surface, byteCount: pixels.count)))
    }
    _ = try writer.finish(at: 400_000_000, status: .complete)
    let result = try RewardRehearsal.run(program: program, directory: directory, assetRoot: root, durationSeconds: 1)
    #expect(result.readinessReached && result.inspectedDecisions == 3 && result.evaluations.count == 2)
    #expect(result.evaluations.map(\.value) == [1, 0])
    let reportURL = try RewardRehearsal.save(result, root: root)
    let decoded = try JSONDecoder().decode(RewardRehearsalReport.self, from: Data(contentsOf: reportURL))
    #expect(decoded.recordingID == manifest.id && decoded.program == program && decoded.evaluations.map(\.value) == [1, 0])
    #expect(throws: (any Error).self) { try RewardRehearsal.save(result, root: root) }
    #expect(throws: CancellationError.self) { try RewardRehearsal.run(program: program, directory: directory, assetRoot: root, cancelled: { true }) }
}

@Test func rewardNumberParsingRejectsAmbiguityAndUsesExplicitLocale() {
    #expect(VisualRewardDetector.parseNumber("Score: 1,234.50", decimalSeparator: ".") == 1234.5)
    #expect(VisualRewardDetector.parseNumber("Score: −1.234,50", decimalSeparator: ",") == -1234.5)
    #expect(VisualRewardDetector.parseNumber("Lives 2  Score 500", decimalSeparator: ".") == nil)
    #expect(VisualRewardDetector.parseNumber("1,23", decimalSeparator: ".") == nil)
    #expect(VisualRewardDetector.parseNumber("1.23.4", decimalSeparator: ".") == nil)
    #expect(VisualRewardDetector.parseNumber("Not readable", decimalSeparator: ".") == nil)
    #expect(VisualRewardDetector.parseNumber("1e9", decimalSeparator: ".") == nil)
}

@Test func rewardImageMatchingUsesVerifiedPixelsAndContentRelativeRegions() throws {
    var bytes = Data(repeating: 0, count: 8 * 8 * 4)
    for y in 0..<8 { for x in 0..<8 {
        let offset = (y * 8 + x) * 4
        bytes[offset + 2] = y < 4 ? 255 : 0
        bytes[offset] = y < 4 ? 0 : 255
        bytes[offset + 3] = 255
    } }
    let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 8, height: 8), pixelWidth: 8, pixelHeight: 8)
    let frame = RewardImageFrame(metadata: .init(eventNanos: 50, observedNanos: 70, surface: surface, byteCount: bytes.count), pixels: bytes)
    let image = try VisualRewardDetector.image(frame)
    let red = try VisualRewardDetector.crop(image, region: .init(x: 0, y: 0, width: 1, height: 0.5), content: surface.contentBounds)
    let png = NSMutableData()
    let destination = try #require(CGImageDestinationCreateWithData(png, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, red, nil); #expect(CGImageDestinationFinalize(destination))
    let data = png as Data
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    #expect(try VisualRewardDetector.similarity(red, templateData: data, expectedDigest: digest) == 1)
    let blue = try VisualRewardDetector.crop(image, region: .init(x: 0, y: 0.5, width: 1, height: 0.5), content: surface.contentBounds)
    #expect(try VisualRewardDetector.similarity(blue, templateData: data, expectedDigest: digest) < 0.4)
    #expect(throws: AstraError.self) { try VisualRewardDetector.similarity(red, templateData: data, expectedDigest: String(repeating: "0", count: 64)) }
    let episode = UUID()
    var signal = RewardSignal(name: "Red indicator", kind: .imageMatch, surfaceID: "fixture", region: .init(x: 0, y: 0, width: 1, height: 0.5))
    signal.templateDigest = digest
    let readings = try VisualRewardDetector.read(signals: [signal], frames: [frame], episodeID: episode, templates: [digest: data])
    #expect(readings.count == 1 && readings[0].value == .number(1))
    #expect(readings[0].episodeID == episode && readings[0].eventNanos == 50 && readings[0].observedNanos == 70)
}

@Test func rewardOCRReadsLocallyGeneratedScoreAndReportsBlankAsUnknown() throws {
    let width = 640, height = 160
    var pixels = Data(repeating: 255, count: width * height * 4)
    try pixels.withUnsafeMutableBytes { buffer in
        let context = try #require(CGContext(data: buffer.baseAddress, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue))
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica-Bold" as CFString, 56, nil),
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0, alpha: 1)]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: "Score: 120", attributes: attributes))
        context.textPosition = CGPoint(x: 20, y: 55); CTLineDraw(line, context)
    }
    let surface = SurfaceDescriptor(id: "generated", globalBounds: .init(x: 0, y: 0, width: Double(width), height: Double(height)), pixelWidth: width, pixelHeight: height)
    let frame = RewardImageFrame(metadata: .init(eventNanos: 100, observedNanos: 110, surface: surface, byteCount: pixels.count), pixels: pixels)
    let signal = RewardSignal(name: "Score", kind: .ocrNumber, surfaceID: "generated", region: .init(x: 0, y: 0, width: 1, height: 1))
    let result = try VisualRewardDetector.read(signals: [signal], frames: [frame], episodeID: UUID())
    #expect(result.count == 1 && result[0].value == .number(120))
    #expect(result[0].confidence >= 0.8)
    let blank = RewardImageFrame(metadata: frame.metadata, pixels: Data(repeating: 255, count: pixels.count))
    let missing = try VisualRewardDetector.read(signals: [signal], frames: [blank], episodeID: UUID())
    if case .unknown = missing.first?.value {} else { Issue.record("Blank OCR became a known reward value") }
}
