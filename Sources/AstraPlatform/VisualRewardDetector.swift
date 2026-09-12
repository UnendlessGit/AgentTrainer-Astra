import Foundation
import CoreGraphics
import CryptoKit
import ImageIO
import Vision
import AstraCore

public struct RewardImageFrame: Sendable {
    public let metadata: FrameMetadata
    public let pixels: Data
    public init(metadata: FrameMetadata, pixels: Data) { self.metadata = metadata; self.pixels = pixels }
}

/// Synchronous, bounded detector work for the environment's analysis queue.
/// It consumes owned pixels, never captures a screen or requests privacy access.
public enum VisualRewardDetector {
    public static func supportedLanguages() throws -> [String] {
        let request = VNRecognizeTextRequest(); request.recognitionLevel = .accurate; request.revision = VNRecognizeTextRequestRevision3
        return try request.supportedRecognitionLanguages()
    }
    public static func read(signals: [RewardSignal], frames: [RewardImageFrame], episodeID: UUID,
                            templates: [String: Data] = [:]) throws -> [SignalReading] {
        guard signals.count <= 32, Set(signals.map(\.id)).count == signals.count, frames.count <= 16,
              Set(frames.map { $0.metadata.surface.id }).count == frames.count else {
            throw AstraError("reward.detectorLimit", "Reward analysis requires a bounded, unique signal and surface snapshot.")
        }
        var images: [String: (RewardImageFrame, CGImage)] = [:]
        for frame in frames {
            _ = try frame.metadata.validated()
            guard frame.pixels.count == frame.metadata.byteCount else { throw AstraError("reward.pixels", "Reward analysis received incomplete frame pixels.") }
            images[frame.metadata.surface.id] = (frame, try image(frame))
        }
        var readings: [SignalReading] = []
        for signal in signals where signal.kind.isVisual {
            _ = try signal.validated()
            guard let id = signal.surfaceID, let (frame, full) = images[id], let region = signal.region else { continue }
            let crop = try crop(full, region: region, content: frame.metadata.surface.contentBounds)
            let result: (SignalValue, Double)
            switch signal.kind {
            case .ocrText, .ocrNumber:
                let text = try recognize(crop, language: signal.language)
                if let text {
                    if signal.kind == .ocrNumber {
                        result = (parseNumber(text.0, decimalSeparator: signal.decimalSeparator).map(SignalValue.number)
                                  ?? .unknown("The region does not contain one unambiguous number."), text.1)
                    } else { result = (.text(text.0), text.1) }
                } else { result = (.unknown("No readable text was found in the selected region."), 0) }
            case .imageMatch:
                guard let digest = signal.templateDigest, let data = templates[digest] else {
                    throw AstraError("reward.templateMissing", "A reward image template is missing from the frozen definition.")
                }
                result = (.number(try similarity(crop, templateData: data, expectedDigest: digest)), 1)
            default: continue
            }
            readings.append(SignalReading(signalID: signal.id, episodeID: episodeID,
                eventNanos: frame.metadata.eventNanos, observedNanos: frame.metadata.observedNanos,
                confidence: result.1, value: result.0, sourceObservationID: frame.metadata.id))
        }
        return readings
    }

    public static func image(_ frame: RewardImageFrame) throws -> CGImage {
        _ = try frame.metadata.validated()
        let surface = frame.metadata.surface
        guard frame.pixels.count == frame.metadata.byteCount, let provider = CGDataProvider(data: frame.pixels as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: surface.pixelWidth, height: surface.pixelHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: surface.pixelWidth * 4, space: space,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).union(.byteOrder32Little),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw AstraError("reward.image", "The reward frame could not be interpreted as native sRGB pixels.")
        }
        return image
    }

    public static func crop(_ image: CGImage, region: Rect2D, content: Rect2D) throws -> CGImage {
        guard region.isValid, region.x >= 0, region.y >= 0, region.x + region.width <= 1, region.y + region.height <= 1,
              content.isValid, content.x >= 0, content.y >= 0,
              content.x + content.width <= Double(image.width), content.y + content.height <= Double(image.height) else {
            throw AstraError("reward.region", "The reward region is outside the captured image.")
        }
        let left = floor(content.x + region.x * content.width), top = floor(content.y + region.y * content.height)
        let right = ceil(content.x + (region.x + region.width) * content.width)
        let bottom = ceil(content.y + (region.y + region.height) * content.height)
        guard let crop = image.cropping(to: CGRect(x: left, y: top, width: right - left, height: bottom - top)) else {
            throw AstraError("reward.crop", "The selected reward region has no pixels.")
        }
        return crop
    }

    private static func recognize(_ image: CGImage, language: String) throws -> (String, Double)? {
        let request = VNRecognizeTextRequest()
        request.revision = VNRecognizeTextRequestRevision3
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.automaticallyDetectsLanguage = false
        guard try request.supportedRecognitionLanguages().contains(language) else {
            throw AstraError("reward.language", "The selected OCR language is not available on this Mac.")
        }
        request.recognitionLanguages = [language]
        try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
        // Vision's bounding boxes have a bottom-left origin. Reading order is
        // top-to-bottom, then left-to-right for lines at the same height.
        let observations = (request.results ?? []).sorted {
            if $0.boundingBox.midY != $1.boundingBox.midY { return $0.boundingBox.midY > $1.boundingBox.midY }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }
        let candidates = observations.compactMap { $0.topCandidates(1).first }
        guard !candidates.isEmpty else { return nil }
        let text = candidates.map(\.string).joined(separator: "\n")
        guard text.utf8.count <= 4096 else { throw AstraError("reward.textLimit", "The OCR region contains too much text. Select a smaller region.") }
        return (text, Double(candidates.map(\.confidence).min() ?? 0))
    }

    /// Explicit decimal convention; multiple values and malformed grouping are
    /// unknown. We never select the first score from an ambiguous HUD.
    public static func parseNumber(_ text: String, decimalSeparator: String) -> Double? {
        guard text.utf8.count <= 4096, [".", ","].contains(decimalSeparator) else { return nil }
        let text = text.replacingOccurrences(of: "−", with: "-")
        guard let pattern = try? NSRegularExpression(pattern: "[-+]?[0-9][0-9.,]*") else { return nil }
        let matches = pattern.matches(in: text, range: NSRange(text.startIndex..., in: text))
        guard matches.count == 1, let range = Range(matches[0].range, in: text) else { return nil }
        var token = String(text[range]), sign = ""
        if token.hasPrefix("-") || token.hasPrefix("+") { sign = String(token.removeFirst()) }
        let parts = token.components(separatedBy: decimalSeparator)
        guard parts.count <= 2 else { return nil }
        let grouping = decimalSeparator == "." ? "," : "."
        let groups = parts[0].components(separatedBy: grouping)
        func digits(_ value: String) -> Bool { !value.isEmpty && value.utf8.allSatisfy { (48...57).contains($0) } }
        guard groups.allSatisfy(digits), groups.count == 1 || ((1...3).contains(groups[0].count) && groups.dropFirst().allSatisfy { $0.count == 3 }),
              parts.count == 1 || digits(parts[1]) else { return nil }
        let normalized = sign + groups.joined() + (parts.count == 2 ? "." + parts[1] : "")
        guard let value = Double(normalized), value.isFinite else { return nil }
        return value
    }

    public static func similarity(_ image: CGImage, templateData: Data, expectedDigest: String) throws -> Double {
        guard templateData.count <= 16 * 1024 * 1024,
              SHA256.hash(data: templateData).map({ String(format: "%02x", $0) }).joined() == expectedDigest,
              let source = CGImageSourceCreateWithData(templateData as CFData, nil), CGImageSourceGetCount(source) == 1,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int,
              (1...1024).contains(width), (1...1024).contains(height),
              let template = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw AstraError("reward.template", "The template is corrupt, changed or larger than 1024 × 1024 pixels.")
        }
        let actual = try rgba(image, width: width, height: height), expected = try rgba(template, width: width, height: height)
        var difference: UInt64 = 0
        for index in stride(from: 0, to: actual.count, by: 4) {
            for channel in 0..<3 { difference += UInt64(abs(Int(actual[index + channel]) - Int(expected[index + channel]))) }
        }
        return 1 - Double(difference) / Double(width * height * 3 * 255)
    }

    private static func rgba(_ image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        try bytes.withUnsafeMutableBytes { buffer in
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                      bytesPerRow: width * 4, space: space,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else {
                throw AstraError("reward.templatePixels", "The image comparison buffer could not be allocated.")
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return bytes
    }
}
