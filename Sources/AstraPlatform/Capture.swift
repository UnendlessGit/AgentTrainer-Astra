import AppKit
import AstraCore
import CoreMedia
import CoreVideo
@preconcurrency import ScreenCaptureKit

public struct CaptureSource: Identifiable, Hashable, Sendable {
    public var id: String
    public var name: String
    public var kind: TargetKind
    public var displayID: UInt32?
    public var windowID: UInt32?
    public var applicationBundleID: String?
    public var applicationPID: Int32?
    public var applicationLaunchDate: Date? = nil
    public var bounds: Rect2D
    public var pixelWidth: Int
    public var pixelHeight: Int
    /// An application/desktop binds these exact leaves until the session ends.
    public var bindings: [CaptureSource]? = nil

    public init(id: String, name: String, kind: TargetKind, displayID: UInt32? = nil, windowID: UInt32? = nil,
                applicationBundleID: String? = nil, applicationPID: Int32? = nil, applicationLaunchDate: Date? = nil,
                bounds: Rect2D, pixelWidth: Int, pixelHeight: Int, bindings: [CaptureSource]? = nil) {
        self.id = id; self.name = name; self.kind = kind; self.displayID = displayID; self.windowID = windowID
        self.applicationBundleID = applicationBundleID; self.applicationPID = applicationPID
        self.applicationLaunchDate = applicationLaunchDate; self.bounds = bounds
        self.pixelWidth = pixelWidth; self.pixelHeight = pixelHeight; self.bindings = bindings
    }

    public func captureBindings() throws -> [CaptureSource] {
        let leaves = bindings ?? [self]
        guard (1...16).contains(leaves.count), Set(leaves.map(\.id)).count == leaves.count,
              leaves.allSatisfy({ $0.bindings == nil && ($0.kind == .window || $0.kind == .display) }) else {
            throw AstraError("capture.bindings", "Choose one to sixteen fixed window or display surfaces.")
        }
        for leaf in leaves {
            _ = try leaf.surfaceDescriptor().validated()
            guard (leaf.kind == .window && leaf.windowID != nil && leaf.applicationPID != nil)
                    || (leaf.kind == .display && leaf.displayID != nil) else {
                throw AstraError("capture.bindingIdentity", "A capture binding has no native source identity.")
            }
        }
        if kind == .application {
            guard windowID == nil, applicationPID != nil, leaves.allSatisfy({ $0.kind == .window
                && $0.applicationPID == applicationPID && $0.applicationLaunchDate == applicationLaunchDate }) else {
                throw AstraError("capture.applicationBindings", "The application surfaces do not belong to the same running application.")
            }
        } else if kind == .desktop {
            guard leaves.allSatisfy({ $0.kind == .display }) else {
                throw AstraError("capture.desktopBindings", "A desktop source must bind only displays.")
            }
        } else if bindings != nil {
            throw AstraError("capture.groupKind", "Only an application or desktop can contain multiple capture bindings.")
        }
        return leaves
    }

    public func surfaceDescriptor() -> SurfaceDescriptor {
        SurfaceDescriptor(id: id, globalBounds: bounds, pixelWidth: pixelWidth, pixelHeight: pixelHeight,
                          nativeWindowID: windowID, nativeDisplayID: displayID)
    }
}

public enum CaptureDiscovery {
    public static func sources() async throws -> [CaptureSource] {
        guard CGPreflightScreenCaptureAccess() else {
            throw AstraError("permission.screenRecording", "Allow Screen Recording for AgentTrainer Astra in System Settings to choose an environment.")
        }
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        let displays = content.displays.map { display in
            CaptureSource(id: "display:\(display.displayID)", name: "Display \(display.displayID)", kind: .display,
                          displayID: display.displayID, bounds: Rect2D(CGDisplayBounds(display.displayID)),
                          pixelWidth: CGDisplayPixelsWide(display.displayID), pixelHeight: CGDisplayPixelsHigh(display.displayID))
        }
        let windows = content.windows.filter { $0.windowLayer == 0 && $0.frame.width > 1 && $0.frame.height > 1 }.compactMap { window -> CaptureSource? in
            guard let app = window.owningApplication, app.processID > 0 else { return nil }
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let scale = Double(filter.pointPixelScale)
            let width = Double(filter.contentRect.width) * scale, height = Double(filter.contentRect.height) * scale
            guard scale.isFinite, (1...4).contains(scale), width.isFinite, height.isFinite,
                  (1...32_768).contains(width), (1...32_768).contains(height), Rect2D(window.frame).isValid else { return nil }
            return CaptureSource(id: "window:\(window.windowID)",
                                 name: [app.applicationName, window.title].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " — "),
                                 kind: .window, windowID: window.windowID,
                                 applicationBundleID: app.bundleIdentifier, applicationPID: app.processID,
                                 applicationLaunchDate: NSRunningApplication(processIdentifier: app.processID)?.launchDate,
                                 bounds: Rect2D(window.frame),
                                 pixelWidth: Int(width.rounded(.up)), pixelHeight: Int(height.rounded(.up)))
        }
        let applications = Dictionary(grouping: windows, by: { $0.applicationPID! }).values.compactMap { leaves -> CaptureSource? in
            guard leaves.count <= 16, let first = leaves.first else { return nil }
            let ordered = leaves.sorted { $0.windowID! < $1.windowID! }
            let name = content.applications.first(where: { $0.processID == first.applicationPID })?.applicationName ?? first.name
            return CaptureSource(id: "application:\(first.applicationPID!)", name: "\(name) — all current windows",
                kind: .application, applicationBundleID: first.applicationBundleID, applicationPID: first.applicationPID,
                applicationLaunchDate: first.applicationLaunchDate, bounds: unionBounds(ordered),
                pixelWidth: first.pixelWidth, pixelHeight: first.pixelHeight, bindings: ordered)
        }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        let desktop: [CaptureSource] = displays.isEmpty || displays.count > 16 ? [] : [
            CaptureSource(id: "desktop", name: "Whole desktop", kind: .desktop, bounds: unionBounds(displays),
                pixelWidth: displays[0].pixelWidth, pixelHeight: displays[0].pixelHeight,
                bindings: displays.sorted { $0.displayID! < $1.displayID! })
        ]
        return desktop + displays + applications + windows.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    private static func unionBounds(_ sources: [CaptureSource]) -> Rect2D {
        Rect2D(sources.dropFirst().reduce(sources[0].bounds.cgRect) { $0.union($1.bounds.cgRect) })
    }
}

public enum CaptureHealth: Sendable {
    case starting, live, idle, unavailable(String), stopped
    case coverage(CaptureFrameCoverage)
}

enum CaptureGeometry {
    static func resolve(previous: SurfaceDescriptor, pixelWidth: Int, pixelHeight: Int,
                        screenRect: CGRect?, contentRect: CGRect?, scaleFactor: Double?,
                        contentScale: Double?) throws -> SurfaceDescriptor {
        guard let screenRect, Rect2D(screenRect).isValid,
              let contentRect, Rect2D(contentRect).isValid,
              let scaleFactor, scaleFactor.isFinite, (1...4).contains(scaleFactor),
              let contentScale, contentScale.isFinite, contentScale > 0 else {
            throw AstraError("capture.geometry", "ScreenCaptureKit returned incomplete or invalid source geometry.")
        }
        // SCStream.h specifies contentRect in surface points. scaleFactor is
        // pixels per point; contentScale is already reflected in that rect.
        // Multiplying by contentScale again would apply resizing twice.
        let pixels = Rect2D(x: contentRect.minX * scaleFactor, y: contentRect.minY * scaleFactor,
                            width: contentRect.width * scaleFactor, height: contentRect.height * scaleFactor)
        var descriptor = SurfaceDescriptor(id: previous.id, globalBounds: Rect2D(screenRect),
                                           pixelWidth: pixelWidth, pixelHeight: pixelHeight,
                                           contentBounds: pixels, geometryRevision: previous.geometryRevision,
                                           nativeWindowID: previous.nativeWindowID, nativeDisplayID: previous.nativeDisplayID)
        _ = try descriptor.validated()
        if previous.globalBounds != descriptor.globalBounds || previous.pixelWidth != descriptor.pixelWidth
            || previous.pixelHeight != descriptor.pixelHeight || previous.contentBounds != descriptor.contentBounds {
            guard previous.geometryRevision < UInt64.max else {
                throw AstraError("capture.geometryRevision", "The capture geometry revision is exhausted.")
            }
            descriptor.geometryRevision += 1
        }
        return descriptor
    }
}

/// The immutable buffer reference has explicit consumer ownership. Callers must
/// bound admitted frames and release this before retaining more SCK surfaces.
public struct CapturedFrame: @unchecked Sendable {
    public let id: UUID
    public let eventNanos: UInt64
    public let observedNanos: UInt64
    public let surface: SurfaceDescriptor
    public let pixelBuffer: CVPixelBuffer
    public let coverage: CaptureFrameCoverage

    public func copyCompactPixels() throws -> Data {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw AstraError("capture.pixelFormat", "The capture stream returned an unsupported pixel format.")
        }
        let width = CVPixelBufferGetWidth(pixelBuffer), height = CVPixelBufferGetHeight(pixelBuffer)
        let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
        let (byteCount, byteOverflow) = pixelCount.multipliedReportingOverflow(by: 4)
        guard width > 0, height > 0, !pixelOverflow, !byteOverflow, byteCount <= FrameArchive.maximumFrameBytes else {
            throw AstraError("capture.frameTooLarge", "This capture exceeds the configured frame memory limit.")
        }
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
            throw AstraError("capture.pixelAccess", "The captured pixels are temporarily unavailable.")
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let source = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw AstraError("capture.pixelAccess", "The capture buffer has no readable pixel data.")
        }
        let sourceStride = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard sourceStride >= width * 4 else { throw AstraError("capture.stride", "Invalid capture row stride.") }
        var pixels = Data(count: byteCount)
        pixels.withUnsafeMutableBytes { destination in
            for row in 0..<height {
                destination.baseAddress!.advanced(by: row * width * 4)
                    .copyMemory(from: source.advanced(by: row * sourceStride), byteCount: width * 4)
            }
        }
        return pixels
    }
}

public final class ScreenCapture: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    public typealias FrameHandler = @Sendable (CapturedFrame) -> Void
    public typealias HealthHandler = @Sendable (CaptureHealth) -> Void
    private let lock = NSLock()
    private let outputQueue: DispatchQueue
    private var stream: SCStream?
    private var generation: UUID?
    private var source: CaptureSource?
    private var surface: SurfaceDescriptor?
    private var coverage: CaptureFrameCoverage?
    private var frameHandler: FrameHandler?
    private var healthHandler: HealthHandler?
    private var startingCompletion: AsyncCompletion?
    private var stopTask: Task<Void, Never>?

    public override init() { outputQueue = DispatchQueue(label: "astra.capture.output", qos: .userInitiated); super.init() }
    init(outputQueue: DispatchQueue) { self.outputQueue = outputQueue; super.init() }

    public func start(source: CaptureSource, fps: Int, showsCursor: Bool,
                      onFrame: @escaping FrameHandler, onHealth: @escaping HealthHandler) async throws {
        guard CGPreflightScreenCaptureAccess() else {
            throw AstraError("permission.screenRecording", "Screen Recording permission is required for this environment.")
        }
        guard (1...120).contains(fps) else { throw AstraError("capture.rate", "Choose a capture rate between 1 and 120 fps.") }
        let initialSurface = try source.surfaceDescriptor().validated()
        guard source.pixelWidth * source.pixelHeight * 4 <= FrameArchive.maximumFrameBytes else {
            throw AstraError("capture.frameTooLarge", "This capture exceeds the recording frame memory limit.")
        }
        let token = UUID()
        let completion = AsyncCompletion()
        try lock.withLock {
            guard generation == nil, stopTask == nil else { throw AstraError("capture.busy", "A capture stream is already starting, running, or stopping.") }
            generation = token; self.source = source; frameHandler = onFrame; healthHandler = onHealth; coverage = nil
            startingCompletion = completion
        }
        defer { completion.finish() }
        onHealth(.starting)
        var startedStream: SCStream?
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
            try Task.checkCancellation()
            let filter: SCContentFilter
            var resolvedSurface = initialSurface
            if source.kind == .window, let id = source.windowID {
                guard let window = content.windows.first(where: { $0.windowID == id && $0.owningApplication?.processID == source.applicationPID }) else {
                    throw AstraError("capture.targetMissing", "The selected window is no longer available. Choose the target again.")
                }
                if let expectedLaunch = source.applicationLaunchDate {
                    guard let pid = source.applicationPID,
                          NSRunningApplication(processIdentifier: pid)?.launchDate == expectedLaunch else {
                        throw AstraError("capture.targetReplaced", "The selected application restarted. Choose the target again.")
                    }
                }
                filter = SCContentFilter(desktopIndependentWindow: window)
                resolvedSurface.globalBounds = Rect2D(window.frame)
            } else if source.kind == .display, let id = source.displayID {
                guard let display = content.displays.first(where: { $0.displayID == id }) else {
                    throw AstraError("capture.targetMissing", "The selected display is no longer connected.")
                }
                // The selected display is the observation scope, including
                // Astra when visible. Excluding UI pixels while recording
                // their global input would produce inconsistent examples.
                filter = SCContentFilter(display: display, excludingWindows: [])
                resolvedSurface.globalBounds = Rect2D(CGDisplayBounds(id))
            } else {
                throw AstraError("capture.source", "This source needs a supported capture binding.")
            }
            let nativeWidth = Double(filter.contentRect.width) * Double(filter.pointPixelScale)
            let nativeHeight = Double(filter.contentRect.height) * Double(filter.pointPixelScale)
            guard nativeWidth.isFinite, nativeHeight.isFinite,
                  (1...32_768).contains(nativeWidth), (1...32_768).contains(nativeHeight) else {
                throw AstraError("capture.nativeSize", "The selected source has no valid native pixel dimensions.")
            }
            resolvedSurface.pixelWidth = Int(nativeWidth.rounded(.up))
            resolvedSurface.pixelHeight = Int(nativeHeight.rounded(.up))
            resolvedSurface.contentBounds = Rect2D(x: 0, y: 0, width: Double(resolvedSurface.pixelWidth), height: Double(resolvedSurface.pixelHeight))
            _ = try resolvedSurface.validated()
            guard resolvedSurface.pixelWidth * resolvedSurface.pixelHeight * 4 <= FrameArchive.maximumFrameBytes else {
                throw AstraError("capture.frameTooLarge", "The current capture size exceeds the recording frame memory limit.")
            }
            let configuration = SCStreamConfiguration()
            configuration.width = resolvedSurface.pixelWidth; configuration.height = resolvedSurface.pixelHeight
            configuration.minimumFrameInterval = CMTime(value: 1, timescale: Int32(fps))
            configuration.queueDepth = 4; configuration.pixelFormat = kCVPixelFormatType_32BGRA
            configuration.colorSpaceName = CGColorSpace.sRGB
            configuration.showsCursor = showsCursor; configuration.capturesAudio = false
            configuration.ignoreShadowsSingleWindow = true; configuration.captureResolution = .best
            configuration.scalesToFit = false; configuration.preservesAspectRatio = true
            let candidate = SCStream(filter: filter, configuration: configuration, delegate: self)
            startedStream = candidate
            try candidate.addStreamOutput(self, type: .screen, sampleHandlerQueue: outputQueue)
            let accepted = lock.withLock {
                guard generation == token, stopTask == nil else { return false }
                stream = candidate; surface = resolvedSurface
                return true
            }
            guard accepted else { throw CancellationError() }
            try await candidate.startCapture()
            try Task.checkCancellation()
            let stillActive = lock.withLock { generation == token && stopTask == nil }
            if !stillActive { throw CancellationError() }
        } catch {
            if let startedStream {
                try? await startedStream.stopCapture()
                await withCheckedContinuation { continuation in outputQueue.async { continuation.resume() } }
                try? startedStream.removeStreamOutput(self, type: .screen)
            }
            let notify = lock.withLock {
                if generation == token {
                    generation = nil; stream = nil; self.source = nil; surface = nil; coverage = nil
                    frameHandler = nil; healthHandler = nil
                    return true
                }
                return false
            }
            if notify { onHealth(.unavailable(error.localizedDescription)) }
            throw error
        }
    }

    public func stop() async {
        let task = lock.withLock { () -> Task<Void, Never> in
            if let stopTask { return stopTask }
            let previous = stream, callback = healthHandler
            let completion = startingCompletion
            generation = nil; stream = nil; frameHandler = nil; healthHandler = nil; source = nil; surface = nil; coverage = nil
            let task = Task { [self] in
                // A concurrent start may not have called startCapture yet. Join
                // it so no late stream can briefly revive after Stop returns.
                await completion?.wait()
                if let previous { try? await previous.stopCapture() }
                await withCheckedContinuation { continuation in outputQueue.async { continuation.resume() } }
                if let previous { try? previous.removeStreamOutput(self, type: .screen) }
                callback?(.stopped)
                lock.withLock { startingCompletion = nil; stopTask = nil }
            }
            stopTask = task
            return task
        }
        await task.value
    }

    public func stream(_ stream: SCStream, didStopWithError error: any Error) {
        let callback = lock.withLock { () -> HealthHandler? in
            guard self.stream === stream else { return nil }
            coverage = nil
            return healthHandler
        }
        callback?(.unavailable(error.localizedDescription))
    }

    public func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid else { return }
        let observed = MonotonicClock.now
        let attachments = (CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]])?.first
        let status = (attachments?[.status] as? Int).flatMap(SCFrameStatus.init(rawValue:))
        let snapshot = lock.withLock { () -> (UUID, SurfaceDescriptor, FrameHandler, HealthHandler)? in
            guard self.stream === stream, let generation, let surface, let frameHandler, let healthHandler else { return nil }
            return (generation, surface, frameHandler, healthHandler)
        }
        guard let (token, initialDescriptor, frameCallback, healthCallback) = snapshot else { return }
        let timestamp = (attachments?[.displayTime] as? NSNumber).map({ MonotonicClock.nanoseconds(fromMachTicks: $0.uint64Value) })
            ?? MonotonicClock.nanoseconds(hostTime: sampleBuffer.presentationTimeStamp)
        if status == .idle {
            healthCallback(.idle)
            // SCK explicitly reports no source change. An idle callback without
            // its source clock and complete geometry cannot establish coverage.
            guard let timestamp, timestamp <= observed,
                  let descriptor = try? CaptureGeometry.resolve(previous: initialDescriptor,
                    pixelWidth: initialDescriptor.pixelWidth, pixelHeight: initialDescriptor.pixelHeight,
                    screenRect: Self.rect(attachments?[.screenRect]), contentRect: Self.rect(attachments?[.contentRect]),
                    scaleFactor: (attachments?[.scaleFactor] as? NSNumber)?.doubleValue,
                    contentScale: (attachments?[.contentScale] as? NSNumber)?.doubleValue) else { return }
            do {
                let verified = try lock.withLock { () throws -> CaptureFrameCoverage? in
                    guard self.stream === stream, generation == token, let coverage else { return nil }
                    // A late source event may precede delivery of its base frame;
                    // it adds no evidence and must not refresh arrival time.
                    guard timestamp >= coverage.throughNanos else { return nil }
                    let next = try coverage.verifyingUnchanged(streamID: token, surface: descriptor,
                        throughNanos: timestamp, verifiedAtNanos: observed)
                    self.coverage = next
                    return next
                }
                if let verified { healthCallback(.coverage(verified)) }
            } catch { healthCallback(.unavailable(error.localizedDescription)) }
            return
        }
        if status == .started, sampleBuffer.imageBuffer == nil { healthCallback(.starting); return }
        guard status == .complete || status == .started else {
            lock.withLock { if self.stream === stream { coverage = nil } }
            healthCallback(.unavailable("The capture surface is unavailable.")); return
        }
        guard let pixels = sampleBuffer.imageBuffer, let timestamp, timestamp <= observed else {
            healthCallback(.unavailable("A capture frame has no valid timestamp or pixels.")); return
        }
        let width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels)
        let descriptor: SurfaceDescriptor
        do {
            descriptor = try CaptureGeometry.resolve(previous: initialDescriptor, pixelWidth: width, pixelHeight: height,
                                                      screenRect: Self.rect(attachments?[.screenRect]),
                                                      contentRect: Self.rect(attachments?[.contentRect]),
                                                      scaleFactor: (attachments?[.scaleFactor] as? NSNumber)?.doubleValue,
                                                      contentScale: (attachments?[.contentScale] as? NSNumber)?.doubleValue)
        } catch { healthCallback(.unavailable(error.localizedDescription)); return }
        let id = UUID()
        let evidence: CaptureFrameCoverage
        do {
            evidence = try CaptureFrameCoverage(streamID: token, frame: FrameMetadata(id: id, eventNanos: timestamp,
                observedNanos: observed, surface: descriptor, byteCount: width * height * 4, codec: "raw"))
            let accepted = try lock.withLock {
                guard self.stream === stream, generation == token else { return false }
                if let coverage, timestamp < (coverage.kind == .unchanged ? coverage.throughNanos : coverage.eventNanos) {
                    throw AstraError("capture.sourceClock", "Capture source time moved backwards.")
                }
                surface = descriptor; coverage = evidence
                return true
            }
            guard accepted else { return }
        } catch { healthCallback(.unavailable(error.localizedDescription)); return }
        healthCallback(.live)
        frameCallback(CapturedFrame(id: id, eventNanos: timestamp, observedNanos: observed,
                                    surface: descriptor, pixelBuffer: pixels, coverage: evidence))
    }

    private static func rect(_ value: Any?) -> CGRect? {
        if let rect = value as? CGRect { return rect }
        if let dictionary = value as? NSDictionary { return CGRect(dictionaryRepresentation: dictionary) }
        return nil
    }
}
