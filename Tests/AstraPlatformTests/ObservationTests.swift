import Foundation
import Testing
import ApplicationServices
import CoreVideo
import CoreMedia
import AstraCore
@testable import AstraPlatform

@Test func inputTranslationKeepsRepeatsTimesAndOriginalPayload() throws {
    let event = try #require(CGEvent(keyboardEventSource: nil, virtualKey: 13, keyDown: true))
    event.timestamp = 123
    event.setIntegerValueField(.keyboardEventAutorepeat, value: 1)
    let translated = try #require(PhysicalInputMonitor.translate(event, sequence: 7, observedNanos: 456))
    #expect(translated.kind == .keyRepeat)
    #expect(translated.keyCode == 13)
    #expect(translated.eventNanos == 123)
    #expect(translated.observedNanos == 456)
    #expect(translated.rawPlatformData == event.data as Data?)
    event.setIntegerValueField(.eventSourceUserData, value: AstraSyntheticInput.tag)
    #expect(PhysicalInputMonitor.translate(event, sequence: 8, observedNanos: 789) == nil)
}

@Test func scrollingAndAdditionalButtonsRetainNativeArguments() throws {
    let scroll = try #require(CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 2,
                                     wheel1: 17, wheel2: -9, wheel3: 0))
    let value = try #require(PhysicalInputMonitor.translate(scroll, sequence: 0, observedNanos: MonotonicClock.now))
    #expect(value.scrollX == -9)
    #expect(value.scrollY == 17)
    let button = try #require(CGEvent(mouseEventSource: nil, mouseType: .otherMouseDown,
                                      mouseCursorPosition: CGPoint(x: -120, y: 35),
                                      mouseButton: CGMouseButton(rawValue: 15)!))
    let click = try #require(PhysicalInputMonitor.translate(button, sequence: 1, observedNanos: MonotonicClock.now))
    #expect(click.button == 15)
    #expect(click.x == -120)
    #expect(click.y == 35)
}

@Test func compactPixelCopyExcludesUninitializedRowPadding() throws {
    var output: CVPixelBuffer?
    let attributes = [kCVPixelBufferBytesPerRowAlignmentKey: 64] as CFDictionary
    #expect(CVPixelBufferCreate(kCFAllocatorDefault, 3, 2, kCVPixelFormatType_32BGRA, attributes, &output) == kCVReturnSuccess)
    let buffer = try #require(output)
    CVPixelBufferLockBaseAddress(buffer, [])
    let pointer = try #require(CVPixelBufferGetBaseAddress(buffer))
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    memset(pointer, 222, stride * 2)
    memset(pointer, 13, 12)
    memset(pointer.advanced(by: stride), 27, 12)
    CVPixelBufferUnlockBaseAddress(buffer, [])
    let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 3, height: 2),
                                    pixelWidth: 3, pixelHeight: 2)
    let metadata = FrameMetadata(eventNanos: 1, observedNanos: 2, surface: surface, byteCount: 24, codec: "raw")
    let coverage = try CaptureFrameCoverage(streamID: UUID(), frame: metadata)
    let frame = CapturedFrame(id: metadata.id, eventNanos: 1, observedNanos: 2, surface: surface, pixelBuffer: buffer, coverage: coverage)
    #expect(try frame.copyCompactPixels() == Data(repeating: 13, count: 12) + Data(repeating: 27, count: 12))
}

@Test func completionBarrierHandlesEarlyAndConcurrentCompletion() async {
    let completion = AsyncCompletion()
    await withTaskGroup(of: Int.self) { group in
        for index in 0..<8 { group.addTask { await completion.wait(); return index } }
        completion.finish()
        completion.finish()
        var results: [Int] = []
        for await value in group { results.append(value) }
        #expect(results.sorted() == Array(0..<8))
    }
    await completion.wait()
}

@Test func hostClockConversionUsesTheSameNativeEpoch() {
    let before = MonotonicClock.now
    let host = CMClockGetTime(CMClockGetHostTimeClock())
    let converted = MonotonicClock.nanoseconds(hostTime: host)
    let after = MonotonicClock.now
    #expect(converted != nil)
    #expect(converted! >= before)
    #expect(converted! <= after)
}

@Test func captureGeometryConvertsSurfacePointsWithoutApplyingContentScaleTwice() throws {
    let original = SurfaceDescriptor(id: "window", globalBounds: .init(x: -1_600, y: 30, width: 1_600, height: 900),
                                     pixelWidth: 1_920, pixelHeight: 1_080)
    let descriptor = try CaptureGeometry.resolve(previous: original, pixelWidth: 1_920, pixelHeight: 1_080,
                                                  screenRect: original.globalBounds.cgRect,
                                                  contentRect: CGRect(x: 0, y: 0, width: 960, height: 540),
                                                  scaleFactor: 2, contentScale: 0.6)
    #expect(descriptor.contentBounds == Rect2D(x: 0, y: 0, width: 1_920, height: 1_080))
    #expect(try descriptor.pixelPoint(global: Point2D(x: -800, y: 480)) == Point2D(x: 960, y: 540))
    #expect(descriptor.geometryRevision == 0)
}

@Test func captureGeometryHandlesFractionalDensityAndPaddedContent() throws {
    let previous = SurfaceDescriptor(id: "window", globalBounds: .init(x: 0, y: 0, width: 100, height: 80),
                                     pixelWidth: 180, pixelHeight: 150)
    let changed = try CaptureGeometry.resolve(previous: previous, pixelWidth: 180, pixelHeight: 150,
                                               screenRect: CGRect(x: -100, y: 20, width: 100, height: 80),
                                               contentRect: CGRect(x: 10, y: 10, width: 100, height: 80),
                                               scaleFactor: 1.5, contentScale: 1)
    #expect(changed.contentBounds == Rect2D(x: 15, y: 15, width: 150, height: 120))
    #expect(changed.geometryRevision == 1)
    #expect(try changed.pixelPoint(global: Point2D(x: -50, y: 60)) == Point2D(x: 90, y: 75))
}

@Test func captureGeometryRejectsMissingOrOutOfBufferMetadata() {
    let previous = SurfaceDescriptor(id: "window", globalBounds: .init(x: 0, y: 0, width: 100, height: 80),
                                     pixelWidth: 200, pixelHeight: 160)
    #expect(throws: AstraError.self) {
        try CaptureGeometry.resolve(previous: previous, pixelWidth: 200, pixelHeight: 160,
                                    screenRect: nil, contentRect: CGRect(x: 0, y: 0, width: 100, height: 80),
                                    scaleFactor: 2, contentScale: 1)
    }
    #expect(throws: AstraError.self) {
        try CaptureGeometry.resolve(previous: previous, pixelWidth: 200, pixelHeight: 160,
                                    screenRect: previous.globalBounds.cgRect,
                                    contentRect: CGRect(x: 1, y: 0, width: 100, height: 80),
                                    scaleFactor: 2, contentScale: 1)
    }
    #expect(throws: AstraError.self) {
        try CaptureGeometry.resolve(previous: previous, pixelWidth: 200, pixelHeight: 160,
                                    screenRect: previous.globalBounds.cgRect, contentRect: previous.globalBounds.cgRect,
                                    scaleFactor: .nan, contentScale: 1)
    }
}

private func scopeFixture(kind: TargetKind = .window) -> (CaptureSource, InputScopeSnapshot) {
    let bounds = Rect2D(x: -200, y: 30, width: 200, height: 100)
    let source = CaptureSource(id: "fixture", name: "Fixture", kind: kind,
                               windowID: 11, applicationPID: 42, applicationLaunchDate: Date(timeIntervalSince1970: 1),
                               bounds: bounds, pixelWidth: 400, pixelHeight: 200)
    let snapshot = InputScopeSnapshot(observedNanos: 100, frontmostPID: 42,
                                      applicationLaunchDate: source.applicationLaunchDate,
                                      windows: [.init(id: 11, pid: 42, layer: 0, bounds: bounds)])
    return (source, snapshot)
}

@Test func windowScopeRequiresForegroundTargetAndRejectsOtherEventRecipients() throws {
    let (source, snapshot) = scopeFixture()
    let event = RawInputEvent(sequence: 0, eventNanos: 110, observedNanos: 120, origin: .physical,
                              kind: .keyDown, keyCode: 0, isDown: true)
    try snapshot.verifyTarget(source)
    try snapshot.verify(event, targetPID: 42, source: source)
    // A missing event recipient uses the bracketed WindowServer evidence; it
    // is not fabricated as proof of application-level event consumption.
    try snapshot.verify(event, targetPID: 0, source: source)
    #expect(throws: AstraError.self) { try snapshot.verify(event, targetPID: 99, source: source) }
    let otherFocus = InputScopeSnapshot(observedNanos: 130, frontmostPID: 99,
                                       applicationLaunchDate: snapshot.applicationLaunchDate, windows: snapshot.windows)
    #expect(throws: AstraError.self) { try otherFocus.verifyTarget(source) }
    let replacement = InputScopeSnapshot(observedNanos: 130, frontmostPID: 42,
                                         applicationLaunchDate: Date(timeIntervalSince1970: 2), windows: snapshot.windows)
    #expect(throws: AstraError.self) { try replacement.verifyTarget(source) }
}

@Test func windowScopeRejectsOccludingPanelsAndOtherWindowsOfTheSameApplication() throws {
    let (source, snapshot) = scopeFixture()
    let click = RawInputEvent(sequence: 0, eventNanos: 110, observedNanos: 120, origin: .physical,
                              kind: .buttonDown, button: 0, x: -150, y: 70, isDown: true)
    try snapshot.verify(click, targetPID: 42, source: source)
    let panel = InputScopeSnapshot.Window(id: 12, pid: 42, layer: 3, bounds: source.bounds)
    let obscured = InputScopeSnapshot(observedNanos: 130, frontmostPID: 42,
                                     applicationLaunchDate: source.applicationLaunchDate, windows: [panel] + snapshot.windows)
    #expect(throws: AstraError.self) { try obscured.verifyTarget(source) }
    #expect(throws: AstraError.self) { try obscured.verify(click, targetPID: 42, source: source) }
    let otherWindow = InputScopeSnapshot.Window(id: 12, pid: 42, layer: 0, bounds: source.bounds)
    let changedWindow = InputScopeSnapshot(observedNanos: 130, frontmostPID: 42,
                                           applicationLaunchDate: source.applicationLaunchDate, windows: [otherWindow] + snapshot.windows)
    #expect(throws: AstraError.self) { try changedWindow.verifyTarget(source) }
}

@Test func annotatedRoutingDistinguishesClickThroughOverlaysFromInputRecipients() throws {
    let (source, snapshot) = scopeFixture()
    let overlay = InputScopeSnapshot.Window(id: 81, pid: 99, layer: 0, bounds: source.bounds)
    let covered = InputScopeSnapshot(observedNanos: 130, frontmostPID: 42,
                                    applicationLaunchDate: source.applicationLaunchDate, windows: [overlay] + snapshot.windows)
    try covered.verifyTarget(source)
    let pointer = RawInputEvent(sequence: 0, eventNanos: 110, observedNanos: 120, origin: .physical,
                                kind: .buttonDown, button: 0, x: -150, y: 70, isDown: true)
    // Neither the overlay's process nor title is special-cased. Only resolved
    // OS routing allows passing through it to the observed window.
    try covered.verify(pointer, targetPID: 42, handlingWindowID: 11, source: source)
    try covered.verify(pointer, targetPID: 42, source: source)
    #expect(throws: AstraError.self) { try covered.verify(pointer, targetPID: nil, source: source) }
    #expect(throws: AstraError.self) { try covered.verify(pointer, targetPID: 99, handlingWindowID: 81, source: source) }
    #expect(throws: AstraError.self) { try covered.verify(pointer, targetPID: 42, handlingWindowID: 81, source: source) }
    #expect(throws: AstraError.self) { try covered.verify(pointer, targetPID: 42, handlingWindowID: 999, source: source) }
}

@Test func annotatedRoutingDoesNotAdmitUnobservedSameAppPopups() throws {
    let (source, snapshot) = scopeFixture()
    let popup = InputScopeSnapshot.Window(id: 12, pid: 42, layer: 101, bounds: source.bounds)
    let shown = InputScopeSnapshot(observedNanos: 130, frontmostPID: 42,
                                  applicationLaunchDate: source.applicationLaunchDate, windows: [popup] + snapshot.windows)
    let pointer = RawInputEvent(sequence: 0, eventNanos: 110, observedNanos: 120, origin: .physical,
                                kind: .scroll, x: -150, y: 70, scrollX: 0, scrollY: 2)
    #expect(throws: AstraError.self) { try shown.verify(pointer, targetPID: 42, handlingWindowID: 12, source: source) }
    #expect(throws: AstraError.self) { try shown.verify(pointer, targetPID: 42, source: source) }
    // A visual popup that explicitly passes this event through is not its
    // recipient. Keyboard target verification remains a separate check.
    try shown.verify(pointer, targetPID: 42, handlingWindowID: 11, source: source)
}

@Test func inputRoutingReadsTheHandlingWindowRatherThanTheVisuallyTopWindow() throws {
    let event = try #require(CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown,
                                    mouseCursorPosition: CGPoint(x: -150, y: 70), mouseButton: .left))
    event.setIntegerValueField(.eventTargetUnixProcessID, value: 42)
    event.setIntegerValueField(.mouseEventWindowUnderMousePointer, value: 81)
    event.setIntegerValueField(.mouseEventWindowUnderMousePointerThatCanHandleThisEvent, value: 11)
    let routing = InputRoutingEvidence(event: event)
    #expect(routing.targetPID == 42)
    #expect(routing.handlingWindowID == 11)
    event.setIntegerValueField(.eventTargetUnixProcessID, value: 0)
    event.setIntegerValueField(.mouseEventWindowUnderMousePointerThatCanHandleThisEvent, value: 0)
    let unknown = InputRoutingEvidence(event: event)
    #expect(unknown.targetPID == nil)
    #expect(unknown.handlingWindowID == nil)
}

@Test func scopeGeometryUsesHalfOpenOwnershipAndDetectsMovement() throws {
    let (source, snapshot) = scopeFixture()
    #expect(InputScopeSnapshot.contains(source.bounds, .init(x: -200, y: 30)))
    #expect(!InputScopeSnapshot.contains(source.bounds, .init(x: 0, y: 30)))
    #expect(!InputScopeSnapshot.contains(source.bounds, .init(x: -200, y: 130)))
    let moved = InputScopeSnapshot(observedNanos: 130, frontmostPID: 42,
                                   applicationLaunchDate: source.applicationLaunchDate,
                                   windows: [.init(id: 11, pid: 42, layer: 0,
                                                    bounds: .init(x: -199, y: 30, width: 200, height: 100))])
    #expect(!moved.sameTarget(as: snapshot, source: source))
    #expect(snapshot.sameTarget(as: snapshot, source: source))
}

@Test func displayAndDesktopScopesDoNotInheritAnApplicationRestriction() throws {
    var (source, snapshot) = scopeFixture(kind: .display)
    let inside = RawInputEvent(sequence: 0, eventNanos: 110, observedNanos: 120, origin: .physical,
                               kind: .pointer, x: -150, y: 70, dx: 3, dy: 4)
    try snapshot.verify(inside, targetPID: 99, source: source)
    var outside = inside; outside.x = 1
    #expect(throws: AstraError.self) { try snapshot.verify(outside, targetPID: 99, source: source) }
    let anotherApplication = InputScopeSnapshot(observedNanos: 140, frontmostPID: 99,
                                                applicationLaunchDate: nil, windows: snapshot.windows)
    #expect(snapshot.sameTarget(as: anotherApplication, source: source))
    source.kind = .desktop
    try snapshot.verify(outside, targetPID: 99, source: source)
}

@Test func producerStopsAreIdempotentWithoutStartingPrivacyAccess() async {
    let input = PhysicalInputMonitor()
    let capture = ScreenCapture()
    await withTaskGroup(of: Void.self) { group in
        for _ in 0..<16 {
            group.addTask { await input.stop() }
            group.addTask { await capture.stop() }
        }
    }
    await input.stop()
    await capture.stop()
}
