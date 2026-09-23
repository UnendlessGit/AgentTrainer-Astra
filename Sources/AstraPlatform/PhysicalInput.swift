import Foundation
import ApplicationServices
import AppKit
import IOKit.hidsystem
import AstraCore

public enum AstraSyntheticInput {
    public static let tag: Int64 = 0x4153545241303031
}

public struct InputObservationBoundary: Sendable {
    /// The conservative beginning of the interval that can no longer be used
    /// as demonstration evidence. This is distinct from detection time.
    public let invalidFromNanos: UInt64
    public let observedNanos: UInt64
    public let message: String
}

struct InputRoutingEvidence: Sendable {
    let targetPID: Int32?
    let handlingWindowID: UInt32?

    init(event: CGEvent? = nil) {
        guard let event else { targetPID = nil; handlingWindowID = nil; return }
        let pid = event.getIntegerValueField(.eventTargetUnixProcessID)
        targetPID = pid > 0 ? Int32(exactly: pid) : nil
        let window = event.getIntegerValueField(.mouseEventWindowUnderMousePointerThatCanHandleThisEvent)
        handlingWindowID = window > 0 ? UInt32(exactly: window) : nil
    }
}

/// WindowServer evidence is collected on the delivery queue, never in the
/// event tap. It describes the observable target; it does not assert that an
/// application consumed a particular event or expose another app's AX tree.
struct InputScopeSnapshot: Equatable, Sendable {
    struct Window: Equatable, Sendable {
        let id: UInt32
        let pid: Int32
        let layer: Int
        let bounds: Rect2D
    }
    let observedNanos: UInt64
    let samplingBeganNanos: UInt64
    let frontmostPID: Int32?
    let applicationLaunchDate: Date?
    let windows: [Window]
    let displays: [UInt32: Rect2D]?

    init(observedNanos: UInt64, frontmostPID: Int32?, applicationLaunchDate: Date?, windows: [Window],
         samplingBeganNanos: UInt64? = nil, displays: [UInt32: Rect2D]? = nil) {
        self.observedNanos = observedNanos; self.samplingBeganNanos = samplingBeganNanos ?? observedNanos
        self.frontmostPID = frontmostPID; self.applicationLaunchDate = applicationLaunchDate; self.windows = windows
        self.displays = displays
    }

    static func current(for source: CaptureSource) throws -> Self {
        let began = MonotonicClock.now
        let beforePID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        guard let raw = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] else {
            throw AstraError("input.scopeUnavailable", "The recording target could not be verified with WindowServer.")
        }
        let windows = raw.compactMap { value -> Window? in
            guard let id = value[kCGWindowNumber as String] as? NSNumber,
                  let pid = value[kCGWindowOwnerPID as String] as? NSNumber,
                  let layer = value[kCGWindowLayer as String] as? NSNumber,
                  let alpha = value[kCGWindowAlpha as String] as? NSNumber, alpha.doubleValue > 0,
                  let data = value[kCGWindowBounds as String] as? NSDictionary,
                  let bounds = CGRect(dictionaryRepresentation: data), Rect2D(bounds).isValid else { return nil }
            return Window(id: id.uint32Value, pid: pid.int32Value, layer: layer.intValue, bounds: Rect2D(bounds))
        }
        let afterPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let launchDate = source.applicationPID.flatMap { NSRunningApplication(processIdentifier: $0)?.launchDate }
        var displayIDs = [CGDirectDisplayID](repeating: 0, count: 32), displayCount: UInt32 = 0
        guard CGGetActiveDisplayList(UInt32(displayIDs.count), &displayIDs, &displayCount) == .success, displayCount <= 16 else {
            throw AstraError("input.displayTopology", "The active display topology could not be verified.")
        }
        let displays = Dictionary(uniqueKeysWithValues: displayIDs.prefix(Int(displayCount)).map { ($0, Rect2D(CGDisplayBounds($0))) })
        return Self(observedNanos: MonotonicClock.now, frontmostPID: beforePID == afterPID ? afterPID : nil,
                    applicationLaunchDate: launchDate, windows: windows, samplingBeganNanos: began, displays: displays)
    }

    var frontmostNormalWindow: Window? {
        // Z order also contains nonactivating, click-through overlays from
        // other processes. They do not determine the active app's key window.
        windows.first { $0.layer == 0 && $0.pid == frontmostPID }
    }

    func verifyTarget(_ source: CaptureSource) throws {
        if source.bindings != nil { try verifyBindings(source) }
        if source.kind == .window || source.kind == .application {
            guard let pid = source.applicationPID, pid > 0, frontmostPID == pid,
                  source.applicationLaunchDate == nil || source.applicationLaunchDate == applicationLaunchDate else {
                throw AstraError("input.targetFocus", "The selected application is no longer in front. The demonstration stopped at this target boundary.")
            }
            if source.kind == .window {
                let foregroundSurface = windows.first { $0.pid == pid && (0..<Int(CGWindowLevelForKey(.dockWindow))).contains($0.layer) }
                guard let target = frontmostNormalWindow, target.id == source.windowID, target.pid == pid,
                      foregroundSurface?.id == source.windowID else {
                    throw AstraError("input.targetWindow", "The selected window is no longer the frontmost normal window. The demonstration stopped at this target boundary.")
                }
            }
        } else if source.kind == .display {
            guard let displayID = source.displayID, CGDisplayIsActive(displayID) != 0,
                  Rect2D(CGDisplayBounds(displayID)) == source.bounds else {
                throw AstraError("input.targetDisplay", "The selected display disconnected or changed its geometry. Choose the target again.")
            }
        } else if source.kind != .desktop {
            throw AstraError("input.scope", "This source has no supported physical-input scope.")
        }
    }

    /// Membership and geometry verification does not require focus, so capture
    /// can perform it independently while the input/control owners check focus.
    func verifyBindings(_ source: CaptureSource) throws {
        let bindings = try source.captureBindings()
        for binding in bindings {
            if binding.kind == .window {
                guard let window = windows.first(where: { $0.id == binding.windowID }),
                      window.pid == binding.applicationPID, window.bounds == binding.bounds else {
                    throw AstraError("capture.bindingChanged", "A bound window moved, resized or disappeared. Choose the source again.")
                }
            } else if let displays {
                guard let id = binding.displayID, displays[id] == binding.bounds else {
                    throw AstraError("capture.bindingChanged", "A bound display disconnected or changed geometry. Choose the source again.")
                }
            }
        }
        if source.kind == .application {
            guard source.applicationLaunchDate == nil || source.applicationLaunchDate == applicationLaunchDate,
                  Set(windows.filter { $0.pid == source.applicationPID && $0.layer == 0 && $0.bounds.width > 1 && $0.bounds.height > 1 }.map(\.id))
                    == Set(bindings.compactMap(\.windowID)) else {
                throw AstraError("capture.applicationTopology", "The application's window set changed. Choose its current windows again.")
            }
        } else if source.kind == .desktop, let displays {
            guard Set(displays.keys) == Set(bindings.compactMap(\.displayID)) else {
                throw AstraError("capture.desktopTopology", "The desktop display set changed. Choose the desktop again.")
            }
        }
    }

    func routedSurfaceID(_ event: RawInputEvent, handlingWindowID: UInt32?, source: CaptureSource) throws -> String? {
        guard [.pointer, .buttonDown, .buttonUp, .scroll].contains(event.kind),
              let x = event.x, let y = event.y else { return nil }
        let point = Point2D(x: x, y: y), bindings = try source.captureBindings()
        if let handlingWindowID, bindings.contains(where: { $0.kind == .window }) {
            return bindings.first { $0.windowID == handlingWindowID && Self.contains($0.bounds, point) }?.id
        }
        let candidates = bindings.filter { Self.contains($0.bounds, point) }
        return candidates.count == 1 ? candidates[0].id : nil
    }

    func verify(_ event: RawInputEvent, targetPID: Int32?, handlingWindowID: UInt32? = nil, source: CaptureSource) throws {
        guard event.origin == .physical else { return }
        if source.kind == .desktop {
            if [.pointer, .buttonDown, .buttonUp, .scroll].contains(event.kind), let x = event.x, let y = event.y,
               let bindings = source.bindings, !bindings.contains(where: { Self.contains($0.bounds, .init(x: x, y: y)) }) {
                throw Self.pointerBoundary()
            }
            return
        }
        if source.kind == .window || source.kind == .application {
            if let targetPID, targetPID > 0, targetPID != source.applicationPID {
                throw AstraError("input.eventTarget", "macOS routed an input event to another application. It was excluded from the demonstration.")
            }
        }
        switch event.kind {
        case .pointer, .buttonDown, .buttonUp, .scroll:
            guard let x = event.x, let y = event.y, x.isFinite, y.isFinite else {
                throw AstraError("input.position", "The input event has no valid pointer position.")
            }
            let point = Point2D(x: x, y: y)
            if source.kind == .display {
                guard Self.contains(source.bounds, point) else { throw Self.pointerBoundary() }
            } else {
                let hit: Window?
                if let handlingWindowID, handlingWindowID != 0 {
                    // At the annotated-session tap, this is the WindowServer
                    // recipient after click-through handling. A resolved but
                    // unknown/mismatching window never falls back to z order.
                    hit = windows.first { $0.id == handlingWindowID }
                } else {
                    // If only the recipient process is known, unrelated
                    // overlays cannot be the recipient. Same-app panels still
                    // participate. With neither annotation, fail closed using
                    // the full visible hit order.
                    hit = windows.first {
                        $0.layer >= 0 && $0.layer < Int(CGWindowLevelForKey(.cursorWindow))
                            && Self.contains($0.bounds, point)
                            && (targetPID == nil || targetPID == 0 || $0.pid == targetPID)
                    }
                }
                guard let hit, Self.contains(hit.bounds, point),
                      hit.pid == source.applicationPID,
                      source.kind != .window || hit.id == source.windowID,
                      source.bindings.map({ $0.contains(where: { $0.windowID == hit.id }) }) ?? true else { throw Self.pointerBoundary() }
            }
        case .keyDown, .keyUp, .keyRepeat, .flags:
            if source.kind == .display {
                guard let target = frontmostNormalWindow, target.pid == frontmostPID,
                      source.bounds.cgRect.intersects(target.bounds.cgRect) else {
                    throw AstraError("input.keyboardBoundary", "Keyboard focus moved outside the selected display. The event was excluded from the demonstration.")
                }
            }
        case .gap: break
        }
    }

    func sameTarget(as other: Self, source: CaptureSource) -> Bool {
        if source.kind == .display || source.kind == .desktop { return true }
        // A moved window creates an ambiguous interval between geometry
        // observations. End it explicitly rather than assigning old pixels to
        // new pointer coordinates. Other windows are checked at both ends.
        guard frontmostPID == other.frontmostPID, applicationLaunchDate == other.applicationLaunchDate else { return false }
        if source.kind == .window {
            return windows.first(where: { $0.id == source.windowID }) == other.windows.first(where: { $0.id == source.windowID })
        }
        if let bindings = source.bindings {
            return bindings.allSatisfy { binding in
                windows.first(where: { $0.id == binding.windowID }) == other.windows.first(where: { $0.id == binding.windowID })
            }
        }
        return true
    }

    static func contains(_ bounds: Rect2D, _ point: Point2D) -> Bool {
        // Desktop pixel/point ownership is half open at adjoining displays.
        bounds.isValid && point.isFinite && point.x >= bounds.x && point.y >= bounds.y
            && point.x < bounds.x + bounds.width && point.y < bounds.y + bounds.height
    }
    private static func pointerBoundary() -> AstraError {
        AstraError("input.pointerBoundary", "The pointer left the observed target or reached an unobserved window. That input was excluded from the demonstration.")
    }
}

public final class PhysicalInputMonitor: @unchecked Sendable {
    public typealias EventHandler = @Sendable ([RawInputEvent]) -> Void
    public typealias FaultHandler = @Sendable (AstraError) -> Void
    private let lock = NSLock()
    private var context: InputTapContext?
    private let keyboardTrust: KeyboardObservationTrust
    private let listenAccess: @Sendable () -> Bool

    public init(keyboardTrust: KeyboardObservationTrust = .shared,
                listenAccess: @escaping @Sendable () -> Bool = { CGPreflightListenEventAccess() }) {
        self.keyboardTrust = keyboardTrust; self.listenAccess = listenAccess
    }
    deinit { context?.requestStop() }

    public func start(source: CaptureSource? = nil, onEvents: @escaping EventHandler, onFault: @escaping FaultHandler,
                      onEmergency: @escaping @Sendable () -> Void,
                      onBoundary: @escaping @Sendable (InputObservationBoundary) -> Void = { _ in }) async throws {
        try keyboardTrust.refreshSynchronously().requireTrusted()
        guard listenAccess() else {
            throw AstraError("permission.inputMonitoring", "Allow Input Monitoring for AgentTrainer Astra before recording controls.")
        }
        let candidate = InputTapContext(source: source, keyboardTrust: keyboardTrust, onEvents: onEvents, onFault: onFault,
                                        onEmergency: onEmergency, onBoundary: onBoundary)
        try lock.withLock {
            guard context == nil else { throw AstraError("input.busy", "Input observation is already running or stopping.") }
            context = candidate
        }
        do {
            try candidate.prepare()
            let thread = Thread { candidate.run() }
            thread.name = "Astra physical input"
            thread.qualityOfService = .userInteractive
            thread.start()
            await candidate.ready.wait()
            try candidate.requireReady()
            try Task.checkCancellation()
        } catch {
            candidate.requestStop()
            // prepare failure starts no thread and completes its own barriers.
            await candidate.finished.wait()
            lock.withLock { if context === candidate { context = nil } }
            throw error
        }
    }

    public func stop() async {
        guard let previous = lock.withLock({ context }) else { return }
        previous.requestStop()
        await previous.finished.wait()
        lock.withLock { if context === previous { context = nil } }
    }

    public static func translate(_ event: CGEvent, sequence: UInt64, observedNanos: UInt64) -> RawInputEvent? {
        guard event.getIntegerValueField(.eventSourceUserData) != AstraSyntheticInput.tag else { return nil }
        let point = event.location
        let common = (event.timestamp, event.flags.rawValue)
        let kind: RawInputKind
        var key: Int?, button: Int?, down: Bool?
        var dx: Double?, dy: Double?, scrollX: Double?, scrollY: Double?
        switch event.type {
        case .keyDown:
            kind = event.getIntegerValueField(.keyboardEventAutorepeat) == 0 ? .keyDown : .keyRepeat
            key = Int(event.getIntegerValueField(.keyboardEventKeycode)); down = true
        case .keyUp: kind = .keyUp; key = Int(event.getIntegerValueField(.keyboardEventKeycode)); down = false
        case .flagsChanged:
            kind = .flags; key = Int(event.getIntegerValueField(.keyboardEventKeycode))
            if let key { down = Self.modifierState(keyCode: key, flags: event.flags) }
        case .leftMouseDown, .rightMouseDown, .otherMouseDown:
            kind = .buttonDown; button = Int(event.getIntegerValueField(.mouseEventButtonNumber)); down = true
        case .leftMouseUp, .rightMouseUp, .otherMouseUp:
            kind = .buttonUp; button = Int(event.getIntegerValueField(.mouseEventButtonNumber)); down = false
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            kind = .pointer
            dx = event.getDoubleValueField(.mouseEventDeltaX); dy = event.getDoubleValueField(.mouseEventDeltaY)
        case .scrollWheel:
            kind = .scroll
            scrollX = event.getDoubleValueField(.scrollWheelEventPointDeltaAxis2)
            scrollY = event.getDoubleValueField(.scrollWheelEventPointDeltaAxis1)
        default: return nil
        }
        return RawInputEvent(sequence: sequence, eventNanos: common.0, observedNanos: observedNanos,
                             origin: .physical, kind: kind, keyCode: key, button: button,
                             x: point.x, y: point.y, dx: dx, dy: dy, scrollX: scrollX, scrollY: scrollY,
                             modifiers: common.1, isDown: down,
                             detail: kind == .scroll ? "scroll units: points; original phase/precision retained in rawPlatformData" : nil,
                             rawPlatformData: event.data as Data?)
    }

    static func modifierState(keyCode: Int, flags: CGEventFlags) -> Bool? {
        // Device-dependent modifier bits belong to this captured event. A HID
        // query here could observe a later physical edge and relabel history.
        let pair: (UInt64, UInt64, CGEventFlags)
        switch keyCode {
        case 56: pair = (UInt64(NX_DEVICELSHIFTKEYMASK), UInt64(NX_DEVICERSHIFTKEYMASK), .maskShift)
        case 60: pair = (UInt64(NX_DEVICERSHIFTKEYMASK), UInt64(NX_DEVICELSHIFTKEYMASK), .maskShift)
        case 59: pair = (UInt64(NX_DEVICELCTLKEYMASK), UInt64(NX_DEVICERCTLKEYMASK), .maskControl)
        case 62: pair = (UInt64(NX_DEVICERCTLKEYMASK), UInt64(NX_DEVICELCTLKEYMASK), .maskControl)
        case 55: pair = (UInt64(NX_DEVICELCMDKEYMASK), UInt64(NX_DEVICERCMDKEYMASK), .maskCommand)
        case 54: pair = (UInt64(NX_DEVICERCMDKEYMASK), UInt64(NX_DEVICELCMDKEYMASK), .maskCommand)
        case 58: pair = (UInt64(NX_DEVICELALTKEYMASK), UInt64(NX_DEVICERALTKEYMASK), .maskAlternate)
        case 61: pair = (UInt64(NX_DEVICERALTKEYMASK), UInt64(NX_DEVICELALTKEYMASK), .maskAlternate)
        case 57: return flags.contains(.maskAlphaShift)
        case 63: return flags.contains(.maskSecondaryFn)
        default: return nil
        }
        let sideBits = flags.rawValue & (pair.0 | pair.1)
        // Some remappers supply aggregate flags only. Preserve ambiguity
        // instead of guessing which physical side changed.
        if sideBits == 0 && flags.contains(pair.2) { return nil }
        return flags.rawValue & pair.0 != 0
    }
}

private final class InputTapContext: @unchecked Sendable {
    let ready = AsyncCompletion()
    let finished = AsyncCompletion()
    private let lock = NSLock()
    private let drainQueue = DispatchQueue(label: "astra.input.delivery", qos: .userInitiated)
    private let onEvents: PhysicalInputMonitor.EventHandler
    private let onFault: PhysicalInputMonitor.FaultHandler
    private let onEmergency: @Sendable () -> Void
    private let onBoundary: @Sendable (InputObservationBoundary) -> Void
    private let scopeSource: CaptureSource?
    private var tap: CFMachPort?
    private var runLoop: CFRunLoop?
    private var stopping = false
    private struct ObservedInput { let value: RawInputEvent; let routing: InputRoutingEvidence }
    private var events: [ObservedInput] = []
    private var retainedBytes = 0
    private var sequence: UInt64 = 0
    private var timer: DispatchSourceTimer?
    private var startupFailure: AstraError?
    // Only prepare (before the thread starts) and drainQueue access this.
    private var scopeSnapshot: InputScopeSnapshot?
    private var scopeFailed = false
    private let keyboardTrust: KeyboardObservationTrust
    private var keyboardContinuity: KeyboardObservationContinuity?

    init(source: CaptureSource?, keyboardTrust: KeyboardObservationTrust, onEvents: @escaping PhysicalInputMonitor.EventHandler,
         onFault: @escaping PhysicalInputMonitor.FaultHandler, onEmergency: @escaping @Sendable () -> Void,
         onBoundary: @escaping @Sendable (InputObservationBoundary) -> Void) {
        self.scopeSource = source; self.onEvents = onEvents; self.onFault = onFault
        self.onEmergency = onEmergency; self.onBoundary = onBoundary
        self.keyboardTrust = keyboardTrust
    }

    func prepare() throws {
        do {
            keyboardContinuity = try KeyboardObservationContinuity(keyboardTrust.refreshSynchronously())
            if let scopeSource {
                let snapshot = try InputScopeSnapshot.current(for: scopeSource)
                try snapshot.verifyTarget(scopeSource)
                scopeSnapshot = snapshot
            }
        } catch {
            ready.finish(); finished.finish(); throw error
        }
        let types: [CGEventType] = [.keyDown, .keyUp, .flagsChanged, .leftMouseDown, .leftMouseUp,
                                    .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp,
                                    .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged, .scrollWheel]
        let mask = types.reduce(CGEventMask(0)) { $0 | (CGEventMask(1) << $1.rawValue) }
        // This passive stage sees the OS's application routing annotations;
        // the earlier session stage can report no recipient yet.
        tap = CGEvent.tapCreate(tap: .cgAnnotatedSessionEventTap, place: .headInsertEventTap,
                               options: .listenOnly, eventsOfInterest: mask,
                               callback: { _, type, event, pointer in
            guard let pointer else { return Unmanaged.passUnretained(event) }
            let context = Unmanaged<InputTapContext>.fromOpaque(pointer).takeUnretainedValue()
            context.receive(type: type, event: event)
            return Unmanaged.passUnretained(event)
        }, userInfo: Unmanaged.passUnretained(self).toOpaque())
        guard tap != nil else {
            ready.finish(); finished.finish()
            throw AstraError("permission.inputTap", "Input observation could not start. Check Input Monitoring permission and relaunch Astra after granting access.")
        }
    }

    func run() {
        autoreleasepool {
            guard let tap, let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0) else {
                let error = AstraError("input.runLoop", "The input observation run loop could not start.")
                lock.withLock { startupFailure = error; stopping = true }
                if let tap { CFMachPortInvalidate(tap) }
                onFault(error)
                ready.finish(); finished.finish(); return
            }
            let loop = CFRunLoopGetCurrent()!
            let shouldRun = lock.withLock {
                runLoop = loop
                return !stopping
            }
            if shouldRun {
                CFRunLoopAddSource(loop, source, .commonModes)
                CGEvent.tapEnable(tap: tap, enable: true)
                do { try seedPhysicalState() }
                catch {
                    let failure = (error as? AstraError) ?? AstraError("input.keyboardTrust", error.localizedDescription)
                    lock.withLock { startupFailure = failure }
                    onFault(failure); requestStop()
                }
                let timer = DispatchSource.makeTimerSource(queue: drainQueue)
                timer.schedule(deadline: .now(), repeating: .milliseconds(10), leeway: .milliseconds(2))
                timer.setEventHandler { [weak self] in self?.drain() }
                self.timer = timer; timer.resume()
                ready.finish()
                // A stop just before entering CFRunLoopRun must not be lost.
                // Bounded turns also guarantee an idle listener can exit.
                while !lock.withLock({ stopping }) {
                    CFRunLoopRunInMode(.defaultMode, 0.1, false)
                }
                CFRunLoopRemoveSource(loop, source, .commonModes)
            } else { ready.finish() }
            CFMachPortInvalidate(tap)
            timer?.cancel(); timer = nil
            drainQueue.sync { drain() }
            lock.withLock { runLoop = nil }
            finished.finish()
        }
    }

    func requestStop() {
        let loop = lock.withLock { stopping = true; return runLoop }
        if let loop { CFRunLoopStop(loop); CFRunLoopWakeUp(loop) }
    }

    func requireReady() throws {
        try lock.withLock {
            if let startupFailure { throw startupFailure }
            if stopping { throw CancellationError() }
        }
    }

    private func receive(type: CGEventType, event: CGEvent) {
        let now = MonotonicClock.now
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            onFault(AstraError("input.interrupted", "macOS interrupted input observation; the recording must mark this gap."))
            requestStop(); return
        }
        guard event.getIntegerValueField(.eventSourceUserData) != AstraSyntheticInput.tag else { return }
        let emergency = type == .keyDown && event.getIntegerValueField(.keyboardEventKeycode) == 53
            && event.flags.contains([.maskControl, .maskAlternate, .maskCommand])
        let overflow = lock.withLock {
            guard !stopping, let value = PhysicalInputMonitor.translate(event, sequence: sequence, observedNanos: now) else { return false }
            let bytes = 256 + (value.rawPlatformData?.count ?? 0)
            guard events.count < 2_048, retainedBytes + bytes <= 4 * 1_024 * 1_024 else { return true }
            sequence += 1; events.append(ObservedInput(value: value, routing: InputRoutingEvidence(event: event))); retainedBytes += bytes
            return false
        }
        if overflow {
            onFault(AstraError("input.backpressure", "Input storage could not keep up. Previously captured events have been preserved."))
            requestStop()
        }
        // Capture the observed time and enqueue the physical edge before a
        // stop callback establishes the recording cutoff.
        if emergency { onEmergency() }
    }

    private func seedPhysicalState() throws {
        let before = keyboardTrust.snapshot(); try before.requireTrusted()
        let now = MonotonicClock.now
        let pointer = CGEvent(source: nil)?.location ?? .zero
        let flags = CGEventSource.flagsState(.hidSystemState).rawValue
        var seed: [RawInputEvent] = [.init(sequence: 0, eventNanos: now, observedNanos: now,
                                          origin: .reconciliation, kind: .pointer, x: pointer.x, y: pointer.y,
                                          modifiers: flags, detail: "initial physical state")]
        for key in 0...127 where CGEventSource.keyState(.hidSystemState, key: CGKeyCode(key)) {
            seed.append(.init(sequence: 0, eventNanos: now, observedNanos: now, origin: .reconciliation,
                              kind: .keyDown, keyCode: key, modifiers: flags, isDown: true, detail: "initial held key"))
        }
        for button in 0...31 where CGEventSource.buttonState(.hidSystemState, button: CGMouseButton(rawValue: UInt32(button))!) {
            seed.append(.init(sequence: 0, eventNanos: now, observedNanos: now, origin: .reconciliation,
                              kind: .buttonDown, button: button, x: pointer.x, y: pointer.y, isDown: true,
                              detail: "initial held button"))
        }
        // Reconciliation is an observed snapshot, not a physical edge. Do not
        // backdate the result to before the hardware queries completed.
        let completed = MonotonicClock.now
        let after = keyboardTrust.snapshot(); try after.requireTrusted()
        guard after.interruptionGeneration == before.interruptionGeneration else {
            throw AstraError("input.keyboardInterrupted", "Keyboard observation changed while its initial state was being read.")
        }
        lock.withLock {
            for var event in seed {
                event.sequence = sequence; event.eventNanos = completed; event.observedNanos = completed
                sequence += 1; events.append(ObservedInput(value: event, routing: InputRoutingEvidence())); retainedBytes += 256
            }
        }
    }

    private func drain() {
        let batch = lock.withLock {
            let batch = events; events.removeAll(keepingCapacity: true); retainedBytes = 0
            return batch
        }
        guard !scopeFailed else { return }
        if let boundary = keyboardContinuity?.inspect(keyboardTrust.snapshot()) {
            scopeFailed = true; requestStop()
            let gap = lock.withLock { () -> RawInputEvent in
                let gap = RawInputEvent(sequence: sequence, eventNanos: boundary.observedNanos, observedNanos: boundary.observedNanos,
                    origin: .boundary, kind: .gap, detail: boundary.message)
                sequence += 1; events.removeAll(); retainedBytes = 0; return gap
            }
            onEvents([gap]); onBoundary(boundary)
            onFault(AstraError("input.keyboardInterrupted", boundary.message))
            return
        }
        var routed = batch.map(\.value)
        if let scopeSource, let previous = scopeSnapshot {
            // Verify every nonempty delivery batch and at least ten times per
            // second while idle. A blocked WindowServer never blocks the tap;
            // its bounded ring will stop observation on backpressure.
            let now = MonotonicClock.now
            guard !batch.isEmpty || now >= previous.observedNanos + 100_000_000 else { return }
            do {
                let current = try InputScopeSnapshot.current(for: scopeSource)
                try current.verifyTarget(scopeSource)
                guard current.sameTarget(as: previous, source: scopeSource) else {
                    throw AstraError("input.targetChanged", "The observed target moved or focus changed during input collection. This interval was excluded from demonstration training.")
                }
                for (index, event) in batch.enumerated() {
                    try previous.verify(event.value, targetPID: event.routing.targetPID,
                                        handlingWindowID: event.routing.handlingWindowID, source: scopeSource)
                    try current.verify(event.value, targetPID: event.routing.targetPID,
                                       handlingWindowID: event.routing.handlingWindowID, source: scopeSource)
                    let before = try previous.routedSurfaceID(event.value, handlingWindowID: event.routing.handlingWindowID, source: scopeSource)
                    let after = try current.routedSurfaceID(event.value, handlingWindowID: event.routing.handlingWindowID, source: scopeSource)
                    if before == after { routed[index].surfaceID = before }
                }
                scopeSnapshot = current
            } catch {
                scopeFailed = true
                requestStop()
                let detected = MonotonicClock.now
                let gap = lock.withLock {
                    let gap = RawInputEvent(sequence: sequence, eventNanos: detected, observedNanos: detected,
                                            origin: .boundary, kind: .gap, detail: error.localizedDescription)
                    sequence += 1; events.removeAll(); retainedBytes = 0
                    return gap
                }
                onEvents([gap])
                onBoundary(.init(invalidFromNanos: previous.samplingBeganNanos, observedNanos: detected,
                                 message: error.localizedDescription))
                onFault((error as? AstraError) ?? AstraError("input.scope", error.localizedDescription))
                return
            }
        }
        if !routed.isEmpty { onEvents(routed) }
    }
}
