import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

private final class SecureProbeState: @unchecked Sendable {
    private let lock = NSLock()
    private var secure = false
    private var time: UInt64 = 1_000_000_000
    private var samples = 0
    private var reads = 0
    var now: UInt64 { lock.withLock { time } }
    var probeCount: Int { lock.withLock { samples } }
    var hidReads: Int { lock.withLock { reads } }
    func setSecure(_ value: Bool) { lock.withLock { secure = value } }
    func advance(_ nanos: UInt64) { lock.withLock { time += nanos } }
    @MainActor func query() -> Bool {
        #expect(Thread.isMainThread)
        return lock.withLock { samples += 1; return secure }
    }
    func hid() -> ControlState {
        lock.withLock {
            reads += 1
            var value = ControlState(); value.valid = true; value.pointer = .init(x: 50, y: 50); value.observedNanos = time
            return value
        }
    }
    func trust() -> KeyboardObservationTrust {
        KeyboardObservationTrust(automaticSampling: false, clock: { self.now }, probe: { self.query() })
    }
}

@Test @MainActor func secureInputProbeUsesMainThreadAndCachedReadsDoNotInvokeIt() async throws {
    let state = SecureProbeState(), trust = KeyboardObservationTrust(automaticSampling: false, probe: { false })
    let checked = state.trust()
    try await withThrowingTaskGroup(of: Void.self) { group in
        for _ in 0..<16 { group.addTask { try checked.refreshSynchronously().requireTrusted() } }
        try await group.waitForAll()
    }
    #expect(state.probeCount == 16)
    for _ in 0..<1000 { #expect(checked.snapshot().trusted) }
    #expect(state.probeCount == 16)
    #expect(!trust.snapshot().trusted) // An unprimed probe cannot authorize input.
}

@Test @MainActor func secureInputRejectsArmingAndPhysicalTrustDespiteGrantedPermissions() throws {
    let state = SecureProbeState(); state.setSecure(true)
    let trust = state.trust()
    let backend = try CGEventControlBackend(keyboardTrust: trust, permissionProbe: { (true, true, true) }, hidProbe: { state.hid() })
    let request = ArmRequest(runID: UUID(), scope: .init(surfaces: [SurfaceDescriptor(id: "fixture",
        globalBounds: .init(x: 0, y: 0, width: 100, height: 100), pixelWidth: 100, pixelHeight: 100)], wholeDesktop: true),
        capabilities: .init(keyCodes: [0]))
    do { _ = try backend.prepare(request); Issue.record("Secure Input incorrectly allowed arming") }
    catch { #expect((error as? AstraError)?.code == "input.secureInput") }
    #expect(!backend.physicalState().valid && !backend.cleanupPhysicalState().valid)
    #expect(state.hidReads == 0)
    state.setSecure(false); _ = trust.refreshSynchronously()
    #expect(backend.cleanupPhysicalState().valid && state.hidReads == 1)
}

@Test @MainActor func secureInputRejectsMonitorBeforeCreatingAnEventTap() async throws {
    let state = SecureProbeState(); state.setSecure(true)
    let monitor = PhysicalInputMonitor(keyboardTrust: state.trust(), listenAccess: { true })
    do {
        try await monitor.start(onEvents: { _ in }, onFault: { _ in }, onEmergency: {})
        Issue.record("Secure Input incorrectly allowed keyboard monitoring")
    } catch { #expect((error as? AstraError)?.code == "input.secureInput") }
    await monitor.stop()
}

@Test @MainActor func secureInputStartingDuringHIDReadInvalidatesCleanupSnapshot() throws {
    let state = SecureProbeState(), trust = state.trust()
    _ = trust.refreshSynchronously()
    let backend = try CGEventControlBackend(keyboardTrust: trust, permissionProbe: { (true, true, true) }, hidProbe: {
        let result = state.hid()
        state.setSecure(true)
        return result
    })
    #expect(!backend.cleanupPhysicalState().valid)
    #expect(state.hidReads == 1 && !trust.snapshot().trusted)
}

@Test @MainActor func secureInputRejectsNativeCleanupBeforeAnyEventPosting() throws {
    let state = SecureProbeState(); state.setSecure(true)
    let backend = try CGEventControlBackend(keyboardTrust: state.trust(), permissionProbe: { (true, true, true) }, hidProbe: { state.hid() })
    do {
        try backend.post(.init(operation: .keyUp, keyCode: 0, location: .zero, delta: .zero, heldKeys: [], heldButtons: [], cleanup: true))
        Issue.record("Secure Input incorrectly permitted owned input release")
    } catch { #expect((error as? AstraError)?.code == "input.secureInput") }
    #expect(state.hidReads == 0)
}

@Test @MainActor func recordingTrustGapStartsAtLastTrustedSampleEvenAfterSecureInputClears() throws {
    let state = SecureProbeState(), trust = state.trust()
    var continuity = try KeyboardObservationContinuity(trust.refreshSynchronously())
    state.advance(10_000_000)
    #expect(continuity.inspect(trust.refreshSynchronously()) == nil)
    let lastTrusted = state.now
    state.advance(10_000_000); state.setSecure(true); _ = trust.refreshSynchronously()
    state.advance(10_000_000); state.setSecure(false); _ = trust.refreshSynchronously()
    let interrupted = continuity.inspect(trust.snapshot())
    let boundary = try #require(interrupted)
    #expect(boundary.invalidFromNanos == lastTrusted && boundary.observedNanos == state.now)
    #expect(boundary.invalidFromNanos < boundary.observedNanos)
    #expect(continuity.inspect(trust.snapshot()) == nil) // One terminal boundary.
}

@Test @MainActor func staleKeyboardProofCannotSilentlyResumeAnExistingRecording() throws {
    let state = SecureProbeState(), trust = state.trust()
    var continuity = try KeyboardObservationContinuity(trust.refreshSynchronously())
    let lastTrusted = state.now
    state.advance(160_000_000)
    #expect(!trust.snapshot().trusted)
    _ = trust.refreshSynchronously()
    #expect(trust.snapshot().trusted)
    let interrupted = continuity.inspect(trust.snapshot())
    #expect(try #require(interrupted).invalidFromNanos == lastTrusted)
}

private final class SecureVirtualBackend: ControlInputBackend, @unchecked Sendable {
    let native: CGEventControlBackend
    let trust: KeyboardObservationTrust
    private let lock = NSLock()
    private var emissions: [InputEmission] = []
    init(state: SecureProbeState, trust: KeyboardObservationTrust) throws {
        self.trust = trust
        native = try CGEventControlBackend(keyboardTrust: trust, permissionProbe: { (true, true, true) }, hidProbe: { state.hid() })
    }
    func prepare(_ request: ArmRequest) throws -> ControlState { try trust.refreshSynchronously().requireTrusted(); return native.physicalState() }
    func physicalState() -> ControlState { native.physicalState() }
    func cleanupPhysicalState() -> ControlState { native.cleanupPhysicalState() }
    func checkHealth() throws { try trust.snapshot().requireTrusted() }
    func validate(_ scope: ControlScope, pointer: Point2D?) throws { try checkHealth() }
    func post(_ emission: InputEmission) throws {
        try trust.snapshot().requireTrusted()
        lock.withLock { emissions.append(emission) } // Never delegates to native posting.
    }
    var posted: [InputEmission] { lock.withLock { emissions } }
}

@Test @MainActor func secureInputKeepsOwnedCleanupAndLeasePendingUntilTrustReturns() async throws {
    let state = SecureProbeState(), trust = state.trust()
    let backend = try SecureVirtualBackend(state: state, trust: trust)
    let lease = FileManager.default.temporaryDirectory.appendingPathComponent("AstraSecureLease-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: lease) }
    let executor = InputExecutor(backend: backend, clock: { state.now }, automaticScheduling: false, desktopLockURL: lease)
    let run = UUID()
    try executor.arm(.init(runID: run, scope: .init(surfaces: [SurfaceDescriptor(id: "fixture",
        globalBounds: .init(x: 0, y: 0, width: 100, height: 100), pixelWidth: 100, pixelHeight: 100)], wholeDesktop: true), capabilities: .init(keyCodes: [0])))
    try executor.execute(.init(runID: run, sequence: 0, observationID: UUID(), geometryRevision: 0, executeAtNanos: state.now,
        durationMs: 10, commands: [.init(offsetMs: 0, operation: .keyDown, keyCode: 0)]))
    executor.service()
    state.setSecure(true); state.advance(10_000_000); _ = trust.refreshSynchronously()
    executor.checkWatchdog()
    let stopped = ContinuousClock.now.advanced(by: .seconds(2))
    while (executor.currentRunID != nil || executor.state().keys != [0]), ContinuousClock.now < stopped { try await Task.sleep(for: .milliseconds(5)) }
    #expect(executor.currentRunID == nil && !executor.cleanupSettled && executor.state().keys == [0])
    #expect(backend.posted.map(\.operation) == [.keyDown])
    #expect(throws: AstraError.self) { _ = try DesktopControlLock(url: lease) }
    state.setSecure(false); state.advance(10_000_000); _ = trust.refreshSynchronously()
    executor.checkWatchdog()
    let settled = ContinuousClock.now.advanced(by: .seconds(2))
    while !executor.cleanupSettled, ContinuousClock.now < settled { try await Task.sleep(for: .milliseconds(5)) }
    #expect(executor.cleanupSettled && backend.posted.map(\.operation) == [.keyDown, .keyUp])
    let acquired = try DesktopControlLock(url: lease); withExtendedLifetime(acquired) {}
}
