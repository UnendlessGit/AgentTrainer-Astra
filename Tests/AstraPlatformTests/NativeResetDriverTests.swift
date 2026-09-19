import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

private final class ResetDriverProbe: @unchecked Sendable {
    let gate = ControlTestGate()
    private let lock = NSLock()
    private var verifications = 0, observationCalls = 0
    private var mode = "valid"
    private var sessions: [UUID] = []
    var scopeChecks: Int { lock.withLock { verifications } }
    var observations: Int { lock.withLock { observationCalls } }
    var sessionIDs: Set<UUID> { lock.withLock { Set(sessions) } }
    func setMode(_ mode: String) { lock.withLock { self.mode = mode } }
    func record(_ event: NativeControlEvent) { lock.withLock { sessions.append(event.sessionID) } }
    func verify(_ source: CaptureSource, _ scope: ControlScope) async throws -> UInt64 {
        let current = lock.withLock { verifications += 1; return mode }
        if current == "blockVerify" { await gate.wait() }
        if current == "changed" { throw AstraError("fixture.geometry", "The captured source moved or its launch identity changed.") }
        if current == "future" { return MonotonicClock.now + 1_000_000_000 }
        if current == "stale" { return MonotonicClock.now - NativeResetDriver.scopeFreshnessNanos }
        guard source.id == scope.surfaces.first?.id, source.bounds == scope.surfaces.first?.globalBounds else {
            throw AstraError("fixture.source", "The capture source and control scope disagree.")
        }
        return MonotonicClock.now
    }
    func observe(_ context: ResetContext) async throws -> ResetObservationSnapshot {
        let current = lock.withLock { observationCalls += 1; return mode }
        if current == "blockObserve" { await gate.wait() }
        let now = MonotonicClock.now
        var surface = context.scope.surfaces[0]
        if current == "foreignCoverage" { surface.globalBounds.x += 1 }
        return .init(context: context, observedNanos: now, sourceCoverage: [
            .init(sourceObservationID: UUID(), surface: surface, eventNanos: now, observedNanos: now,
                  throughNanos: now, verifiedAtNanos: now, kind: .frame)
        ], readings: [])
    }
}
private struct ResetDriverSetup {
    let probe = ResetDriverProbe()
    let runtime: ControlRuntimeFixture
    let owner = NativeControlOwner()
    let context: ResetContext
    let source: CaptureSource
    init(runtime: ControlRuntimeFixture = .init(earlyTerminal: true)) throws {
        self.runtime = runtime
        let config = try controlConfiguration(root: FileManager.default.temporaryDirectory)
        context = try .init(nextEpisodeID: UUID(), environmentID: UUID(), scope: config.scope)
        let surface = config.scope.surfaces[0]
        source = CaptureSource(id: surface.id, name: "Owned fixture capture", kind: .display, bounds: surface.globalBounds,
                               pixelWidth: surface.pixelWidth, pixelHeight: surface.pixelHeight)
    }
    func driver(priorJoined: Bool = true) -> NativeResetDriver {
        NativeResetDriver(environmentID: context.environmentID, source: source,
            observations: { try await probe.observe($0) }, priorOwnersJoined: { priorJoined }, owner: owner,
            runtimeFactory: runtime.factory, recoveryDirectory: FileManager.default.temporaryDirectory,
            scopeVerifier: { try await probe.verify($0, $1) }, onControlEvent: { probe.record($0) })
    }
    func packet(_ context: ResetContext) -> ActionPacket {
        .init(runID: context.resetID, sequence: 0, observationID: UUID(), geometryRevision: context.scope.surfaces[0].geometryRevision,
              executeAtNanos: MonotonicClock.now, durationMs: 1, commands: [.init(offsetMs: 0, operation: .keyDown, keyCode: 36)])
    }
}

@Test func nativeResetManualPreparationUsesOnlyOwnedObservationAndNeverCreatesAHelper() async throws {
    let fixture = try ResetDriverSetup(), driver = fixture.driver()
    let binding = try await driver.prepare(context: fixture.context, capabilities: .init())
    #expect(binding.priorOwnersJoined && binding.context == fixture.context && fixture.runtime.startCount == 0)
    try driver.checkHealth(resetID: fixture.context.resetID)
    let before = try await driver.observe(context: fixture.context)
    #expect(before.readings.isEmpty && before.sourceCoverage.count == 1)
    let proof = await driver.release(context: fixture.context)
    #expect(proof.confirmed && fixture.runtime.startCount == 0)
    // The final Ready observation is produced after joined release.
    let after = try await driver.observe(context: fixture.context)
    #expect(after.observedNanos >= proof.observedNanos && fixture.probe.observations == 2)
}

@Test func nativeResetEachRetryCreatesANewSessionAndKeepsResetActionsOutsidePolicyIdentity() async throws {
    let fixture = try ResetDriverSetup(), driver = fixture.driver()
    // ResetRunner retries the same reset context; physical helper instances
    // must still be fresh and restart the reset-only sequence at zero.
    for context in [fixture.context, fixture.context] {
        _ = try await driver.prepare(context: context, capabilities: .init(keyCodes: [36]))
        let receipt = try await driver.execute(fixture.packet(context), context: context)
        #expect(receipt.status == .executed && receipt.runID == context.resetID && receipt.runID != context.nextEpisodeID)
        #expect(await driver.release(context: context).confirmed)
    }
    #expect(fixture.runtime.startCount == 2 && fixture.probe.sessionIDs.count == 2 && fixture.owner.priorCleanupJoined)
}

@Test(arguments: ["prior", "changed", "stale", "future"])
func nativeResetRejectsUnjoinedOwnersAndUntrustworthyScopeBeforeArm(mode: String) async throws {
    let fixture = try ResetDriverSetup(), driver = fixture.driver(priorJoined: mode != "prior")
    fixture.probe.setMode(mode)
    await #expect(throws: AstraError.self) { try await driver.prepare(context: fixture.context, capabilities: .init(keyCodes: [36])) }
    #expect(fixture.runtime.startCount == 0 && fixture.runtime.sentPackets.isEmpty)
    #expect(await driver.release(context: fixture.context).confirmed)
}

@Test func nativeResetScopeChangeStopsAdmissionAndForeignCoverageCannotSatisfyReady() async throws {
    let fixture = try ResetDriverSetup(), driver = fixture.driver()
    _ = try await driver.prepare(context: fixture.context, capabilities: .init(keyCodes: [36]))
    fixture.probe.setMode("foreignCoverage")
    await #expect(throws: AstraError.self) { try await driver.observe(context: fixture.context) }
    fixture.probe.setMode("changed")
    try await controlWait { fixture.runtime.log.contains("disarm") }
    await #expect(throws: AstraError.self) { try await driver.execute(fixture.packet(fixture.context), context: fixture.context) }
    let proof = await driver.release(context: fixture.context)
    #expect(fixture.runtime.sentPackets.isEmpty && proof.confirmed)
    fixture.probe.setMode("valid")
    // Returning to the old geometry cannot erase an observed reset boundary.
    await #expect(throws: AstraError.self) { try await driver.observe(context: fixture.context) }
}

@Test func nativeResetCancellationJoinsLatePreparationBeforeAllowingAnotherOwner() async throws {
    let fixture = try ResetDriverSetup(), driver = fixture.driver()
    fixture.probe.setMode("blockVerify")
    let preparing = Task { try await driver.prepare(context: fixture.context, capabilities: .init(keyCodes: [36])) }
    try await controlWait { fixture.probe.scopeChecks > 0 }
    preparing.cancel()
    let finished = ControlTestFlag()
    let releasing = Task { defer { finished.set() }; return await driver.release(context: fixture.context) }
    try await Task.sleep(for: .milliseconds(10))
    #expect(!finished.value && fixture.runtime.startCount == 0)
    fixture.probe.gate.open()
    await #expect(throws: CancellationError.self) { try await preparing.value }
    #expect(await releasing.value.confirmed && fixture.runtime.startCount == 0)
}

@Test func nativeResetReleaseJoinsOwnedObservationAndForeignReleaseDoesNotBorrowItsProof() async throws {
    let fixture = try ResetDriverSetup(), driver = fixture.driver()
    _ = try await driver.prepare(context: fixture.context, capabilities: .init())
    fixture.probe.setMode("blockObserve")
    let observing = Task { try await driver.observe(context: fixture.context) }
    try await controlWait { fixture.probe.observations > 0 }
    let done = ControlTestFlag()
    let releasing = Task { defer { done.set() }; return await driver.release(context: fixture.context) }
    try await Task.sleep(for: .milliseconds(10))
    #expect(!done.value)
    let foreign = try ResetContext(nextEpisodeID: UUID(), environmentID: fixture.context.environmentID, scope: fixture.context.scope)
    #expect(!(await driver.release(context: foreign)).confirmed)
    await #expect(throws: AstraError.self) { try await driver.prepare(context: foreign, capabilities: .init()) }
    fixture.probe.gate.open()
    await #expect(throws: CancellationError.self) { try await observing.value }
    #expect(await releasing.value.confirmed)
}

@Test func nativeResetRepeatedReleaseNeverUpgradesUnconfirmedCleanup() async throws {
    let fixture = try ResetDriverSetup(runtime: .init(exitStatus: 7)), driver = fixture.driver()
    _ = try await driver.prepare(context: fixture.context, capabilities: .init(keyCodes: [36]))
    let proof = await driver.release(context: fixture.context)
    #expect(!proof.confirmed)
    let again = await driver.release(context: fixture.context)
    #expect(!again.confirmed && again.observedNanos == proof.observedNanos && again.issue == proof.issue)
}
