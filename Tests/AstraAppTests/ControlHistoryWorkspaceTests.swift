import Foundation
import Testing
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

private final class HistoryGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var pending: [CheckedContinuation<Void, Never>] = []
    func wait() async { await withCheckedContinuation { continuation in
        let ready = lock.withLock { if opened { return true }; pending.append(continuation); return false }
        if ready { continuation.resume() }
    } }
    func open() { let callbacks = lock.withLock { opened = true; let result = pending; pending = []; return result }; callbacks.forEach { $0.resume() } }
}
private final class HistoryProbe: @unchecked Sendable {
    private let lock = NSLock(); private var didStart = false
    var started: Bool { lock.withLock { didStart } }
    func mark() { lock.withLock { didStart = true } }
}
@MainActor private struct HistoryWorkspaceFixture {
    let root: URL, leaseURL: URL
    let store: LibraryStore
    let owner = NativeControlOwner()
    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraHistoryWorkspace-" + UUID().uuidString)
        leaseURL = root.appendingPathComponent("desktop.lock")
        store = try LibraryStore(root: root)
    }
    func warning() async throws -> ControlHistoryReview {
        let run = UUID(), directory = root.appendingPathComponent("Runs/\(run.uuidString.lowercased())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(JSONValue.object(["runID": .string(run.uuidString), "cleanupConfirmed": .bool(false)]))
            .write(to: directory.appendingPathComponent("results.json"))
        try await store.inspectPriorInferenceRuns()
        return try #require(await store.snapshot().issues.compactMap(\.controlHistory).first)
    }
    func model(inference: InferenceCoordinator? = nil,
               inspection: @escaping @Sendable (LibraryStore) async throws -> Void = WorkspaceModel.inspectControlHistory) -> WorkspaceModel {
        WorkspaceModel(inferenceCoordinator: inference, historyStore: store, historyRoot: root, controlOwner: owner,
                       desktopLeaseURL: leaseURL, controlPreflightInspection: inspection)
    }
    func native() throws -> NativeControlSession {
        let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 32, height: 32), pixelWidth: 32, pixelHeight: 32)
        let configuration = try NativeControlConfiguration(runID: UUID(), scope: .init(surfaces: [surface]), capabilities: .init(keyCodes: [0]), recoveryDirectory: root)
        return NativeControlSession(configuration: configuration, owner: owner, runtimeFactory: .init(protectsPhysicalInputs: false) { _, _ in
            .init(start: { .init(kind: "hello", sequence: 0, payload: .object(["role": .string("control"), "protocolVersion": .integer(1)])) },
                request: { kind, _, run, _ in
                    .init(kind: "ack", sequence: 1, requestID: UUID(), runID: run, payload: .object(kind == "arm"
                        ? ["armed": .bool(true), "nextPacketSequence": .integer(0)]
                        : kind == "disarm" ? ["stopped": .bool(true), "cleanupSettled": .bool(true)] : [:]))
                }, shutdown: { 0 })
        })
    }
    func inference(_ probe: HistoryProbe) -> InferenceCoordinator {
        let dependencies = InferenceDependencies(runtime: { _, _, _ in
            .init(start: { probe.mark(); throw AstraError("fixture.unexpected", "History gating should prevent runtime startup.") },
                  request: { _, _, _, _, _ in throw AstraError("fixture.unexpected", "No runtime request expected.") }, shutdown: { 0 })
        }, capture: { _ in .init(start: { _, _ in probe.mark(); throw AstraError("fixture.unexpected", "History gating should prevent capture.") }, stop: {}) },
            activate: { _ in }, countdownSeconds: 0, controlOwner: owner)
        return InferenceCoordinator(store: store, root: root, dependencies: dependencies)
    }
    var source: CaptureSource {
        .init(id: "display:1", name: "Owned fixture", kind: .display, displayID: 1,
              bounds: .init(x: 0, y: 0, width: 32, height: 32), pixelWidth: 32, pixelHeight: 32)
    }
    func artifacts() async throws -> (AgentDocument, CheckpointDocument) {
        let agent = AgentDocument(name: "History fixture")
        let checkpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: nil, name: "Policy", kind: "initial",
            trainingStep: 0, policySignature: String(repeating: "a", count: 64), parameterCount: 1)
        try await store.save(agent); try await store.saveCheckpoint(checkpoint); return (agent, checkpoint)
    }
}
@MainActor private func historyWait(_ condition: () -> Bool) async throws {
    let deadline = ContinuousClock.now + .seconds(3)
    while !condition() { guard ContinuousClock.now < deadline else { throw AstraError("fixture.timeout", "History workspace fixture timed out.") }; try await Task.sleep(for: .milliseconds(2)) }
}

@Test @MainActor func historyAcknowledgementCannotBypassACurrentNativeOwner() async throws {
    let fixture = try HistoryWorkspaceFixture(); defer { try? FileManager.default.removeItem(at: fixture.root) }
    let review = try await fixture.warning(), model = fixture.model(), session = try fixture.native()
    await model.refreshControlHistory(); _ = try await session.start()
    await #expect(throws: AstraError.self) { try await model.acknowledgeControlHistory(review) }
    #expect(try await fixture.store.snapshot().issues.contains { $0.controlHistory?.id == review.id })
    _ = await session.shutdown()
    try await model.acknowledgeControlHistory(review)
    #expect(model.pendingControlHistory.isEmpty && fixture.owner.priorCleanupJoined)
}

@Test @MainActor func historyAcknowledgementWaitsForAnOrphanHelpersGlobalLease() async throws {
    let fixture = try HistoryWorkspaceFixture(); defer { try? FileManager.default.removeItem(at: fixture.root) }
    let review = try await fixture.warning(), model = fixture.model()
    await model.refreshControlHistory()
    do {
        let orphan = try DesktopControlLock(url: fixture.leaseURL)
        defer { withExtendedLifetime(orphan) {} }
        await #expect(throws: AstraError.self) { try await model.acknowledgeControlHistory(review) }
        #expect(!model.pendingControlHistory.isEmpty)
    }
    try await model.acknowledgeControlHistory(review)
    #expect(model.pendingControlHistory.isEmpty)
}

@Test @MainActor func newHistoryIsRescannedBeforeLiveControlAdmission() async throws {
    let fixture = try HistoryWorkspaceFixture(); defer { try? FileManager.default.removeItem(at: fixture.root) }
    let probe = HistoryProbe(), coordinator = fixture.inference(probe), model = fixture.model(inference: coordinator)
    let (agent, checkpoint) = try await fixture.artifacts()
    await model.refreshControlHistory()
    #expect(model.inferenceUnavailableReason == nil)
    _ = try await fixture.warning() // Appears after the model's last snapshot.
    model.startInference(agent: agent, checkpoint: checkpoint, source: fixture.source, options: .init())
    try await historyWait { !model.controlHistoryBusy }
    #expect(!probe.started && !coordinator.isBusy && !model.pendingControlHistory.isEmpty && model.errorMessage != nil)
    await coordinator.stopAndWait()
}

@Test @MainActor func quitDuringHistoryInspectionJoinsItAndNeverStartsLateControl() async throws {
    let fixture = try HistoryWorkspaceFixture(); defer { try? FileManager.default.removeItem(at: fixture.root) }
    let entered = HistoryProbe(), runtime = HistoryProbe(), gate = HistoryGate()
    let coordinator = fixture.inference(runtime)
    let model = fixture.model(inference: coordinator, inspection: { store in entered.mark(); await gate.wait(); try await store.inspectPriorInferenceRuns() })
    let (agent, checkpoint) = try await fixture.artifacts()
    model.startInference(agent: agent, checkpoint: checkpoint, source: fixture.source, options: .init())
    try await historyWait { entered.started }
    let closed = HistoryProbe()
    let closing = Task { let allowed = await model.prepareForTermination(); closed.mark(); return allowed }
    try await historyWait { model.isClosing }
    #expect(model.controlHistoryBusy && model.isRunningAgent && !closed.started && !runtime.started)
    gate.open()
    #expect(await closing.value)
    #expect(!model.controlHistoryBusy && !runtime.started && !coordinator.isBusy)
}
