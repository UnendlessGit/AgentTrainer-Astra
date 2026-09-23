import Foundation
import AstraCore
@testable import AstraPlatform
@testable import AgentTrainerAstra

/// No Core Graphics posting, capture, event tap or privacy preflight. Physical
/// input is always empty; injected effects belong only to this private model.
final class HostQualificationBackend: ControlInputBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var keys: Set<Int> = [], buttons: Set<Int> = []
    private var posts = 0
    private let surfaces: [SurfaceDescriptor]
    init(surfaces: [SurfaceDescriptor]) { self.surfaces = surfaces }
    var clean: Bool { lock.withLock { keys.isEmpty && buttons.isEmpty } }
    var posted: Int { lock.withLock { posts } }
    func prepare(_ request: ArmRequest) throws -> ControlState { physicalState() }
    func physicalState() -> ControlState {
        var state = ControlState(); state.valid = true; state.observedNanos = MonotonicClock.now
        state.pointer = .init(x: 16, y: 16); return state
    }
    func checkHealth() throws {}
    func validate(_ scope: ControlScope, pointer: Point2D?) throws {
        _ = try scope.validated()
        guard scope.surfaces == surfaces, pointer?.isFinite != false else {
            throw AstraError("qualification.scope", "Virtual input escaped its generated source.")
        }
    }
    func post(_ emission: InputEmission) throws {
        lock.withLock {
            posts += 1
            switch emission.operation {
            case .keyDown: if let key = emission.keyCode { keys.insert(key) }
            case .keyUp: if let key = emission.keyCode { keys.remove(key) }
            case .buttonDown: if let button = emission.button { buttons.insert(button) }
            case .buttonUp: if let button = emission.button { buttons.remove(button) }
            default: break
            }
        }
    }
}

/// The control transport is injected; its scheduler, admission, histories,
/// deadlines, receipts and cleanup are the actual production InputExecutor.
final class HostQualificationControl: @unchecked Sendable {
    let backend: HostQualificationBackend
    private let lock = NSLock()
    private let output = DispatchQueue(label: "astra.qualification.control-events")
    private let events: ComputeProcess.EventHandler
    private var executor: InputExecutor?
    private var requests: [UUID: UUID] = [:]
    private var sequence: UInt64 = 1
    private var recorded: [ExecutionReceipt] = []
    private var joined = false
    init(events: @escaping ComputeProcess.EventHandler, surfaces: [SurfaceDescriptor]) {
        self.events = events; backend = HostQualificationBackend(surfaces: surfaces)
        executor = InputExecutor(backend: backend, onReceipt: { [weak self] in self?.receipt($0) },
            onStop: { [weak self] run, cause, reason in
                self?.emit("control.stopped", .object(["cause": .string(cause.rawValue), "reason": .string(reason)]), run: run)
            })
    }
    var receipts: [ExecutionReceipt] { lock.withLock { recorded } }
    var hasJoined: Bool { lock.withLock { joined } }
    var runtime: NativeControlRuntime {
        .init(start: {
            .init(kind: "hello", sequence: 0, payload: .object(["role": .string("control"),
                "protocolVersion": .integer(1), "initialPacketSequenceVersion": .integer(1)]))
        }, request: { kind, payload, run, _ in try await self.request(kind, payload, run) }, shutdown: { await self.shutdown() })
    }
    private func request(_ kind: String, _ payload: JSONValue, _ run: UUID) async throws -> WireMessage {
        guard let executor = lock.withLock({ executor }) else { throw AstraError("qualification.joined", "Virtual control is already joined.") }
        let id = UUID()
        do {
            let value: JSONValue
            switch kind {
            case "arm":
                try executor.arm(payload.decode(ArmRequest.self))
                value = .object(["armed": .bool(true), "nextPacketSequence": .unsigned(executor.nextPacketSequence)])
            case "heartbeat": try executor.heartbeat(runID: run); value = .object(["alive": .bool(true)])
            case "observation": value = try .encode(executor.observation(afterSequence: payload.fields?["afterSequence"]?.uint64))
            case "execute":
                let packet = try payload.decode(ActionPacket.self)
                lock.withLock { requests[packet.id] = id }
                try executor.execute(packet); value = .object(["admitted": .bool(true)])
            case "disarm":
                executor.disarm(reason: "Virtual native owner requested release.", cause: .requested)
                try await settle(executor)
                value = .object(["stopped": .bool(true), "cleanupSettled": .bool(executor.cleanupSettled)])
            default: throw AstraError("qualification.operation", "Unsupported virtual control operation: \(kind)")
            }
            return .init(kind: "ack", sequence: 0, requestID: id, runID: run, payload: value)
        } catch {
            let value = (error as? AstraError) ?? .init("qualification.control", error.localizedDescription)
            return .init(kind: "error", sequence: 0, requestID: id, runID: run,
                payload: .object(["code": .string(value.code), "message": .string(value.message), "recoverable": .bool(true)]))
        }
    }
    private func receipt(_ value: ExecutionReceipt) {
        let id = lock.withLock { () -> UUID? in
            recorded.append(value)
            let request = requests[value.packetID]
            if value.status != .admitted { requests.removeValue(forKey: value.packetID) }
            return request
        }
        do { emit("control.receipt", .object(["receipt": try .encode(value)]), request: id, run: value.runID) }
        catch { preconditionFailure("Production receipt failed encoding: \(error)") }
    }
    private func emit(_ kind: String, _ payload: JSONValue, request: UUID? = nil, run: UUID?) {
        output.async { [self] in
            events(.init(kind: kind, sequence: sequence, requestID: request, runID: run, payload: payload)); sequence += 1
        }
    }
    private func settle(_ executor: InputExecutor) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while !executor.cleanupSettled {
            guard ContinuousClock.now < deadline else { throw AstraError("qualification.cleanup", "Virtual InputExecutor cleanup did not settle.") }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
    private func shutdown() async -> Int32? {
        guard let value = lock.withLock({ executor }) else { return 0 }
        value.disarm(reason: "Virtual runtime is shutting down.", cause: .shutdown)
        do { try await settle(value) } catch { return 1 }
        await withCheckedContinuation { continuation in output.async { continuation.resume() } }
        lock.withLock { executor = nil; joined = true }
        return backend.clean ? 0 : 1
    }
}

final class HostQualificationControls: @unchecked Sendable {
    private let lock = NSLock()
    private var children: [HostQualificationControl] = []
    private let surfaces: [SurfaceDescriptor]
    init(surfaces: [SurfaceDescriptor] = HostQualificationCapture().surfaces) { self.surfaces = surfaces }
    var all: [HostQualificationControl] { lock.withLock { children } }
    var factory: NativeControlRuntimeFactory {
        .init(protectsPhysicalInputs: false) { events, _ in
            let child = HostQualificationControl(events: events, surfaces: self.surfaces)
            self.lock.withLock { self.children.append(child) }; return child.runtime
        }
    }
}

final class HostQualificationCapture: @unchecked Sendable {
    let surfaces: [SurfaceDescriptor]
    var surface: SurfaceDescriptor { surfaces[0] }
    private let queue = DispatchQueue(label: "astra.qualification.capture")
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var frames = 0, joined = false
    private var producedMetadata: [FrameMetadata] = []
    private let buffers: [Data]
    init(multiSurface: Bool = false) {
        var sources = [SurfaceDescriptor(id: "generated-host-source", globalBounds: .init(x: 0, y: 0, width: 32, height: 32),
            pixelWidth: 32, pixelHeight: 32, geometryRevision: 7, nativeDisplayID: 7_001)]
        if multiSurface {
            sources.append(.init(id: "generated-host-secondary", globalBounds: .init(x: 32, y: 0, width: 48, height: 32),
                pixelWidth: 48, pixelHeight: 32, geometryRevision: 19, nativeDisplayID: 7_002))
        }
        surfaces = sources
        buffers = sources.enumerated().map { index, value in
            Data((0..<(value.pixelWidth * value.pixelHeight * 4)).map { UInt8(($0 * 17 + index * 31) % 256) })
        }
    }
    var hasJoined: Bool { lock.withLock { joined } }
    var produced: Int { lock.withLock { frames } }
    var metadata: [FrameMetadata] { lock.withLock { producedMetadata } }
    var source: CaptureSource {
        let leaves = surfaces.map { value in
            CaptureSource(id: value.id, name: "Generated private fixture", kind: .display, displayID: value.nativeDisplayID,
                bounds: value.globalBounds, pixelWidth: value.pixelWidth, pixelHeight: value.pixelHeight)
        }
        guard leaves.count > 1 else { return leaves[0] }
        return .init(id: "generated-host-desktop", name: "Generated two-source desktop", kind: .desktop,
            bounds: .init(x: 0, y: 0, width: 80, height: 32), pixelWidth: 32, pixelHeight: 32, bindings: leaves)
    }
    var runtime: InferenceCapture {
        .init(start: { frame, _ in
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(deadline: .now(), repeating: .milliseconds(10), leeway: .milliseconds(1))
            timer.setEventHandler { [self] in
                for (index, surface) in surfaces.enumerated() {
                    let now = MonotonicClock.now, pixels = buffers[index]
                    let metadata = FrameMetadata(id: UUID(), eventNanos: now - UInt64(index + 1) * 1_000_000,
                        observedNanos: now, surface: surface, byteCount: pixels.count, codec: "raw")
                    lock.withLock {
                        frames += 1
                        if producedMetadata.count < 20_000 { producedMetadata.append(metadata) }
                    }
                    frame(.init(metadata: metadata, pixels: { pixels }))
                }
            }
            self.lock.withLock { self.timer = timer; self.joined = false }; timer.resume()
        }, stop: {
            let timer = self.lock.withLock { let value = self.timer; self.timer = nil; return value }
            timer?.cancel()
            await withCheckedContinuation { continuation in self.queue.async { continuation.resume() } }
            self.lock.withLock { self.joined = true }
        })
    }
}

final class HostQualificationProcesses: @unchecked Sendable {
    let executable: URL
    private let lock = NSLock()
    private var workers: [(String, ComputeProcess)] = []
    init(executable: URL) { self.executable = executable }
    private func make(_ role: String, events: @escaping ComputeProcess.EventHandler, failure: @escaping ComputeProcess.FailureHandler) -> ComputeProcess {
        let process = ComputeProcess(executable: executable, arguments: ["--role", role], expectedRole: role,
            allowedEvents: role == "collector" ? ["collector.framesConsumed", "collector.applied", "collector.fault", "collector.sealed", "collector.audited"] : [],
            onEvent: events, onFailure: failure)
        lock.withLock { workers.append((role, process)) }; return process
    }
    var actor: @Sendable (String, @escaping ComputeProcess.EventHandler, @escaping ComputeProcess.FailureHandler) -> InferenceRuntime {
        { role, events, failure in
            let child = self.make(role, events: events, failure: failure)
            return .init(start: { try await child.start() }, request: { kind, payload, run, timeout, accepting in
                try await child.request(kind: kind, payload: payload, runID: run, timeout: timeout, acceptingError: accepting)
            }, shutdown: { await child.shutdown(); return await child.terminationStatus() })
        }
    }
    var collector: CollectorRuntime.Factory {
        { events, failure in
            let child = self.make("collector", events: events, failure: failure)
            return .init(start: { try await child.start() }, request: { kind, payload, run in
                try await child.request(kind: kind, payload: payload, runID: run, timeout: .seconds(30))
            }, shutdown: { await child.shutdown() })
        }
    }
    func statuses() async -> [String: [Int32]] {
        let copy = lock.withLock { workers }; var result: [String: [Int32]] = [:]
        for (role, worker) in copy { result[role, default: []].append(await worker.terminationStatus() ?? -999) }
        return result
    }
}
