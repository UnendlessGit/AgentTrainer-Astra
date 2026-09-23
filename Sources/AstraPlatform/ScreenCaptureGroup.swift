import Foundation
import AstraCore

/// Fixed source membership with one shared delivery queue. Every timestamp,
/// frame ID, coverage proof and per-source revision comes from its own stream.
public final class ScreenCaptureGroup: @unchecked Sendable {
    public typealias HealthHandler = @Sendable (String, CaptureHealth) -> Void
    private let lock = NSLock()
    private let outputQueue = DispatchQueue(label: "astra.capture.group", qos: .userInitiated)
    private var generation: UUID?
    private var captures: [ScreenCapture] = []
    private var surfaces: [String: SurfaceDescriptor] = [:]
    private var startup: AsyncCompletion?
    private var stopping: Task<Void, Never>?
    private var topologyWatch: Task<Void, Never>?

    public init() {}

    deinit {
        topologyWatch?.cancel()
        let children = captures
        Task {
            await withTaskGroup(of: Void.self) { group in
                for child in children { group.addTask { await child.stop() } }
            }
        }
    }

    public func start(source: CaptureSource, fps: Int, showsCursor: Bool,
                      onFrame: @escaping ScreenCapture.FrameHandler, onHealth: @escaping HealthHandler) async throws {
        let bindings = try source.captureBindings(), token = UUID(), completion = AsyncCompletion()
        let children = bindings.map { _ in ScreenCapture(outputQueue: outputQueue) }
        try lock.withLock {
            guard generation == nil, stopping == nil else { throw AstraError("capture.busy", "Capture is already active or stopping.") }
            generation = token; captures = children; startup = completion; surfaces = [:]
        }
        defer { completion.finish() }
        do {
            for (binding, child) in zip(bindings, children) {
                try Task.checkCancellation()
                guard lock.withLock({ generation == token }) else { throw CancellationError() }
                try await child.start(source: binding, fps: fps, showsCursor: showsCursor, onFrame: { [weak self] frame in
                    guard let self else { return }
                    let accepted = lock.withLock { () -> Bool? in
                        guard generation == token else { return nil }
                        guard frame.surface.id == binding.id, frame.surface.globalBounds == binding.bounds,
                              surfaces[binding.id].map({ $0 == frame.surface }) ?? true else { return false }
                        surfaces[binding.id] = frame.surface; return true
                    }
                    if accepted == true { onFrame(frame) }
                    else if accepted == false { onHealth(binding.id, .unavailable("A bound surface changed geometry. End this session and choose the source again.")) }
                }, onHealth: { [weak self] health in
                    guard self?.lock.withLock({ self?.generation == token }) == true else { return }
                    onHealth(binding.id, health)
                })
            }
            guard lock.withLock({ generation == token }) else { throw CancellationError() }
            let watch = Task.detached { [weak self] in
                while !Task.isCancelled {
                    do {
                        try await Task.sleep(for: .milliseconds(250))
                        let snapshot = try InputScopeSnapshot.current(for: source)
                        try snapshot.verifyBindings(source)
                    } catch is CancellationError { return }
                    catch {
                        guard self?.lock.withLock({ self?.generation == token }) == true else { return }
                        onHealth(source.id, .unavailable(error.localizedDescription)); return
                    }
                }
            }
            lock.withLock {
                if generation == token { topologyWatch = watch } else { watch.cancel() }
            }
        } catch {
            // Never call stop here: stop joins this startup barrier.
            await withTaskGroup(of: Void.self) { group in
                for child in children { group.addTask { await child.stop() } }
            }
            lock.withLock { if generation == token { generation = nil; captures = []; surfaces = [:] } }
            throw error
        }
    }

    public func stop() async {
        let task = lock.withLock { () -> Task<Void, Never> in
            if let stopping { return stopping }
            generation = nil
            let children = captures, completion = startup, watch = topologyWatch
            topologyWatch = nil; watch?.cancel()
            let task = Task { [self] in
                await completion?.wait(); await watch?.value
                await withTaskGroup(of: Void.self) { group in
                    for child in children { group.addTask { await child.stop() } }
                }
                lock.withLock { captures = []; surfaces = [:]; startup = nil; stopping = nil }
            }
            stopping = task; return task
        }
        await task.value
    }
}
