import Foundation
import AppKit
import AstraCore

/// Binds reset orchestration to an already-owned capture stream and a fresh
/// protected control helper. This class never starts capture or requests TCC.
public final class NativeResetDriver: ResetControlDriver, @unchecked Sendable {
    public typealias ObservationProvider = @Sendable (ResetContext) async throws -> ResetObservationSnapshot
    public typealias ScopeVerifier = @Sendable (CaptureSource, ControlScope) async throws -> UInt64
    private let environmentID: UUID
    private let source: CaptureSource
    private let observations: ObservationProvider
    private let verify: ScopeVerifier
    private let priorOwnersJoined: @Sendable () async -> Bool
    private let owner: NativeControlOwner
    private let factory: NativeControlRuntimeFactory
    private let recoveryDirectory: URL
    private let onControlEvent: @Sendable (NativeControlEvent) throws -> Void
    private let lock = NSLock()
    private var context: ResetContext?
    private var generation = UUID()
    private var stopped = true, released = true
    private var session: NativeControlSession?
    private var preparation: Task<ResetBinding, any Error>?
    private var releaseWork: Task<ResetReleaseProof, Never>?
    private var releaseID: UUID?
    private var lastRelease: ResetReleaseProof?
    private var scopeWatch: Task<Void, Never>?
    private var observationWork: [UUID: Task<ResetObservationSnapshot, any Error>] = [:]
    private var scopeCheckedAt: UInt64 = 0
    private var healthError: AstraError?
    public static let scopeFreshnessNanos: UInt64 = 250_000_000

    public init(environmentID: UUID, source: CaptureSource, observations: @escaping ObservationProvider,
                priorOwnersJoined: @escaping @Sendable () async -> Bool, owner: NativeControlOwner = .shared,
                runtimeFactory: NativeControlRuntimeFactory, recoveryDirectory: URL,
                scopeVerifier: @escaping ScopeVerifier = NativeResetDriver.verifyLiveScope,
                onControlEvent: @escaping @Sendable (NativeControlEvent) throws -> Void = { _ in }) {
        self.environmentID = environmentID; self.source = source; self.observations = observations; verify = scopeVerifier
        self.priorOwnersJoined = priorOwnersJoined; self.owner = owner; factory = runtimeFactory
        self.recoveryDirectory = recoveryDirectory; self.onControlEvent = onControlEvent
    }
    deinit {
        scopeWatch?.cancel(); observationWork.values.forEach { $0.cancel() }
        let session = session
        session?.requestStop()
        if let session { Task { _ = await session.shutdown() } }
    }

    public func prepare(context: ResetContext, capabilities: ActionCapabilities) async throws -> ResetBinding {
        guard context.environmentID == environmentID else { throw AstraError("reset.environment", "The reset belongs to a different bound environment.") }
        _ = try capabilities.validated()
        let selected = try lock.withLock { () throws -> (Task<ResetBinding, any Error>, UUID) in
            guard released, preparation == nil, releaseWork == nil, observationWork.isEmpty else {
                throw AstraError("reset.previousOwner", "The previous reset preparation, observation or control cleanup has not joined.")
            }
            let id = UUID(); generation = id; self.context = context; stopped = false; released = false; healthError = nil; lastRelease = nil; scopeCheckedAt = 0
            let task = Task { try await self.prepareWork(context: context, capabilities: capabilities, generation: id) }
            preparation = task; return (task, id)
        }
        let task = selected.0
        do {
            let result = try await withTaskCancellationHandler { try await task.value } onCancel: { self.requestStop(resetID: context.resetID) }
            lock.withLock { if generation == selected.1 { preparation = nil } }; return result
        } catch { lock.withLock { if generation == selected.1 { preparation = nil } }; throw error }
    }
    private func prepareWork(context: ResetContext, capabilities: ActionCapabilities, generation id: UUID) async throws -> ResetBinding {
        guard await priorOwnersJoined(), owner.priorCleanupJoined else { throw AstraError("reset.previousOwner", "Join the prior actor and control episode before preparing reset.") }
        let before = try await verify(source, context.scope)
        try requirePreparing(id)
        try storeProof(before, generation: id)
        if !capabilities.isEmpty {
            let config = try NativeControlConfiguration(runID: context.resetID, scope: context.scope, capabilities: capabilities,
                packetCapacity: 64, recoveryDirectory: recoveryDirectory)
            let native = NativeControlSession(configuration: config, owner: owner, runtimeFactory: factory,
                onEvent: onControlEvent, onFailure: { [weak self] _, error in self?.record(error, generation: id) })
            try lock.withLock {
                guard generation == id, !stopped else { throw CancellationError() }
                session = native
            }
            _ = try await native.start()
            try requirePreparing(id)
        }
        let after = try await verify(source, context.scope)
        try requirePreparing(id); try storeProof(after, generation: id)
        startScopeWatch(context: context, generation: id)
        return .init(context: context, priorOwnersJoined: true, verifiedAtNanos: after)
    }
    public func observe(context: ResetContext) async throws -> ResetObservationSnapshot {
        let id = UUID()
        let task = try lock.withLock { () throws -> Task<ResetObservationSnapshot, any Error> in
            if let healthError { throw healthError }
            guard self.context == context, preparation == nil, releaseWork == nil, observationWork.count < 4 else { throw AstraError("reset.observationOwner", "Reset observation has a foreign owner, is still preparing/releasing, or exceeded its queue bound.") }
            let expected = generation
            let task = Task { [source, verify, observations] in
                try Task.checkCancellation()
                let before = try await verify(source, context.scope)
                try self.storeProof(before, generation: expected)
                let result = try await observations(context)
                try Task.checkCancellation()
                let after = try await verify(source, context.scope)
                try self.storeProof(after, generation: expected)
                try self.lock.withLock { if let healthError = self.healthError { throw healthError } }
                guard result.context == context, result.observedNanos <= MonotonicClock.now, result.sourceCoverage.count == context.scope.surfaces.count,
                      Set(result.sourceCoverage.map { $0.surface.id }).count == result.sourceCoverage.count else {
                    throw AstraError("reset.observationOwner", "The bound capture producer changed reset identity or source geometry.")
                }
                for source in result.sourceCoverage { _ = try source.validated(scope: context.scope, cutoffNanos: result.observedNanos) }
                return result
            }
            observationWork[id] = task; return task
        }
        defer { _ = lock.withLock { observationWork.removeValue(forKey: id) } }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
    }
    public func execute(_ packet: ActionPacket, context: ResetContext) async throws -> ExecutionReceipt {
        try checkHealth(resetID: context.resetID)
        guard packet.runID == context.resetID, let native = lock.withLock({ self.context == context && !stopped ? session : nil }) else {
            throw AstraError("reset.controlOwner", "No protected helper accepts commands for this reset.")
        }
        return try await native.execute(packet)
    }
    public func checkHealth(resetID: UUID) throws {
        let native = try lock.withLock { () throws -> NativeControlSession? in
            if let healthError { throw healthError }
            let now = MonotonicClock.now
            guard context?.resetID == resetID, !stopped, !released, now >= scopeCheckedAt,
                  now - scopeCheckedAt < Self.scopeFreshnessNanos else { throw AstraError("reset.scopeHealth", "Reset scope verification stopped or expired.") }
            return session
        }
        try native?.checkHealth()
    }
    public func requestStop(resetID: UUID) {
        let values = lock.withLock { () -> (NativeControlSession?, [Task<ResetObservationSnapshot, any Error>]) in
            guard context?.resetID == resetID else { return (nil, []) }
            stopped = true; return (session, Array(observationWork.values))
        }
        values.0?.requestStop(); values.1.forEach { $0.cancel() }
    }
    public func release(context: ResetContext) async -> ResetReleaseProof {
        let selected = lock.withLock { () -> (Task<ResetReleaseProof, Never>, UUID?) in
            guard self.context == context else {
                let confirmed = self.context == nil && owner.priorCleanupJoined
                return (Task { ResetReleaseProof(resetID: context.resetID, observedNanos: MonotonicClock.now, confirmed: confirmed,
                                                issue: confirmed ? nil : "This is a foreign reset or the previous owner is not joined.") }, nil)
            }
            if let releaseWork { return (releaseWork, releaseID) }
            if let lastRelease { return (Task { lastRelease }, nil) }
            stopped = true
            let preparing = preparation
            let id = UUID()
            let task = Task<ResetReleaseProof, Never> { [self] in
                requestStop(resetID: context.resetID)
                _ = try? await preparing?.value
                let pending = lock.withLock { Array(observationWork.values) }
                pending.forEach { $0.cancel() }
                for work in pending { _ = try? await work.value }
                let watch = lock.withLock { scopeWatch }; watch?.cancel(); await watch?.value
                let native = lock.withLock { session }
                let completion = await native?.shutdown()
                let confirmed = completion?.cleanupConfirmed ?? owner.priorCleanupJoined
                let failure = lock.withLock { healthError = healthError ?? completion?.issue; return healthError }
                let proof = ResetReleaseProof(resetID: context.resetID, observedNanos: MonotonicClock.now, confirmed: confirmed,
                                              issue: failure?.message ?? (confirmed ? nil : "The previous control owner remains unconfirmed."))
                lock.withLock { session = nil; scopeWatch = nil; preparation = nil; observationWork = [:]; released = true; lastRelease = proof }
                return proof
            }
            releaseWork = task; releaseID = id; return (task, id)
        }
        let result = await selected.0.value
        lock.withLock { if let id = selected.1, releaseID == id { releaseWork = nil; releaseID = nil } }
        return result
    }
    private func requirePreparing(_ id: UUID) throws {
        try lock.withLock { guard generation == id, !stopped, !released else { throw CancellationError() }; if let healthError { throw healthError } }
    }
    private func storeProof(_ timestamp: UInt64, generation id: UUID) throws {
        let now = MonotonicClock.now
        guard timestamp <= now, now - timestamp < Self.scopeFreshnessNanos else { throw AstraError("reset.scopeHealth", "The source verifier returned stale or future evidence.") }
        try lock.withLock {
            guard generation == id else { throw AstraError("reset.scopeGeneration", "A previous reset verifier returned after its owner changed.") }
            scopeCheckedAt = max(scopeCheckedAt, timestamp)
        }
    }
    private func record(_ error: AstraError, generation id: UUID) {
        let native = lock.withLock { () -> NativeControlSession? in
            guard generation == id, !released else { return nil }
            healthError = healthError ?? error; return session
        }
        native?.requestStop()
    }
    private func startScopeWatch(context: ResetContext, generation id: UUID) {
        let source = source, verify = verify
        let task = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .milliseconds(50))
                    let proof = try await verify(source, context.scope)
                    try self?.storeProof(proof, generation: id)
                } catch {
                    if !Task.isCancelled { self?.record((error as? AstraError) ?? .init("reset.scope", error.localizedDescription), generation: id) }
                    return
                }
            }
        }
        lock.withLock { scopeWatch = task }
    }

    public static func verifyLiveScope(_ source: CaptureSource, _ scope: ControlScope) async throws -> UInt64 {
        try await Task.detached {
            _ = try scope.validated()
            let began = MonotonicClock.now
            guard source.applicationPID == scope.applicationPID, source.windowID == scope.windowID else {
                throw AstraError("reset.scope", "The reset control scope differs from its captured source.")
            }
            if source.applicationPID != nil, source.applicationLaunchDate == nil { throw AstraError("reset.launchIdentity", "The original application launch identity is unavailable.") }
            let snapshot = try InputScopeSnapshot.current(for: source)
            try snapshot.verifyTarget(source)
            for surface in scope.surfaces {
                if surface.id.hasPrefix("window:"), let id = UInt32(surface.id.dropFirst(7)) {
                    guard let window = snapshot.windows.first(where: { $0.id == id }), window.pid == source.applicationPID,
                          window.bounds == surface.globalBounds else { throw AstraError("reset.geometry", "A reset window moved, resized or changed ownership.") }
                } else if surface.id.hasPrefix("display:"), let id = UInt32(surface.id.dropFirst(8)) {
                    guard CGDisplayIsActive(id) != 0, Rect2D(CGDisplayBounds(id)) == surface.globalBounds else {
                        throw AstraError("reset.geometry", "A reset display disconnected or changed geometry.")
                    }
                } else { throw AstraError("reset.surface", "This reset surface has no verifiable native window or display binding.") }
            }
            return began
        }.value
    }
}
