import Foundation
import AstraCore

public struct RecordingProgress: Sendable {
    public let id: UUID
    public let frames: Int
    public let events: Int
    public let bytes: UInt64
    public let elapsedSeconds: Double
    public let pendingFrames: Int
}

/// Bounded capture -> parallel lossless compression -> ordered publication.
/// The UI receives snapshots and never owns capture buffers or disk I/O.
public final class RecordingSession: @unchecked Sendable {
    private let lock = NSLock()
    private let capture = ScreenCaptureGroup()
    private let input = PhysicalInputMonitor()
    private let writerQueue = DispatchQueue(label: "astra.recording.writer", qos: .userInitiated)
    private let compressors: OperationQueue = {
        let queue = OperationQueue(); queue.name = "astra.recording.compression"
        queue.maxConcurrentOperationCount = 4; queue.qualityOfService = .userInitiated
        return queue
    }()
    private let writer: RecordingWriter
    private let correction: CorrectionRecordingSeed?
    private let source: CaptureSource
    private let surfaceIDs: Set<String>
    private let fps: Int
    private let showsCursor: Bool
    private let onProgress: @Sendable (RecordingProgress) -> Void
    private let onFault: @Sendable (String) -> Void
    private let startingCompletion = AsyncCompletion()
    private var timer: DispatchSourceTimer?
    private var started = false
    private var acceptingFrames = false
    private var acceptingEvents = false
    private var stopAt: UInt64?
    private var firstFailure: String?
    private var finishTask: Task<RecordingManifest, any Error>?
    private var admittedFrames = 0
    private var reservedBytes = 0
    private var admittedEvents = 0
    private var admittedCoverage = 0
    private var receivedSurfaceIDs: Set<String> = []
    private var frameSequence: UInt64 = 0
    private var captureRequestedAt: UInt64?
    // The following fields are owned only by writerQueue.
    private var nextWrittenSequence: UInt64 = 0
    private var completedFrames: [UInt64: (Result<PreparedFrame, any Error>, Int)] = [:]
    private var healthWasAvailable: [String: Bool] = [:]
    private var finalized = false

    public init(directory: URL, manifest: RecordingManifest, source: CaptureSource, correction: CorrectionRecordingSeed? = nil,
                onProgress: @escaping @Sendable (RecordingProgress) -> Void,
                onFault: @escaping @Sendable (String) -> Void) throws {
        self.correction = correction
        self.source = source; fps = manifest.environment.captureFPS; showsCursor = manifest.environment.showsCursor
        let bindings = try source.captureBindings()
        surfaceIDs = Set(bindings.map(\.id))
        self.onProgress = onProgress; self.onFault = onFault
        var boundManifest = manifest
        boundManifest.surfaceIDs = bindings.map(\.id)
        writer = try RecordingWriter(directory: directory, manifest: boundManifest)
    }

    deinit {
        timer?.cancel()
        compressors.cancelAllOperations()
        // Normal ownership always awaits stop(). If a coordinator abandons a
        // session, still release the native producers and leave the append-only
        // package recoverable instead of keeping a hidden capture alive.
        Task { [capture, input] in
            async let captureStop: Void = capture.stop()
            async let inputStop: Void = input.stop()
            _ = await (captureStop, inputStop)
        }
    }

    public func start() async throws {
        try lock.withLock {
            guard !started, finishTask == nil else { throw AstraError("recording.busy", "This recording session has already started or stopped.") }
            started = true; acceptingFrames = true; acceptingEvents = true
        }
        defer { startingCompletion.finish() }
        do {
            try checkDisk()
            if let correction {
                let gate = try lock.withLock {
                    guard acceptingFrames, finishTask == nil else { throw CancellationError() }
                    return MonotonicClock.now
                }
                try await Task.detached { [writer] in try writer.attachCorrection(correction, supervisionStartNanos: gate) }.value
                try requireStarting()
            }
            try await input.start(source: source, onEvents: { [weak self] in self?.admitEvents($0) },
                                  onFault: { [weak self] in self?.fail($0.localizedDescription) },
                                  onEmergency: { [weak self] in self?.fail("Recording stopped by the emergency shortcut.") },
                                  onBoundary: { [weak self] in self?.inputBoundary($0) })
            try requireStarting()
            lock.withLock { captureRequestedAt = MonotonicClock.now }
            try await capture.start(source: source, fps: fps, showsCursor: showsCursor,
                                    onFrame: { [weak self] in self?.admitFrame($0) },
                                    onHealth: { [weak self] in self?.captureHealth($1, surfaceID: $0) })
            try requireStarting()
            let timer = DispatchSource.makeTimerSource(queue: writerQueue)
            timer.schedule(deadline: .now() + .milliseconds(250), repeating: .milliseconds(250), leeway: .milliseconds(25))
            timer.setEventHandler { [weak self] in self?.flushAndPublish() }
            self.timer = timer; timer.resume()
        } catch {
            async let captureStop: Void = capture.stop()
            async let inputStop: Void = input.stop()
            _ = await (captureStop, inputStop)
            fail(error.localizedDescription)
            throw error
        }
    }

    public func stop(issue: String? = nil) async throws -> RecordingManifest {
        let task = lock.withLock { () -> Task<RecordingManifest, any Error> in
            if let finishTask { return finishTask }
            acceptingFrames = false
            stopAt = stopAt ?? MonotonicClock.now
            if let issue, firstFailure == nil { firstFailure = issue }
            if !started { startingCompletion.finish() }
            let task = Task { [self] in
                // Joining startup prevents a producer from starting after this
                // stop's first attempt to close it.
                await startingCompletion.wait()
                async let captureStop: Void = capture.stop()
                async let inputStop: Void = input.stop()
                _ = await (captureStop, inputStop)
                lock.withLock { acceptingEvents = false }
                await withCheckedContinuation { continuation in
                    DispatchQueue.global(qos: .userInitiated).async { [compressors] in
                        compressors.waitUntilAllOperationsAreFinished()
                        continuation.resume()
                    }
                }
                return try await withCheckedThrowingContinuation { continuation in
                    writerQueue.async { [self] in
                        timer?.cancel(); timer = nil
                        let (end, failure) = lock.withLock { (stopAt ?? MonotonicClock.now, firstFailure) }
                        do {
                            let manifest = try writer.finish(at: end, status: failure == nil ? .complete : .interrupted, issue: failure)
                            finalized = true
                            continuation.resume(returning: manifest)
                        } catch { continuation.resume(throwing: error) }
                    }
                }
            }
            finishTask = task
            return task
        }
        return try await task.value
    }

    private func requireStarting() throws {
        try Task.checkCancellation()
        try lock.withLock {
            guard acceptingFrames, finishTask == nil else { throw CancellationError() }
            if let firstFailure { throw AstraError("recording.start", firstFailure) }
        }
    }

    private func admitFrame(_ frame: CapturedFrame) {
        let cost = frame.surface.pixelWidth * frame.surface.pixelHeight * 4 * 3
        let sequence: UInt64? = lock.withLock {
            guard acceptingFrames else { return nil }
            guard surfaceIDs.contains(frame.surface.id), admittedFrames < max(8, surfaceIDs.count * 2), reservedBytes + cost <= 512 * 1_024 * 1_024 else { return UInt64.max }
            receivedSurfaceIDs.insert(frame.surface.id)
            admittedFrames += 1; reservedBytes += cost
            let value = frameSequence; frameSequence += 1
            return value
        }
        guard let sequence else { return }
        guard sequence != UInt64.max else {
            fail("Lossless recording could not keep up at this resolution and frame rate. The captured prefix is preserved.")
            return
        }
        do {
            // Copy only compact pixels, then let SCK release the original
            // IOSurface before asynchronous compression begins.
            let pixels = try frame.copyCompactPixels()
            let metadata = FrameMetadata(id: frame.id, eventNanos: frame.eventNanos, observedNanos: frame.observedNanos,
                                         surface: frame.surface, byteCount: pixels.count)
            compressors.addOperation { [weak self] in
                let prepared = Result { try FrameArchive.prepare(pixels: pixels, metadata: metadata) }
                self?.prepared(prepared, sequence: sequence, cost: cost, observedNanos: metadata.observedNanos)
            }
        } catch { prepared(.failure(error), sequence: sequence, cost: cost, observedNanos: frame.observedNanos) }
    }

    private func prepared(_ result: Result<PreparedFrame, any Error>, sequence: UInt64, cost: Int, observedNanos: UInt64) {
        writerQueue.async { [self] in
            if case .failure(let error) = result {
                writer.markInvalid(from: observedNanos, message: error.localizedDescription)
            }
            completedFrames[sequence] = (result, cost)
            while let (next, reservation) = completedFrames.removeValue(forKey: nextWrittenSequence) {
                do {
                    let frame = try next.get()
                    do { try writer.append(frame) }
                    catch { writer.markInvalid(from: frame.metadata.observedNanos, message: error.localizedDescription); throw error }
                } catch { fail(error.localizedDescription) }
                lock.withLock { admittedFrames -= 1; reservedBytes -= reservation }
                nextWrittenSequence += 1
            }
        }
    }

    private func admitEvents(_ events: [RawInputEvent]) {
        let accepted: [RawInputEvent]? = lock.withLock {
            guard acceptingEvents else { return nil }
            let included = stopAt.map { cutoff in events.filter { $0.observedNanos <= cutoff } } ?? events
            guard admittedEvents + included.count <= 4_096 else { return nil }
            admittedEvents += included.count
            return included
        }
        guard let accepted else {
            if lock.withLock({ acceptingEvents }) { fail("Input storage could not keep up. The recorded prefix is preserved.") }
            return
        }
        writerQueue.async { [self] in
            defer { lock.withLock { admittedEvents -= accepted.count } }
            guard !finalized else { return }
            do { try writer.append(events: accepted) }
            catch { fail(error.localizedDescription) }
        }
    }

    private func inputBoundary(_ boundary: InputObservationBoundary) {
        writerQueue.async { [self] in
            guard !finalized else { return }
            writer.markInvalid(from: boundary.invalidFromNanos, message: boundary.message)
            do { try writer.health(observedNanos: boundary.observedNanos, status: "inputBoundary", message: boundary.message) }
            catch { fail(error.localizedDescription) }
        }
    }

    private func captureHealth(_ health: CaptureHealth, surfaceID: String) {
        if case .coverage(let proof) = health {
            let admission = lock.withLock { () -> Bool? in
                guard acceptingFrames, stopAt.map({ proof.verifiedAtNanos <= $0 }) ?? true else { return nil }
                guard admittedCoverage < 512 else { return false }
                admittedCoverage += 1; return true
            }
            guard let admission else { return }
            guard admission else { fail("Capture coverage storage could not keep up. The recorded prefix is preserved."); return }
            writerQueue.async { [self] in
                defer { lock.withLock { admittedCoverage -= 1 } }
                guard !finalized else { return }
                do { try writer.append(coverage: proof) } catch { fail(error.localizedDescription) }
            }
            return
        }
        let available: Bool
        switch health {
        case .live, .idle: available = true
        case .starting, .stopped, .coverage: return
        case .unavailable(let message): fail(message); available = false
        }
        let time = MonotonicClock.now
        writerQueue.async { [self] in
            guard !finalized, healthWasAvailable[surfaceID] != available else { return }
            healthWasAvailable[surfaceID] = available
            if !available { writer.markInvalid(from: time, message: "Capture became unavailable.") }
            do { try writer.health(observedNanos: time, status: available ? "available" : "unavailable", message: surfaceID) }
            catch { fail(error.localizedDescription) }
        }
    }

    private func flushAndPublish() {
        guard !finalized else { return }
        let missingFirstFrame = lock.withLock { () -> Bool in
            guard let requested = captureRequestedAt, stopAt == nil else { return false }
            return MonotonicClock.now - requested >= 10_000_000_000 && receivedSurfaceIDs != surfaceIDs
        }
        if missingFirstFrame {
            fail("A bound capture surface supplied no complete frame within 10 seconds. Check that every target is visible and Screen Recording access is still enabled.")
        }
        do { try checkDisk(); try writer.flush() }
        catch { fail(error.localizedDescription) }
        let manifest = writer.snapshot
        let now = lock.withLock { stopAt ?? MonotonicClock.now }
        let seconds = manifest.firstObservedNanos.map { Double(now >= $0 ? now - $0 : 0) / 1_000_000_000 } ?? 0
        onProgress(.init(id: manifest.id, frames: manifest.frameCount, events: manifest.eventCount,
                         bytes: manifest.storedBytes, elapsedSeconds: seconds,
                         pendingFrames: lock.withLock { admittedFrames }))
    }

    private func checkDisk() throws {
        let values = try writer.directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        guard let available = values.volumeAvailableCapacityForImportantUsage, available >= 10 * 1_024 * 1_024 * 1_024 else {
            throw AstraError("recording.diskSpace", "Recording needs at least 10 GB of free space to preserve and finalize its data.")
        }
    }

    private func fail(_ message: String) {
        let first = lock.withLock {
            guard firstFailure == nil else { return false }
            firstFailure = message; acceptingFrames = false; stopAt = stopAt ?? MonotonicClock.now
            return true
        }
        if first {
            let detected = lock.withLock { stopAt ?? MonotonicClock.now }
            writerQueue.async { [self] in
                guard !finalized else { return }
                writer.markInvalid(from: detected, message: message)
            }
            // Resource failure must close producers even if the UI is busy.
            // The caller receives the same finishTask result when publishing.
            Task { [weak self] in _ = try? await self?.stop(issue: message) }
            onFault(message)
        }
    }
}
