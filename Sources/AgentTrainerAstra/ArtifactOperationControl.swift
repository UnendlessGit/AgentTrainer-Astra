import Foundation
import AstraCore

/// Transfer I/O reports into one bounded slot. The UI samples it at 10 Hz,
/// rather than creating a MainActor task for every copied chunk.
final class ArtifactOperationControl: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var latest: ArtifactTransferProgress?
    var isCancelled: Bool { lock.withLock { cancelled } }
    var progress: ArtifactTransferProgress? { lock.withLock { latest } }
    func cancel() { lock.withLock { cancelled = true } }
    func report(_ progress: ArtifactTransferProgress) { lock.withLock { latest = progress } }
}
