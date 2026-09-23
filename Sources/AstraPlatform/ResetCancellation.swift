import Foundation
import AstraCore

/// A single reset's sticky admission stop. The host creates this before any
/// async dispatch or setup, so an early Stop cannot miss a not-yet-created run.
public final class ResetCancellation: @unchecked Sendable {
    public let resetID: UUID
    private let lock = NSLock()
    private var requested = false
    private var consumed = false
    private var cancellation: (@Sendable () -> Void)?
    public init(resetID: UUID = UUID()) { self.resetID = resetID }

    public func cancel() {
        let callback = lock.withLock { requested = true; return cancellation }
        callback?()
    }
    func attach(resetID: UUID, cancellation: @escaping @Sendable () -> Void) throws {
        let alreadyRequested = try lock.withLock { () throws -> Bool in
            guard self.resetID == resetID, !consumed else {
                throw AstraError("reset.cancellationIdentity", "Reset cancellation belongs to a different or already consumed attempt.")
            }
            consumed = true; self.cancellation = cancellation; return requested
        }
        if alreadyRequested { cancellation() }
    }
    func detach() { lock.withLock { cancellation = nil } }
}
