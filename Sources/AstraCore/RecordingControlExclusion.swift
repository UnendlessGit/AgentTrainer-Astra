import Foundation

/// Attestation of an uninterrupted recording-owned desktop exclusion lease.
/// It certifies absence of Astra queues, not the absence of other OS automation
/// or the validity of physical controls. An unfinished proof stays unavailable.
public struct RecordingControlExclusion: Codable, Hashable, Sendable {
    public var schemaVersion = 1
    public let ownershipID: UUID
    public let recordingID: UUID
    public let startedNanos: UInt64
    public var throughNanos: UInt64?
    public var producersJoinedNanos: UInt64?
    public init(ownershipID: UUID = UUID(), recordingID: UUID, startedNanos: UInt64) {
        self.ownershipID = ownershipID; self.recordingID = recordingID; self.startedNanos = startedNanos
    }
    @discardableResult public func validated(recordingID expected: UUID) throws -> Self {
        guard schemaVersion == 1, recordingID == expected, startedNanos <= UInt64(Int64.max),
              (throughNanos == nil) == (producersJoinedNanos == nil) else {
            throw AstraError("recording.controlExclusion", "The recording's control-exclusion provenance is invalid.")
        }
        if let throughNanos, let producersJoinedNanos {
            guard startedNanos <= throughNanos, throughNanos <= producersJoinedNanos,
                  producersJoinedNanos <= UInt64(Int64.max) else {
                throw AstraError("recording.controlExclusion", "Control-exclusion coverage must end after it starts and before its producers join.")
            }
        }
        return self
    }
    public func covers(_ cutoff: UInt64) -> Bool {
        guard let throughNanos, producersJoinedNanos != nil else { return false }
        return startedNanos <= cutoff && cutoff <= throughNanos
    }
}
