import Foundation

public struct RecordingTimeRange: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var startNanos: UInt64
    public var endNanos: UInt64
    public init(id: UUID = UUID(), startNanos: UInt64, endNanos: UInt64) {
        self.id = id; self.startNanos = startNanos; self.endNanos = endNanos
    }
    public var durationSeconds: Double { endNanos >= startNanos ? Double(endNanos - startNanos) / 1e9 : 0 }
}

/// Metadata on an agent's link, never an edit to the shared recording. Nil
/// ranges means the whole usable prefix; an empty explicit range list is invalid.
public struct RecordingTrainingSelection: Codable, Hashable, Sendable {
    public var schemaVersion = 1
    public var ranges: [RecordingTimeRange]?
    public init(ranges: [RecordingTimeRange]? = nil) { self.ranges = ranges }
    public static let whole = RecordingTrainingSelection()

    public func validated() throws -> Self {
        guard schemaVersion == 1 else { throw AstraError("selection.version", "This recording selection uses an unsupported format.") }
        if let ranges {
            guard (1...256).contains(ranges.count), Set(ranges.map(\.id)).count == ranges.count else {
                throw AstraError("selection.ranges", "Choose between 1 and 256 distinct training intervals.")
            }
            var previousEnd: UInt64 = 0
            for range in ranges {
                guard range.startNanos < range.endNanos, range.endNanos <= UInt64(Int64.max), range.startNanos >= previousEnd else {
                    throw AstraError("selection.bounds", "Training intervals must be ordered, have positive duration, and must not overlap.")
                }
                previousEnd = range.endNanos
            }
        }
        return self
    }

    public func resolved(for recording: RecordingManifest) throws -> [RecordingTimeRange] {
        _ = try validated(); _ = try recording.validated()
        guard recording.status != .recording, recording.frameCount > 0,
              let first = recording.firstObservedNanos, let stopped = recording.stoppedNanos else {
            throw AstraError("selection.source", "Finish saving a recording with complete frames before choosing its training intervals.")
        }
        let end = min(stopped, recording.firstInvalidObservedNanos ?? stopped)
        guard end > first else { throw AstraError("selection.coverage", "This recording has no verified interval available for training.") }
        let result = ranges ?? [.init(id: recording.id, startNanos: first, endNanos: end)]
        guard result.allSatisfy({ $0.startNanos >= first && $0.endNanos <= end }) else {
            throw AstraError("selection.coverage", "A saved interval extends beyond this recording's usable source. Review and save the selection again.")
        }
        return result
    }

    public func payload(recordingID: UUID) throws -> JSONValue {
        _ = try validated()
        var value: [String: JSONValue] = ["recording_id": .string(recordingID.uuidString.lowercased()), "context_ids": .array([])]
        if let ranges {
            value["ranges"] = .array(ranges.map { .object(["start_nanos": .integer(Int64($0.startNanos)), "end_nanos": .integer(Int64($0.endNanos))]) })
        }
        return .object(value)
    }
}
