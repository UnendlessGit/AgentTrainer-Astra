import Foundation

/// Opt-in evidence of admitted plans. This is separate from actual held state.
public enum ControlFeedbackLimits {
    public static let version = 1
    public static let maximumPackets = 64
    public static let maximumChanges = 128
    public static let maximumBytes = 262_144
    public static let envelopeReservation = 4_096
    public static func reservation(for packet: ActionPacket) throws -> Int {
        guard packet.commands.count <= 64 else { throw AstraError("control.feedbackCapacity", "The feedback packet exceeds its command limit.") }
        let bytes = try JSONEncoder().encode(packet).count
        guard bytes <= maximumBytes else { throw AstraError("control.feedbackCapacity", "The original packet exceeds the feedback byte limit.") }
        return bytes + 2_048 + 512 * packet.commands.count
    }
}

public struct ControlFeedbackCursor: Codable, Equatable, Sendable {
    public var version: Int
    public var controlEpochID: UUID
    public var afterSequence: UInt64?
    public init(controlEpochID: UUID, afterSequence: UInt64? = nil, version: Int = 1) {
        self.version = version; self.controlEpochID = controlEpochID; self.afterSequence = afterSequence
    }
}

public enum ControlFeedbackUnavailable: String, Codable, Sendable {
    case postInFlight, arming, stopping, untrustedState, historyGap, overflow
}
public enum ControlFeedbackChangeKind: String, Codable, Sendable { case admitted, terminal }
public struct ControlFeedbackChange: Codable, Equatable, Sendable {
    public var sequence: UInt64
    public var packetID: UUID
    public var kind: ControlFeedbackChangeKind
    public var availableNanos: UInt64
    public init(sequence: UInt64, packetID: UUID, kind: ControlFeedbackChangeKind, availableNanos: UInt64) {
        self.sequence = sequence; self.packetID = packetID; self.kind = kind; self.availableNanos = availableNanos
    }
}
public enum ControlFeedbackProgressStatus: String, Codable, Sendable { case pending, partial, posted, noOp, cancelled, failed }
public struct ControlFeedbackProgress: Codable, Equatable, Sendable {
    public var commandIndex: Int
    public var status: ControlFeedbackProgressStatus = .pending
    public var completedSampleCount = 0
    public var lastCompletedOffsetMs: Int?
    public var lastCompletedAvailableNanos: UInt64?
    public var lastPostedNanos: UInt64?
    public var emittedDx: Int?
    public var emittedDy: Int?
    public init(commandIndex: Int, relative: Bool) {
        self.commandIndex = commandIndex; emittedDx = relative ? 0 : nil; emittedDy = relative ? 0 : nil
    }
}
public struct ControlFeedbackTerminal: Codable, Equatable, Sendable {
    public var sequence: UInt64
    public var status: ReceiptStatus
    public var availableNanos: UInt64
    public init(sequence: UInt64, status: ReceiptStatus, availableNanos: UInt64) {
        self.sequence = sequence; self.status = status; self.availableNanos = availableNanos
    }
}
public struct ControlFeedbackPacket: Codable, Equatable, Sendable {
    public var packet: ActionPacket
    public var admissionSequence: UInt64
    public var admittedNanos: UInt64
    public var progress: [ControlFeedbackProgress]
    public var terminal: ControlFeedbackTerminal?
    public init(packet: ActionPacket, admissionSequence: UInt64, admittedNanos: UInt64) {
        self.packet = packet; self.admissionSequence = admissionSequence; self.admittedNanos = admittedNanos
        progress = packet.commands.enumerated().map { ControlFeedbackProgress(commandIndex: $0.offset, relative: $0.element.operation == .pointerRelative) }
    }
}
public struct ControlFeedbackSnapshot: Codable, Equatable, Sendable {
    public var version = ControlFeedbackLimits.version
    public var controlEpochID: UUID
    public var runID: UUID
    public var geometryRevision: UInt64
    public var cutoffNanos: UInt64
    public var coverageNanos: UInt64?
    public var unavailableReason: ControlFeedbackUnavailable?
    public var acknowledgedThrough: UInt64?
    public var throughSequence: UInt64?
    public var changes: [ControlFeedbackChange]
    public var packets: [ControlFeedbackPacket]
    public init(controlEpochID: UUID, runID: UUID, geometryRevision: UInt64, cutoffNanos: UInt64,
                unavailableReason: ControlFeedbackUnavailable?, acknowledgedThrough: UInt64?, throughSequence: UInt64?,
                changes: [ControlFeedbackChange], packets: [ControlFeedbackPacket]) {
        self.controlEpochID = controlEpochID; self.runID = runID; self.geometryRevision = geometryRevision; self.cutoffNanos = cutoffNanos
        self.unavailableReason = unavailableReason; coverageNanos = unavailableReason == nil ? cutoffNanos : nil
        self.acknowledgedThrough = acknowledgedThrough; self.throughSequence = throughSequence
        self.changes = changes; self.packets = packets
    }

    /// Transport validation; availability remains explicit and must also agree
    /// with the enclosing actual-control coverage before use as actor input.
    public func validate(cursor: ControlFeedbackCursor, runID: UUID, geometryRevision: UInt64, cutoffNanos: UInt64) throws {
        func require(_ valid: Bool) throws { if !valid { throw AstraError("control.feedbackEvidence", "Queued control evidence is incomplete or belongs to a different observation.") } }
        try require(version == 1 && cursor.version == 1 && controlEpochID == cursor.controlEpochID && self.runID == runID
            && self.geometryRevision == geometryRevision && self.cutoffNanos == cutoffNanos && acknowledgedThrough == cursor.afterSequence)
        try require(unavailableReason == nil ? coverageNanos == cutoffNanos : coverageNanos == nil)
        try require(packets.count <= ControlFeedbackLimits.maximumPackets && changes.count <= ControlFeedbackLimits.maximumChanges)
        try require(Set(packets.map { $0.packet.id }).count == packets.count)
        try require(Set(packets.map(\.admissionSequence)).count == packets.count)
        try require(zip(packets, packets.dropFirst()).allSatisfy {
            $0.packet.sequence < $1.packet.sequence && $0.admissionSequence < $1.admissionSequence && $0.admittedNanos <= $1.admittedNanos
        })
        let byID = Dictionary(uniqueKeysWithValues: packets.map { ($0.packet.id, $0) })
        var preceding = acknowledgedThrough
        var precedingTime: UInt64 = 0
        for change in changes {
            let expected = preceding.map { $0.addingReportingOverflow(1) } ?? (partialValue: 0, overflow: false)
            try require(!expected.overflow && change.sequence == expected.partialValue && change.availableNanos <= cutoffNanos && change.availableNanos >= precedingTime)
            guard let row = byID[change.packetID] else { try require(false); return }
            try require(change.kind == .admitted
                ? row.admissionSequence == change.sequence && row.admittedNanos == change.availableNanos
                : row.terminal?.sequence == change.sequence && row.terminal?.availableNanos == change.availableNanos)
            preceding = change.sequence
            precedingTime = change.availableNanos
        }
        try require(preceding == throughSequence)
        for row in packets {
            let packet = row.packet
            try require(packet.runID == runID && packet.geometryRevision == geometryRevision && packet.commands.count <= 64
                && row.admittedNanos <= cutoffNanos
                && row.progress.count == packet.commands.count && throughSequence.map { row.admissionSequence <= $0 } == true
                && (1...1_000).contains(packet.durationMs))
            if acknowledgedThrough.map({ row.admissionSequence > $0 }) ?? true {
                try require(changes.contains { $0.kind == .admitted && $0.packetID == packet.id && $0.sequence == row.admissionSequence })
            }
            if let terminal = row.terminal {
                try require([ReceiptStatus.executed, .cancelled, .late].contains(terminal.status) && terminal.sequence > row.admissionSequence
                    && terminal.availableNanos >= row.admittedNanos && terminal.availableNanos <= cutoffNanos
                    && changes.contains { $0.sequence == terminal.sequence && $0.packetID == packet.id && $0.kind == .terminal })
                if terminal.status == .executed {
                    let end = packet.executeAtNanos.addingReportingOverflow(UInt64(packet.durationMs) * 1_000_000)
                    try require(!end.overflow && terminal.availableNanos >= end.partialValue
                        && row.progress.allSatisfy { $0.status == .posted || $0.status == .noOp })
                } else { try require(unavailableReason != nil) }
            }
            var previousMotion: TimedCommand?
            var previousOffset = 0
            for (index, progress) in row.progress.enumerated() {
                let command = packet.commands[index]
                try require(command.offsetMs >= previousOffset && command.offsetMs <= packet.durationMs
                    && (command.operation.isMotion || command.offsetMs < packet.durationMs))
                previousOffset = command.offsetMs
                if command.operation == .pointerRelative {
                    try require([command.dx, command.dy].allSatisfy { $0.map { $0.isFinite && (-32_768...32_767).contains($0) && $0.rounded(.toNearestOrEven) == $0 } == true })
                }
                if command.operation.isMotion, let previousMotion { try require(previousMotion.operation == command.operation) }
                let interpolated = command.operation.isMotion && previousMotion.map {
                    command.offsetMs > $0.offsetMs && (command.operation == .pointerRelative || command.surfaceID == $0.surfaceID)
                } == true
                let sampleLimit = interpolated ? command.offsetMs - previousMotion!.offsetMs : 1
                try require(progress.commandIndex == index && (0...1_001).contains(progress.completedSampleCount))
                try require(progress.completedSampleCount <= sampleLimit)
                try require((progress.completedSampleCount > 0) == (progress.lastCompletedOffsetMs != nil)
                    && (progress.completedSampleCount > 0) == (progress.lastCompletedAvailableNanos != nil))
                if let offset = progress.lastCompletedOffsetMs, let available = progress.lastCompletedAvailableNanos {
                    try require((0...1_000).contains(offset))
                    let scheduled = packet.executeAtNanos.addingReportingOverflow(UInt64(offset) * 1_000_000)
                    try require(offset <= command.offsetMs && !scheduled.overflow
                        && available >= scheduled.partialValue && available <= cutoffNanos && available >= row.admittedNanos)
                    try require(offset == (interpolated ? previousMotion!.offsetMs + progress.completedSampleCount : command.offsetMs))
                    try require(progress.lastPostedNanos.map { $0 >= scheduled.partialValue && $0 <= available } ?? true)
                    if let terminal = row.terminal { try require(available <= terminal.availableNanos) }
                }
                switch progress.status {
                case .pending: try require(progress.completedSampleCount == 0 && progress.lastPostedNanos == nil)
                case .partial: try require(command.operation.isMotion && progress.completedSampleCount > 0 && progress.lastCompletedOffsetMs! < command.offsetMs && progress.lastPostedNanos != nil)
                case .posted: try require(progress.completedSampleCount > 0 && progress.lastCompletedOffsetMs == command.offsetMs && progress.lastPostedNanos != nil)
                case .noOp: try require(!command.operation.isMotion && progress.completedSampleCount == 1 && progress.lastCompletedOffsetMs == command.offsetMs && progress.lastPostedNanos == nil)
                case .cancelled, .failed: break
                }
                try require(command.operation == .pointerRelative
                    ? progress.emittedDx.map { (-32_768...32_767).contains($0) } == true && progress.emittedDy.map { (-32_768...32_767).contains($0) } == true
                    : progress.emittedDx == nil && progress.emittedDy == nil)
                if command.operation == .pointerRelative {
                    let fraction = Double(progress.completedSampleCount) / Double(sampleLimit)
                    try require(progress.emittedDx == Int((command.dx! * fraction).rounded(.toNearestOrEven))
                        && progress.emittedDy == Int((command.dy! * fraction).rounded(.toNearestOrEven)))
                }
                if progress.completedSampleCount == 0 { try require(progress.lastPostedNanos == nil) }
                if command.operation.isMotion { previousMotion = command }
                if unavailableReason == nil { try require(![ControlFeedbackProgressStatus.failed, .cancelled].contains(progress.status)) }
            }
        }
        try require(try JSONEncoder().encode(self).count <= ControlFeedbackLimits.maximumBytes)
    }
}
