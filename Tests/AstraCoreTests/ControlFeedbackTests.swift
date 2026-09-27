import Foundation
import Testing
@testable import AstraCore

private func feedbackWorstWidthPacket(capacity: Int) -> ControlFeedbackPacket {
    let packet = ActionPacket(runID: UUID(), sequence: UInt64.max, observationID: UUID(), geometryRevision: UInt64.max,
        executeAtNanos: UInt64.max - 1_000_000_001, durationMs: 1_000,
        commands: (0..<capacity).map { .init(offsetMs: $0 * 15, operation: .pointerRelative, dx: 32_767, dy: -32_768) })
    var row = ControlFeedbackPacket(packet: packet, admissionSequence: 0, admittedNanos: packet.executeAtNanos - 1)
    for index in row.progress.indices {
        row.progress[index].status = .posted
        row.progress[index].completedSampleCount = index == 0 ? 1 : 15
        row.progress[index].lastCompletedOffsetMs = packet.commands[index].offsetMs
        row.progress[index].lastCompletedAvailableNanos = UInt64.max - 1
        row.progress[index].lastPostedNanos = UInt64.max - 2
        row.progress[index].emittedDx = 32_767; row.progress[index].emittedDy = -32_768
    }
    row.terminal = .init(sequence: 1, status: .executed, availableNanos: UInt64.max)
    return row
}
private func feedbackSnapshot(_ row: ControlFeedbackPacket) -> ControlFeedbackSnapshot {
    .init(controlEpochID: UUID(), runID: row.packet.runID, geometryRevision: row.packet.geometryRevision, cutoffNanos: UInt64.max,
          unavailableReason: nil, acknowledgedThrough: nil, throughSequence: 1,
          changes: [.init(sequence: 0, packetID: row.packet.id, kind: .admitted, availableNanos: row.admittedNanos),
                    .init(sequence: 1, packetID: row.packet.id, kind: .terminal, availableNanos: UInt64.max)], packets: [row])
}

@Test(arguments: [16, 32, 64]) func feedbackReservationCoversWorstWidthProgressAndTerminal(capacity: Int) throws {
    let row = feedbackWorstWidthPacket(capacity: capacity), snapshot = feedbackSnapshot(feedbackWorstWidthPacket(capacity: capacity))
    let rowBytes = try JSONEncoder().encode(row).count
    let reservation = try ControlFeedbackLimits.reservation(for: row.packet)
    #expect(rowBytes + 512 < reservation)
    #expect(try JSONEncoder().encode(snapshot).count < reservation + ControlFeedbackLimits.envelopeReservation)
    let cursor = ControlFeedbackCursor(controlEpochID: snapshot.controlEpochID)
    try snapshot.validate(cursor: cursor, runID: snapshot.runID, geometryRevision: snapshot.geometryRevision, cutoffNanos: snapshot.cutoffNanos)
    #expect(try JSONDecoder().decode(ControlFeedbackSnapshot.self, from: JSONEncoder().encode(snapshot)) == snapshot)
}

@Test func feedbackEvidenceRejectsCursorGapsIncompleteProgressAndOverflowingTimes() throws {
    let original = feedbackSnapshot(feedbackWorstWidthPacket(capacity: 16))
    let cursor = ControlFeedbackCursor(controlEpochID: original.controlEpochID)
    for fault in 0..<7 {
        var value = original
        switch fault {
        case 0: value.changes.removeFirst()
        case 1: value.packets[0].progress[0].lastCompletedAvailableNanos = nil
        case 2: value.packets[0].progress[0].lastCompletedOffsetMs = Int.max
        case 3: value.packets[0].packet.executeAtNanos = UInt64.max
        case 4: value.packets[0].progress[0].status = .pending
        case 5: value.packets.append(value.packets[0])
        default: value.coverageNanos = UInt64.max - 1
        }
        #expect(throws: AstraError.self) { try value.validate(cursor: cursor, runID: original.runID, geometryRevision: original.geometryRevision, cutoffNanos: original.cutoffNanos) }
    }
}

@Test func feedbackEvidenceRejectsBorrowedAdmissionAndNonmonotonicAvailability() throws {
    var original = feedbackSnapshot(feedbackWorstWidthPacket(capacity: 16))
    original.packets[0].packet.sequence = 0
    let cursor = ControlFeedbackCursor(controlEpochID: original.controlEpochID)
    for sequence in [UInt64(0), 1] {
        var value = original, extra = original.packets[0]
        extra.packet.id = UUID(); extra.packet.sequence = 1; extra.admissionSequence = sequence; extra.terminal = nil
        value.packets.append(extra)
        #expect(throws: AstraError.self) { try value.validate(cursor: cursor, runID: original.runID, geometryRevision: original.geometryRevision, cutoffNanos: original.cutoffNanos) }
    }
    var value = original
    value.changes[1].availableNanos = original.changes[0].availableNanos - 1
    value.packets[0].terminal?.availableNanos = value.changes[1].availableNanos
    #expect(throws: AstraError.self) { try value.validate(cursor: cursor, runID: original.runID, geometryRevision: original.geometryRevision, cutoffNanos: original.cutoffNanos) }
}
