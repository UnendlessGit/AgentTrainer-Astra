import Foundation
import AstraCore

/// All access is serialized by InputExecutor's lock. Lifetime reservations
/// keep terminal recording independent of transport and JSON allocation.
final class ControlFeedbackLedger {
    let epochID = UUID()
    let runID: UUID
    let geometryRevision: UInt64
    private var rows: [UUID: ControlFeedbackPacket] = [:]
    private var reservations: [UUID: Int] = [:]
    private var changes: [ControlFeedbackChange] = []
    private var bytes = ControlFeedbackLimits.envelopeReservation
    private var nextSequence: UInt64 = 0
    private var acknowledged: UInt64?
    private var delivered: UInt64?
    private(set) var interrupted = false
    init(request: ArmRequest) { runID = request.runID; geometryRevision = request.scope.geometryRevision; changes.reserveCapacity(128) }

    func validate(_ cursor: ControlFeedbackCursor?) throws {
        guard let cursor, cursor.version == 1, cursor.controlEpochID == epochID else {
            throw AstraError("control.feedbackCursor", "The queued-control cursor does not belong to this physical control epoch.")
        }
        if let after = cursor.afterSequence {
            guard let delivered, after <= delivered, acknowledged.map({ after >= $0 }) ?? true else {
                throw AstraError("control.feedbackCursor", "The queued-control acknowledgement is unavailable or precedes released history.")
            }
        } else if acknowledged != nil { throw AstraError("control.feedbackCursor", "Queued-control history requires its previous acknowledgement.") }
    }
    func acknowledge(_ sequence: UInt64?) {
        guard let sequence else { return }
        acknowledged = sequence
        changes.removeAll { $0.sequence <= sequence }
        let removed = rows.values.filter { $0.terminal.map { $0.sequence <= sequence } ?? false }.map { $0.packet.id }
        for id in removed { rows.removeValue(forKey: id); bytes -= reservations.removeValue(forKey: id)! }
    }
    func reserveAdmission(_ packet: ActionPacket, bytes reservation: Int) throws {
        let terminals = rows.values.filter { $0.terminal == nil }.count
        guard rows[packet.id] == nil, rows.count < ControlFeedbackLimits.maximumPackets,
              reservation > 0, reservation <= ControlFeedbackLimits.maximumBytes - bytes,
              changes.count + terminals + 2 <= ControlFeedbackLimits.maximumChanges,
              UInt64.max - nextSequence >= UInt64(terminals + 2) else {
            throw AstraError("control.feedbackCapacity", "Queued-control history requires acknowledgement before another packet can be admitted.")
        }
    }
    func admit(_ packet: ActionPacket, reservation: Int, at now: UInt64) {
        rows[packet.id] = ControlFeedbackPacket(packet: packet, admissionSequence: nextSequence, admittedNanos: now)
        reservations[packet.id] = reservation; bytes += reservation
        changes.append(.init(sequence: nextSequence, packetID: packet.id, kind: .admitted, availableNanos: now)); nextSequence += 1
    }
    func completeSample(packetID: UUID, commandIndex: Int, command: TimedCommand, endpoint: Bool,
                        status: CommandStatus, postedNanos: UInt64?, availableNanos: UInt64) {
        guard var row = rows[packetID], row.terminal == nil else { return }
        var value = row.progress[commandIndex]
        if status == .failed { value.status = .failed }
        else {
            value.completedSampleCount += 1; value.lastCompletedOffsetMs = command.offsetMs; value.lastCompletedAvailableNanos = availableNanos
            if let postedNanos {
                value.lastPostedNanos = postedNanos
                if command.operation == .pointerRelative { value.emittedDx! += Int(command.dx!); value.emittedDy! += Int(command.dy!) }
            }
            value.status = endpoint ? (status == .noOp ? .noOp : .posted) : .partial
        }
        row.progress[commandIndex] = value; rows[packetID] = row
    }
    func finish(_ packetID: UUID, status: ReceiptStatus, at now: UInt64) {
        guard var row = rows[packetID], row.terminal == nil else { return }
        if status != .executed { interrupted = true }
        for index in row.progress.indices where row.progress[index].status == .pending || row.progress[index].status == .partial {
            row.progress[index].status = .cancelled
        }
        row.terminal = .init(sequence: nextSequence, status: status, availableNanos: now); rows[packetID] = row
        changes.append(.init(sequence: nextSequence, packetID: packetID, kind: .terminal, availableNanos: now)); nextSequence += 1
    }
    func snapshot(cutoff: UInt64, unavailable: ControlFeedbackUnavailable?) -> ControlFeedbackSnapshot {
        let through = nextSequence == 0 ? nil : nextSequence - 1
        if unavailable == nil { delivered = through }
        return .init(controlEpochID: epochID, runID: runID, geometryRevision: geometryRevision, cutoffNanos: cutoff,
                     unavailableReason: unavailable, acknowledgedThrough: acknowledged, throughSequence: through,
                     changes: changes, packets: rows.values.sorted { $0.packet.sequence < $1.packet.sequence })
    }
}
