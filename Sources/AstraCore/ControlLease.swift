import Foundation

public struct ControlScope: Codable, Hashable, Sendable {
    public var surfaces: [SurfaceDescriptor]
    public var applicationPID: Int32?
    public var windowID: UInt32?
    public var wholeDesktop: Bool
    public var stopOnPhysicalInput: Bool
    public var geometryRevision: UInt64

    public init(surfaces: [SurfaceDescriptor], applicationPID: Int32? = nil, windowID: UInt32? = nil,
                wholeDesktop: Bool = false, stopOnPhysicalInput: Bool = true, geometryRevision: UInt64 = 0) {
        self.surfaces = surfaces; self.applicationPID = applicationPID; self.windowID = windowID
        self.wholeDesktop = wholeDesktop; self.stopOnPhysicalInput = stopOnPhysicalInput
        self.geometryRevision = geometryRevision
    }

    public func validated() throws -> Self {
        guard !surfaces.isEmpty, surfaces.count <= 16,
              Set(surfaces.map(\.id)).count == surfaces.count,
              applicationPID.map({ $0 > 0 }) ?? true,
              windowID == nil || applicationPID != nil else {
            throw AstraError("control.scope", "The control scope has invalid surfaces or target identity.")
        }
        for surface in surfaces {
            _ = try surface.validated()
            guard surface.geometryRevision == geometryRevision else {
                throw AstraError("control.geometry", "The control scope contains inconsistent surface revisions.")
            }
        }
        return self
    }
}

public struct ArmRequest: Codable, Hashable, Sendable {
    public var runID: UUID
    public var scope: ControlScope
    public var capabilities: ActionCapabilities
    public var packetCapacity: Int
    public init(runID: UUID, scope: ControlScope, capabilities: ActionCapabilities, packetCapacity: Int = 16) {
        self.runID = runID; self.scope = scope; self.capabilities = capabilities; self.packetCapacity = packetCapacity
    }
}

/// Pure admission state. The native scheduler serializes this with input posting
/// and must release its physical ledger whenever disarm/expiry occurs.
public struct ControlLease: Sendable {
    public static let durationNanos: UInt64 = 500_000_000
    public static let maximumLookaheadNanos: UInt64 = 2_000_000_000
    public private(set) var request: ArmRequest?
    public private(set) var expiresAtNanos: UInt64 = 0
    public private(set) var nextSequence: UInt64 = 0
    public init() {}

    public mutating func arm(_ request: ArmRequest, now: UInt64) throws {
        guard self.request == nil else {
            throw AstraError("control.busy", "Another session already owns desktop control.")
        }
        _ = try request.scope.validated()
        _ = try request.capabilities.validated()
        guard !request.capabilities.isEmpty, [16, 32, 64].contains(request.packetCapacity),
              !now.addingReportingOverflow(Self.durationNanos).overflow else {
            throw AstraError("control.configuration", "Select valid action capabilities before starting control.")
        }
        self.request = request
        expiresAtNanos = now + Self.durationNanos
        nextSequence = 0
    }

    public mutating func heartbeat(runID: UUID, now: UInt64) throws {
        try requireLive(runID: runID, now: now)
        guard !now.addingReportingOverflow(Self.durationNanos).overflow else {
            throw AstraError("control.clock", "The control clock is outside its supported range.")
        }
        expiresAtNanos = now + Self.durationNanos
    }

    public func requireLive(runID: UUID, now: UInt64) throws {
        guard request?.runID == runID else { throw AstraError("control.session", "The action belongs to an inactive session.") }
        guard now < expiresAtNanos else { throw AstraError("control.expired", "Desktop control stopped because its owner stopped responding.") }
    }

    public mutating func admit(_ packet: ActionPacket, now: UInt64) throws {
        try requireLive(runID: packet.runID, now: now)
        guard let request else { throw AstraError("control.disarmed", "Desktop control is not armed.") }
        guard packet.sequence == nextSequence, nextSequence < UInt64.max else {
            throw AstraError("control.sequence", "A duplicate, missing, or stale action packet was rejected.")
        }
        guard packet.executeAtNanos >= now,
              packet.executeAtNanos - now <= Self.maximumLookaheadNanos else {
            throw AstraError("control.deadline", "An action missed its deadline or was scheduled too far ahead.")
        }
        guard packet.geometryRevision == request.scope.geometryRevision else {
            throw AstraError("control.geometry", "The action uses a previous target geometry.")
        }
        _ = try packet.validated(capabilities: request.capabilities, surfaces: request.scope.surfaces,
                                 capacity: request.packetCapacity)
        nextSequence += 1
    }

    public mutating func disarm() {
        request = nil; expiresAtNanos = 0; nextSequence = 0
    }
}

public enum ReceiptStatus: String, Codable, Sendable { case admitted, executed, cancelled, rejected, late }
public enum CommandStatus: String, Codable, Sendable { case posted, noOp, cancelled, failed }
public struct CommandResult: Codable, Hashable, Sendable {
    public var commandIndex: Int
    public var scheduledNanos: UInt64
    public var postedNanos: UInt64?
    public var status: CommandStatus
    public var message: String?
    public init(commandIndex: Int, scheduledNanos: UInt64, postedNanos: UInt64? = nil,
                status: CommandStatus, message: String? = nil) {
        self.commandIndex = commandIndex; self.scheduledNanos = scheduledNanos
        self.postedNanos = postedNanos; self.status = status; self.message = message
    }
}
public struct ExecutionReceipt: Codable, Hashable, Sendable {
    public var packetID: UUID
    public var runID: UUID
    public var sequence: UInt64
    public var status: ReceiptStatus
    public var observedNanos: UInt64
    public var commandResults: [CommandResult]
    public var resultingState: ControlState
    public init(packet: ActionPacket, status: ReceiptStatus, observedNanos: UInt64,
                commandResults: [CommandResult] = [], resultingState: ControlState) {
        packetID = packet.id; runID = packet.runID; sequence = packet.sequence
        self.status = status; self.observedNanos = observedNanos
        self.commandResults = commandResults; self.resultingState = resultingState
    }
}
