import Foundation

public struct ActionCapabilities: Codable, Hashable, Sendable {
    public var keyCodes: Set<Int>
    public var mouseButtons: Set<Int>
    public var absolutePointer: Bool
    public var relativePointer: Bool
    public var scroll: Bool

    public init(keyCodes: Set<Int> = [], mouseButtons: Set<Int> = [],
                absolutePointer: Bool = false, relativePointer: Bool = false, scroll: Bool = false) {
        self.keyCodes = keyCodes; self.mouseButtons = mouseButtons
        self.absolutePointer = absolutePointer; self.relativePointer = relativePointer; self.scroll = scroll
    }
    public var isEmpty: Bool {
        keyCodes.isEmpty && mouseButtons.isEmpty && !absolutePointer && !relativePointer && !scroll
    }
    public func validated() throws -> Self {
        guard keyCodes.allSatisfy({ (0...127).contains($0) }),
              mouseButtons.allSatisfy({ (0...31).contains($0) }) else {
            throw AstraError("actions.capabilities", "Unsupported keyboard or mouse-button capability.")
        }
        return self
    }
    public func intersection(_ other: Self) -> Self {
        Self(keyCodes: keyCodes.intersection(other.keyCodes), mouseButtons: mouseButtons.intersection(other.mouseButtons),
             absolutePointer: absolutePointer && other.absolutePointer,
             relativePointer: relativePointer && other.relativePointer, scroll: scroll && other.scroll)
    }

    private enum CodingKeys: String, CodingKey {
        case keyCodes, mouseButtons, absolutePointer, relativePointer, scroll
    }
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let keys = try container.decode([Int].self, forKey: .keyCodes)
        let buttons = try container.decode([Int].self, forKey: .mouseButtons)
        guard Set(keys).count == keys.count, Set(buttons).count == buttons.count else {
            throw AstraError("actions.duplicateCapability", "Action capabilities must not contain duplicates.")
        }
        keyCodes = Set(keys); mouseButtons = Set(buttons)
        absolutePointer = try container.decode(Bool.self, forKey: .absolutePointer)
        relativePointer = try container.decode(Bool.self, forKey: .relativePointer)
        scroll = try container.decode(Bool.self, forKey: .scroll)
        _ = try validated()
    }
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(keyCodes.sorted(), forKey: .keyCodes)
        try container.encode(mouseButtons.sorted(), forKey: .mouseButtons)
        try container.encode(absolutePointer, forKey: .absolutePointer)
        try container.encode(relativePointer, forKey: .relativePointer)
        try container.encode(scroll, forKey: .scroll)
    }
}

public enum CommandOperation: String, Codable, CaseIterable, Sendable {
    case keyDown, keyUp, keyRepeat, buttonDown, buttonUp, pointerAbsolute, pointerRelative, scroll
    public var isMotion: Bool { self == .pointerAbsolute || self == .pointerRelative }
}

public struct TimedCommand: Codable, Hashable, Sendable {
    public var offsetMs: Int
    public var operation: CommandOperation
    public var keyCode: Int?
    public var button: Int?
    public var surfaceID: String?
    public var x: Double?
    public var y: Double?
    public var dx: Double?
    public var dy: Double?

    public init(offsetMs: Int, operation: CommandOperation, keyCode: Int? = nil, button: Int? = nil,
                surfaceID: String? = nil, x: Double? = nil, y: Double? = nil, dx: Double? = nil, dy: Double? = nil) {
        self.offsetMs = offsetMs; self.operation = operation; self.keyCode = keyCode
        self.button = button; self.surfaceID = surfaceID; self.x = x; self.y = y; self.dx = dx; self.dy = dy
    }

    public func validated(capabilities: ActionCapabilities, surfaces: [SurfaceDescriptor], durationMs: Int) throws {
        guard offsetMs >= 0, offsetMs < durationMs || (operation.isMotion && offsetMs == durationMs),
              [x, y, dx, dy].compactMap({ $0 }).allSatisfy(\.isFinite) else {
            throw AstraError("actions.arguments", "Invalid command time or nonfinite arguments.")
        }
        let hasPoint = x != nil || y != nil
        let hasDelta = dx != nil || dy != nil
        switch operation {
        case .keyDown, .keyUp, .keyRepeat:
            guard let keyCode, capabilities.keyCodes.contains(keyCode), button == nil,
                  surfaceID == nil, !hasPoint, !hasDelta else { throw invalidCapability() }
        case .buttonDown, .buttonUp:
            guard let button, capabilities.mouseButtons.contains(button), keyCode == nil,
                  surfaceID == nil, !hasPoint, !hasDelta else { throw invalidCapability() }
        case .pointerAbsolute:
            guard capabilities.absolutePointer, keyCode == nil, button == nil, !hasDelta,
                  let surfaceID, surfaces.contains(where: { $0.id == surfaceID }),
                  let x, let y, (0...1).contains(x), (0...1).contains(y) else { throw invalidCapability() }
        case .pointerRelative, .scroll:
            let enabled = operation == .scroll ? capabilities.scroll : capabilities.relativePointer
            guard enabled, keyCode == nil, button == nil, surfaceID == nil, !hasPoint,
                  let dx, let dy, (-32_768...32_767).contains(dx), (-32_768...32_767).contains(dy) else {
                throw invalidCapability()
            }
        }
    }

    private func invalidCapability() -> AstraError {
        AstraError("actions.notAllowed", "The command is outside the run's action capabilities or contains ambiguous arguments.")
    }
}

public struct ActionPacket: Codable, Hashable, Sendable, Identifiable {
    public var id: UUID
    public var runID: UUID
    public var sequence: UInt64
    public var observationID: UUID
    public var geometryRevision: UInt64
    public var executeAtNanos: UInt64
    public var durationMs: Int
    public var commands: [TimedCommand]

    public init(id: UUID = UUID(), runID: UUID, sequence: UInt64, observationID: UUID,
                geometryRevision: UInt64, executeAtNanos: UInt64, durationMs: Int, commands: [TimedCommand]) {
        self.id = id; self.runID = runID; self.sequence = sequence; self.observationID = observationID
        self.geometryRevision = geometryRevision; self.executeAtNanos = executeAtNanos
        self.durationMs = durationMs; self.commands = commands
    }

    public func validated(capabilities: ActionCapabilities, surfaces: [SurfaceDescriptor], capacity: Int) throws -> Self {
        _ = try capabilities.validated()
        guard [16, 32, 64].contains(capacity), commands.count <= capacity,
              (1...1_000).contains(durationMs), !surfaces.isEmpty,
              Set(surfaces.map(\.id)).count == surfaces.count else {
            throw AstraError("actions.packet", "Invalid command packet size, cadence, or surfaces.")
        }
        for surface in surfaces {
            _ = try surface.validated()
            guard surface.geometryRevision == geometryRevision else {
                throw AstraError("actions.geometryChanged", "The surface moved after this action was planned.")
            }
        }
        let duration = UInt64(durationMs) * 1_000_000
        guard !executeAtNanos.addingReportingOverflow(duration).overflow else {
            throw AstraError("actions.timeOverflow", "The command deadline is outside the monotonic clock range.")
        }
        var previousOffset = 0
        var motion: CommandOperation?
        for command in commands {
            guard command.offsetMs >= previousOffset else {
                throw AstraError("actions.order", "Command times must preserve packet order.")
            }
            try command.validated(capabilities: capabilities, surfaces: surfaces, durationMs: durationMs)
            if command.operation.isMotion {
                guard motion == nil || motion == command.operation else {
                    throw AstraError("actions.mixedMotion", "Absolute and relative trajectories cannot share a packet.")
                }
                motion = command.operation
            }
            previousOffset = command.offsetMs
        }
        return self
    }
}

public struct ControlState: Codable, Hashable, Sendable {
    public var keys: Set<Int> = []
    public var buttons: Set<Int> = []
    public var modifiers: UInt64 = 0
    public var pointer: Point2D = .zero
    public var observedNanos: UInt64 = 0
    public var revision: UInt64 = 0
    public var valid = false
    public init() {}

    private enum CodingKeys: String, CodingKey { case keys, buttons, modifiers, pointer, observedNanos, revision, valid }
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(keys.sorted(), forKey: .keys)
        try container.encode(buttons.sorted(), forKey: .buttons)
        try container.encode(modifiers, forKey: .modifiers)
        try container.encode(pointer, forKey: .pointer)
        try container.encode(observedNanos, forKey: .observedNanos)
        try container.encode(revision, forKey: .revision)
        try container.encode(valid, forKey: .valid)
    }
}

public enum InputOrigin: String, Codable, Sendable { case physical, agent, reconciliation, boundary }
public enum RawInputKind: String, Codable, Sendable {
    case keyDown, keyUp, keyRepeat, buttonDown, buttonUp, pointer, scroll, flags, gap
}

public struct RawInputEvent: Codable, Hashable, Sendable {
    public var sequence: UInt64
    public var eventNanos: UInt64
    public var observedNanos: UInt64
    public var origin: InputOrigin
    public var kind: RawInputKind
    public var keyCode: Int?
    public var button: Int?
    public var x: Double?
    public var y: Double?
    public var dx: Double?
    public var dy: Double?
    public var scrollX: Double?
    public var scrollY: Double?
    public var modifiers: UInt64?
    public var isDown: Bool?
    public var detail: String?

    public init(sequence: UInt64, eventNanos: UInt64, observedNanos: UInt64, origin: InputOrigin,
                kind: RawInputKind, keyCode: Int? = nil, button: Int? = nil,
                x: Double? = nil, y: Double? = nil, dx: Double? = nil, dy: Double? = nil,
                scrollX: Double? = nil, scrollY: Double? = nil, modifiers: UInt64? = nil,
                isDown: Bool? = nil, detail: String? = nil) {
        self.sequence = sequence; self.eventNanos = eventNanos; self.observedNanos = observedNanos
        self.origin = origin; self.kind = kind; self.keyCode = keyCode; self.button = button
        self.x = x; self.y = y; self.dx = dx; self.dy = dy; self.scrollX = scrollX; self.scrollY = scrollY
        self.modifiers = modifiers; self.isDown = isDown; self.detail = detail
    }
}
