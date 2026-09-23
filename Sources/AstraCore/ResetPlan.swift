import Foundation

public struct ResetPacketTemplate: Codable, Hashable, Sendable {
    public var durationMS: Int
    public var commands: [TimedCommand]
    public init(durationMS: Int, commands: [TimedCommand]) { self.durationMS = durationMS; self.commands = commands }
    public var capabilities: ActionCapabilities {
        .init(keyCodes: Set(commands.compactMap(\.keyCode)), mouseButtons: Set(commands.compactMap(\.button)),
              absolutePointer: commands.contains { $0.operation == .pointerAbsolute },
              relativePointer: commands.contains { $0.operation == .pointerRelative }, scroll: commands.contains { $0.operation == .scroll })
    }
    public func validated() throws -> Self {
        guard !commands.isEmpty, commands.count <= 64,
              capabilities.keyCodes.isDisjoint(with: [57, 63, 72, 73, 74, 127]) else {
            throw AstraError("reset.packet", "Reset packets need 1–64 commands; Caps Lock, Fn, media and power keys are unsupported.")
        }
        let names = Set(commands.compactMap(\.surfaceID))
        guard names.count <= 16, names.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 256 }) else {
            throw AstraError("reset.surface", "Choose a valid observed surface for each pointer command.")
        }
        let surfaces = (names.isEmpty ? ["reset.validation"] : names.sorted()).map {
            SurfaceDescriptor(id: $0, globalBounds: .init(x: 0, y: 0, width: 1, height: 1), pixelWidth: 1, pixelHeight: 1)
        }
        _ = try ActionPacket(runID: UUID(), sequence: 0, observationID: UUID(), geometryRevision: 0,
            executeAtNanos: 0, durationMs: durationMS, commands: commands).validated(capabilities: capabilities, surfaces: surfaces, capacity: 64)
        guard commands.filter({ $0.operation == .pointerRelative }).allSatisfy({
            $0.dx.map { $0.rounded() == $0 } == true && $0.dy.map { $0.rounded() == $0 } == true
        }) else { throw AstraError("reset.relativeMotion", "Relative pointer movement uses whole raw counts.") }
        return self
    }
    public func packet(context: ResetContext, observationID: UUID, sequence: UInt64, executeAtNanos: UInt64) throws -> ActionPacket {
        _ = try validated()
        return try ActionPacket(runID: context.resetID, sequence: sequence, observationID: observationID,
            geometryRevision: context.scope.geometryRevision, executeAtNanos: executeAtNanos,
            durationMs: durationMS, commands: commands).validated(capabilities: capabilities, surfaces: context.scope.surfaces, capacity: 64, expectedGeometryRevision: context.scope.geometryRevision)
    }
}

public enum ResetStepKind: String, Codable, CaseIterable, Sendable { case packet, pause, wait }
public struct ResetStep: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var name: String
    public var kind: ResetStepKind
    public var packet: ResetPacketTemplate?
    public var pauseMS: Int?
    public var condition: RewardPredicate?
    public var timeoutMS: Int?
    public init(id: UUID = UUID(), name: String, packet: ResetPacketTemplate) {
        self.id = id; self.name = name; kind = .packet; self.packet = packet
    }
    public init(id: UUID = UUID(), name: String = "Pause", pauseMS: Int) {
        self.id = id; self.name = name; kind = .pause; self.pauseMS = pauseMS
    }
    public init(id: UUID = UUID(), name: String = "Wait for a condition", condition: RewardPredicate, timeoutMS: Int = 10_000) {
        self.id = id; self.name = name; kind = .wait; self.condition = condition; self.timeoutMS = timeoutMS
    }
    public func validated(signals: [UUID: RewardSignal], maximumDurationMS: Int) throws -> Self {
        var copy = self; copy.name = try DocumentNames.validated(name)
        switch kind {
        case .packet:
            guard let packet, pauseMS == nil, condition == nil, timeoutMS == nil else { throw invalid() }
            copy.packet = try packet.validated()
        case .pause:
            guard let pauseMS, (1...min(60_000, maximumDurationMS)).contains(pauseMS),
                  packet == nil, condition == nil, timeoutMS == nil else { throw invalid() }
        case .wait:
            guard let condition, let timeoutMS, (1...min(60_000, maximumDurationMS)).contains(timeoutMS),
                  packet == nil, pauseMS == nil else { throw invalid() }
            _ = try condition.validated(signals: signals)
            guard condition.conditions.allSatisfy({ signals[$0.signalID]?.kind != .elapsedSeconds }) else {
                throw AstraError("reset.episodeClock", "Episode time starts after reset and cannot satisfy a reset condition.")
            }
        }
        return copy
    }
    private func invalid() -> AstraError { .init("reset.step", "A reset step has missing, conflicting or out-of-range settings.") }
}

/// A bounded sequence on the current environment. Absence of a plan selects
/// explicit Manual Ready; it never means automatic readiness.
public struct ResetPlan: Codable, Hashable, Sendable {
    public static let maximumEncodedBytes = 262_144
    public var schemaVersion = 1
    public var steps: [ResetStep]
    public var maximumAttempts: Int = 1
    public var maximumDurationMS: Int = 120_000
    public var readinessTimeoutMS: Int = 30_000
    public init(steps: [ResetStep] = []) { self.steps = steps }
    public var capabilities: ActionCapabilities {
        var result = ActionCapabilities()
        for step in steps {
            guard let value = step.packet?.capabilities else { continue }
            result.keyCodes.formUnion(value.keyCodes); result.mouseButtons.formUnion(value.mouseButtons)
            result.absolutePointer = result.absolutePointer || value.absolutePointer
            result.relativePointer = result.relativePointer || value.relativePointer; result.scroll = result.scroll || value.scroll
        }
        return result
    }
    public func validated(signals: [UUID: RewardSignal]) throws -> Self {
        guard schemaVersion == 1, steps.count <= 64, Set(steps.map(\.id)).count == steps.count,
              (1...3).contains(maximumAttempts), (1_000...600_000).contains(maximumDurationMS),
              (1...min(60_000, maximumDurationMS)).contains(readinessTimeoutMS) else {
            throw AstraError("reset.plan", "Reset settings exceed the supported version, step, attempt or timeout limits.")
        }
        var copy = self; copy.steps = try steps.map { try $0.validated(signals: signals, maximumDurationMS: maximumDurationMS) }
        guard try JSONEncoder().encode(copy).count <= Self.maximumEncodedBytes else {
            throw AstraError("reset.size", "The reset definition exceeds its supported size.")
        }
        return copy
    }
}

public struct ResetContext: Hashable, Sendable {
    public let resetID: UUID
    public let nextEpisodeID: UUID
    public let environmentID: UUID
    public let scope: ControlScope
    public init(resetID: UUID = UUID(), nextEpisodeID: UUID, environmentID: UUID, scope: ControlScope) throws {
        guard resetID != nextEpisodeID else { throw AstraError("reset.identity", "Reset control and policy episodes need distinct identities.") }
        self.resetID = resetID; self.nextEpisodeID = nextEpisodeID; self.environmentID = environmentID
        self.scope = try scope.validated()
    }
}
