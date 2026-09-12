import Foundation

public enum RewardSignalKind: String, Codable, CaseIterable, Sendable {
    case manual, ocrText, ocrNumber, imageMatch, elapsedSeconds
    public var isVisual: Bool { self == .ocrText || self == .ocrNumber || self == .imageMatch }
}

public struct RewardSignal: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var name: String
    public var kind: RewardSignalKind
    public var surfaceID: String?
    public var region: Rect2D?
    public var minimumConfidence: Double = 0.8
    public var maximumAgeMS: Int = 500
    public var language: String = "en-US"
    public var ocrRevision: Int = 3
    public var decimalSeparator: String = "."
    public var templateDigest: String?
    public init(id: UUID = UUID(), name: String, kind: RewardSignalKind, surfaceID: String? = nil, region: Rect2D? = nil) {
        self.id = id; self.name = name; self.kind = kind; self.surfaceID = surfaceID; self.region = region
    }
    public func validated() throws -> Self {
        var value = self; value.name = try DocumentNames.validated(name)
        guard minimumConfidence.isFinite, (0...1).contains(minimumConfidence), (1...60_000).contains(maximumAgeMS),
              !language.isEmpty, language.utf8.count <= 64, ocrRevision == 3, [".", ","].contains(decimalSeparator) else {
            throw AstraError("reward.signal", "A reward signal has invalid confidence, age or language settings.")
        }
        if kind.isVisual {
            guard let surfaceID, !surfaceID.isEmpty, surfaceID.utf8.count <= 256, let region, region.isValid,
                  region.x >= 0, region.y >= 0, region.x + region.width <= 1, region.y + region.height <= 1 else {
                throw AstraError("reward.region", "Choose a surface and a region fully inside its visible content.")
            }
        } else if surfaceID != nil || region != nil {
            throw AstraError("reward.region", "Only visual signals may select a surface region.")
        }
        if kind == .imageMatch {
            guard let templateDigest, templateDigest.count == 64,
                  templateDigest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw AstraError("reward.template", "Choose an image template with a verified SHA-256 identity.")
            }
        } else if templateDigest != nil { throw AstraError("reward.template", "Only image matching uses a template.") }
        return value
    }
}

public enum SignalValue: Codable, Hashable, Sendable {
    case number(Double), text(String), flag(Bool), unknown(String)
    public func validated() throws -> Self {
        switch self {
        case .number(let number):
            guard number.isFinite else { throw AstraError("reward.number", "Reward signals must be finite.") }
        case .text(let text), .unknown(let text):
            guard text.utf8.count <= 4096 else { throw AstraError("reward.text", "A reward signal exceeds the text limit.") }
        case .flag: break
        }
        return self
    }
}

public struct SignalReading: Sendable {
    public let signalID: UUID
    public let episodeID: UUID
    public let eventNanos: UInt64
    public let observedNanos: UInt64
    public let confidence: Double
    public let value: SignalValue
    public let sourceObservationID: UUID?
    public init(signalID: UUID, episodeID: UUID, eventNanos: UInt64, observedNanos: UInt64,
                confidence: Double = 1, value: SignalValue, sourceObservationID: UUID? = nil) {
        self.signalID = signalID; self.episodeID = episodeID; self.eventNanos = eventNanos
        self.observedNanos = observedNanos; self.confidence = confidence; self.value = value
        self.sourceObservationID = sourceObservationID
    }
}

public enum RewardComparison: String, Codable, CaseIterable, Sendable { case atLeast, atMost, equalText, containsText, isTrue, isFalse }
public struct RewardCondition: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var signalID: UUID
    public var comparison: RewardComparison
    public var number: Double?
    public var text: String?
    public init(id: UUID = UUID(), signalID: UUID, comparison: RewardComparison, number: Double? = nil, text: String? = nil) {
        self.id = id; self.signalID = signalID; self.comparison = comparison; self.number = number; self.text = text
    }
}
public enum RewardLogic: String, Codable, CaseIterable, Sendable { case all, any }
public struct RewardPredicate: Codable, Hashable, Sendable {
    public var logic: RewardLogic
    public var conditions: [RewardCondition]
    public init(logic: RewardLogic = .all, conditions: [RewardCondition]) { self.logic = logic; self.conditions = conditions }
    public func validated(signals: [UUID: RewardSignal]) throws -> Self {
        guard (1...32).contains(conditions.count), Set(conditions.map(\.id)).count == conditions.count else {
            throw AstraError("reward.predicate", "Use 1–32 uniquely identified conditions per rule.")
        }
        for condition in conditions {
            guard let signal = signals[condition.signalID] else { throw AstraError("reward.reference", "A reward condition refers to a missing signal.") }
            switch condition.comparison {
            case .atLeast, .atMost:
                guard let number = condition.number, number.isFinite, condition.text == nil,
                      signal.kind != .ocrText else { throw AstraError("reward.condition", "Numeric conditions need a numeric signal and finite threshold.") }
            case .equalText, .containsText:
                guard let text = condition.text, !text.isEmpty, text.utf8.count <= 1024, condition.number == nil,
                      [.ocrText, .manual].contains(signal.kind) else { throw AstraError("reward.condition", "Text conditions need a text signal and nonempty comparison.") }
            case .isTrue, .isFalse:
                guard signal.kind == .manual, condition.number == nil, condition.text == nil else {
                    throw AstraError("reward.condition", "Boolean conditions require a manual boolean signal.")
                }
            }
        }
        return self
    }
}

public enum RewardRuleKind: String, Codable, CaseIterable, Sendable { case risingEdge, ratePerSecond, scoreDelta, manualMarker }
public struct RewardRule: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var name: String
    public var kind: RewardRuleKind
    public var amount: Double
    public var predicate: RewardPredicate?
    public var signalID: UUID?
    public var maximumDelta: Double = 100_000
    public init(id: UUID = UUID(), name: String, kind: RewardRuleKind, amount: Double,
                predicate: RewardPredicate? = nil, signalID: UUID? = nil) {
        self.id = id; self.name = name; self.kind = kind; self.amount = amount
        self.predicate = predicate; self.signalID = signalID
    }
}

public struct RewardProgram: Codable, Hashable, Identifiable, Sendable {
    public var schemaVersion = 1
    public var id: UUID
    public var name: String
    public var signals: [RewardSignal]
    public var rules: [RewardRule]
    public var success: RewardPredicate?
    public var failure: RewardPredicate?
    public var ready: RewardPredicate?
    public var resetPlan: ResetPlan?
    public var maximumEpisodeMS: Int = 120_000
    public init(id: UUID = UUID(), name: String, signals: [RewardSignal] = [], rules: [RewardRule] = []) {
        self.id = id; self.name = name; self.signals = signals; self.rules = rules
    }
    public func validated() throws -> Self {
        var value = self; value.name = try DocumentNames.validated(name)
        guard schemaVersion == 1, signals.count <= 32, rules.count <= 64,
              Set(signals.map(\.id)).count == signals.count, Set(rules.map(\.id)).count == rules.count,
              (100...3_600_000).contains(maximumEpisodeMS) else {
            throw AstraError("reward.program", "The reward definition is unsupported, duplicated or exceeds its limits.")
        }
        value.signals = try signals.map { try $0.validated() }
        let lookup = Dictionary(uniqueKeysWithValues: value.signals.map { ($0.id, $0) })
        for predicate in [success, failure, ready].compactMap({ $0 }) { _ = try predicate.validated(signals: lookup) }
        if ready?.conditions.contains(where: { lookup[$0.signalID]?.kind == .elapsedSeconds }) == true {
            throw AstraError("reward.readyClock", "Elapsed episode time starts after readiness and cannot confirm the starting condition.")
        }
        if let resetPlan {
            guard ready != nil else { throw AstraError("reset.readiness", "An authored reset requires an explicit starting condition.") }
            value.resetPlan = try resetPlan.validated(signals: lookup)
        }
        for rule in rules {
            _ = try DocumentNames.validated(rule.name)
            guard rule.amount.isFinite, abs(rule.amount) <= 1_000_000, rule.maximumDelta.isFinite,
                  rule.maximumDelta > 0, rule.maximumDelta <= 1_000_000_000 else {
                throw AstraError("reward.rule", "Reward scales and change limits must be finite and within the supported range.")
            }
            switch rule.kind {
            case .risingEdge, .ratePerSecond:
                guard rule.signalID == nil else { throw AstraError("reward.rule", "Conditional rules select signals through their conditions.") }
                if let predicate = rule.predicate { _ = try predicate.validated(signals: lookup) }
                else if rule.kind == .risingEdge { throw AstraError("reward.rule", "An edge rule needs a condition.") }
            case .scoreDelta:
                guard rule.predicate == nil, let id = rule.signalID, let signal = lookup[id],
                      [.ocrNumber, .manual].contains(signal.kind) else { throw AstraError("reward.rule", "A score change needs a numeric score signal.") }
            case .manualMarker:
                guard rule.predicate == nil, rule.signalID == nil else { throw AstraError("reward.rule", "Manual feedback uses explicit markers.") }
            }
        }
        return value
    }
}
