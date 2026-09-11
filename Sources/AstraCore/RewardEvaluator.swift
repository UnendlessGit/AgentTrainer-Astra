import Foundation

public enum RewardTruth: String, Codable, Sendable { case yes, no, unknown }
public enum RewardOutcome: String, Codable, Sendable { case continuing, succeeded, failed, truncated, unknown }
public struct ManualRewardMarker: Sendable {
    public let sequence: UInt64
    public let episodeID: UUID
    public let ruleID: UUID
    public let eventNanos: UInt64
    public let observedNanos: UInt64
    public init(sequence: UInt64, episodeID: UUID, ruleID: UUID, eventNanos: UInt64, observedNanos: UInt64) {
        self.sequence = sequence; self.episodeID = episodeID; self.ruleID = ruleID
        self.eventNanos = eventNanos; self.observedNanos = observedNanos
    }
}
/// The input producer issues this only after draining its ordered marker stream
/// through the interval boundary. An empty array alone is not proof of no input.
public struct ManualRewardCoverage: Sendable {
    public let episodeID: UUID
    public let startNanos: UInt64
    public let endNanos: UInt64
    public let lastSequence: UInt64?
    public init(episodeID: UUID, startNanos: UInt64, endNanos: UInt64, lastSequence: UInt64?) {
        self.episodeID = episodeID; self.startNanos = startNanos; self.endNanos = endNanos; self.lastSequence = lastSequence
    }
}
public struct RewardEvaluation: Codable, Sendable, Identifiable {
    public var id: String { episodeID.uuidString + ":" + String(endNanos) }
    public let episodeID: UUID
    public let startNanos: UInt64
    public let endNanos: UInt64
    public let value: Double?
    public let components: [UUID: Double]
    public let unknownRules: [UUID]
    public let outcome: RewardOutcome
}

/// Value-owned episode state. A failed evaluation leaves it unchanged. Detectors
/// resolve observations elsewhere; this evaluator never reads pixels or OS state.
public struct RewardEvaluator: Sendable {
    public let program: RewardProgram
    private let signals: [UUID: RewardSignal]
    private var episodeID: UUID?
    private var startedAt: UInt64 = 0
    private var cutoff: UInt64 = 0
    private var previous: [UUID: SignalValue] = [:]
    private var deltaPrevious: [UUID: Double] = [:]
    private var lastMarker: UInt64?
    private var finished = false

    public init(program: RewardProgram) throws {
        self.program = try program.validated()
        signals = Dictionary(uniqueKeysWithValues: self.program.signals.map { ($0.id, $0) })
    }

    public func readiness(episodeID: UUID, cutoffNanos: UInt64, readings: [SignalReading]) throws -> RewardTruth {
        let values = try resolve(readings, episodeID: episodeID, cutoff: cutoffNanos, startedAt: cutoffNanos, allowBeforeStart: true)
        return program.ready.map { Self.test($0, values: values) } ?? .yes
    }

    public mutating func reset(episodeID: UUID, readyAtNanos: UInt64, readings: [SignalReading], controlsReleased: Bool) throws {
        guard controlsReleased else { throw AstraError("reward.reset", "Release owned controls before confirming episode readiness.") }
        let values = try resolve(readings, episodeID: episodeID, cutoff: readyAtNanos, startedAt: readyAtNanos, allowBeforeStart: true)
        guard program.ready.map({ Self.test($0, values: values) }) ?? .yes == .yes else {
            throw AstraError("reward.notReady", "The starting condition is false or could not be read.")
        }
        self.episodeID = episodeID; startedAt = readyAtNanos; cutoff = readyAtNanos
        previous = values; lastMarker = nil; finished = false
        deltaPrevious = [:]
        for rule in program.rules where rule.kind == .scoreDelta {
            if let id = rule.signalID, case .number(let number) = values[id] { deltaPrevious[rule.id] = number }
        }
    }

    public mutating func evaluate(endNanos: UInt64, readings: [SignalReading], markers: [ManualRewardMarker] = [],
                                  markerCoverage: ManualRewardCoverage? = nil) throws -> RewardEvaluation {
        guard let episodeID, !finished, endNanos > cutoff, endNanos >= startedAt, markers.count <= 4096 else {
            throw AstraError("reward.interval", "Reward evaluation requires an active episode and the next increasing cutoff.")
        }
        let values = try resolve(readings, episodeID: episodeID, cutoff: endNanos, startedAt: startedAt)
        var nextMarker = lastMarker, markerCounts: [UUID: Int] = [:]
        let manualRules = Set(program.rules.filter { $0.kind == .manualMarker }.map(\.id))
        for marker in markers {
            let expected = nextMarker.map { $0.addingReportingOverflow(1) }
            guard marker.episodeID == episodeID, manualRules.contains(marker.ruleID),
                  marker.eventNanos >= cutoff, marker.eventNanos < endNanos,
                  marker.observedNanos >= marker.eventNanos, marker.observedNanos <= endNanos,
                  expected.map({ !$0.overflow && marker.sequence == $0.partialValue }) ?? (marker.sequence == 0) else {
                throw AstraError("reward.marker", "Manual feedback is stale, duplicated, unavailable or belongs to another episode.")
            }
            nextMarker = marker.sequence; markerCounts[marker.ruleID, default: 0] += 1
        }
        if let markerCoverage {
            guard markerCoverage.episodeID == episodeID, markerCoverage.startNanos == cutoff,
                  markerCoverage.endNanos == endNanos, markerCoverage.lastSequence == nextMarker else {
                throw AstraError("reward.markerCoverage", "Manual feedback coverage does not match its complete ordered decision interval.")
            }
        }
        var components: [UUID: Double] = [:], unknown: [UUID] = []
        var nextDeltas = deltaPrevious
        for rule in program.rules {
            var reward: Double?
            switch rule.kind {
            case .manualMarker:
                if markerCoverage != nil { reward = rule.amount * Double(markerCounts[rule.id, default: 0]) }
            case .scoreDelta:
                if let id = rule.signalID, case .number(let new) = values[id] {
                    nextDeltas[rule.id] = new
                    if let old = deltaPrevious[rule.id] {
                        let delta = new - old
                        if delta.isFinite, abs(delta) <= rule.maximumDelta { reward = delta * rule.amount }
                        else { nextDeltas[rule.id] = nil }
                    }
                } else { nextDeltas[rule.id] = nil
                } // After a gap, the new value establishes a baseline but the
                  // reward for the gap's final interval is still unknown.
            case .risingEdge:
                if let predicate = rule.predicate {
                    let before = Self.test(predicate, values: previous), now = Self.test(predicate, values: values)
                    if now == .no { reward = 0 }
                    else if now == .yes && before != .unknown { reward = before == .no ? rule.amount : 0 }
                }
            case .ratePerSecond:
                // A left-endpoint rate integral. A condition at the end cannot
                // retroactively turn an entire elapsed interval on or off.
                switch rule.predicate.map({ Self.test($0, values: previous) }) ?? .yes {
                case .yes: reward = rule.amount * (Double(endNanos - cutoff) / 1_000_000_000)
                case .no: reward = 0
                case .unknown: break
                }
            }
            if let reward {
                guard reward.isFinite else { throw AstraError("reward.nonfinite", "The reward overflowed its numeric range.") }
                components[rule.id] = reward
            } else { unknown.append(rule.id) }
        }
        let success = program.success.map { Self.test($0, values: values) } ?? .no
        let failure = program.failure.map { Self.test($0, values: values) } ?? .no
        guard success != .yes || failure != .yes else { throw AstraError("reward.conflictingOutcome", "Success and failure conditions are both true. Refine the episode rules.") }
        let outcome: RewardOutcome
        if success == .yes { outcome = .succeeded }
        else if failure == .yes { outcome = .failed }
        else if endNanos - startedAt >= UInt64(program.maximumEpisodeMS) * 1_000_000 { outcome = .truncated }
        else if success == .unknown || failure == .unknown { outcome = .unknown }
        else { outcome = .continuing }
        let total = program.rules.reduce(0.0) { $0 + (components[$1.id] ?? 0) }
        guard total.isFinite else { throw AstraError("reward.nonfinite", "The combined reward overflowed its numeric range.") }
        let result = RewardEvaluation(episodeID: episodeID, startNanos: cutoff, endNanos: endNanos,
                                      value: unknown.isEmpty ? total : nil, components: components,
                                      unknownRules: unknown, outcome: outcome)
        cutoff = endNanos; previous = values; deltaPrevious = nextDeltas; lastMarker = nextMarker
        finished = [.succeeded, .failed, .truncated].contains(outcome)
        return result
    }

    private func resolve(_ readings: [SignalReading], episodeID: UUID, cutoff: UInt64, startedAt: UInt64,
                         allowBeforeStart: Bool = false) throws -> [UUID: SignalValue] {
        guard readings.count <= signals.count, Set(readings.map(\.signalID)).count == readings.count else {
            throw AstraError("reward.readings", "A signal snapshot contains duplicate or excess readings.")
        }
        var result: [UUID: SignalValue] = [:]
        for reading in readings {
            guard let signal = signals[reading.signalID], signal.kind != .elapsedSeconds, reading.episodeID == episodeID,
                  reading.confidence.isFinite, (0...1).contains(reading.confidence), reading.eventNanos <= reading.observedNanos else {
                throw AstraError("reward.reading", "A signal has an invalid identity, confidence or timestamp.")
            }
            _ = try reading.value.validated()
            switch (signal.kind, reading.value) {
            case (_, .unknown), (.manual, _), (.ocrText, .text), (.ocrNumber, .number), (.imageMatch, .number): break
            default: throw AstraError("reward.signalType", "A reward detector supplied the wrong kind of value.")
            }
            if reading.observedNanos > cutoff || (!allowBeforeStart && reading.eventNanos < startedAt) || cutoff < reading.eventNanos {
                result[signal.id] = .unknown("The signal is outside this episode's observation cutoff.")
            } else if cutoff - reading.eventNanos > UInt64(signal.maximumAgeMS) * 1_000_000 {
                result[signal.id] = .unknown("The signal is stale.")
            } else if reading.confidence < signal.minimumConfidence { result[signal.id] = .unknown("The signal confidence is too low.") }
            else { result[signal.id] = reading.value }
        }
        for signal in program.signals {
            if signal.kind == .elapsedSeconds { result[signal.id] = .number(Double(cutoff - startedAt) / 1_000_000_000) }
            else if result[signal.id] == nil { result[signal.id] = .unknown("The signal is missing.") }
        }
        return result
    }

    public static func test(_ predicate: RewardPredicate, values: [UUID: SignalValue]) -> RewardTruth {
        let results: [RewardTruth] = predicate.conditions.map { condition in
            switch (values[condition.signalID], condition.comparison) {
            case (.number(let value), .atLeast): return condition.number.map { value >= $0 ? .yes : .no } ?? .unknown
            case (.number(let value), .atMost): return condition.number.map { value <= $0 ? .yes : .no } ?? .unknown
            case (.text(let value), .equalText): return condition.text.map { value == $0 ? .yes : .no } ?? .unknown
            case (.text(let value), .containsText): return condition.text.map { value.contains($0) ? .yes : .no } ?? .unknown
            case (.flag(let value), .isTrue): return value ? .yes : .no
            case (.flag(let value), .isFalse): return value ? .no : .yes
            default: return .unknown
            }
        }
        if predicate.logic == .all {
            return results.contains(.no) ? .no : results.contains(.unknown) ? .unknown : .yes
        }
        return results.contains(.yes) ? .yes : results.contains(.unknown) ? .unknown : .no
    }
}
