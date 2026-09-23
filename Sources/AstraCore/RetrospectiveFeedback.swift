import Foundation
import CryptoKit

public enum FeedbackLimits {
    public static let maximumIntervals = 65_536
    public static let maximumPairs = 262_144
    public static let maximumAnnotations = 65_536
    public static let maximumReviews = 4_096
    public static let maximumCount = 4_096
    public static let maximumBytes = 128 * 1024 * 1024
}

public struct FeedbackIntervalIdentity: Codable, Hashable, Sendable {
    public let episodeID: UUID, observationID: UUID, packetID: UUID
    public let startNanos: UInt64, endNanos: UInt64
    public init(episodeID: UUID, observationID: UUID, packetID: UUID, startNanos: UInt64, endNanos: UInt64) {
        self.episodeID = episodeID; self.observationID = observationID; self.packetID = packetID
        self.startNanos = startNanos; self.endNanos = endNanos
    }
}

/// An existing source value, never authored or inferred by the reviewer.
public struct FeedbackBootstrap: Codable, Equatable, Sendable {
    public let episodeID: UUID, policyID: UUID, observationID: UUID
    public let cutoffNanos: UInt64
    public let value: Double
    public let recurrentReset: Bool
    public init(episodeID: UUID, policyID: UUID, observationID: UUID, cutoffNanos: UInt64, value: Double, recurrentReset: Bool = false) {
        self.episodeID = episodeID; self.policyID = policyID; self.observationID = observationID
        self.cutoffNanos = cutoffNanos; self.value = value; self.recurrentReset = recurrentReset
    }
}

public struct FeedbackEligibleInterval: Codable, Equatable, Sendable {
    public let target: FeedbackIntervalIdentity
    public let packetSequence: UInt64, drawIndex: UInt64
    public let endpointObservationID: UUID
    public let execution: String
    public let outcome: String
    public let eligible: Bool
    public let bootstrap: FeedbackBootstrap?
    public init(target: FeedbackIntervalIdentity, packetSequence: UInt64, drawIndex: UInt64, endpointObservationID: UUID,
                execution: String, outcome: String, eligible: Bool, bootstrap: FeedbackBootstrap? = nil) {
        self.target = target; self.packetSequence = packetSequence; self.drawIndex = drawIndex
        self.endpointObservationID = endpointObservationID; self.execution = execution; self.outcome = outcome
        self.eligible = eligible; self.bootstrap = bootstrap
    }
}

/// The only reward-definition fields interpreted by the annotation reader.
/// Native producers derive these from the full validated, digest-bound program.
public struct FeedbackManualRule: Codable, Equatable, Sendable {
    public let id: UUID
    public let kind: String
    public let amount: Double
    public init(id: UUID, amount: Double) { self.id = id; kind = "manualMarker"; self.amount = amount }
}

/// The collector's frozen, authenticated eligibility projection. This module
/// checks its contract; it cannot derive physical execution from a digest alone.
public struct FeedbackTrajectoryMetadata: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let collectionID: UUID, runID: UUID, clockID: UUID, sourceSessionID: UUID, policyID: UUID, programID: UUID
    public let trajectorySHA256: String, policySignature: String, programSHA256: String
    public let status: String
    public let controlClosureKnown: Bool
    public let closedNanos: UInt64, closedWallTimeMS: UInt64
    public let intervals: [FeedbackEligibleInterval]
    public let manualRules: [FeedbackManualRule]
    public init(collectionID: UUID, runID: UUID, clockID: UUID, sourceSessionID: UUID, policyID: UUID, program: RewardProgram,
                trajectorySHA256: String, policySignature: String, programSHA256: String, status: String,
                controlClosureKnown: Bool, closedNanos: UInt64, closedWallTimeMS: UInt64, intervals: [FeedbackEligibleInterval]) throws {
        let definition = try program.validated()
        manualRules = definition.rules.filter { $0.kind == .manualMarker }.sorted { $0.id.uuidString < $1.id.uuidString }
            .map { FeedbackManualRule(id: $0.id, amount: $0.amount) }
        schemaVersion = 1; self.collectionID = collectionID; self.runID = runID; self.clockID = clockID
        self.sourceSessionID = sourceSessionID; self.policyID = policyID; self.programID = definition.id
        self.trajectorySHA256 = trajectorySHA256; self.policySignature = policySignature; self.programSHA256 = programSHA256
        self.status = status; self.controlClosureKnown = controlClosureKnown; self.closedNanos = closedNanos
        self.closedWallTimeMS = closedWallTimeMS; self.intervals = intervals
    }
}

/// On a shared monotonic clock, authoring follows the joined source boundary.
/// Across clocks, require a distinct session and a later wall-time audit instead
/// of comparing unrelated monotonic timestamps.
public struct FeedbackAuthorship: Codable, Equatable, Sendable {
    public let sessionID: UUID, clockID: UUID
    public let observedNanos: UInt64, wallTimeMS: UInt64
    public init(sessionID: UUID, clockID: UUID, observedNanos: UInt64, wallTimeMS: UInt64) {
        self.sessionID = sessionID; self.clockID = clockID; self.observedNanos = observedNanos; self.wallTimeMS = wallTimeMS
    }
}

public struct FeedbackAnnotation: Codable, Equatable, Sendable {
    public let id: UUID
    public let sequence: Int
    public let target: FeedbackIntervalIdentity
    public let ruleID: UUID
    public let count: Int
    public let authored: FeedbackAuthorship
    public init(id: UUID = UUID(), sequence: Int, target: FeedbackIntervalIdentity, ruleID: UUID, count: Int, authored: FeedbackAuthorship) {
        self.id = id; self.sequence = sequence; self.target = target; self.ruleID = ruleID; self.count = count; self.authored = authored
    }
}

public struct FeedbackReviewPair: Codable, Hashable, Sendable {
    public let packetID: UUID, ruleID: UUID
    public init(packetID: UUID, ruleID: UUID) { self.packetID = packetID; self.ruleID = ruleID }
}

/// The operator explicitly finished reviewing these pairs. A missing pair is
/// unknown even when there were no clicks; zero is never inferred from silence.
public struct FeedbackReviewCompletion: Codable, Equatable, Sendable {
    public let id: UUID
    public let sequence: Int
    public let pairs: [FeedbackReviewPair]
    public let authored: FeedbackAuthorship
    public init(id: UUID = UUID(), sequence: Int, pairs: [FeedbackReviewPair], authored: FeedbackAuthorship) {
        self.id = id; self.sequence = sequence; self.pairs = pairs; self.authored = authored
    }
}

public struct FeedbackRevisionReference: Codable, Equatable, Sendable {
    public let id: UUID
    public let sha256: String
    public init(id: UUID, sha256: String) { self.id = id; self.sha256 = sha256 }
}

public struct FeedbackRuleResolution: Codable, Equatable, Sendable {
    public let ruleID: UUID
    public let reviewed: Bool
    public let count: Int?
    public let value: Double?
}
public struct FeedbackIntervalResolution: Codable, Equatable, Sendable {
    public let packetID: UUID
    public let components: [FeedbackRuleResolution]
    /// Sum of manual-marker components only. Other reward rules are untouched.
    public let manualReward: Double?
}

public struct FeedbackRewardRevision: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let id: UUID
    public let revision: UInt64
    public let parent: FeedbackRevisionReference?
    public let sourceSHA256: String
    public let authored: FeedbackAuthorship
    public let annotations: [FeedbackAnnotation]
    public let reviews: [FeedbackReviewCompletion]
    public let resolved: [FeedbackIntervalResolution]

    public static func create(id: UUID = UUID(), source: VerifiedFeedbackSource, authored: FeedbackAuthorship,
                              annotations: [FeedbackAnnotation], reviews: [FeedbackReviewCompletion],
                              parent: LoadedFeedbackRevision? = nil) throws -> Self {
        if let parent { guard parent.document.revision < UInt64.max else { throw FeedbackChecks.error("revision", "The revision counter is exhausted.") } }
        let result = Self(schemaVersion: 1, id: id, revision: parent.map { $0.document.revision + 1 } ?? 0,
            parent: parent?.reference, sourceSHA256: source.sha256, authored: authored,
            annotations: annotations, reviews: reviews, resolved: try source.resolve(authored: authored, annotations: annotations, reviews: reviews))
        try result.validate(source: source, parent: parent)
        return result
    }

    func validate(source: VerifiedFeedbackSource, parent: LoadedFeedbackRevision?) throws {
        guard schemaVersion == 1, sourceSHA256 == source.sha256, self.parent == parent?.reference,
              (parent?.document.sourceSHA256 ?? source.sha256) == source.sha256 else {
            throw FeedbackChecks.error("revision", "The reward revision differs from its immutable source or verified parent.")
        }
        if let parent {
            guard id != parent.reference.id, parent.document.revision < UInt64.max, revision == parent.document.revision + 1 else {
                throw FeedbackChecks.error("revision", "A correction requires a new ID and the next verified revision.")
            }
            try FeedbackChecks.follows(authored, parent.document.authored)
            let oldLabels = Dictionary(uniqueKeysWithValues: parent.document.annotations.map { ($0.id, $0) })
            let oldReviews = Dictionary(uniqueKeysWithValues: parent.document.reviews.map { ($0.id, $0) })
            for label in annotations {
                if oldLabels[label.id] == nil { try FeedbackChecks.follows(label.authored, parent.document.authored) }
                guard oldReviews[label.id] == nil, oldLabels[label.id].map({ $0 == label }) ?? true else {
                    throw FeedbackChecks.error("revisionIdentity", "A correction cannot change an existing annotation identity.")
                }
            }
            for review in reviews {
                if oldReviews[review.id] == nil { try FeedbackChecks.follows(review.authored, parent.document.authored) }
                guard oldLabels[review.id] == nil, oldReviews[review.id].map({ $0 == review }) ?? true else {
                    throw FeedbackChecks.error("revisionIdentity", "A correction cannot change an existing review identity.")
                }
            }
        } else if revision != 0 { throw FeedbackChecks.error("revision", "An initial reward revision must have index zero.") }
        guard resolved == (try source.resolve(authored: authored, annotations: annotations, reviews: reviews)) else {
            throw FeedbackChecks.error("resolution", "Stored manual rewards disagree with their explicit annotations and review coverage.")
        }
    }
}

public struct VerifiedFeedbackSource: Sendable {
    public let metadata: FeedbackTrajectoryMetadata
    public let sha256: String
    public let program: RewardProgram
    private let byPacket: [UUID: FeedbackEligibleInterval]
    private let rules: [RewardRule]

    public init(metadataBytes: Data, expectedSHA256: String, programBytes: Data) throws {
        try FeedbackChecks.digest(expectedSHA256)
        guard metadataBytes.count <= FeedbackLimits.maximumBytes, FeedbackChecks.sha256(metadataBytes) == expectedSHA256 else {
            throw FeedbackChecks.error("sourceIntegrity", "The frozen source projection failed its expected digest.")
        }
        metadata = try FeedbackChecks.decode(FeedbackTrajectoryMetadata.self, data: metadataBytes)
        sha256 = expectedSHA256
        guard programBytes.count <= 1_048_576, FeedbackChecks.sha256(programBytes) == metadata.programSHA256 else {
            throw FeedbackChecks.error("programIntegrity", "The reward definition differs from the frozen program identity.")
        }
        program = try FeedbackChecks.decode(RewardProgram.self, data: programBytes).validated()
        try FeedbackChecks.digest(metadata.trajectorySHA256); try FeedbackChecks.digest(metadata.policySignature)
        rules = program.rules.filter { $0.kind == .manualMarker }.sorted { $0.id.uuidString < $1.id.uuidString }
        guard metadata.schemaVersion == 1, metadata.status == "awaiting_manual_review", metadata.controlClosureKnown,
              metadata.programID == program.id,
              metadata.manualRules == rules.map({ FeedbackManualRule(id: $0.id, amount: $0.amount) }),
              (1...FeedbackLimits.maximumIntervals).contains(metadata.intervals.count),
              !rules.isEmpty, metadata.intervals.count <= FeedbackLimits.maximumPairs / rules.count,
              metadata.closedWallTimeMS > 0, metadata.closedWallTimeMS <= FeedbackChecks.maximumWallMS else {
            throw FeedbackChecks.error("source", "Only a bounded, joined collector projection awaiting manual rewards is reviewable.")
        }
        var lookup: [UUID: FeedbackEligibleInterval] = [:], observations: Set<UUID> = [], previousSequence: UInt64?
        var episodeEnds: [UUID: UInt64] = [:]
        for interval in metadata.intervals {
            let target = interval.target
            guard interval.eligible, ["executed", "semanticBoundaryPrefix"].contains(interval.execution),
                  ["continuing", "terminated", "truncated"].contains(interval.outcome),
                  interval.execution != "semanticBoundaryPrefix" || interval.outcome != "continuing",
                  target.startNanos < target.endNanos, target.endNanos <= metadata.closedNanos,
                  interval.endpointObservationID != target.observationID,
                  interval.packetSequence == interval.drawIndex,
                  previousSequence.map({ interval.packetSequence > $0 }) ?? true,
                  lookup[target.packetID] == nil, observations.insert(target.observationID).inserted,
                  episodeEnds[target.episodeID].map({ target.startNanos >= $0 }) ?? true else {
                throw FeedbackChecks.error("interval", "A review interval is duplicated, ineligible, overlapping or differs from its original behavior stream.")
            }
            if let bootstrap = interval.bootstrap {
                guard interval.outcome == "truncated", bootstrap.episodeID == target.episodeID,
                      bootstrap.policyID == metadata.policyID, bootstrap.observationID == interval.endpointObservationID,
                      bootstrap.cutoffNanos == target.endNanos, bootstrap.value.isFinite, !bootstrap.recurrentReset else {
                    throw FeedbackChecks.error("bootstrap", "A truncation must preserve its exact same-episode, pre-reset source bootstrap.")
                }
            } else if interval.outcome == "truncated" { throw FeedbackChecks.error("bootstrap", "The source truncation is missing its bootstrap.") }
            lookup[target.packetID] = interval; previousSequence = interval.packetSequence; episodeEnds[target.episodeID] = target.endNanos
        }
        byPacket = lookup
    }

    private func validate(_ authored: FeedbackAuthorship) throws {
        guard authored.wallTimeMS > 0, authored.wallTimeMS <= FeedbackChecks.maximumWallMS else {
            throw FeedbackChecks.error("authorship", "Annotation wall-time audit is missing or outside its supported range.")
        }
        if authored.clockID == metadata.clockID {
            guard authored.observedNanos >= metadata.closedNanos else { throw FeedbackChecks.error("authorship", "Retrospective feedback precedes the joined source boundary.") }
        } else {
            guard authored.sessionID != metadata.sourceSessionID, authored.wallTimeMS >= metadata.closedWallTimeMS else {
                throw FeedbackChecks.error("authorship", "A different authoring clock requires a new session and a later wall-time audit.")
            }
        }
    }

    func resolve(authored: FeedbackAuthorship, annotations: [FeedbackAnnotation], reviews: [FeedbackReviewCompletion]) throws -> [FeedbackIntervalResolution] {
        try validate(authored)
        guard annotations.count <= FeedbackLimits.maximumAnnotations, reviews.count <= FeedbackLimits.maximumReviews else {
            throw FeedbackChecks.error("capacity", "Manual feedback exceeds its bounded annotation or review count.")
        }
        let allowedRules = Set(rules.map(\.id))
        var ids: Set<UUID> = [], counts: [FeedbackReviewPair: Int] = [:], lastLabel: [FeedbackReviewPair: FeedbackAuthorship] = [:]
        var lastAnnotationTime: FeedbackAuthorship?
        for (index, label) in annotations.enumerated() {
            try validate(label.authored); try FeedbackChecks.follows(authored, label.authored)
            if let lastAnnotationTime { try FeedbackChecks.follows(label.authored, lastAnnotationTime) }
            let pair = FeedbackReviewPair(packetID: label.target.packetID, ruleID: label.ruleID)
            guard label.sequence == index, ids.insert(label.id).inserted, byPacket[label.target.packetID]?.target == label.target,
                  allowedRules.contains(label.ruleID), (1...FeedbackLimits.maximumCount).contains(label.count) else {
                throw FeedbackChecks.error("annotation", "A label has an invalid target, rule, count, sequence or authoring order.")
            }
            let sum = (counts[pair] ?? 0) + label.count
            guard sum <= FeedbackLimits.maximumCount else { throw FeedbackChecks.error("capacity", "A rule's interval annotation count exceeds its bound.") }
            counts[pair] = sum; lastLabel[pair] = label.authored; lastAnnotationTime = label.authored
        }
        var covered: Set<FeedbackReviewPair> = [], total = 0
        var lastReviewTime: FeedbackAuthorship?
        for (index, review) in reviews.enumerated() {
            try validate(review.authored); try FeedbackChecks.follows(authored, review.authored)
            if let lastReviewTime { try FeedbackChecks.follows(review.authored, lastReviewTime) }
            total += review.pairs.count
            guard review.sequence == index, ids.insert(review.id).inserted, !review.pairs.isEmpty,
                  total <= FeedbackLimits.maximumPairs else {
                throw FeedbackChecks.error("review", "Review completion is duplicated, unbounded or out of order.")
            }
            for pair in review.pairs {
                if let labelTime = lastLabel[pair] { try FeedbackChecks.follows(review.authored, labelTime) }
                guard byPacket[pair.packetID] != nil, allowedRules.contains(pair.ruleID), covered.insert(pair).inserted else {
                    throw FeedbackChecks.error("review", "Review completion must uniquely cover known pairs after their last annotation.")
                }
            }
            lastReviewTime = review.authored
        }
        guard Set(counts.keys).isSubset(of: covered) else { throw FeedbackChecks.error("unreviewedLabel", "An annotated pair needs explicit review completion before publication.") }
        return try metadata.intervals.map { interval in
            var sum = 0.0, complete = true
            let components = try rules.map { rule -> FeedbackRuleResolution in
                let pair = FeedbackReviewPair(packetID: interval.target.packetID, ruleID: rule.id)
                guard covered.contains(pair) else { complete = false; return .init(ruleID: rule.id, reviewed: false, count: nil, value: nil) }
                let count = counts[pair] ?? 0, value = Double(count) * rule.amount
                sum += value
                guard value.isFinite, sum.isFinite else { throw FeedbackChecks.error("nonfinite", "Manual reward arithmetic must remain finite.") }
                return .init(ruleID: rule.id, reviewed: true, count: count, value: value)
            }
            return .init(packetID: interval.target.packetID, components: components, manualReward: complete ? sum : nil)
        }
    }
}

enum FeedbackChecks {
    static let maximumWallMS: UInt64 = 253_402_300_799_999
    static func error(_ code: String, _ message: String) -> AstraError { .init("feedback." + code, message) }
    static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func digest(_ value: String) throws {
        guard value.utf8.count == 64, value.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw error("digest", "Feedback artifacts require a complete lower-case SHA-256 digest.")
        }
    }
    static func follows(_ value: FeedbackAuthorship, _ old: FeedbackAuthorship) throws {
        if value.clockID == old.clockID {
            guard value.observedNanos >= old.observedNanos else { throw error("revisionTime", "The correction predates its parent revision.") }
        } else {
            guard value.sessionID != old.sessionID, value.wallTimeMS >= old.wallTimeMS else {
                throw error("revisionTime", "A correction on another clock requires a new session and later wall-time audit.")
            }
        }
    }
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        guard data.count <= FeedbackLimits.maximumBytes else { throw error("capacity", "Feedback metadata exceeds its byte limit.") }
        return data
    }
    static func decode<T: Codable>(_ type: T.Type, data: Data) throws -> T {
        guard data.count <= FeedbackLimits.maximumBytes else { throw error("capacity", "Feedback metadata exceeds its byte limit.") }
        let value = try JSONDecoder().decode(type, from: data)
        let original = try JSONDecoder().decode(JSONValue.self, from: data)
        let expected = try JSONValue.encode(value)
        try sameShape(original, expected)
        return value
    }
    private static func sameShape(_ value: JSONValue, _ expected: JSONValue) throws {
        switch (value, expected) {
        case (.object(let left), .object(let right)):
            guard Set(left.keys) == Set(right.keys) else { throw error("fields", "Feedback metadata contains missing or unknown fields.") }
            for key in left.keys { try sameShape(left[key]!, right[key]!) }
        case (.array(let left), .array(let right)):
            guard left.count == right.count else { throw error("fields", "Feedback metadata has an invalid array shape.") }
            for (a, b) in zip(left, right) { try sameShape(a, b) }
        default: break
        }
    }
}
