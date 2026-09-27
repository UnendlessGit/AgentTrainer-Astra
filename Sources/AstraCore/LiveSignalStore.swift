import Foundation

public struct LiveSignalDescriptor: Codable, Equatable, Sendable {
    public let id: UUID
    public let name: String
    public let maximumAgeMS: Int
    public let minimumConfidence: Double
}

/// A source generation begins before reset observation. It is independent of
/// the episode runner's callback generation, which starts after reset completes.
public struct LiveSignalBinding: Codable, Equatable, Sendable {
    public let sessionID: UUID
    public let bindingID: UUID
    public let episodeID: UUID
    public let resetID: UUID
    public let publishedAtNanos: UInt64
    public let signals: [LiveSignalDescriptor]
}
public struct LiveSignalValueUpdate: Sendable {
    public let signalID: UUID
    public let value: SignalValue
    public let confidence: Double
    public init(signalID: UUID, value: SignalValue, confidence: Double = 1) {
        self.signalID = signalID; self.value = value; self.confidence = confidence
    }
}
public struct LiveSignalReceipt: Codable, Sendable {
    public let bindingID: UUID
    public let sequence: UInt64
    public let nextSequence: UInt64
    public let receivedAtNanos: UInt64
}
public struct LiveSignalStoreStatus: Sendable {
    public let binding: LiveSignalBinding?
    public let nextSequence: UInt64
    public let lastReceivedNanos: UInt64?
    public let retainedReadings: Int
    public let closed: Bool
    public let issue: String?
}

/// As-of scalar evidence, never a reward marker or a promise that no feedback
/// occurred. All clocks are minted at receive-side admission. History loss,
/// absence and stale values remain unknown to the existing reward evaluator.
public final class LiveSignalStore: @unchecked Sendable {
    public static let maximumReadings = 4096
    public static let maximumBytes = 4 * 1024 * 1024
    public nonisolated let sessionID: UUID
    private let descriptors: [LiveSignalDescriptor]
    private let allowed: Set<UUID>
    private let clock: @Sendable () -> UInt64
    private let lock = NSLock()
    private var binding: LiveSignalBinding?
    private var nextSequence: UInt64 = 0
    private var lastReceived: UInt64?
    private var sealedThrough: UInt64?
    private var sealedIDs: Set<UUID> = []
    private var entries: [Entry] = []
    private var head = 0, retainedBytes = 0
    private var closed = false
    private var discontinuity = false
    private var issue: String?
    private struct Entry {
        let reading: SignalReading
        let bytes: Int
    }

    public init(sessionID: UUID, signals: [RewardSignal], clock: @escaping @Sendable () -> UInt64) throws {
        guard (1...32).contains(signals.count), Set(signals.map(\.id)).count == signals.count,
              signals.allSatisfy({ $0.kind == .manual }) else {
            throw AstraError("liveSignal.signals", "A live value source needs 1–32 distinct manual state signals.")
        }
        for signal in signals { _ = try signal.validated() }
        self.sessionID = sessionID; self.clock = clock
        descriptors = signals.map { .init(id: $0.id, name: $0.name, maximumAgeMS: $0.maximumAgeMS, minimumConfidence: $0.minimumConfidence) }
        allowed = Set(signals.map(\.id))
    }
    public var status: LiveSignalStoreStatus {
        lock.withLock { .init(binding: binding, nextSequence: nextSequence, lastReceivedNanos: lastReceived,
                             retainedReadings: entries.count - head, closed: closed, issue: issue) }
    }
    public func bind(episodeID: UUID, resetID: UUID) throws -> LiveSignalBinding {
        try lock.withLock {
            guard !closed else { throw AstraError("liveSignal.closed", "The live state source has closed.") }
            clearValues()
            let published = clock()
            let value = LiveSignalBinding(sessionID: sessionID, bindingID: UUID(), episodeID: episodeID,
                resetID: resetID, publishedAtNanos: published, signals: descriptors)
            binding = value
            return value
        }
    }
    public func deactivate(binding expected: LiveSignalBinding) {
        lock.withLock {
            guard binding?.bindingID == expected.bindingID else { return }
            binding = nil; clearValues()
        }
    }
    public func close() {
        lock.withLock { closed = true; binding = nil; clearValues() }
    }

    public func accept(sessionID: UUID, bindingID: UUID, episodeID: UUID, sequence: UInt64,
                       values: [LiveSignalValueUpdate]) throws -> LiveSignalReceipt {
        // Validate finite bounded values before taking ownership. The receive
        // timestamp is deliberately assigned only after this validation.
        guard (1...32).contains(values.count), Set(values.map(\.signalID)).count == values.count,
              values.allSatisfy({ allowed.contains($0.signalID) && $0.confidence.isFinite && (0...1).contains($0.confidence) }) else {
            throw AstraError("liveSignal.values", "Publish distinct allowed state signals with confidence between zero and one.")
        }
        for value in values { _ = try value.value.validated() }
        return try lock.withLock {
            guard !closed, let binding, binding.sessionID == sessionID, binding.bindingID == bindingID, binding.episodeID == episodeID else {
                throw AstraError("liveSignal.binding", "Publish only to the currently announced session, binding and episode.")
            }
            guard !discontinuity else {
                throw AstraError("liveSignal.discontinuity", "The state source skipped an update. A fresh reset binding is required before its values can be trusted again.")
            }
            let received = clock()
            guard received >= binding.publishedAtNanos, lastReceived.map({ received >= $0 }) ?? true,
                  sealedThrough.map({ received > $0 }) ?? true else {
                throw AstraError("liveSignal.causality", "A state update cannot change an already sealed observation cutoff.")
            }
            guard sequence == nextSequence, nextSequence < UInt64.max else {
                if sequence > nextSequence {
                    // Known missing history must not disappear between reward
                    // cutoffs even if a newer current value arrives. Only a
                    // fresh reset binding clears this discontinuity.
                    discontinuity = true
                    issue = "The state source skipped an update. A fresh reset binding is required."
                    for signal in descriptors { append(.init(signalID: signal.id, episodeID: episodeID, eventNanos: received,
                        observedNanos: received, value: .unknown(issue!))) }
                    lastReceived = received
                }
                throw AstraError("liveSignal.sequence", "Read the binding’s nextSequence before retrying; update sequences must be consecutive.")
            }
            for value in values {
                append(.init(signalID: value.signalID, episodeID: episodeID, eventNanos: received,
                    observedNanos: received, confidence: value.confidence, value: value.value))
            }
            nextSequence += 1; lastReceived = received; issue = nil
            return .init(bindingID: bindingID, sequence: sequence, nextSequence: nextSequence, receivedAtNanos: received)
        }
    }

    /// Reset observations remain independent immutable values. The reward
    /// evaluator still checks its post-cleanup minimumEvidenceNanos barrier.
    public func readings(binding expected: LiveSignalBinding, cutoffNanos: UInt64) throws -> [SignalReading] {
        try lock.withLock {
            try validate(expected, cutoff: cutoffNanos)
            return snapshot(cutoff: cutoffNanos)
        }
    }
    /// Called once per actor observation before analysis. A later receive may
    /// serve the next cutoff, but never rewrite this seal or insert into its past.
    public func seal(binding expected: LiveSignalBinding, observationID: UUID, cutoffNanos: UInt64) throws -> [SignalReading] {
        try lock.withLock {
            try validate(expected, cutoff: cutoffNanos)
            guard sealedThrough.map({ cutoffNanos > $0 }) ?? true, !sealedIDs.contains(observationID), sealedIDs.count < 65_536 else {
                throw AstraError("liveSignal.seal", "Each live state observation needs a distinct identity and advancing cutoff.")
            }
            let result = snapshot(cutoff: cutoffNanos)
            sealedThrough = cutoffNanos; sealedIDs.insert(observationID)
            return result
        }
    }
    private func validate(_ expected: LiveSignalBinding, cutoff: UInt64) throws {
        guard !closed, binding == expected, cutoff >= expected.publishedAtNanos, cutoff <= clock() else {
            throw AstraError("liveSignal.observation", "The state observation belongs to another binding or an unavailable time.")
        }
    }
    private func snapshot(cutoff: UInt64) -> [SignalReading] {
        var newest: [UUID: SignalReading] = [:]
        for entry in entries[head...].reversed() where entry.reading.observedNanos <= cutoff {
            if newest[entry.reading.signalID] == nil { newest[entry.reading.signalID] = entry.reading }
            if newest.count == descriptors.count { break }
        }
        return descriptors.compactMap { newest[$0.id] }
    }
    private func append(_ reading: SignalReading) {
        let textBytes: Int
        switch reading.value { case .text(let text), .unknown(let text): textBytes = text.utf8.count; default: textBytes = 0 }
        let count = 192 + textBytes
        entries.append(.init(reading: reading, bytes: count)); retainedBytes += count
        while entries.count - head > Self.maximumReadings || retainedBytes > Self.maximumBytes {
            retainedBytes -= entries[head].bytes; head += 1
        }
        if head >= 1024 { entries.removeFirst(head); head = 0 }
    }
    private func clearValues() {
        nextSequence = 0; lastReceived = nil; sealedThrough = nil; sealedIDs.removeAll(keepingCapacity: true)
        entries.removeAll(keepingCapacity: true); head = 0; retainedBytes = 0; issue = nil; discontinuity = false
    }
}
