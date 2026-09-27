import Foundation
import AstraCore
import AstraPlatform

struct DesktopLiveValue: Identifiable, Sendable {
    let id: UUID
    let name: String
    let text: String
    let current: Bool
}

enum DesktopLiveValues {
    static func rows(store: LiveSignalStore, binding: LiveSignalBinding, cutoff: UInt64) throws -> [DesktopLiveValue] {
        let readings = try store.readings(binding: binding, cutoffNanos: cutoff)
        let byID = Dictionary(uniqueKeysWithValues: readings.map { ($0.signalID, $0) })
        return binding.signals.map { signal in
            guard let reading = byID[signal.id], reading.observedNanos <= cutoff,
                  cutoff - reading.observedNanos <= UInt64(signal.maximumAgeMS) * 1_000_000,
                  reading.confidence >= signal.minimumConfidence else {
                return .init(id: signal.id, name: signal.name, text: "Waiting for a current value", current: false)
            }
            let text: String, known: Bool
            switch reading.value {
            case .number(let value): text = String(format: "%.7g", value); known = true
            case .flag(let value): text = value ? "True" : "False"; known = true
            case .text(let value): text = String(value.prefix(160)); known = true
            case .unknown(let reason): text = String(reason.prefix(160)); known = false
            }
            return .init(id: signal.id, name: signal.name, text: text, current: known)
        }
    }
}

/// The synchronous callback is joined by the same lock as closure. Ending an
/// episode retires only its source binding; the run's listener stays available
/// for the next reset and never creates manual-marker coverage.
final class DesktopLiveValueProducer: @unchecked Sendable {
    private let source: LiveSignalStore
    private let binding: LiveSignalBinding
    private let lock = NSLock()
    private var closed = false
    private var generationID: UUID?
    init(source: LiveSignalStore, binding: LiveSignalBinding) { self.source = source; self.binding = binding }
    var producer: DesktopEpisodeManualProducer {
        .init(observe: { [self] observation, deliver in
            try lock.withLock {
                guard !closed, observation.episodeID == binding.episodeID,
                      generationID == nil || generationID == observation.generationID else {
                    throw AstraError("liveSignal.episode", "The live values belong to a different or closed episode.")
                }
                generationID = observation.generationID
                let readings = try source.seal(binding: binding, observationID: observation.observationID, cutoffNanos: observation.cutoffNanos)
                try deliver(.init(observationID: observation.observationID, episodeID: observation.episodeID,
                    cutoffNanos: observation.cutoffNanos, readings: readings, markers: [], coverage: nil))
            }
        }, closeAndJoin: { [self] in
            lock.withLock { closed = true; source.deactivate(binding: binding) }
        })
    }
}
