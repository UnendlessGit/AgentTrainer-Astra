import Foundation
import AstraCore
import AstraPlatform

/// One independently sized transport per fixed source role. A large display
/// does not multiply every smaller window's allocation. The parent owns process
/// lifetime and retires these mappings only after the consumer has joined.
final class ObservationFrameRings: Sendable {
    let rings: [SharedFrameRing]
    var primary: SharedFrameRing { rings[0] }
    var descriptors: [JSONValue] {
        rings.map { .object(["path": .string($0.url.path), "ringID": .string($0.ringID.uuidString.lowercased())]) }
    }

    init(url: URL, runID: UUID, capacities: [Int], maximumSlots: Int = 2, maximumBytes: Int = 512 * 1024 * 1024) throws {
        guard (1...16).contains(capacities.count), capacities.allSatisfy({ (4...FrameArchive.maximumFrameBytes).contains($0) }),
              capacities.reduce(0, +) <= 256 * 1024 * 1024, (1...4).contains(maximumSlots) else {
            throw AstraError("inference.frameBudget", "An observation requires 1–16 bounded sources and at most 256 MiB of pixels.")
        }
        let slots = max(1, min(maximumSlots, maximumBytes / capacities.reduce(0, +)))
        var created: [SharedFrameRing] = []
        do {
            for (index, capacity) in capacities.enumerated() {
                let path = index == 0 ? url : url.deletingLastPathComponent().appendingPathComponent("\(url.lastPathComponent).\(index)")
                created.append(try SharedFrameRing(url: path, runID: runID, slotCount: slots, slotCapacity: capacity))
            }
        } catch {
            // No reader can have opened these unpublished mappings yet.
            created.forEach { $0.closeAfterConsumerExit() }
            throw error
        }
        rings = created
    }

    func publish(_ frames: [PolicyActorOwnedObservation.Frame]) throws -> [SharedFrameReference] {
        guard frames.count == rings.count else { throw AstraError("inference.sourceMembership", "The observation differs from its prepared source group.") }
        var published: [SharedFrameReference] = []
        do {
            for (frame, ring) in zip(frames, rings) { published.append(try ring.publish(pixels: frame.pixels, metadata: frame.metadata)) }
            return published
        } catch {
            // The caller has not sent a partial observation to the consumer.
            for reference in published { try? release(reference.acknowledgement) }
            throw error
        }
    }

    func release(_ acknowledgement: SharedFrameAcknowledgement) throws {
        guard let ring = rings.first(where: { $0.ringID == acknowledgement.ringID }) else {
            throw AstraError("inference.frameAcknowledgement", "The consumer released a foreign frame transport.")
        }
        try ring.release(acknowledgement)
    }
    func closeAfterConsumerExit() { rings.forEach { $0.closeAfterConsumerExit() } }
}
