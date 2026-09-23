import Foundation
import AstraCore

/// Original native proof retained when a batch is suspended. Reopening also
/// authenticates its immutable source and requires an idle control owner.
struct PendingReviewBoundary: Codable, Sendable {
    let identity: DesktopEvidenceIdentity
    let joins: [DesktopEpisodeJoin]
    let actor: PolicyActorState
    let result: WireMessage

    func validated(checkpoint: CheckpointDocument) throws -> DesktopCollectionBoundary {
        try .init(identity: identity, checkpoint: checkpoint,
            destination: URL(fileURLWithPath: try result.payload.required("path").decode(String.self)),
            result: result, joins: joins, actorState: actor)
    }
}
