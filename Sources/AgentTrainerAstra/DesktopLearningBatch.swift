import Foundation
import AstraCore

/// The immutable collector output and native join evidence for one update.
/// The compute worker independently authenticates package bytes, likelihoods
/// and recurrent state; this admission checks the live owners and identities.
struct DesktopCollectionBoundary: Sendable {
    let identity: DesktopEvidenceIdentity
    let checkpoint: CheckpointDocument
    let collectionID: UUID
    let rolloutID: UUID?
    let path: URL
    let manifest: JSONValue
    let actorProgress: JSONValue
    let decisions: Int

    init(identity: DesktopEvidenceIdentity, checkpoint: CheckpointDocument, destination: URL,
         result: WireMessage, joins: [DesktopEpisodeJoin], actorState: PolicyActorState) throws {
        _ = try checkpoint.validated()
        _ = try result.validated()
        let manifest = try result.payload.required("manifest"), binding = try manifest.required("binding")
        let progress = try result.payload.required("actorProgress")
        try PolicyActorValidation.progress(progress)
        let sealed = result.kind == "collector.sealed"
        let retrospective = manifest.fields?["schemaVersion"] == .integer(2)
            && manifest.fields?["status"] == .string("awaiting_manual_review")
        guard sealed || result.kind == "collector.audited", result.runID == identity.runID,
              result.payload.fields?["learningEligible"] == .bool(sealed),
              result.payload.fields?["controlClosureKnown"] == .bool(true),
              destination.isFileURL, destination.standardizedFileURL.path == destination.path,
              result.payload.fields?["path"] == .string(destination.path),
              let collectionID = result.payload.fields?["collectionID"]?.uuid,
              destination.lastPathComponent == collectionID.uuidString.lowercased(),
              manifest.fields?["id"]?.uuid == collectionID, [1, 2].contains(manifest.fields?["schemaVersion"]?.int ?? 0),
              manifest.fields?["status"] == .string(sealed ? "sealed" : retrospective ? "awaiting_manual_review" : "audited"), manifest.fields?["controlClosureKnown"] == .bool(true),
              let decisions = manifest.fields?["decisions"]?.int, (0...65_536).contains(decisions),
              ["learning", "audit", "retrospective"].contains(binding.fields?["purpose"]?.text ?? "") else {
            throw AstraError("desktop.learningPackage", "The collector did not finish the expected validated local package with confirmed control closure.")
        }
        guard
              binding.fields?["runID"]?.uuid == identity.runID, binding.fields?["clockID"]?.uuid == identity.clockID,
              binding.fields?["actorSourceID"]?.uuid == identity.actorSourceID,
              binding.fields?["environmentSourceID"]?.uuid == identity.environmentSourceID,
              binding.fields?["policyID"]?.uuid == checkpoint.id,
              binding.fields?["policySignature"] == .string(checkpoint.policySignature) else {
            throw AstraError("desktop.learningIdentity", "The collection belongs to a different policy, source or clock.")
        }
        guard
              progress.fields?["runID"]?.uuid == identity.runID,
              let draw = progress.fields?["drawIndex"]?.uint64, draw < UInt64.max,
              actorState.sampledProgressKnown, !actorState.joined,
              actorState.nextPacketSequence == draw + 1, actorState.nextDrawIndex == draw + 1,
              let recordedGeneration = progress.fields?["actorResetGeneration"]?.uint64,
              actorState.actorResetGeneration >= recordedGeneration,
              try JSONValue.encode(manifest.required("actorProgress")) == JSONValue.encode(progress),
              try actorState.actorProgress.map(JSONValue.encode) == JSONValue.encode(progress) else {
            throw AstraError("desktop.learningProgress", "The sealed collection does not match the actor's last real random draw and reset state.")
        }
        guard
              !joins.isEmpty, joins.count <= 65_536, Set(joins.map(\.episodeID)).count == joins.count,
              joins.allSatisfy({ $0.runID == identity.runID && $0.actorJoined && $0.controlJoined && $0.manualProducerJoined
                  && $0.predictionResolved && $0.cleanupConfirmed }),
              joins.compactMap(\.lastProducedSequence).max() == draw else {
            throw AstraError("desktop.learningBoundary", "Learning requires the sealed collection and the exact joined actor, feedback and control boundary.")
        }
        self.identity = identity; self.checkpoint = checkpoint; self.collectionID = collectionID
        rolloutID = manifest.fields?["rolloutID"]?.uuid; path = destination; self.manifest = manifest
        actorProgress = try JSONValue.encode(progress); self.decisions = decisions
    }

    func validateHeldPause(actor: PolicyActorSession, token: PolicyActorLearningPause) async throws {
        try await actor.validateLearningPause(token)
        guard token.binding.runID == identity.runID, token.binding.collecting,
              token.binding.checkpoint.matchesIdentity(of: checkpoint),
              try token.binding.state.actorProgress.map(JSONValue.encode) == actorProgress else {
            throw AstraError("desktop.learningPause", "The held actor pause belongs to a different collection, checkpoint or sampled boundary.")
        }
    }
}

struct DesktopLearningBatch: Sendable {
    let boundary: DesktopCollectionBoundary
    let rolloutID: UUID
    private let learningPath: URL
    private let learningManifest: JSONValue
    var identity: DesktopEvidenceIdentity { boundary.identity }
    var checkpoint: CheckpointDocument { boundary.checkpoint }
    var collectionID: UUID { boundary.collectionID }
    var path: URL { learningPath }
    var manifest: JSONValue { learningManifest }
    var actorProgress: JSONValue { boundary.actorProgress }
    var decisions: Int { learningManifest.fields?["decisions"]?.int ?? 0 }

    init(identity: DesktopEvidenceIdentity, checkpoint: CheckpointDocument, destination: URL,
         result: WireMessage, joins: [DesktopEpisodeJoin], actorState: PolicyActorState) throws {
        let boundary = try DesktopCollectionBoundary(identity: identity, checkpoint: checkpoint, destination: destination,
            result: result, joins: joins, actorState: actorState)
        let reviewed = boundary.manifest.fields?["schemaVersion"] == .integer(2) && boundary.manifest.fields?["reviewSource"] != nil
        guard result.kind == "collector.sealed", let rolloutID = boundary.rolloutID, boundary.decisions > 0,
              boundary.manifest.fields?["binding"]?.fields?["purpose"] == .string("learning") || reviewed,
              case .array(let sampling) = boundary.manifest.fields?["actorSampling"], sampling.count == 4,
              sampling[0] == .string("categorical"), sampling[1].double == 1,
              sampling[2] == .string("none"), sampling[3].int == 1 else {
            throw AstraError("desktop.learningSampling", "This collection does not contain sealed eligible experience from the declared categorical behavior policy.")
        }
        self.boundary = boundary; self.rolloutID = rolloutID; learningPath = destination; learningManifest = boundary.manifest
    }
    /// The compute job authenticates each source/revision and preserves its
    /// original run/clock. Publication still uses the last actual native join.
    init(reviewed result: JSONValue, boundary: DesktopCollectionBoundary) throws {
        let manifest = try result.required("manifest"), binding = try manifest.required("binding")
        let path = URL(fileURLWithPath: try result.required("rolloutPath").decode(String.self))
        guard result.fields?["learningEligible"] == .bool(true), [2, 3].contains(manifest.fields?["schemaVersion"]?.int ?? 0),
              manifest.fields?["status"] == .string("sealed"), manifest.fields?["controlClosureKnown"] == .bool(true),
              let id = manifest.fields?["id"]?.uuid, path.lastPathComponent == id.uuidString.lowercased(),
              let rollout = manifest.fields?["rolloutID"]?.uuid,
              rollout == manifest.fields?["behaviorBatchID"]?.uuid,
              rollout == boundary.manifest.fields?["behaviorBatchID"]?.uuid,
              binding.fields?["policyID"]?.uuid == boundary.checkpoint.id,
              binding.fields?["policySignature"] == .string(boundary.checkpoint.policySignature),
              try JSONValue.encode(manifest.required("actorProgress")) == boundary.actorProgress,
              let count = manifest.fields?["decisions"]?.int, count > 0,
              let minimum = binding.fields?["training"]?.fields?["rollout_decisions"]?.int, count >= minimum else {
            throw AstraError("feedback.learningBatch", "The reviewed batch differs from its policy, complete experience or final actor boundary.")
        }
        for key in ["environment", "model", "training", "contextIDs"] {
            guard try JSONValue.encode(binding.required(key)) == JSONValue.encode(boundary.manifest.required("binding").required(key)) else {
                throw AstraError("feedback.learningTask", "Reviewed fragments have inconsistent task or learning settings.")
            }
        }
        self.boundary = boundary; rolloutID = rollout; learningPath = path; learningManifest = manifest
    }
    func validateHeldPause(actor: PolicyActorSession, token: PolicyActorLearningPause) async throws {
        try await boundary.validateHeldPause(actor: actor, token: token)
    }
}

struct ExternalLearningResult: Sendable {
    let runID: UUID
    let checkpoint: CheckpointDocument?
    let cancelled: Bool
    let actorProgress: JSONValue?
}

struct PreparedDesktopPolicy: Sendable {
    let checkpoint: PolicyActorCheckpoint
    let manifest: JSONValue
}
