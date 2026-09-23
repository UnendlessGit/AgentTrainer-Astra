import Foundation
import AstraCore

struct DesktopLearningOptions: Sendable {
    var contextVocabulary = ContextVocabulary.empty
    var initialCheckpointID: UUID?
    var resume = false
    var iterations = 20
    var periodMS = 100
    var leadMS = 100
    var packetCapacity = 16
    var actions = ActionCapabilities(mouseButtons: [0], absolutePointer: true)
    var contextIDs: [Int] = []
    var training = ReinforcementSettings()
    var surfaceBindings: [String: String] = [:]

    func validated() throws -> Self {
        _ = try contextVocabulary.validated()
        guard surfaceBindings.count <= 16, surfaceBindings.allSatisfy({ !$0.key.isEmpty && $0.key.utf8.count <= 256
            && !$0.value.isEmpty && $0.value.utf8.count <= 256 }) else {
            throw AstraError("desktop.surfaceBindings", "The reward source selections are invalid.")
        }
        guard (1...100_000).contains(iterations), !resume || initialCheckpointID != nil else {
            throw AstraError("desktop.options", "Choose an iteration target and a saved desktop checkpoint when resuming.")
        }
        if !resume { _ = try training.validated() }
        if initialCheckpointID == nil {
            _ = try actions.validated()
            guard !actions.isEmpty, [50, 100].contains(periodMS), (1...2000).contains(leadMS), [16, 32, 64].contains(packetCapacity) else {
                throw AstraError("desktop.policyOptions", "Choose controls, a decision cadence and a positive execution lead for the new policy.")
            }
        }
        return self
    }
    var model: JSONValue {
        contextVocabulary.applying(to: .object(["period_ms": .integer(Int64(periodMS)), "lead_ms": .integer(Int64(leadMS)),
                 "packet_capacity": .integer(Int64(packetCapacity))]))
    }
}

/// Fully bound task settings, separate from live clock/process/window identity.
/// Resumption retains the checkpoint's optimizer and task contract even if the
/// editor's current defaults have changed.
struct DesktopLearningConfiguration: Sendable {
    let environmentID: UUID
    let environment: JSONValue
    let training: JSONValue
    let model: JSONValue
    let contextIDs: [Int]
    let reward: RewardProgramBinding
    let program: RewardProgram
    let minimumDecisions: Int
    var retrospective: Bool { program.rules.contains { $0.kind == .manualMarker } }

    init(prepared: PreparedDesktopPolicy, reward: RewardProgramBinding, scope: ControlScope,
         options: DesktopLearningOptions, resumedConfiguration: JSONValue? = nil, suspended: PendingFeedbackDocument? = nil) throws {
        _ = try options.validated()
        self.reward = try reward.validated(scope: scope); program = try reward.resolved(scope: scope)
        model = try prepared.manifest.required("model")
        let vocabulary = try prepared.manifest.required("actions")
        guard let fields = vocabulary.fields,
              Set(fields.keys) == ["keyCodes", "mouseButtons", "absolutePointer", "relativePointer", "scroll", "scrollUnitsPerPoint"],
              let units = fields["scrollUnitsPerPoint"]?.int, [1, 2, 4, 8, 16].contains(units) else {
            throw AstraError("desktop.actionVocabulary", "The checkpoint's complete action vocabulary is missing or invalid.")
        }
        let actions = try vocabulary.decode(ActionCapabilities.self)
        _ = try actions.validated()
        guard !actions.isEmpty, let period = model.fields?["period_ms"]?.int, (1...1000).contains(period),
              let lead = model.fields?["lead_ms"]?.int, (1...2000).contains(lead),
              let capacity = model.fields?["packet_capacity"]?.int, [16, 32, 64].contains(capacity),
              let sizes = try? model.required("context_sizes").decode([Int].self), sizes.count <= 32,
              sizes.allSatisfy({ (1...65_536).contains($0) }) else {
            throw AstraError("desktop.policyContract", "This checkpoint has incompatible desktop controls, contexts or timing.")
        }
        if let suspended {
            guard suspended.checkpoint.matchesIdentity(of: prepared.checkpoint.document) else {
                throw AstraError("feedback.originalPolicy", "Continue with the exact original behavior checkpoint.")
            }
            environment = try suspended.configuration.required("environment")
            training = try suspended.configuration.required("training")
            contextIDs = try suspended.configuration.required("contextIDs").decode([Int].self)
            environmentID = try environment.requiredUUID("identity")
            guard environment.fields?["reward_signature"] == .string(reward.definitionSignature),
                  environment.fields?["reset_signature"] == .string(reward.resetSignature),
                  try JSONValue.encode(suspended.configuration.required("model")) == JSONValue.encode(model),
                  try JSONValue.encode(environment.required("action_vocabulary")) == JSONValue.encode(vocabulary) else {
                throw AstraError("feedback.changedTask", "The suspended experience must retain its original task, controls and model.")
            }
        } else if options.resume {
            guard let previous = resumedConfiguration, let sourceRun = prepared.checkpoint.document.runID,
                  prepared.checkpoint.document.kind == "reinforcement",
                  ["train.reinforcement.external", "checkpoint.externalBoundary"].contains(previous.fields?["operation"]?.text ?? ""),
                  previous.fields?["runID"]?.uuid == sourceRun,
                  previous.fields?["agentID"]?.uuid == prepared.checkpoint.document.agentID,
                  previous.fields?["destinationCheckpointID"]?.uuid == prepared.checkpoint.document.id,
                  prepared.manifest.fields?["metrics"]?.fields?["sourceKind"] == .string("external_rollout"),
                  prepared.manifest.fields?["metrics"]?.fields?["requiresEnvironmentReset"] == .bool(true) else {
                throw AstraError("desktop.resumeSource", "Resume requires an external reinforcement checkpoint and its matching saved run configuration.")
            }
            environment = try previous.required("environment")
            training = try prepared.manifest.required("trainingConfig")
            contextIDs = try previous.required("contextIDs").decode([Int].self)
            guard let identity = environment.fields?["identity"]?.uuid else {
                throw AstraError("desktop.resumeEnvironment", "The saved desktop task identity is invalid.")
            }
            environmentID = identity
            guard environment.fields?["reward_signature"] == .string(reward.definitionSignature),
                  environment.fields?["reset_signature"] == .string(reward.resetSignature),
                  environment.fields?["maximum_episode_ms"]?.int == program.maximumEpisodeMS,
                  environment.fields?["period_ms"]?.int == period, environment.fields?["lead_ms"]?.int == lead,
                  try JSONValue.encode(environment.required("action_vocabulary")) == JSONValue.encode(vocabulary),
                  try JSONValue.encode(previous.required("model")) == JSONValue.encode(model) else {
                throw AstraError("desktop.resumeTask", "The reward definition, reset, controls or model differ from the saved task. Start a new learning run to change them.")
            }
        } else {
            environmentID = UUID()
            training = options.training.payload
            contextIDs = options.contextIDs.isEmpty ? Array(repeating: 0, count: sizes.count) : options.contextIDs
            environment = .object(["schema_version": .integer(1), "identity": .string(environmentID.uuidString.lowercased()),
                "action_vocabulary": vocabulary, "period_ms": .integer(Int64(period)), "lead_ms": .integer(Int64(lead)),
                "maximum_episode_ms": .integer(Int64(program.maximumEpisodeMS)),
                "maximum_observation_bytes": .integer(256 * 1024 * 1024), "maximum_surfaces": .integer(16),
                "maximum_frame_age_ms": .integer(250), "discount_half_life_ms": .number(options.training.discountHalfLifeSeconds * 1000),
                "seed": .integer(Int64(options.training.seed)), "seeded_reset": .bool(false),
                "reward_signature": .string(reward.definitionSignature), "reset_signature": .string(reward.resetSignature)])
        }
        guard contextIDs.count == sizes.count, zip(contextIDs, sizes).allSatisfy({ $0 >= 0 && $0 < $1 }),
              let minimum = training.fields?["rollout_decisions"]?.int, (1...65_536).contains(minimum) else {
            throw AstraError("desktop.trainingContract", "The saved learning settings or context values are invalid.")
        }
        minimumDecisions = minimum
    }

    func collector(identity: DesktopEvidenceIdentity, checkpoint: CheckpointDocument, destination: URL,
                   previousActorProgress: JSONValue?, continuation: PendingFeedbackDocument? = nil) throws -> JSONValue {
        guard identity.environmentID == environmentID else {
            throw AstraError("desktop.environmentIdentity", "This collector is bound to a different desktop task.")
        }
        var value: [String: JSONValue] = ["schemaVersion": .integer(1),
            "clockID": .string(identity.clockID.uuidString.lowercased()), "environment": environment,
            "model": model, "training": training, "policyID": .string(checkpoint.id.uuidString.lowercased()),
            "policySignature": .string(checkpoint.policySignature), "actorSourceID": .string(identity.actorSourceID.uuidString.lowercased()),
            "environmentSourceID": .string(identity.environmentSourceID.uuidString.lowercased()),
            "contextIDs": try .encode(contextIDs), "destination": .string(destination.path), "purpose": .string("learning")]
        if retrospective {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let bytes = try encoder.encode(reward.definition)
            value["schemaVersion"] = .integer(2); value["purpose"] = .string("retrospective")
            value["behaviorBatchID"] = .string(destination.lastPathComponent)
            value["retrospective"] = .object(["programBase64": .string(bytes.base64EncodedString()),
                "programSHA256": .string(reward.definitionSignature), "sourceSessionID": .string(identity.runID.uuidString.lowercased())])
        }
        if let previousActorProgress {
            try PolicyActorValidation.progress(previousActorProgress)
            // A new process run can continue the same authenticated RNG stream.
            // Only its transport binding changes; the checkpoint remains intact.
            var rebound = previousActorProgress.fields!
            rebound["runID"] = .string(identity.runID.uuidString.lowercased())
            value["previousActorProgress"] = .object(rebound)
        }
        if let continuation, let last = continuation.fragments.last {
            guard retrospective, continuation.checkpoint.matchesIdentity(of: checkpoint) else {
                throw AstraError("feedback.continuation", "Suspended collection requires its original retrospective policy.")
            }
            value["behaviorBatchID"] = .string(continuation.id.uuidString.lowercased())
            value["continuationSource"] = .object(["sourcePath": .string(last.sourceDirectory), "manifestSHA256": .string(last.manifestSHA256)])
            value["previousActorProgress"] = nil // The collector authenticates the original cursor without rewriting its run.
        }
        return .object(value)
    }
}
