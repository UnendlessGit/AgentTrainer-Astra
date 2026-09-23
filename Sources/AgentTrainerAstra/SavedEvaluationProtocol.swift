import Foundation
import AstraCore

enum SavedEvaluationProtocol {
    /// Resolve once for the whole comparison. Candidate checkpoints never choose
    /// their own demonstrations, splits, or canonicalization configuration.
    static func resolve(checkpoint: CheckpointDocument, root: URL, split: String, layout: ArtifactStorageLayout? = nil) async throws -> EvaluationProtocol {
        guard let runID = checkpoint.runID else {
            throw AstraError("evaluation.dataset", "Choose a checkpoint with a saved demonstration dataset.")
        }
        let source = try await LearningFiles.read(root.appendingPathComponent("Jobs/\(runID.uuidString.lowercased())/configuration.json"))
        guard source.fields?["agentID"]?.text.flatMap(UUID.init(uuidString:)) == checkpoint.agentID,
              source.fields?["runID"]?.text.flatMap(UUID.init(uuidString:)) == runID,
              source.fields?["operation"] == .string("train.behavioral") else {
            throw AstraError("evaluation.sourceIdentity", "The saved demonstration configuration belongs to another run or has no behavioral dataset.")
        }
        let layout = layout ?? .defaults(catalogRoot: root)
        let manifest = try await LearningFiles.read(layout.modelsRoot.appendingPathComponent("\(checkpoint.id.uuidString.lowercased())/manifest.json"))
        let dataset = try SavedArtifactLocations.dataset(source.required("dataset"), checkpointManifest: manifest, layout: layout)
        guard manifest.fields?["id"]?.text.flatMap(UUID.init(uuidString:)) == checkpoint.id,
              manifest.fields?["policySignature"]?.text == checkpoint.policySignature else {
            throw AstraError("evaluation.sourceCheckpoint", "The source checkpoint no longer matches its saved catalog identity.")
        }
        let identity: JSONValue
        let datasetID: UUID?
        let provenance: String
        switch dataset.fields?["kind"]?.text {
        case "recordings":
            guard let path = dataset.fields?["path"]?.text else { throw AstraError("evaluation.dataset", "The recorded dataset path is missing.") }
            let revision = try await LearningFiles.read(URL(fileURLWithPath: path, isDirectory: true).appendingPathComponent("manifest.json"))
            guard let identifier = revision.fields?["id"]?.text.flatMap(UUID.init(uuidString:)),
                  revision.fields?["indexSHA256"]?.text != nil,
                  revision.fields?["sources"] != nil else {
                throw AstraError("evaluation.dataset", "The saved demonstration revision has no verifiable identity or source provenance.")
            }
            datasetID = identifier; provenance = "recorded_demonstrations"
            identity = .object(["revision": revision])
        case "practice_oracle":
            datasetID = nil; provenance = "practice_oracle"
            identity = .object(["demonstrations": dataset, "model": try manifest.required("model"), "actions": try manifest.required("actions")])
        default: throw AstraError("evaluation.dataset", "This checkpoint has no supported saved demonstration dataset.")
        }
        return try EvaluationProtocol(sourceRunID: runID, sourceCheckpointID: checkpoint.id,
            sourceName: checkpoint.name, dataset: dataset, identity: identity, expectedDatasetID: datasetID,
            provenance: provenance, policySignature: checkpoint.policySignature, split: split,
            verificationMode: source.fields?["verificationMode"] == .bool(true))
    }
}
