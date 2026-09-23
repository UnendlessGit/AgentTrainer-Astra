import Foundation
import AstraCore

/// Resolve immutable artifact identities into this workspace's current storage.
/// Saved configurations remain byte-for-byte historical evidence; only the
/// runtime request receives current paths.
enum SavedArtifactLocations {
    static func dataset(_ saved: JSONValue, checkpointManifest: JSONValue, layout: ArtifactStorageLayout) throws -> JSONValue {
        guard saved.fields?["kind"] == .string("recordings") else { return saved }
        guard var fields = saved.fields, let path = fields["path"]?.text,
              let id = UUID(uuidString: URL(fileURLWithPath: path).lastPathComponent) else {
            throw AstraError("dataset.location", "The saved dataset has no stable artifact identity.")
        }
        if let expected = checkpointManifest.fields?["datasetID"], expected != .null {
            guard expected.uuid == id else { throw AstraError("dataset.identity", "The saved configuration and checkpoint reference different datasets.") }
        }
        fields["path"] = .string(layout.catalogRoot.appendingPathComponent("Datasets/\(id.uuidString.lowercased())").path)
        fields["recordingRoot"] = .string(layout.recordingsRoot.path)
        return .object(fields)
    }
}
