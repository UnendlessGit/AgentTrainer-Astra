import Foundation
import CryptoKit

enum ArtifactArchive {
    static let filename = "archive.json"
    static let metadataLimit = 32 * 1024 * 1024
    static func encoded(_ manifest: ArtifactArchiveManifest) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .millisecondsSince1970
        // Swift UUID-keyed dictionaries encode as alternating array pairs.
        // Sort those pairs and normalize catalog-only dates to milliseconds so
        // a decoded recovery plan reproduces exactly the same archive header.
        func canonical(_ value: Any, key: String? = nil) -> Any {
            if let object = value as? [String: Any] {
                return object.reduce(into: [String: Any]()) { result, pair in
                    if ["createdAt", "modifiedAt", "finishedAt"].contains(pair.key), let number = pair.value as? NSNumber {
                        result[pair.key] = number.doubleValue.rounded()
                    } else { result[pair.key] = canonical(pair.value, key: pair.key) }
                }
            }
            if let values = value as? [Any] {
                if ["recordingSelections", "contextValues"].contains(key ?? ""), values.count.isMultiple(of: 2),
                   stride(from: 0, to: values.count, by: 2).allSatisfy({ (values[$0] as? String).flatMap(UUID.init(uuidString:)) != nil }) {
                    return stride(from: 0, to: values.count, by: 2).map { (values[$0] as! String, canonical(values[$0 + 1])) }
                        .sorted { $0.0 < $1.0 }.flatMap { [$0.0 as Any, $0.1] }
                }
                return values.map { canonical($0) }
            }
            return value
        }
        let bytes = try JSONSerialization.data(withJSONObject: canonical(JSONSerialization.jsonObject(with: encoder.encode(manifest))), options: [.sortedKeys, .withoutEscapingSlashes])
        guard bytes.count <= metadataLimit else { throw error("The archive metadata exceeds its size limit.") }
        return bytes
    }
    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func read(_ directory: URL, cancelled: @Sendable () -> Bool = { false }) throws -> (ArtifactArchiveManifest, String) {
        try ArtifactTransferFiles.requireDirectory(directory)
        let bytes = try ArtifactTransferFiles.read(directory.appendingPathComponent(filename), limit: metadataLimit)
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
        let manifest = try decoder.decode(ArtifactArchiveManifest.self, from: bytes)
        try validate(manifest)
        var expected = [ArtifactFileEntry(path: filename, byteCount: UInt64(bytes.count), sha256: digest(bytes))]
        for item in manifest.items {
            expected += item.files.map { .init(path: item.relativePath + "/" + $0.path, byteCount: $0.byteCount, sha256: $0.sha256) }
        }
        guard try ArtifactTransferFiles.inventory(directory, cancelled: cancelled) == expected.sorted(by: { $0.path < $1.path }) else {
            throw error("The archive contains changed, missing or unlisted files.")
        }
        for item in manifest.items {
            try ArtifactTransferFiles.check(cancelled)
            let source = directory.appendingPathComponent(item.relativePath, isDirectory: true)
            try validateContent(item, directory: source, catalog: manifest.catalog, items: manifest.items)
        }
        return (manifest, digest(bytes))
    }
    static func validate(_ manifest: ArtifactArchiveManifest) throws {
        guard manifest.schemaVersion == 1, manifest.createdAt.timeIntervalSince1970.isFinite,
              (1...16_384).contains(manifest.items.count), Set(manifest.items.map(\.id)).count == manifest.items.count,
              manifest.notices.count <= 1024, manifest.notices.allSatisfy({ $0.utf8.count <= 8192 }) else {
            throw error("The archive has unsupported, duplicate or oversized metadata.")
        }
        var count = 0, bytes: UInt64 = 0
        for item in manifest.items {
            guard item.relativePath == "payload/\(item.kind.rawValue)/\(item.identity)",
                  item.kind == .rewardAsset ? isDigest(item.identity) : UUID(uuidString: item.identity)?.uuidString.lowercased() == item.identity,
                  (try? DocumentNames.validated(item.name)) != nil, !item.files.isEmpty,
                  item.files == item.files.sorted(by: { $0.path < $1.path }),
                  Set(item.files.map({ $0.path.precomposedStringWithCanonicalMapping.lowercased() })).count == item.files.count else {
                throw error("An archive item has an invalid identity, path or file list.")
            }
            for file in item.files {
                let sum = bytes.addingReportingOverflow(file.byteCount)
                guard ArtifactTransferFiles.validPath(file.path), isDigest(file.sha256), !sum.overflow,
                      sum.partialValue <= ArtifactTransferFiles.maximumBytes else { throw error("An archive file exceeds its path or resource limits.") }
                bytes = sum.partialValue; count += 1
                guard count <= ArtifactTransferFiles.maximumFiles else { throw error("The archive contains too many files.") }
            }
        }
        let catalog = manifest.catalog
        try unique(catalog.agents); try unique(catalog.environments); try unique(catalog.contexts); try unique(catalog.rewards)
        try unique(catalog.recordings); try unique(catalog.checkpoints); try unique(catalog.runs)
        for value in catalog.agents { _ = try value.validated() }
        for value in catalog.environments { _ = try value.validated() }
        for value in catalog.contexts { _ = try value.validated() }
        for value in catalog.rewards { _ = try value.validated() }
        for value in catalog.recordings { _ = try value.validated(); guard value.status != .recording else { throw error("Live recordings cannot be imported.") } }
        for value in catalog.checkpoints { _ = try value.validated() }
        for value in catalog.runs {
            _ = try value.validated()
            guard ![.preparing, .running, .cancelling].contains(value.status) else { throw error("Live learning runs cannot be imported.") }
        }
        let agents = Set(catalog.agents.map(\.id)), recordings = Set(catalog.recordings.map(\.id)), checkpoints = Set(catalog.checkpoints.map(\.id))
        let contexts = Set(catalog.contexts.map(\.id)), rewards = Set(catalog.rewards.map(\.id))
        let runConfigurations = Set(manifest.items.filter { $0.kind == .runConfiguration }.compactMap { UUID(uuidString: $0.identity) })
        let templates = Set(manifest.items.filter { $0.kind == .rewardAsset }.map(\.identity))
        guard Set(catalog.links.map(\.agentID)).count == catalog.links.count,
              catalog.links.allSatisfy({ agents.contains($0.agentID) && Set($0.recordingSelections.keys).isSubset(of: recordings)
                  && Set($0.checkpointIDs).isSubset(of: checkpoints) && Set($0.checkpointIDs).count == $0.checkpointIDs.count }),
              catalog.checkpoints.allSatisfy({ agents.contains($0.agentID) }), catalog.runs.allSatisfy({ agents.contains($0.agentID) }),
              catalog.agents.allSatisfy({ Set($0.contextFieldIDs ?? []).isSubset(of: contexts)
                  && ($0.rewardProgramID.map(rewards.contains) ?? true) && ($0.selectedCheckpointID.map(checkpoints.contains) ?? true) }),
              catalog.checkpoints.allSatisfy({ $0.kind == "initial" || ($0.runID.map(runConfigurations.contains) ?? false) }),
              Set(catalog.rewards.flatMap { $0.signals.compactMap(\.templateDigest) }).isSubset(of: templates),
              Set(manifest.items.filter { $0.kind == .recording }.compactMap { UUID(uuidString: $0.identity) }) == recordings,
              Set(manifest.items.filter { $0.kind == .checkpoint }.compactMap { UUID(uuidString: $0.identity) }) == checkpoints else {
            throw error("The archive catalog links do not match its included immutable artifacts.")
        }
        for link in catalog.links { for selection in link.recordingSelections.values { _ = try selection.validated() } }
    }
    static func validateContent(_ item: ArtifactTransferItem, directory: URL, catalog: ArtifactCatalogBundle, items: [ArtifactTransferItem]) throws {
        switch item.kind {
        case .recording:
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
            let manifest = try decoder.decode(RecordingManifest.self, from: ArtifactTransferFiles.read(directory.appendingPathComponent("manifest.json"), limit: 262_144)).validated()
            guard let expected = catalog.recordings.first(where: { $0.id.uuidString.lowercased() == item.identity }), try sameMetadata(manifest, expected) else {
                throw error("A recording manifest disagrees with its archive catalog identity.")
            }
            _ = try manifest.indexURL(in: directory)
            for file in item.files where file.path.hasSuffix("-wal") && file.byteCount > 0 { throw error("A recording still has an unpublished WAL; recover and seal it before export.") }
        case .checkpoint:
            let manifest = try json(directory.appendingPathComponent("manifest.json"))
            guard let checkpoint = catalog.checkpoints.first(where: { $0.id.uuidString.lowercased() == item.identity }),
                  manifest.fields?["id"]?.uuid == checkpoint.id, manifest.fields?["policySignature"]?.text == checkpoint.policySignature,
                  manifest.fields?["kind"]?.text == checkpoint.kind, manifest.fields?["step"]?.int == checkpoint.trainingStep,
                  let artifacts = manifest.fields?["artifacts"]?.fields, artifacts["policy.safetensors"] != nil,
                  Set(artifacts.keys).isSubset(of: ["policy.safetensors", "training.safetensors", "training.json"]),
                  Set(item.files.map(\.path)) == Set(artifacts.keys).union(["manifest.json"]) else {
                throw error("A checkpoint disagrees with its immutable artifact manifest.")
            }
            for (name, spec) in artifacts {
                guard let file = item.files.first(where: { $0.path == name }), spec.fields?["bytes"]?.uint64 == file.byteCount,
                      spec.fields?["sha256"]?.text == file.sha256 else { throw error("A checkpoint tensor/state checksum is invalid.") }
            }
            if checkpoint.kind == "behavioral", let datasetID = manifest.fields?["datasetID"]?.uuid,
               let run = catalog.runs.first(where: { $0.id == checkpoint.runID }), run.sourceKind == "recordings" {
                guard items.contains(where: { $0.kind == .dataset && UUID(uuidString: $0.identity) == datasetID }) else {
                    throw error("The behavioral checkpoint is missing its exact dataset dependency.")
                }
            }
        case .dataset:
            let manifest = try json(directory.appendingPathComponent("manifest.json"))
            guard manifest.fields?["id"]?.uuid == UUID(uuidString: item.identity),
                  let index = item.files.first(where: { $0.path == "index.sqlite" }),
                  manifest.fields?["indexSHA256"]?.text == index.sha256,
                  let sources = try? manifest.required("sources").decode([JSONValue].self),
                  sources.allSatisfy({ source in catalog.recordings.contains { $0.id == source.fields?["id"]?.uuid } }) else {
                throw error("The dataset has no matching index or complete recording dependency set.")
            }
        case .runConfiguration, .desktopConfiguration:
            let configuration = try json(directory.appendingPathComponent("configuration.json"))
            guard item.files.map(\.path) == ["configuration.json"], configuration.fields?["runID"]?.uuid == UUID(uuidString: item.identity) else {
                throw error("A saved run configuration has a different identity.")
            }
            if item.kind == .desktopConfiguration {
                let binding = try configuration.required("rewardBinding").decode(RewardProgramBinding.self)
                _ = try binding.validated(scope: configuration.required("scope").decode(ControlScope.self))
                guard catalog.rewards.contains(binding.definition) else { throw error("The archive is missing a desktop checkpoint's frozen reward definition.") }
            } else if let actorID = configuration.fields?["actorRunID"]?.uuid {
                guard items.contains(where: { $0.kind == .desktopConfiguration && UUID(uuidString: $0.identity) == actorID }) else {
                    throw error("The desktop checkpoint is missing its original task configuration.")
                }
            }
        case .rewardAsset:
            guard item.files.count == 1, item.files[0].path == item.identity + ".image", item.files[0].sha256 == item.identity else {
                throw error("A reward template's content address is invalid.")
            }
        }
    }
    static func json(_ path: URL) throws -> JSONValue { try JSONDecoder().decode(JSONValue.self, from: ArtifactTransferFiles.read(path, limit: 8 * 1024 * 1024)) }
    /// Date serialization precision is presentation, not source identity. Raw
    /// artifact bytes are checked separately and are never normalized/copied anew.
    static func sameMetadata<T: Encodable>(_ lhs: T, _ rhs: T) throws -> Bool {
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .millisecondsSince1970
        func normalized(_ value: JSONValue) -> JSONValue {
            switch value {
            case .object(let fields): return .object(fields.filter { !["createdAt", "modifiedAt", "finishedAt"].contains($0.key) }.mapValues(normalized))
            case .array(let values): return .array(values.map(normalized))
            default: return value
            }
        }
        return try normalized(JSONDecoder().decode(JSONValue.self, from: encoder.encode(lhs)))
            == normalized(JSONDecoder().decode(JSONValue.self, from: encoder.encode(rhs)))
    }
    static func isDigest(_ value: String) -> Bool { value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }
    static func unique<T: Identifiable>(_ values: [T]) throws where T.ID == UUID {
        guard values.count <= 16_384, Set(values.map(\.id)).count == values.count else { throw error("The archive catalog contains too many or duplicate identities.") }
    }
    static func error(_ message: String) -> AstraError { .init("artifact.archive", message) }
}
