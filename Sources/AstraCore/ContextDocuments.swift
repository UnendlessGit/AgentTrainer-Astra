import Foundation

/// IDs carry meaning. Names are presentation only; a renamed value keeps the
/// same model index in every previously frozen vocabulary.
public struct ContextValue: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var name: String
    public init(id: UUID = UUID(), name: String) { self.id = id; self.name = name }
}

public struct ContextFieldDocument: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var name: String
    public var values: [ContextValue]
    public init(id: UUID = UUID(), name: String, values: [ContextValue] = []) {
        self.id = id; self.name = name; self.values = values
    }
    public func validated() throws -> Self {
        var result = self; result.name = try DocumentNames.validated(name)
        guard values.count <= 255, Set(values.map(\.id)).count == values.count,
              !values.contains(where: { $0.id == id }) else {
            throw AstraError("context.values", "Use at most 255 distinct values in a context field.")
        }
        result.values = try values.map { .init(id: $0.id, name: try DocumentNames.validated($0.name)) }
        guard Set(result.values.map { $0.name.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current) }).count == values.count else {
            throw AstraError("context.names", "Use distinct names for the values in a context field.")
        }
        return result
    }
}

/// A value snapshot belongs to one model/dataset. Index zero is always unknown;
/// authored values retain this snapshot's order even after library edits.
public struct ContextVocabulary: Codable, Hashable, Sendable {
    public var fields: [ContextFieldDocument]
    public init(fields: [ContextFieldDocument] = []) { self.fields = fields }
    public static let empty = ContextVocabulary()
    public var sizes: [Int] { fields.map { $0.values.count + 1 } }
    public func validated() throws -> Self {
        guard fields.count <= 32, Set(fields.map(\.id)).count == fields.count else {
            throw AstraError("context.fields", "Choose up to 32 distinct context fields.")
        }
        return .init(fields: try fields.map { try $0.validated() })
    }
    public func indices(for assignments: [UUID: UUID]) throws -> [Int] {
        _ = try validated()
        return try fields.map { field in
            guard let value = assignments[field.id] else { return 0 }
            guard let index = field.values.firstIndex(where: { $0.id == value }) else {
                throw AstraError("context.valueMissing", "The selected value for “\(field.name)” is not in this model’s vocabulary. Choose an available value or Unknown.")
            }
            return index + 1
        }
    }
    public var payload: JSONValue {
        .array(fields.map { field in
            .object(["id": .string(field.id.uuidString.lowercased()), "name": .string(field.name),
                "values": .array(field.values.map { .object(["id": .string($0.id.uuidString.lowercased()), "name": .string($0.name)]) })])
        })
    }
    public func applying(to model: JSONValue) -> JSONValue {
        guard var fields = model.fields else { return model }
        fields["context_sizes"] = .array(sizes.map { .integer(Int64($0)) })
        if !self.fields.isEmpty { fields["context_vocabulary"] = payload }
        else { fields["context_vocabulary"] = nil }
        return .object(fields)
    }
    public static func from(model: JSONValue) throws -> Self? {
        guard let value = model.fields?["context_vocabulary"] else { return nil }
        let vocabulary = try Self(fields: value.decode([ContextFieldDocument].self)).validated()
        guard try model.required("context_sizes").decode([Int].self) == vocabulary.sizes else {
            throw AstraError("context.configuration", "The checkpoint’s context names and embedding dimensions disagree.")
        }
        return vocabulary
    }
}
