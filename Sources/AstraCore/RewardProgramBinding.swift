import Foundation
import CryptoKit

/// A saved definition names reference surfaces. A run explicitly binds those
/// references to its currently verified surfaces without editing the library
/// definition or making its fingerprint depend on transient window IDs.
public struct RewardProgramBinding: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let definition: RewardProgram
    public let surfaces: [String: SurfaceDescriptor]
    public let definitionSignature: String
    public let resetSignature: String

    public init(definition: RewardProgram, surfaces: [String: SurfaceDescriptor], scope: ControlScope) throws {
        let definition = try definition.validated()
        _ = try scope.validated()
        guard surfaces.count <= 16, Set(surfaces.keys) == Self.referencedSurfaces(in: definition),
              surfaces.values.allSatisfy(scope.surfaces.contains) else {
            throw AstraError("reward.binding", "Bind every reference surface to a verified surface in this environment.")
        }
        schemaVersion = 1; self.definition = definition; self.surfaces = surfaces
        definitionSignature = try Self.fingerprint(definition)
        // Manual readiness has its own stable identity; absence is not an
        // automatic reset and must not collide with an authored empty plan.
        resetSignature = try Self.fingerprint(definition.resetPlan.map { try JSONValue.encode($0) }
            ?? .object(["mode": .string("manualReady"), "schemaVersion": .integer(1)]))
    }

    public static func referencedSurfaces(in definition: RewardProgram) -> Set<String> {
        var result = Set(definition.signals.compactMap(\.surfaceID))
        for step in definition.resetPlan?.steps ?? [] {
            result.formUnion(step.packet?.commands.compactMap(\.surfaceID) ?? [])
        }
        return result
    }

    /// The single-source UI can resolve one reference without guessing among
    /// multiple recorded surfaces. Multi-source callers supply an explicit map.
    public static func singleSource(_ definition: RewardProgram, surface: SurfaceDescriptor, scope: ControlScope) throws -> Self {
        let names = referencedSurfaces(in: definition)
        guard names.count <= 1, scope.surfaces.count == 1, scope.surfaces.first == surface else {
            throw AstraError("reward.multipleBindings", "This definition uses several reference surfaces. Bind each surface explicitly before starting.")
        }
        return try Self(definition: definition, surfaces: Dictionary(uniqueKeysWithValues: names.map { ($0, surface) }), scope: scope)
    }

    /// Resolve an operator-reviewed reference map against this run's verified
    /// surfaces. Native IDs are never matched by title, location or array order.
    public static func bound(_ definition: RewardProgram, sourceIDs: [String: String], scope: ControlScope) throws -> Self {
        let references = referencedSurfaces(in: definition)
        if sourceIDs.isEmpty, references.count <= 1, scope.surfaces.count == 1, let surface = scope.surfaces.first {
            return try singleSource(definition, surface: surface, scope: scope)
        }
        guard Set(sourceIDs.keys) == references else {
            throw AstraError("reward.binding", "Choose a current capture surface for every reward and reset source before starting.")
        }
        var resolved: [String: SurfaceDescriptor] = [:]
        for (reference, id) in sourceIDs {
            guard let surface = scope.surfaces.first(where: { $0.id == id }) else {
                throw AstraError("reward.bindingMissing", "A chosen reward or reset surface is no longer part of this environment. Choose its current source again.")
            }
            resolved[reference] = surface
        }
        return try Self(definition: definition, surfaces: resolved, scope: scope)
    }

    public func validated(scope: ControlScope) throws -> Self {
        let checked = try Self(definition: definition, surfaces: surfaces, scope: scope)
        guard schemaVersion == 1, definitionSignature == checked.definitionSignature, resetSignature == checked.resetSignature else {
            throw AstraError("reward.bindingIntegrity", "The saved reward binding does not match its immutable definition.")
        }
        return checked
    }

    public func resolved(scope: ControlScope) throws -> RewardProgram {
        _ = try validated(scope: scope)
        var result = definition
        for index in result.signals.indices {
            if let source = result.signals[index].surfaceID { result.signals[index].surfaceID = surfaces[source]?.id }
        }
        if var plan = result.resetPlan {
            for index in plan.steps.indices {
                guard var packet = plan.steps[index].packet else { continue }
                for command in packet.commands.indices {
                    if let source = packet.commands[command].surfaceID { packet.commands[command].surfaceID = surfaces[source]?.id }
                }
                plan.steps[index].packet = packet
            }
            result.resetPlan = plan
        }
        return try result.validated()
    }

    private static func fingerprint<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try SHA256.hash(data: encoder.encode(value)).map { String(format: "%02x", $0) }.joined()
    }
}
