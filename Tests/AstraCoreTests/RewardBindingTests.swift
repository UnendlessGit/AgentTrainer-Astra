import Foundation
import Testing
@testable import AstraCore

private func boundRewardFixture() -> RewardProgram {
    let signal = RewardSignal(name: "Score", kind: .ocrNumber, surfaceID: "window:10", region: .init(x: 0.1, y: 0.1, width: 0.2, height: 0.1))
    var definition = RewardProgram(name: "Target task", signals: [signal],
        rules: [.init(name: "Score change", kind: .scoreDelta, amount: 1, signalID: signal.id)])
    definition.ready = .init(conditions: [.init(signalID: signal.id, comparison: .atMost, number: 0)])
    definition.resetPlan = ResetPlan(steps: [.init(name: "Select restart", packet: .init(durationMS: 100,
        commands: [.init(offsetMs: 0, operation: .pointerAbsolute, surfaceID: "window:10", x: 0.5, y: 0.7)]))])
    return definition
}
private func liveRewardSurface(_ id: String) -> SurfaceDescriptor {
    .init(id: id, globalBounds: .init(x: 30, y: 40, width: 800, height: 600), pixelWidth: 1600, pixelHeight: 1200)
}

@Test func rewardBindingUpdatesOnlyRuntimeSurfaceReferencesAndKeepsTaskSignatures() throws {
    let definition = boundRewardFixture(), first = liveRewardSurface("window:21"), second = liveRewardSurface("window:89")
    let firstScope = ControlScope(surfaces: [first]), secondScope = ControlScope(surfaces: [second])
    let one = try RewardProgramBinding.singleSource(definition, surface: first, scope: firstScope)
    let two = try RewardProgramBinding.singleSource(definition, surface: second, scope: secondScope)
    let resolved = try one.resolved(scope: firstScope)
    #expect(one.definition == definition && definition.signals[0].surfaceID == "window:10")
    #expect(resolved.signals[0].surfaceID == "window:21")
    #expect(resolved.resetPlan?.steps[0].packet?.commands[0].surfaceID == "window:21")
    #expect(resolved.signals[0].id == definition.signals[0].id && resolved.rules == definition.rules)
    #expect(one.definitionSignature == two.definitionSignature && one.resetSignature == two.resetSignature)
    let decoded = try JSONDecoder().decode(RewardProgramBinding.self, from: JSONEncoder().encode(one))
    #expect(try decoded.resolved(scope: firstScope) == resolved)
}

@Test func rewardBindingRefusesAmbiguousMissingAndChangedSources() throws {
    var definition = boundRewardFixture()
    let actual = liveRewardSurface("window:21"), scope = ControlScope(surfaces: [actual])
    #expect(throws: AstraError.self) { try RewardProgramBinding(definition: definition, surfaces: [:], scope: scope) }
    #expect(throws: AstraError.self) {
        try RewardProgramBinding(definition: definition, surfaces: ["window:10": liveRewardSurface("window:99")], scope: scope)
    }
    let binding = try RewardProgramBinding.singleSource(definition, surface: actual, scope: scope)
    var moved = actual; moved.globalBounds.x += 1; moved.geometryRevision = 1
    #expect(throws: AstraError.self) { try binding.resolved(scope: .init(surfaces: [moved], geometryRevision: 1)) }
    definition.signals.append(.init(name: "Second source", kind: .ocrText, surfaceID: "display:4", region: .init(x: 0, y: 0, width: 1, height: 1)))
    #expect(throws: AstraError.self) { try RewardProgramBinding.singleSource(definition, surface: actual, scope: scope) }
}

@Test func rewardBindingRejectsChangedSerializedDefinitionFingerprint() throws {
    let surface = liveRewardSurface("window:21"), scope = ControlScope(surfaces: [surface])
    let binding = try RewardProgramBinding.singleSource(boundRewardFixture(), surface: surface, scope: scope)
    var value = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(binding)) as? [String: Any])
    value["definitionSignature"] = String(repeating: "a", count: 64)
    let changed = try JSONDecoder().decode(RewardProgramBinding.self, from: JSONSerialization.data(withJSONObject: value))
    #expect(throws: AstraError.self) { try changed.resolved(scope: scope) }
}
