import Foundation
import Testing
@testable import AstraCore

private let rewardTick: UInt64 = 100_000_000

@Test func scoreChangeBoundsDoNotCorruptOtherRulesSharingTheSameSignal() throws {
    let score = RewardSignal(name: "Score", kind: .manual)
    var strict = RewardRule(name: "Small increments", kind: .scoreDelta, amount: 1, signalID: score.id)
    strict.maximumDelta = 5
    let broad = RewardRule(name: "Large increments", kind: .scoreDelta, amount: 1, signalID: score.id)
    var evaluator = try RewardEvaluator(program: .init(name: "Independent bounds", signals: [score], rules: [strict, broad]))
    let episode = UUID()
    func value(_ time: UInt64, _ number: Double) -> [SignalReading] {
        [.init(signalID: score.id, episodeID: episode, eventNanos: time, observedNanos: time, value: .number(number))]
    }
    try evaluator.reset(episodeID: episode, readyAtNanos: 0, readings: value(0, 0), controlsReleased: true)
    let jump = try evaluator.evaluate(endNanos: rewardTick, readings: value(rewardTick, 10))
    #expect(jump.value == nil && jump.components[broad.id] == 10 && jump.unknownRules == [strict.id])
    let next = try evaluator.evaluate(endNanos: rewardTick * 2, readings: value(rewardTick * 2, 11))
    #expect(next.value == nil && next.components[broad.id] == 1)
    #expect(try evaluator.evaluate(endNanos: rewardTick * 3, readings: value(rewardTick * 3, 12)).value == 2)
}

@Test func rewardDefinitionsRemainImmutableAndSharedAcrossAgentCopies() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), agent = AgentDocument(name: "Reward author")
    try await store.save(agent)
    let program = RewardProgram(name: "Manual", rules: [.init(name: "Good", kind: .manualMarker, amount: 1)])
    try await store.saveRewardProgram(program, for: agent.id)
    var snapshot = try await store.snapshot()
    #expect(snapshot.rewardPrograms == [program] && snapshot.agents.first?.rewardProgramID == program.id)
    let copy = try await store.duplicateAgent(#require(snapshot.agents.first))
    #expect(copy.rewardProgramID == program.id)
    var changed = program; changed.rules[0].amount = 2
    await #expect(throws: AstraError.self) { try await store.saveRewardProgram(changed, for: agent.id) }
    changed.id = UUID(); try await store.saveRewardProgram(changed, for: agent.id)
    snapshot = try await store.snapshot()
    #expect(snapshot.rewardPrograms.count == 2)
    #expect(snapshot.agents.first(where: { $0.id == copy.id })?.rewardProgramID == program.id)
    #expect(snapshot.agents.first(where: { $0.id == agent.id })?.rewardProgramID == changed.id)
}

@Test func rewardTemplateIntegrityIsCheckedBeforeCatalogPublication() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), agent = AgentDocument(name: "Visual rewards")
    try await store.save(agent)
    let bytes = Data("Content-addressed image fixture".utf8)
    let digest = try RewardAssets.save(bytes, root: root)
    #expect(try RewardAssets.read(digest, root: root) == bytes)
    var signal = RewardSignal(name: "Image", kind: .imageMatch, surfaceID: "fixture", region: .init(x: 0, y: 0, width: 1, height: 1))
    signal.templateDigest = digest
    let program = RewardProgram(name: "Visual", signals: [signal])
    try Data("Changed".utf8).write(to: root.appendingPathComponent("RewardTemplates/\(digest).image"))
    await #expect(throws: AstraError.self) { try await store.saveRewardProgram(program, for: agent.id) }
    #expect(try await store.snapshot().rewardPrograms.isEmpty)
}

@Test func rewardMissingValuesBreakDeltaAndEdgeBaselinesWithoutPhantomChanges() throws {
    let score = RewardSignal(name: "Score", kind: .manual)
    let won = RewardSignal(name: "Won", kind: .manual)
    let delta = RewardRule(name: "Score change", kind: .scoreDelta, amount: 0.5, signalID: score.id)
    let edge = RewardRule(name: "Victory", kind: .risingEdge, amount: 10,
        predicate: .init(conditions: [.init(signalID: won.id, comparison: .isTrue)]))
    var evaluator = try RewardEvaluator(program: .init(name: "Rewards", signals: [score, won], rules: [delta, edge]))
    let episode = UUID()
    func values(_ at: UInt64, _ number: Double, _ flag: Bool) -> [SignalReading] {
        [.init(signalID: score.id, episodeID: episode, eventNanos: at, observedNanos: at, value: .number(number)),
         .init(signalID: won.id, episodeID: episode, eventNanos: at, observedNanos: at, value: .flag(flag))]
    }
    try evaluator.reset(episodeID: episode, readyAtNanos: 0, readings: values(0, 100, false), controlsReleased: true)
    #expect(try evaluator.evaluate(endNanos: rewardTick, readings: values(rewardTick, 102, true)).value == 11)
    #expect(try evaluator.evaluate(endNanos: rewardTick * 2, readings: []).value == nil)
    #expect(try evaluator.evaluate(endNanos: rewardTick * 3, readings: values(rewardTick * 3, 999, true)).value == nil)
    #expect(try evaluator.evaluate(endNanos: rewardTick * 4, readings: values(rewardTick * 4, 1001, true)).value == 1)
    try evaluator.reset(episodeID: episode, readyAtNanos: rewardTick * 5, readings: values(rewardTick * 5, 0, true), controlsReleased: true)
    #expect(try evaluator.evaluate(endNanos: rewardTick * 6, readings: values(rewardTick * 6, 0, true)).value == 0)
}

@Test func rewardMarkersUseDecisionWindowsAndDoNotMutateStateOnRejection() throws {
    let rule = RewardRule(name: "Good action", kind: .manualMarker, amount: 1)
    var evaluator = try RewardEvaluator(program: .init(name: "Manual", rules: [rule]))
    let episode = UUID()
    try evaluator.reset(episodeID: episode, readyAtNanos: 100, readings: [], controlsReleased: true)
    func marker(_ sequence: UInt64, _ event: UInt64, _ observed: UInt64) -> ManualRewardMarker {
        .init(sequence: sequence, episodeID: episode, ruleID: rule.id, eventNanos: event, observedNanos: observed)
    }
    func coverage(_ start: UInt64, _ end: UInt64, _ sequence: UInt64?) -> ManualRewardCoverage {
        .init(episodeID: episode, startNanos: start, endNanos: end, lastSequence: sequence)
    }
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: 200, readings: [], markers: [marker(0, 200, 200)]) }
    #expect(try evaluator.evaluate(endNanos: 200, readings: [], markers: [marker(0, 100, 150)], markerCoverage: coverage(100, 200, 0)).value == 1)
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: 300, readings: [], markers: [marker(0, 210, 220)]) }
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: 300, readings: [], markers: [marker(1, 190, 220)]) }
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: 300, readings: [], markers: [marker(1, 210, 301)]) }
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: 300, readings: [], markers: [marker(2, 210, 220)]) }
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: 300, readings: [], markerCoverage: coverage(200, 300, 1)) }
    #expect(try evaluator.evaluate(endNanos: 300, readings: [], markers: [marker(1, 200, 220)], markerCoverage: coverage(200, 300, 1)).value == 1)
    #expect(try evaluator.evaluate(endNanos: 400, readings: []).value == nil)
    #expect(try evaluator.evaluate(endNanos: 500, readings: [], markerCoverage: coverage(400, 500, 1)).value == 0)
}

@Test func rewardReadinessUnknownConfidenceAndTerminalConflictAreExplicit() throws {
    let signal = RewardSignal(name: "State", kind: .manual)
    let predicate = RewardPredicate(conditions: [.init(signalID: signal.id, comparison: .isTrue)])
    var program = RewardProgram(name: "Episode", signals: [signal]); program.ready = predicate
    var evaluator = try RewardEvaluator(program: program)
    let episode = UUID()
    let reading = SignalReading(signalID: signal.id, episodeID: episode, eventNanos: 50, observedNanos: 75, value: .flag(true))
    #expect(try evaluator.readiness(episodeID: episode, cutoffNanos: 100, readings: []) == .unknown)
    #expect(throws: AstraError.self) { try evaluator.reset(episodeID: episode, readyAtNanos: 100, readings: [reading], controlsReleased: false) }
    try evaluator.reset(episodeID: episode, readyAtNanos: 100, readings: [reading], controlsReleased: true)
    program.ready = nil; program.success = predicate; program.failure = predicate
    evaluator = try RewardEvaluator(program: program)
    try evaluator.reset(episodeID: episode, readyAtNanos: 0, readings: [], controlsReleased: true)
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: 100, readings: [reading]) }
    #expect(try evaluator.evaluate(endNanos: 100, readings: []).outcome == .unknown)
    let low = SignalReading(signalID: signal.id, episodeID: episode, eventNanos: 101, observedNanos: 101, confidence: 0.1, value: .flag(true))
    #expect(try evaluator.evaluate(endNanos: 200, readings: [low]).outcome == .unknown)
}

@Test func rewardRatesUseActualDurationAndLeftEndpointConditions() throws {
    let flag = RewardSignal(name: "Active", kind: .manual)
    let rate = RewardRule(name: "Time cost", kind: .ratePerSecond, amount: -2,
        predicate: .init(conditions: [.init(signalID: flag.id, comparison: .isTrue)]))
    var program = RewardProgram(name: "Timing", signals: [flag], rules: [rate]); program.maximumEpisodeMS = 1000
    var evaluator = try RewardEvaluator(program: program)
    let episode = UUID()
    func reading(_ time: UInt64, _ active: Bool) -> [SignalReading] {
        [.init(signalID: flag.id, episodeID: episode, eventNanos: time, observedNanos: time, value: .flag(active))]
    }
    try evaluator.reset(episodeID: episode, readyAtNanos: 0, readings: reading(0, true), controlsReleased: true)
    #expect(try evaluator.evaluate(endNanos: rewardTick, readings: reading(rewardTick, false)).value == -0.2)
    #expect(try evaluator.evaluate(endNanos: rewardTick * 3, readings: reading(rewardTick * 3, true)).value == 0)
    let end = try evaluator.evaluate(endNanos: rewardTick * 10, readings: reading(rewardTick * 10, true))
    #expect(end.value == -1.4 && end.outcome == .truncated)
    #expect(throws: AstraError.self) { try evaluator.evaluate(endNanos: rewardTick * 11, readings: []) }
}

@Test func rewardPredicateUsesThreeValuedLogicAndRejectsInvalidDefinitions() throws {
    let one = RewardSignal(name: "One", kind: .manual), two = RewardSignal(name: "Two", kind: .manual)
    let conditions: [RewardCondition] = [.init(signalID: one.id, comparison: .isTrue), .init(signalID: two.id, comparison: .isTrue)]
    #expect(RewardEvaluator.test(.init(logic: .all, conditions: conditions), values: [one.id: .flag(false)]) == .no)
    #expect(RewardEvaluator.test(.init(logic: .any, conditions: conditions), values: [one.id: .flag(true)]) == .yes)
    #expect(RewardEvaluator.test(.init(logic: .all, conditions: conditions), values: [one.id: .flag(true)]) == .unknown)
    var program = RewardProgram(name: "Invalid", signals: [one, one])
    #expect(throws: AstraError.self) { try program.validated() }
    program.signals = [one]; program.success = .init(conditions: conditions)
    #expect(throws: AstraError.self) { try program.validated() }
    var visual = RewardSignal(name: "Screen score", kind: .ocrNumber, surfaceID: "display:1", region: .init(x: 0.9, y: 0, width: 0.2, height: 1))
    #expect(throws: AstraError.self) { try visual.validated() }
    visual.region = .init(x: 0, y: 0, width: 1, height: 1)
    #expect(try visual.validated() == visual)
}
