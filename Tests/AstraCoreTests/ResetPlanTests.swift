import Foundation
import Testing
@testable import AstraCore

@Test func resetPlanIsOptionalAndValidatesEveryAuthoredStepBeforeExecution() throws {
    let original = RewardProgram(name: "Manual default")
    let data = try JSONEncoder().encode(original)
    #expect(try JSONDecoder().decode(RewardProgram.self, from: data).validated().resetPlan == nil)
    let signal = RewardSignal(name: "Ready", kind: .manual)
    var program = RewardProgram(name: "Authored", signals: [signal])
    program.resetPlan = .init(steps: [.init(name: "Return", packet: .init(durationMS: 100,
        commands: [.init(offsetMs: 0, operation: .keyDown, keyCode: 36), .init(offsetMs: 80, operation: .keyUp, keyCode: 36)]))])
    #expect(throws: AstraError.self) { _ = try program.validated() }
    program.ready = .init(conditions: [.init(signalID: signal.id, comparison: .isTrue)])
    #expect(try program.validated().resetPlan?.capabilities.keyCodes == [36])
    program.resetPlan?.steps.append(.init(pauseMS: 100))
    program.resetPlan?.steps.append(.init(condition: program.ready!, timeoutMS: 1000))
    let encoded = try JSONEncoder().encode(program)
    #expect(try JSONDecoder().decode(RewardProgram.self, from: encoded).validated() == program)
    program.resetPlan?.steps[0].packet?.commands[0].keyCode = 57
    #expect(throws: AstraError.self) { _ = try program.validated() }
}

@Test func resetPlanRejectsAmbiguousStepsUnboundedRetriesAndEpisodeTimeWaits() throws {
    let elapsed = RewardSignal(name: "Episode time", kind: .elapsedSeconds)
    var plan = ResetPlan(steps: [.init(condition: .init(conditions: [.init(signalID: elapsed.id, comparison: .atLeast, number: 1)]))])
    #expect(throws: AstraError.self) { _ = try plan.validated(signals: [elapsed.id: elapsed]) }
    plan = .init(steps: [.init(pauseMS: 100)])
    plan.steps[0].timeoutMS = 100
    #expect(throws: AstraError.self) { _ = try plan.validated(signals: [:]) }
    plan.steps[0].timeoutMS = nil; plan.maximumAttempts = 4
    #expect(throws: AstraError.self) { _ = try plan.validated(signals: [:]) }
    plan.maximumAttempts = 1; plan.steps.append(plan.steps[0])
    #expect(throws: AstraError.self) { _ = try plan.validated(signals: [:]) }
    let fraction = ResetPacketTemplate(durationMS: 100, commands: [.init(offsetMs: 0, operation: .pointerRelative, dx: 0.5, dy: 0)])
    #expect(throws: AstraError.self) { _ = try fraction.validated() }
}

@Test func rewardCoverageKeepsOriginalTimesAndUsesTheSameReadyAndBaselineSnapshot() throws {
    let surface = SurfaceDescriptor(id: "frame", globalBounds: .init(x: 0, y: 0, width: 10, height: 10), pixelWidth: 10, pixelHeight: 10)
    let scope = ControlScope(surfaces: [surface], wholeDesktop: true)
    var score = RewardSignal(name: "Score", kind: .ocrNumber, surfaceID: surface.id, region: .init(x: 0, y: 0, width: 1, height: 1))
    score.maximumAgeMS = 1
    let rule = RewardRule(name: "Change", kind: .scoreDelta, amount: 1, signalID: score.id)
    var program = RewardProgram(name: "Coverage", signals: [score], rules: [rule])
    program.ready = .init(conditions: [.init(signalID: score.id, comparison: .atLeast, number: 40)])
    var evaluator = try RewardEvaluator(program: program)
    let episode = UUID(), sourceID = UUID(), cutoff: UInt64 = 1_000_000_000
    let reading = SignalReading(signalID: score.id, episodeID: episode, eventNanos: 10, observedNanos: 20,
                                value: .number(42), sourceObservationID: sourceID)
    #expect(try evaluator.readiness(episodeID: episode, cutoffNanos: cutoff, readings: [reading]) == .unknown)
    let source = RewardSourceCoverage(sourceObservationID: sourceID, surface: surface, eventNanos: 10, observedNanos: 20,
                                     throughNanos: cutoff, verifiedAtNanos: cutoff, kind: .unchanged)
    let snapshot = try evaluator.resolveSnapshot(episodeID: episode, cutoffNanos: cutoff, readings: [reading],
        coverage: [.init(signalID: score.id, source: source)], scope: scope, minimumEvidenceNanos: cutoff)
    #expect(snapshot.readings[0].eventNanos == 10 && snapshot.readings[0].observedNanos == 20)
    #expect(try evaluator.readiness(snapshot: snapshot) == .yes)
    try evaluator.reset(snapshot: snapshot, controlsReleased: true)
    let next = SignalReading(signalID: score.id, episodeID: episode, eventNanos: cutoff + 100, observedNanos: cutoff + 100, value: .number(43))
    #expect(try evaluator.evaluate(endNanos: cutoff + 100, readings: [next]).value == 1)
    var changed = program; changed.rules[0].amount = 2
    var other = try RewardEvaluator(program: changed)
    #expect(throws: AstraError.self) { try other.reset(snapshot: snapshot, controlsReleased: true) }
}

@Test func unrelatedOrRetimestampedCoverageCannotRefreshARewardSignal() throws {
    let one = SurfaceDescriptor(id: "one", globalBounds: .init(x: 0, y: 0, width: 10, height: 10), pixelWidth: 10, pixelHeight: 10)
    let two = SurfaceDescriptor(id: "two", globalBounds: .init(x: 10, y: 0, width: 10, height: 10), pixelWidth: 10, pixelHeight: 10)
    let scope = ControlScope(surfaces: [one, two], wholeDesktop: true)
    var signal = RewardSignal(name: "Ready text", kind: .ocrText, surfaceID: one.id, region: .init(x: 0, y: 0, width: 1, height: 1)); signal.maximumAgeMS = 1
    var program = RewardProgram(name: "Fresh source", signals: [signal]); program.ready = .init(conditions: [.init(signalID: signal.id, comparison: .equalText, text: "Ready")])
    let evaluator = try RewardEvaluator(program: program), episode = UUID(), sourceID = UUID()
    let reading = SignalReading(signalID: signal.id, episodeID: episode, eventNanos: 10, observedNanos: 20, value: .text("Ready"), sourceObservationID: sourceID)
    func evidence(surface: SurfaceDescriptor = one, id: UUID? = nil, observed: UInt64 = 20, through: UInt64 = 100_000_000,
                  verified: UInt64 = 100_000_000, kind: RewardCoverageKind = .unchanged) -> RewardReadingCoverage {
        .init(signalID: signal.id, source: .init(sourceObservationID: id ?? sourceID, surface: surface, eventNanos: 10,
              observedNanos: observed, throughNanos: through, verifiedAtNanos: verified, kind: kind))
    }
    for item in [evidence(surface: two), evidence(id: UUID()), evidence(observed: 21), evidence(verified: 100_000_001), evidence(kind: .frame)] {
        #expect(throws: AstraError.self) { _ = try evaluator.resolveSnapshot(episodeID: episode, cutoffNanos: 100_000_000, readings: [reading], coverage: [item], scope: scope) }
    }
    let wallPoll = try evaluator.resolveSnapshot(episodeID: episode, cutoffNanos: 100_000_000, readings: [reading],
        coverage: [evidence(through: 10, kind: .frame)], scope: scope, minimumEvidenceNanos: 100)
    #expect(try evaluator.readiness(snapshot: wallPoll) == .unknown)
    let low = SignalReading(signalID: signal.id, episodeID: episode, eventNanos: 10, observedNanos: 20, confidence: 0.1,
                            value: .text("Ready"), sourceObservationID: sourceID)
    let lowSnapshot = try evaluator.resolveSnapshot(episodeID: episode, cutoffNanos: 100_000_000, readings: [low], coverage: [evidence()], scope: scope)
    #expect(try evaluator.readiness(snapshot: lowSnapshot) == .unknown)
}
