import Testing
import AstraCore
@testable import AgentTrainerAstra

private let baseReinforcementMetric: [String: JSONValue] = [
    "iteration": .integer(1), "elapsed_seconds": .number(12), "mean_reward": .number(0.1),
    "mean_value_loss": .number(0.2), "mean_policy_loss": .number(-0.01),
    "maximum_sampled_kl": .number(0.18), "clip_fraction": .number(0.1), "optimizer_updates": .integer(4),
]

@Test func reinforcementMetricsPreferAcceptedPolicyAndKeepUnknownBacktrackingAbsent() throws {
    let legacy = try #require(ReinforcementMetric(baseReinforcementMetric))
    #expect(legacy.kl == 0.18 && !legacy.klIsAccepted)
    #expect(legacy.backtrackCount == nil && legacy.rejectedUpdates == nil && !legacy.hasCandidateKL)
    var fields = baseReinforcementMetric
    fields["maximum_accepted_kl"] = .number(0.008)
    fields["maximum_candidate_kl"] = .null
    fields["backtrack_count"] = .integer(3)
    fields["rejected_optimizer_steps"] = .integer(1)
    fields["minimum_step_scale"] = .number(0.125)
    let updated = try #require(ReinforcementMetric(fields))
    #expect(updated.kl == 0.008 && updated.klIsAccepted)
    #expect(updated.hasCandidateKL && updated.candidateKL == nil)
    #expect(updated.backtrackCount == 3 && updated.rejectedUpdates == 1 && updated.minimumStepScale == 0.125)
}

@Test func reinforcementMetricsRejectInvalidAcceptedValuesInsteadOfFallingBackToOldKL() {
    for invalid in [JSONValue.null, .number(-0.1), .string("0.02")] {
        var fields = baseReinforcementMetric; fields["maximum_accepted_kl"] = invalid
        #expect(ReinforcementMetric(fields) == nil)
    }
    for (key, invalid) in [("backtrack_count", JSONValue.integer(-1)), ("minimum_step_scale", .number(0)), ("maximum_candidate_kl", .number(-1))] {
        var fields = baseReinforcementMetric; fields[key] = invalid
        #expect(ReinforcementMetric(fields) == nil)
    }
}
