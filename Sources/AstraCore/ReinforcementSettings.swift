import Foundation

/// Learning settings independent of the environment or its reset mechanism.
/// Checkpoints retain the compute runtime's fully resolved configuration.
public struct ReinforcementSettings: Codable, Hashable, Sendable {
    public var rolloutDecisions = 512
    public var epochs = 4
    public var sequenceLength = 64
    public var burnIn = 32
    public var effectiveBatchDecisions = 256
    public var learningRate = 0.0001
    public var pretrainedLearningRate = 0.00001
    public var clipRatio = 0.2
    public var targetKL = 0.02
    public var entropyCoefficient = 0.01
    public var discountHalfLifeSeconds = 30.0
    public var seed = 0

    public init() {}

    public func validated() throws -> Self {
        guard (1...65_536).contains(rolloutDecisions), (1...100).contains(epochs),
              (1...512).contains(sequenceLength), (0...4096).contains(burnIn),
              (1...65_536).contains(effectiveBatchDecisions), (0...1_000_000_000).contains(seed),
              [learningRate, pretrainedLearningRate, clipRatio, targetKL, entropyCoefficient, discountHalfLifeSeconds].allSatisfy(\.isFinite),
              learningRate > 0, learningRate <= 1, pretrainedLearningRate > 0, pretrainedLearningRate <= 1,
              clipRatio > 0, clipRatio < 1, targetKL > 0, targetKL <= 1,
              entropyCoefficient >= 0, entropyCoefficient <= 1, discountHalfLifeSeconds > 0 else {
            throw AstraError("reinforcement.settings", "Choose valid sequence, batching, optimizer and reward-discount settings.")
        }
        return self
    }

    public var payload: JSONValue {
        .object(["rollout_decisions": .integer(Int64(rolloutDecisions)), "epochs": .integer(Int64(epochs)),
            "sequence_length": .integer(Int64(sequenceLength)), "burn_in": .integer(Int64(burnIn)),
            "effective_batch_decisions": .integer(Int64(effectiveBatchDecisions)),
            "learning_rate": .number(learningRate), "pretrained_learning_rate": .number(pretrainedLearningRate),
            "seed": .integer(Int64(seed)), "ppo": .object(["clip_ratio": .number(clipRatio),
                "target_kl": .number(targetKL), "entropy_coefficient": .number(entropyCoefficient)]),
            "returns": .object(["discount_half_life_seconds": .number(discountHalfLifeSeconds)])])
    }
}
