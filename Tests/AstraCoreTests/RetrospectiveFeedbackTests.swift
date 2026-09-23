import Foundation
import Testing
@testable import AstraCore

private struct FeedbackFixture {
    let directory: URL
    let positive = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
    let negative = UUID(uuidString: "00000000-0000-0000-0000-000000000002")!
    let metadata: FeedbackTrajectoryMetadata
    let metadataBytes: Data, programBytes: Data
    let source: VerifiedFeedbackSource

    init() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("astra-feedback-" + UUID().uuidString)
        let program = RewardProgram(name: "Reviewed experience", rules: [
            .init(id: positive, name: "Good", kind: .manualMarker, amount: 1),
            .init(id: negative, name: "Bad", kind: .manualMarker, amount: -2),
            .init(name: "Other reward", kind: .ratePerSecond, amount: 0.5)])
        programBytes = try FeedbackArtifactStore.encoded(program)
        let episode = UUID(), policy = UUID(), secondObservation = UUID(), endpoint = UUID()
        let intervals: [FeedbackEligibleInterval] = [
            .init(target: .init(episodeID: episode, observationID: UUID(), packetID: UUID(), startNanos: 100, endNanos: 200),
                  packetSequence: 7, drawIndex: 7, endpointObservationID: secondObservation, execution: "executed", outcome: "continuing", eligible: true),
            .init(target: .init(episodeID: episode, observationID: secondObservation, packetID: UUID(), startNanos: 200, endNanos: 300),
                  packetSequence: 8, drawIndex: 8, endpointObservationID: endpoint, execution: "semanticBoundaryPrefix", outcome: "truncated", eligible: true,
                  bootstrap: .init(episodeID: episode, policyID: policy, observationID: endpoint, cutoffNanos: 300, value: 0.75))]
        metadata = try .init(collectionID: UUID(), runID: UUID(), clockID: UUID(), sourceSessionID: UUID(), policyID: policy, program: program,
            trajectorySHA256: String(repeating: "a", count: 64), policySignature: String(repeating: "b", count: 64),
            programSHA256: FeedbackArtifactStore.digest(programBytes), status: "awaiting_manual_review", controlClosureKnown: true,
            closedNanos: 400, closedWallTimeMS: 1000, intervals: intervals)
        metadataBytes = try FeedbackArtifactStore.encoded(metadata)
        source = try VerifiedFeedbackSource(metadataBytes: metadataBytes, expectedSHA256: FeedbackArtifactStore.digest(metadataBytes), programBytes: programBytes)
    }
    func authored(_ nanos: UInt64 = 600) -> FeedbackAuthorship {
        .init(sessionID: metadata.sourceSessionID, clockID: metadata.clockID, observedNanos: nanos, wallTimeMS: 1000 + nanos)
    }
    func label(count: Int = 2, id: UUID = UUID(), sequence: Int = 0, rule: UUID? = nil,
               target: FeedbackIntervalIdentity? = nil, authored: FeedbackAuthorship? = nil) -> FeedbackAnnotation {
        .init(id: id, sequence: sequence, target: target ?? metadata.intervals[0].target,
              ruleID: rule ?? positive, count: count, authored: authored ?? self.authored(500))
    }
    func review(pairs: [FeedbackReviewPair]? = nil, authored: FeedbackAuthorship? = nil) -> FeedbackReviewCompletion {
        .init(sequence: 0, pairs: pairs ?? [
            .init(packetID: metadata.intervals[0].target.packetID, ruleID: positive),
            .init(packetID: metadata.intervals[0].target.packetID, ruleID: negative),
            .init(packetID: metadata.intervals[1].target.packetID, ruleID: negative)], authored: authored ?? self.authored(550))
    }
    func revision() throws -> FeedbackRewardRevision {
        try .create(source: source, authored: authored(), annotations: [label()], reviews: [review()])
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
    func alteredSource(_ change: (inout [String: JSONValue]) throws -> Void) throws -> VerifiedFeedbackSource {
        var fields = try JSONDecoder().decode(JSONValue.self, from: metadataBytes).fields!
        try change(&fields)
        let data = try FeedbackArtifactStore.encoded(JSONValue.object(fields))
        return try VerifiedFeedbackSource(metadataBytes: data, expectedSHA256: FeedbackArtifactStore.digest(data), programBytes: programBytes)
    }
}

@Suite struct RetrospectiveFeedbackTests {
@Test func exactManualRulesResolveReviewedZeroAndPreserveUnknownAndSourceEligibility() throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let revision = try fixture.revision()
    #expect(revision.resolved.count == 2)
    #expect(revision.resolved[0].manualReward == 2)
    #expect(revision.resolved[0].components.first(where: { $0.ruleID == fixture.negative })?.value == 0)
    #expect(revision.resolved[1].manualReward == nil)
    #expect(revision.resolved[1].components.first(where: { $0.ruleID == fixture.positive })?.reviewed == false)
    #expect(fixture.source.metadata.intervals[1].execution == "semanticBoundaryPrefix")
    #expect(fixture.source.metadata.intervals[1].bootstrap?.value == 0.75)
}

@Test(arguments: ["aborted", "audited", "unjoined", "rejected", "ineligible", "missingBootstrap", "changedBootstrap", "changedDraw", "manualProjection"])
func unsuitableSourceCannotBeMadeTrainableByFeedback(mode: String) throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    #expect(throws: (any Error).self) {
        try fixture.alteredSource { fields in
            if ["aborted", "audited"].contains(mode) { fields["status"] = .string(mode); return }
            if mode == "unjoined" { fields["controlClosureKnown"] = .bool(false); return }
            if mode == "manualProjection" { fields["manualRules"] = .array([]); return }
            var intervals = try fields["intervals"]!.decode([JSONValue].self)
            var interval = intervals[1].fields!
            switch mode {
            case "rejected": interval["execution"] = .string("rejected")
            case "ineligible": interval["eligible"] = .bool(false)
            case "missingBootstrap": interval.removeValue(forKey: "bootstrap")
            case "changedBootstrap": var bootstrap = interval["bootstrap"]!.fields!; bootstrap["cutoffNanos"] = .integer(301); interval["bootstrap"] = .object(bootstrap)
            default: interval["drawIndex"] = .integer(9)
            }
            intervals[1] = .object(interval); fields["intervals"] = .array(intervals)
        }
    }
}

@Test(arguments: ["unreviewed", "negativeCount", "tooMany", "wrongRule", "wrongInterval", "duplicatePair", "reviewBeforeLabel", "duplicateAnnotation"])
func annotationAndReviewMustMatchExplicitOriginalPairs(mode: String) throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    var labels = [fixture.label()], reviews = [fixture.review()]
    switch mode {
    case "unreviewed": reviews = []
    case "negativeCount": labels = [fixture.label(count: -1)]
    case "tooMany": labels = [fixture.label(count: FeedbackLimits.maximumCount + 1)]
    case "wrongRule": labels = [fixture.label(rule: UUID())]
    case "wrongInterval":
        let value = fixture.metadata.intervals[0].target
        labels = [fixture.label(target: .init(episodeID: value.episodeID, observationID: value.observationID, packetID: value.packetID,
                                             startNanos: value.startNanos, endNanos: value.endNanos + 1))]
    case "duplicatePair": reviews = [fixture.review(pairs: [fixture.review().pairs[0], fixture.review().pairs[0]])]
    case "reviewBeforeLabel": reviews = [fixture.review(authored: fixture.authored(450))]
    default: labels.append(fixture.label(id: labels[0].id, sequence: 1))
    }
    #expect(throws: (any Error).self) { try FeedbackRewardRevision.create(source: fixture.source, authored: fixture.authored(), annotations: labels, reviews: reviews) }
}

@Test func unrelatedMonotonicClocksUseDistinctSessionAndWallAudit() throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let other = FeedbackAuthorship(sessionID: UUID(), clockID: UUID(), observedNanos: 1, wallTimeMS: 2000)
    let valid = try FeedbackRewardRevision.create(source: fixture.source, authored: other,
        annotations: [fixture.label(authored: other)], reviews: [fixture.review(authored: other)])
    #expect(valid.resolved[0].manualReward == 2)
    let bad = [fixture.authored(399),
        FeedbackAuthorship(sessionID: fixture.metadata.sourceSessionID, clockID: other.clockID, observedNanos: 5000, wallTimeMS: 2000),
        FeedbackAuthorship(sessionID: other.sessionID, clockID: other.clockID, observedNanos: 5000, wallTimeMS: 999)]
    for value in bad {
        #expect(throws: (any Error).self) { try FeedbackRewardRevision.create(source: fixture.source, authored: value, annotations: [], reviews: []) }
    }
}

@Test func atomicPublicationRejectsOverwriteAndLoadsOnlyExpectedSourceAndDigest() throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let revision = try fixture.revision()
    let first = try FeedbackArtifactStore.publish(revision, source: fixture.source, directory: fixture.directory)
    let loaded = try FeedbackArtifactStore.load(first.reference, source: fixture.source, directory: fixture.directory)
    #expect(loaded.document == revision)
    #expect(throws: (any Error).self) { try FeedbackArtifactStore.publish(revision, source: fixture.source, directory: fixture.directory) }
    let wrong = FeedbackRevisionReference(id: first.reference.id, sha256: String(repeating: "0", count: 64))
    #expect(throws: (any Error).self) { try FeedbackArtifactStore.load(wrong, source: fixture.source, directory: fixture.directory) }
    let files = try FileManager.default.contentsOfDirectory(atPath: fixture.directory.path)
    #expect(files == [FeedbackArtifactStore.filename(revision.id)])
    #expect(throws: (any Error).self) {
        try VerifiedFeedbackSource(metadataBytes: fixture.metadataBytes + Data(" ".utf8), expectedSHA256: fixture.source.sha256, programBytes: fixture.programBytes)
    }
}

@Test(arguments: ["reward", "feature", "source", "interval", "nonfinite"])
func artifactEditsCannotForgeResolvedRewardsOrInputFeatures(mode: String) throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let saved = try FeedbackArtifactStore.publish(fixture.revision(), source: fixture.source, directory: fixture.directory)
    let path = fixture.directory.appendingPathComponent(FeedbackArtifactStore.filename(saved.reference.id))
    var fields = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: path)).fields!
    switch mode {
    case "feature": fields["controlState"] = .object(["labels": .array([.integer(1)])])
    case "source": fields["sourceSHA256"] = .string(String(repeating: "0", count: 64))
    case "interval":
        var annotations = try fields["annotations"]!.decode([JSONValue].self), label = annotations[0].fields!, target = label["target"]!.fields!
        target["endNanos"] = .integer(201); label["target"] = .object(target); annotations[0] = .object(label); fields["annotations"] = .array(annotations)
    default:
        var rows = try fields["resolved"]!.decode([JSONValue].self), row = rows[0].fields!
        row["manualReward"] = .integer(999); rows[0] = .object(row); fields["resolved"] = .array(rows)
    }
    var edited = try FeedbackArtifactStore.encoded(JSONValue.object(fields))
    if mode == "nonfinite" {
        edited = Data(String(decoding: edited, as: UTF8.self).replacingOccurrences(of: "\"manualReward\":999", with: "\"manualReward\":1e400").utf8)
    }
    try edited.write(to: path)
    #expect(throws: (any Error).self) { try FeedbackArtifactStore.load(saved.reference, source: fixture.source, directory: fixture.directory) }
    let changedReference = FeedbackRevisionReference(id: saved.reference.id, sha256: FeedbackArtifactStore.digest(edited))
    #expect(throws: (any Error).self) { try FeedbackArtifactStore.load(changedReference, source: fixture.source, directory: fixture.directory) }
    #expect(FeedbackArtifactStore.digest(fixture.metadataBytes) == fixture.source.sha256)
}

@Test func correctionKeepsImmutableParentAndCannotRewriteItsRecordIdentities() throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let initial = try fixture.revision()
    let first = try FeedbackArtifactStore.publish(initial, source: fixture.source, directory: fixture.directory)
    let authored = FeedbackAuthorship(sessionID: UUID(), clockID: UUID(), observedNanos: 5, wallTimeMS: 2000)
    let changed = try FeedbackRewardRevision.create(source: fixture.source, authored: authored,
        annotations: [fixture.label(count: 1, authored: authored)], reviews: [fixture.review(authored: authored)], parent: first)
    let second = try FeedbackArtifactStore.publish(changed, source: fixture.source, directory: fixture.directory, parent: first)
    #expect(second.document.revision == 1 && second.document.resolved[0].manualReward == 1)
    #expect(throws: (any Error).self) { try FeedbackArtifactStore.load(second.reference, source: fixture.source, directory: fixture.directory) }
    #expect(try FeedbackArtifactStore.load(second.reference, source: fixture.source, directory: fixture.directory, parent: first).document == changed)
    #expect(try FeedbackArtifactStore.load(first.reference, source: fixture.source, directory: fixture.directory).document == initial)
    #expect(throws: (any Error).self) {
        try FeedbackRewardRevision.create(source: fixture.source, authored: authored,
            annotations: [fixture.label(count: 1, id: initial.annotations[0].id, authored: authored)],
            reviews: [fixture.review(authored: authored)], parent: first)
    }
}

@Test func newCorrectionRecordsCannotPredateTheVerifiedParent() throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let initial = try fixture.revision()
    let parent = try FeedbackArtifactStore.publish(initial, source: fixture.source, directory: fixture.directory)
    #expect(throws: (any Error).self) {
        try FeedbackRewardRevision.create(source: fixture.source, authored: fixture.authored(700),
            annotations: [fixture.label(count: 1, authored: fixture.authored(550))],
            reviews: [fixture.review(authored: fixture.authored(650))], parent: parent)
    }
    #expect(throws: (any Error).self) {
        try FeedbackRewardRevision.create(source: fixture.source, authored: fixture.authored(700),
            annotations: initial.annotations, reviews: [fixture.review()], parent: parent)
    }
    let carried = try FeedbackRewardRevision.create(source: fixture.source, authored: fixture.authored(700),
        annotations: initial.annotations, reviews: initial.reviews, parent: parent)
    #expect(carried.resolved == initial.resolved)
}

@Test func sourceAndArtifactSymlinksAreRejected() throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let saved = try FeedbackArtifactStore.publish(fixture.revision(), source: fixture.source, directory: fixture.directory)
    let file = fixture.directory.appendingPathComponent(FeedbackArtifactStore.filename(saved.reference.id))
    let moved = fixture.directory.appendingPathComponent("original.json")
    try FileManager.default.moveItem(at: file, to: moved)
    try FileManager.default.createSymbolicLink(at: file, withDestinationURL: moved)
    #expect(throws: (any Error).self) { try FeedbackArtifactStore.load(saved.reference, source: fixture.source, directory: fixture.directory) }
}

@Test func concurrentPublicationExposesOnlyOneCompleteImmutableRevision() async throws {
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let revision = try fixture.revision()
    let winners = await withTaskGroup(of: Bool.self) { group in
        for _ in 0..<2 {
            group.addTask {
                do { _ = try FeedbackArtifactStore.publish(revision, source: fixture.source, directory: fixture.directory); return true }
                catch { return false }
            }
        }
        var successes = 0
        for await success in group where success { successes += 1 }
        return successes
    }
    #expect(winners == 1)
    let reference = FeedbackRevisionReference(id: revision.id, sha256: FeedbackArtifactStore.digest(try FeedbackArtifactStore.encoded(revision)))
    #expect(try FeedbackArtifactStore.load(reference, source: fixture.source, directory: fixture.directory).document == revision)
}

/// Optional direct native-to-Python fixture for the CPU-only interoperability
/// check. Normal tests keep all artifacts in their temporary directory.
@Test func feedbackCrossLanguageFixture() throws {
    guard let destination = ProcessInfo.processInfo.environment["ASTRA_FEEDBACK_FIXTURE"] else { return }
    let fixture = try FeedbackFixture(); defer { fixture.remove() }
    let folder = URL(fileURLWithPath: destination, isDirectory: true)
    let saved = try FeedbackArtifactStore.publish(fixture.revision(), source: fixture.source, directory: folder)
    try fixture.metadataBytes.write(to: folder.appendingPathComponent("source.json"), options: .withoutOverwriting)
    try fixture.programBytes.write(to: folder.appendingPathComponent("program.json"), options: .withoutOverwriting)
    try FeedbackArtifactStore.encoded(saved.reference).write(to: folder.appendingPathComponent("reference.json"), options: .withoutOverwriting)
}
}
