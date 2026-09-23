import Foundation
import Testing
@testable import AstraCore

private let pendingDigest = String(repeating: "a", count: 64)

private func pendingFragment(collectionID: UUID = UUID(), sourceDirectory: String? = nil,
                             manifestSHA256: String = pendingDigest, revisions: [FeedbackRevisionReference] = [],
                             boundary: JSONValue = .object(["closed": .bool(true)])) -> PendingFeedbackFragment {
    .init(collectionID: collectionID, sourceDirectory: sourceDirectory ?? "/catalog/\(collectionID)/source",
          manifestSHA256: manifestSHA256, revisionDirectory: "/catalog/\(collectionID)/revisions",
          revisions: revisions, boundary: boundary)
}

private func pendingFixture(fragments: [PendingFeedbackFragment]? = nil) -> PendingFeedbackDocument {
    let agentID = UUID()
    let checkpoint = CheckpointDocument(id: UUID(), agentID: agentID, runID: UUID(), name: "Original policy",
        kind: "reinforcement", trainingStep: 10, policySignature: pendingDigest, parameterCount: 10)
    return .init(behaviorBatchID: UUID(), agentID: agentID, checkpoint: checkpoint,
                 createdAt: Date(timeIntervalSince1970: 1_000), configuration: .object([
                    "environment": .object(["kind": .string("practice")]), "resumeLearner": .bool(true),
                    "training": .object(["batch": .integer(32)]), "context": .array([]), "target": .object([:])
                 ]), fragments: fragments ?? [pendingFragment()])
}

private func pendingArtifact() -> PendingFeedbackArtifactReference {
    let id = UUID()
    return .init(id: id, sha256: pendingDigest, path: "/catalog/artifacts/\(id).json")
}

@Test func pendingFeedbackPersistsDraftRevisionAndOrderedFragmentsAcrossReopen() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    var document = pendingFixture()
    document.fragments[0].draft = pendingArtifact()
    document.status = .reviewing
    try await store.savePendingFeedback(document)
    let reopened = try LibraryStore(root: root)
    let saved = try #require(try await reopened.snapshot().pendingFeedback.first)
    #expect(saved.behaviorBatchID == document.behaviorBatchID)
    #expect(saved.checkpoint.matchesIdentity(of: document.checkpoint))
    #expect(saved.fragments == document.fragments)
    #expect(saved.configuration == document.configuration)
    #expect(saved.status == .reviewing) // Reopening cannot claim completion.

    document.fragments[0].draft = pendingArtifact()
    document.modifiedAt += 1
    try await reopened.savePendingFeedback(document)
    document.fragments[0].revisions.append(.init(id: UUID(), sha256: pendingDigest))
    document.fragments[0].reviewedPackage = pendingArtifact()
    document.fragments[0].draft = nil
    document.fragments.append(pendingFragment())
    document.modifiedAt += 1
    try await reopened.savePendingFeedback(document)
    try await reopened.savePendingFeedback(document) // Idempotent catalog save.
    let final = try #require(try await store.snapshot().pendingFeedback.first)
    #expect(final.fragments == document.fragments)
    #expect(final.fragments.map(\.collectionID) == document.fragments.map(\.collectionID))
    #expect(try await store.snapshot().issues.isEmpty)
}

@Test func pendingFeedbackRejectsRewritingFrozenSourcesConfigurationAndRevisionHistory() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let revision = FeedbackRevisionReference(id: UUID(), sha256: pendingDigest)
    let document = pendingFixture(fragments: [pendingFragment(revisions: [revision]), pendingFragment()])
    try await store.savePendingFeedback(document)

    var changed = document
    changed.fragments.removeLast()
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(changed) }
    changed = document; changed.fragments.reverse()
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(changed) }
    changed = document; changed.fragments[0].revisions[0] = .init(id: revision.id, sha256: String(repeating: "b", count: 64))
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(changed) }
    changed = document
    changed.fragments[0] = pendingFragment(collectionID: document.fragments[0].collectionID,
        manifestSHA256: String(repeating: "b", count: 64), revisions: [revision])
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(changed) }
    changed = document
    changed.fragments[0] = pendingFragment(collectionID: document.fragments[0].collectionID,
        revisions: [revision], boundary: .object(["closed": .bool(false)]))
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(changed) }
    let newConfiguration = PendingFeedbackDocument(behaviorBatchID: document.id, agentID: document.agentID,
        checkpoint: document.checkpoint, createdAt: document.createdAt, configuration: .object([:]), fragments: document.fragments)
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(newConfiguration) }
    var checkpoint = document.checkpoint; checkpoint.id = UUID()
    let newPolicy = PendingFeedbackDocument(behaviorBatchID: document.id, agentID: document.agentID,
        checkpoint: checkpoint, createdAt: document.createdAt, configuration: document.configuration, fragments: document.fragments)
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(newPolicy) }
    let retained = try #require(try await store.snapshot().pendingFeedback.first)
    #expect(retained.fragments == document.fragments)
    #expect(retained.checkpoint.matchesIdentity(of: document.checkpoint))
}

@Test func pendingFeedbackPackageReplacementNeedsNewRevisionAndArtifactIdentity() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    var document = pendingFixture()
    document.fragments[0].revisions = [.init(id: UUID(), sha256: pendingDigest)]
    document.fragments[0].reviewedPackage = pendingArtifact()
    try await store.savePendingFeedback(document)
    var changed = document
    changed.fragments[0].reviewedPackage = pendingArtifact()
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(changed) }
    changed.fragments[0].revisions.append(.init(id: UUID(), sha256: pendingDigest))
    try await store.savePendingFeedback(changed)
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(document) }
    let previousPackage = try #require(changed.fragments[0].reviewedPackage)
    changed.fragments[0].revisions.append(.init(id: UUID(), sha256: pendingDigest))
    changed.fragments[0].reviewedPackage = .init(id: previousPackage.id, sha256: String(repeating: "b", count: 64), path: previousPackage.path)
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(changed) }
}

@Test func pendingFeedbackValidatesBoundsPathsDigestsAndDuplicateIdentities() throws {
    let fixture = pendingFixture()
    var changed = fixture; changed.fragments.append(changed.fragments[0])
    #expect(throws: AstraError.self) { try changed.validated() }
    for path in ["relative/source", "/catalog/../source", "/catalog//source", "/catalog/source\u{0}", "/" + String(repeating: "a", count: 4_096)] {
        changed = fixture; changed.fragments[0] = pendingFragment(sourceDirectory: path)
        #expect(throws: AstraError.self) { try changed.validated() }
    }
    changed = fixture; changed.fragments[0] = pendingFragment(manifestSHA256: String(repeating: "G", count: 64))
    #expect(throws: AstraError.self) { try changed.validated() }
    changed = fixture; changed.modifiedAt = fixture.createdAt - 1
    #expect(throws: AstraError.self) { try changed.validated() }
    changed = fixture; changed.fragments[0].reviewedPackage = pendingArtifact()
    #expect(throws: AstraError.self) { try changed.validated() }
    let revision = FeedbackRevisionReference(id: UUID(), sha256: pendingDigest)
    changed = fixture; changed.fragments[0].revisions = [revision, revision]
    #expect(throws: AstraError.self) { try changed.validated() }
    var nested: JSONValue = .object([:])
    for _ in 0..<34 { nested = .object(["nested": nested]) }
    for boundary in [nested, .object(["value": .number(.nan)]), .object(["value": .string(String(repeating: "x", count: 16_385))]), .array([])] {
        changed = fixture; changed.fragments[0] = pendingFragment(boundary: boundary)
        #expect(throws: AstraError.self) { try changed.validated() }
    }
    // Numeric representation may normalize on JSON round-trip; its value is immutable.
    let original = pendingFixture(fragments: [pendingFragment(boundary: .object(["nanos": .unsigned(10)]))])
    let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .millisecondsSince1970
    let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .millisecondsSince1970
    let decoded = try decoder.decode(PendingFeedbackDocument.self, from: encoder.encode(original))
    try original.validateUpdate(from: decoded)
}

@Test func pendingFeedbackCorruptOrOversizedCatalogRowsBecomeRecoverableIssues() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let valid = pendingFixture()
    try await store.savePendingFeedback(valid)
    let database = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    let malformedID = UUID(), oversizedID = UUID(), mismatchedID = UUID()
    try database.execute("INSERT INTO pending_feedback(id,document,created) VALUES(?,?,?)", [.text(malformedID.uuidString), .blob(Data("invalid json".utf8)), .real(0)])
    try database.execute("INSERT INTO pending_feedback(id,document,created) VALUES(?,zeroblob(?),?)", [.text(oversizedID.uuidString), .integer(Int64(PendingFeedbackDocument.maximumBytes + 1)), .real(0)])
    try database.execute("INSERT INTO pending_feedback(id,document,created) SELECT ?,document,created FROM pending_feedback WHERE id=?", [.text(mismatchedID.uuidString), .text(valid.id.uuidString)])
    let snapshot = try await store.snapshot()
    #expect(snapshot.pendingFeedback.map(\.id) == [valid.id])
    #expect(Set(snapshot.issues.filter { $0.collection == "pending_feedback" }.map(\.id)) == Set([malformedID, oversizedID, mismatchedID].map(\.uuidString)))
    let replacement = PendingFeedbackDocument(behaviorBatchID: oversizedID, agentID: valid.agentID,
        checkpoint: valid.checkpoint, configuration: valid.configuration, fragments: valid.fragments)
    await #expect(throws: AstraError.self) { try await store.savePendingFeedback(replacement) }
}
