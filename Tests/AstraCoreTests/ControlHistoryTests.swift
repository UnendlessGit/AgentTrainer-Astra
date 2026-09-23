import Foundation
import Testing
@testable import AstraCore

private func historyRoot() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraHistory-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}
private func historyFiles(_ root: URL, location: ControlHistoryLocation, id: UUID, result: Bool? = false) throws -> URL {
    let directory = root.appendingPathComponent(location.rawValue).appendingPathComponent(id.uuidString.lowercased())
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let configuration: JSONValue = .object(["runID": .string(id.uuidString.lowercased()), "sourceID": .string("fixture")])
    try JSONEncoder().encode(configuration).write(to: directory.appendingPathComponent("configuration.json"))
    if let result { try historyResult(id, location: location, confirmed: result).write(to: directory.appendingPathComponent("results.json")) }
    return directory
}
private func historyResult(_ id: UUID, location: ControlHistoryLocation, confirmed: Bool, note: String = "first") throws -> Data {
    try JSONEncoder().encode(JSONValue.object(["schemaVersion": .integer(1), "runID": .string(id.uuidString.lowercased()),
        location.cleanupKey: .bool(confirmed), "note": .string(note)]))
}

@Test func desktopControlHistoryAcknowledgementSurvivesReopenWithoutChangingNativeEvidence() async throws {
    let root = try historyRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), id = UUID()
    let directory = try historyFiles(root, location: .desktop, id: id)
    let original = try Data(contentsOf: directory.appendingPathComponent("results.json"))
    let configuration = try Data(contentsOf: directory.appendingPathComponent("configuration.json"))
    try await store.inspectPriorInferenceRuns()
    let review = try #require(await store.snapshot().issues.first?.controlHistory)
    #expect(review.location == .desktop && review.runID == id)
    try await store.acknowledgeControlHistory(review)
    try await store.acknowledgeControlHistory(review) // Exact acknowledgement is idempotent.
    let reopened = try LibraryStore(root: root)
    try await reopened.inspectPriorInferenceRuns()
    #expect(try await reopened.snapshot().issues.isEmpty)
    #expect(try Data(contentsOf: directory.appendingPathComponent("results.json")) == original)
    #expect(try Data(contentsOf: directory.appendingPathComponent("configuration.json")) == configuration)
    let db = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"), readOnly: true)
    #expect(try db.query("SELECT fingerprint FROM control_history_acknowledgements").count == 1)
}

@Test func changedOrNewRunHistoryNeverInheritsAnEarlierAcknowledgement() async throws {
    let root = try historyRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), id = UUID()
    let directory = try historyFiles(root, location: .inference, id: id)
    try await store.inspectPriorInferenceRuns()
    let first = try #require(await store.snapshot().issues.first?.controlHistory)
    try await store.acknowledgeControlHistory(first)
    // Rewrite equal-length bytes in the same inode; content, not just file
    // identity/size, must change the review fingerprint.
    let file = try FileHandle(forWritingTo: directory.appendingPathComponent("results.json"))
    try file.write(contentsOf: historyResult(id, location: .inference, confirmed: false, note: "other")); try file.close()
    await #expect(throws: AstraError.self) { try await store.acknowledgeControlHistory(first) }
    try await store.inspectPriorInferenceRuns()
    let changed = try #require(await store.snapshot().issues.first?.controlHistory)
    #expect(changed.runID == first.runID && changed.fingerprint != first.fingerprint)
    try await store.acknowledgeControlHistory(changed)
    let next = UUID(); _ = try historyFiles(root, location: .inference, id: next)
    try await store.inspectPriorInferenceRuns()
    #expect(try await store.snapshot().issues.compactMap(\.controlHistory).map(\.runID) == [next])
}

@Test func interruptedHistoryAcknowledgementDoesNotHideALaterResultOrChangedConfiguration() async throws {
    let root = try historyRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), id = UUID()
    let directory = try historyFiles(root, location: .desktop, id: id, result: nil)
    let incomplete = try #require(await store.controlHistoryReview(location: .desktop, runID: id))
    try await store.acknowledgeControlHistory(incomplete)
    try historyResult(id, location: .desktop, confirmed: false).write(to: directory.appendingPathComponent("results.json"))
    try await store.inspectPriorInferenceRuns()
    let completed = try #require(await store.snapshot().issues.first?.controlHistory)
    #expect(completed.fingerprint != incomplete.fingerprint)
    try await store.acknowledgeControlHistory(completed)
    try Data("changed configuration".utf8).write(to: directory.appendingPathComponent("configuration.json"))
    try await store.inspectPriorInferenceRuns()
    #expect(try await store.snapshot().issues.first?.controlHistory?.fingerprint != completed.fingerprint)
}

@Test func corruptBoundedRunResultsCanBeAcknowledgedOnlyForTheirExactBytes() async throws {
    let root = try historyRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), id = UUID()
    let directory = try historyFiles(root, location: .desktop, id: id)
    try Data("{broken".utf8).write(to: directory.appendingPathComponent("results.json"))
    try await store.inspectPriorInferenceRuns()
    let review = try #require(await store.snapshot().issues.first?.controlHistory)
    try await store.acknowledgeControlHistory(review)
    try await store.inspectPriorInferenceRuns()
    #expect(try await store.snapshot().issues.isEmpty)
    try Data("{other!".utf8).write(to: directory.appendingPathComponent("results.json"))
    try await store.inspectPriorInferenceRuns()
    #expect(try await store.snapshot().issues.first?.controlHistory?.fingerprint != review.fingerprint)
}

@Test func aDamagedHistoryDirectoryCannotHideTheOtherControlNamespaceOrCatalog() async throws {
    let root = try historyRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), agent = AgentDocument(name: "Existing", createdAt: Date(timeIntervalSince1970: 1000))
    try await store.save(agent)
    try Data("not a directory".utf8).write(to: root.appendingPathComponent("Runs"))
    let id = UUID(); _ = try historyFiles(root, location: .desktop, id: id)
    try await store.inspectPriorInferenceRuns()
    let snapshot = try await store.snapshot()
    #expect(snapshot.agents == [agent])
    #expect(snapshot.issues.contains { $0.id == "Runs.history" && $0.blocksLiveControl && $0.controlHistory == nil })
    let review = try #require(snapshot.issues.compactMap(\.controlHistory).first)
    #expect(review.runID == id)
    try await store.acknowledgeControlHistory(review)
    #expect(try await store.snapshot().issues.map(\.id) == ["Runs.history"])
}

@Test func linkedAndOversizedHistoryFilesCannotBeTurnedIntoAnAcknowledgement() async throws {
    let root = try historyRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root), linked = UUID(), huge = UUID()
    let first = try historyFiles(root, location: .inference, id: linked, result: nil)
    let outside = root.appendingPathComponent("unrelated.json")
    try historyResult(linked, location: .inference, confirmed: false).write(to: outside)
    try FileManager.default.createSymbolicLink(at: first.appendingPathComponent("results.json"), withDestinationURL: outside)
    let second = try historyFiles(root, location: .desktop, id: huge, result: nil)
    try Data(repeating: 0, count: ControlHistoryReader.maximumResultBytes + 1).write(to: second.appendingPathComponent("results.json"))
    try await store.inspectPriorInferenceRuns()
    let issues = try await store.snapshot().issues
    #expect(issues.count == 2 && issues.allSatisfy { $0.blocksLiveControl && $0.controlHistory == nil })
    await #expect(throws: AstraError.self) { try await store.controlHistoryReview(location: .inference, runID: linked) }
    await #expect(throws: AstraError.self) { try await store.controlHistoryReview(location: .desktop, runID: huge) }
}
