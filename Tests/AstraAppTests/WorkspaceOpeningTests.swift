import Foundation
import Testing
import AstraCore
@testable import AgentTrainerAstra

@Test @MainActor func workspaceOpeningFailureRequiresDeliberateRetryAfterPathRepair() async throws {
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    let parent = temporary.appendingPathComponent("blocked-parent")
    let original = Data("Temporary path obstruction".utf8)
    try original.write(to: parent)
    let root = parent.appendingPathComponent("workspace")
    let model = WorkspaceModel(historyRoot: root)
    await model.start()
    #expect(!model.loading && !model.workspaceReady)
    #expect(model.openingFailure?.root == root && model.openingFailure?.message.isEmpty == false)
    #expect(model.inferenceUnavailableReason != nil && model.artifactUnavailableReason != nil)
    #expect(model.learning == nil && model.inference == nil && model.desktopLearning == nil)
    #expect(try Data(contentsOf: parent) == original)
    await model.createAgent(name: "Blocked")
    #expect(model.agents.isEmpty)
    try FileManager.default.removeItem(at: parent)
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    await model.start() // A recreated view task must not retry access implicitly.
    #expect(model.openingFailure != nil && !FileManager.default.fileExists(atPath: root.path))
    model.errorMessage = "Previous message"
    await model.retryOpening()
    #expect(model.workspaceReady && model.openingFailure == nil && model.errorMessage == nil)
    await model.createAgent(name: "After repair")
    #expect(model.agents.map(\.name) == ["After repair"])
    let saved = try await LibraryStore(root: root).snapshot()
    #expect(saved.agents.map(\.name) == ["After repair"])
}

@Test @MainActor func workspaceOpeningReleasesPartialOwnersBeforeCatalogRetry() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    _ = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Retained agent")
    try await LibraryStore(root: root).save(agent)
    let transferID = UUID()
    let database = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    try database.execute("INSERT INTO agents(id,name,document,created) VALUES(?,?,?,0)",
                         [.text(UUID().uuidString), .text("Nonfatal corrupt agent"), .blob(Data([0]))])
    try database.execute("INSERT INTO artifact_transfers(id,operation,status,document,created) VALUES(?,'import','needsAttention',?,0)",
                         [.text(transferID.uuidString), .blob(Data([0]))])
    try database.close()
    let marker = root.appendingPathComponent("preserve.txt"), bytes = Data("Existing data".utf8)
    try bytes.write(to: marker)
    let model = WorkspaceModel(historyRoot: root)
    await model.start()
    #expect(model.openingFailure != nil && !model.workspaceReady)
    #expect(model.learning == nil && model.inference == nil && model.desktopLearning == nil)
    do {
        let released = try LibraryLease(root: root)
        withExtendedLifetime(released) {} // Failed startup cannot retain its old lease through retry.
    }
    let repair = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    try repair.execute("DELETE FROM artifact_transfers WHERE id=?", [.text(transferID.uuidString)])
    try repair.close()
    await model.retryOpening()
    #expect(model.workspaceReady && model.agents.map(\.id) == [agent.id])
    #expect(model.openingFailure == nil && model.issues.contains { $0.collection == "agents" })
    #expect(try Data(contentsOf: marker) == bytes)
}
