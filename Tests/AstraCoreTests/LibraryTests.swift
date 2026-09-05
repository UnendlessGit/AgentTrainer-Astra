import Foundation
import Testing
@testable import AstraCore

@Test func catalogPersistsAgentsWithoutDuplicatingSharedEnvironment() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let environment = EnvironmentDocument(name: "Workspace", kind: .desktop)
    try await store.save(environment)
    let agent = AgentDocument(name: "First agent", environmentID: environment.id)
    try await store.save(agent)
    let copy = try await store.duplicateAgent(agent)
    #expect(copy.id != agent.id)
    #expect(copy.environmentID == environment.id)
    try await store.archiveAgent(id: agent.id, archived: true)
    #expect(try await store.snapshot().agents.map(\.id) == [copy.id])
    try await store.archiveAgent(id: agent.id, archived: false)
    let reopened = try LibraryStore(root: root)
    let snapshot = try await reopened.snapshot()
    #expect(snapshot.agents.count == 2)
    #expect(snapshot.environments.count == 1)
    #expect(snapshot.issues.isEmpty)
}

@Test func corruptCatalogItemIsVisibleAsAnIssueAndIsNotDeleted() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    try await store.save(AgentDocument(name: "Readable"))
    let database = try SQLiteDatabase(url: root.appendingPathComponent("library.sqlite"))
    try database.execute("INSERT INTO agents(id,name,document,created) VALUES(?,?,?,?)", [
        .text("broken"), .text("Broken"), .blob(Data([255])), .real(0)
    ])
    let snapshot = try await store.snapshot()
    #expect(snapshot.agents.count == 1)
    #expect(snapshot.issues.count == 1)
    #expect(snapshot.issues.first?.id == "broken")
    #expect(try database.query("SELECT id FROM agents").count == 2)
}
