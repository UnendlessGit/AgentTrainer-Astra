import AppKit
import SwiftUI
import Testing
import AstraCore
@testable import AgentTrainerAstra

/// Owned view pixels only; never a screen or another application's hierarchy.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_HISTORY_RENDER_DIR"] != nil))
@MainActor func renderControlCleanupHistoryRecovery() async throws {
    let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_HISTORY_RENDER_DIR"]))
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraHistoryRender-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Desktop agent", createdAt: Date(timeIntervalSince1970: 1000))
    try await store.save(agent)
    for location in [ControlHistoryLocation.inference, .desktop] {
        let id = UUID(), path = root.appendingPathComponent(location.rawValue + "/" + id.uuidString.lowercased())
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
        let key = location == .inference ? "cleanupConfirmed" : "controlCleanupConfirmed"
        try JSONEncoder().encode(JSONValue.object(["runID": .string(id.uuidString), key: .bool(false)]))
            .write(to: path.appendingPathComponent("results.json"))
    }
    let model = WorkspaceModel(historyStore: store, historyRoot: root)
    await model.refreshControlHistory(); model.destination = .agent(agent.id); model.section = .run
    NSApplication.shared.setActivationPolicy(.prohibited)
    for size in [NSSize(width: 1120, height: 760), NSSize(width: 860, height: 580)] {
        for scheme in [ColorScheme.light, .dark] {
            let rect = NSRect(origin: .zero, size: size)
            let hosting = NSHostingView(rootView: WorkspaceView(model: model).environment(\.colorScheme, scheme).preferredColorScheme(scheme)
                .frame(width: size.width, height: size.height).background(Color(nsColor: .windowBackgroundColor)))
            hosting.sizingOptions = []
            let window = NSWindow(contentRect: NSRect(x: -20_000, y: -20_000, width: size.width, height: size.height),
                                  styleMask: [.borderless], backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            defer { window.orderOut(nil); window.contentView = nil; window.close() }
            let appearance = try #require(NSAppearance(named: scheme == .dark ? .darkAqua : .aqua))
            window.appearance = appearance; hosting.appearance = appearance
            window.contentView = hosting; hosting.frame = rect; window.setContentSize(size); window.orderFront(nil)
            try await Task.sleep(for: .milliseconds(160)); hosting.layoutSubtreeIfNeeded(); hosting.displayIfNeeded()
            try await Task.sleep(for: .milliseconds(80))
            let bitmap = try #require(hosting.bitmapImageRepForCachingDisplay(in: rect))
            hosting.cacheDisplay(in: rect, to: bitmap)
            let png = try #require(bitmap.representation(using: .png, properties: [:]))
            #expect(png.count > 10_000)
            try png.write(to: output.appendingPathComponent("history-\(scheme == .dark ? "dark" : "light")-\(Int(size.width))x\(Int(size.height)).png"))
        }
    }
}
