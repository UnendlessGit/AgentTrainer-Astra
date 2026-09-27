import AppKit
import SwiftUI
import Testing
import AstraCore
@testable import AgentTrainerAstra

/// Non-secret presentation data only. No listener, credentials, clipboard,
/// operating-system input or ML runtime is created by this owned-view render.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_LIVE_SIGNAL_RENDER_DIR"] != nil))
@MainActor func renderLiveSignalConnectionPanel() async throws {
    let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_LIVE_SIGNAL_RENDER_DIR"]))
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    NSApplication.shared.setActivationPolicy(.prohibited)
    let values: [DesktopLiveValue] = [
        .init(id: UUID(), name: "Score", text: "240", current: true),
        .init(id: UUID(), name: "Ready after reset", text: "True", current: true),
        .init(id: UUID(), name: "Application state with a longer descriptive signal name", text: "Waiting for a current value", current: false),
        .init(id: UUID(), name: "Source continuity", text: "The state source skipped an update. A fresh reset binding is required.", current: false)
    ]
    let size = NSSize(width: 430, height: 670)
    var renders: [[String: Any]] = []
    for scheme in [ColorScheme.light, .dark] {
        let panel = LiveSignalConnectionPanel(address: "127.0.0.1:49152", sessionID: UUID(),
            status: "Local client connected · two signals are waiting for current values", values: values,
            canCopySignalIDs: true, copyConnection: { false }, copyToken: { false }, copySignalIDs: { false }, openGuide: {})
        renders.append(try await renderOwnedView(AnyView(panel.padding(20)), name: "local-state-connection-narrow",
            size: size, scheme: scheme, output: directory))
    }
    try JSONSerialization.data(withJSONObject: ["renderer": "owned NSHostingView", "generatedPresentationOnly": true,
        "credentialsCreated": false, "listenerStarted": false, "clipboardChanged": false, "images": renders], options: [.prettyPrinted, .sortedKeys])
        .write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
}
