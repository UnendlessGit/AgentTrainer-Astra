import AppKit
import SwiftUI
import Testing
@testable import AgentTrainerAstra

@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_QUEUED_MODEL_RENDER_DIR"] != nil))
@MainActor func renderQueuedActionMemoryChoice() async throws {
    let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_QUEUED_MODEL_RENDER_DIR"]))
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    NSApplication.shared.setActivationPolicy(.prohibited)
    for scheme in [ColorScheme.light, .dark] {
        let panel = VStack(alignment: .leading, spacing: 18) {
            QueuedActionCheckpointDescription(sourceName: "Careful desktop control · saved behavioral checkpoint")
            Divider()
            QueuedActionMemoryOption(enabled: .constant(true))
            HStack { Spacer(); Button("Cancel") {}; Button("Create Copy") {}.buttonStyle(.borderedProminent) }
        }.padding(24)
        _ = try await renderOwnedView(AnyView(panel), name: "queued-action-choice", size: .init(width: 540, height: 480), scheme: scheme, output: output)
    }
}
