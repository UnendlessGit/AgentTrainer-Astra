import AppKit
import SwiftUI
import Testing
import AstraCore
@testable import AgentTrainerAstra

/// Four populated owned-view renders. No panels, capture or user artifact copy.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_ARTIFACT_RENDER_DIR"] != nil))
@MainActor func renderArtifactTransferAndStorageSettings() async throws {
    let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_ARTIFACT_RENDER_DIR"]))
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let root = output.appendingPathComponent("GeneratedLibrary-" + UUID().uuidString, isDirectory: true)
    let store = try LibraryStore(root: root)
    let agent = AgentDocument(name: "Desktop skills · precise interaction across familiar and unfamiliar layouts")
    try await store.save(agent)
    let environment = EnvironmentDocument(name: "Generated visual fixture", kind: .practice)
    try await store.save(environment)
    let surface = SurfaceDescriptor(id: "generated-transfer-fixture", globalBounds: .init(x: 0, y: 0, width: 32, height: 32),
        pixelWidth: 32, pixelHeight: 32)
    let pixels = Data(repeating: 100, count: 4096)
    for name in ["Careful pointing · browser settings and navigation", "Recovery demonstration · alternate windows and changed layouts"] {
        let manifest = RecordingManifest(name: name, environment: environment, recordedForAgentID: agent.id)
        let writer = try RecordingWriter(directory: store.layout.recordingDirectory(id: manifest.id), manifest: manifest)
        for time in [UInt64(1_000_000_000), 1_100_000_000] {
            try writer.append(FrameArchive.prepare(pixels: pixels, metadata: .init(eventNanos: time, observedNanos: time,
                surface: surface, byteCount: pixels.count)))
        }
        try await store.saveRecording(writer.finish(at: 1_200_000_000, status: .complete), linkTo: agent.id)
    }
    for (index, name) in ["Behavioral model · clean demonstrations and held-out layouts", "Reinforcement model · fine control after 24 learning iterations"].enumerated() {
        let checkpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: nil, name: name,
            kind: index == 0 ? "behavioral" : "reinforcement", trainingStep: (index + 1) * 240,
            policySignature: String(repeating: "a", count: 64), parameterCount: 34_638_639)
        try await store.saveCheckpoint(checkpoint)
    }
    let model = WorkspaceModel(historyStore: store, historyRoot: root)
    await model.refreshStorageAvailability(); model.destination = .agent(agent.id)
    #expect(model.errorMessage == nil && model.recordings.count == 2 && model.checkpoints.count == 2)
    #expect(model.artifactUnavailableReason == nil)
    NSApplication.shared.setActivationPolicy(.prohibited)
    var evidence: [[String: Any]] = []
    for scheme in [ColorScheme.light, .dark] {
        evidence.append(try await renderOwnedView(AnyView(ArtifactTransferView(model: model, mode: .exportArchive)),
            name: "artifact-export-selections", size: .init(width: 780, height: 690), scheme: scheme, output: output))
        evidence.append(try await renderOwnedView(AnyView(StorageSettingsView(model: model)),
            name: "artifact-storage-settings", size: .init(width: 640, height: 620), scheme: scheme, output: output))
    }
    let manifest: [String: Any] = ["generatedLibraryOnly": true, "library": root.path, "screenCapture": false, "filePanelsOpened": false,
        "artifactsCopied": false, "renderer": "owned NSHostingView", "images": evidence]
    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("manifest.json"), options: .atomic)
}
