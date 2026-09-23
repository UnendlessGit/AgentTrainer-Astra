import Foundation
import AppKit
import SwiftUI
import Testing
import AstraCore
@testable import AgentTrainerAstra

struct CorrectionPreludeReviewFixture {
    let directory: URL
    let reference: CorrectionReference
    let frames: [[FrameMetadata]]
    let sourceRunID = UUID(), checkpointID = UUID()
    init(empty: Bool = false) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("astra-correction-review-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var allFrames: [[FrameMetadata]] = []
        var observations: [CorrectionSourceObservation] = []
        for index in 0..<(empty ? 0 : 2) {
            let cutoff = UInt64(800 + index * 200) * 1_000_000
            let inputs: [CorrectionSourceFrame] = (0..<2).map { role in
                let surface = SurfaceDescriptor(id: "fixture-\(role)", globalBounds: .init(x: Double(role * 32), y: 0, width: 32, height: 32),
                    pixelWidth: 32, pixelHeight: 32, geometryRevision: UInt64(role * 7))
                let metadata = FrameMetadata(eventNanos: cutoff - 10_000_000, observedNanos: cutoff - 5_000_000,
                    surface: surface, byteCount: 4096, codec: "raw")
                var pixels = Data(repeating: 255, count: 4096)
                pixels.withUnsafeMutableBytes { raw in
                    let bytes = raw.bindMemory(to: UInt8.self)
                    for i in stride(from: 0, to: bytes.count, by: 4) {
                        bytes[i] = UInt8(40 + index * 30); bytes[i + 1] = UInt8(80 + role * 30); bytes[i + 2] = 160
                    }
                }
                return .init(metadata: metadata, pixels: pixels, coverage: nil)
            }
            allFrames.append(inputs.map(\.metadata))
            observations.append(.init(actorInput: .object(["observationID": .string(UUID().uuidString), "cutoffNanos": .unsigned(cutoff),
                "contextIDs": .array([])]), frames: inputs))
        }
        frames = allFrames
        reference = try CorrectionPrelude.write(.init(sourceRunID: sourceRunID, sourceCheckpointID: checkpointID,
            sourcePolicySignature: String(repeating: "b", count: 64), contextIDs: [], requestedAtNanos: 1_200_000_000,
            controlJoinedAtNanos: 1_300_000_000, observations: observations), supervisionStartNanos: 1_600_000_000, in: directory)
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
}

actor CorrectionPreludeReadGate {
    var active = 0, peak = 0
    var loaded: [UUID] = []
    var waiting: CheckedContinuation<Void, Never>?
    var holdNext = true
    var isWaiting: Bool { waiting != nil }
    func hold() { holdNext = true }
    func release() { waiting?.resume(); waiting = nil }
    func load(_ prelude: CorrectionPrelude, frame: CorrectionPrelude.Frame, directory: URL) async throws -> Data {
        active += 1; peak = max(active, peak); loaded.append(frame.block.metadata.id)
        defer { active -= 1 }
        if holdNext { holdNext = false; await withCheckedContinuation { waiting = $0 } }
        return try prelude.pixels(for: frame, in: directory)
    }
}

@MainActor private func correctionEventually(_ condition: () async -> Bool) async throws {
    let deadline = ContinuousClock.now + .seconds(3)
    while !(await condition()) {
        guard ContinuousClock.now < deadline else { throw AstraError("fixture.timeout", "Correction review did not reach its expected read boundary.") }
        try await Task.sleep(for: .milliseconds(2))
    }
}

@Suite @MainActor struct CorrectionPreludeReviewTests {
    @Test func rapidScrubbingAndReopeningShareOneOriginalFrameRead() async throws {
        let fixture = try CorrectionPreludeReviewFixture(); defer { fixture.remove() }
        let gate = CorrectionPreludeReadGate()
        let model = CorrectionPreludeReviewModel(dependencies: .init(loadPrelude: CorrectionPreludeReviewDependencies.live.loadPrelude,
            loadPixels: { prelude, frame, directory in try await gate.load(prelude, frame: frame, directory: directory) }))
        model.open(reference: fixture.reference, directory: fixture.directory)
        try await correctionEventually { await gate.isWaiting }
        #expect(model.count == 2 && model.selectedIndex == 1 && model.secondsBeforeRequest == 0.2)
        #expect(model.retainedSeconds == 0.2)
        #expect(model.prelude?.sourceRunID == fixture.sourceRunID && model.prelude?.sourceCheckpointID == fixture.checkpointID)
        model.select(0); model.selectSurface(1)
        await gate.release(); await model.waitForIdle()
        #expect(model.frame?.id == fixture.frames[0][1].id && model.image != nil && model.issue == nil)
        #expect(model.secondsBeforeRequest == 0.4 && model.selectedImageAgeMS == 10)
        let peakBefore = await gate.peak, countBefore = await gate.loaded.count
        #expect(peakBefore == 1 && countBefore == 2)
        await gate.hold(); model.select(1)
        try await correctionEventually { await gate.isWaiting }
        model.close(); model.open(reference: fixture.reference, directory: fixture.directory)
        #expect(model.image == nil && model.prelude == nil)
        await gate.release(); await model.waitForIdle()
        #expect(model.selectedIndex == 1 && model.selectedSurface == 0)
        #expect(model.frame?.id == fixture.frames[1][0].id && model.image != nil && model.issue == nil)
        let peakAfter = await gate.peak, activeAfter = await gate.active
        #expect(peakAfter == 1 && activeAfter == 0)
        model.close()
    }

    @Test func sourceIntegrityErrorsAndEmptyHistoryKeepProvenanceHonest() async throws {
        let fixture = try CorrectionPreludeReviewFixture(); defer { fixture.remove() }
        let model = CorrectionPreludeReviewModel()
        model.open(reference: fixture.reference, directory: fixture.directory); await model.waitForIdle()
        #expect(model.image != nil && model.issue == nil)
        let path = fixture.directory.appendingPathComponent(fixture.reference.path), bytes = try Data(contentsOf: path)
        try Data("changed".utf8).write(to: path)
        model.retry(); await model.waitForIdle()
        #expect(model.prelude == nil && model.image == nil && model.issue?.contains("integrity") == true)
        try bytes.write(to: path); model.retry(); await model.waitForIdle()
        #expect(model.image != nil && model.issue == nil)
        let empty = try CorrectionPreludeReviewFixture(empty: true); defer { empty.remove() }
        model.open(reference: empty.reference, directory: empty.directory); await model.waitForIdle()
        #expect(model.count == 0 && model.image == nil && model.issue == nil && !model.loading)
        #expect(model.prelude?.sourceRunID == empty.sourceRunID && model.prelude?.continuityProven == false)
        model.close()
    }
}

// Only this owned, generated preview is rendered; no screen or accessibility read.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_CORRECTION_PRELUDE_RENDER_DIR"] != nil))
@MainActor func renderCorrectionPreludeReview() async throws {
    let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_CORRECTION_PRELUDE_RENDER_DIR"]))
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    NSApplication.shared.setActivationPolicy(.prohibited)
    let fixture = try CorrectionPreludeReviewFixture(); defer { fixture.remove() }
    for scheme in [ColorScheme.light, .dark] {
        let size = NSSize(width: 760, height: 720), bounds = NSRect(x: 0, y: 0, width: 760, height: 720)
        let view = CorrectionPreludeReview(reference: fixture.reference, directory: fixture.directory)
            .padding(24).frame(width: size.width, height: size.height, alignment: .top)
            .environment(\.colorScheme, scheme).preferredColorScheme(scheme)
            .background(Color(nsColor: .windowBackgroundColor))
        let hosting = NSHostingView(rootView: view); hosting.sizingOptions = []
        let window = NSWindow(contentRect: NSRect(x: -20_000, y: -20_000, width: size.width, height: size.height),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let appearance = try #require(NSAppearance(named: scheme == .dark ? .darkAqua : .aqua))
        window.appearance = appearance; hosting.appearance = appearance; window.contentView = hosting
        hosting.frame = bounds; window.setContentSize(size); window.orderFront(nil)
        defer { window.orderOut(nil); window.contentView = nil; window.close() }
        try await Task.sleep(for: .milliseconds(500)); hosting.layoutSubtreeIfNeeded(); hosting.displayIfNeeded()
        let bitmap = try #require(hosting.bitmapImageRepForCachingDisplay(in: bounds)); hosting.cacheDisplay(in: bounds, to: bitmap)
        var sourcePixels = 0
        for y in stride(from: 0, to: bitmap.pixelsHigh, by: 8) {
            for x in stride(from: 0, to: bitmap.pixelsWide, by: 8) {
                if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
                   (0.55...0.7).contains(color.redComponent), (0.2...0.55).contains(color.greenComponent),
                   (0.15...0.4).contains(color.blueComponent) { sourcePixels += 1 }
            }
        }
        #expect(sourcePixels > 100, "The retained original source image must actually appear in the native preview")
        let name = scheme == .dark ? "correction-prelude-dark.png" : "correction-prelude-light.png"
        try #require(bitmap.representation(using: .png, properties: [:])).write(to: directory.appendingPathComponent(name))
    }
}
