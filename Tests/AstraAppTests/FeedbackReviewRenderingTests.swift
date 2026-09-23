import AppKit
import SwiftUI
import Testing
@testable import AgentTrainerAstra

/// Own NSHostingView pixels only: no screen/accessibility capture or TCC.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ASTRA_FEEDBACK_RENDER_DIR"] != nil))
@MainActor func renderFeedbackReviewViews() async throws {
    let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ASTRA_FEEDBACK_RENDER_DIR"]), isDirectory: true)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    NSApplication.shared.setActivationPolicy(.prohibited)
    var files: [String] = []
    for state in ["partial", "complete", "missing", "blocked"] {
        for scheme in [ColorScheme.light, .dark] {
            for size in [NSSize(width: 960, height: 760), NSSize(width: 480, height: 740)] {
                let fixture = try FeedbackReviewFixture(longNames: true)
                let model = try fixture.model()
                if state == "missing" { await fixture.harness.invalid("missing") }
                if state == "blocked" { await fixture.harness.reject() }
                await model.open()
                if state == "partial" {
                    try await viewFeedbackInterval(model, index: 0)
                    await model.setCount(packetID: model.selected.target.packetID, ruleID: fixture.positive, count: 2)
                    await model.markReviewed(ruleID: fixture.positive)
                } else if state == "complete" {
                    for index in 0..<3 {
                        try await viewFeedbackInterval(model, index: index)
                        await model.setCount(packetID: model.selected.target.packetID, ruleID: fixture.positive, count: index + 1)
                        await model.markReviewed()
                    }
                } else { try await feedbackEventually { !model.loadingFrames } }
                let name = "feedback-\(state)-\(scheme == .dark ? "dark" : "light")-\(Int(size.width))x\(Int(size.height))"
                files += try await renderFeedback(model, name: name, size: size, scheme: scheme, output: output)
                await model.cancel(); fixture.remove()
            }
        }
    }
    let manifest: [String: Any] = ["generatedSourceOnly": true, "screenCapture": false, "accessibilityReads": false,
                                   "renderer": "owned NSHostingView", "files": files]
    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("manifest.json"), options: .atomic)
}

@MainActor private func renderFeedback(_ model: FeedbackReviewModel, name: String, size: NSSize,
                                      scheme: ColorScheme, output: URL) async throws -> [String] {
    let bounds = NSRect(origin: .zero, size: size)
    let view = FeedbackReviewView(model: model, onFinish: { _ in })
        .environment(\.colorScheme, scheme).preferredColorScheme(scheme)
        .frame(width: size.width, height: size.height).background(Color(nsColor: .windowBackgroundColor))
    let hosting = NSHostingView(rootView: view); hosting.sizingOptions = []
    let window = NSWindow(contentRect: NSRect(x: -20_000, y: -20_000, width: size.width, height: size.height),
                          styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    let appearance = try #require(NSAppearance(named: scheme == .dark ? .darkAqua : .aqua))
    window.appearance = appearance; hosting.appearance = appearance
    window.contentView = hosting; hosting.frame = bounds; window.setContentSize(size); window.orderFront(nil)
    defer { window.orderOut(nil); window.contentView = nil; window.close() }
    try await Task.sleep(for: .milliseconds(160)); hosting.layoutSubtreeIfNeeded(); hosting.displayIfNeeded()
    try await Task.sleep(for: .milliseconds(80)); hosting.layoutSubtreeIfNeeded(); hosting.displayIfNeeded()
    let first = try #require(hosting.bitmapImageRepForCachingDisplay(in: bounds)); hosting.cacheDisplay(in: bounds, to: first)
    if model.image != nil {
        var sourceColorSamples = 0
        for y in stride(from: 0, to: first.pixelsHigh, by: 8) {
            for x in stride(from: 0, to: first.pixelsWide, by: 8) {
                if let color = first.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
                   color.redComponent < 0.4, (0.3...0.7).contains(color.greenComponent), color.blueComponent > 0.5 {
                    sourceColorSamples += 1
                }
            }
        }
        #expect(sourceColorSamples > 100, "The owned original preview must actually render, not just allocate an image")
    }
    let png = try #require(first.representation(using: .png, properties: [:]))
    #expect(png.count > 5000)
    try png.write(to: output.appendingPathComponent(name + ".png"), options: .atomic)
    var files = [name + ".png"]
    if size.width < 700, let scroll = scrolls(hosting).max(by: { extent($0) < extent($1) }), extent(scroll) > 1,
       let content = scroll.documentView {
        let y = content.isFlipped ? content.bounds.maxY - scroll.contentView.bounds.height : content.bounds.minY
        scroll.contentView.scroll(to: .init(x: 0, y: y)); scroll.reflectScrolledClipView(scroll.contentView)
        try await Task.sleep(for: .milliseconds(80)); hosting.layoutSubtreeIfNeeded()
        let last = try #require(hosting.bitmapImageRepForCachingDisplay(in: bounds)); hosting.cacheDisplay(in: bounds, to: last)
        try #require(last.representation(using: .png, properties: [:])).write(to: output.appendingPathComponent(name + "-bottom.png"), options: .atomic)
        files.append(name + "-bottom.png")
    }
    return files
}
@MainActor private func scrolls(_ view: NSView) -> [NSScrollView] {
    (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap { scrolls($0) }
}
@MainActor private func extent(_ view: NSScrollView) -> CGFloat { (view.documentView?.bounds.height ?? 0) - view.contentView.bounds.height }
