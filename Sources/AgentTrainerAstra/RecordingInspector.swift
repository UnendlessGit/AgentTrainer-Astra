import SwiftUI
import AppKit
import AstraCore

private struct PreviewEvent: Identifiable {
    let event: RawInputEvent
    var id: UInt64 { event.sequence }
}

@MainActor @Observable private final class RecordingInspectorModel {
    var inspection: RecordingInspection?
    var preview: RecordingPreview?
    var image: NSImage?
    var issue: String?
    var seconds: Double = 0
    var loading = false
    private var reader: RecordingReader?
    private var generation = UUID()
    private var work: Task<Void, Never>?

    var maximumSeconds: Double {
        guard let first = inspection?.firstFrameNanos, let end = inspection?.manifest.stoppedNanos, end >= first else { return 0 }
        return Double(end - first) / 1e9
    }
    func open(_ directory: URL) async {
        loading = true
        let token = UUID(); generation = token
        do {
            let pair = try await Task.detached {
                let reader = try RecordingReader(directory: directory)
                return (reader, try reader.inspect())
            }.value
            guard generation == token, !Task.isCancelled else { return }
            reader = pair.0; inspection = pair.1
            seek(0)
        } catch {
            if generation == token { issue = error.localizedDescription; loading = false }
        }
    }
    func seek(_ seconds: Double) {
        guard let reader, let first = inspection?.firstFrameNanos else { loading = false; return }
        self.seconds = max(0, min(seconds, maximumSeconds))
        generation = UUID(); loading = true
        guard work == nil else { return }
        work = Task { [weak self] in
            guard let self else { return }
            defer { if !Task.isCancelled { work = nil; loading = false } }
            // Exactly one disk read is in flight. A scrub while it runs changes
            // only the desired time; the next iteration reads the newest one.
            while !Task.isCancelled {
                let token = generation
                let offset = UInt64(self.seconds * 1e9)
                guard first <= UInt64(Int64.max) - offset else { issue = "The selected time exceeds the recording range."; return }
                do {
                    try await Task.sleep(for: .milliseconds(35))
                    if generation != token { continue }
                    let result = try await Task.detached { try reader.preview(at: first + offset) }.value
                    if Task.isCancelled { return }
                    if generation != token { continue }
                    preview = result; image = result.flatMap(Self.previewImage); issue = nil
                    return
                } catch {
                    if Task.isCancelled { return }
                    if generation != token { continue }
                    issue = error.localizedDescription
                    return
                }
            }
        }
    }
    func close() { generation = UUID(); work?.cancel(); work = nil; reader = nil }
    private static func previewImage(_ preview: RecordingPreview) -> NSImage? {
        let surface = preview.frame.surface
        guard let provider = CGDataProvider(data: preview.pixels as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: surface.pixelWidth, height: surface.pixelHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: surface.pixelWidth * 4, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).union(.byteOrder32Little),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return NSImage(cgImage: image, size: NSSize(width: surface.pixelWidth, height: surface.pixelHeight))
    }
}

struct RecordingInspector: View {
    let recording: RecordingManifest
    let directory: URL
    @Environment(\.dismiss) private var dismiss
    @State private var model = RecordingInspectorModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(recording.name).font(.title2.weight(.semibold))
                    Text(recording.environment.name).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            if let issue = recording.issue {
                AttentionLabel(message: issue).font(.callout).textSelection(.enabled)
            }
            ZStack {
                RoundedRectangle(cornerRadius: 10).fill(.quaternary)
                if let image = model.image {
                    Image(nsImage: image).resizable().interpolation(.none).scaledToFit().padding(1)
                        .accessibilityLabel("Recorded screen at \(model.seconds.formatted(.number.precision(.fractionLength(2)))) seconds")
                } else if let issue = model.issue {
                    ContentUnavailableView("Preview unavailable", systemImage: "exclamationmark.triangle", description: Text(issue))
                } else if recording.frameCount == 0 {
                    ContentUnavailableView("No complete frames", systemImage: "photo", description: Text("The captured source has been preserved for inspection."))
                }
                if model.loading { ProgressView().controlSize(.large) }
            }.frame(minHeight: 180, idealHeight: 300, maxHeight: 440).clipShape(RoundedRectangle(cornerRadius: 10))
            HStack(spacing: 12) {
                Button { model.seek(model.seconds - 1.0 / Double(recording.environment.captureFPS)) } label: { Image(systemName: "backward.frame") }
                    .help("Previous capture interval").accessibilityLabel("Previous capture interval")
                Slider(value: Binding(get: { model.seconds }, set: { model.seek($0) }), in: 0...max(model.maximumSeconds, 0.001))
                    .accessibilityLabel("Recording time").disabled(model.inspection?.firstFrameNanos == nil)
                Button { model.seek(model.seconds + 1.0 / Double(recording.environment.captureFPS)) } label: { Image(systemName: "forward.frame") }
                    .help("Next capture interval").accessibilityLabel("Next capture interval")
                Text("\(model.seconds, specifier: "%.2f") / \(model.maximumSeconds, specifier: "%.2f") s")
                    .font(.callout.monospacedDigit()).frame(minWidth: 110, alignment: .trailing)
            }
            if let preview = model.preview {
                HStack {
                    Text("\(preview.frame.surface.pixelWidth) × \(preview.frame.surface.pixelHeight) · Lossless source")
                    Spacer()
                    Text("\(recording.frameCount.formatted()) frames · \(recording.eventCount.formatted()) input events")
                }.font(.caption).foregroundStyle(.secondary)
                Text("Nearby controls").font(.headline)
                Table(preview.events.map { PreviewEvent(event: $0) }) {
                    TableColumn("Time") { row in
                        Text(relativeTime(row.event.observedNanos)).monospacedDigit()
                    }.width(80)
                    TableColumn("Control") { row in Text(eventName(row.event)) }
                    TableColumn("Origin") { row in Text(row.event.origin.rawValue.capitalized) }.width(120)
                }.frame(minHeight: 70, idealHeight: 110, maxHeight: 130)
                if preview.moreEvents { Text("Showing the first 256 events in this interval.").font(.caption).foregroundStyle(.secondary) }
            }
            if let issue = model.issue, model.image != nil { AttentionLabel(message: issue) }
        }.padding(24).frame(minWidth: 730, idealWidth: 880, maxWidth: 1100, minHeight: 560)
            .task { await model.open(directory) }.onDisappear { model.close() }
    }

    private func relativeTime(_ nanos: UInt64) -> String {
        guard let first = model.inspection?.firstFrameNanos else { return "—" }
        let delta = nanos >= first ? Double(nanos - first) : -Double(first - nanos)
        return (delta / 1e9).formatted(.number.precision(.fractionLength(3))) + " s"
    }
    private func eventName(_ event: RawInputEvent) -> String {
        let suffix = event.keyCode.map { " · key \($0)" } ?? event.button.map { " · button \($0)" } ?? ""
        switch event.kind {
        case .keyDown: return "Press" + suffix
        case .keyUp: return "Release" + suffix
        case .keyRepeat: return "Repeat" + suffix
        case .buttonDown: return "Press" + suffix
        case .buttonUp: return "Release" + suffix
        case .pointer: return "Pointer movement"
        case .scroll: return "Scroll"
        case .flags: return "Modifier change" + suffix
        case .gap: return "Input boundary"
        }
    }
}
