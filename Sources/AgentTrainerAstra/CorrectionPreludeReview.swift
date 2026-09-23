import SwiftUI
import AppKit
import AstraCore

struct CorrectionPreludeReviewSource: Hashable, Sendable {
    let reference: CorrectionReference
    let directory: URL
}

struct CorrectionPreludeReviewDependencies: Sendable {
    var loadPrelude: @Sendable (CorrectionPreludeReviewSource) async throws -> CorrectionPrelude
    var loadPixels: @Sendable (CorrectionPrelude, CorrectionPrelude.Frame, URL) async throws -> Data
    static let live = Self(loadPrelude: { source in
        try await Task.detached(priority: .userInitiated) {
            try CorrectionPrelude.load(in: source.directory, reference: source.reference)
        }.value
    }, loadPixels: { prelude, frame, directory in
        try await Task.detached(priority: .userInitiated) { try prelude.pixels(for: frame, in: directory) }.value
    })
}

/// One metadata/image operation at a time, even across close/reopen. Scrubbing
/// replaces only the desired observation; obsolete reads finish without display.
@MainActor @Observable final class CorrectionPreludeReviewModel {
    private(set) var prelude: CorrectionPrelude?
    private(set) var selectedIndex = 0
    private(set) var selectedSurface = 0
    private(set) var image: CGImage?
    private(set) var frame: FrameMetadata?
    private(set) var issue: String?
    private(set) var loading = false
    private let dependencies: CorrectionPreludeReviewDependencies
    private var source: CorrectionPreludeReviewSource?
    private var sourceGeneration = UUID()
    private var imageGeneration = UUID()
    private var work: Task<Void, Never>?

    init(dependencies: CorrectionPreludeReviewDependencies = .live) { self.dependencies = dependencies }
    var count: Int { prelude?.observations.count ?? 0 }
    var observation: CorrectionPrelude.Observation? {
        guard let prelude, prelude.observations.indices.contains(selectedIndex) else { return nil }
        return prelude.observations[selectedIndex]
    }
    var surfaces: [SurfaceDescriptor] { observation?.frames.map(\.block.metadata.surface) ?? [] }
    var cutoffNanos: UInt64? { observation?.actorInput.fields?["cutoffNanos"]?.uint64 }
    var retainedSeconds: Double {
        guard let first = prelude?.observations.first?.actorInput.fields?["cutoffNanos"]?.uint64,
              let last = prelude?.observations.last?.actorInput.fields?["cutoffNanos"]?.uint64, last >= first else { return 0 }
        return Double(last - first) / 1e9
    }
    var secondsBeforeRequest: Double? {
        guard let prelude, let cutoffNanos, cutoffNanos <= prelude.requestedAtNanos else { return nil }
        return Double(prelude.requestedAtNanos - cutoffNanos) / 1e9
    }
    var selectedImageAgeMS: Double? {
        guard let frame, let cutoffNanos, frame.eventNanos <= cutoffNanos else { return nil }
        return Double(cutoffNanos - frame.eventNanos) / 1e6
    }

    func open(reference: CorrectionReference, directory: URL) {
        source = .init(reference: reference, directory: directory)
        sourceGeneration = UUID(); imageGeneration = UUID()
        prelude = nil; selectedIndex = 0; selectedSurface = 0
        image = nil; frame = nil; issue = nil; loading = true
        beginWork()
    }
    func select(_ index: Int) {
        guard (0..<count).contains(index), selectedIndex != index else { return }
        selectedIndex = index; requestImage()
    }
    func selectSurface(_ index: Int) {
        guard surfaces.indices.contains(index), selectedSurface != index else { return }
        selectedSurface = index; requestImage()
    }
    func retry() {
        guard let source else { return }
        open(reference: source.reference, directory: source.directory)
    }
    func close() {
        source = nil; sourceGeneration = UUID(); imageGeneration = UUID()
        prelude = nil; image = nil; frame = nil; issue = nil; loading = false
        // Cancellation cannot join a file read. Keep this serial owner until
        // the current operation returns; a rapid reopen shares the same owner.
    }
    func waitForIdle() async { await work?.value }

    private func requestImage() {
        imageGeneration = UUID(); image = nil; frame = nil; issue = nil; loading = true
        beginWork()
    }
    private func beginWork() {
        guard work == nil else { return }
        work = Task { [weak self] in
            guard let self else { return }
            defer { work = nil; loading = false }
            while let desired = source {
                let sourceToken = sourceGeneration
                if prelude == nil {
                    do {
                        let loaded = try await dependencies.loadPrelude(desired)
                        guard sourceGeneration == sourceToken, source != nil else { continue }
                        prelude = loaded
                        selectedIndex = max(0, loaded.observations.count - 1); selectedSurface = 0
                    } catch {
                        guard sourceGeneration == sourceToken, source != nil else { continue }
                        issue = error.localizedDescription; return
                    }
                }
                guard let prelude, let observation else { return }
                let imageToken = imageGeneration
                guard observation.frames.indices.contains(selectedSurface) else {
                    issue = "The selected observation does not contain this surface."; return
                }
                let selected = observation.frames[selectedSurface]
                do {
                    // Coalesce a scrub before allocating the next decoded image.
                    try await Task.sleep(for: .milliseconds(30))
                    guard sourceGeneration == sourceToken, imageGeneration == imageToken, source != nil else { continue }
                    let pixels = try await dependencies.loadPixels(prelude, selected, desired.directory)
                    guard sourceGeneration == sourceToken, imageGeneration == imageToken, source != nil else { continue }
                    image = try Self.makeImage(pixels: pixels, metadata: selected.block.metadata)
                    frame = selected.block.metadata; issue = nil; return
                } catch {
                    guard sourceGeneration == sourceToken, imageGeneration == imageToken, source != nil else { continue }
                    issue = error.localizedDescription; image = nil; frame = nil; return
                }
            }
        }
    }
    private static func makeImage(pixels: Data, metadata: FrameMetadata) throws -> CGImage {
        let surface = try metadata.validated().surface
        guard pixels.count == metadata.byteCount,
              let provider = CGDataProvider(data: pixels as CFData), let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: surface.pixelWidth, height: surface.pixelHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: surface.pixelWidth * 4, space: space,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).union(.byteOrder32Little),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw AstraError("correction.preview", "The original correction frame could not be displayed.")
        }
        return image
    }
}

/// Embedded in a recording inspector; it cannot create labels or training ranges.
struct CorrectionPreludeReview: View {
    let reference: CorrectionReference
    let directory: URL
    @State private var model = CorrectionPreludeReviewModel()

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Text("These earlier agent observations are for review only. Your demonstration starts after the handoff, as a separate training sequence.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if let prelude = model.prelude {
                    history(prelude)
                    provenance(prelude)
                } else if let issue = model.issue {
                    failure(issue)
                } else {
                    ProgressView("Loading the correction history…").padding(.vertical, 12)
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(6)
        } label: {
            HStack {
                Label("Before the correction", systemImage: "clock.arrow.circlepath")
                Spacer()
                Text("Review only").font(.caption).foregroundStyle(.secondary)
            }
        }
        .task(id: CorrectionPreludeReviewSource(reference: reference, directory: directory)) {
            model.open(reference: reference, directory: directory)
        }
        .onDisappear { model.close() }
    }

    @ViewBuilder private func history(_ prelude: CorrectionPrelude) -> some View {
        if model.count == 0 {
            ContentUnavailableView("No earlier observations retained", systemImage: "clock.badge.exclamationmark",
                description: Text("The correction still records its source and handoff below. Its human demonstration can be reviewed separately."))
                .frame(minHeight: 130)
        } else {
            HStack {
                Text("\(model.count.formatted()) observations · \(model.retainedSeconds.formatted(.number.precision(.fractionLength(3)))) s retained")
                Spacer()
                Text("Observation \(model.selectedIndex + 1) of \(model.count)").monospacedDigit()
            }.font(.caption).foregroundStyle(.secondary)
            if model.surfaces.count > 1 {
                Picker("Observed surface", selection: Binding(get: { model.selectedSurface }, set: { model.selectSurface($0) })) {
                    ForEach(Array(model.surfaces.enumerated()), id: \.element.id) { index, surface in
                        Text("Surface \(index + 1) · \(surface.id)").tag(index)
                    }
                }.pickerStyle(.menu)
            }
            ZStack {
                RoundedRectangle(cornerRadius: 8).fill(.quaternary)
                if let image = model.image {
                    Image(image, scale: 1, label: Text("Original agent observation \(model.selectedIndex + 1), surface \(model.selectedSurface + 1)"))
                        .resizable().interpolation(.none).scaledToFit().padding(1)
                } else if let issue = model.issue {
                    failure(issue)
                }
                if model.loading { ProgressView("Loading observation…") }
            }.frame(height: 230).clipShape(RoundedRectangle(cornerRadius: 8))
            HStack(spacing: 12) {
                Button { model.select(model.selectedIndex - 1) } label: { Image(systemName: "backward.frame") }
                    .disabled(model.selectedIndex == 0).help("Previous retained observation")
                    .accessibilityLabel("Previous retained observation")
                Slider(value: Binding(get: { Double(model.selectedIndex) }, set: { model.select(Int($0.rounded())) }),
                       in: 0...Double(max(1, model.count - 1)), step: 1)
                    .disabled(model.count < 2).accessibilityLabel("Correction history position")
                    .accessibilityValue("Observation \(model.selectedIndex + 1) of \(model.count)")
                Button { model.select(model.selectedIndex + 1) } label: { Image(systemName: "forward.frame") }
                    .disabled(model.selectedIndex + 1 >= model.count).help("Next retained observation")
                    .accessibilityLabel("Next retained observation")
                if let seconds = model.secondsBeforeRequest {
                    Text("−\(seconds.formatted(.number.precision(.fractionLength(3)))) s")
                        .font(.callout.monospacedDigit()).frame(minWidth: 70, alignment: .trailing)
                        .accessibilityLabel("\(seconds.formatted(.number.precision(.fractionLength(3)))) seconds before the correction request")
                }
            }
            Text("Times are relative to the correction request. Each surface retains its original capture time.")
                .font(.caption).foregroundStyle(.secondary)
            if let frame = model.frame, let age = model.selectedImageAgeMS {
                Text("\(frame.surface.pixelWidth) × \(frame.surface.pixelHeight) · Frame captured \(age.formatted(.number.precision(.fractionLength(1)))) ms before this observation")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func provenance(_ prelude: CorrectionPrelude) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            LabeledContent("Source checkpoint") { Text(prelude.sourceCheckpointID.uuidString).font(.caption.monospaced()) }
            LabeledContent("Source run") { Text(prelude.sourceRunID.uuidString).font(.caption.monospaced()) }
            HStack(alignment: .firstTextBaseline, spacing: 18) {
                Text("Controls released \(offset(prelude.controlJoinedAtNanos, from: prelude.requestedAtNanos))")
                Text("Demonstration starts \(offset(prelude.supervisionStartNanos, from: prelude.requestedAtNanos))")
            }.foregroundStyle(.secondary)
            Text("Continuity across this handoff is unverified. Earlier agent actions are not treated as your demonstration.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.font(.caption).textSelection(.enabled)
    }
    private func failure(_ issue: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Correction preview unavailable", systemImage: "exclamationmark.triangle").font(.headline)
            Text(issue).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
            Button("Retry Preview") { model.retry() }
        }.padding(12)
    }
    private func offset(_ value: UInt64, from origin: UInt64) -> String {
        let positive = value >= origin, difference = positive ? value - origin : origin - value
        return (positive ? "+" : "−") + (Double(difference) / 1e9).formatted(.number.precision(.fractionLength(3))) + " s"
    }
}
