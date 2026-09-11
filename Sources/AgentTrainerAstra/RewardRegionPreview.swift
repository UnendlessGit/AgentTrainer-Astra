import SwiftUI
import AstraCore

struct RewardRegionPreview: View {
    let image: NSImage
    let surface: SurfaceDescriptor
    let region: Binding<Rect2D?>?
    var body: some View {
        GeometryReader { geometry in
            let scale = min(geometry.size.width / Double(surface.pixelWidth), geometry.size.height / Double(surface.pixelHeight))
            let width = Double(surface.pixelWidth) * scale, height = Double(surface.pixelHeight) * scale
            let left = (geometry.size.width - width) / 2, top = (geometry.size.height - height) / 2
            ZStack(alignment: .topLeading) {
                Image(nsImage: image).resizable().frame(width: width, height: height).offset(x: left, y: top)
                if let value = region?.wrappedValue, value.isValid {
                    let content = surface.contentBounds
                    Rectangle().fill(Color.accentColor.opacity(0.12)).overlay { Rectangle().stroke(.white, lineWidth: 1).padding(1) }
                        .overlay { Rectangle().stroke(Color.accentColor, lineWidth: 2) }
                        .frame(width: value.width * content.width * scale, height: value.height * content.height * scale)
                        .offset(x: left + (content.x + value.x * content.width) * scale, y: top + (content.y + value.y * content.height) * scale)
                        .allowsHitTesting(false)
                }
            }.frame(width: geometry.size.width, height: geometry.size.height, alignment: .topLeading).contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 3).onChanged { drag in
                    guard let region else { return }
                    let content = surface.contentBounds
                    func point(_ location: CGPoint) -> Point2D {
                        .init(x: min(1, max(0, ((location.x - left) / scale - content.x) / content.width)),
                              y: min(1, max(0, ((location.y - top) / scale - content.y) / content.height)))
                    }
                    let first = point(drag.startLocation), last = point(drag.location)
                    let value = Rect2D(x: min(first.x, last.x), y: min(first.y, last.y), width: abs(first.x - last.x), height: abs(first.y - last.y))
                    if value.width * content.width >= 1 && value.height * content.height >= 1 { region.wrappedValue = value }
                })
        }.frame(height: 210)
            .accessibilityLabel(region == nil ? "Recorded frame" : "Reward region in recorded frame")
            .accessibilityHint(region == nil ? "" : "Use the precise region fields below to adjust this region with the keyboard.")
    }
}
