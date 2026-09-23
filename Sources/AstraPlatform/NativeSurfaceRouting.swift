import Foundation
import AstraCore

enum NativeSurfaceRouting {
    static func verifyWindowRecipient(_ windowID: UInt32, scope: ControlScope, requestedSurfaceID: String?) throws {
        let candidates = scope.surfaces.filter { surface in
            let nativeID = surface.nativeWindowID ?? (surface.id.hasPrefix("window:") ? UInt32(surface.id.dropFirst(7)) : nil)
            return nativeID == windowID && (requestedSurfaceID == nil || requestedSurfaceID == surface.id)
        }
        guard candidates.count == 1 else {
            throw AstraError("control.surfaceRecipient", "The requested surface is covered by another window or the input recipient is not observed.")
        }
    }
}
