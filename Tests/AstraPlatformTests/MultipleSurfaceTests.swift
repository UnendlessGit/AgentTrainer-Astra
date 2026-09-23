import Foundation
import Testing
import AstraCore
@testable import AstraPlatform

@Test func multiSurfaceRoutingRetainsRecipientAndIndependentGeometry() throws {
    let bounds = Rect2D(x: 0, y: 0, width: 100, height: 80), launched = Date(timeIntervalSince1970: 100)
    let leaves = [11, 12].map { id in CaptureSource(id: "window:\(id)", name: "Window \(id)", kind: .window,
        windowID: UInt32(id), applicationPID: 42, applicationLaunchDate: launched,
        bounds: bounds, pixelWidth: 200, pixelHeight: 160) }
    let source = CaptureSource(id: "application:42", name: "Application", kind: .application, applicationPID: 42,
        applicationLaunchDate: launched, bounds: bounds, pixelWidth: 200, pixelHeight: 160, bindings: leaves)
    let snapshot = InputScopeSnapshot(observedNanos: 100, frontmostPID: 42, applicationLaunchDate: launched,
        windows: [.init(id: 12, pid: 42, layer: 0, bounds: bounds), .init(id: 11, pid: 42, layer: 0, bounds: bounds)])
    try snapshot.verifyTarget(source)
    let event = RawInputEvent(sequence: 0, eventNanos: 100, observedNanos: 100, origin: .physical,
        kind: .pointer, x: 50, y: 40)
    #expect(try snapshot.routedSurfaceID(event, handlingWindowID: 11, source: source) == "window:11")
    #expect(try snapshot.routedSurfaceID(event, handlingWindowID: nil, source: source) == nil)
    #expect(try snapshot.routedSurfaceID(event, handlingWindowID: 99, source: source) == nil)
    var surfaces = leaves.map { $0.surfaceDescriptor() }
    surfaces[0].geometryRevision = 7; surfaces[1].geometryRevision = 2
    let scope = try ControlScope(surfaces: surfaces, applicationPID: 42, geometryRevision: 91).validated()
    try NativeSurfaceRouting.verifyWindowRecipient(12, scope: scope, requestedSurfaceID: "window:12")
    #expect(throws: AstraError.self) { try NativeSurfaceRouting.verifyWindowRecipient(12, scope: scope, requestedSurfaceID: "window:11") }
    let changed = InputScopeSnapshot(observedNanos: 110, frontmostPID: 42, applicationLaunchDate: launched,
        windows: [.init(id: 12, pid: 42, layer: 0, bounds: bounds)])
    #expect(throws: AstraError.self) { try changed.verifyBindings(source) }
    #expect(scope.surfaces.map(\.geometryRevision) == [7, 2])
}

@Test func multiSurfaceDisplayRoutingUsesHalfOpenOwnership() throws {
    let left = CaptureSource(id: "display:1", name: "Left", kind: .display, displayID: 1,
        bounds: .init(x: -100, y: 0, width: 100, height: 80), pixelWidth: 100, pixelHeight: 80)
    let right = CaptureSource(id: "display:2", name: "Right", kind: .display, displayID: 2,
        bounds: .init(x: 0, y: 0, width: 100, height: 80), pixelWidth: 100, pixelHeight: 80)
    let desktop = CaptureSource(id: "desktop", name: "Desktop", kind: .desktop,
        bounds: .init(x: -100, y: 0, width: 200, height: 80), pixelWidth: 100, pixelHeight: 80, bindings: [left, right])
    let snapshot = InputScopeSnapshot(observedNanos: 100, frontmostPID: nil, applicationLaunchDate: nil,
        windows: [], displays: [1: left.bounds, 2: right.bounds])
    try snapshot.verifyBindings(desktop)
    let event = RawInputEvent(sequence: 0, eventNanos: 100, observedNanos: 100, origin: .physical, kind: .pointer, x: 0, y: 40)
    #expect(try snapshot.routedSurfaceID(event, handlingWindowID: nil, source: desktop) == "display:2")
    let changed = InputScopeSnapshot(observedNanos: 110, frontmostPID: nil, applicationLaunchDate: nil,
        windows: [], displays: [1: left.bounds])
    #expect(throws: AstraError.self) { try changed.verifyBindings(desktop) }
}
