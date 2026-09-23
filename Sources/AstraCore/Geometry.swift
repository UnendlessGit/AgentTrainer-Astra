import Foundation

public struct Point2D: Codable, Hashable, Sendable {
    public var x: Double
    public var y: Double
    public init(x: Double, y: Double) { self.x = x; self.y = y }
    public static let zero = Point2D(x: 0, y: 0)
    public var isFinite: Bool { x.isFinite && y.isFinite }
}

public struct Rect2D: Codable, Hashable, Sendable {
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x; self.y = y; self.width = width; self.height = height
    }

    public var isValid: Bool {
        x.isFinite && y.isFinite && width.isFinite && height.isFinite
            && width > 0 && height > 0 && (x + width).isFinite && (y + height).isFinite
    }

    public func contains(_ point: Point2D) -> Bool {
        isValid && point.isFinite && point.x >= x && point.x <= x + width
            && point.y >= y && point.y <= y + height
    }

    public func globalPoint(normalized point: Point2D) throws -> Point2D {
        guard isValid, point.isFinite, (0...1).contains(point.x), (0...1).contains(point.y) else {
            throw AstraError("geometry.invalidPoint", "The pointer target is outside the observed surface.")
        }
        return Point2D(x: x + point.x * width, y: y + point.y * height)
    }

    public func normalizedPoint(global point: Point2D) throws -> Point2D {
        guard contains(point) else {
            throw AstraError("geometry.outsideSurface", "The pointer does not belong to this surface.")
        }
        return Point2D(x: (point.x - x) / width, y: (point.y - y) / height)
    }
}

public struct SurfaceDescriptor: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var globalBounds: Rect2D
    public var pixelWidth: Int
    public var pixelHeight: Int
    public var contentBounds: Rect2D
    public var geometryRevision: UInt64
    /// Native recipient identity, independent of the model's ordered surface role.
    public var nativeWindowID: UInt32?
    public var nativeDisplayID: UInt32?

    public init(id: String, globalBounds: Rect2D, pixelWidth: Int, pixelHeight: Int,
                contentBounds: Rect2D? = nil, geometryRevision: UInt64 = 0,
                nativeWindowID: UInt32? = nil, nativeDisplayID: UInt32? = nil) {
        self.id = id; self.globalBounds = globalBounds
        self.pixelWidth = pixelWidth; self.pixelHeight = pixelHeight
        self.contentBounds = contentBounds ?? Rect2D(x: 0, y: 0, width: Double(pixelWidth), height: Double(pixelHeight))
        self.geometryRevision = geometryRevision
        self.nativeWindowID = nativeWindowID; self.nativeDisplayID = nativeDisplayID
    }

    public func validated() throws -> Self {
        guard !id.isEmpty, id.utf8.count <= 256, globalBounds.isValid, contentBounds.isValid,
              (1...32_768).contains(pixelWidth), (1...32_768).contains(pixelHeight),
              contentBounds.x >= 0, contentBounds.y >= 0,
              nativeWindowID.map({ $0 > 0 }) ?? true, nativeDisplayID.map({ $0 > 0 }) ?? true,
              nativeWindowID == nil || nativeDisplayID == nil,
              contentBounds.x + contentBounds.width <= Double(pixelWidth),
              contentBounds.y + contentBounds.height <= Double(pixelHeight) else {
            throw AstraError("geometry.invalidSurface", "The capture surface has invalid dimensions or bounds.")
        }
        return self
    }

    public func pixelPoint(global point: Point2D) throws -> Point2D {
        _ = try validated()
        let normalized = try globalBounds.normalizedPoint(global: point)
        return Point2D(x: contentBounds.x + normalized.x * contentBounds.width,
                       y: contentBounds.y + normalized.y * contentBounds.height)
    }
}
