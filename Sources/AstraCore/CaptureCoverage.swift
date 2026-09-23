import Foundation

/// Evidence about one immutable source frame. Availability is distinct from
/// source time: a delayed callback cannot prove that pixels stayed unchanged
/// until callback arrival. Silence never advances this evidence.
public struct CaptureFrameCoverage: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable { case frame, unchanged }
    public let streamID: UUID
    public let frameID: UUID
    public let surface: SurfaceDescriptor
    public let eventNanos: UInt64
    public let observedNanos: UInt64
    public let throughNanos: UInt64
    public let verifiedAtNanos: UInt64
    public let kind: Kind

    public init(streamID: UUID, frame: FrameMetadata) throws {
        _ = try frame.validated()
        guard frame.eventNanos <= frame.observedNanos else {
            throw AstraError("capture.sourceClock", "The capture frame is unavailable at its declared observation time.")
        }
        self.streamID = streamID; frameID = frame.id; surface = frame.surface
        eventNanos = frame.eventNanos; observedNanos = frame.observedNanos
        throughNanos = frame.observedNanos; verifiedAtNanos = frame.observedNanos; kind = .frame
    }

    private init(previous: Self, through: UInt64, verifiedAt: UInt64) {
        streamID = previous.streamID; frameID = previous.frameID; surface = previous.surface
        eventNanos = previous.eventNanos; observedNanos = previous.observedNanos
        throughNanos = through; verifiedAtNanos = verifiedAt; kind = .unchanged
    }

    /// Only an explicit unchanged-source event with matching geometry can call
    /// this. Missing geometry/timestamps do not authorize an extension.
    public func verifyingUnchanged(streamID: UUID, surface: SurfaceDescriptor,
                                   throughNanos: UInt64, verifiedAtNanos: UInt64) throws -> Self {
        guard streamID == self.streamID, surface == self.surface,
              throughNanos >= self.throughNanos, verifiedAtNanos >= self.verifiedAtNanos,
              throughNanos <= verifiedAtNanos else {
            throw AstraError("capture.coverage", "Unchanged-frame evidence has a different source, geometry or invalid clock.")
        }
        return Self(previous: self, through: throughNanos, verifiedAt: verifiedAtNanos)
    }

    @discardableResult
    public func validated(frame: FrameMetadata, cutoffNanos: UInt64, maximumAgeMS: Int? = nil) throws -> Self {
        _ = try frame.validated()
        guard frame.id == frameID, frame.surface == surface, frame.eventNanos == eventNanos,
              frame.observedNanos == observedNanos, eventNanos <= observedNanos,
              observedNanos <= throughNanos, throughNanos <= verifiedAtNanos, verifiedAtNanos <= cutoffNanos,
              kind != .frame || (throughNanos == observedNanos && verifiedAtNanos == observedNanos) else {
            throw AstraError("capture.coverageBinding", "Visual coverage does not belong to the available source frame.")
        }
        if let maximumAgeMS {
            guard (1...60_000).contains(maximumAgeMS),
                  cutoffNanos - (kind == .frame ? eventNanos : throughNanos) <= UInt64(maximumAgeMS) * 1_000_000 else {
                throw AstraError("capture.stale", "The capture stream has not provided recent pixels or verified unchanged-source evidence.")
            }
        }
        return self
    }
}
