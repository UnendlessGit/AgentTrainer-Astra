import AppKit
import ApplicationServices
import CoreMedia
import AstraCore
import Darwin

public enum MonotonicClock {
    private static let timebase: (UInt64, UInt64) = {
        var value = mach_timebase_info_data_t()
        mach_timebase_info(&value)
        return (UInt64(value.numer), UInt64(value.denom))
    }()
    public static var now: UInt64 { nanoseconds(fromMachTicks: mach_absolute_time()) }
    public static func nanoseconds(fromMachTicks ticks: UInt64) -> UInt64 {
        let (numerator, denominator) = timebase
        return (ticks / denominator) * numerator + (ticks % denominator) * numerator / denominator
    }
    public static func nanoseconds(hostTime: CMTime) -> UInt64? {
        guard hostTime.isNumeric, hostTime.value >= 0 else { return nil }
        return nanoseconds(fromMachTicks: CMClockConvertHostTimeToSystemUnits(hostTime))
    }
}

public struct PermissionSnapshot: Codable, Sendable {
    public let screenRecording: Bool
    public let inputMonitoring: Bool
    public let accessibility: Bool
    public let eventPosting: Bool
    public static func current() -> Self {
        Self(screenRecording: CGPreflightScreenCaptureAccess(), inputMonitoring: CGPreflightListenEventAccess(),
             accessibility: AXIsProcessTrusted(), eventPosting: CGPreflightPostEventAccess())
    }
}

public enum PrivacyPane: String, Sendable {
    case screenRecording = "Privacy_ScreenCapture"
    case inputMonitoring = "Privacy_ListenEvent"
    case accessibility = "Privacy_Accessibility"
    @MainActor public func open() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(rawValue)") {
            NSWorkspace.shared.open(url)
        }
    }
}

extension Rect2D {
    public init(_ rect: CGRect) { self.init(x: rect.origin.x, y: rect.origin.y, width: rect.width, height: rect.height) }
    public var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}
extension Point2D {
    public init(_ point: CGPoint) { self.init(x: point.x, y: point.y) }
    public var cgPoint: CGPoint { CGPoint(x: x, y: y) }
}
