import Testing
import CoreGraphics
import IOKit.hidsystem
@testable import AstraPlatform

@Test func modifierEdgesUseCapturedSideBitsInsteadOfLaterHardwareState() {
    let both = CGEventFlags(rawValue: CGEventFlags.maskShift.rawValue | UInt64(NX_DEVICELSHIFTKEYMASK | NX_DEVICERSHIFTKEYMASK))
    let rightOnly = CGEventFlags(rawValue: CGEventFlags.maskShift.rawValue | UInt64(NX_DEVICERSHIFTKEYMASK))
    #expect(PhysicalInputMonitor.modifierState(keyCode: 56, flags: both) == true)
    #expect(PhysicalInputMonitor.modifierState(keyCode: 56, flags: rightOnly) == false)
    #expect(PhysicalInputMonitor.modifierState(keyCode: 60, flags: rightOnly) == true)
    #expect(PhysicalInputMonitor.modifierState(keyCode: 60, flags: []) == false)
    #expect(PhysicalInputMonitor.modifierState(keyCode: 56, flags: .maskShift) == nil)
    let control = CGEventFlags(rawValue: CGEventFlags.maskControl.rawValue | UInt64(NX_DEVICERCTLKEYMASK))
    #expect(PhysicalInputMonitor.modifierState(keyCode: 62, flags: control) == true)
    #expect(PhysicalInputMonitor.modifierState(keyCode: 59, flags: control) == false)
}
