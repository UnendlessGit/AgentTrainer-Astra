import Foundation
import Testing
import AstraCore
@testable import AgentTrainerAstra

@Test func selectionEditorPreservesUntouchedNanosecondsAndRejectsNonFiniteTimes() throws {
    let range = RecordingTimeRange(startNanos: 9_007_199_254_740_993, endNanos: 9_007_199_255_740_995)
    var draft = RecordingRangeDraft(range, origin: 1)
    #expect(try draft.range(origin: 1) == range)
    draft.start = .infinity
    #expect(throws: AstraError.self) { _ = try draft.range(origin: 1) }
    draft.start = -1
    #expect(throws: AstraError.self) { _ = try draft.range(origin: 1) }
}
