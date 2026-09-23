import Foundation

/// One parent-owned review. Shutdown joins playback, source reads and draft
/// publication before its caller can release the actor/control boundary.
@MainActor final class FeedbackReviewPresentation: Identifiable {
    let id = UUID()
    let model: FeedbackReviewModel
    private var result: FeedbackReviewOutcome?
    private var continuation: CheckedContinuation<FeedbackReviewOutcome, Never>?
    init(model: FeedbackReviewModel) { self.model = model }
    func wait() async -> FeedbackReviewOutcome {
        if let result { return result }
        return await withCheckedContinuation { continuation = $0 }
    }
    func finish(_ outcome: FeedbackReviewOutcome) {
        guard result == nil else { return }
        result = outcome; let waiting = continuation; continuation = nil; waiting?.resume(returning: outcome)
    }
    func cancelAndJoin() async {
        await model.cancel()
        if let outcome = model.takeOutcome() { finish(outcome) }
    }
}
