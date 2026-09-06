import Foundation

final class AsyncCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var completed = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        await withCheckedContinuation { continuation in
            let ready = lock.withLock {
                if completed { return true }
                waiters.append(continuation)
                return false
            }
            if ready { continuation.resume() }
        }
    }

    func finish() {
        let pending = lock.withLock {
            completed = true
            let pending = waiters
            waiters.removeAll()
            return pending
        }
        pending.forEach { $0.resume() }
    }
}
