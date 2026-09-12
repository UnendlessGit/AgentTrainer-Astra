import Foundation
import Carbon.HIToolbox
import AstraCore

public struct KeyboardObservationProof: Sendable {
    public let trusted: Bool
    public let secureInputEnabled: Bool?
    public let sampledNanos: UInt64?
    public let observedNanos: UInt64
    public let interruptionGeneration: UInt64

    public func requireTrusted() throws {
        guard trusted else {
            throw AstraError(secureInputEnabled == true ? "input.secureInput" : "input.keyboardTrust",
                secureInputEnabled == true ? "macOS Secure Input prevents reliable keyboard observation."
                    : "Keyboard observation could not be verified recently enough.")
        }
    }
}

/// The Carbon SDK declares IsSecureEventInputEnabled not thread-safe. Every
/// query goes through the main thread. Scheduler/watchdog reads only take a
/// short lock over cached values; they never wait for the main queue or an OS API.
public final class KeyboardObservationTrust: @unchecked Sendable {
    public static let shared = KeyboardObservationTrust()
    private let lock = NSLock()
    private let probe: @MainActor @Sendable () -> Bool
    private let clock: @Sendable () -> UInt64
    private let freshnessNanos: UInt64
    private var sampledNanos: UInt64?
    private var secureInput: Bool?
    private var unavailable = true
    private var generation: UInt64 = 0
    private var timer: DispatchSourceTimer?

    public init(freshnessNanos: UInt64 = 150_000_000, automaticSampling: Bool = true,
                clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now },
                probe: @escaping @MainActor @Sendable () -> Bool = { IsSecureEventInputEnabled() }) {
        precondition(freshnessNanos > 0)
        self.clock = clock; self.probe = probe; self.freshnessNanos = freshnessNanos
        if automaticSampling {
            let timer = DispatchSource.makeTimerSource(queue: .main)
            timer.schedule(deadline: .now(), repeating: .milliseconds(20), leeway: .milliseconds(2))
            timer.setEventHandler { @Sendable [weak self] in
                MainActor.assumeIsolated { _ = self?.refreshOnMainThread() }
            }
            self.timer = timer; timer.resume()
        }
    }
    deinit { timer?.cancel() }

    public func snapshot() -> KeyboardObservationProof {
        lock.withLock { proof(at: clock()) }
    }

    /// Cold arming and cleanup may obtain fresh proof. Do not call this from a
    /// deadline/event-tap/watchdog callback. The guardian's own main-thread
    /// recovery loop executes directly, without a self-dispatch deadlock.
    public func refreshSynchronously() -> KeyboardObservationProof {
        if Thread.isMainThread { return MainActor.assumeIsolated { refreshOnMainThread() } }
        return DispatchQueue.main.sync { MainActor.assumeIsolated { refreshOnMainThread() } }
    }

    @MainActor private func refreshOnMainThread() -> KeyboardObservationProof {
        let began = clock(), secure = probe()
        return lock.withLock {
            expire(at: began)
            if secure, !unavailable { generation &+= 1 }
            unavailable = secure; secureInput = secure; sampledNanos = began
            return proof(at: clock())
        }
    }
    private func expire(at now: UInt64) {
        if let sampledNanos, (now < sampledNanos || now - sampledNanos >= freshnessNanos), !unavailable {
            generation &+= 1; unavailable = true
        }
    }
    private func proof(at now: UInt64) -> KeyboardObservationProof {
        expire(at: now)
        return KeyboardObservationProof(trusted: sampledNanos != nil && secureInput == false && !unavailable,
            secureInputEnabled: secureInput, sampledNanos: sampledNanos, observedNanos: now, interruptionGeneration: generation)
    }
}

/// Per-recording continuity survives a secure interval that starts and ends
/// between delivery callbacks. The exclusion begins at this consumer's last
/// trusted sample, not at the later detection time.
struct KeyboardObservationContinuity: Sendable {
    private let generation: UInt64
    private(set) var lastTrustedNanos: UInt64
    private var failed = false
    init(_ proof: KeyboardObservationProof) throws {
        try proof.requireTrusted()
        generation = proof.interruptionGeneration; lastTrustedNanos = proof.sampledNanos!
    }
    mutating func inspect(_ proof: KeyboardObservationProof) -> InputObservationBoundary? {
        guard !failed else { return nil }
        if !proof.trusted || proof.interruptionGeneration != generation {
            failed = true
            return InputObservationBoundary(invalidFromNanos: lastTrustedNanos, observedNanos: proof.observedNanos,
                message: "Keyboard observation was interrupted by macOS Secure Input or unavailable verification. The interval after the last trusted sample was excluded.")
        }
        lastTrustedNanos = proof.sampledNanos!
        return nil
    }
}
