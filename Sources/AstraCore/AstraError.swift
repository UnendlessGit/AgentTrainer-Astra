import Foundation

public struct AstraError: Error, LocalizedError, Codable, Equatable, Sendable {
    public let code: String
    public let message: String
    public let recoverable: Bool

    public init(_ code: String, _ message: String, recoverable: Bool = true) {
        self.code = code
        self.message = message
        self.recoverable = recoverable
    }

    public var errorDescription: String? { message }
}

public enum AstraVersion {
    public static let protocolVersion = 1
    public static let dataVersion = 1
    public static let maximumMessageBytes = 1_048_576
}
