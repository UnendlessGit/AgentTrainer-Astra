import Foundation

/// Integer cases preserve nanosecond clocks beyond Double's exact range.
public indirect enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue]), array([JSONValue]), string(String)
    case integer(Int64), unsigned(UInt64), number(Double), bool(Bool), null

    public init(from decoder: any Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let bool = try? value.decode(Bool.self) { self = .bool(bool) }
        else if let integer = try? value.decode(Int64.self) { self = .integer(integer) }
        else if let unsigned = try? value.decode(UInt64.self) { self = .unsigned(unsigned) }
        else if let number = try? value.decode(Double.self), number.isFinite { self = .number(number) }
        else if let string = try? value.decode(String.self) { self = .string(string) }
        else if let array = try? value.decode([JSONValue].self) { self = .array(array) }
        else if let object = try? value.decode([String: JSONValue].self) { self = .object(object) }
        else { throw AstraError("protocol.invalidJSON", "Unsupported JSON value.") }
    }

    public func encode(to encoder: any Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .object(let object): try value.encode(object)
        case .array(let array): try value.encode(array)
        case .string(let string): try value.encode(string)
        case .integer(let integer): try value.encode(integer)
        case .unsigned(let unsigned): try value.encode(unsigned)
        case .number(let number):
            guard number.isFinite else { throw AstraError("protocol.nonFinite", "Nonfinite values cannot be sent.") }
            try value.encode(number)
        case .bool(let bool): try value.encode(bool)
        case .null: try value.encodeNil()
        }
    }

    public static func encode<T: Encodable>(_ value: T) throws -> Self {
        try JSONDecoder().decode(Self.self, from: JSONEncoder().encode(value))
    }

    public func decode<T: Decodable>(_ type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: JSONEncoder().encode(self))
    }

    public var fields: [String: JSONValue]? { if case .object(let value) = self { value } else { nil } }
    public var text: String? { if case .string(let value) = self { value } else { nil } }
    public var uuid: UUID? { text.flatMap(UUID.init(uuidString:)) }
    public var int: Int? {
        switch self { case .integer(let value): Int(exactly: value); case .unsigned(let value): Int(exactly: value); default: nil }
    }
    public var uint64: UInt64? {
        switch self { case .integer(let value): UInt64(exactly: value); case .unsigned(let value): value; default: nil }
    }
    public var double: Double? {
        switch self { case .number(let value): value; case .integer(let value): Double(value); case .unsigned(let value): Double(value); default: nil }
    }
    public func required(_ key: String) throws -> JSONValue {
        guard let value = fields?[key], value != .null else {
            throw AstraError("protocol.missingField", "Required metadata is missing \(key).")
        }
        return value
    }
    public func requiredUUID(_ key: String) throws -> UUID {
        guard let value = fields?[key]?.uuid else {
            throw AstraError("protocol.invalidIdentity", "The runtime did not provide a valid \(key).")
        }
        return value
    }
}

public struct WireMessage: Codable, Equatable, Sendable {
    public var version: Int
    public var kind: String
    public var sequence: UInt64
    public var requestID: UUID?
    public var runID: UUID?
    public var payload: JSONValue

    public init(kind: String, sequence: UInt64, requestID: UUID? = nil,
                runID: UUID? = nil, payload: JSONValue = .object([:])) {
        self.version = AstraVersion.protocolVersion
        self.kind = kind; self.sequence = sequence
        self.requestID = requestID; self.runID = runID; self.payload = payload
    }

    public func validated() throws -> Self {
        guard version == AstraVersion.protocolVersion else {
            throw AstraError("protocol.version", "The helper uses an incompatible protocol version.", recoverable: false)
        }
        guard !kind.isEmpty, kind.utf8.count <= 80,
              kind.utf8.allSatisfy({ (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 46 }),
              case .object = payload else {
            throw AstraError("protocol.envelope", "Invalid message kind or payload.")
        }
        return self
    }

    public func framed() throws -> Data {
        _ = try validated()
        var data = try JSONEncoder().encode(self)
        guard data.count < AstraVersion.maximumMessageBytes else {
            throw AstraError("protocol.tooLarge", "The helper message exceeds its size limit.")
        }
        data.append(10)
        return data
    }
}

/// The transport is terminal after malformed framing; never recover by interpreting
/// a suffix of an oversized command as a fresh actionable message.
public struct MessageFramer: Sendable {
    private var pending = Data()
    private var failed = false

    public init() {}

    public mutating func append(_ bytes: Data) throws -> [WireMessage] {
        guard !failed else { throw AstraError("protocol.closed", "The malformed transport is closed.") }
        var messages: [WireMessage] = []
        do {
            for byte in bytes {
                if byte == 10 {
                    guard !pending.isEmpty else { throw AstraError("protocol.empty", "An empty protocol line is invalid.") }
                    let message = try JSONDecoder().decode(WireMessage.self, from: pending).validated()
                    messages.append(message)
                    pending.removeAll(keepingCapacity: true)
                } else {
                    guard pending.count < AstraVersion.maximumMessageBytes - 1 else {
                        throw AstraError("protocol.tooLarge", "The helper message exceeds its size limit.")
                    }
                    pending.append(byte)
                }
            }
            return messages
        } catch {
            failed = true
            pending.removeAll()
            throw error
        }
    }

    public mutating func finish() throws {
        guard !failed, pending.isEmpty else {
            failed = true
            throw AstraError("protocol.truncated", "The helper closed in the middle of a message.")
        }
    }
}
