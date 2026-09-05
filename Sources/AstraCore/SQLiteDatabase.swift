import Foundation
import CSQLite

public enum SQLValue: Sendable, Equatable {
    case integer(Int64), real(Double), text(String), blob(Data), null
    public var data: Data? { if case .blob(let data) = self { return data }; return nil }
    public var string: String? { if case .text(let text) = self { return text }; return nil }
    public var integer: Int64? { if case .integer(let number) = self { return number }; return nil }
}

/// Serialized connection; nested operations inside a transaction share the same
/// recursive lock. No SQLite handle or statement escapes this object.
public final class SQLiteDatabase: @unchecked Sendable {
    private var connection: OpaquePointer?
    private let lock = NSRecursiveLock()
    private var inTransaction = false

    public init(url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let result = sqlite3_open_v2(url.path, &connection,
                                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nil)
        guard result == SQLITE_OK, connection != nil else {
            let message = connection.map { String(cString: sqlite3_errmsg($0)) } ?? "Cannot open database."
            if let connection { sqlite3_close_v2(connection) }
            connection = nil
            throw AstraError("storage.open", message)
        }
        do {
            sqlite3_busy_timeout(connection, 5_000)
            try execute("PRAGMA foreign_keys = ON")
            try execute("PRAGMA journal_mode = WAL")
            try execute("PRAGMA synchronous = FULL")
        } catch {
            if let connection { sqlite3_close_v2(connection) }
            connection = nil
            throw error
        }
    }

    deinit { if let connection { sqlite3_close_v2(connection) } }

    public func execute(_ sql: String, _ bindings: [SQLValue] = []) throws {
        try lock.withLock {
            let statement = try prepare(sql, bindings)
            defer { sqlite3_finalize(statement) }
            var result = sqlite3_step(statement)
            while result == SQLITE_ROW { result = sqlite3_step(statement) }
            guard result == SQLITE_DONE else { throw failure("storage.write") }
        }
    }

    public func query(_ sql: String, _ bindings: [SQLValue] = []) throws -> [[String: SQLValue]] {
        try lock.withLock {
            let statement = try prepare(sql, bindings)
            defer { sqlite3_finalize(statement) }
            var rows: [[String: SQLValue]] = []
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { return rows }
                guard result == SQLITE_ROW else { throw failure("storage.read") }
                var row: [String: SQLValue] = [:]
                for column in 0..<sqlite3_column_count(statement) {
                    let name = String(cString: sqlite3_column_name(statement, column))
                    switch sqlite3_column_type(statement, column) {
                    case SQLITE_INTEGER: row[name] = .integer(sqlite3_column_int64(statement, column))
                    case SQLITE_FLOAT: row[name] = .real(sqlite3_column_double(statement, column))
                    case SQLITE_TEXT:
                        let count = Int(sqlite3_column_bytes(statement, column))
                        guard let pointer = sqlite3_column_text(statement, column),
                              let value = String(data: Data(bytes: pointer, count: count), encoding: .utf8) else {
                            throw AstraError("storage.text", "The database contains invalid UTF-8 text.")
                        }
                        row[name] = .text(value)
                    case SQLITE_BLOB:
                        let count = Int(sqlite3_column_bytes(statement, column))
                        if count == 0 { row[name] = .blob(Data()) }
                        else if let pointer = sqlite3_column_blob(statement, column) {
                            row[name] = .blob(Data(bytes: pointer, count: count))
                        } else { throw failure("storage.blob") }
                    default: row[name] = .null
                    }
                }
                rows.append(row)
            }
        }
    }

    public func transaction<T>(_ operation: () throws -> T) throws -> T {
        try lock.withLock {
            guard !inTransaction else { throw AstraError("storage.transaction", "Nested write transactions are not supported.") }
            try execute("BEGIN IMMEDIATE")
            inTransaction = true
            defer { inTransaction = false }
            do {
                let result = try operation()
                try execute("COMMIT")
                return result
            } catch {
                do { try execute("ROLLBACK") }
                catch { throw AstraError("storage.rollback", "The transaction could not be rolled back; reopen the library before continuing.", recoverable: false) }
                throw error
            }
        }
    }

    public func checkpoint() throws { try execute("PRAGMA wal_checkpoint(TRUNCATE)") }

    public func backup(to url: URL) throws {
        try lock.withLock {
            guard !FileManager.default.fileExists(atPath: url.path) else {
                throw AstraError("storage.destinationExists", "The backup destination already exists.")
            }
            var destination: OpaquePointer?
            guard sqlite3_open_v2(url.path, &destination, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil) == SQLITE_OK,
                  let destination else { throw AstraError("storage.backup", "Cannot create the backup database.") }
            defer { sqlite3_close_v2(destination) }
            guard let backup = sqlite3_backup_init(destination, "main", connection, "main") else {
                throw AstraError("storage.backup", "Cannot begin a consistent database backup.")
            }
            let result = sqlite3_backup_step(backup, -1)
            let finished = sqlite3_backup_finish(backup)
            guard result == SQLITE_DONE, finished == SQLITE_OK else {
                throw AstraError("storage.backup", "The database backup did not complete.")
            }
        }
    }

    private func prepare(_ sql: String, _ bindings: [SQLValue]) throws -> OpaquePointer {
        guard let connection else { throw AstraError("storage.closed", "The database is closed.") }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(connection, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            throw failure("storage.statement")
        }
        do {
            guard sqlite3_bind_parameter_count(statement) == Int32(bindings.count) else {
                throw AstraError("storage.bindings", "The database statement has incorrect bindings.")
            }
            let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            for (offset, value) in bindings.enumerated() {
                let index = Int32(offset + 1)
                let result: Int32
                switch value {
                case .integer(let value): result = sqlite3_bind_int64(statement, index, value)
                case .real(let value):
                    guard value.isFinite else { throw AstraError("storage.nonFinite", "Nonfinite numbers cannot be stored.") }
                    result = sqlite3_bind_double(statement, index, value)
                case .text(let value):
                    result = value.withCString { sqlite3_bind_text(statement, index, $0, Int32(value.utf8.count), transient) }
                case .blob(let data):
                    if data.isEmpty { result = sqlite3_bind_zeroblob(statement, index, 0) }
                    else { result = data.withUnsafeBytes { sqlite3_bind_blob64(statement, index, $0.baseAddress, UInt64($0.count), transient) } }
                case .null: result = sqlite3_bind_null(statement, index)
                }
                guard result == SQLITE_OK else { throw failure("storage.binding") }
            }
            return statement
        } catch { sqlite3_finalize(statement); throw error }
    }

    private func failure(_ code: String) -> AstraError {
        AstraError(code, connection.map { String(cString: sqlite3_errmsg($0)) } ?? "The database is closed.")
    }
}
