import Foundation
import CryptoKit
import Darwin

public enum ControlHistoryLocation: String, Codable, Sendable {
    case inference = "Runs"
    case desktop = "DesktopRuns"
    public var title: String { self == .inference ? "Agent run" : "Desktop learning" }
    var cleanupKey: String { self == .inference ? "cleanupConfirmed" : "controlCleanupConfirmed" }
}

/// An acknowledgement identifies the exact local history that the operator
/// reviewed. It is not native cleanup evidence and never changes the report.
public struct ControlHistoryReview: Codable, Hashable, Identifiable, Sendable {
    public let location: ControlHistoryLocation
    public let runID: UUID
    public let directoryName: String
    public let fingerprint: String
    public let message: String
    public var id: String { location.rawValue + "/" + directoryName + "/" + fingerprint }
    public var title: String { location.title + " · " + String(runID.uuidString.prefix(8)) }
}

/// Small, no-follow descriptor reads keep review identity separate from display
/// diagnostics. A bad entry cannot prevent the other history directory scanning.
enum ControlHistoryReader {
    static let maximumResultBytes = 262_144
    static let maximumConfigurationBytes = 8 * 1024 * 1024
    private struct Directory {
        let fd: Int32
        let information: stat
    }
    private struct FileSnapshot {
        let data: Data?
        let identity: String
    }
    static func names(root: URL, location: ControlHistoryLocation) throws -> [String] {
        guard let directory = try openDirectory(root.appendingPathComponent(location.rawValue), missingAllowed: true) else { return [] }
        defer { close(directory.fd) }
        let duplicate = dup(directory.fd)
        guard duplicate >= 0 else { throw failure("The run history could not be enumerated.") }
        guard let stream = fdopendir(duplicate) else { close(duplicate); throw failure("The run history could not be enumerated.") }
        defer { closedir(stream) }
        var names: [String] = []
        while true {
            try Task.checkCancellation()
            errno = 0
            guard let entry = readdir(stream) else {
                guard errno == 0 else { throw failure("The run history directory could not be completely read.") }
                break
            }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                String(cString: UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self))
            }
            if name == "." || name == ".." { continue }
            guard names.count < 4096 else { throw failure("Run history exceeds its automatic review limit. Archive older run folders before starting live control.") }
            names.append(name)
        }
        return names.sorted()
    }

    static func review(root: URL, location: ControlHistoryLocation, directoryName: String) throws -> ControlHistoryReview? {
        guard let runID = UUID(uuidString: directoryName), directoryName.utf8.count == 36 else {
            throw failure("The run history identity is invalid.")
        }
        guard let parent = try openDirectory(root.appendingPathComponent(location.rawValue), missingAllowed: false) else { throw failure("Run history is missing.") }
        defer { close(parent.fd) }
        let descriptor = openat(parent.fd, directoryName, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw failure("A run history entry is missing, linked or not a directory.") }
        defer { close(descriptor) }
        var information = stat()
        guard fstat(descriptor, &information) == 0, information.st_mode & S_IFMT == S_IFDIR else { throw failure("The run history entry changed while being read.") }
        func verifyLocation() throws {
            var currentDirectory = stat(), currentParent = stat()
            guard fstatat(parent.fd, directoryName, &currentDirectory, AT_SYMLINK_NOFOLLOW) == 0,
                  lstat(root.appendingPathComponent(location.rawValue).path, &currentParent) == 0,
                  currentDirectory.st_mode & S_IFMT == S_IFDIR, currentParent.st_mode & S_IFMT == S_IFDIR,
                  identity(information) == identity(currentDirectory), identity(parent.information) == identity(currentParent) else {
                throw failure("The run history location changed while it was being reviewed. Refresh before acknowledging it.")
            }
        }
        let result = try read("results.json", directory: descriptor, maximum: maximumResultBytes)
        try verifyLocation()
        let message: String
        if let data = result.data {
            do {
                let document = try JSONDecoder().decode(JSONValue.self, from: data)
                guard let fields = document.fields, fields["runID"]?.uuid == runID,
                      fields["schemaVersion"] == nil || fields["schemaVersion"] == .integer(1) else {
                    throw failure("A previous control result has an inconsistent or unsupported identity.")
                }
                if fields[location.cleanupKey] == .bool(true) { return nil }
                message = "Native cleanup was not confirmed for this run. Release any remaining held keys or mouse buttons before acknowledging it."
            } catch {
                message = "This run’s cleanup result could not be verified. Release any remaining held keys or mouse buttons before acknowledging this exact history."
            }
        } else {
            message = "This run ended before a cleanup result was saved. Verify that no keys or mouse buttons remain held before acknowledging this exact history."
        }
        let configuration = try read("configuration.json", directory: descriptor, maximum: maximumConfigurationBytes)
        try verifyLocation()
        var digest = SHA256()
        func include(_ data: Data) {
            var size = UInt64(data.count).bigEndian
            withUnsafeBytes(of: &size) { digest.update(data: Data($0)) }
            digest.update(data: data)
        }
        for value in ["AstraControlHistoryReview1", location.rawValue, directoryName, identity(information),
                      result.identity, configuration.identity] { include(Data(value.utf8)) }
        include(result.data ?? Data()); include(configuration.data ?? Data())
        let fingerprint = digest.finalize().map { String(format: "%02x", $0) }.joined()
        return .init(location: location, runID: runID, directoryName: directoryName, fingerprint: fingerprint, message: message)
    }

    private static func openDirectory(_ url: URL, missingAllowed: Bool) throws -> Directory? {
        let descriptor = open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        if descriptor < 0, errno == ENOENT, missingAllowed { return nil }
        guard descriptor >= 0 else { throw failure("The local run history must be a readable regular directory.") }
        var information = stat()
        guard fstat(descriptor, &information) == 0, information.st_mode & S_IFMT == S_IFDIR else {
            close(descriptor); throw failure("The local run history changed while being opened.")
        }
        return Directory(fd: descriptor, information: information)
    }
    private static func identity(_ info: stat) -> String {
        "\(info.st_dev):\(info.st_ino):\(info.st_birthtimespec.tv_sec):\(info.st_birthtimespec.tv_nsec)"
    }
    private static func read(_ name: String, directory: Int32, maximum: Int) throws -> FileSnapshot {
        let descriptor = openat(directory, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        if descriptor < 0, errno == ENOENT { return FileSnapshot(data: nil, identity: "missing:" + name) }
        guard descriptor >= 0 else { throw failure("A previous run file is missing or linked. Repair its history before starting live control.") }
        defer { close(descriptor) }
        var before = stat()
        guard fstat(descriptor, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
              before.st_size >= 0, before.st_size <= maximum else {
            throw failure("A previous run file is not a regular bounded report. Repair its history before starting live control.")
        }
        var data = Data(), buffer = [UInt8](repeating: 0, count: 65_536)
        while true {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes { Darwin.read(descriptor, $0.baseAddress!, $0.count) }
            if count < 0, errno == EINTR { continue }
            guard count >= 0, count <= maximum - data.count else { throw failure("A run file changed or could not be read completely.") }
            if count == 0 { break }
            data.append(contentsOf: buffer.prefix(count))
        }
        var after = stat(), path = stat()
        guard fstat(descriptor, &after) == 0, fstatat(directory, name, &path, AT_SYMLINK_NOFOLLOW) == 0,
              identity(before) == identity(after), identity(after) == identity(path), after.st_mode & S_IFMT == S_IFREG,
              before.st_size == after.st_size, data.count == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec, before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec, before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else {
            throw failure("A previous run changed while it was being reviewed. Refresh its history and review the current report.")
        }
        return FileSnapshot(data: data, identity: name + ":" + identity(after))
    }
    private static func failure(_ message: String) -> AstraError { .init("inference.history", message) }
}
