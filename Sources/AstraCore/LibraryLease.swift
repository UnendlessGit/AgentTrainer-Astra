import Foundation
import Darwin

/// One app coordinator per library. The descriptor is close-on-exec so a child
/// runtime cannot accidentally keep a crashed UI's library ownership alive.
public final class LibraryLease: @unchecked Sendable {
    private let descriptor: Int32
    public init(root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let path = root.appendingPathComponent(".coordinator.lock").path
        descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw AstraError("library.lease", "The workspace ownership file could not be opened.") }
        var information = stat()
        guard fstat(descriptor, &information) == 0, information.st_mode & S_IFMT == S_IFREG else {
            close(descriptor)
            throw AstraError("library.lease", "The workspace ownership path must be a regular file.")
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            throw AstraError("library.inUse", "This workspace is already open in another AgentTrainer Astra process. Close that copy before opening it here.")
        }
    }
    deinit { flock(descriptor, LOCK_UN); close(descriptor) }
}
