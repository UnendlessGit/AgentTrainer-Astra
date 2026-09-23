import AppKit
import UniformTypeIdentifiers

/// SwiftUI owns selection and transfer state. These short-lived AppKit panels
/// only choose a local URL; choosing it never starts an import or export.
@MainActor enum NativeArtifactPanels {
    static func chooseArchive() async -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose an Astra Archive"
        panel.message = "Choose an AgentTrainer Astra archive (.astraarchive) to preview its contents."
        panel.prompt = "Preview Archive"
        panel.canChooseFiles = true; panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false; panel.treatsFilePackagesAsDirectories = false
        if let type = UTType(filenameExtension: "astraarchive", conformingTo: .package) { panel.allowedContentTypes = [type] }
        return await ArtifactPanelSession(panel).choose()
    }

    static func saveArchive(suggestedName: String = "AgentTrainer Astra") async -> URL? {
        let panel = NSSavePanel()
        panel.title = "Export Astra Archive"
        panel.message = "Choose a new location for the archive shown in the preview."
        panel.prompt = "Export Archive"
        panel.canCreateDirectories = true; panel.isExtensionHidden = false
        if let type = UTType(filenameExtension: "astraarchive", conformingTo: .package) { panel.allowedContentTypes = [type] }
        var name = suggestedName.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
        if name.lowercased().hasSuffix(".astraarchive") { name.removeLast(".astraarchive".count) }
        while name.utf8.count > 180 { name.removeLast() }
        panel.nameFieldStringValue = (name.isEmpty ? "AgentTrainer Astra" : name) + ".astraarchive"
        return await ArtifactPanelSession(panel).choose()
    }

    static func chooseFolder(title: String = "Choose Storage Folder", message: String? = nil) async -> URL? {
        let panel = NSOpenPanel()
        panel.title = title; panel.message = message ?? "Choose the folder for these local artifacts."
        panel.prompt = "Choose Folder"
        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        return await ArtifactPanelSession(panel).choose()
    }
}

@MainActor private final class ArtifactPanelSession {
    private let panel: NSSavePanel
    init(_ panel: NSSavePanel) { self.panel = panel }
    func choose() async -> URL? {
        await withTaskCancellationHandler {
            guard !Task.isCancelled else { return nil }
            return await withCheckedContinuation { continuation in
                panel.begin { [self] response in
                    continuation.resume(returning: response == .OK ? panel.url : nil)
                }
            }
        } onCancel: { Task { @MainActor [self] in panel.cancel(nil) } }
    }
}
