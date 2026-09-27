import SwiftUI
import AppKit
import AstraCore
import AstraPlatform

/// The host owns credentials and lifecycle. The presentation below receives
/// only non-secret state and explicit user actions, never the session token.
struct LiveSignalConnectionView: View {
    let host: DesktopLearningHost
    var body: some View {
        if let endpoint = host.liveSignalEndpoint {
            LiveSignalConnectionPanel(address: endpoint.address, sessionID: endpoint.sessionID,
                status: host.liveSignalStatus, values: host.liveSignalValues,
                canCopySignalIDs: host.liveSignalBinding != nil,
                copyConnection: {
                    let value: JSONValue = .object(["host": .string(endpoint.host), "port": .integer(Int64(endpoint.port)),
                        "sessionID": .string(endpoint.sessionID.uuidString.lowercased()), "token": .string(endpoint.token)])
                    guard let data = try? JSONEncoder().encode(value), let text = String(data: data, encoding: .utf8) else { return false }
                    return Self.copy(text)
                }, copyToken: { Self.copy(endpoint.token) }, copySignalIDs: {
                    guard let binding = host.liveSignalBinding else { return false }
                    return Self.copy(binding.signals.map { "\($0.name): \($0.id.uuidString.lowercased())" }.joined(separator: "\n"))
                }, openGuide: exampleAction)
        }
    }
    private static func copy(_ text: String) -> Bool {
        NSPasteboard.general.clearContents()
        return NSPasteboard.general.setString(text, forType: .string)
    }
    private var exampleAction: (() -> Void)? {
        guard let folder = Bundle.main.resourceURL?.appendingPathComponent("Examples"), FileManager.default.fileExists(atPath: folder.path) else { return nil }
        return { NSWorkspace.shared.open(folder) }
    }
}

struct LiveSignalConnectionPanel: View {
    let address: String
    let sessionID: UUID
    let status: String
    let values: [DesktopLiveValue]
    let canCopySignalIDs: Bool
    let copyConnection: () -> Bool
    let copyToken: () -> Bool
    let copySignalIDs: () -> Bool
    var openGuide: (() -> Void)? = nil
    @State private var copyMessage: String?

    var body: some View {
        GroupBox("Local state source") {
            VStack(alignment: .leading, spacing: 12) {
                LabeledContent("Connection") { Text(address).monospacedDigit().textSelection(.enabled) }
                Text(status).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                ForEach(values) { value in signalRow(value) }
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) { copyButtons }.fixedSize(horizontal: true, vertical: false)
                    VStack(alignment: .leading, spacing: 8) { copyButtons }
                }
                if let copyMessage { Text(copyMessage).font(.caption).foregroundStyle(.secondary) }
                if let openGuide {
                    Button("Show Client Example and Guide", systemImage: "folder", action: openGuide)
                }
                Text("Connect a local telemetry client. Astra waits for current values before each reset. A missing or interrupted source stops required reward evaluation. The session token expires when this run ends.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
        }.onChange(of: sessionID) { _, _ in copyMessage = nil }
    }

    @ViewBuilder private var copyButtons: some View {
        Button("Copy Connection Settings", systemImage: "doc.on.doc") { copy(copyConnection, success: "Connection settings copied") }
            .help("Copies the local address, session identity and private token. Keep these settings on this Mac.")
        Button("Copy Token") { copy(copyToken, success: "Session token copied") }
            .help("Copies the private token without displaying it. The token expires when the run ends.")
        if canCopySignalIDs {
            Button("Copy Signal IDs") { copy(copySignalIDs, success: "Signal IDs copied") }
        }
    }
    private func copy(_ action: () -> Bool, success: String) { copyMessage = action() ? success : "Couldn’t copy to the clipboard. Try again." }

    private func signalRow(_ value: DesktopLiveValue) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                signalName(value).fixedSize(horizontal: true, vertical: false)
                Spacer(minLength: 12)
                Text(value.text).foregroundStyle(.secondary).fixedSize(horizontal: true, vertical: false)
            }
            VStack(alignment: .leading, spacing: 4) {
                signalName(value).fixedSize(horizontal: false, vertical: true)
                Text(value.text).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.leading, 24)
            }
        }
        .font(.callout).accessibilityElement(children: .combine)
        .accessibilityValue(value.current ? "Current value" : "Waiting for a current value")
    }
    private func signalName(_ value: DesktopLiveValue) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: value.current ? "checkmark.circle" : "clock")
                .foregroundStyle(value.current ? Color.green : .secondary).accessibilityHidden(true)
            Text(value.name).fontWeight(.medium)
        }
    }
}
