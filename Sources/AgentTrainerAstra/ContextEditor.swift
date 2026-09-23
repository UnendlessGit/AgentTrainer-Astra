import SwiftUI
import AstraCore

/// Uses the vocabulary embedded in the chosen checkpoint, never today's
/// editable catalog, so old model indices cannot silently change meaning.
struct ContextValuePickers: View {
    var vocabulary: ContextVocabulary?
    var sizes: [Int]
    @Binding var indices: [Int]
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(sizes.indices, id: \.self) { index in
                let field = vocabulary?.fields.indices.contains(index) == true ? vocabulary?.fields[index] : nil
                let selection = Binding(get: { indices.indices.contains(index) ? indices[index] : 0 }, set: { value in
                    if indices.count != sizes.count { indices = sizes.map { _ in 0 } }
                    indices[index] = value
                })
                if let field {
                    Picker(field.name, selection: selection) {
                        Text("Unknown / not specified").tag(0)
                        ForEach(Array(field.values.enumerated()), id: \.element.id) { offset, value in Text(value.name).tag(offset + 1) }
                    }
                } else {
                    TextField("Context \(index + 1) · 0–\(sizes[index] - 1)", value: selection, format: .number.grouping(.never))
                        .textFieldStyle(.roundedBorder)
                }
            }
            if !sizes.isEmpty {
                Text("Values remain fixed throughout this run. Stop and start a new run to change them.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

struct ContextAssignmentPickers: View {
    let vocabulary: ContextVocabulary
    @Binding var values: [UUID: UUID]
    var body: some View {
        ForEach(vocabulary.fields) { field in
            Picker(field.name, selection: Binding<UUID?>(get: { values[field.id] }, set: { values[field.id] = $0 })) {
                Text("Unknown / not specified").tag(nil as UUID?)
                ForEach(field.values) { value in Text(value.name).tag(Optional(value.id)) }
                if let current = values[field.id], !field.values.contains(where: { $0.id == current }) {
                    Text("Unavailable value — choose again").tag(Optional(current))
                }
            }
        }
    }
}

struct ContextEditor: View {
    let agent: AgentDocument
    let model: WorkspaceModel
    @Environment(\.dismiss) private var dismiss
    @State private var fields: [ContextFieldDocument]
    @State private var selectedIDs: [UUID]
    @State private var editingID: UUID?
    @State private var saving = false
    @State private var issue: String?
    init(agent: AgentDocument, model: WorkspaceModel) {
        self.agent = agent; self.model = model
        _fields = State(initialValue: model.contextFields)
        _selectedIDs = State(initialValue: agent.contextFieldIDs ?? [])
    }
    private var canSave: Bool {
        fields.allSatisfy { (try? $0.validated()) != nil } && selectedIDs.count <= 32
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Contexts for \(agent.name)").font(.title2.weight(.semibold))
            Text("Give the agent categorical information such as a task, mode or preference. New models use the selected fields. Existing datasets and checkpoints retain their original vocabulary.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading) {
                    Text("Saved fields").font(.headline)
                    List(selection: $editingID) {
                        ForEach(fields) { field in
                            HStack {
                                Toggle(isOn: Binding(get: { selectedIDs.contains(field.id) }, set: { enabled in
                                    if enabled { selectedIDs.append(field.id) } else { selectedIDs.removeAll { $0 == field.id } }
                                })) { Text(field.name.isEmpty ? "Untitled field" : field.name) }.toggleStyle(.checkbox)
                            }.tag(field.id)
                        }
                    }.frame(minWidth: 220, minHeight: 290)
                    Button("Add Field", systemImage: "plus") {
                        let field = ContextFieldDocument(name: "New field")
                        fields.append(field); editingID = field.id
                        if selectedIDs.count < 32 { selectedIDs.append(field.id) }
                    }.disabled(fields.count >= 256)
                }.frame(width: 245)
                if let id = editingID, let index = fields.firstIndex(where: { $0.id == id }) {
                    VStack(alignment: .leading, spacing: 12) {
                        TextField("Field name", text: $fields[index].name).textFieldStyle(.roundedBorder)
                        Text("Values").font(.headline)
                        Text("Unknown / not specified is always available.").font(.caption).foregroundStyle(.secondary)
                        ScrollView {
                            VStack(spacing: 8) {
                                ForEach($fields[index].values) { $value in
                                    HStack {
                                        TextField("Value name", text: $value.name).textFieldStyle(.roundedBorder)
                                        Button { fields[index].values.removeAll { $0.id == value.id } } label: { Image(systemName: "minus.circle") }
                                            .accessibilityLabel("Remove \(value.name)")
                                    }
                                }
                            }
                        }.frame(maxHeight: 270)
                        Button("Add Value", systemImage: "plus") { fields[index].values.append(.init(name: "New value")) }
                            .disabled(fields[index].values.count >= 255)
                        Text("Renaming keeps a field or value’s meaning. Add a new value when its meaning changes. Saved models keep their own labels.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, alignment: .topLeading)
                } else {
                    ContentUnavailableView("Choose a context field", systemImage: "tag", description: Text("Select a field to edit its name and values."))
                        .frame(maxWidth: .infinity)
                }
            }
            if let issue { AttentionLabel(message: issue) }
            HStack {
                Text("\(selectedIDs.count) of 32 fields selected").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(saving ? "Saving…" : "Save Contexts") {
                    saving = true; issue = nil
                    Task {
                        do { try await model.saveContexts(fields, selectedIDs: selectedIDs, for: agent.id); dismiss() }
                        catch { issue = error.localizedDescription }
                        saving = false
                    }
                }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction).disabled(!canSave)
            }
        }.padding(24).frame(width: 760, height: 530).disabled(saving).interactiveDismissDisabled(saving)
            .onAppear { editingID = fields.first?.id }
    }
}
