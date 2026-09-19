import SwiftUI
import AstraCore

enum ResetPacketPresets {
    struct KeyPress { let key: Int; let modifiers: Set<Int>; let holdMS: Int }
    static let modifiers: [(Int, String)] = [(55, "Command"), (58, "Option"), (56, "Shift"), (59, "Control")]
    static let supportedKeys = (0...127).filter { ![57, 63, 72, 73, 74, 127].contains($0) }
    static func keyPress(key: Int = 36, modifiers: Set<Int> = [], holdMS: Int = 80) -> ResetPacketTemplate {
        let keys = modifiers.sorted() + [key]
        let end = holdMS.addingReportingOverflow(20)
        return .init(durationMS: end.overflow ? Int.max : end.partialValue,
            commands: keys.map { .init(offsetMs: 0, operation: .keyDown, keyCode: $0) }
                + keys.reversed().map { .init(offsetMs: holdMS, operation: .keyUp, keyCode: $0) })
    }
    static func keyPress(_ packet: ResetPacketTemplate) -> KeyPress? {
        let downs = Array(packet.commands.prefix { $0.operation == .keyDown && $0.offsetMs == 0 })
        let keys = downs.compactMap(\.keyCode)
        let ups = Array(packet.commands.dropFirst(downs.count))
        guard !keys.isEmpty, keys.count == downs.count, Set(keys).count == keys.count,
              Set(keys.dropLast()).isSubset(of: Set(modifiers.map(\.0))),
              ups.count == downs.count, ups.allSatisfy({ $0.operation == .keyUp && $0.offsetMs == ups.first?.offsetMs }),
              ups.compactMap(\.keyCode) == Array(keys.reversed()), let hold = ups.first?.offsetMs else { return nil }
        return .init(key: keys.last!, modifiers: Set(keys.dropLast()), holdMS: hold)
    }
    static func click(surfaceID: String) -> ResetPacketTemplate {
        .init(durationMS: 100, commands: [.init(offsetMs: 0, operation: .pointerAbsolute, surfaceID: surfaceID, x: 0.5, y: 0.5),
            .init(offsetMs: 0, operation: .buttonDown, button: 0), .init(offsetMs: 80, operation: .buttonUp, button: 0)])
    }
    static func isClick(_ packet: ResetPacketTemplate) -> Bool {
        packet.commands.count == 3 && packet.commands[0].operation == .pointerAbsolute && packet.commands[1].operation == .buttonDown
            && packet.commands[2].operation == .buttonUp && packet.commands[1].button == packet.commands[2].button
    }
}

struct ResetPlanEditor: View {
    @Binding var plan: ResetPlan?
    let signals: [RewardSignal]
    let referenceSurface: SurfaceDescriptor?
    @State private var previous = ResetPlan()
    private var waitSignals: [RewardSignal] { signals.filter { $0.kind != .elapsedSeconds } }
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Picker("Reset between episodes", selection: Binding(get: { plan != nil }, set: { automatic in
                    if automatic { plan = previous } else { previous = plan ?? previous; plan = nil }
                })) {
                    Text("Manual Ready").tag(false)
                    Text("Authored reset").tag(true)
                }.pickerStyle(.radioGroup)
                if let plan = Binding($plan) {
                    Text("Runs on the same environment between policy episodes. A fresh starting condition and released controls are required before training continues.")
                        .font(.callout).foregroundStyle(.secondary)
                    ForEach(plan.steps) { step in
                        ResetStepEditor(step: step, signals: waitSignals, referenceSurface: referenceSurface,
                            move: { direction in move(step.wrappedValue.id, direction: direction) },
                            remove: { self.plan?.steps.removeAll { $0.id == step.wrappedValue.id } })
                    }
                    Menu("Add Step", systemImage: "plus") {
                        Button("Key press or shortcut") { append(.init(name: "Key press", packet: ResetPacketPresets.keyPress())) }
                        Button("Click") { if let referenceSurface { append(.init(name: "Click", packet: ResetPacketPresets.click(surfaceID: referenceSurface.id))) } }
                            .disabled(referenceSurface == nil)
                        Button("Move pointer") {
                            if let referenceSurface { append(.init(name: "Move pointer", packet: .init(durationMS: 100,
                                commands: [.init(offsetMs: 0, operation: .pointerAbsolute, surfaceID: referenceSurface.id, x: 0.5, y: 0.5)]))) }
                        }.disabled(referenceSurface == nil)
                        Button("Relative pointer movement") { append(.init(name: "Relative movement", packet: .init(durationMS: 100,
                            commands: [.init(offsetMs: 0, operation: .pointerRelative, dx: 0, dy: 0)]))) }
                        Button("Scroll") { append(.init(name: "Scroll", packet: .init(durationMS: 100,
                            commands: [.init(offsetMs: 0, operation: .scroll, dx: 0, dy: 120)]))) }
                        Button("Pause") { append(.init(pauseMS: 500)) }
                        Button("Wait for condition") {
                            if let first = waitSignals.first { append(.init(condition: .init(conditions: [RewardPredicateEditor.condition(for: first)]))) }
                        }.disabled(waitSignals.isEmpty)
                    }.disabled(plan.wrappedValue.steps.count >= 64)
                    if plan.wrappedValue.steps.isEmpty { Text("No actions: wait only for the starting condition.").font(.caption).foregroundStyle(.secondary) }
                    if referenceSurface == nil { Text("Choose a reference recording above to add absolute pointer actions.").font(.caption).foregroundStyle(.secondary) }
                    Divider()
                    HStack {
                        ResetDurationField(title: "Total timeout", value: plan.maximumDurationMS, seconds: true)
                        ResetDurationField(title: "Starting condition timeout", value: plan.readinessTimeoutMS, seconds: true)
                    }
                    Stepper("Maximum attempts: \(plan.wrappedValue.maximumAttempts)", value: plan.maximumAttempts, in: 1...3)
                    Text("Additional attempts repeat the authored steps only after a condition timeout and confirmed cleanup. A failed action, changed target or cancellation stops the reset.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Reset the environment yourself, then confirm Ready. No reset actions run automatically.").foregroundStyle(.secondary)
                }
            }.padding(8)
        } label: { Text("Reset").font(.headline) }
        .onAppear { if let plan { previous = plan } }
    }
    private func append(_ step: ResetStep) { plan?.steps.append(step) }
    private func move(_ id: UUID, direction: Int) {
        guard let index = plan?.steps.firstIndex(where: { $0.id == id }), let count = plan?.steps.count,
              (0..<count).contains(index + direction) else { return }
        plan?.steps.swapAt(index, index + direction)
    }
}

private struct ResetStepEditor: View {
    @Binding var step: ResetStep
    let signals: [RewardSignal]
    let referenceSurface: SurfaceDescriptor?
    let move: (Int) -> Void
    let remove: () -> Void
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    TextField("Step name", text: $step.name).fontWeight(.medium)
                    Button { move(-1) } label: { Image(systemName: "arrow.up") }.help("Move step earlier").accessibilityLabel("Move step earlier")
                    Button { move(1) } label: { Image(systemName: "arrow.down") }.help("Move step later").accessibilityLabel("Move step later")
                    Button(action: remove) { Image(systemName: "trash") }.help("Remove step").accessibilityLabel("Remove reset step")
                }
                switch step.kind {
                case .packet:
                    if let packet = Binding($step.packet) { ResetPacketEditor(packet: packet, referenceSurface: referenceSurface) }
                case .pause:
                    if let value = Binding($step.pauseMS) { ResetDurationField(title: "Pause", value: value, seconds: true) }
                case .wait:
                    if let condition = Binding($step.condition) { RewardPredicateEditor(predicate: condition, signals: signals) }
                    if let timeout = Binding($step.timeoutMS) { ResetDurationField(title: "Wait timeout", value: timeout, seconds: true) }
                    Text("Missing or unreadable signals keep this step waiting; they do not become false or zero.").font(.caption).foregroundStyle(.secondary)
                }
            }.padding(6)
        }
    }
}

private struct ResetDurationField: View {
    let title: String
    @Binding var value: Int
    var seconds = false
    var body: some View {
        LabeledContent(title) {
            if seconds {
                TextField(title, value: Binding(get: { Double(value) / 1000 }, set: { value = Int(exactly: ($0 * 1000).rounded()) ?? Int.max }),
                          format: .number.precision(.fractionLength(0...3))).labelsHidden().frame(minWidth: 70, idealWidth: 90, maxWidth: 120)
            } else {
                TextField(title, value: $value, format: .number).labelsHidden().frame(minWidth: 70, idealWidth: 90, maxWidth: 120)
            }
            Text(seconds ? "s" : "ms").foregroundStyle(.secondary)
        }
    }
}

private struct ResetPacketEditor: View {
    @Binding var packet: ResetPacketTemplate
    let referenceSurface: SurfaceDescriptor?
    @State private var advanced = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let key = ResetPacketPresets.keyPress(packet) {
                Picker("Key", selection: Binding(get: { key.key }, set: { packet = ResetPacketPresets.keyPress(key: $0, modifiers: key.modifiers, holdMS: key.holdMS) })) {
                    ForEach(ResetPacketPresets.supportedKeys.filter { ![54, 55, 56, 58, 59, 60, 61, 62].contains($0) }, id: \.self) { Text(KeyNames.name($0)).tag($0) }
                }
                HStack {
                    ForEach(ResetPacketPresets.modifiers, id: \.0) { code, title in
                        Toggle(title, isOn: Binding(get: { key.modifiers.contains(code) }, set: { included in
                            var modifiers = key.modifiers
                            if included { modifiers.insert(code) } else { modifiers.remove(code) }
                            packet = ResetPacketPresets.keyPress(key: key.key, modifiers: modifiers, holdMS: key.holdMS)
                        })).toggleStyle(.checkbox)
                    }
                }
                ResetDurationField(title: "Hold key", value: Binding(get: { key.holdMS }, set: { packet = ResetPacketPresets.keyPress(key: key.key, modifiers: key.modifiers, holdMS: $0) }))
            } else if ResetPacketPresets.isClick(packet) {
                pointer(command: command(0))
                Picker("Mouse button", selection: Binding(get: { packet.commands.indices.contains(1) ? (packet.commands[1].button ?? 0) : 0 }, set: { value in
                    guard packet.commands.indices.contains(2) else { return }
                    packet.commands[1].button = value; packet.commands[2].button = value
                })) {
                    ForEach(0..<32, id: \.self) { Text(KeyNames.button($0)).tag($0) }
                }
            } else if packet.commands.count == 1 {
                if packet.commands[0].operation == .pointerAbsolute { pointer(command: command(0)) }
                else if packet.commands[0].operation == .pointerRelative || packet.commands[0].operation == .scroll {
                    HStack { delta("Horizontal", command: command(0), horizontal: true); delta("Vertical", command: command(0), horizontal: false) }
                    Text(packet.commands[0].operation == .scroll ? "Scroll values are measured in points." : "Relative movement uses whole raw pointer counts.").font(.caption).foregroundStyle(.secondary)
                }
            }
            DisclosureGroup("Timing & individual commands", isExpanded: $advanced) {
                VStack(alignment: .leading, spacing: 10) {
                    ResetDurationField(title: "Packet duration", value: $packet.durationMS)
                    ForEach(Array(packet.commands.indices), id: \.self) { index in
                        ResetCommandEditor(command: command(index), referenceSurface: referenceSurface,
                            remove: { if packet.commands.indices.contains(index) { packet.commands.remove(at: index) } })
                    }
                    Button("Add Command", systemImage: "plus") {
                        packet.commands.append(.init(offsetMs: packet.commands.last?.offsetMs ?? 0, operation: .keyUp, keyCode: 36))
                    }.disabled(packet.commands.count >= 64)
                    Text("Commands stay in this order; timestamps must not move backward. Packets are at most one second. Use separate packets and pauses for longer holds.")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding(.top, 8)
            }
        }
    }
    private func command(_ index: Int) -> Binding<TimedCommand> {
        Binding(get: { packet.commands.indices.contains(index) ? packet.commands[index] : .init(offsetMs: 0, operation: .keyUp, keyCode: 36) },
                set: { if packet.commands.indices.contains(index) { packet.commands[index] = $0 } })
    }
    private func pointer(command: Binding<TimedCommand>) -> some View { ResetPointerFields(command: command, referenceSurface: referenceSurface) }
    private func delta(_ title: String, command: Binding<TimedCommand>, horizontal: Bool) -> some View {
        RewardNumberField(title: title, value: Binding(get: { (horizontal ? command.wrappedValue.dx : command.wrappedValue.dy) ?? 0 },
            set: { if horizontal { command.wrappedValue.dx = $0 } else { command.wrappedValue.dy = $0 } }))
    }
}

private struct ResetPointerFields: View {
    @Binding var command: TimedCommand
    let referenceSurface: SurfaceDescriptor?
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(command.surfaceID == referenceSurface?.id && referenceSurface != nil ? "Using Reference Surface" : "Use Reference Surface") {
                command.surfaceID = referenceSurface?.id
            }.disabled(referenceSurface == nil)
            HStack { percentage("Horizontal position", horizontal: true); percentage("Vertical position", horizontal: false) }
            Text("Positions are percentages of the observed surface, from 0 up to 100. The right and bottom edges are excluded.").font(.caption).foregroundStyle(.secondary)
        }
    }
    private func percentage(_ title: String, horizontal: Bool) -> some View {
        RewardNumberField(title: title + " (%)", value: Binding(get: { ((horizontal ? command.x : command.y) ?? 0) * 100 },
            set: { if horizontal { command.x = $0 / 100 } else { command.y = $0 / 100 } }))
    }
}

private struct ResetCommandEditor: View {
    @Binding var command: TimedCommand
    let referenceSurface: SurfaceDescriptor?
    let remove: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                ResetDurationField(title: "At", value: $command.offsetMs)
                Picker("Command", selection: Binding(get: { command.operation }, set: { operation in
                    var next = TimedCommand(offsetMs: command.offsetMs, operation: operation)
                    switch operation {
                    case .keyDown, .keyUp, .keyRepeat: next.keyCode = 36
                    case .buttonDown, .buttonUp: next.button = 0
                    case .pointerAbsolute: next.surfaceID = referenceSurface?.id; next.x = 0.5; next.y = 0.5
                    case .pointerRelative, .scroll: next.dx = 0; next.dy = 0
                    }
                    command = next
                })) { ForEach(CommandOperation.allCases, id: \.self) { Text($0.resetTitle).tag($0) } }
                Button(action: remove) { Image(systemName: "minus.circle") }.accessibilityLabel("Remove timed command")
            }
            switch command.operation {
            case .keyDown, .keyUp, .keyRepeat:
                Picker("Key", selection: Binding(get: { command.keyCode ?? 36 }, set: { command.keyCode = $0 })) {
                    ForEach(ResetPacketPresets.supportedKeys, id: \.self) { Text(KeyNames.name($0)).tag($0) }
                }
            case .buttonDown, .buttonUp:
                Picker("Button", selection: Binding(get: { command.button ?? 0 }, set: { command.button = $0 })) {
                    ForEach(0..<32, id: \.self) { Text(KeyNames.button($0)).tag($0) }
                }
            case .pointerAbsolute: ResetPointerFields(command: $command, referenceSurface: referenceSurface)
            case .pointerRelative, .scroll:
                HStack {
                    RewardNumberField(title: "Horizontal", value: Binding(get: { command.dx ?? 0 }, set: { command.dx = $0 }))
                    RewardNumberField(title: "Vertical", value: Binding(get: { command.dy ?? 0 }, set: { command.dy = $0 }))
                }
            }
            Divider()
        }
    }
}

private extension CommandOperation {
    var resetTitle: String { switch self {
    case .keyDown: "Key down"; case .keyUp: "Key up"; case .keyRepeat: "Key repeat"; case .buttonDown: "Button down"; case .buttonUp: "Button up"
    case .pointerAbsolute: "Pointer position"; case .pointerRelative: "Relative movement"; case .scroll: "Scroll"
    } }
}
