import SwiftUI
import AstraCore

struct AgentEvaluationView: View {
    let agent: AgentDocument
    @Bindable var model: WorkspaceModel
    @State private var practiceOutcomes = false
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Picker("Evaluation method", selection: $practiceOutcomes) {
                Text("Demonstration likelihood").tag(false)
                Text("Practice outcomes").tag(true)
            }.pickerStyle(.segmented).frame(maxWidth: 460)
            if practiceOutcomes { ClosedLoopEvaluationView(agent: agent, model: model) }
            else { LearningEvaluationView(agent: agent, model: model) }
        }
    }
}
