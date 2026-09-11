import Foundation
import Testing
import AstraCore
@testable import AgentTrainerAstra

/// A protocol/lifecycle fixture only. It never imports MLX and does not claim
/// model correctness; real compute training is covered by Python job tests.
private let fixtureSource = #"""
import json, os, pathlib, sys, threading, uuid
root = pathlib.Path(__file__).parent
(root/'pid').write_text(str(os.getpid()))
mode = '__MODE__'
sequence = 0
lock = threading.Lock()
cancel = threading.Event()
thread = None
def send(kind, payload, request=None):
    global sequence
    with lock:
        message = dict(version=1,kind=kind,sequence=sequence,payload=payload)
        if request:
            for field in ('requestID','runID'):
                if field in request: message[field]=request[field]
        print(json.dumps(message),flush=True); sequence+=1
def artifact(destination, kind, model, actions):
    destination=pathlib.Path(destination);destination.mkdir(parents=True)
    manifest=dict(id=destination.name,kind=kind,step=0 if kind=='initial' else 1,
                  policySignature='a'*64,model=model,actions=actions,
                  metrics=dict(epoch=1,updates=1,decisions=2),datasetID=str(uuid.uuid4()))
    (destination/'manifest.json').write_text(json.dumps(manifest))
    (destination/'policy.safetensors').write_bytes(b'protocol-fixture-only')
    return manifest
def perform(request, job):
    value=request['payload'];op=request['kind']
    if op=='checkpoint.inspect' and mode=='inspectCancelled':
        send('job.cancelled',dict(jobID=job,result=dict(cancelled=True,checkpointPublished=False)),request);return
    if op=='dataset.prepare' and mode=='prepareFailure':
        send('job.failed',dict(jobID=job,error=dict(message='Incompatible recorded control',recoverable=True)),request);return
    if op=='dataset.prepare':
        result=dict(manifest=dict(model=value['model'],actions=value['actions'],steps=2),datasetPath=value['destination'])
    elif op=='checkpoint.create':
        manifest=artifact(value['destination'],'initial',value['model'],value['actions'])
        result=dict(manifest=manifest,checkpointPath=value['destination'],checkpointPublished=True,parameterCount=10)
    elif op=='checkpoint.inspect':
        manifest=json.loads((pathlib.Path(value['path'])/'manifest.json').read_text())
        result=dict(manifest=manifest,parameterCount=10,integrityVerified=True)
    elif op=='evaluate.behavioral':
        result=dict(evaluation=dict(available=True,split=value['split'],decisions=2,meanNLL=.75))
    else:
        if mode=='waitBeforeTraining':
            send('job.progress',dict(jobID=job,phase='preparing'),request)
            cancel.wait(10)
            send('job.cancelled',dict(jobID=job,result=dict(cancelled=True,checkpointPublished=False)),request);return
        reinforcement=op=='train.reinforcement'
        metric=dict(iteration=1,elapsed_seconds=1.5,mean_reward=.25,mean_value_loss=.3,mean_policy_loss=-.1,
                    maximum_sampled_kl=.001,clip_fraction=.05,optimizer_updates=1,decisions=2,peak_memory_bytes=1024)
        if reinforcement:
            progress_request={**request,'requestID':str(uuid.uuid4())} if mode=='wrongProgressRequest' else request
            progress_job=str(uuid.uuid4()) if mode=='wrongProgressJob' else job
            send('job.progress',dict(jobID=progress_job,phase='updating',sourceKind='practice_rollout',last_iteration=metric,**metric),progress_request)
        else:
            send('job.progress',dict(jobID=job,phase='training',epoch=0,updates=1,decisions=2,mean_nll=1.25,
                                     decisions_per_second=10,peak_memory_bytes=1024),request)
        if mode=='waitForCancel' and not value.get('resume',False):cancel.wait(10)
        initial=json.loads((pathlib.Path(value['checkpointPath'])/'manifest.json').read_text())
        manifest=artifact(value['destination'],'reinforcement' if reinforcement else 'behavioral',initial['model'],initial['actions'])
        manifest['trainingConfig']=value['training']
        (pathlib.Path(value['destination'])/'manifest.json').write_text(json.dumps(manifest))
        if reinforcement:manifest['metrics']=dict(iteration=1,optimizer_updates=1,decisions=2,elapsed_seconds=1.5)
        if mode=='wrongCheckpoint':manifest['id']=str(uuid.uuid4())
        result=dict(manifest=manifest,checkpointPath=value['destination'],checkpointPublished=True,
                    cancelled=cancel.is_set(),parameterCount=10)
        if reinforcement:result.update(sourceKind='practice_rollout',iterationMetrics=[metric],requiresEnvironmentReset=True)
    if mode=='wrongRequest':request={**request,'requestID':str(uuid.uuid4())}
    if mode=='wrongJob':job=str(uuid.uuid4())
    send('job.cancelled' if cancel.is_set() else 'job.completed',dict(jobID=job,result=result),request)
send('hello',dict(role='compute',protocolVersion=1))
for line in sys.stdin:
    request=json.loads(line)
    with (root/'requests.jsonl').open('a') as file:file.write(json.dumps(request)+'\n')
    kind=request['kind']
    if kind=='shutdown':
        if thread:thread.join(2)
        (root/'shutdown').write_text('joined')
        send('ack',dict(stopped=True),request)
        if mode=='lateFailure':print('{broken',flush=True)
        break
    if kind=='cancel':
        cancel.set();send('ack',dict(status='cancelling'),request);continue
    job=str(uuid.uuid4())
    if mode=='earlyEverything' or (mode=='earlyTerminal' and not kind.startswith('train.')):
        perform(request,job);send('ack',dict(jobID=job,status='queued'),request)
    else:
        send('ack',dict(jobID=job,status='queued'),{**request,'runID':str(uuid.uuid4())} if mode=='wrongAckRun' else request)
        thread=threading.Thread(target=perform,args=(request,job));thread.start()
"""#

private final class Fixture {
    let directory: URL
    let root: URL
    let bundle: Bundle
    let log: URL
    init(_ mode: String) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("AstraCoordinator-" + UUID().uuidString)
        root = directory.appendingPathComponent("Library")
        let application = directory.appendingPathComponent("Fixture.app")
        let contents = application.appendingPathComponent("Contents")
        let resources = contents.appendingPathComponent("Resources/Weights")
        let executable = contents.appendingPathComponent("Helpers/AstraCompute.app/Contents/MacOS/AstraCompute")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: executable.deletingLastPathComponent(), withIntermediateDirectories: true)
        let info: [String: Any] = ["CFBundleIdentifier": "test.astra.coordinator." + UUID().uuidString,
                                   "CFBundlePackageType": "APPL", "CFBundleName": "Fixture"]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(to: contents.appendingPathComponent("Info.plist"))
        try Data([0]).write(to: resources.appendingPathComponent("convnext_tiny.safetensors"))
        let script = directory.appendingPathComponent("compute.py")
        try fixtureSource.replacingOccurrences(of: "__MODE__", with: mode).write(to: script, atomically: true, encoding: .utf8)
        let python = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent(".venv/bin/python")
        func quoted(_ value: String) -> String { "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'" }
        try ("#!/bin/sh\nexec " + quoted(python.path) + " -u " + quoted(script.path) + "\n").write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        bundle = try #require(Bundle(url: application))
        log = directory.appendingPathComponent("requests.jsonl")
    }
    deinit { try? FileManager.default.removeItem(at: directory) }
    func operations() throws -> [String] {
        try String(contentsOf: log, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(WireMessage.self, from: Data($0.utf8)).kind
        }
    }
}

@MainActor private func waitUntil(_ condition: () -> Bool) async throws {
    let limit = ContinuousClock.now.advanced(by: .seconds(10))
    while !condition() {
        guard ContinuousClock.now < limit else { throw AstraError("test.timeout", "Coordinator did not reach the expected state") }
        try await Task.sleep(for: .milliseconds(10))
    }
}

@Suite(.serialized) @MainActor struct LearningCoordinatorTests {
    @Test func reinforcementPipelinePublishesRealOperationAndMetricsWithoutOracleFlags() async throws {
        let fixture = try Fixture("earlyTerminal")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Practice reinforcement")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = ReinforcementOptions(); options.iterations = 1
        try coordinator.startReinforcement(agent: agent, options: options)
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil)
        let snapshot = try await store.snapshot()
        let run = try #require(snapshot.learningRuns.first)
        #expect(run.kind == .reinforcement && run.sourceKind == "practice_rollout")
        #expect(run.status == .completed && run.meanNLL == nil && run.epoch == 1)
        let checkpoint = try #require(snapshot.checkpoints.first { $0.id == run.checkpointID })
        #expect(checkpoint.kind == "reinforcement" && snapshot.agents.first?.selectedCheckpointID == checkpoint.id)
        #expect(coordinator.reinforcementMetrics.first?.reward == 0.25 && coordinator.metrics.isEmpty)
        let saved = try await LearningFiles.read(fixture.root.appendingPathComponent("Jobs/\(run.id.uuidString.lowercased())/configuration.json"))
        #expect(saved.fields?["verificationMode"] == nil)
        #expect(saved.fields?["training"]?.fields?["rollout_decisions"] == .integer(512))
        #expect(saved.fields?["training"]?.fields?["sequence_length"] == .integer(64))
        #expect(try fixture.operations() == ["checkpoint.create", "train.reinforcement", "shutdown"])
        options.initialCheckpointID = checkpoint.id; options.resume = true; options.iterations = 2
        // Resume must load the stored settings rather than these edited values.
        options.task = "delayed_memory"; options.rolloutDecisions = 4
        try coordinator.startReinforcement(agent: agent, options: options)
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil)
        let resumed = try #require(coordinator.activeRun)
        let resumedConfiguration = try await LearningFiles.read(fixture.root.appendingPathComponent("Jobs/\(resumed.id.uuidString.lowercased())/configuration.json"))
        #expect(resumedConfiguration.fields?["environment"]?.fields?["task"] == .string("pointing"))
        #expect(resumedConfiguration.fields?["training"]?.fields?["rollout_decisions"] == .integer(512))
        #expect(try fixture.operations().suffix(3) == ["checkpoint.inspect", "train.reinforcement", "shutdown"])
    }

    @Test(arguments: ["wrongProgressRequest", "wrongProgressJob"])
    func reinforcementRejectsMismatchedProgressBeforePublication(mode: String) async throws {
        let fixture = try Fixture(mode)
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Progress identity")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        try coordinator.startReinforcement(agent: agent, options: ReinforcementOptions())
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure?.contains("different job or request") == true)
        #expect(coordinator.reinforcementMetrics.isEmpty)
        #expect(coordinator.activeRun?.status == .failed && coordinator.activeRun?.checkpointID == nil)
        #expect(try await store.snapshot().checkpoints.allSatisfy { $0.kind == "initial" })
    }

    @Test func reinforcementPreservesCorrelatedProgressDeliveredBeforeAcknowledgement() async throws {
        let fixture = try Fixture("earlyEverything")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Early progress")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        try coordinator.startReinforcement(agent: agent, options: ReinforcementOptions())
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil && coordinator.activeRun?.status == .completed)
        #expect(coordinator.reinforcementMetrics.first?.reward == 0.25)
    }

    @Test func reinforcementStopJoinsCheckpointPublicationAndChildShutdown() async throws {
        let fixture = try Fixture("waitForCancel")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Stop reinforcement")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        try coordinator.startReinforcement(agent: agent, options: ReinforcementOptions())
        try await waitUntil { !coordinator.reinforcementMetrics.isEmpty }
        await coordinator.stopAndWait()
        #expect(!coordinator.isBusy && coordinator.failure == nil)
        let run = try #require(try await store.snapshot().learningRuns.first)
        #expect(run.status == .cancelled && run.checkpointID != nil)
        #expect(try fixture.operations().suffix(2) == ["cancel", "shutdown"])
    }

    @Test func reinforcementCanStartFromACopiedAgentsSharedBehavioralCheckpoint() async throws {
        let fixture = try Fixture("earlyTerminal")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Shared BC origin")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var behavior = BehaviorOptions(); behavior.source = .practice; behavior.epochs = 1
        try coordinator.start(agent: agent, options: behavior, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let snapshot = try await store.snapshot()
        let original = try #require(snapshot.agents.first)
        let checkpointID = try #require(original.selectedCheckpointID)
        let copied = try await store.duplicateAgent(original)
        var options = ReinforcementOptions(); options.initialCheckpointID = checkpointID; options.iterations = 1
        try coordinator.startReinforcement(agent: copied, options: options)
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil && coordinator.activeRun?.kind == .reinforcement)
        let result = try await store.snapshot()
        #expect(result.agents.first { $0.id == original.id }?.selectedCheckpointID == checkpointID)
        #expect(result.agents.first { $0.id == copied.id }?.selectedCheckpointID != checkpointID)
        #expect(try fixture.operations().suffix(3) == ["checkpoint.inspect", "train.reinforcement", "shutdown"])
    }

    @Test func completedPipelinePublishesCatalogAndSurvivesEarlyTerminalDelivery() async throws {
        let fixture = try Fixture("earlyTerminal")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Native coordinator")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice; options.epochs = 1
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil)
        let snapshot = try await store.snapshot()
        let run = try #require(snapshot.learningRuns.first)
        #expect(run.status == .completed && run.checkpointID != nil)
        #expect(snapshot.checkpoints.count == 2)
        #expect(snapshot.agents.first?.selectedCheckpointID == run.checkpointID)
        #expect(coordinator.metrics.first?.epoch == 1)
        let saved = fixture.root.appendingPathComponent("Jobs/\(run.id.uuidString.lowercased())/configuration.json")
        #expect(try await LearningFiles.read(saved).fields?["verificationMode"] == .bool(true))
        #expect(try fixture.operations() == ["checkpoint.create", "train.behavioral", "shutdown"])
        #expect(FileManager.default.fileExists(atPath: fixture.directory.appendingPathComponent("shutdown").path))
        var checkpoint = try #require(snapshot.checkpoints.first { $0.id == run.checkpointID })
        checkpoint.name = "Renamed in an already-open view"
        checkpoint.createdAt += 0.0000001
        try coordinator.evaluate(checkpoint: checkpoint, split: "validation")
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.evaluation?.meanNLL == 0.75)
        #expect(coordinator.evaluation?.available == true)
        #expect(try await store.snapshot().learningRuns.count == 1) // Evaluation is not a fabricated training run.
    }

    @Test func cancellationWaitsForCheckpointPublicationBeforeShutdown() async throws {
        let fixture = try Fixture("waitForCancel")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Cancellation")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.metrics.isEmpty }
        await coordinator.stopAndWait()
        #expect(!coordinator.isBusy && coordinator.failure == nil)
        let snapshot = try await store.snapshot()
        let run = try #require(snapshot.learningRuns.first)
        #expect(run.status == .cancelled && run.checkpointID != nil)
        #expect(snapshot.checkpoints.contains { $0.id == run.checkpointID })
        let operations = try fixture.operations()
        #expect(operations.suffix(2) == ["cancel", "shutdown"])
        #expect(FileManager.default.fileExists(atPath: fixture.directory.appendingPathComponent("shutdown").path))
    }

    @Test func datasetFailureStopsBeforeCreatingOrSelectingAModel() async throws {
        let fixture = try Fixture("prepareFailure")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Incompatible recording")
        try await store.save(agent)
        var recording = RecordingManifest(name: "Recorded controls", environment: EnvironmentDocument(name: "Environment", kind: .window))
        recording.frameCount = 1; recording.storedBytes = 4; recording.firstObservedNanos = 1; recording.stoppedNanos = 2; recording.status = .complete
        var options = BehaviorOptions(); options.recordingIDs = [recording.id]
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        try coordinator.start(agent: agent, options: options, recordings: [recording])
        try await waitUntil { !coordinator.isBusy }
        let snapshot = try await store.snapshot()
        #expect(snapshot.learningRuns.first?.status == .failed)
        #expect(coordinator.failure?.contains("Incompatible recorded control") == true)
        #expect(snapshot.checkpoints.isEmpty && snapshot.agents.first?.selectedCheckpointID == nil)
        #expect(try fixture.operations() == ["dataset.prepare", "shutdown"])
    }

    @Test func mismatchedCheckpointCannotEnterCatalog() async throws {
        let fixture = try Fixture("wrongCheckpoint")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Checkpoint identity")
        try await store.save(agent)
        var options = BehaviorOptions(); options.source = .practice
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let snapshot = try await store.snapshot()
        #expect(snapshot.learningRuns.first?.status == .failed)
        #expect(snapshot.checkpoints.count == 1 && snapshot.checkpoints.first?.kind == "initial")
        #expect(snapshot.agents.first?.selectedCheckpointID == nil)
    }

    @Test(arguments: ["wrongRequest", "wrongJob", "wrongAckRun"])
    func mismatchedJobIdentityCannotAdvancePipeline(mode: String) async throws {
        let fixture = try Fixture(mode)
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Request identity")
        try await store.save(agent)
        var options = BehaviorOptions(); options.source = .practice
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        #expect(try await store.snapshot().learningRuns.first?.status == .failed)
        #expect(try await store.snapshot().checkpoints.isEmpty)
        let expected = mode == "wrongAckRun" ? ["checkpoint.create"] : ["checkpoint.create", "shutdown"]
        #expect(try fixture.operations() == expected)
        if mode == "wrongAckRun" {
            // A wrong envelope run closes the malformed transport immediately;
            // it must not send another request over that channel. Shutdown still
            // joins the actual child before the coordinator declares completion.
            let pid = Int32(try String(contentsOf: fixture.directory.appendingPathComponent("pid"), encoding: .utf8))!
            #expect(kill(pid, 0) == -1 && errno == ESRCH)
        }
    }

    @Test func behavioralResumeUsesSavedDatasetOptimizerConfigurationAndTarget() async throws {
        let fixture = try Fixture("waitForCancel")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Resume behavior")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var initial = BehaviorOptions(); initial.source = .practice; initial.epochs = 7; initial.learningRate = 0.0004
        try coordinator.start(agent: agent, options: initial, recordings: [])
        try await waitUntil { !coordinator.metrics.isEmpty }
        await coordinator.stopAndWait()
        let before = try await store.snapshot()
        let stopped = try #require(before.learningRuns.first)
        #expect(stopped.status == .cancelled)
        let checkpoint = try #require(before.checkpoints.first { $0.id == stopped.checkpointID })
        let copy = try await store.duplicateAgent(try #require(before.agents.first))
        var resume = BehaviorOptions()
        resume.resume = true; resume.initialCheckpointID = checkpoint.id
        resume.epochs = 99; resume.learningRate = -.infinity; resume.source = .practice; resume.practiceEpisodes = -10; resume.recordingIDs = []
        try coordinator.start(agent: copy, options: resume, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil)
        let after = try await store.snapshot()
        let run = try #require(after.learningRuns.first { $0.agentID == copy.id })
        #expect(run.status == .completed && run.sourceKind == "practice_oracle")
        let stored = try await LearningFiles.read(fixture.root.appendingPathComponent("Jobs/\(run.id.uuidString.lowercased())/configuration.json"))
        #expect(stored.fields?["resume"] == .bool(true))
        #expect(stored.fields?["training"]?.fields?["epochs"] == .integer(7))
        #expect(stored.fields?["training"]?.fields?["learning_rate"]?.double == 0.0004)
        #expect(stored.fields?["dataset"]?.fields?["kind"] == .string("practice_oracle"))
        #expect(after.agents.first { $0.id == agent.id }?.selectedCheckpointID == checkpoint.id)
        #expect(try fixture.operations() == ["checkpoint.create", "train.behavioral", "cancel", "shutdown", "checkpoint.inspect", "train.behavioral", "shutdown"])
    }

    @Test func cancellationBeforeTrainingDoesNotClaimACheckpointWasSaved() async throws {
        let fixture = try Fixture("waitBeforeTraining")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Early cancellation")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { coordinator.phase == "Learning from demonstrations…" }
        await coordinator.stopAndWait()
        let snapshot = try await store.snapshot()
        #expect(snapshot.learningRuns.first?.status == .cancelled)
        #expect(snapshot.learningRuns.first?.checkpointID == nil && snapshot.checkpoints.count == 1)
        #expect(coordinator.phase == "Training stopped before a checkpoint was saved" || coordinator.phase == "Training stopped")
        #expect(coordinator.failure == nil)
    }

    @Test func cancelledCheckpointInspectionRemainsCancellation() async throws {
        let fixture = try Fixture("inspectCancelled")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Inspection cancellation")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        options.initialCheckpointID = try #require(try await store.snapshot().checkpoints.first { $0.kind == "behavioral" }?.id)
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.activeRun?.status == .cancelled && coordinator.failure == nil)
        #expect(try await store.snapshot().checkpoints.count == 2)
        #expect(try fixture.operations().suffix(2) == ["checkpoint.inspect", "shutdown"])
    }

    @Test func copiedAgentEvaluatesAndLearnsFromSharedImmutableCheckpoint() async throws {
        let fixture = try Fixture("earlyTerminal")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Original learning project")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice; options.epochs = 1
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let snapshot = try await store.snapshot()
        let original = try #require(snapshot.agents.first)
        let checkpoint = try #require(snapshot.checkpoints.first { $0.id == original.selectedCheckpointID })
        let copy = try await store.duplicateAgent(original)
        #expect(copy.selectedCheckpointID == checkpoint.id)
        try coordinator.evaluate(checkpoint: checkpoint, agentID: copy.id, split: "validation")
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil && coordinator.evaluation?.available == true)
        #expect(coordinator.resultAgentID == copy.id)
        options.initialCheckpointID = checkpoint.id
        try coordinator.start(agent: copy, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil && coordinator.activeRun?.status == .completed)
        let final = try await store.snapshot()
        let selected = try #require(final.agents.first { $0.id == copy.id }?.selectedCheckpointID)
        #expect(selected != checkpoint.id)
        #expect(final.agents.first { $0.id == original.id }?.selectedCheckpointID == checkpoint.id)
        #expect(final.checkpoints.first { $0.id == selected }?.agentID == copy.id)
        #expect(final.checkpoints.filter { $0.id == checkpoint.id }.count == 1)
    }

    @Test func unlinkedCheckpointCannotBeUsedByAnotherAgent() async throws {
        let fixture = try Fixture("earlyTerminal")
        let store = try LibraryStore(root: fixture.root)
        let owner = AgentDocument(name: "Checkpoint owner")
        let other = AgentDocument(name: "Unlinked agent")
        try await store.save(owner); try await store.save(other)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice
        try coordinator.start(agent: owner, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let checkpoint = try #require(try await store.snapshot().checkpoints.first { $0.kind == "behavioral" })
        let operations = try fixture.operations()
        try coordinator.evaluate(checkpoint: checkpoint, agentID: other.id, split: "validation")
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure?.contains("not linked") == true && coordinator.evaluation == nil)
        options.initialCheckpointID = checkpoint.id
        try coordinator.start(agent: other, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.activeRun?.status == .failed)
        #expect(try fixture.operations() == operations)
    }

    @Test func closingChildFailureDoesNotPoisonNextEvaluation() async throws {
        let fixture = try Fixture("lateFailure")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Child generation")
        try await store.save(agent)
        var options = BehaviorOptions(); options.source = .practice
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let checkpoint = try #require(try await store.snapshot().checkpoints.first { $0.kind == "behavioral" })
        try coordinator.evaluate(checkpoint: checkpoint, split: "validation")
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil && coordinator.evaluation?.available == true)
    }

    @Test func immutableRunArtifactPublicationHasOneWinnerUnderRace() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("AstraArtifact-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let destination = directory.appendingPathComponent("configuration.json")
        let winners = await withTaskGroup(of: Int?.self, returning: [Int].self) { group in
            for value in 0..<12 {
                group.addTask {
                    do {
                        try await LearningFiles.write(.object(["winner": .integer(Int64(value))]), to: destination, exclusive: true)
                        return value
                    } catch { return nil }
                }
            }
            var result: [Int] = []
            for await winner in group { if let winner { result.append(winner) } }
            return result
        }
        #expect(winners.count == 1)
        #expect(try await LearningFiles.read(destination).fields?["winner"]?.int == winners.first)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["configuration.json"])
    }

    @Test func capabilityDiscoveryIsReadOnlyAndRejectsIndexMismatch() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("AstraCapabilities-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = RecordingManifest(name: "Controls", environment: EnvironmentDocument(name: "Fixture", kind: .practice))
        let package = directory.appendingPathComponent("Recordings/\(manifest.id.uuidString).astrarecord")
        let writer = try RecordingWriter(directory: package, manifest: manifest)
        try writer.append(events: [RawInputEvent(sequence: 0, eventNanos: 1, observedNanos: 1, origin: .physical, kind: .keyDown, keyCode: 13),
                                   RawInputEvent(sequence: 1, eventNanos: 2, observedNanos: 2, origin: .physical, kind: .buttonDown, button: 0)])
        let sealed = try writer.finish(at: 3, status: .failed, issue: "Input-only test fixture")
        let index = try sealed.indexURL(in: package)
        let before = try Data(contentsOf: index)
        let capabilities = try await LearningFiles.recordedCapabilities([sealed], root: directory)
        #expect(capabilities.keyCodes == [13] && capabilities.mouseButtons == [0])
        #expect(try Data(contentsOf: index) == before)
        let database = try SQLiteDatabase(url: index)
        try database.execute("UPDATE events SET observed=observed+1 WHERE sequence=0")
        try database.checkpoint(); try database.close()
        await #expect(throws: AstraError.self) { try await LearningFiles.recordedCapabilities([sealed], root: directory) }
    }
}
