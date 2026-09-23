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
boundary = threading.Event()
waiting_sent = threading.Event()
active_request = active_job = None
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
        checkpoint=json.loads((pathlib.Path(value['checkpointPath'])/'manifest.json').read_text())
        score=dict(available=False,split=value['split'],reason='No independent sessions in this split') if mode=='evaluationUnavailable' else dict(available=True,split=value['split'],decisions=2,meanNLL=.75)
        result=dict(evaluation=score,checkpointID=checkpoint['id'],provenance='practice_oracle',
                    datasetID=str(uuid.uuid5(uuid.NAMESPACE_URL,json.dumps(value['dataset'],sort_keys=True))))
    elif op in ('train.reinforcement.external','checkpoint.externalBoundary'):
        preserving=op=='checkpoint.externalBoundary'
        package=json.loads((pathlib.Path(value['auditPath' if preserving else 'rolloutPath'])/'manifest.json').read_text())
        metric=dict(iteration=1,elapsed_seconds=1.5,mean_reward=.25,mean_value_loss=.3,mean_policy_loss=-.1,
                    maximum_sampled_kl=.001,clip_fraction=.05,optimizer_updates=1,decisions=2,peak_memory_bytes=1024)
        if not preserving:
            send('job.progress',dict(jobID=job,phase='updating',sourceKind='external_rollout',**metric),request)
            if mode=='externalWaitCancel':cancel.wait(10)
            send('job.progress',dict(jobID=job,phase='waiting_for_actor_boundary',sourceKind='external_rollout',**metric),request)
            waiting_sent.set()
            if not boundary.wait(5):
                send('job.failed',dict(jobID=job,error=dict(message='No fixture boundary')),request);return
        initial=json.loads((pathlib.Path(value['checkpointPath'])/'manifest.json').read_text())
        manifest=artifact(value['destination'],'reinforcement',initial['model'],initial['actions'])
        progress=package['actorProgress']
        if mode=='externalWrongProgress':progress={**progress,'drawIndex':progress['drawIndex']+1}
        manifest.update(parentID=initial['id'],metrics=dict(iteration=0 if preserving else 1,optimizer_updates=0 if preserving or cancel.is_set() else 1))
        (pathlib.Path(value['destination'])/'manifest.json').write_text(json.dumps(manifest))
        result=dict(manifest=manifest,checkpointPath=value['destination'],checkpointPublished=True,
            parameterCount=10,cancelled=cancel.is_set(),sourceKind='external_rollout',provenance='external_rollout',
            resumable=True,requiresEnvironmentReset=True,actorProgress=progress,
            boundaryCollectionID=package['id'],metrics=None if preserving or cancel.is_set() else metric)
        if preserving:result['boundaryOnly']=True
        else:result['rolloutID']=package['rolloutID']
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
    if kind=='job.externalBoundary':
        assert request['runID']==active_request['runID'] and request['payload']['jobID']==active_job
        assert request['payload']['auditPath']==active_request['payload']['rolloutPath']
        send('ack',dict(jobID=str(uuid.uuid4()) if mode=='externalWrongAck' else active_job,status='boundary_queued'),request)
        boundary.set();continue
    job=str(uuid.uuid4())
    active_request,active_job=request,job
    if mode=='externalEarlyBoundary' and kind=='train.reinforcement.external':
        thread=threading.Thread(target=perform,args=(request,job));thread.start()
        assert waiting_sent.wait(5)
        send('ack',dict(jobID=job,status='queued'),request);continue
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

@MainActor private func makeExternalBatch(fixture: Fixture, store: LibraryStore, agent: AgentDocument) async throws -> DesktopLearningBatch {
    let identity = DesktopEvidenceIdentity(runID: UUID(), clockID: UUID(), environmentID: UUID(), actorSourceID: UUID(), environmentSourceID: UUID())
    let checkpoint = CheckpointDocument(id: UUID(), agentID: agent.id, runID: nil, name: "Behavior fixture", kind: "behavioral",
        trainingStep: 1, policySignature: String(repeating: "a", count: 64), parameterCount: 10)
    try await store.saveCheckpoint(checkpoint)
    let source = fixture.root.appendingPathComponent("Models/\(checkpoint.id.uuidString.lowercased())")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try JSONEncoder().encode(JSONValue.object(["id": .string(checkpoint.id.uuidString.lowercased()), "model": .object([:]), "actions": .object([:])]))
        .write(to: source.appendingPathComponent("manifest.json"))
    let id = UUID(), episode = UUID(), destination = fixture.root.appendingPathComponent("Rollouts/\(id.uuidString.lowercased())")
    let progress: JSONValue = .object(["schemaVersion": .integer(1), "runID": .string(identity.runID.uuidString.lowercased()),
        "rngStreamID": .string(UUID().uuidString.lowercased()), "drawIndex": .integer(1),
        "rngState": .array([.integer(10), .integer(20)]), "actorResetGeneration": .integer(1)])
    let binding: JSONValue = .object(["runID": .string(identity.runID.uuidString.lowercased()),
        "clockID": .string(identity.clockID.uuidString.lowercased()), "actorSourceID": .string(identity.actorSourceID.uuidString.lowercased()),
        "environmentSourceID": .string(identity.environmentSourceID.uuidString.lowercased()),
        "policyID": .string(checkpoint.id.uuidString.lowercased()), "policySignature": .string(checkpoint.policySignature),
        "purpose": .string("learning"), "model": .object([:]), "training": .object([:]), "environment": .object([:]), "contextIDs": .array([])])
    let manifest: JSONValue = .object(["schemaVersion": .integer(1), "id": .string(id.uuidString.lowercased()),
        "rolloutID": .string(UUID().uuidString.lowercased()), "status": .string("sealed"), "controlClosureKnown": .bool(true),
        "actorProgress": progress, "decisions": .integer(2), "binding": binding,
        "actorSampling": .array([.string("categorical"), .integer(1), .string("none"), .integer(1)])])
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    try JSONEncoder().encode(manifest).write(to: destination.appendingPathComponent("manifest.json"))
    let result = WireMessage(kind: "collector.sealed", sequence: 0, runID: identity.runID, payload: .object([
        "collectionID": .string(id.uuidString.lowercased()), "path": .string(destination.path), "manifest": manifest,
        "actorProgress": progress, "learningEligible": .bool(true), "controlClosureKnown": .bool(true)]))
    let join = DesktopEpisodeJoin(runID: identity.runID, episodeID: episode, generationID: UUID(), actorJoined: true,
        controlJoined: true, manualProducerJoined: true, predictionResolved: true, cleanupConfirmed: true,
        stoppedNanos: 100, lastProducedSequence: 1, stop: .semanticBoundary(100))
    let state = PolicyActorState(nextPacketSequence: 2, nextDrawIndex: 2, actorResetGeneration: 1, episodeStep: 2,
        episodeID: episode, stateID: UUID(), actorProgress: progress, sampledProgressKnown: true, stopped: false, joined: false)
    return try DesktopLearningBatch(identity: identity, checkpoint: checkpoint, destination: destination, result: result, joins: [join], actorState: state)
}

@MainActor private func waitUntil(_ condition: () -> Bool) async throws {
    let limit = ContinuousClock.now.advanced(by: .seconds(10))
    while !condition() {
        guard ContinuousClock.now < limit else { throw AstraError("test.timeout", "Coordinator did not reach the expected state") }
        try await Task.sleep(for: .milliseconds(10))
    }
}

@Suite(.serialized) @MainActor struct LearningCoordinatorTests {
    @Test func stoppedActorBoundaryPublishesWithoutLaunchingAnUpdate() async throws {
        let fixture = try Fixture("externalNormal"), store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Stopped collection"); try await store.save(agent)
        let batch = try await makeExternalBatch(fixture: fixture, store: store, agent: agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        let result = try await coordinator.preserveExternalBoundary(agent: agent, boundary: batch.boundary, resume: false, validateBoundary: {})
        let checkpoint = try #require(result.checkpoint)
        #expect(result.actorProgress == batch.actorProgress && checkpoint.trainingStep == batch.checkpoint.trainingStep)
        #expect(try fixture.operations() == ["checkpoint.externalBoundary", "shutdown"])
        #expect(coordinator.activeRun?.updates == 0 && coordinator.reinforcementMetrics.isEmpty)
        #expect(coordinator.phase == "Desktop checkpoint saved · ready for a fresh reset")
        let configuration = try await LearningFiles.read(fixture.root.appendingPathComponent("Jobs/\(result.runID.uuidString.lowercased())/configuration.json"))
        #expect(configuration.fields?["operation"] == .string("checkpoint.externalBoundary"))
        #expect(configuration.fields?["rolloutID"] == .null)
    }

    @Test func desktopPolicyPreparationCreatesAndInspectsWithoutClaimingTraining() async throws {
        let fixture = try Fixture("earlyEverything"), store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Fresh desktop policy"); try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        let actions = ActionCapabilities(keyCodes: [13])
        let first = try await coordinator.prepareDesktopPolicy(agent: agent, checkpoint: nil,
            model: BehaviorOptions().model, actions: actions, seed: 17)
        #expect(first.checkpoint.document.kind == "initial" && first.checkpoint.document.runID == nil)
        #expect(first.checkpoint.document.parameterCount == 10 && !coordinator.isBusy)
        #expect(first.manifest.fields?["actions"]?.fields?["scrollUnitsPerPoint"]?.int == 8)
        let second = try await coordinator.prepareDesktopPolicy(agent: agent, checkpoint: first.checkpoint.document,
            model: .object([:]), actions: .init(), seed: -1)
        #expect(second.checkpoint.document.matchesIdentity(of: first.checkpoint.document))
        #expect(second.manifest == first.manifest)
        #expect(try await store.snapshot().learningRuns.isEmpty)
        #expect(try await store.snapshot().checkpoints.count == 1)
        #expect(try fixture.operations() == ["checkpoint.create", "shutdown", "checkpoint.inspect", "shutdown"])
    }

    @Test func cancelledDesktopPolicyInspectionDoesNotStartAnActorOrAnotherModel() async throws {
        let fixture = try Fixture("inspectCancelled"), store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Cancelled preparation"); try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        let actions = ActionCapabilities(keyCodes: [13])
        let first = try await coordinator.prepareDesktopPolicy(agent: agent, checkpoint: nil,
            model: BehaviorOptions().model, actions: actions, seed: 0)
        await #expect(throws: CancellationError.self) {
            try await coordinator.prepareDesktopPolicy(agent: agent, checkpoint: first.checkpoint.document,
                model: .object([:]), actions: actions, seed: 0)
        }
        #expect(coordinator.failure == nil && !coordinator.isBusy)
        #expect(try await store.snapshot().checkpoints.count == 1)
        #expect(try fixture.operations().suffix(2) == ["checkpoint.inspect", "shutdown"])
    }

    @Test(arguments: ["externalNormal", "externalEarlyBoundary", "externalWaitCancel"])
    func externalUpdatePublishesOnlyAfterItsCorrelatedBoundary(mode: String) async throws {
        let fixture = try Fixture(mode), store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Desktop fixture"); try await store.save(agent)
        let batch = try await makeExternalBatch(fixture: fixture, store: store, agent: agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        let operation = Task { try await coordinator.updateExternal(agent: agent, batch: batch, resume: false, validateBoundary: {}) }
        if mode == "externalWaitCancel" {
            try await waitUntil { coordinator.phase == "Updating the learner…" }
            await coordinator.stopAndWait()
        }
        let result = try await operation.value
        #expect(!coordinator.isBusy && coordinator.failure == nil)
        #expect(result.cancelled == (mode == "externalWaitCancel"))
        let checkpoint = try #require(result.checkpoint)
        #expect(checkpoint.kind == "reinforcement" && checkpoint.runID == result.runID)
        #expect(result.actorProgress == batch.actorProgress)
        #expect(try await store.snapshot().agents.first?.selectedCheckpointID == checkpoint.id)
        let requests = try String(contentsOf: fixture.log, encoding: .utf8).split(separator: "\n")
            .map { try JSONDecoder().decode(WireMessage.self, from: Data($0.utf8)) }
        let training = try #require(requests.first { $0.kind == "train.reinforcement.external" })
        let boundary = try #require(requests.first { $0.kind == "job.externalBoundary" })
        #expect(training.runID == boundary.runID && training.runID != batch.identity.runID)
        #expect(boundary.payload.fields?["auditPath"] == .string(batch.path.path))
        #expect(requests.filter { $0.kind == "job.externalBoundary" }.count == 1)
        #expect(requests.last?.kind == "shutdown")
        #expect(coordinator.reinforcementMetrics.first?.iteration == 1)
    }

    @Test(arguments: ["externalWrongAck", "externalWrongProgress"])
    func externalPublicationMismatchCannotEnterCatalog(mode: String) async throws {
        let fixture = try Fixture(mode), store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Boundary mismatch"); try await store.save(agent)
        let batch = try await makeExternalBatch(fixture: fixture, store: store, agent: agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        await #expect(throws: AstraError.self) {
            try await coordinator.updateExternal(agent: agent, batch: batch, resume: false, validateBoundary: {})
        }
        #expect(!coordinator.isBusy && coordinator.failure != nil)
        #expect(try await store.snapshot().checkpoints.count == 1)
        #expect(coordinator.activeRun?.status == .failed && coordinator.activeRun?.checkpointID == nil)
        #expect(try fixture.operations().last == "shutdown")
    }

    @Test func externalUpdateRefusesLostActorPauseBeforeLaunchingCompute() async throws {
        let fixture = try Fixture("externalNormal"), store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Pause lost"); try await store.save(agent)
        let batch = try await makeExternalBatch(fixture: fixture, store: store, agent: agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        await #expect(throws: AstraError.self) {
            try await coordinator.updateExternal(agent: agent, batch: batch, resume: false, validateBoundary: {
                throw AstraError("test.actorNotPaused", "The actor pause is no longer held")
            })
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.log.path))
        #expect(coordinator.activeRun?.status == .failed && !coordinator.isBusy)
    }

    @Test func recordedRangeSnapshotAndExactResumeIgnoreLaterAgentSelectionEdits() async throws {
        let fixture = try Fixture("waitForCancel")
        let library = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Selection snapshot")
        try await library.save(agent)
        var recording = RecordingManifest(name: "Protocol fixture", environment: .init(name: "Virtual source", kind: .desktop))
        recording.frameCount = 1; recording.storedBytes = 4; recording.firstObservedNanos = 1_000_000_000
        recording.stoppedNanos = 5_000_000_000; recording.status = .complete
        try await library.saveRecording(recording, linkTo: agent.id)
        let field = ContextFieldDocument(name: "Task", values: [.init(name: "Inspect")])
        let vocabulary = ContextVocabulary(fields: [field])
        let selection = RecordingTrainingSelection(ranges: [.init(startNanos: 1_000_000_000, endNanos: 2_000_000_000), .init(startNanos: 3_000_000_000, endNanos: 4_000_000_000)], contextValues: [field.id: field.values[0].id])
        try await library.saveRecordingSelection(selection, recordingID: recording.id, agentID: agent.id)
        let coordinator = LearningCoordinator(store: library, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.recordingIDs = [recording.id]; options.contextVocabulary = vocabulary
        try coordinator.start(agent: agent, options: options, recordings: [recording], selections: [recording.id: selection])
        try await library.saveRecordingSelection(.whole, recordingID: recording.id, agentID: agent.id)
        try await waitUntil { (coordinator.activeRun?.updates ?? 0) > 0 || !coordinator.isBusy }
        await coordinator.stopAndWait()
        #expect(coordinator.failure == nil)
        let run = try #require(coordinator.activeRun), checkpointID = try #require(run.checkpointID)
        let expected: JSONValue = .array([try selection.payload(recordingID: recording.id, vocabulary: vocabulary)])
        let configuration = try await LearningFiles.read(fixture.root.appendingPathComponent("Jobs/\(run.id.uuidString.lowercased())/configuration.json"))
        #expect(configuration.fields?["sourceSelections"] == expected)
        #expect(try ContextVocabulary.from(model: configuration.required("model")) == vocabulary)
        let requests = try String(contentsOf: fixture.log, encoding: .utf8).split(separator: "\n").map { try JSONDecoder().decode(WireMessage.self, from: Data($0.utf8)) }
        #expect(requests.first { $0.kind == "dataset.prepare" }?.payload.fields?["selections"] == expected)
        try await library.unlinkRecording(recording.id, from: agent.id)
        options.resume = true; options.initialCheckpointID = checkpointID; options.recordingIDs = []
        try coordinator.start(agent: agent, options: options, recordings: [], selections: [:])
        try await waitUntil { !coordinator.isBusy }
        #expect(coordinator.failure == nil)
        let resumed = try #require(coordinator.activeRun)
        let resumedConfiguration = try await LearningFiles.read(fixture.root.appendingPathComponent("Jobs/\(resumed.id.uuidString.lowercased())/configuration.json"))
        #expect(resumedConfiguration.fields?["sourceSelections"] == expected)
        #expect(resumedConfiguration.fields?["dataset"] == configuration.fields?["dataset"])
        #expect(try fixture.operations().filter { $0 == "dataset.prepare" }.count == 1)
    }

    @Test func unreadableSavedRangeCannotSilentlyTrainTheWholeRecording() async throws {
        let fixture = try Fixture("earlyTerminal"), agent = AgentDocument(name: "Corrupt range")
        let store = try LibraryStore(root: fixture.root)
        try await store.save(agent)
        var recording = RecordingManifest(name: "Protocol fixture", environment: .init(name: "Virtual", kind: .desktop))
        recording.frameCount = 1; recording.storedBytes = 4; recording.firstObservedNanos = 1; recording.stoppedNanos = 100; recording.status = .complete
        var options = BehaviorOptions(); options.recordingIDs = [recording.id]
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        #expect(throws: AstraError.self) { try coordinator.start(agent: agent, options: options, recordings: [recording], selections: [:]) }
        #expect(!coordinator.isBusy && coordinator.activeRun == nil && !FileManager.default.fileExists(atPath: fixture.log.path))
    }

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

    @Test func comparisonUsesOneExplicitDatasetAndPersistsFailuresAlongsideScores() async throws {
        let fixture = try Fixture("earlyTerminal")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "Comparison")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice; options.epochs = 1
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let first = try #require(try await store.snapshot().checkpoints.first { $0.kind == "behavioral" })
        options.seed = 10
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let second = try #require(try await store.snapshot().checkpoints.first { $0.kind == "behavioral" && $0.id != first.id })
        let incompatible = CheckpointDocument(id: UUID(), agentID: agent.id, runID: first.runID, name: "Different controls", kind: "initial",
            trainingStep: 0, policySignature: String(repeating: "b", count: 64), parameterCount: 10)
        try await store.saveCheckpoint(incompatible)
        try coordinator.evaluate(checkpoints: [first, second, incompatible], agentID: agent.id, datasetCheckpoint: first, split: "test")
        try await waitUntil { !coordinator.isBusy }
        let history = try await LibraryStore(root: fixture.root).snapshot().evaluations
        #expect(history.count == 3 && Set(history.map(\.comparisonID)).count == 1)
        let scored = history.filter { $0.status == .completed }
        #expect(scored.count == 2 && scored[0].comparisonIssue(with: scored[1]) == nil)
        #expect(history.first { $0.checkpointID == incompatible.id }?.status == .failed)
        #expect(history.first { $0.checkpointID == incompatible.id }?.meanNLL == nil)
        let requests = try String(contentsOf: fixture.log, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8))
        }.filter { $0.fields?["kind"] == .string("evaluate.behavioral") }
        #expect(requests.count == 2)
        let expected = try await LearningFiles.read(fixture.root.appendingPathComponent("Jobs/\(first.runID!.uuidString.lowercased())/configuration.json")).required("dataset")
        #expect(requests.allSatisfy { $0.fields?["payload"]?.fields?["dataset"] == expected })
        #expect(history.allSatisfy { $0.protocolDefinition.sourceRunID == first.runID && $0.protocolDefinition.split == "test" })
    }

    @Test func unavailableEvaluationPersistsWithoutInventingMetrics() async throws {
        let fixture = try Fixture("evaluationUnavailable")
        let store = try LibraryStore(root: fixture.root)
        let agent = AgentDocument(name: "No held-out sessions")
        try await store.save(agent)
        let coordinator = LearningCoordinator(store: store, root: fixture.root, bundle: fixture.bundle, changed: {})
        var options = BehaviorOptions(); options.source = .practice; options.epochs = 1
        try coordinator.start(agent: agent, options: options, recordings: [])
        try await waitUntil { !coordinator.isBusy }
        let checkpoint = try #require(try await store.snapshot().checkpoints.first { $0.kind == "behavioral" })
        try coordinator.evaluate(checkpoint: checkpoint, split: "validation")
        try await waitUntil { !coordinator.isBusy }
        let document = try #require(try await store.snapshot().evaluations.first)
        #expect(document.status == .unavailable && document.datasetID != nil)
        #expect(document.meanNLL == nil && document.decisions == nil)
        #expect(document.issue == "No independent sessions in this split")
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

    @Test func capabilityDiscoveryExcludesGapKeysAndKeepsPointerAnchorsForSelectedClicks() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("AstraSelectedControls-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = RecordingManifest(name: "Selected controls", environment: .init(name: "Fixture", kind: .practice))
        let package = directory.appendingPathComponent("Recordings/\(manifest.id.uuidString).astrarecord")
        let writer = try RecordingWriter(directory: package, manifest: manifest)
        let surface = SurfaceDescriptor(id: "fixture", globalBounds: .init(x: 0, y: 0, width: 2, height: 2), pixelWidth: 2, pixelHeight: 2)
        try writer.append(FrameArchive.prepare(pixels: Data(repeating: 0, count: 16), metadata: .init(eventNanos: 1, observedNanos: 1, surface: surface, byteCount: 16)))
        try writer.append(events: [
            .init(sequence: 0, eventNanos: 5, observedNanos: 5, origin: .reconciliation, kind: .flags, keyCode: 57, modifiers: 0),
            .init(sequence: 1, eventNanos: 10, observedNanos: 10, origin: .physical, kind: .keyDown, keyCode: 57),
            .init(sequence: 2, eventNanos: 22, observedNanos: 22, origin: .physical, kind: .keyDown, keyCode: 13),
            .init(sequence: 3, eventNanos: 25, observedNanos: 25, origin: .physical, kind: .buttonDown, button: 0, x: 1, y: 1),
            .init(sequence: 4, eventNanos: 30, observedNanos: 30, origin: .physical, kind: .keyDown, keyCode: 59),
            .init(sequence: 5, eventNanos: 21, observedNanos: 50, origin: .physical, kind: .keyDown, keyCode: 1)])
        let sealed = try writer.finish(at: 100, status: .complete)
        let selection = RecordingTrainingSelection(ranges: [.init(startNanos: 20, endNanos: 30)])
        let result = try await LearningFiles.recordedCapabilities([sealed], root: directory, selections: [sealed.id: selection])
        #expect(result.keyCodes == [1, 13] && result.mouseButtons == [0] && result.absolutePointer)
    }
}
