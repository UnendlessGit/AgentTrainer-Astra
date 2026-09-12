import Foundation
import Testing
import AstraCore
import AstraPlatform
@testable import AgentTrainerAstra

private actor CollectorHarness {
    let mode: String
    var events: ComputeProcess.EventHandler?
    var runID: UUID?
    var collectionID: UUID?
    var requests: [(String, JSONValue)] = []
    var ringPath: String?
    var firstLease: SharedFrameReference?
    var deliveredPixels: Data?
    var blocked = false
    var suspended: CheckedContinuation<Void, Never>?
    var exited = false
    init(_ mode: String = "normal") { self.mode = mode }
    nonisolated var factory: CollectorRuntime.Factory {
        { events, _ in
            CollectorRuntime(start: { await self.start(events) }, request: { kind, payload, run in
                try await self.request(kind, payload, run)
            }, shutdown: { await self.shutdown() })
        }
    }
    func start(_ events: @escaping ComputeProcess.EventHandler) -> WireMessage {
        self.events = events
        return WireMessage(kind: "hello", sequence: 0, payload: .object(["role": .string("collector"), "protocolVersion": .integer(1)]))
    }
    func request(_ kind: String, _ payload: JSONValue, _ run: UUID) async throws -> WireMessage {
        requests.append((kind, payload)); runID = run
        if kind == "collector.prepare" {
            let path = try payload.required("destination").decode(String.self)
            collectionID = UUID(uuidString: URL(fileURLWithPath: path).lastPathComponent)
            ringPath = try payload.required("rings").decode([[String: String]].self)[0]["path"]
            if mode == "cancelPrepare" {
                blocked = true
                await withCheckedContinuation { suspended = $0 }
            }
            return ack(run, "ready", extra: ["collectionVersion": .integer(1)])
        }
        if kind == "collector.begin", mode == "blocked" {
            blocked = true
            await withCheckedContinuation { suspended = $0 }
        }
        if kind == "collector.actor" {
            let frames = try payload.required("observation").required("frames").decode([JSONValue].self)
            let reference = try frames[0].required("reference").decode(SharedFrameReference.self)
            guard try frames[0].required("metadata").decode(FrameMetadata.self) == reference.metadata else {
                throw AstraError("fixture.frameMetadata", "Outer frame metadata differs from the leased raw pixels")
            }
            firstLease = reference
            let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: ringPath!)); defer { try? file.close() }
            try file.seek(toOffset: UInt64(reference.offset)); deliveredPixels = try file.read(upToCount: reference.size)
            if mode == "transportFailure" { throw AstraError("fixture.transport", "Fixture collector pipe closed") }
            let release = WireMessage(kind: "collector.framesConsumed", sequence: 2, runID: run,
                payload: .object(["collectionID": .string(collectionID!.uuidString),
                    "observationID": try payload.required("observation").required("id"),
                    "acknowledgements": try .encode([reference.acknowledgement])]))
            events?(release)
            if mode == "duplicateRelease" { events?(release) }
        }
        if kind == "collector.finish" {
            events?(WireMessage(kind: "collector.audited", sequence: 4, runID: run,
                payload: .object(["collectionID": .string(collectionID!.uuidString), "learningEligible": .bool(false)])))
        }
        return ack(run, "queued")
    }
    func ack(_ run: UUID, _ status: String, extra: [String: JSONValue] = [:]) -> WireMessage {
        WireMessage(kind: "ack", sequence: 1, requestID: UUID(), runID: run,
            payload: .object(extra.merging(["collectionID": .string(collectionID!.uuidString), "status": .string(status)]) { _, new in new }))
    }
    func releaseBlock() { suspended?.resume(); suspended = nil }
    func shutdown() { exited = true }
}

@Test func collectorCancellationAfterPrepareAcknowledgementJoinsWithoutPublishingSession() async throws {
    let harness = CollectorHarness("cancelPrepare")
    let task = Task { try await collectorFixture(harness) }
    for _ in 0..<1000 { if await harness.blocked { break }; try await Task.sleep(for: .milliseconds(1)) }
    #expect(await harness.blocked)
    task.cancel()
    await harness.releaseBlock()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await harness.exited)
    let path = try #require(await harness.ringPath)
    #expect(!FileManager.default.fileExists(atPath: path))
}

private func collectorFixture(_ harness: CollectorHarness, maximumItems: Int = 128) async throws -> (CollectorSession, URL) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("AstraCollectorTests-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let destination = root.appendingPathComponent(UUID().uuidString.lowercased())
    do {
        let session = try await CollectorSession.start(runID: UUID(), configuration: .object(["schemaVersion": .integer(1), "destination": .string(destination.path)]),
            journalURL: root.appendingPathComponent("native.jsonl"), ringURL: root.appendingPathComponent("frames.astraring"),
            slotCapacity: 4096, maximumQueuedItems: maximumItems, factory: harness.factory, onFault: { _ in })
        return (session, root)
    } catch { try? FileManager.default.removeItem(at: root); throw error }
}

private func collectorObservation(run: UUID) throws -> InferenceCollectedObservation {
    let cutoff: UInt64 = 9_007_199_254_740_999
    let frame = FrameMetadata(eventNanos: cutoff - 10, observedNanos: cutoff - 5,
        surface: SurfaceDescriptor(id: "surface", globalBounds: .init(x: 0, y: 0, width: 32, height: 32), pixelWidth: 32, pixelHeight: 32),
        byteCount: 4096)
    var controls = ControlState(); controls.valid = true; controls.observedNanos = cutoff
    return .init(runID: run, actorInput: .object(["observationID": .string(UUID().uuidString), "episodeID": .string(UUID().uuidString),
        "cutoffNanos": .unsigned(cutoff), "geometryRevision": .unsigned(frame.surface.geometryRevision),
        "controlState": try .encode(controls), "executedEvents": .array([])]), frame: frame, pixels: Data(repeating: 19, count: 4096))
}

@Test func collectorUsesIndependentLeasesAndExactClockAndJoinsBeforeRetiring() async throws {
    let harness = CollectorHarness(), (session, root) = try await collectorFixture(harness)
    defer { try? FileManager.default.removeItem(at: root) }
    let observation = try collectorObservation(run: session.runID)
    try session.offer(.request("collector.begin", .object([:])))
    try session.offer(.actor(sourceID: UUID(), response: .object(["fixture": .bool(true)]), observation: observation))
    try session.offer(.controlAudit(.init(kind: "control.stopped", sequence: 5, runID: session.runID)))
    try session.offer(.request("collector.end", .object([:])))
    async let first = session.finish()
    async let second = session.finish()
    let (result, duplicate) = try await (first, second)
    #expect(result == duplicate)
    #expect(await harness.exited)
    #expect(await harness.deliveredPixels == observation.pixels)
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("frames.astraring").path))
    let requests = await harness.requests
    #expect(requests.map(\.0) == ["collector.prepare", "collector.begin", "collector.actor", "collector.end", "collector.finish"])
    #expect(requests.last?.1.fields?["throughSequence"] == .unsigned(3))
    let snapshot = try requests[2].1.required("observation")
    #expect(try snapshot.required("cutoffNanos").decode(UInt64.self) == 9_007_199_254_740_999)
    let journal = try String(contentsOf: session.journalURL, encoding: .utf8)
    #expect(journal.contains("control.stopped"))
    #expect(journal.contains("native.actor"))
    #expect(journal.contains("pixelsPersistedByNativeAudit"))
}

@Test func collectorBackpressureIncludesTheInFlightRequestAndStillDrainsAudit() async throws {
    let harness = CollectorHarness("blocked"), (session, root) = try await collectorFixture(harness, maximumItems: 1)
    defer { try? FileManager.default.removeItem(at: root) }
    try session.offer(.request("collector.begin", .object([:])))
    for _ in 0..<1000 { if await harness.blocked { break }; try await Task.sleep(for: .milliseconds(1)) }
    #expect(await harness.blocked)
    #expect(throws: AstraError.self) { try session.offer(.request("collector.end", .object([:]))) }
    try session.offer(.controlAudit(.init(kind: "control.stopped", sequence: 9, runID: session.runID)))
    await harness.releaseBlock()
    await #expect(throws: AstraError.self) { try await session.finish() }
    #expect(await harness.exited)
    #expect(try String(contentsOf: session.journalURL, encoding: .utf8).contains("collector.begin"))
    #expect(try String(contentsOf: session.journalURL, encoding: .utf8).contains("control.stopped"))
}

@Test(arguments: ["transportFailure", "duplicateRelease"])
func collectorFailureKeepsNativeCommandsAndFinalReceipts(mode: String) async throws {
    let harness = CollectorHarness(mode), (session, root) = try await collectorFixture(harness)
    defer { try? FileManager.default.removeItem(at: root) }
    try session.offer(.actor(sourceID: UUID(), response: .object(["packet": .object(["id": .string(UUID().uuidString)])]),
        observation: try collectorObservation(run: session.runID)))
    try session.offer(.controlAudit(.init(kind: "control.receipt", sequence: 7, runID: session.runID,
                                       payload: .object(["status": .string("executed")]))))
    try session.offer(.controlAudit(.init(kind: "control.stopped", sequence: 8, runID: session.runID)))
    await #expect(throws: AstraError.self) { try await session.finish() }
    #expect(await harness.exited)
    let journal = try String(contentsOf: session.journalURL, encoding: .utf8)
    #expect(journal.contains("native.actor"))
    #expect(journal.contains("control.receipt"))
    #expect(journal.contains("executed"))
    #expect(journal.contains("control.stopped"))
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("frames.astraring").path))
}
