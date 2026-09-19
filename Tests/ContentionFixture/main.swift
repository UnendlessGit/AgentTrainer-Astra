import Foundation
import Darwin
import AstraCore
import AstraPlatform

private struct Configuration: Decodable {
    var mode: String
    var worker: String
    var checkpoint: String
    var frames: [String]
    var ring: String
    var width: Int = 1280
    var height: Int = 720
    var relativePacing: Bool?
}
private struct Window: Decodable { var startAtNanos: UInt64; var durationNanos: UInt64 }
private extension JSONValue {
    func get<T: Decodable>(_ key: String, _ type: T.Type = T.self) throws -> T {
        guard case .object(let fields) = self, let value = fields[key] else {
            throw AstraError("benchmark.field", "Missing field \(key)")
        }
        return try value.decode(type)
    }
}
private func emit(_ object: [String: JSONValue]) throws {
    var data = try JSONEncoder().encode(JSONValue.object(object)); data.append(10)
    try FileHandle.standardOutput.write(contentsOf: data)
}
private func wait(until time: UInt64) async throws {
    let now = MonotonicClock.now
    if time > now { try await Task.sleep(for: .nanoseconds(Int64(time - now))) }
}
private func readWindow() throws -> Window {
    guard let line = readLine() else { throw AstraError("benchmark.start", "No measurement window") }
    let window = try JSONDecoder().decode(Window.self, from: Data(line.utf8))
    guard window.durationNanos > 0, window.durationNanos <= 65_000_000_000,
          window.startAtNanos > MonotonicClock.now else { throw AstraError("benchmark.window", "Invalid measurement window") }
    return window
}
private func memory() -> JSONValue {
    var value = rusage()
    getrusage(RUSAGE_SELF, &value)
    var child = rusage(); getrusage(RUSAGE_CHILDREN, &child)
    return .object(["peakRSSBytes": .integer(Int64(value.ru_maxrss)), "joinedChildPeakRSSBytes": .integer(Int64(child.ru_maxrss))])
}

@main struct ContentionFixture {
    static func main() async {
        do {
            guard CommandLine.arguments.count == 2 else { throw AstraError("benchmark.arguments", "Pass one fixture configuration") }
            let config = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
            guard config.width == 1280, config.height == 720, config.frames.count == 3 else { throw AstraError("benchmark.quality", "Expected three full-size scenes") }
            let frames = try config.frames.map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
            guard frames.allSatisfy({ $0.count == config.width * config.height * 4 }) else { throw AstraError("benchmark.pixels", "Incomplete owned BGRA pixels") }
            switch config.mode {
            case "actor": try await actor(config, frames: frames)
            case "ocr": try await ocr(config, frames: frames)
            default: throw AstraError("benchmark.mode", "Unknown fixture mode")
            }
        } catch {
            try? emit(["kind": .string("error"), "error": .string(String(describing: error))])
            exit(1)
        }
    }

    private static func actor(_ config: Configuration, frames: [Data]) async throws {
        let began = MonotonicClock.now
        let run = UUID(), episode = UUID()
        let ring = try SharedFrameRing(url: URL(fileURLWithPath: config.ring), runID: run, slotCount: 4,
                                       slotCapacity: config.width * config.height * 4)
        let process = ComputeProcess(executable: URL(fileURLWithPath: config.worker), arguments: ["--role", "actor"],
                                     expectedRole: "actor", allowedEvents: [])
        do {
            _ = try await process.start()
            let prepared = try await process.request(kind: "inference.prepare", payload: .object([
                "checkpointPath": .string(config.checkpoint),
                "ring": .object(["path": .string(ring.url.path), "ringID": .string(ring.ringID.uuidString.lowercased())]),
                "seed": .integer(817), "deterministic": .bool(false), "collection": .bool(true)]), runID: run, timeout: .seconds(120))
            let policyID: UUID = try prepared.payload.get("checkpointID")
            let signature: String = try prepared.payload.get("policySignature")
            let rngID: UUID = try prepared.payload.get("rngStreamID")
            let capabilities: ActionCapabilities = try prepared.payload.get("actions")
            let model: JSONValue = try prepared.payload.get("model")
            guard try model.get("period_ms", Int.self) == 100, try model.get("lead_ms", Int.self) == 100,
                  try model.get("packet_capacity", Int.self) == 16 else { throw AstraError("benchmark.model", "Model timing/capacity changed") }
            let reset = try await process.request(kind: "inference.reset", payload: .object([
                "confirmed": .bool(true), "episodeID": .string(episode.uuidString.lowercased()), "contextIDs": .array([])]), runID: run)
            var stateID: UUID = try reset.payload.get("stateID")
            let surface = SurfaceDescriptor(id: "benchmark", globalBounds: .init(x: 0, y: 0, width: 1280, height: 720),
                                            pixelWidth: 1280, pixelHeight: 720)
            var expectedKey: [UInt32]?
            var sequence: UInt64 = 0
            var lastCutoff: UInt64?
            func predict(index: Int, warmup: Bool) async throws -> [String: JSONValue] {
                let started = MonotonicClock.now, id = UUID(), observation = UUID()
                let metadata = FrameMetadata(id: id, eventNanos: started, observedNanos: started, surface: surface,
                                             byteCount: frames[index % 3].count, codec: "raw")
                let reference = try ring.publish(pixels: frames[index % 3], metadata: metadata)
                let cutoff = MonotonicClock.now
                var controls = ControlState(); controls.valid = true; controls.observedNanos = cutoff
                controls.pointer = .init(x: 640, y: 360)
                let response = try await process.request(kind: warmup ? "inference.warmup" : "inference.step", payload: .object([
                    "observationID": .string(observation.uuidString.lowercased()), "episodeID": .string(episode.uuidString.lowercased()),
                    "previousStateID": .string(stateID.uuidString.lowercased()), "cutoffNanos": .unsigned(cutoff),
                    "geometryRevision": .unsigned(0), "frames": .array([try .encode(reference)]),
                    "controlState": try .encode(controls), "executedEvents": .array([]), "intervalCovered": .bool(true),
                    "contextIDs": .array([])]), runID: run, timeout: .seconds(warmup ? 120 : 5))
                let received = MonotonicClock.now
                let released: [SharedFrameAcknowledgement] = try response.payload.get("releasedFrames")
                guard released == [reference.acknowledgement] else { throw AstraError("benchmark.lease", "Actor did not release the exact image lease") }
                try ring.release(released[0])
                let packet: ActionPacket = try response.payload.get("packet")
                guard response.kind == "ack", response.runID == run, packet.runID == run,
                      packet.observationID == observation, packet.sequence == sequence, packet.executeAtNanos == cutoff + 100_000_000,
                      packet.durationMs == 100, packet.geometryRevision == 0,
                      try response.payload.get("checkpointID", UUID.self) == policyID,
                      try response.payload.get("policySignature", String.self) == signature else { throw AstraError("benchmark.identity", "Actor packet identity/timing changed") }
                _ = try packet.validated(capabilities: capabilities, surfaces: [surface], capacity: 16)
                for field in ["logProbability", "value", "conditionalEntropy"] {
                    guard try response.payload.get(field, Double.self).isFinite else { throw AstraError("benchmark.nonfinite", "Nonfinite actor output") }
                }
                if warmup {
                    guard try response.payload.get("warmup", Bool.self) else { throw AstraError("benchmark.warmup", "Warmup was not isolated") }
                } else {
                    let record: JSONValue = try response.payload.get("collectionRecord")
                    let sampler: JSONValue = try record.get("sampler")
                    let before: [UInt32] = try sampler.get("stateBefore")
                    guard try sampler.get("rngStreamID", UUID.self) == rngID,
                          try sampler.get("drawIndex", UInt64.self) == sequence,
                          try record.get("episodeStep", UInt64.self) == sequence,
                          try record.get("cutoffNanos", UInt64.self) == cutoff,
                          try record.get("frameIDs", [UUID].self) == [id],
                          expectedKey == nil || expectedKey == before else { throw AstraError("benchmark.counter", "Packet/RNG/observation continuity changed") }
                    expectedKey = try sampler.get("stateAfter")
                    stateID = try response.payload.get("stateID"); sequence += 1; lastCutoff = cutoff
                }
                return ["sequence": .unsigned(packet.sequence), "cutoffNanos": .unsigned(cutoff), "receivedNanos": .unsigned(received),
                    "latencyMS": .number(Double(received - started) / 1e6), "ringPublicationMS": .number(Double(cutoff - started) / 1e6),
                    "roundTripMS": .number(Double(received - cutoff) / 1e6), "missedLead": .bool(received >= packet.executeAtNanos),
                    "packet": try .encode(packet), "logProbability": try response.payload.get("logProbability", JSONValue.self)]
            }
            var warmups: [JSONValue] = []
            for index in 0..<3 { warmups.append(.object(try await predict(index: index, warmup: true))) }
            let collectionWarmup = try await predict(index: 0, warmup: false)
            try emit(["kind": .string("ready"), "mode": .string("actor"), "collectionWarmup": .object(collectionWarmup), "setupAndWarmupSeconds": .number(Double(MonotonicClock.now - began) / 1e9),
                      "warmups": .array(warmups), "policyID": .string(policyID.uuidString.lowercased()), "memory": memory()])
            let window = try readWindow(), end = window.startAtNanos + window.durationNanos
            try await wait(until: window.startAtNanos)
            var slot: UInt64 = 0, missedSlots: UInt64 = 0
            while MonotonicClock.now < end {
                let intended = config.relativePacing == true
                    ? max(window.startAtNanos, lastCutoff.map { $0 + 100_000_000 } ?? window.startAtNanos)
                    : window.startAtNanos + slot * 100_000_000
                try await wait(until: max(intended, lastCutoff.map { $0 + 100_000_000 } ?? intended))
                if MonotonicClock.now >= end { break }
                var row = try await predict(index: Int(sequence), warmup: false)
                row["kind"] = .string("sample"); row["intendedNanos"] = .unsigned(intended)
                try emit(row)
                let following = max(slot + 1, (MonotonicClock.now - window.startAtNanos) / 100_000_000 + 1)
                if config.relativePacing != true { missedSlots += following - slot - 1 }
                slot = following
            }
            let measuredEnd = MonotonicClock.now
            guard ring.pendingLeaseCount == 0 else { throw AstraError("benchmark.pendingFrames", "Actor left image leases outstanding") }
            await process.shutdown(); ring.closeAfterConsumerExit()
            try emit(["kind": .string("complete"), "mode": .string("actor"), "decisions": .unsigned(sequence),
                "windowEndNanos": .unsigned(end), "measuredEndNanos": .unsigned(measuredEnd), "missedCadenceSlots": .unsigned(missedSlots),
                "allFramesReleased": .bool(true), "packetCounterMatchesSampler": .bool(true), "memory": memory(),
                "pacing": .string(config.relativePacing == true ? "lastActualCutoff+period" : "nominalGridWithSkippedSlots"),
                "teardownSeconds": .number(Double(MonotonicClock.now - measuredEnd) / 1e9)])
        } catch {
            await process.shutdown(); ring.closeAfterConsumerExit(); throw error
        }
    }

    private static func ocr(_ config: Configuration, frames: [Data]) async throws {
        let began = MonotonicClock.now, episode = UUID()
        let surface = SurfaceDescriptor(id: "benchmark", globalBounds: .init(x: 0, y: 0, width: 1280, height: 720), pixelWidth: 1280, pixelHeight: 720)
        let signal = RewardSignal(name: "Benchmark OCR", kind: .ocrText, surfaceID: "benchmark", region: .init(x: 0, y: 0, width: 1, height: 1))
        func recognize(_ index: Int) throws -> String {
            let now = MonotonicClock.now
            let image = RewardImageFrame(metadata: .init(eventNanos: now, observedNanos: now, surface: surface, byteCount: frames[index % 3].count, codec: "raw"), pixels: frames[index % 3])
            let reading = try VisualRewardDetector.read(signals: [signal], frames: [image], episodeID: episode, templates: [:])
            guard reading.count == 1, case .text(let text) = reading[0].value, text.contains("12345") else {
                throw AstraError("benchmark.ocr", "Real OCR did not read the generated score")
            }
            return text
        }
        let text = try recognize(0)
        try emit(["kind": .string("ready"), "mode": .string("ocr"), "setupAndWarmupSeconds": .number(Double(MonotonicClock.now - began) / 1e9),
                  "recognizedText": .string(text), "routing": .string("Production Vision default; no CPU-only restriction"), "memory": memory()])
        let window = try readWindow(), end = window.startAtNanos + window.durationNanos
        try await wait(until: window.startAtNanos)
        var count = 0
        while MonotonicClock.now < end {
            let start = MonotonicClock.now
            _ = try recognize(count); count += 1
            try emit(["kind": .string("ocr"), "latencyMS": .number(Double(MonotonicClock.now - start) / 1e6), "sourceNanos": .unsigned(start)])
            try await wait(until: start + 100_000_000)
        }
        try emit(["kind": .string("complete"), "mode": .string("ocr"), "observations": .integer(Int64(count)),
                  "measuredEndNanos": .unsigned(MonotonicClock.now), "memory": memory()])
    }
}
