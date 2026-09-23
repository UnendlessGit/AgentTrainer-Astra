import Foundation
import AstraCore
import AstraPlatform

/// Shared wire/causality checks for native actor owners. These checks do not
/// retime, resample, repair or otherwise rewrite execution evidence.
enum PolicyActorValidation {
    static func hello(_ reply: WireMessage) throws {
        guard reply.kind == "hello", reply.version == AstraVersion.protocolVersion,
              reply.payload.fields?["role"] == .string("actor"),
              reply.payload.fields?["protocolVersion"] == .integer(Int64(AstraVersion.protocolVersion)) else {
            throw AstraError("inference.runtime", "The local runtime did not identify the supported actor protocol.")
        }
    }

    static func success(_ reply: WireMessage, runID: UUID) throws {
        guard reply.runID == runID else { throw AstraError("inference.run", "The actor replied for another run.") }
        guard reply.kind == "ack" else {
            throw AstraError(reply.payload.fields?["code"]?.text ?? "inference.actor",
                             reply.payload.fields?["message"]?.text ?? "The actor rejected its request.")
        }
    }

    static func counters(_ value: JSONValue) throws -> (next: UInt64, generation: UInt64) {
        guard let sequence = value.fields?["nextPacketSequence"]?.uint64, sequence < UInt64.max,
              value.fields?["nextDrawIndex"]?.uint64 == sequence,
              let generation = value.fields?["actorResetGeneration"]?.uint64 else {
            throw AstraError("inference.counters", "The actor's persistent packet and random-draw counters disagree.")
        }
        return (sequence, generation)
    }

    static func progress(_ value: JSONValue) throws {
        guard value.fields?.count == 6, value.fields?["schemaVersion"] == .integer(1),
              value.fields?["runID"]?.uuid != nil, value.fields?["rngStreamID"]?.uuid != nil,
              value.fields?["drawIndex"]?.uint64 != nil, value.fields?["actorResetGeneration"]?.uint64 != nil,
              try value.required("rngState").decode([UInt32].self).count == 2 else {
            throw AstraError("inference.progress", "Actor progress requires a complete real-result stream watermark.")
        }
    }

    static func snapshot(_ snapshot: PolicyActorSnapshot, now: UInt64, previousCutoff: UInt64?, periodMS: Int,
                         episodeStep: UInt64, previousEvent: UInt64?, collecting: Bool) throws -> [SurfaceDescriptor] {
        let input = snapshot.controls, cutoff = input.cutoffNanos
        guard (1...16).contains(snapshot.frames.count),
              Set(snapshot.frames.map { $0.metadata.surface.id }).count == snapshot.frames.count,
              Set(snapshot.frames.map { $0.metadata.id }).count == snapshot.frames.count,
              snapshot.frames.allSatisfy({ (4...FrameArchive.maximumFrameBytes).contains($0.metadata.byteCount) }),
              snapshot.frames.reduce(0, { $0 + $1.metadata.byteCount }) <= 256 * 1024 * 1024, input.intervalCovered, input.controlState.valid,
              input.controlState.pointer.isFinite, input.controlState.observedNanos <= cutoff, cutoff <= now,
              input.executedEvents.count <= 2048 else {
            throw AstraError("inference.inputCoverage", "A complete causal source and settled input observation are required.")
        }
        try input.validateControlCoverage()
        if let previousCutoff {
            let bound = previousCutoff.addingReportingOverflow(episodeStep == 0 ? 1 : UInt64(periodMS) * 1_000_000)
            guard !bound.overflow, cutoff >= bound.partialValue else {
                throw AstraError("inference.causality", "Actor observations cannot replay or overlap the immutable policy cadence.")
            }
        }
        if collecting {
            guard episodeStep != 0 || (input.executedEvents.isEmpty && input.lastSequence == nil),
                  input.executedEvents.allSatisfy({ [.agent, .reconciliation].contains($0.origin) && $0.kind != .gap }) else {
                throw AstraError("inference.collectionHistory", "Collecting observations cannot contain pre-control events, interventions or input gaps.")
            }
        }
        var last = previousEvent
        for event in input.executedEvents {
            guard last.map({ event.sequence > $0 }) ?? true, event.eventNanos <= event.observedNanos,
                  event.observedNanos <= cutoff, previousCutoff.map({ event.observedNanos > $0 }) ?? true else {
                throw AstraError("inference.inputCausality", "Executed input history is repeated, unavailable or outside the causal interval.")
            }
            last = event.sequence
        }
        guard last == input.lastSequence else { throw AstraError("inference.inputCursor", "Input history does not match its acknowledged cursor.") }
        return try snapshot.frames.map { frame in
            _ = try frame.metadata.validated()
            if let coverage = frame.coverage { try coverage.validated(frame: frame.metadata, cutoffNanos: cutoff, maximumAgeMS: 250) }
            else {
                guard frame.metadata.eventNanos <= frame.metadata.observedNanos, frame.metadata.observedNanos <= cutoff,
                      cutoff - frame.metadata.eventNanos <= 250_000_000 else {
                    throw AstraError("capture.stale", "The actor observation has no recent source coverage.")
                }
            }
            return frame.metadata.surface
        }
    }

    static func result(_ value: JSONValue, checkpoint: CheckpointDocument, runID: UUID, episodeID: UUID,
                       previousState: UUID, observationID: UUID, cutoff: UInt64, sequence: UInt64,
                       surfaces: [SurfaceDescriptor], geometryRevision: UInt64? = nil, policy: InferencePolicyDetails) throws -> ActionPacket {
        let packet = try value.required("packet").decode(ActionPacket.self)
        let deadline = cutoff.addingReportingOverflow(UInt64(policy.leadMS) * 1_000_000)
        let end = deadline.partialValue.addingReportingOverflow(UInt64(policy.periodMS) * 1_000_000)
        guard value.fields?["runID"]?.uuid == runID, value.fields?["checkpointID"]?.uuid == checkpoint.id,
              value.fields?["policySignature"]?.text == checkpoint.policySignature,
              value.fields?["episodeID"]?.uuid == episodeID, let state = value.fields?["stateID"]?.uuid,
              state != previousState, value.fields?["needsReset"] == .bool(false),
              ["logProbability", "conditionalEntropy", "value"].allSatisfy({ value.fields?[$0]?.double?.isFinite == true }),
              try value.required("surfaces").decode([SurfaceDescriptor].self) == surfaces,
              packet.runID == runID, packet.sequence == sequence, packet.observationID == observationID,
              packet.geometryRevision == (geometryRevision ?? surfaces.first?.geometryRevision),
              !deadline.overflow, !end.overflow, packet.executeAtNanos == deadline.partialValue,
              packet.durationMs == policy.periodMS else {
            throw AstraError("inference.resultIdentity", "The actor returned inconsistent policy, state, observation or packet evidence.")
        }
        return try packet.validated(capabilities: policy.capabilities, surfaces: surfaces, capacity: policy.capacity, expectedGeometryRevision: geometryRevision ?? surfaces.first?.geometryRevision)
    }

    static func collection(_ value: JSONValue, packet: ActionPacket, checkpoint: CheckpointDocument,
                           observation: PolicyActorOwnedObservation, episodeStep: UInt64, resetGeneration: UInt64,
                           rngStreamID: UUID?, expectedRNG: [UInt32]?) throws -> JSONValue {
        let record = try value.required("collectionRecord"), input = observation.actorInput
        let sampler = try record.required("sampler")
        let suppliedCoverage = try record.fields?["controlCoverageNanos"].flatMap { value in
            value == .null ? nil : try value.decode(UInt64.self)
        }
        let observedCoverage = try input.fields?["controlCoverageNanos"].flatMap { value in
            value == .null ? nil : try value.decode(UInt64.self)
        }
        let before = try sampler.required("stateBefore").decode([UInt32].self)
        let after = try sampler.required("stateAfter").decode([UInt32].self)
        let sampleKey = try sampler.required("sampleKey").decode([UInt32].self)
        guard record.fields?["schemaVersion"] == .integer(1), record.fields?["checkpointID"]?.uuid == checkpoint.id,
              record.fields?["policySignature"]?.text == checkpoint.policySignature,
              record.fields?["episodeID"]?.uuid == input.fields?["episodeID"]?.uuid,
              record.fields?["observationID"]?.uuid == packet.observationID,
              record.fields?["previousStateID"]?.uuid == input.fields?["previousStateID"]?.uuid,
              record.fields?["nextStateID"]?.uuid == value.fields?["stateID"]?.uuid,
              record.fields?["cutoffNanos"]?.uint64 == input.fields?["cutoffNanos"]?.uint64,
              suppliedCoverage == observedCoverage,
              record.fields?["geometryRevision"]?.uint64 == packet.geometryRevision,
              try record.required("frameIDs").decode([UUID].self) == observation.frames.map(\.metadata.id),
              record.fields?["contextIDs"] == input.fields?["contextIDs"],
              record.fields?["episodeStep"]?.uint64 == episodeStep,
              record.fields?["recurrentReset"] == .bool(episodeStep == 0),
              record.fields?["logProbability"]?.double == value.fields?["logProbability"]?.double,
              record.fields?["value"]?.double == value.fields?["value"]?.double,
              record.fields?["environmentResets"]?.uint64 == resetGeneration,
              let stream = sampler.fields?["rngStreamID"]?.uuid, stream == rngStreamID,
              sampler.fields?["drawIndex"]?.uint64 == packet.sequence,
              sampler.fields?["kind"] == .string("categorical"), sampler.fields?["version"] == .integer(1),
              sampler.fields?["temperature"]?.double == 1, sampler.fields?["mixture"] == .string("none"),
              before.count == 2, after.count == 2, sampleKey.count == 2, expectedRNG == nil || expectedRNG == before else {
            throw AstraError("inference.collectionIdentity", "The actor's original packet and collection cursor do not describe the same real draw.")
        }
        let progress: JSONValue = .object(["schemaVersion": .integer(1), "runID": .string(observation.runID.uuidString.lowercased()),
            "rngStreamID": .string(stream.uuidString.lowercased()), "drawIndex": .unsigned(packet.sequence),
            "rngState": try .encode(after), "actorResetGeneration": .unsigned(resetGeneration)])
        try Self.progress(progress)
        return progress
    }
}
