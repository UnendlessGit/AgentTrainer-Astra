"""Bounded actor-record assembly; observations never wait for reward labels.

This worker does not sample policies, emit actions, detect rewards or reset a
world. Callers bind trusted actor/environment producers and join control stop
before closing an episode. Delayed labels are attached to original decisions.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field, replace
import copy
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

from astra.environments.interface import (EnvironmentObservation, EnvironmentSpec, EnvironmentError,
    DecisionContext, SurfaceObservation, fields, integer, same_id, uuid_key, owned_bgra, control_observation)
from astra.environments.evidence import DecisionEvidence, OUTCOMES
from astra.environments.external import validate_commands, json_geometry
from astra.environments.observation_transport import SnapshotDecoder
from astra.recordings import validate_event
from astra.model.actions import PacketBatch
from astra.model.config import ModelConfig
from astra.protocol import Message
from .reinforcement import (ObservationRecord, CollectedDecision, CollectedRollout, ReinforcementConfig,
                            _frozen_array)
from .rollout_store import FrameSpool
from .rl import Transition, RewardWindow, BootstrapObservation, Outcome, Rollout


def _bounded_size(value, limit):
    """Conservative JSON-size admission without constructing a giant string."""
    pending=[iter((value,))]; count=0
    while pending:
        try: item=next(pending[-1])
        except StopIteration:
            pending.pop(); continue
        count+=32
        if type(item) is str: count+=len(item)*6
        elif type(item) in (int,float):
            if type(item) is float and not math.isfinite(item):
                raise EnvironmentError('Actor metadata must contain finite JSON values')
        elif item is None or type(item) is bool: pass
        elif type(item) in (list,tuple,dict):
            count+=len(item)*16
            if count>limit or len(pending)>=128:
                raise EnvironmentError('Actor metadata exceeds ingress capacity')
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise EnvironmentError('Actor metadata requires JSON object keys')
                pending.append(iter(item.keys())); pending.append(iter(item.values()))
            else: pending.append(iter(item))
        else: raise EnvironmentError('Actor metadata must contain JSON values')
        if count>limit: raise EnvironmentError('Actor metadata exceeds ingress capacity')
    return count


@dataclass(frozen=True)
class AssemblyLimits:
    maximum_queue_bytes: int = 256 * 1024**2
    maximum_queue_items: int = 128
    maximum_unsealed_decisions: int = 256
    maximum_label_lag_seconds: float = 30
    maximum_cadence_delay_nanos: int = 500_000_000
    maximum_audit_bytes: int = 32 * 1024**2
    maximum_journal_bytes: int = 256 * 1024**2

    def validate(self):
        for name, low, high in (('maximum_queue_bytes', 1024, 1024**3), ('maximum_queue_items', 1, 4096),
            ('maximum_unsealed_decisions', 1, 65536), ('maximum_cadence_delay_nanos', 0, 60_000_000_000),
            ('maximum_audit_bytes', 1024, 1024**3), ('maximum_journal_bytes', 1024, 1024**3)):
            integer(getattr(self, name), high, low)
        if type(self.maximum_label_lag_seconds) not in (int, float) or not math.isfinite(self.maximum_label_lag_seconds) or not 0 < self.maximum_label_lag_seconds <= 3600:
            raise EnvironmentError('Label lag must be a bounded positive duration')
        return self


@dataclass(frozen=True)
class ActorRecord:
    packet: dict
    collection: dict
    observation: EnvironmentObservation
    events: tuple[dict, ...]

    @classmethod
    def from_wire(cls, response, snapshot, *, spec, resolve_frame, on_consumed):
        """Bind an opt-in actor response to a separately retained raw snapshot.

        Frame references must belong to the collector's ownership path; this
        cannot reuse a single-consumer actor lease after its acknowledgement.
        The actor callback never needs to wait for reward or outcome labels.
        """
        if type(response) is not dict or response.get('warmup') is True or 'collectionRecord' not in response or 'packet' not in response:
            raise EnvironmentError('Only real opt-in categorical actor responses may enter collection')
        collection=response['collectionRecord']
        if type(collection) is not dict or 'episodeID' not in collection:
            raise EnvironmentError('Actor collection record is missing its episode binding')
        coverage=collection.get('controlCoverageNanos')
        if coverage is not None:integer(coverage)
        if type(snapshot) is not dict or coverage!=snapshot.get('controlCoverageNanos'):
            raise EnvironmentError('Actor and snapshot control coverage disagree before acquisition')
        decoded,events=SnapshotDecoder(spec,resolve_frame,on_consumed).decode(snapshot,episode=collection['episodeID'],
            expected_cutoff=collection.get('cutoffNanos'))
        return cls(copy.deepcopy(response['packet']),copy.deepcopy(collection),decoded,events)

    def owned(self):
        observation = self.observation
        images = tuple(SurfaceObservation(owned_bgra(frame.pixels), copy.deepcopy(frame.metadata),
                                          frame.coverage_nanos, frame.coverage_kind) for frame in observation.frames)
        return ActorRecord(copy.deepcopy(self.packet), copy.deepcopy(self.collection),
            replace(observation, frames=images, control_state=copy.deepcopy(observation.control_state)),
            tuple(copy.deepcopy(self.events)))

    def byte_count(self, limit):
        pixels=sum(frame.pixels.nbytes for frame in self.observation.frames)
        if pixels>=limit: raise EnvironmentError('One actor record exceeds ingress capacity')
        return pixels+_bounded_size([self.packet,self.collection,self.events,
            self.observation.control_state,[frame.metadata for frame in self.observation.frames]],limit-pixels)


@dataclass
class _ActorRow:
    record: ObservationRecord
    collection: dict
    evidence: DecisionEvidence
    arrived: float
    retrospective: dict | None = None


@dataclass
class _Episode:
    id: str
    ready_nanos: int
    reset_id: str
    rows: list[_ActorRow] = field(default_factory=list)
    watermark: int = 0
    boundary: int | None = None
    closed_nanos: int | None = None
    last_actor_sequence: int | None = None
    geometry: tuple | None = None
    last_event_sequence: int | None = None
    last_input_nanos: int | None = None
    actor_generation: int | None = None
    bootstraps: dict = field(default_factory=dict)


class AsyncRolloutAssembler:
    """One immutable-policy collection, with a nonblocking bounded ingress.

    Actor records may continue while label messages are absent. Once the
    caller closes all physical episodes and finishes collection, only delayed
    evidence for those records remains admissible. A subsequent actor episode
    uses another collection or an explicit continuity/audit sink.
    """
    def __init__(self, *, spec: EnvironmentSpec, model: ModelConfig, training: ReinforcementConfig,
                 run_id: str, clock_id: str, policy_id: str, policy_signature: str,
                 actor_source_id: str, environment_source_id: str, audit_path: Path,
                 context_ids=(), limits: AssemblyLimits = AssemblyLimits(), scratch_directory: Path | None = None,
                 on_fault=lambda _: None, audit_only=False, previous_actor_progress=None, retrospective_program=None, behavior_batch_id=None):
        self.spec, self.model, self.training, self.limits = spec.validate(), model.validate(), training.validate(), limits.validate()
        self.run_id, self.clock_id = uuid_key(run_id), uuid_key(clock_id)
        self.actor_source_id, self.environment_source_id = uuid_key(actor_source_id), uuid_key(environment_source_id)
        if self.actor_source_id == self.environment_source_id:
            raise EnvironmentError('Actor and label producers require distinct explicit identities')
        if type(policy_id) is not str or not 1 <= len(policy_id) <= 256 or type(policy_signature) is not str or len(policy_signature) != 64:
            raise EnvironmentError('Actor policy identities must be pinned before collection')
        if (model.period_ms, model.lead_ms) != (spec.period_ms, spec.lead_ms) or spec.maximum_surfaces > model.maximum_surfaces:
            raise EnvironmentError('Environment and policy timing/surface contracts disagree')
        if type(context_ids) is not tuple or len(context_ids) != len(model.context_sizes) or any(type(v) is not int or not 0 <= v < size for v, size in zip(context_ids, model.context_sizes)):
            raise EnvironmentError('Actor contexts do not match the immutable vocabulary')
        self.policy_id, self.policy_signature, self.context_ids = uuid_key(policy_id), policy_signature, context_ids
        self.audit_path = Path(audit_path)
        if not self.audit_path.parent.is_dir() or self.audit_path.exists() or self.audit_path.is_symlink():
            raise EnvironmentError('Allocate a new audit file in an existing run directory')
        self._spool = FrameSpool(memory_bytes=training.maximum_rollout_bytes,
                                disk_bytes=training.maximum_rollout_disk_bytes, directory=scratch_directory)
        if type(audit_only) is not bool: raise EnvironmentError('Audit-only collection must be explicit')
        self.audit_only = audit_only
        self.retrospective_program = retrospective_program
        self.behavior_batch_id = None if behavior_batch_id is None else uuid_key(behavior_batch_id)
        if previous_actor_progress is not None:
            from .rollout_artifacts import progress
            progress(previous_actor_progress,self.run_id if retrospective_program is None else None)
        self._retrospective_excluded=set()
        self._previous_actor_progress=previous_actor_progress
        self._on_fault = on_fault
        self._condition = threading.Condition()
        self._queue = queue.Queue(maxsize=limits.maximum_queue_items)
        self._queue_bytes = 0
        self._ingress_items = 0
        self._processing = False
        self._failure = None
        self._closed = False
        self._finished_collection = False
        self._episodes: dict[str, _Episode] = {}
        self._active_episode = None
        self._rows: dict[str, _ActorRow] = {}
        self._last_actor_sequence = self._last_label_sequence = None
        self._last_received_label_sequence = None
        self._pending_evidence = []
        self._pending_evidence_bytes = 0
        self._sampler = None
        self._sample_keys = set()
        self._snapshots = set()
        self._started = time.monotonic()
        self._sealed = None
        self._sealing = False
        self._transferred = False
        self._thread = threading.Thread(target=self._work, name='Astra rollout evidence owner', daemon=True)
        self._thread.start()

    def _submit(self, kind, value, *, size=0, wait=False, prepare=None):
        future = Future() if wait else None
        with self._condition:
            if self._failure is not None: raise self._failure
            if self._closed: raise EnvironmentError('Rollout assembler is closed')
            if self._sealing or self._sealed is not None: raise EnvironmentError('Rollout evidence is already sealing or sealed')
            if self._queue_bytes + size > self.limits.maximum_queue_bytes or self._ingress_items>=self.limits.maximum_queue_items:
                raise EnvironmentError('Rollout ingress byte budget exceeded')
            self._queue_bytes += size; self._ingress_items+=1
        try:
            # Reserve before a generic mutable pixel source is detached. A
            # second producer cannot allocate another full image past budget.
            if prepare is not None: value=prepare()
            with self._condition:
                if self._failure is not None: raise self._failure
                if self._closed or self._sealing: raise EnvironmentError('Rollout assembler stopped during ingress transfer')
                try:
                    self._queue.put_nowait((kind, value, size, future, time.monotonic()))
                except queue.Full as error:
                    raise EnvironmentError('Rollout ingress item budget exceeded') from error
        except BaseException:
            with self._condition:
                self._queue_bytes-=size; self._ingress_items-=1
            raise
        return future.result(timeout=10) if wait else None

    def begin_episode(self, *, episode_id, ready_nanos, reset_id, controls_released, pending_packets, source_id):
        if not same_id(source_id, self.environment_source_id): raise EnvironmentError('Unbound readiness producer')
        if controls_released is not True or type(pending_packets) is not int or pending_packets != 0:
            raise EnvironmentError('Readiness must prove a cleared control ledger')
        return self._submit('begin', (uuid_key(episode_id), integer(ready_nanos), uuid_key(reset_id)), wait=True)

    def submit_actor(self, record: ActorRecord, *, source_id):
        if not same_id(source_id, self.actor_source_id) or not isinstance(record, ActorRecord):
            raise EnvironmentError('Unbound actor record producer')
        record.observation.validate(self.spec)
        size = record.byte_count(self.limits.maximum_queue_bytes)
        self._submit('actor', None, size=size, prepare=record.owned)

    def wire_byte_count(self,response,snapshot):
        frames=snapshot.get('frames') if type(snapshot) is dict else None
        if type(frames) is not list or not 1<=len(frames)<=self.spec.maximum_surfaces:
            raise EnvironmentError('Invalid wire observation surface count')
        pixels=sum(integer(frame['metadata']['byteCount'],self.spec.maximum_observation_bytes,1) for frame in frames)
        if pixels>self.spec.maximum_observation_bytes or pixels>=self.limits.maximum_queue_bytes:
            raise EnvironmentError('Wire frame copies exceed collector ingress capacity')
        return pixels+_bounded_size([response,snapshot],self.limits.maximum_queue_bytes-pixels)

    def submit_actor_wire(self,response,snapshot,*,source_id,resolve_frame,on_consumed):
        if not same_id(source_id,self.actor_source_id):raise EnvironmentError('Unbound actor record producer')
        self._submit('actor',None,size=self.wire_byte_count(response,snapshot),
            prepare=lambda:ActorRecord.from_wire(response,snapshot,spec=self.spec,
                resolve_frame=resolve_frame,on_consumed=on_consumed))

    def submit_bootstrap_wire(self,*,episode_id,snapshot,value,policy_id,preceding_packet_id,source_id,resolve_frame,on_consumed,endpoint=False):
        if not same_id(source_id,self.actor_source_id) or not same_id(policy_id,self.policy_id):
            raise EnvironmentError('Bootstrap did not come from the bound behavior actor')
        if not (endpoint and self.retrospective_program is not None and value is None) and (type(value) not in (int,float) or not math.isfinite(value)):
            raise EnvironmentError('Bootstrap must have a finite value')
        def prepare():
            observed,events=SnapshotDecoder(self.spec,resolve_frame,on_consumed).decode(snapshot,episode=episode_id)
            return uuid_key(episode_id),observed,None if value is None else float(value),uuid_key(preceding_packet_id),events
        self._submit('bootstrap',None,size=self.wire_byte_count({},snapshot),prepare=prepare)

    def submit_evidence(self, message: Message, *, source_id):
        if not same_id(source_id, self.environment_source_id): raise EnvironmentError('Unbound label/receipt producer')
        message.validate()
        size = _bounded_size([message.kind,message.sequence,message.payload,message.request_id,message.run_id],
                             self.limits.maximum_queue_bytes)
        if not same_id(message.run_id, self.run_id) or not same_id(message.payload.get('clockID'), self.clock_id):
            raise EnvironmentError('Evidence run/clock identity is incompatible')
        self._submit('evidence', None, size=size, prepare=lambda:copy.deepcopy(message))

    def close_episode(self, *, episode_id, stopped_nanos, last_actor_sequence, controls_released, pending_packets, source_id):
        if not same_id(source_id, self.environment_source_id) or controls_released is not True or type(pending_packets) is not int or pending_packets != 0:
            raise EnvironmentError('Episode closure needs explicit joined control cleanup')
        self._submit('end', (uuid_key(episode_id), integer(stopped_nanos), integer(last_actor_sequence)), wait=True)

    def submit_bootstrap(self, *, episode_id, observation, value, policy_id, preceding_packet_id, source_id, executed_events=()):
        if not same_id(source_id, self.actor_source_id) or not same_id(policy_id,self.policy_id):
            raise EnvironmentError('Bootstrap did not come from the bound behavior actor')
        observation.validate(self.spec)
        if type(value) not in (int, float) or not math.isfinite(value): raise EnvironmentError('Bootstrap value must be finite')
        item=ActorRecord({}, {}, observation, tuple(executed_events))
        self._submit('bootstrap', None, size=item.byte_count(self.limits.maximum_queue_bytes),
            prepare=lambda:(uuid_key(episode_id),item.owned().observation,float(value),uuid_key(preceding_packet_id),tuple(copy.deepcopy(executed_events))))

    def abort_learning(self):
        self._submit('abort_learning',None,wait=True)

    def finish_collection(self):
        self._submit('finish', None, wait=True)

    def take_ready(self):
        """Nonblocking one-time transfer for a collector process owner loop."""
        with self._condition:
            if self._failure is not None: raise self._failure
            if self._sealed is None:return None
            if self._transferred:raise EnvironmentError('A collection can be transferred only once')
            self._transferred=True
            return self._sealed

    def seal(self, *, cancelled=lambda: False, timeout_seconds=30) -> CollectedRollout:
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600:
            raise EnvironmentError('Sealing needs a finite bounded timeout')
        deadline = time.monotonic() + timeout_seconds
        while True:
            if cancelled():
                self.close()
                raise InterruptedError('Rollout assembly cancelled before learner admission')
            with self._condition:
                if self._failure is not None: raise self._failure
                if self._sealed is not None:
                    if self._transferred: raise EnvironmentError('A rollout may be transferred to only one learner')
                    self._transferred = True
                    return self._sealed
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise EnvironmentError('Rollout evidence did not seal before the requested timeout')
                self._condition.wait(min(.05, remaining))

    def _work(self):
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                try: item = self._queue.get(timeout=.05)
                except queue.Empty:
                    with self._condition:
                        if self._closed: return
                    self._check_lag(); self._try_seal()
                    continue
                kind, value, size, future, arrived = item
                self._processing = True
                try:
                    if kind == 'begin': self._begin(*value)
                    elif kind == 'actor': self._actor(value, arrived)
                    elif kind == 'evidence': self._evidence(value)
                    elif kind == 'end': self._end(*value)
                    elif kind == 'bootstrap': self._bootstrap(*value)
                    elif kind == 'abort_learning':
                        if self.retrospective_program is None:self.audit_only=True
                        else:self._retrospective_excluded.update(identifier for identifier,episode in self._episodes.items()
                            if episode.closed_nanos is None or episode.boundary is None or any(row.evidence.outcome=='aborted' for row in episode.rows))
                    elif kind == 'finish':
                        if self._active_episode is not None: raise EnvironmentError('Join episode stop before finishing actor collection')
                        self._finished_collection = True
                    self._check_lag(); self._try_seal()
                    if future is not None: future.set_result(None)
                except BaseException as error:
                    if future is not None: future.set_exception(error)
                    raise
                finally:
                    with self._condition:
                        self._queue_bytes-=size; self._ingress_items-=1
                        self._processing = False
                    value=None
                    self._queue.task_done()
                if self._sealed is not None:
                    self._episodes.clear(); self._rows.clear(); self._pending_evidence.clear()
                    self._sample_keys.clear(); self._snapshots.clear()
                    return
        except BaseException as error:
            with self._condition:
                self._failure = error
                self._condition.notify_all()
            try: self._on_fault(error)
            except Exception: pass
            if not self._transferred: self._spool.close()
            while True:
                try: _, _, size, future, _ = self._queue.get_nowait()
                except queue.Empty: break
                if future is not None: future.set_exception(error)
                with self._condition: self._queue_bytes-=size; self._ingress_items-=1
                self._queue.task_done()

    def _begin(self, episode_id, ready_nanos, reset_id):
        if self._finished_collection or self._active_episode is not None or episode_id in self._episodes:
            raise EnvironmentError('Episode readiness is duplicated or crosses an active episode')
        if self._episodes and ready_nanos < next(reversed(self._episodes.values())).closed_nanos:
            raise EnvironmentError('Readiness moved the host clock backwards')
        self._episodes[episode_id] = _Episode(episode_id, ready_nanos, reset_id)
        self._active_episode = episode_id

    def _actor(self, actor, arrived):
        packet, value, observed = actor.packet, actor.collection, actor.observation
        required = ('schemaVersion','checkpointID','policySignature','modelSignature','episodeID','episodeStep',
            'observationID','cutoffNanos','geometryRevision','frameIDs','contextIDs','previousStateID','nextStateID',
            'recurrentReset','elapsedSeconds','stateBefore','packetFields','logProbability','value','sampler','environmentResets')
        fields(value, required, ('controlCoverageNanos',))
        fields(packet, ('id','runID','sequence','observationID','geometryRevision','executeAtNanos','durationMs','commands'))
        if integer(value['schemaVersion'],1,1) != 1 or not same_id(value['checkpointID'],self.policy_id) or value['policySignature'] != self.policy_signature or value['modelSignature'] != self.model.signature:
            raise EnvironmentError('Actor used a different policy or model configuration')
        episode_id = uuid_key(value['episodeID']); episode = self._episodes.get(episode_id)
        if episode is None: raise EnvironmentError('Actor decision has no confirmed episode readiness')
        sequence, step = integer(packet['sequence']), integer(value['episodeStep'])
        if self._last_actor_sequence is not None and sequence != self._last_actor_sequence + 1:
            raise EnvironmentError('Actor packet sequence skipped or repeated a decision')
        if episode.closed_nanos is not None and sequence > episode.last_actor_sequence:
            raise EnvironmentError('Actor decision exceeds the joined stop watermark')
        if step != len(episode.rows) or type(value['recurrentReset']) is not bool or value['recurrentReset'] != (step == 0):
            raise EnvironmentError('Actor recurrence did not follow the confirmed episode')
        cutoff = integer(value['cutoffNanos']); integer(value['geometryRevision'])
        if not same_id(observed.episode_id, episode_id) or not same_id(value['observationID'], observed.id) or not same_id(packet['observationID'], observed.id) or not same_id(packet['runID'], self.run_id):
            raise EnvironmentError('Actor snapshot and packet identities disagree')
        if value.get('controlCoverageNanos') is not None:integer(value['controlCoverageNanos'])
        if value.get('controlCoverageNanos')!=observed.control_coverage_nanos:
            raise EnvironmentError('Actor and retained observation control coverage disagree')
        if cutoff != observed.cutoff_nanos or value['geometryRevision'] != observed.geometry_revision or packet['geometryRevision'] != observed.geometry_revision:
            raise EnvironmentError('Actor snapshot clocks/geometry disagree')
        if type(value['frameIDs']) is not list or [uuid_key(item) for item in value['frameIDs']] != [uuid_key(frame.metadata['id']) for frame in observed.frames] or value['contextIDs'] != list(self.context_ids):
            raise EnvironmentError('Actor inputs differ from the retained source observation')
        maximum = self.spec.period_ms * 1000000 + self.limits.maximum_cadence_delay_nanos
        if step == 0:
            if cutoff < episode.ready_nanos or actor.events: raise EnvironmentError('Initial actor input precedes readiness or includes pre-control events')
            elapsed = self.spec.period_ms / 1000
        else:
            previous = episode.rows[-1]
            delta = cutoff - previous.record.cutoff_nanos
            if not self.spec.period_ms * 1000000 <= delta <= maximum: raise EnvironmentError('Actor cadence exceeded its explicit delay allowance')
            elapsed = delta / 1e9
            if not same_id(value['previousStateID'], previous.collection['nextStateID']): raise EnvironmentError('Actor recurrent state identity is discontinuous')
        if type(value['elapsedSeconds']) not in (int,float) or not math.isfinite(value['elapsedSeconds']) or value['elapsedSeconds'] != elapsed:
            raise EnvironmentError('Actor elapsed-time feature was rounded or changed')
        if same_id(value['previousStateID'], value['nextStateID']): raise EnvironmentError('Actor state identity did not advance')
        generation = integer(value['environmentResets'])
        if episode.actor_generation is not None and generation != episode.actor_generation: raise EnvironmentError('Actor reset generation changed inside an episode')
        if step == 0 and self._sampler is not None and generation <= self._sampler[3]: raise EnvironmentError('Actor reset generation did not advance')
        geometry = (observed.geometry_revision, tuple(json_geometry(frame.metadata['surface']) for frame in observed.frames))
        if episode.geometry is not None and geometry != episode.geometry: raise EnvironmentError('Within-episode resize requires an explicit boundary')
        if packet['executeAtNanos'] != cutoff + self.spec.lead_ms * 1000000 or packet['durationMs'] != self.spec.period_ms:
            raise EnvironmentError('Actor changed the fixed packet lead or duration')
        validate_commands(packet['commands'], self.spec, observed)
        for name in ('logProbability','value'):
            if type(value[name]) not in (int,float) or not math.isfinite(value[name]) or (name=='logProbability' and value[name]>1e-6):
                raise EnvironmentError('Actor likelihood/value must be finite behavior outputs')
        fields(value['packetFields'], PacketBatch.__dataclass_fields__)
        for values in value['packetFields'].values():
            if type(values) is not list or len(values) != self.model.packet_capacity+1 or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in values):
                raise EnvironmentError('Actor packet fields must retain original Int32 token arrays')
        states = value['stateBefore']
        if type(states) is not list or len(states) != self.model.recurrent_layers or any(type(row) is not list or len(row)!=self.model.recurrent_width or any(type(v) not in (int,float) or not math.isfinite(v) for v in row) for row in states):
            raise EnvironmentError('Actor recurrent anchors have incompatible shapes or nonfinite values')
        sampler = value['sampler']; fields(sampler, ('kind','temperature','mixture','version','rngStreamID','drawIndex','stateBefore','sampleKey','stateAfter'))
        if sampler['kind']!='categorical' or type(sampler['temperature']) not in (int,float) or sampler['temperature']!=1 or sampler['mixture']!='none' or type(sampler['version']) is not int or sampler['version']!=1:
            raise EnvironmentError('PPO requires categorical temperature-one sampling without greedy fallback or mixtures')
        keys=[]
        for name in ('stateBefore','sampleKey','stateAfter'):
            data=sampler[name]
            if type(data) is not list or len(data)!=2: raise EnvironmentError('Actor RNG keys must contain two UInt32 words')
            keys.append(tuple(integer(v,2**32-1) for v in data))
        stream, draw = uuid_key(sampler['rngStreamID']), integer(sampler['drawIndex'])
        if sequence!=draw:raise EnvironmentError('Collection schema 1 requires packet sequence to equal the actor RNG draw index')
        # Validate the declared key split on CPU. This does not sample an
        # action or consume global RNG state, and cannot block actor GPU work.
        import mlx.core as mx
        split=np.asarray(mx.random.split(mx.array(keys[0],dtype=mx.uint32),stream=mx.cpu))
        if tuple(split[0])!=keys[2] or tuple(split[1])!=keys[1]:
            raise EnvironmentError('Actor RNG evidence does not match its declared split')
        if self._sampler is None and self._previous_actor_progress is not None:
            prior=self._previous_actor_progress
            if stream!=uuid_key(prior['rngStreamID']) or draw!=prior['drawIndex']+1 or keys[0]!=tuple(prior['rngState']) or generation<=prior['actorResetGeneration']:
                raise EnvironmentError('Actor collection does not continue its bound prior progress after reset')
        if self._sampler is not None and (stream != self._sampler[0] or draw != self._sampler[1]+1 or keys[0] != self._sampler[2]):
            raise EnvironmentError('Actor RNG stream repeated, reseeded or skipped independently of learning')
        if keys[1] in self._sample_keys: raise EnvironmentError('An actor sampling key was reused')
        if len(self._rows) >= self.training.maximum_rollout_decisions: raise EnvironmentError('Actor collection exceeds its bounded decision count')
        packet_id=uuid_key(packet['id'])
        if packet_id in self._rows or uuid_key(observed.id) in self._snapshots: raise EnvironmentError('Actor packet/observation identity was duplicated')
        last_event=episode.last_event_sequence
        for raw in actor.events:
            event=validate_event(raw, maximum_timestamp=2**64-1)
            if event['origin'] not in ('agent','reconciliation') or event['kind']=='gap' or event['eventNanos']>event['observedNanos'] or event['observedNanos']>cutoff or (last_event is not None and event['sequence']<=last_event):
                raise EnvironmentError('Actor input contains feedback/intervention, a gap or noncausal history')
            if step and event['observedNanos']<=episode.rows[-1].record.cutoff_nanos: raise EnvironmentError('Actor input history was delivered outside its observation interval')
            last_event=event['sequence']
        control_observation(observed.control_state,cutoff,observed.control_coverage_nanos,
                            maximum_age_ms=self.spec.maximum_frame_age_ms)
        observed=replace(observed,id=uuid_key(observed.id),episode_id=episode_id)
        if self.audit_only or episode_id in self._retrospective_excluded:
            # Continuity is explicitly ineligible for learning. Retain timing,
            # controls and exact actor/receipt evidence, without spooling visual
            # tensors that no learner may consume. CPU lease validation still
            # completed before this branch.
            record=ObservationRecord((),observed.id,episode_id,cutoff,observed.geometry_revision,
                json.dumps(observed.control_state,separators=(',',':')).encode(),
                json.dumps(actor.events,separators=(',',':')).encode(),elapsed,step==0,episode.last_input_nanos,
                observed.control_coverage_nanos)
            self._spool.reserve_metadata(record.metadata_byte_count)
        else:
            record=ObservationRecord.capture(observed, events=actor.events, elapsed_seconds=elapsed, reset=step==0,
                                             last_input_nanos=episode.last_input_nanos, spool=self._spool)
        context=DecisionContext(self.run_id,episode_id,self.policy_id,observed.id,packet['id'],step,cutoff,observed.geometry_revision)
        row=_ActorRow(record,value,DecisionEvidence(context,sequence,tuple(packet['commands'])),arrived)
        self._spool.reserve_metadata(_bounded_size([value,packet],self.training.maximum_rollout_bytes)*3)
        episode.rows.append(row); self._rows[packet_id]=row; self._snapshots.add(uuid_key(observed.id))
        episode.geometry=geometry; episode.actor_generation=generation; episode.last_event_sequence=last_event
        times=[event['observedNanos'] for event in actor.events if event['origin']=='agent']
        if times: episode.last_input_nanos=max(episode.last_input_nanos or 0,max(times))
        self._sample_keys.add(keys[1]); self._sampler=(stream,draw,keys[2],generation)
        self._last_actor_sequence=sequence
        self._drain_evidence()

    def _evidence(self, message):
        if self._last_received_label_sequence is not None and message.sequence<=self._last_received_label_sequence:
            raise EnvironmentError('Label producer sequence repeated or moved backwards')
        size=len(message.encode())
        if len(self._pending_evidence)>=self.limits.maximum_queue_items or self._pending_evidence_bytes+size>self.limits.maximum_queue_bytes:
            raise EnvironmentError('Evidence awaiting actor identity exceeded its bounded capacity')
        self._last_received_label_sequence=message.sequence
        self._pending_evidence.append((message,size,time.monotonic()))
        self._pending_evidence_bytes+=size
        self._drain_evidence()

    def _drain_evidence(self):
        while self._pending_evidence:
            message,size,_=self._pending_evidence[0]
            if message.kind!='environment.watermark' and uuid_key(message.request_id) not in self._rows:
                break
            self._apply_evidence(message)
            self._pending_evidence.pop(0); self._pending_evidence_bytes-=size

    def _apply_evidence(self, message):
        payload=message.payload; episode=self._episodes.get(uuid_key(payload.get('episodeID')))
        if episode is None: raise EnvironmentError('Evidence belongs to an unconfirmed episode')
        if message.kind=='environment.watermark':
            fields(payload,('clockID','episodeID','throughNanos','throughSequence','complete'))
            covered=payload['throughSequence']
            if covered is not None: integer(covered)
            if payload['complete'] is not True or covered!=self._last_label_sequence:
                raise EnvironmentError('Watermark does not certify the explicit producer evidence prefix')
            through=integer(payload['throughNanos'])
            if through<episode.watermark: raise EnvironmentError('Label watermark moved backwards')
            for row in episode.rows:
                evidence=row.evidence
                if evidence.end is not None and evidence.end<=through and (evidence.reward is None or evidence.outcome in (None,'unknown')):
                    raise EnvironmentError('A sealed reward/outcome is unknown; no zero or continuation may be inferred')
            episode.watermark=through
        else:
            row=self._rows.get(uuid_key(message.request_id))
            if row is None or row.record.episode_id!=episode.id: raise EnvironmentError('Evidence has no matching actor decision')
            if message.kind in ('environment.reward','environment.outcome'):
                if self.retrospective_program is not None and message.kind=='environment.reward':
                    from .feedback_program import automatic_reward
                    row.retrospective=automatic_reward(payload,self.retrospective_program)
                    payload={key:value for key,value in payload.items() if key not in ('automaticComponents','deferredManualRuleIDs')}
                boundary=row.evidence.accept_label(message.kind,payload,period_ms=self.spec.period_ms,watermark=episode.watermark,
                    maximum_interval_nanos=self.spec.period_ms*1000000+self.limits.maximum_cadence_delay_nanos)
                if episode.rows and row.evidence.end-episode.rows[0].record.cutoff_nanos>self.spec.maximum_episode_ms*1000000+self.limits.maximum_cadence_delay_nanos:
                    raise EnvironmentError('Semantic episode exceeded its configured duration bound')
                if boundary is not None:
                    if episode.boundary is not None and episode.boundary!=boundary: raise EnvironmentError('Episode terminal cutoffs disagree')
                    episode.boundary=boundary
            elif message.kind=='environment.receipt': row.evidence.accept_receipt(payload,lead_ms=self.spec.lead_ms,retain_failure=True)
            else: raise EnvironmentError('Unsupported delayed evidence kind')
        self._last_label_sequence=message.sequence
        self._spool.reserve_metadata(len(message.encode()))

    def _end(self, episode_id, stopped, last_sequence):
        episode=self._episodes.get(episode_id)
        if episode is None or episode.closed_nanos is not None or self._active_episode!=episode_id:
            raise EnvironmentError('Episode closure is stale or duplicated')
        if stopped<episode.ready_nanos or (episode.rows and (stopped<episode.rows[-1].record.cutoff_nanos or last_sequence<episode.rows[-1].evidence.sequence)):
            raise EnvironmentError('Episode stop precedes admitted actor data')
        episode.closed_nanos=stopped; episode.last_actor_sequence=last_sequence; self._active_episode=None

    def _bootstrap(self, episode_id, observed, value, packet_id, events):
        episode=self._episodes.get(episode_id); row=self._rows.get(packet_id)
        if episode is None or row is None or row.record.episode_id!=episode_id or not same_id(observed.episode_id,episode_id):
            raise EnvironmentError('Bootstrap does not belong to its preceding actor decision')
        if observed.cutoff_nanos<=row.record.cutoff_nanos: raise EnvironmentError('Bootstrap must follow its actor decision')
        if observed.cutoff_nanos-row.record.cutoff_nanos>self.spec.period_ms*1000000+self.limits.maximum_cadence_delay_nanos:
            raise EnvironmentError('Bootstrap interval exceeds the cadence delay bound')
        control_observation(observed.control_state,observed.cutoff_nanos,observed.control_coverage_nanos,
                            maximum_age_ms=self.spec.maximum_frame_age_ms)
        if packet_id in episode.bootstraps: raise EnvironmentError('Bootstrap actor evidence was duplicated')
        if (observed.geometry_revision,tuple(json_geometry(frame.metadata['surface']) for frame in observed.frames))!=episode.geometry:
            raise EnvironmentError('Bootstrap geometry is not the pre-reset episode geometry')
        last=row.record.last_input_nanos
        past=json.loads(row.record.events_json)
        times=[event['observedNanos'] for event in past if event['origin']=='agent']
        if times:last=max(last or 0,max(times))
        sequence=None
        for event in events:
            validate_event(event,maximum_timestamp=2**64-1)
            if event['origin'] not in ('agent','reconciliation') or event['kind']=='gap' or event['eventNanos']>event['observedNanos'] or not row.record.cutoff_nanos<event['observedNanos']<=observed.cutoff_nanos or (sequence is not None and event['sequence']<=sequence):
                raise EnvironmentError('Bootstrap input history is not causal pre-reset evidence')
            sequence=event['sequence']
        observed=replace(observed,id=uuid_key(observed.id),episode_id=episode_id)
        record=ObservationRecord.capture(observed,elapsed_seconds=(observed.cutoff_nanos-row.record.cutoff_nanos)/1e9,
                                         reset=False,last_input_nanos=last,events=events,spool=self._spool)
        episode.bootstraps[packet_id]=(record,value)

    def _check_lag(self):
        if self._sealed is not None: return
        if self._pending_evidence and time.monotonic()-self._pending_evidence[0][2]>self.limits.maximum_label_lag_seconds:
            raise EnvironmentError('Evidence did not obtain its actor identity within the lag budget')
        pending=[]
        for episode in self._episodes.values():
            for index,row in enumerate(episode.rows):
                evidence=row.evidence
                suffix=self.audit_only or episode.id in self._retrospective_excluded or (episode.boundary is not None and row.record.cutoff_nanos>=episode.boundary)
                if not suffix and evidence.end is None:
                    expected_end=episode.rows[index+1].record.cutoff_nanos if index+1<len(episode.rows) else episode.boundary
                    if expected_end is not None and expected_end<=episode.watermark:
                        raise EnvironmentError('Watermark sealed a missing actor reward/outcome window')
                labels=suffix or (evidence.end is not None and evidence.end<=episode.watermark and evidence.reward is not None and evidence.outcome in OUTCOMES)
                missing_bootstrap=(not suffix and (evidence.outcome=='truncated' or self.retrospective_program is not None and evidence.outcome in OUTCOMES) and
                    not any(other.record.cutoff_nanos==evidence.end for other in episode.rows) and
                    uuid_key(evidence.context.packet_id) not in episode.bootstraps)
                if not labels or evidence.receipt is None or (not suffix and not evidence.admission_seen) or missing_bootstrap:
                    pending.append(row)
        if len(pending)>self.limits.maximum_unsealed_decisions: raise EnvironmentError('Unsealed actor decisions exceeded the configured lag capacity')
        if pending and time.monotonic()-min(row.arrived for row in pending)>self.limits.maximum_label_lag_seconds:
            raise EnvironmentError('Delayed actor evidence exceeded its explicit wall-time lag budget')

    def _try_seal(self):
        if not self._finished_collection or self._sealed is not None: return
        if not self._episodes: raise EnvironmentError('Cannot seal an empty actor collection')
        if self._pending_evidence:
            if all(episode.closed_nanos is not None and episode.rows and episode.rows[-1].evidence.sequence==episode.last_actor_sequence
                   for episode in self._episodes.values()):
                raise EnvironmentError('Closed actor stream cannot account for pending evidence identities')
            return
        if self.audit_only:
            self._try_seal_audit();return
        if self.retrospective_program is not None and self._retrospective_excluded==set(self._episodes):
            self._try_seal_audit();return
        decisions=[]; audit=[]
        for episode in self._episodes.values():
            if episode.id in self._retrospective_excluded:
                if episode.closed_nanos is None or not episode.rows or episode.rows[-1].evidence.sequence!=episode.last_actor_sequence or any(row.evidence.receipt is None for row in episode.rows):return
                audit.append({'episodeID':episode.id,'stoppedNanos':episode.closed_nanos,'excludedEpisode':True,
                    'reason':'Operator stopped before a completed joined semantic episode',
                    'decisions':[{'packetID':row.evidence.context.packet_id,'observationID':row.record.observation_id,
                        'decisionNanos':row.record.cutoff_nanos,'receipt':row.evidence.receipt,'commands':row.evidence.commands} for row in episode.rows]})
                continue
            if episode.closed_nanos is None or not episode.rows or episode.boundary is None: return
            if episode.rows[-1].evidence.sequence!=episode.last_actor_sequence: return
            if episode.boundary>episode.closed_nanos: raise EnvironmentError('Semantic outcome lies after joined episode stop')
            if any(row.evidence.receipt is None for row in episode.rows): return
            prefix=[row for row in episode.rows if row.record.cutoff_nanos<episode.boundary]
            if not prefix: raise EnvironmentError('Terminal episode has no positive-duration actor decision')
            for index,row in enumerate(prefix):
                evidence=row.evidence
                if not evidence.execution_complete(episode.boundary) or not evidence.admission_seen: return
                if evidence.end is None or evidence.end>episode.watermark or evidence.reward is None or evidence.outcome not in OUTCOMES: return
                if evidence.outcome=='aborted': raise EnvironmentError('An aborted episode cannot enter PPO')
                expected_end=prefix[index+1].record.cutoff_nanos if index+1<len(prefix) else episode.boundary
                if evidence.end!=expected_end or (index+1<len(prefix) and evidence.outcome!='continuing') or (index+1==len(prefix) and evidence.outcome=='continuing'):
                    raise EnvironmentError('Reward/outcome windows do not partition the actual actor cutoffs')
                outcome=Outcome(OUTCOMES[evidence.outcome]); bootstrap_record=None; bootstrap=None
                endpoint_record=None
                if outcome in (Outcome.CONTINUING,Outcome.TRUNCATED) or self.retrospective_program is not None:
                    next_row=next((item for item in episode.rows if item.record.cutoff_nanos==evidence.end),None)
                    supplied=episode.bootstraps.get(uuid_key(evidence.context.packet_id))
                    if next_row is not None: bootstrap_record,bootstrap_value=replace(next_row.record,reset=False),next_row.collection['value']
                    elif supplied is not None: bootstrap_record,bootstrap_value=supplied
                    else: return
                    if bootstrap_record.cutoff_nanos!=evidence.end: raise EnvironmentError('Bootstrap cutoff is not the original pre-reset outcome cutoff')
                    endpoint_record=bootstrap_record
                    if outcome in (Outcome.CONTINUING,Outcome.TRUNCATED):
                        if bootstrap_value is None:raise EnvironmentError('Truncation/continuation requires its original behavior value')
                        bootstrap=BootstrapObservation(episode.id,self.policy_id,bootstrap_record.observation_id,evidence.end,bootstrap_value)
                    else:bootstrap_record=None
                scalar=Transition(self.run_id,episode.id,self.policy_id,index,row.record.observation_id,evidence.context.packet_id,
                    row.record.cutoff_nanos,evidence.end,RewardWindow(row.record.cutoff_nanos,evidence.end,evidence.reward),
                    row.collection['logProbability'],row.collection['value'],bootstrap,outcome,index==0)
                tokens=tuple(tuple(row.collection['packetFields'][name]) for name in PacketBatch.__dataclass_fields__)
                states=tuple(_frozen_array([values],np.float32) for values in row.collection['stateBefore'])
                decisions.append(CollectedDecision(scalar,replace(row.record,episode_id=episode.id),tokens,states,bootstrap_record,
                    evidence.outcome,json.dumps(evidence.commands,separators=(',',':'),allow_nan=False).encode(),
                    endpoint_record if self.retrospective_program is not None else None,
                    None if self.retrospective_program is None else json.dumps({
                        'packetSequence':evidence.sequence,'drawIndex':row.collection['sampler']['drawIndex'],
                        'execution':'semanticBoundaryPrefix' if evidence.receipt['status']=='cancelled' else 'executed',
                        'receipt':evidence.receipt,**row.retrospective},separators=(',',':'),allow_nan=False).encode()))
            suffix=[]; post_terminal=[]
            for row in episode.rows:
                receipt=copy.deepcopy(row.evidence.receipt)
                if row.record.cutoff_nanos>=episode.boundary:
                    suffix.append({'packetID':row.evidence.context.packet_id,'observationID':row.record.observation_id,
                        'decisionNanos':row.record.cutoff_nanos,'receipt':receipt,'commands':row.evidence.commands})
                for result in receipt['commandResults']:
                    if result.get('postedNanos',-1)>=episode.boundary:
                        post_terminal.append({'packetID':row.evidence.context.packet_id,**result})
            audit.append({'episodeID':episode.id,'terminalNanos':episode.boundary,'stoppedNanos':episode.closed_nanos,
                          'excludedDecisions':suffix,'postTerminalEffects':post_terminal})
        report={'schemaVersion':1,'runID':self.run_id,'policyID':self.policy_id,'episodes':audit,
                'sampler':{'kind':'categorical','temperature':1,'mixture':'none','version':1},
                'limitation':'Actual excluded effects are preserved; they were not retroactively cancelled.'}
        progress={'schemaVersion':1,'runID':self.run_id,'rngStreamID':self._sampler[0],
                  'drawIndex':self._sampler[1],'rngState':list(self._sampler[2]),
                  'actorResetGeneration':self._sampler[3]}
        report['actorProgress']=progress
        if len(decisions)<self.training.rollout_decisions and self.retrospective_program is None:
            raise EnvironmentError('Complete learning episodes have not reached the configured rollout minimum')
        encoded=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(encoded)>self.limits.maximum_audit_bytes: raise EnvironmentError('Rollout audit exceeds its explicit byte budget')
        with self._condition:
            if self._closed or not self._queue.empty() or self._ingress_items>int(self._processing): return
            self._sealing=True
        import os, tempfile
        descriptor, temporary=tempfile.mkstemp(prefix='.rollout-audit-',dir=self.audit_path.parent)
        try:
            with os.fdopen(descriptor,'wb') as stream: stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary,self.audit_path)
        finally: Path(temporary).unlink(missing_ok=True)
        self._spool.seal()
        rollout=CollectedRollout(self.behavior_batch_id or str(uuid.uuid4()),Rollout(tuple(item.transition for item in decisions)),tuple(decisions),
            self.model.signature,self.spec.signature,self.context_ids,self._spool.disk_bytes+self._spool.metadata_bytes,
            time.monotonic()-self._started,0,self._spool,('categorical',1.0,'none',1),
            json.dumps(progress,sort_keys=True,separators=(',',':')).encode())
        with self._condition:
            self._sealed=rollout; self._condition.notify_all()

    def _try_seal_audit(self):
        for episode in self._episodes.values():
            if episode.closed_nanos is None or not episode.rows or episode.rows[-1].evidence.sequence!=episode.last_actor_sequence:
                return
            if any(row.evidence.receipt is None for row in episode.rows):return
        with self._condition:
            if self._closed or not self._queue.empty() or self._ingress_items>int(self._processing):return
            self._sealing=True
        progress={'schemaVersion':1,'runID':self.run_id,'rngStreamID':self._sampler[0],
            'drawIndex':self._sampler[1],'rngState':list(self._sampler[2]),'actorResetGeneration':self._sampler[3]}
        report={'schemaVersion':1,'runID':self.run_id,'policyID':self.policy_id,'actorProgress':progress,
            'auditOnly':True,'episodes':[{'episodeID':episode.id,'stoppedNanos':episode.closed_nanos,
                'lastActorSequence':episode.last_actor_sequence,'decisions':len(episode.rows)} for episode in self._episodes.values()]}
        import os
        data=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        with self.audit_path.open('xb') as target:target.write(data);target.flush();os.fsync(target.fileno())
        self._spool.close()
        with self._condition:self._sealed=report;self._condition.notify_all()

    def close(self):
        with self._condition:
            if self._sealed is None and self._failure is None:
                self._failure=InterruptedError('Rollout assembly closed before sealing')
            self._closed=True; self._condition.notify_all()
        self._thread.join(timeout=10)
        if self._thread.is_alive(): raise EnvironmentError('Rollout evidence worker did not join')
        while True:
            try: _,_,size,future,_=self._queue.get_nowait()
            except queue.Empty: break
            if future is not None: future.set_exception(InterruptedError('Rollout assembly closed'))
            with self._condition: self._queue_bytes-=size; self._ingress_items-=1
            self._queue.task_done()
        if not self._transferred: self._spool.close()
        self._episodes.clear(); self._rows.clear(); self._pending_evidence.clear(); self._sample_keys.clear()
        self._snapshots.clear()
