"""Environment-neutral actor boundary. Observation clocks are not frame clocks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable, Protocol, runtime_checkable
import uuid

import numpy as np

from astra.model.actions import ActionVocabulary
from astra.recordings import validate_frame

CancelCheck = Callable[[], bool]


class EnvironmentError(ValueError):
    pass


def integer(value, maximum=2**64 - 1, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise EnvironmentError('Environment integer is outside its contract')
    return value


def identifier(value):
    if type(value) is not str or len(value) != 36:
        raise EnvironmentError('Environment identity must be a UUID')
    try:
        uuid.UUID(value)
    except ValueError as error:
        raise EnvironmentError('Environment identity must be a UUID') from error
    return value


def controls(value, cutoff):
    if not isinstance(value, dict) or set(value) != {'keys', 'buttons', 'modifiers', 'pointer', 'observedNanos', 'revision', 'valid'}:
        raise EnvironmentError('Environment control snapshot has incompatible fields')
    for field, maximum in (('keys', 127), ('buttons', 31)):
        values = value[field]
        if type(values) is not list or any(type(item) is not int or not 0 <= item <= maximum for item in values) or values != sorted(set(values)):
            raise EnvironmentError('Control keys/buttons must be canonical bounded integer arrays')
    integer(value['modifiers']); integer(value['revision'])
    if integer(value['observedNanos']) > cutoff or type(value['valid']) is not bool:
        raise EnvironmentError('Control snapshot is unavailable at the observation cutoff')
    point = value['pointer']
    if not isinstance(point, dict) or set(point) != {'x', 'y'} or any(type(v) not in (int, float) or not math.isfinite(v) for v in point.values()):
        raise EnvironmentError('Control pointer must contain finite global coordinates')
    return value


def owned_bgra(pixels):
    """Reuse proven immutable byte backing; detach other producer storage."""
    base = pixels
    while type(base) is np.ndarray:
        base = base.base
    if type(base) is bytes and not pixels.flags.writeable and pixels.flags.c_contiguous:
        return pixels
    return np.frombuffer(pixels.tobytes(), dtype=pixels.dtype).reshape(pixels.shape)


def fields(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise EnvironmentError('External environment fields do not match the negotiated contract')
    return value


def uuid_key(value):
    return str(uuid.UUID(identifier(value)))


def same_id(left, right):
    if left is None or right is None: return left is right
    return uuid_key(left) == uuid_key(right)


@dataclass(frozen=True)
class EnvironmentSpec:
    """Stable data/learning settings; live run, clock and process IDs are separate."""
    identity: str
    action_vocabulary: ActionVocabulary
    period_ms: int = 100
    lead_ms: int = 100
    maximum_episode_ms: int = 120000
    maximum_observation_bytes: int = 256 * 1024**2
    maximum_surfaces: int = 16
    maximum_frame_age_ms: int = 250
    discount_half_life_ms: float = 30000
    seed: int = 0
    seeded_reset: bool = False
    reward_signature: str = ''
    reset_signature: str = ''
    schema_version: int = 1

    def validate(self):
        if type(self.identity) is not str or not 1 <= len(self.identity.encode()) <= 256:
            raise EnvironmentError('Environment identity must be a bounded stable name')
        self.action_vocabulary.validate()
        integer(self.schema_version, 1, 1); integer(self.period_ms, 1000, 1); integer(self.lead_ms, 2000)
        integer(self.maximum_episode_ms, 3600000, 1); integer(self.maximum_observation_bytes, 1024**3, 4)
        integer(self.maximum_surfaces, 16, 1); integer(self.maximum_frame_age_ms, 60000, 1)
        integer(self.seed, 2**63 - 1)
        if type(self.seeded_reset) is not bool or type(self.discount_half_life_ms) not in (int, float) or not math.isfinite(self.discount_half_life_ms) or self.discount_half_life_ms <= 0:
            raise EnvironmentError('Invalid environment reset/discount settings')
        for value in (self.reward_signature, self.reset_signature):
            if type(value) is not str or (value and (len(value) != 64 or any(c not in '0123456789abcdef' for c in value))):
                raise EnvironmentError('Environment programs require SHA-256 fingerprints')
        return self

    @property
    def signature(self):
        self.validate()
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def to_dict(self):
        self.validate()
        return {**asdict(self), 'action_vocabulary': self.action_vocabulary.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or value.keys() != cls.__dataclass_fields__.keys():
            raise EnvironmentError('Environment specification must contain every versioned field')
        return cls(**{**value, 'action_vocabulary': ActionVocabulary.from_dict(value['action_vocabulary'])}).validate()


@dataclass(frozen=True)
class SurfaceObservation:
    pixels: np.ndarray
    metadata: dict
    coverage_nanos: int
    coverage_kind: str = 'frame'

    def validate(self, *, cutoff: int, maximum_age_ms: int):
        validate_frame(self.metadata, maximum_timestamp=2**64 - 1)
        source, available = self.metadata['eventNanos'], self.metadata['observedNanos']
        coverage = integer(self.coverage_nanos)
        if not source <= available <= coverage <= cutoff or self.coverage_kind not in ('frame', 'unchanged'):
            raise EnvironmentError('Visual source/availability/coverage clocks disagree with the observation cutoff')
        if cutoff - coverage > maximum_age_ms * 1000000:
            raise EnvironmentError('Visual coverage is stale at this observation cutoff')
        if self.coverage_kind == 'frame' and (coverage != available or cutoff - source > maximum_age_ms * 1000000):
            raise EnvironmentError('A stale frame cannot masquerade as a fresh observation')
        surface = self.metadata['surface']
        if not isinstance(self.pixels, np.ndarray) or self.pixels.dtype != np.uint8 or self.pixels.shape != (surface['pixelHeight'], surface['pixelWidth'], 4):
            raise EnvironmentError('Owned BGRA pixels do not match their surface metadata')
        return self


@dataclass(frozen=True)
class EnvironmentObservation:
    id: str
    episode_id: str
    cutoff_nanos: int
    geometry_revision: int
    frames: tuple[SurfaceObservation, ...]
    control_state: dict

    def validate(self, spec: EnvironmentSpec):
        identifier(self.id); identifier(self.episode_id); integer(self.cutoff_nanos); integer(self.geometry_revision)
        if type(self.frames) is not tuple or not 1 <= len(self.frames) <= spec.maximum_surfaces:
            raise EnvironmentError('Observation exceeds its declared surface capacity')
        for frame in self.frames:
            frame.validate(cutoff=self.cutoff_nanos, maximum_age_ms=spec.maximum_frame_age_ms)
        if len({frame.metadata['surface']['id'] for frame in self.frames}) != len(self.frames):
            raise EnvironmentError('Observation surface roles must be unique')
        if sum(frame.pixels.nbytes for frame in self.frames) > spec.maximum_observation_bytes:
            raise EnvironmentError('Observation exceeds its declared byte capacity')
        controls(self.control_state, self.cutoff_nanos)
        return self


@dataclass(frozen=True)
class DecisionContext:
    run_id: str
    episode_id: str
    policy_id: str
    observation_id: str
    packet_id: str
    episode_step: int
    cutoff_nanos: int
    geometry_revision: int

    def validate(self):
        for value in (self.run_id, self.episode_id, self.observation_id, self.packet_id): identifier(value)
        if type(self.policy_id) is not str or not 1 <= len(self.policy_id.encode()) <= 256:
            raise EnvironmentError('Decision policy identity is invalid')
        integer(self.episode_step); integer(self.cutoff_nanos); integer(self.geometry_revision)
        return self


@dataclass(frozen=True)
class EnvironmentTransition:
    observation: EnvironmentObservation
    reward: float
    duration_ms: int
    outcome: str
    raw_events: tuple[dict, ...]
    command_results: tuple[dict, ...] = ()
    provenance: str = 'agent'
    reason: str | None = None
    outcome_detail: str | None = None


@runtime_checkable
class EnvironmentAdapter(Protocol):
    spec: EnvironmentSpec
    run_id: str
    @property
    def signature(self) -> str: ...
    @property
    def outcome(self) -> str: ...
    def reset(self, *, seed: int, cancelled: CancelCheck) -> EnvironmentObservation: ...
    def step(self, commands: list[dict], *, context: DecisionContext, cancelled: CancelCheck) -> EnvironmentTransition: ...
    def seal_episode(self, *, cancelled: CancelCheck) -> None: ...
    def abort(self, reason: str) -> None: ...
