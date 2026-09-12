"""Owned multi-surface snapshots shared by synchronous and asynchronous consumers."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
from astra.recordings import validate_frame, validate_event
from .interface import (EnvironmentError, EnvironmentObservation, SurfaceObservation, integer, identifier,
                        fields, uuid_key, same_id, owned_bgra)


def json_geometry(surface):
    import json
    return json.dumps(surface, sort_keys=True, separators=(',', ':'))

@dataclass(frozen=True)
class ResolvedFrame:
    pixels: np.ndarray
    acknowledgement: dict


class FrameRingResolver:
    """Resolve only pre-bound rings; message paths never grant file access."""
    def __init__(self, readers):
        from astra.frame_ring import FrameRingReader
        if not isinstance(readers, dict) or not readers or any(not isinstance(reader, FrameRingReader) or key != str(reader.ring_id)
                                                             for key, reader in readers.items()):
            raise EnvironmentError('Frame resolver requires explicit ring bindings')
        self._readers = dict(readers)

    def __call__(self, reference, metadata):
        if type(reference) is not dict or reference.get('metadata') != metadata:
            raise EnvironmentError('Observation metadata disagrees with its bound frame reference')
        reader = self._readers.get(uuid_key(reference.get('ringID')))
        if reader is None:
            raise EnvironmentError('Observation references an unbound frame ring')
        owned = reader.copy_cpu_frame(reference)
        return ResolvedFrame(owned.pixels, owned.acknowledgement)




class SnapshotDecoder:
    def __init__(self, spec, resolve_frame, on_consumed):
        self.spec=spec
        self._resolve_frame=resolve_frame
        self._on_consumed=on_consumed
        self.reset()

    def reset(self):
        self._geometry=None
        self._observations=set()
        self._last_event_sequence=None

    def decode(self, value, *, episode, expected_cutoff=None):
        fields(value, ('id', 'episodeID', 'cutoffNanos', 'geometryRevision', 'frames', 'controlState', 'events'))
        obs_id = identifier(value['id'])
        cutoff = integer(value['cutoffNanos'])
        if expected_cutoff is not None and cutoff != expected_cutoff:
            raise EnvironmentError('Observation cutoff disagrees with the independently sealed reward window')
        if not same_id(value['episodeID'], episode) or uuid_key(obs_id) in self._observations:
            raise EnvironmentError('Observation identity is stale or belongs to another episode')
        if type(value['frames']) is not list or not 1 <= len(value['frames']) <= self.spec.maximum_surfaces:
            raise EnvironmentError('Observation exceeds its declared surface capacity')
        prepared, acknowledgements = [], []
        byte_count = 0
        for frame in value['frames']:
            fields(frame, ('metadata', 'reference', 'coverageNanos', 'coverageKind'))
            metadata = copy.deepcopy(validate_frame(frame['metadata'], maximum_timestamp=2**64 - 1))
            byte_count += metadata['byteCount']
            if byte_count > self.spec.maximum_observation_bytes:
                raise EnvironmentError('Observation exceeds its declared byte capacity')
            # Validate clocks before acquiring/copying an out-of-band frame.
            source, available, coverage = metadata['eventNanos'], metadata['observedNanos'], integer(frame['coverageNanos'])
            if not source <= available <= coverage <= cutoff or frame['coverageKind'] not in ('frame', 'unchanged'):
                raise EnvironmentError('Visual frame clocks are inconsistent with their independent observation cutoff')
            if cutoff - coverage > self.spec.maximum_frame_age_ms * 1000000 or (
                frame['coverageKind'] == 'frame' and (coverage != available or cutoff - source > self.spec.maximum_frame_age_ms * 1000000)):
                raise EnvironmentError('Visual frame/coverage is stale')
            resolved = self._resolve_frame(copy.deepcopy(frame['reference']), metadata)
            if not isinstance(resolved, ResolvedFrame):
                raise EnvironmentError('Frame resolver did not establish owned pixel storage')
            surface = metadata['surface']
            if not isinstance(resolved.pixels, np.ndarray) or resolved.pixels.dtype != np.uint8 or resolved.pixels.shape != (surface['pixelHeight'], surface['pixelWidth'], 4):
                raise EnvironmentError('Resolved pixels disagree with their bound frame metadata')
            owned = owned_bgra(resolved.pixels)
            prepared.append(SurfaceObservation(owned, metadata, coverage, frame['coverageKind']))
            acknowledgements.append(copy.deepcopy(resolved.acknowledgement))
            # Every successful ownership transfer is released even if another
            # surface or later control metadata invalidates this observation.
            self._on_consumed(obs_id, [acknowledgements[-1]])
        observation = EnvironmentObservation(obs_id, episode, cutoff, integer(value['geometryRevision']),
            tuple(prepared), copy.deepcopy(value['controlState'])).validate(self.spec)
        if not observation.control_state['valid']:
            raise EnvironmentError('Authoritative controls are unavailable for this observation')
        if cutoff - observation.control_state['observedNanos'] > self.spec.maximum_frame_age_ms * 1000000:
            raise EnvironmentError('Authoritative controls are stale at this observation cutoff')
        geometry = (observation.geometry_revision, tuple(json_geometry(frame.metadata['surface']) for frame in prepared))
        if self._geometry is not None and geometry != self._geometry:
            raise EnvironmentError('Surface roles or geometry changed without a confirmed reset')
        self._geometry = geometry
        if type(value['events']) is not list or len(value['events']) > 4096:
            raise EnvironmentError('Executed-input history exceeds its bounded window')
        events = []
        for raw in value['events']:
            event = copy.deepcopy(validate_event(raw, maximum_timestamp=2**64 - 1))
            if event['eventNanos'] > event['observedNanos'] or max(event['eventNanos'], event['observedNanos']) > cutoff or (
                self._last_event_sequence is not None and event['sequence'] <= self._last_event_sequence):
                raise EnvironmentError('Executed-input history is future, duplicated or out of order')
            if event['origin'] in ('physical', 'boundary') or event['kind'] == 'gap':
                raise EnvironmentError('External intervention or an input-history gap invalidates the rollout')
            self._last_event_sequence = event['sequence']
            events.append(event)
        self._observations.add(uuid_key(obs_id))
        return observation, tuple(events)
