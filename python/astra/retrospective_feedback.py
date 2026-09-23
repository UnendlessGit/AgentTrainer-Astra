"""CPU-only validation of immutable retrospective manual-reward revisions.

The collector remains the authority on source eligibility. This module accepts
its digest-bound projection; it never upgrades an aborted/audited source, edits
observations/actions/outcomes, or turns late annotations into live markers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import uuid

MAX_BYTES = 128 * 1024 * 1024
MAX_INTERVALS = 65_536
MAX_PAIRS = 262_144
MAX_ANNOTATIONS = 65_536
MAX_REVIEWS = 4_096
MAX_COUNT = 4_096
MAX_WALL_MS = 253_402_300_799_999


class FeedbackError(ValueError):
    pass


def _fields(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise FeedbackError('Missing or unknown feedback fields')


def _integer(value, maximum=2**64 - 1, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise FeedbackError('Feedback integer is outside its supported range')
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise FeedbackError('Feedback values must be finite numbers')
    return value


def _id(value):
    if type(value) is not str or len(value) != 36:
        raise FeedbackError('Feedback identity must be a UUID')
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise FeedbackError('Invalid feedback UUID') from error


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise FeedbackError('Feedback requires a lower-case SHA-256 digest')
    return value


def _json(data):
    if type(data) is not bytes or not 0 < len(data) <= MAX_BYTES:
        raise FeedbackError('Feedback artifact exceeds its byte limit')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FeedbackError('Duplicate feedback JSON key')
            result[key] = value
        return result
    def reject(_):
        raise FeedbackError('Nonfinite feedback JSON')
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise FeedbackError('Invalid feedback JSON') from error


def _author(value):
    _fields(value, ('sessionID', 'clockID', 'observedNanos', 'wallTimeMS'))
    return (_id(value['sessionID']), _id(value['clockID']), _integer(value['observedNanos']),
            _integer(value['wallTimeMS'], MAX_WALL_MS, 1))


def _follows(value, previous):
    current, old = _author(value), _author(previous)
    if current[1] == old[1]:
        if current[2] < old[2]:
            raise FeedbackError('Feedback authoring time moved backwards')
    elif current[0] == old[0] or current[3] < old[3]:
        raise FeedbackError('A new authoring clock requires a distinct session and later wall-time audit')


def _target(value):
    _fields(value, ('episodeID', 'observationID', 'packetID', 'startNanos', 'endNanos'))
    result = tuple(_id(value[name]) for name in ('episodeID', 'observationID', 'packetID')) + (
        _integer(value['startNanos']), _integer(value['endNanos']))
    if result[3] >= result[4]:
        raise FeedbackError('Feedback interval must have positive duration')
    return result


@dataclass(frozen=True, init=False)
class VerifiedFeedbackSource:
    """Collector-authenticated projection and hash-bound original program.

    The native producer owns full visual/reset program validation. This reader
    independently checks the exact bounded manual subset, never silently drops
    a manual rule, and does not accept arbitrary reviewer-created eligibility.
    """
    _metadata_bytes: bytes
    _program_bytes: bytes
    sha256: str
    def __init__(self, metadata_bytes: bytes, expected_sha256: str, program_bytes: bytes):
        if type(metadata_bytes) is not bytes or type(program_bytes) is not bytes or len(metadata_bytes) > MAX_BYTES or len(program_bytes) > 1_048_576:
            raise FeedbackError('Feedback source/program byte limit exceeded')
        object.__setattr__(self, '_metadata_bytes', metadata_bytes)
        object.__setattr__(self, '_program_bytes', program_bytes)
        object.__setattr__(self, 'sha256', _digest(expected_sha256))
        if hashlib.sha256(metadata_bytes).hexdigest() != self.sha256:
            raise FeedbackError('Frozen feedback source digest mismatch')
        metadata, program = _json(metadata_bytes), _json(program_bytes)
        _fields(metadata, ('schemaVersion', 'collectionID', 'runID', 'clockID', 'sourceSessionID', 'policyID',
            'programID', 'trajectorySHA256', 'policySignature', 'programSHA256', 'status', 'controlClosureKnown',
            'closedNanos', 'closedWallTimeMS', 'intervals', 'manualRules'))
        if _integer(metadata['schemaVersion'], 1, 1) != 1 or metadata['status'] != 'awaiting_manual_review' or metadata['controlClosureKnown'] is not True:
            raise FeedbackError('Only joined collector experience awaiting manual rewards is reviewable')
        for key in ('collectionID', 'runID', 'clockID', 'sourceSessionID', 'policyID', 'programID'):
            _id(metadata[key])
        for key in ('trajectorySHA256', 'policySignature', 'programSHA256'):
            _digest(metadata[key])
        _integer(metadata['closedNanos']); _integer(metadata['closedWallTimeMS'], MAX_WALL_MS, 1)
        if len(program_bytes) > 1_048_576 or hashlib.sha256(program_bytes).hexdigest() != metadata['programSHA256']:
            raise FeedbackError('Frozen reward-program digest mismatch')
        _fields(program, ('schemaVersion', 'id', 'name', 'signals', 'rules', 'maximumEpisodeMS'), ('success', 'failure', 'ready', 'resetPlan'))
        if _integer(program['schemaVersion'], 1, 1) != 1 or _id(program['id']) != _id(metadata['programID']):
            raise FeedbackError('Frozen reward-program identity mismatch')
        if type(program['signals']) is not list or len(program['signals']) > 32 or type(program['rules']) is not list or len(program['rules']) > 64:
            raise FeedbackError('Reward-program capacity exceeded')
        rules, ids = [], set()
        for rule in program['rules']:
            _fields(rule, ('id', 'name', 'kind', 'amount', 'maximumDelta'), ('predicate', 'signalID'))
            identifier = _id(rule['id'])
            if identifier in ids or rule['kind'] not in ('risingEdge', 'ratePerSecond', 'scoreDelta', 'manualMarker'):
                raise FeedbackError('Duplicate or unsupported reward rule')
            ids.add(identifier)
            if abs(_number(rule['amount'])) > 1_000_000 or not 0 < _number(rule['maximumDelta']) <= 1_000_000_000:
                raise FeedbackError('Invalid reward scale')
            if rule['kind'] == 'manualMarker':
                if 'predicate' in rule or 'signalID' in rule:
                    raise FeedbackError('Manual rules cannot impersonate another reward source')
                rules.append(rule)
        # The native collector validates the full program, including visual
        # and reset semantics. This narrow reader interprets only the complete
        # manual-rule projection under that authenticated original hash.
        projected = metadata['manualRules']
        if type(projected) is not list or not 1 <= len(projected) <= 64:
            raise FeedbackError('Invalid bounded manual-rule projection')
        actual_rules, seen = [], set()
        for rule in projected:
            _fields(rule, ('id', 'kind', 'amount'))
            identifier = _id(rule['id']); amount = _number(rule['amount'])
            if identifier in seen or rule['kind'] != 'manualMarker' or abs(amount) > 1_000_000:
                raise FeedbackError('Duplicate or unsupported projected manual rule')
            seen.add(identifier); actual_rules.append((identifier, rule['kind'], amount))
        expected_rules = sorted((_id(rule['id']), rule['kind'], rule['amount']) for rule in rules)
        if actual_rules != expected_rules:
            raise FeedbackError('Manual-rule projection differs from the complete original program')
        intervals = metadata['intervals']
        if type(intervals) is not list or not 1 <= len(intervals) <= MAX_INTERVALS or not rules or len(intervals) * len(rules) > MAX_PAIRS:
            raise FeedbackError('Review interval/rule capacity exceeded')
        packets, observations, episode_ends, last_sequence = set(), set(), {}, None
        for interval in intervals:
            _fields(interval, ('target', 'packetSequence', 'drawIndex', 'endpointObservationID', 'execution', 'outcome', 'eligible'), ('bootstrap',))
            episode, observed, packet, start, end = _target(interval['target'])
            sequence = _integer(interval['packetSequence'])
            if interval['eligible'] is not True or interval['execution'] not in ('executed', 'semanticBoundaryPrefix') or interval['outcome'] not in ('continuing', 'terminated', 'truncated'):
                raise FeedbackError('The source interval is not eligible for manual reward review')
            if interval['execution'] == 'semanticBoundaryPrefix' and interval['outcome'] == 'continuing':
                raise FeedbackError('Boundary prefixes require their original semantic endpoint')
            endpoint = _id(interval['endpointObservationID'])
            if end > metadata['closedNanos'] or endpoint == observed or sequence != _integer(interval['drawIndex']) or (last_sequence is not None and sequence <= last_sequence) or packet in packets or observed in observations or start < episode_ends.get(episode, 0):
                raise FeedbackError('Source interval identity, timing or sampled sequence mismatch')
            if 'bootstrap' in interval:
                value = interval['bootstrap']
                _fields(value, ('episodeID', 'policyID', 'observationID', 'cutoffNanos', 'value', 'recurrentReset'))
                if interval['outcome'] != 'truncated' or _id(value['episodeID']) != episode or _id(value['policyID']) != _id(metadata['policyID']) or _id(value['observationID']) != endpoint or _integer(value['cutoffNanos']) != end or value['recurrentReset'] is not False:
                    raise FeedbackError('Truncation must retain its exact same-episode pre-reset bootstrap')
                _number(value['value'])
            elif interval['outcome'] == 'truncated':
                raise FeedbackError('Truncated source interval is missing its bootstrap')
            packets.add(packet); observations.add(observed); episode_ends[episode] = end; last_sequence = sequence

    @property
    def metadata(self):
        return _json(self._metadata_bytes)

    @property
    def program(self):
        return _json(self._program_bytes)


def resolve_feedback(source: VerifiedFeedbackSource, authored: dict, annotations: list, reviews: list):
    metadata = source.metadata
    def check_author(value):
        session, clock, observed, wall = _author(value)
        if clock == _id(metadata['clockID']):
            if observed < metadata['closedNanos']:
                raise FeedbackError('Retrospective feedback precedes the joined source boundary')
        elif session == _id(metadata['sourceSessionID']) or wall < metadata['closedWallTimeMS']:
            raise FeedbackError('Different authoring clock requires a distinct session and later wall-time audit')
    check_author(authored)
    if type(annotations) is not list or len(annotations) > MAX_ANNOTATIONS or type(reviews) is not list or len(reviews) > MAX_REVIEWS:
        raise FeedbackError('Annotation/review capacity exceeded')
    intervals = metadata['intervals']
    by_packet = {_id(row['target']['packetID']): _target(row['target']) for row in intervals}
    rules = sorted((r for r in source.program['rules'] if r['kind'] == 'manualMarker'), key=lambda r: _id(r['id']))
    rule_ids = {_id(r['id']) for r in rules}
    ids, counts, last_label, previous = set(), {}, {}, None
    for index, label in enumerate(annotations):
        _fields(label, ('id', 'sequence', 'target', 'ruleID', 'count', 'authored'))
        check_author(label['authored']); _follows(authored, label['authored'])
        if previous is not None:
            _follows(label['authored'], previous)
        identifier, target = _id(label['id']), _target(label['target'])
        pair = target[2], _id(label['ruleID'])
        if _integer(label['sequence'], MAX_ANNOTATIONS) != index or identifier in ids or by_packet.get(target[2]) != target or pair[1] not in rule_ids:
            raise FeedbackError('Annotation target/rule/sequence mismatch')
        total = counts.get(pair, 0) + _integer(label['count'], MAX_COUNT, 1)
        if total > MAX_COUNT:
            raise FeedbackError('Interval manual count exceeds its bound')
        ids.add(identifier); counts[pair] = total; last_label[pair] = label['authored']; previous = label['authored']
    covered, total, previous = set(), 0, None
    for index, review in enumerate(reviews):
        _fields(review, ('id', 'sequence', 'pairs', 'authored'))
        check_author(review['authored']); _follows(authored, review['authored'])
        if previous is not None:
            _follows(review['authored'], previous)
        identifier = _id(review['id'])
        if type(review['pairs']) is not list or not review['pairs'] or _integer(review['sequence'], MAX_REVIEWS) != index or identifier in ids:
            raise FeedbackError('Review sequence/identity mismatch')
        total += len(review['pairs'])
        if total > MAX_PAIRS:
            raise FeedbackError('Review pair capacity exceeded')
        ids.add(identifier)
        for entry in review['pairs']:
            _fields(entry, ('packetID', 'ruleID'))
            pair = _id(entry['packetID']), _id(entry['ruleID'])
            if pair[0] not in by_packet or pair[1] not in rule_ids or pair in covered:
                raise FeedbackError('Review pairs must uniquely identify original interval/rule pairs')
            if pair in last_label:
                _follows(review['authored'], last_label[pair])
            covered.add(pair)
        previous = review['authored']
    if not counts.keys() <= covered:
        raise FeedbackError('Annotated pairs require explicit review completion')
    result = []
    for interval in intervals:
        packet = _id(interval['target']['packetID']); total, complete, components = 0.0, True, []
        for rule in rules:
            pair = packet, _id(rule['id'])
            component = {'ruleID': rule['id'], 'reviewed': pair in covered}
            if pair not in covered:
                complete = False
            else:
                count = counts.get(pair, 0); value = count * float(rule['amount']); total += value
                _number(value); _number(total)
                component.update(count=count, value=value)
            components.append(component)
        value = {'packetID': interval['target']['packetID'], 'components': components}
        if complete:
            value['manualReward'] = total
        result.append(value)
    return result


_LOADED_KEY = object()


@dataclass(frozen=True, init=False)
class LoadedFeedbackRevision:
    _data: bytes
    _reference: bytes
    def __init__(self, data: bytes, reference: dict, *, _key=None):
        if _key is not _LOADED_KEY:
            raise FeedbackError('Load and verify an artifact before using it as a revision parent')
        object.__setattr__(self, '_data', bytes(data))
        object.__setattr__(self, '_reference', json.dumps(reference).encode())

    @property
    def document(self):
        return _json(self._data)

    @property
    def reference(self):
        return _json(self._reference)


def _normalized_resolution(value):
    if type(value) is not list:
        raise FeedbackError('Reward resolution requires an array')
    result = []
    for row in value:
        _fields(row, ('packetID', 'components'), ('manualReward',))
        if type(row['components']) is not list or len(row['components']) > 64:
            raise FeedbackError('Invalid manual reward components')
        components = []
        for item in row['components']:
            _fields(item, ('ruleID', 'reviewed'), ('count', 'value'))
            if type(item['reviewed']) is not bool:
                raise FeedbackError('Review coverage must be Boolean')
            if item['reviewed']:
                if 'count' not in item or 'value' not in item:
                    raise FeedbackError('Reviewed component is missing its count/value')
                components.append((_id(item['ruleID']), True, _integer(item['count'], MAX_COUNT), _number(item['value'])))
            else:
                if 'count' in item or 'value' in item:
                    raise FeedbackError('Unreviewed reward cannot masquerade as zero')
                components.append((_id(item['ruleID']), False))
        result.append((_id(row['packetID']), components, _number(row['manualReward']) if 'manualReward' in row else None))
    return result


def validate_revision(document, source: VerifiedFeedbackSource, parent: LoadedFeedbackRevision | None = None):
    _fields(document, ('schemaVersion', 'id', 'revision', 'sourceSHA256', 'authored', 'annotations', 'reviews', 'resolved'), ('parent',))
    _integer(document['schemaVersion'], 1, 1); identifier = _id(document['id']); revision = _integer(document['revision'])
    if document['sourceSHA256'] != source.sha256:
        raise FeedbackError('Revision source digest mismatch')
    expected = resolve_feedback(source, document['authored'], document['annotations'], document['reviews'])
    if parent is None:
        if 'parent' in document or revision != 0:
            raise FeedbackError('Initial revision cannot claim an unverified parent')
    else:
        if type(parent) is not LoadedFeedbackRevision:
            raise FeedbackError('A correction requires a loaded and verified parent')
        prior, reference = parent.document, parent.reference
        _fields(document.get('parent'), ('id', 'sha256'))
        if _id(document['parent']['id']) != _id(reference['id']) or document['parent']['sha256'] != reference['sha256'] or prior['sourceSHA256'] != source.sha256 or identifier == _id(reference['id']) or revision != prior['revision'] + 1:
            raise FeedbackError('Reward revision does not continue its verified parent')
        _follows(document['authored'], prior['authored'])
        old_labels = {_id(v['id']): v for v in prior['annotations']}
        old_reviews = {_id(v['id']): v for v in prior['reviews']}
        for name, old, other in (('annotations', old_labels, old_reviews), ('reviews', old_reviews, old_labels)):
            for value in document[name]:
                key = _id(value['id'])
                if key not in old:
                    _follows(value['authored'], prior['authored'])
                if key in other or (key in old and value != old[key]):
                    raise FeedbackError('A correction cannot change an existing annotation/review identity')
    if _normalized_resolution(document['resolved']) != _normalized_resolution(expected):
        raise FeedbackError('Stored manual rewards differ from their annotations and explicit review coverage')


def load_revision(directory, reference, source: VerifiedFeedbackSource, parent: LoadedFeedbackRevision | None = None):
    _fields(reference, ('id', 'sha256'))
    identifier, digest = _id(reference['id']), _digest(reference['sha256'])
    directory = Path(directory)
    if not directory.is_absolute():
        raise FeedbackError('Feedback revision directory must be absolute')
    folder = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fd = os.open(identifier + '.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=folder)
        with os.fdopen(fd, 'rb') as source_file:
            info = os.fstat(source_file.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
                raise FeedbackError('Invalid feedback artifact file type or size')
            data = source_file.read(MAX_BYTES + 1)
        if len(data) != info.st_size or hashlib.sha256(data).hexdigest() != digest:
            raise FeedbackError('Feedback revision expected digest mismatch')
        document = _json(data)
        if _id(document.get('id')) != identifier:
            raise FeedbackError('Feedback revision file identity mismatch')
        validate_revision(document, source, parent)
        return LoadedFeedbackRevision(data, reference, _key=_LOADED_KEY)
    finally:
        os.close(folder)
