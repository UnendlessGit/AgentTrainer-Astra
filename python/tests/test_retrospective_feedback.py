from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from astra.retrospective_feedback import (FeedbackError, VerifiedFeedbackSource, LoadedFeedbackRevision,
    load_revision, resolve_feedback, validate_revision)


def uid(): return str(uuid.uuid4())
def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return hashlib.sha256(value).hexdigest()


@pytest.fixture
def feedback():
    positive, negative, episode, policy, second, endpoint = [uid() for _ in range(6)]
    program = dict(schemaVersion=1, id=uid(), name='Feedback', signals=[], maximumEpisodeMS=120000,
        rules=[dict(id=positive, name='Good', kind='manualMarker', amount=1.0, maximumDelta=100000),
               dict(id=negative, name='Bad', kind='manualMarker', amount=-2.0, maximumDelta=100000)])
    targets = [dict(episodeID=episode, observationID=uid(), packetID=uid(), startNanos=100, endNanos=200),
               dict(episodeID=episode, observationID=second, packetID=uid(), startNanos=200, endNanos=300)]
    metadata = dict(schemaVersion=1, collectionID=uid(), runID=uid(), clockID=uid(), sourceSessionID=uid(), policyID=policy,
        programID=program['id'], trajectorySHA256='a'*64, policySignature='b'*64, programSHA256=digest(encoded(program)),
        status='awaiting_manual_review', controlClosureKnown=True, closedNanos=400, closedWallTimeMS=1000,
        manualRules=sorted([dict(id=r['id'], kind=r['kind'], amount=r['amount']) for r in program['rules']], key=lambda r:r['id']),
        intervals=[dict(target=targets[0], packetSequence=7, drawIndex=7, endpointObservationID=second,
                        execution='executed', outcome='continuing', eligible=True),
                   dict(target=targets[1], packetSequence=8, drawIndex=8, endpointObservationID=endpoint,
                        execution='semanticBoundaryPrefix', outcome='truncated', eligible=True,
                        bootstrap=dict(episodeID=episode, policyID=policy, observationID=endpoint, cutoffNanos=300,
                                       value=.75, recurrentReset=False))])
    source = VerifiedFeedbackSource(encoded(metadata), digest(encoded(metadata)), encoded(program))
    def authored(nanos):
        return dict(sessionID=metadata['sourceSessionID'], clockID=metadata['clockID'], observedNanos=nanos, wallTimeMS=1000+nanos)
    annotations = [dict(id=uid(), sequence=0, target=targets[0], ruleID=positive, count=2, authored=authored(500))]
    reviews = [dict(id=uid(), sequence=0, authored=authored(550), pairs=[
        dict(packetID=targets[0]['packetID'], ruleID=positive), dict(packetID=targets[0]['packetID'], ruleID=negative),
        dict(packetID=targets[1]['packetID'], ruleID=negative)])]
    document = dict(schemaVersion=1, id=uid(), revision=0, sourceSHA256=source.sha256, authored=authored(600),
                    annotations=annotations, reviews=reviews,
                    resolved=resolve_feedback(source, authored(600), annotations, reviews))
    return source, metadata, program, document


def save_fixture(folder, document):
    data = encoded(document); reference = dict(id=document['id'], sha256=digest(data))
    (folder / (document['id'] + '.json')).write_bytes(data)
    return reference


def test_reviewed_zero_unknown_and_exact_boundary_prefix(feedback, tmp_path):
    source, metadata, _, document = feedback
    reference = save_fixture(tmp_path, document)
    loaded = load_revision(tmp_path, reference, source)
    assert loaded.document['resolved'][0]['manualReward'] == 2
    assert 'manualReward' not in loaded.document['resolved'][1]
    assert any(item == {'ruleID': item['ruleID'], 'reviewed': False} for item in loaded.document['resolved'][1]['components'])
    assert any(item['reviewed'] and item['count'] == 0 and item['value'] == 0 for item in loaded.document['resolved'][0]['components'])
    assert source.metadata['intervals'][1]['bootstrap'] == metadata['intervals'][1]['bootstrap']
    assert source.metadata['intervals'][1]['execution'] == 'semanticBoundaryPrefix'
    copy = loaded.document; copy['resolved'][0]['manualReward'] = 88
    assert loaded.document['resolved'][0]['manualReward'] == 2
    with pytest.raises(FrozenInstanceError): source.sha256 = '0'*64


@pytest.mark.parametrize('mode', ['aborted', 'audited', 'unjoined', 'rejected', 'ineligible', 'bootstrap', 'draw', 'unknownField'])
def test_projection_cannot_bless_ineligible_source(feedback, mode):
    _, metadata, program, _ = feedback
    if mode in ('aborted', 'audited'): metadata['status'] = mode
    elif mode == 'unjoined': metadata['controlClosureKnown'] = False
    elif mode == 'rejected': metadata['intervals'][0]['execution'] = 'rejected'
    elif mode == 'ineligible': metadata['intervals'][0]['eligible'] = False
    elif mode == 'bootstrap': metadata['intervals'][1]['bootstrap']['cutoffNanos'] += 1
    elif mode == 'draw': metadata['intervals'][0]['drawIndex'] += 1
    else: metadata['modelFeatures'] = [1]
    with pytest.raises(FeedbackError):
        VerifiedFeedbackSource(encoded(metadata), digest(encoded(metadata)), encoded(program))


@pytest.mark.parametrize('mode', ['unreviewed', 'negative', 'countLimit', 'wrongRule', 'wrongInterval', 'duplicatePair', 'earlyReview', 'duplicateID', 'floatCount', 'boolCount'])
def test_annotations_require_real_review_of_exact_pairs(feedback, mode):
    source, _, _, document = feedback
    label, review = document['annotations'][0], document['reviews'][0]
    if mode == 'unreviewed': document['reviews'] = []
    elif mode == 'negative': label['count'] = -1
    elif mode == 'countLimit': label['count'] = 4097
    elif mode == 'wrongRule': label['ruleID'] = uid()
    elif mode == 'wrongInterval': label['target']['endNanos'] += 1
    elif mode == 'duplicatePair': review['pairs'].append(deepcopy(review['pairs'][0]))
    elif mode == 'earlyReview': review['authored']['observedNanos'] = 450
    elif mode == 'duplicateID': document['annotations'].append({**deepcopy(label), 'sequence': 1})
    elif mode == 'floatCount': label['count'] = 2.0
    else: label['count'] = True
    with pytest.raises(FeedbackError): validate_revision(document, source)


def test_cross_clock_annotations_use_wall_audit_without_comparing_monotonic_times(feedback):
    source, metadata, _, document = feedback
    authored = dict(sessionID=uid(), clockID=uid(), observedNanos=1, wallTimeMS=2000)
    document['authored'] = authored
    document['annotations'][0]['authored'] = authored
    document['reviews'][0]['authored'] = authored
    validate_revision(document, source)
    for bad in ({**authored, 'wallTimeMS': 999}, {**authored, 'sessionID': metadata['sourceSessionID']},
                {**authored, 'clockID': metadata['clockID'], 'observedNanos': 399}):
        document['authored'] = bad
        with pytest.raises(FeedbackError): validate_revision(document, source)


@pytest.mark.parametrize('mode', ['reward', 'feature', 'source', 'nonfinite', 'unknownZero'])
def test_even_rehashed_artifacts_must_rederive_without_feature_leakage(feedback, tmp_path, mode):
    source, metadata, _, document = feedback
    original_source = encoded(metadata)
    if mode == 'reward': document['resolved'][0]['manualReward'] = 999
    elif mode == 'feature': document['packetFields'] = {'reward': [1]}
    elif mode == 'source': document['sourceSHA256'] = '0'*64
    elif mode == 'nonfinite': document['resolved'][0]['manualReward'] = 1e300 * 1e300
    else: document['resolved'][1]['manualReward'] = 0
    if mode == 'nonfinite':
        data = json.dumps(document).encode(); reference = dict(id=document['id'], sha256=digest(data))
        (tmp_path / (document['id'] + '.json')).write_bytes(data)
    else: reference = save_fixture(tmp_path, document)
    with pytest.raises(FeedbackError): load_revision(tmp_path, reference, source)
    assert encoded(source.metadata) == original_source
    assert not {'observations', 'frames', 'commands', 'packetFields', 'stateBefore'} & source.metadata.keys()


def test_corrections_require_verified_immutable_parent(feedback, tmp_path):
    source, _, _, document = feedback
    first_ref = save_fixture(tmp_path, document); first = load_revision(tmp_path, first_ref, source)
    correction = deepcopy(document)
    correction.update(id=uid(), revision=1, parent=first_ref)
    correction['authored']['observedNanos'] = 700
    correction['authored']['wallTimeMS'] = 1700
    correction['annotations'][0].update(id=uid(), count=1)
    correction['annotations'][0]['authored']['observedNanos'] = 650
    correction['reviews'][0]['id'] = uid()
    correction['reviews'][0]['authored']['observedNanos'] = 675
    correction['resolved'] = resolve_feedback(source, correction['authored'], correction['annotations'], correction['reviews'])
    second_ref = save_fixture(tmp_path, correction)
    with pytest.raises(FeedbackError): load_revision(tmp_path, second_ref, source)
    assert load_revision(tmp_path, second_ref, source, first).document['resolved'][0]['manualReward'] == 1
    assert load_revision(tmp_path, first_ref, source).document == document
    correction['annotations'][0]['id'] = document['annotations'][0]['id']
    with pytest.raises(FeedbackError): validate_revision(correction, source, first)
    with pytest.raises(FeedbackError): LoadedFeedbackRevision(encoded(document), first_ref)


def test_digest_symlinks_duplicate_keys_and_sparse_oversize_are_rejected(feedback, tmp_path):
    source, _, _, document = feedback
    reference = save_fixture(tmp_path, document); path = tmp_path / (document['id'] + '.json')
    with pytest.raises(FeedbackError): load_revision(tmp_path, {**reference, 'sha256': '0'*64}, source)
    original = tmp_path / 'original'; path.rename(original); path.symlink_to(original)
    with pytest.raises(OSError): load_revision(tmp_path, reference, source)
    path.unlink()
    data = encoded(document).replace(b'{', b'{"id":"' + document['id'].encode() + b'",', 1)
    path.write_bytes(data)
    with pytest.raises(FeedbackError): load_revision(tmp_path, {**reference, 'sha256': digest(data)}, source)
    with path.open('wb') as file: file.truncate(128 * 1024 * 1024 + 1)
    with pytest.raises(FeedbackError): load_revision(tmp_path, reference, source)


@pytest.mark.parametrize('kind', ['annotations', 'reviews'])
def test_new_correction_records_cannot_be_backdated_before_verified_parent(feedback, tmp_path, kind):
    source, _, _, document = feedback
    reference = save_fixture(tmp_path, document); parent = load_revision(tmp_path, reference, source)
    correction = deepcopy(document); correction.update(id=uid(), revision=1, parent=reference)
    correction['authored']['observedNanos'] = 700
    correction[kind][0]['id'] = uid()
    with pytest.raises(FeedbackError, match='backwards'):
        validate_revision(correction, source, parent)


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'amount', 'kind', 'extraOriginalManual'])
def test_manual_projection_exactly_covers_hash_bound_original_rules(feedback, mode):
    _, metadata, program, _ = feedback
    if mode == 'missing': metadata['manualRules'].pop()
    elif mode == 'duplicate': metadata['manualRules'].append(deepcopy(metadata['manualRules'][0]))
    elif mode == 'amount': metadata['manualRules'][0]['amount'] += 1
    elif mode == 'kind': metadata['manualRules'][0]['kind'] = 'scoreDelta'
    else:
        program['rules'].append(dict(id=uid(), name='Additional manual', kind='manualMarker', amount=3, maximumDelta=100000))
        metadata['programSHA256'] = digest(encoded(program))
    with pytest.raises(FeedbackError):
        VerifiedFeedbackSource(encoded(metadata), digest(encoded(metadata)), encoded(program))
