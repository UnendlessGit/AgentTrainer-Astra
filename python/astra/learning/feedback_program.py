"""Frozen manual deferral contract, separate from live marker semantics."""
from __future__ import annotations
import base64
import hashlib
import math

from astra.environments.interface import EnvironmentError, fields, uuid_key
from astra.retrospective_feedback import _json, _digest, _fields, _number, _integer


def program_binding(value):
    fields(value, ('programBase64', 'programSHA256', 'sourceSessionID'))
    uuid_key(value['sourceSessionID']); _digest(value['programSHA256'])
    if type(value['programBase64']) is not str or len(value['programBase64']) > 1_398_104:
        raise EnvironmentError('Frozen reward program exceeds its byte limit')
    try: data = base64.b64decode(value['programBase64'], validate=True)
    except (ValueError, TypeError) as error: raise EnvironmentError('Invalid reward program encoding') from error
    if not 0 < len(data) <= 1_048_576 or hashlib.sha256(data).hexdigest() != value['programSHA256']:
        raise EnvironmentError('Frozen reward program digest mismatch')
    program = _json(data)
    _fields(program, ('schemaVersion', 'id', 'name', 'signals', 'rules', 'maximumEpisodeMS'),
            ('success', 'failure', 'ready', 'resetPlan'))
    _integer(program['schemaVersion'], 1, 1); uuid_key(program['id'])
    if type(program['signals']) is not list or len(program['signals']) > 32 or type(program['rules']) is not list or not 1 <= len(program['rules']) <= 64:
        raise EnvironmentError('Invalid bounded reward program')
    seen = set()
    for rule in program['rules']:
        _fields(rule, ('id', 'name', 'kind', 'amount', 'maximumDelta'), ('predicate', 'signalID'))
        identifier = uuid_key(rule['id'])
        if identifier in seen or rule['kind'] not in ('risingEdge', 'ratePerSecond', 'scoreDelta', 'manualMarker'):
            raise EnvironmentError('Duplicate or unsupported frozen reward rule')
        if abs(_number(rule['amount'])) > 1_000_000 or not 0 < _number(rule['maximumDelta']) <= 1_000_000_000:
            raise EnvironmentError('Invalid reward scale')
        if rule['kind'] == 'manualMarker' and ('predicate' in rule or 'signalID' in rule):
            raise EnvironmentError('Manual rules cannot impersonate automatic signals')
        seen.add(identifier)
    if not any(rule['kind'] == 'manualMarker' for rule in program['rules']):
        raise EnvironmentError('Retrospective collection requires a manual reward rule')
    return data, program


def automatic_reward(value, program):
    """Check the complete ordered split; no omitted rule is treated as zero."""
    fields(value, ('clockID', 'episodeID', 'startNanos', 'endNanos', 'value',
                   'automaticComponents', 'deferredManualRuleIDs'))
    automatic = [r for r in program['rules'] if r['kind'] != 'manualMarker']
    manual = [uuid_key(r['id']) for r in program['rules'] if r['kind'] == 'manualMarker']
    components, deferred = value['automaticComponents'], value['deferredManualRuleIDs']
    if type(components) is not list or len(components) != len(automatic) or type(deferred) is not list or [uuid_key(v) for v in deferred] != manual:
        raise EnvironmentError('Reward deferral must enumerate every original rule in program order')
    subtotal = 0.0; known = True
    for component, rule in zip(components, automatic, strict=True):
        fields(component, ('ruleID', 'value'))
        if uuid_key(component['ruleID']) != uuid_key(rule['id']):
            raise EnvironmentError('Automatic reward component order differs from frozen program')
        if component['value'] is None: known = False
        else:
            amount=_number(component['value']);duration=(_integer(value['endNanos'])-_integer(value['startNanos']))/1_000_000_000
            if duration<=0:raise EnvironmentError('Automatic reward interval must have positive duration')
            if rule['kind']=='risingEdge' and amount not in (0,rule['amount']):
                raise EnvironmentError('Rising-edge component changed its frozen reward amount')
            if rule['kind']=='ratePerSecond':
                expected=rule['amount']*duration
                if not math.isclose(amount,expected,rel_tol=1e-12,abs_tol=0) and not ('predicate' in rule and amount==0):
                    raise EnvironmentError('Rate component changed its original interval integral')
            if rule['kind']=='scoreDelta' and abs(amount)>abs(rule['amount'])*rule['maximumDelta']:
                raise EnvironmentError('Score component exceeded its frozen delta limit')
            subtotal+=amount
    if not math.isfinite(subtotal): raise EnvironmentError('Automatic reward subtotal overflow')
    if known:
        if _number(value['value']) != subtotal: raise EnvironmentError('Automatic reward subtotal disagrees with components')
    elif value['value'] is not None:
        raise EnvironmentError('Unknown automatic components cannot become known reward')
    return {key: value[key] for key in ('automaticComponents', 'deferredManualRuleIDs')}
