#!/usr/bin/env python3
"""Summarize only authenticated matching completed GRU diagnostic endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    with Path(path).open('rb') as source: return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    progress = json.loads((root / 'campaign-progress.json').read_text())
    evaluations = {(item['seed'], item['phase']): Path(item['result']['evaluation']) for item in progress['operations'] if item['mode'] == 'evaluate'}
    rows = []; pending = []
    for seed in plan['headSeeds']:
        for arm in ('A', 'B', 'C'):
            path = root / f'run-{seed}-{arm}.json'; evaluation_path = evaluations.get((seed, arm))
            if not path.exists() or evaluation_path is None:
                pending.append([seed, arm]); continue
            report = json.loads(path.read_text()); evaluation = json.loads(evaluation_path.read_text())
            if report['completedUpdates'] != 512 or report['phase'] != 'completed':
                pending.append([seed, arm]); continue
            reference = report['checkpoint']; manifest = Path(reference['path']) / 'manifest.json'
            if (evaluation['checkpoint'] != reference or digest(manifest) != reference['manifestSHA256'] or
                digest(manifest.parent / 'policy.safetensors') != reference['policySHA256']):
                raise ValueError('Evaluation does not identify the unchanged matched checkpoint')
            if [item['episodeIDs'] for item in report['metrics']] != plan['armOrder']:
                raise ValueError('An arm changed the fixed exposure/order')
            if report['validDecisions'] != 27648 or report['supervisedPackets'] != 1024:
                raise ValueError('An arm has unmatched exposure counts')
            scored = []
            for split, layouts in [('train', plan['trainingLayouts']), ('development', plan['developmentLayouts'])]:
                for delay in (2000, 8000, 30000):
                    selected = [item for item in evaluation['rows'] if item['seed'] in layouts and item['delayMS'] == delay]
                    if len(selected) != len(layouts): raise ValueError('Evaluation is incomplete')
                    entry = {'split': split, 'delayMS': delay, 'rankingAccuracy': sum(item['correct'] for item in selected) / (2 * len(selected)),
                        'cueFlippedPreference': sum(item['cueChangedChoice'] for item in selected) / len(selected)}
                    for name, greedy in [('greedy', True), ('sampled', False)]:
                        packets = [packet for item in selected for packet in item['readoutPackets'] if packet['greedy'] == greedy]
                        outcomes = [packet.get('execution', {}).get('outcome', 'invalid_packet') for packet in packets]
                        entry[name] = {'packets': len(packets), 'successRate': outcomes.count('success') / len(outcomes),
                            'outcomes': {outcome: outcomes.count(outcome) for outcome in sorted(set(outcomes))},
                            'firstOperationEND': sum(packet['firstOperation'] == 0 for packet in packets)}
                    scored.append(entry)
            rows.append({'headSeed': seed, 'arm': arm, 'checkpoint': reference, 'evaluation': str(evaluation_path),
                'updates': report['completedUpdates'], 'validDecisions': report['validDecisions'], 'supervisedPackets': report['supervisedPackets'],
                'clipFraction': sum(item['clipMultiplier'] < 1 for item in report['metrics']) / 512,
                'meanChoiceNLLLast16': statistics.mean(item['choiceNLL'] for item in report['metrics'][-16:]),
                'medianPreClipNorm': statistics.median(item['preClipNorm'] for item in report['metrics']),
                'peakMLXBytes': report['peakMLXBytes'], 'scores': scored})
    summary = {'schemaVersion': 1, 'scope': 'three_head_seed_frozen_visual_GRU_clipping_diagnostic',
        'phase': 'completed' if len(rows) == 9 else 'incomplete', 'rows': rows, 'pending': pending,
        'productionDefaultsChanged': False, 'reservedTestUsed': False,
        'executionScope': 'one actual packet at passive readiness, followed only by empty packets; not autonomous waiting behavior'}
    path = root / 'summary.json'; temporary = root / 'summary.tmp'
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n'); temporary.replace(path)
    print('| Seed | Arm | Clipped | Train rank | Development rank | Development gated greedy | Development gated sampled |')
    print('|---:|:---:|---:|---:|---:|---:|---:|')
    for row in rows:
        train = next(item for item in row['scores'] if item['split'] == 'train' and item['delayMS'] == 30000)
        dev = next(item for item in row['scores'] if item['split'] == 'development' and item['delayMS'] == 30000)
        print(f"| {row['headSeed']} | {row['arm']} | {row['clipFraction']:.1%} | {train['rankingAccuracy']:.1%} | {dev['rankingAccuracy']:.1%} | {dev['greedy']['successRate']:.1%} | {dev['sampled']['successRate']:.1%} |")
    print('All displayed scores are at30-second delay; complete per-delay values and exact counts are in', path)
    print('Pending:', pending)


if __name__ == '__main__': main()
