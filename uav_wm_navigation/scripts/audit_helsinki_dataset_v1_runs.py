#!/usr/bin/env python3
"""Read-only, exact multi-run Helsinki QA; process restarts are not auto-resets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import _bootstrap  # noqa: F401

ROOT = _bootstrap.PROJECT_ROOT.parent
sys.path.insert(0, str(ROOT))
from uav_wm_navigation.data.helsinki_dataset_v1_qa import audit_helsinki_collection


def audit_runs(roots, expected_episodes):
    reports, manifests, resets = [], [], []
    all_dt, all_clearance = [], []
    for root in map(Path, roots):
        root = root.resolve()
        summary_path = next((root / name for name in (
            'collection_summary.json', 'collection_failure.json', 'collection_progress.json'
        ) if (root / name).exists()), None)
        summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path else {}
        run_resets = summary.get('reset_transitions', [])
        report = audit_helsinki_collection(root, expected_episodes=len(list(root.glob('*.h5'))),
                                          reset_transitions=run_resets)
        reports.append(report)
        resets.extend(run_resets)
        manifests.append({'directory': str(root), 'summary': str(summary_path) if summary_path else None,
                          'episode_count': report['episode_count'], 'reset': report['reset'],
                          'collector_process_ids': summary.get('collector_process_ids', []),
                          'first_record': (summary.get('records') or [None])[0],
                          'last_record': (summary.get('records') or [None])[-1]})
        for episode in report['episodes']:
            with h5py.File(episode['path'], 'r') as handle:
                all_dt.append(handle['timestamps/dt'][:])
                all_clearance.append(handle['labels/minimum_clearance'][:])

    episodes = [e for r in reports for e in r['episodes']]
    def index(item):
        match = re.search(r'_smoke_(\d+)_', item['episode_id'])
        return int(match.group(1)) if match else -1
    episodes.sort(key=index)
    indices = [index(e) for e in episodes]
    transitions = sum(e['steps'] for e in episodes)
    stale_count = sum(e['stale_action_count'] for e in episodes)
    success_count = sum(e['success'] for e in episodes)
    collision_count = sum(e['collision'] for e in episodes)
    dt = np.concatenate(all_dt) if all_dt else np.empty(0)
    clearance = np.concatenate(all_clearance) if all_clearance else np.empty(0)
    tasks = {}
    for task in ('building_blocked', 'street_canyon', 'rooftop_to_ground',
                 'ground_to_rooftop', 'rooftop_to_rooftop'):
        selected = [e for e in episodes if e['task_type'] == task]
        steps = sum(e['steps'] for e in selected)
        stale = sum(e['stale_action_count'] for e in selected)
        tasks[task] = {'episodes': len(selected), 'successes': sum(e['success'] for e in selected),
                       'collisions': sum(e['collision'] for e in selected), 'transitions': steps,
                       'stale_action_count': stale, 'stale_action_ratio': stale / steps if steps else 0,
                       'maximum_stale_action_burst': max((e['maximum_stale_action_burst'] for e in selected), default=0)}
    phases = {}
    for phase in ('start', 'middle', 'end'):
        steps = sum(e['stale_action_by_phase'][phase]['transitions'] for e in episodes)
        stale = sum(e['stale_action_by_phase'][phase]['stale_action_count'] for e in episodes)
        phases[phase] = {'transitions': steps, 'stale_action_count': stale,
                         'stale_action_ratio': stale / steps if steps else 0}

    boundaries = []
    nonempty = [m for m in manifests if m['episode_count']]
    for previous, current in zip(nonempty, nonempty[1:]):
        first = current['first_record'] or {}
        last = previous['last_record'] or {}
        evidence = first.get('reset_evidence', {})
        checks = {
            'new_policy_step_zero': evidence.get('first_policy_step_id') == 0,
            'action_buffer_reset': evidence.get('action_buffer_reset') is True,
            'new_policy_step_observed': evidence.get('new_policy_step_observed') is True,
            'initial_pose_reset': evidence.get('start_position_error_m', float('inf')) <= 0.25,
            'initial_speed_reset': first.get('initial_speed_mps', float('inf')) <= 0.25,
            'initial_acceleration_reset': first.get('initial_acceleration_mps2', float('inf')) <= 0.25,
            'initial_yaw_reset': evidence.get('initial_yaw_error_degrees', float('inf')) <= 2.0,
            'writer_closed': last.get('reset_evidence', {}).get('writer_flush_closed') is True,
        }
        boundaries.append({'from_episode': last.get('episode_id'), 'to_episode': first.get('episode_id'),
                           'kind': 'process_restart_not_continuous_auto_reset', 'checks': checks,
                           'status': 'PASS' if all(checks.values()) else 'LIMITATION'})
    reset_expected = sum(max(0, r['episode_count'] - 1) for r in reports)
    reset_pass = sum(x.get('automatic_reset') == 'PASS' for x in resets)
    cross_stale = sum(r['stale_action']['cross_episode_inheritance_count'] for r in reports)
    corrupted = [e for r in reports for e in r['corrupted_hdf5']]
    partials = [e for r in reports for e in r['partial_files']]
    hover_correct = all(e['stale_executed_timeout_hover_correct'] for e in episodes)
    gates = {
        'episode_count': len(episodes) == expected_episodes,
        'unique_contiguous_episode_ids': indices == list(range(expected_episodes)),
        'balanced_task_counts': max(t['episodes'] for t in tasks.values()) - min(t['episodes'] for t in tasks.values()) <= 1,
        'success_rate_at_least_98_percent': bool(episodes) and success_count / len(episodes) >= .98,
        'collision_count_zero': collision_count == 0,
        'minimum_clearance_at_least_2p5m': bool(clearance.size) and float(clearance.min()) >= 2.5,
        'within_run_reset_count_and_success': len(resets) == reset_expected and reset_pass == reset_expected,
        'cross_run_fresh_start_evidence': all(b['status'] == 'PASS' for b in boundaries),
        'corrupted_hdf5_zero': not corrupted, 'partial_count_zero': not partials,
        'cross_episode_stale_action_zero': cross_stale == 0 and all(b['status'] == 'PASS' for b in boundaries),
        'all_stale_executed_actions_factual_hover': hover_correct,
        'all_dataset_integrity_checks': all(e['integrity_status'] == 'PASS' and all(e['integrity_checks'].values()) for e in episodes),
    }
    return {'schema': 'urbanfly-helsinki-multirun-qa-v1', 'status': 'PASS' if all(gates.values()) else 'FAIL',
            'gate_scope': 'Dataset integrity/outcomes across explicit continuation runs, not one uninterrupted collector',
            'expected_episodes': expected_episodes, 'episode_count': len(episodes), 'transition_count': transitions,
            'success_count': success_count, 'success_rate': success_count / len(episodes) if episodes else 0,
            'collision_count': collision_count, 'collision_rate': collision_count / len(episodes) if episodes else 0,
            'clearance_m': {'minimum': float(clearance.min()) if clearance.size else None,
                            'median': float(np.median(clearance)) if clearance.size else None},
            'dt_s': {'mean': float(dt.mean()) if dt.size else None,
                     'p95': float(np.percentile(dt, 95)) if dt.size else None, 'maximum': float(dt.max()) if dt.size else None},
            'stale_action': {'count': stale_count, 'ratio': stale_count / transitions if transitions else 0,
                             'maximum_burst': max((e['maximum_stale_action_burst'] for e in episodes), default=0),
                             'by_phase': phases, 'by_task': tasks, 'within_run_inheritance_count': cross_stale,
                             'cross_run_boundary_evidence_complete': all(b['status'] == 'PASS' for b in boundaries),
                             'executed_timeout_hover_correct': hover_correct},
            'reset': {'expected': reset_expected, 'observed': len(resets), 'passes': reset_pass,
                      'scope': 'within-run automatic resets only', 'process_restart_boundaries': boundaries,
                      'single_continuous_run': len(nonempty) == 1},
            'task_summary': tasks, 'corrupted_hdf5': corrupted, 'partial_files': partials,
            'gate_checks': gates, 'episodes': episodes,
            'runs': [{k: v for k, v in m.items() if k not in ('first_record', 'last_record')} for m in manifests]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--expected-episodes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_runs(args.directories, args.expected_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'episode_count', 'transition_count', 'success_rate',
                                           'collision_count', 'clearance_m', 'dt_s', 'gate_checks')}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
