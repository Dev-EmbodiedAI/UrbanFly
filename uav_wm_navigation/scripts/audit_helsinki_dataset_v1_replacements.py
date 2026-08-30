#!/usr/bin/env python3
"""Read-only Helsinki multi-run QA with explicit, auditable replacements."""
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


def episode_index(item):
    match = re.search(r'_smoke_(\d+)_', item['episode_id'])
    return int(match.group(1)) if match else -1


def _summary_path(root):
    return next((root / name for name in (
        'collection_summary.json', 'collection_failure.json', 'collection_progress.json'
    ) if (root / name).exists()), None)


def _load_metadata(path):
    with h5py.File(path, 'r') as handle:
        raw = handle.attrs['metadata_json']
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return json.loads(raw)


def _route_signature(path):
    metadata = _load_metadata(path)
    fields = ('episode_id', 'task_type', 'start_world', 'global_goal_world',
              'global_route_world', 'global_route_backend')
    return {field: metadata.get(field) for field in fields}


def _fresh_start_boundary(previous, current):
    first = current['record'] or {}
    last = previous['record'] or {}
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
    return {
        'from_episode': previous['episode']['episode_id'],
        'to_episode': current['episode']['episode_id'],
        'kind': 'process_or_replacement_boundary_not_continuous_auto_reset',
        'checks': checks,
        'status': 'PASS' if all(checks.values()) else 'LIMITATION',
    }


def audit_runs(roots, expected_episodes, replacement_roots=()):
    roots = [Path(root).resolve() for root in roots]
    replacement_roots = [Path(root).resolve() for root in replacement_roots]
    if set(roots) & set(replacement_roots):
        raise ValueError('A directory cannot be both a base and replacement directory')

    run_data = []
    all_corrupted, all_partials = [], []
    for role, run_roots in (('base', roots), ('replacement', replacement_roots)):
        for root in run_roots:
            summary_path = _summary_path(root)
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path else {}
            reset_transitions = summary.get('reset_transitions', [])
            report = audit_helsinki_collection(
                root,
                expected_episodes=len(list(root.glob('*.h5'))),
                reset_transitions=reset_transitions,
            )
            records = {record.get('episode_id'): record for record in summary.get('records', [])}
            transitions = {
                (transition.get('from_episode'), transition.get('to_episode')): transition
                for transition in reset_transitions
            }
            run = {
                'role': role,
                'root': root,
                'summary_path': summary_path,
                'summary': summary,
                'report': report,
                'records': records,
                'transitions': transitions,
            }
            run_data.append(run)
            all_corrupted.extend(report['corrupted_hdf5'])
            all_partials.extend(report['partial_files'])

    base_entries, replacement_entries = {}, {}
    for run in run_data:
        target = base_entries if run['role'] == 'base' else replacement_entries
        for episode in run['report']['episodes']:
            entry = {
                'episode': episode,
                'run': run,
                'record': run['records'].get(episode['episode_id']),
            }
            target.setdefault(episode_index(episode), []).append(entry)

    base_duplicates = {index: entries for index, entries in base_entries.items() if len(entries) != 1}
    replacement_duplicates = {
        index: entries for index, entries in replacement_entries.items() if len(entries) != 1
    }
    unexpected_replacements = sorted(set(replacement_entries) - set(base_entries))
    selected_by_index = {
        index: entries[0] for index, entries in base_entries.items() if entries
    }
    replacements = []
    route_checks = []
    for index, entries in sorted(replacement_entries.items()):
        if len(entries) != 1 or index not in base_entries or not base_entries[index]:
            continue
        old_entry = base_entries[index][0]
        new_entry = entries[0]
        old_signature = _route_signature(old_entry['episode']['path'])
        new_signature = _route_signature(new_entry['episode']['path'])
        match = old_signature == new_signature
        route_checks.append({
            'episode_index': index,
            'status': 'PASS' if match else 'FAIL',
            'old_path': old_entry['episode']['path'],
            'replacement_path': new_entry['episode']['path'],
            'matching_fields': [
                field for field in old_signature if old_signature[field] == new_signature[field]
            ],
        })
        selected_by_index[index] = new_entry
        replacements.append({
            'episode_index': index,
            'episode_id': new_entry['episode']['episode_id'],
            'old_path': old_entry['episode']['path'],
            'replacement_path': new_entry['episode']['path'],
            'route_signature_status': 'PASS' if match else 'FAIL',
        })

    selected = [selected_by_index[index] for index in sorted(selected_by_index)]
    episodes = [entry['episode'] for entry in selected]
    indices = [episode_index(episode) for episode in episodes]

    all_dt, all_clearance = [], []
    for entry in selected:
        with h5py.File(entry['episode']['path'], 'r') as handle:
            all_dt.append(handle['timestamps/dt'][:])
            all_clearance.append(handle['labels/minimum_clearance'][:])
    dt = np.concatenate(all_dt) if all_dt else np.empty(0)
    clearance = np.concatenate(all_clearance) if all_clearance else np.empty(0)

    transitions = sum(episode['steps'] for episode in episodes)
    stale_count = sum(episode['stale_action_count'] for episode in episodes)
    success_count = sum(episode['success'] for episode in episodes)
    collision_count = sum(episode['collision'] for episode in episodes)
    tasks = {}
    for task in ('building_blocked', 'street_canyon', 'rooftop_to_ground',
                 'ground_to_rooftop', 'rooftop_to_rooftop'):
        task_episodes = [episode for episode in episodes if episode['task_type'] == task]
        steps = sum(episode['steps'] for episode in task_episodes)
        stale = sum(episode['stale_action_count'] for episode in task_episodes)
        tasks[task] = {
            'episodes': len(task_episodes),
            'successes': sum(episode['success'] for episode in task_episodes),
            'collisions': sum(episode['collision'] for episode in task_episodes),
            'transitions': steps,
            'stale_action_count': stale,
            'stale_action_ratio': stale / steps if steps else 0,
            'maximum_stale_action_burst': max(
                (episode['maximum_stale_action_burst'] for episode in task_episodes), default=0
            ),
        }
    phases = {}
    for phase in ('start', 'middle', 'end'):
        steps = sum(episode['stale_action_by_phase'][phase]['transitions'] for episode in episodes)
        stale = sum(episode['stale_action_by_phase'][phase]['stale_action_count'] for episode in episodes)
        phases[phase] = {
            'transitions': steps,
            'stale_action_count': stale,
            'stale_action_ratio': stale / steps if steps else 0,
        }

    automatic_boundaries, fresh_boundaries = [], []
    for previous, current in zip(selected, selected[1:]):
        pair = (previous['episode']['episode_id'], current['episode']['episode_id'])
        if previous['run'] is current['run'] and pair in previous['run']['transitions']:
            transition = previous['run']['transitions'][pair]
            checks = transition.get('checks', {})
            passed = transition.get('automatic_reset') == 'PASS' and all(checks.values())
            automatic_boundaries.append({
                'from_episode': pair[0],
                'to_episode': pair[1],
                'kind': 'within_run_automatic_reset',
                'checks': checks,
                'status': 'PASS' if passed else 'FAIL',
            })
        else:
            fresh_boundaries.append(_fresh_start_boundary(previous, current))

    maximum_burst = max((episode['maximum_stale_action_burst'] for episode in episodes), default=0)
    hover_correct = all(episode['stale_executed_timeout_hover_correct'] for episode in episodes)
    all_boundaries = automatic_boundaries + fresh_boundaries
    gates = {
        'episode_count': len(episodes) == expected_episodes,
        'unique_contiguous_episode_ids': (
            not base_duplicates and not replacement_duplicates and not unexpected_replacements
            and indices == list(range(expected_episodes))
        ),
        'explicit_replacements_unique': (
            len(replacements) == len(replacement_entries) and not replacement_duplicates
            and not unexpected_replacements
        ),
        'replacement_routes_match_original': all(check['status'] == 'PASS' for check in route_checks),
        'balanced_task_counts': (
            max(task['episodes'] for task in tasks.values())
            - min(task['episodes'] for task in tasks.values()) <= 1
        ),
        'success_rate_at_least_98_percent': bool(episodes) and success_count / len(episodes) >= .98,
        'collision_count_zero': collision_count == 0,
        'minimum_clearance_at_least_2p5m': bool(clearance.size) and float(clearance.min()) >= 2.5,
        'all_99_episode_boundaries_accounted': len(all_boundaries) == max(0, expected_episodes - 1),
        'within_run_reset_count_and_success': all(
            boundary['status'] == 'PASS' for boundary in automatic_boundaries
        ),
        'cross_run_fresh_start_evidence': all(
            boundary['status'] == 'PASS' for boundary in fresh_boundaries
        ),
        'corrupted_hdf5_zero': not all_corrupted,
        'partial_count_zero': not all_partials,
        'stale_action_zero': stale_count == 0 and maximum_burst == 0,
        'cross_episode_stale_action_zero': all(
            boundary['checks'].get('stale_action_inheritance_absent', True)
            for boundary in automatic_boundaries
        ) and all(boundary['status'] == 'PASS' for boundary in fresh_boundaries),
        'all_stale_executed_actions_factual_hover': hover_correct,
        'all_dataset_integrity_checks': all(
            episode['integrity_status'] == 'PASS' and all(episode['integrity_checks'].values())
            for episode in episodes
        ),
    }
    return {
        'schema': 'urbanfly-helsinki-replacement-aware-multirun-qa-v2',
        'status': 'PASS' if all(gates.values()) else 'FAIL',
        'gate_scope': 'Selected Dataset episodes with explicit replacement precedence and factual boundary accounting',
        'expected_episodes': expected_episodes,
        'episode_count': len(episodes),
        'transition_count': transitions,
        'success_count': success_count,
        'success_rate': success_count / len(episodes) if episodes else 0,
        'collision_count': collision_count,
        'collision_rate': collision_count / len(episodes) if episodes else 0,
        'clearance_m': {
            'minimum': float(clearance.min()) if clearance.size else None,
            'median': float(np.median(clearance)) if clearance.size else None,
        },
        'dt_s': {
            'mean': float(dt.mean()) if dt.size else None,
            'p95': float(np.percentile(dt, 95)) if dt.size else None,
            'maximum': float(dt.max()) if dt.size else None,
        },
        'stale_action': {
            'count': stale_count,
            'ratio': stale_count / transitions if transitions else 0,
            'maximum_burst': maximum_burst,
            'by_phase': phases,
            'by_task': tasks,
            'executed_timeout_hover_correct': hover_correct,
        },
        'reset': {
            'expected_total_episode_boundaries': max(0, expected_episodes - 1),
            'within_run_automatic_resets': automatic_boundaries,
            'process_or_replacement_boundaries': fresh_boundaries,
            'automatic_reset_passes': sum(
                boundary['status'] == 'PASS' for boundary in automatic_boundaries
            ),
            'fresh_boundary_passes': sum(
                boundary['status'] == 'PASS' for boundary in fresh_boundaries
            ),
        },
        'replacement_selection': {
            'replacement_count': len(replacements),
            'replacements': replacements,
            'route_checks': route_checks,
            'base_duplicate_indices': sorted(base_duplicates),
            'replacement_duplicate_indices': sorted(replacement_duplicates),
            'unexpected_replacement_indices': unexpected_replacements,
        },
        'task_summary': tasks,
        'corrupted_hdf5': all_corrupted,
        'partial_files': all_partials,
        'gate_checks': gates,
        'episodes': episodes,
        'runs': [{
            'directory': str(run['root']),
            'role': run['role'],
            'summary': str(run['summary_path']) if run['summary_path'] else None,
            'episode_count': run['report']['episode_count'],
            'collector_process_ids': run['summary'].get('collector_process_ids', []),
        } for run in run_data],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+', help='Base collection directories')
    parser.add_argument('--replacement-directory', type=Path, action='append', default=[])
    parser.add_argument('--expected-episodes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_runs(args.directories, args.expected_episodes, args.replacement_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'status', 'episode_count', 'transition_count', 'success_rate', 'collision_count',
        'clearance_m', 'dt_s', 'stale_action', 'replacement_selection', 'gate_checks'
    )}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
