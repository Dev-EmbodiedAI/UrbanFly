"""Replacement selection must be explicit, route-identical, and zero-stale."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import h5py

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    'helsinki_replacement_audit', scripts / 'audit_helsinki_dataset_v1_replacements.py'
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_run(root, index, *, route_offset=0.0, stale=0):
    root.mkdir()
    task = ('building_blocked', 'street_canyon')[index % 2]
    episode_id = f'HelsinkiCentral1km_real_smoke_{index:03d}_{task}'
    path = root / f'{episode_id}.h5'
    route = [[float(index), 0.0, 5.0], [float(index) + 1.0 + route_offset, 0.0, 5.0]]
    metadata = {
        'episode_id': episode_id,
        'task_type': task,
        'start_world': route[0],
        'global_goal_world': route[-1],
        'global_route_world': route,
        'global_route_backend': route,
    }
    with h5py.File(path, 'w') as handle:
        handle['timestamps/dt'] = [.1, .1]
        handle['labels/minimum_clearance'] = [4., 8.]
        handle.attrs['metadata_json'] = json.dumps(metadata)
    record = {
        'episode_id': episode_id,
        'initial_speed_mps': 0.,
        'initial_acceleration_mps2': 0.,
        'reset_evidence': {
            'first_policy_step_id': 0,
            'action_buffer_reset': True,
            'new_policy_step_observed': True,
            'start_position_error_m': 0.,
            'initial_yaw_error_degrees': 0.,
            'writer_flush_closed': True,
        },
    }
    (root / 'collection_summary.json').write_text(json.dumps({
        'records': [record], 'reset_transitions': [], 'collector_process_ids': [index + 10]
    }))
    episode = {
        'path': str(path),
        'episode_id': episode_id,
        'task_type': task,
        'steps': 2,
        'success': True,
        'collision': False,
        'stale_action_count': stale,
        'maximum_stale_action_burst': stale,
        'stale_executed_timeout_hover_correct': True,
        'integrity_status': 'PASS',
        'integrity_checks': {'test': True},
        'stale_action_by_phase': {
            phase: {'transitions': 1 if phase != 'middle' else 0, 'stale_action_count': 0}
            for phase in ('start', 'middle', 'end')
        },
    }
    report = {
        'episodes': [episode],
        'episode_count': 1,
        'corrupted_hdf5': [],
        'partial_files': [],
    }
    return report


def test_explicit_replacement_overrides_stale_base_and_preserves_route(tmp_path):
    base0, base1, replacement1 = tmp_path / 'base0', tmp_path / 'base1', tmp_path / 'replacement1'
    reports = [
        fixture_run(base0, 0),
        fixture_run(base1, 1, stale=1),
        fixture_run(replacement1, 1),
    ]
    with patch.object(audit, 'audit_helsinki_collection', side_effect=reports):
        result = audit.audit_runs([base0, base1], 2, [replacement1])
    assert result['status'] == 'PASS'
    assert result['replacement_selection']['replacement_count'] == 1
    assert result['replacement_selection']['route_checks'][0]['status'] == 'PASS'
    assert result['stale_action']['count'] == 0
    assert [audit.episode_index(episode) for episode in result['episodes']] == [0, 1]


def test_route_mismatch_cannot_pass(tmp_path):
    base0, base1, replacement1 = tmp_path / 'base0', tmp_path / 'base1', tmp_path / 'replacement1'
    reports = [
        fixture_run(base0, 0),
        fixture_run(base1, 1),
        fixture_run(replacement1, 1, route_offset=3.0),
    ]
    with patch.object(audit, 'audit_helsinki_collection', side_effect=reports):
        result = audit.audit_runs([base0, base1], 2, [replacement1])
    assert result['status'] == 'FAIL'
    assert not result['gate_checks']['replacement_routes_match_original']
