"""Aggregation semantics only; synthetic fixtures never count as real episodes."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import h5py
import pytest

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('helsinki_multirun_audit', scripts / 'audit_helsinki_dataset_v1_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_run(root, index, valid_start=True):
    root.mkdir()
    task = ('building_blocked', 'street_canyon')[index % 2]
    episode_id = f'HelsinkiCentral1km_real_smoke_{index:03d}_{task}'
    path = root / f'{episode_id}.h5'
    with h5py.File(path, 'w') as handle:
        handle['timestamps/dt'] = [.1, .3]
        handle['labels/minimum_clearance'] = [4., 8.]
    record = {'episode_id': episode_id, 'initial_speed_mps': 0., 'initial_acceleration_mps2': 0.,
              'reset_evidence': {'first_policy_step_id': 0 if valid_start else 9,
                                 'action_buffer_reset': True, 'new_policy_step_observed': True,
                                 'start_position_error_m': 0., 'initial_yaw_error_degrees': 0.,
                                 'writer_flush_closed': True}}
    (root / 'collection_summary.json').write_text(json.dumps({'records': [record], 'reset_transitions': []}))
    episode = {'path': str(path), 'episode_id': episode_id, 'task_type': task, 'steps': 2,
               'success': True, 'collision': False, 'stale_action_count': 0,
               'maximum_stale_action_burst': 0, 'stale_executed_timeout_hover_correct': True,
               'integrity_status': 'PASS', 'integrity_checks': {'test': True},
               'stale_action_by_phase': {phase: {'transitions': 1 if phase != 'middle' else 0,
                                                'stale_action_count': 0} for phase in ('start', 'middle', 'end')}}
    return {'episodes': [episode], 'episode_count': 1, 'reset': {'observed': 0, 'passes': 0},
            'stale_action': {'cross_episode_inheritance_count': 0}, 'corrupted_hdf5': [], 'partial_files': []}


def test_two_runs_keep_restart_separate_from_auto_reset(tmp_path):
    roots = [tmp_path / 'a', tmp_path / 'b']
    reports = [fixture_run(root, index) for index, root in enumerate(roots)]
    with patch.object(audit, 'audit_helsinki_collection', side_effect=reports):
        result = audit.audit_runs(roots, 2)
    assert result['status'] == 'PASS'
    assert result['reset']['expected'] == 0
    assert len(result['reset']['process_restart_boundaries']) == 1
    assert not result['reset']['single_continuous_run']
    assert result['dt_s']['mean'] == pytest.approx(.2)
    assert result['clearance_m']['median'] == 6.


@pytest.mark.parametrize('case', ['duplicate', 'missing_start', 'corrupt', 'partial'])
def test_invalid_aggregate_cannot_pass(tmp_path, case):
    roots = [tmp_path / 'a', tmp_path / 'b']
    reports = [fixture_run(roots[0], 0), fixture_run(roots[1], 0 if case == 'duplicate' else 1,
                                                   valid_start=case != 'missing_start')]
    if case == 'corrupt': reports[1]['corrupted_hdf5'] = [{'path': 'bad.h5'}]
    if case == 'partial': reports[1]['partial_files'] = ['bad.h5.partial']
    with patch.object(audit, 'audit_helsinki_collection', side_effect=reports):
        result = audit.audit_runs(roots, 2)
    assert result['status'] == 'FAIL'
