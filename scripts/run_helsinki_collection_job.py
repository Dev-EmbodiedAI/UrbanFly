"""Durable local collector job. No retries, no mock fallback, no core changes."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import aiohttp

ROOT = Path(__file__).resolve().parents[1]


def health():
    with urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=5) as response:
        return json.load(response)


async def control(action, **values):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
        async with session.ws_connect('http://127.0.0.1:8765/ws') as socket:
            await socket.send_json({'type': 'control', 'payload': {'action': action, **values}})
            await socket.send_json({'type': 'control', 'payload': {'action': 'get_status'}})
            async for message in socket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    if data.get('type') == 'sim_state':
                        return data['payload']
            raise RuntimeError('No simulator acknowledgement')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task-manifest-in', type=Path, required=True)
    parser.add_argument('--episode-index-offset', type=int, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # Never reuse a previous run or let the atomic writer overwrite episodes.
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'PREFLIGHT', 'supervisor_pid': os.getpid(),
              'offset': args.episode_index_offset, 'requested_episodes': args.episodes,
              'started_unix_s': time.time(), 'output_dir': str(output)}

    def save():
        temporary = output / 'job_status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        os.replace(temporary, output / 'job_status.json')

    save()
    owns_run = False
    try:
        initial = health()
        surfaces = [s for s in initial['surfaces'] if s['age_s'] < 5]
        if initial['clients']['policy'] != 0 or len(surfaces) != 1 or not surfaces[0]['scene_ready']:
            raise RuntimeError('Requires exactly one ready sensor surface and no existing policy')
        if initial['simulator']['state'] != 'stopped':
            raise RuntimeError('Refusing to replace an already-running simulator task')
        state = asyncio.run(control('set_speed', value=1.0))
        if state.get('speed_multiplier') != 1.0:
            raise RuntimeError('Simulation speed 1x acknowledgement failed')
        status['health_before'] = initial
        status['frozen_source_sha256'] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ['backend/engine/planner.py', 'backend/engine/simulator.py',
                         'backend/engine/multirotor_dynamics.py', 'backend/engine/triangle_geometry.py',
                         'backend/engine/helsinki_urban_sampling.py', 'backend/engine/local_goal.py',
                         'backend/engine/helsinki_navigation.py']
        }
        command = [sys.executable, '-u', str(ROOT / 'uav_wm_navigation/scripts/collect_helsinki_dataset_v1.py'),
                   '--output-dir', str(output), '--task-manifest-in', str(args.task_manifest_in.resolve()),
                   '--episode-index-offset', str(args.episode_index_offset), '--episodes', str(args.episodes)]
        status['command'] = command
        owns_run = True
        with (output / 'collector.stdout.log').open('w', encoding='utf-8') as stdout, \
                (output / 'collector.stderr.log').open('w', encoding='utf-8') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            status.update(status='RUNNING', collector_pid=child.pid)
            save()
            code = child.wait()
        status.update(status='PASS' if code == 0 else 'FAIL', collector_exit_code=code)
    except Exception as error:
        status.update(status='FAIL', error=repr(error))
    finally:
        if owns_run:
            try:
                current = health()
                if current['clients']['policy'] != 0:
                    raise RuntimeError('Policy still active; refusing cleanup of another owner')
                asyncio.run(control('stop'))
                status['health_after'] = health()
            except Exception as error:
                status.update(status='FAIL', cleanup_error=repr(error))
        status['completed_hdf5'] = len(list(output.glob('*.h5')))
        status['finished_unix_s'] = time.time()
        save()
    return 0 if status['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
