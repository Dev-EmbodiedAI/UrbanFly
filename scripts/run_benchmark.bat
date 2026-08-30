@echo off
REM 算法对比基准测试
cd /d %~dp0..

echo Running algorithm benchmark...
python -c "
import sys, json
sys.path.insert(0, '.')
from backend.engine.models import DroneStateData, Task, TaskStatus
from backend.engine.allocator.benchmark import AlgorithmBenchmark
from backend.config import FLEET_COMPOSITION, DRONE_TYPES, TASK_GENERATION, TASK_TYPES, BASE_STATIONS
import numpy as np
import random

# Create 30 dummy drones
drones = []
drone_id = 0
base_idx = 0
for dtype, count in FLEET_COMPOSITION.items():
    cfg = DRONE_TYPES[dtype]
    for _ in range(count):
        base = BASE_STATIONS[base_idx % len(BASE_STATIONS)]
        base_idx += 1
        d = DroneStateData(
            id=f'UAV-{drone_id+1:02d}',
            drone_type=dtype,
            position=np.array(base['pos'], dtype=float),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=0.0,
            battery_remaining=cfg['battery_capacity'],
            payload_current=0.0,
            **{k:v for k,v in cfg.items() if k not in ('label','color')}
        )
        drones.append(d)
        drone_id += 1

# Create 100 dummy tasks
tasks = []
for i in range(100):
    rand = random.random()
    cumulative = 0
    ttype = 'regular'
    for tt, cfg in TASK_TYPES.items():
        cumulative += cfg['proportion']
        if rand <= cumulative:
            ttype = tt
            break
    cfg = TASK_TYPES[ttype]
    t = Task(
        id=f'T-{i+1:03d}',
        task_type=ttype,
        priority=cfg['priority'],
        pickup_pos=np.random.uniform(-400, 400, 3),
        delivery_pos=np.random.uniform(-400, 400, 3),
        time_window=(0, cfg['time_window'][1]),
        payload_weight=random.uniform(*cfg['payload_range']),
        reward=cfg['reward'],
        deadline_penalty=cfg['deadline_penalty'],
        required_comms=cfg['required_comms'],
        created_at=0.0,
    )
    tasks.append(t)

print(f'Benchmark: {len(drones)} drones, {len(tasks)} tasks')
print()

# Run benchmark
bench = AlgorithmBenchmark(drones, tasks)
results = bench.run_all(runs=3)
bench.print_report(results)

# Export
bench.export_to_json(results, 'data/benchmark_results.json')
print()
print('Results saved to data/benchmark_results.json')
"
pause
