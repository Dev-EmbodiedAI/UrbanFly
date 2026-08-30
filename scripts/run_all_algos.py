"""
Full algorithm benchmark: 13 algorithms × 1 scenario
"""
import sys, json, random, time, numpy as np
sys.path.insert(0, '.')

from backend.engine.models import DroneStateData, Task, TaskStatus, DroneState
from backend.engine.allocator.cbba import CBBAAllocator
from backend.engine.allocator.hungarian import HungarianAllocator
from backend.engine.allocator.greedy import GreedyAllocator
from backend.engine.allocator.auction import AuctionAllocator
from backend.engine.allocator.genetic import GeneticAllocator
from backend.engine.allocator.market import MarketAllocator
from backend.engine.allocator.pso import PSOAllocator
from backend.engine.allocator.gwo import GWOAllocator
from backend.engine.allocator.aco import ACOAllocator
from backend.engine.allocator.woa import WOAAllocator
from backend.engine.allocator.simanneal import SAAllocator
from backend.engine.allocator.de import DEAllocator
from backend.engine.allocator.gnn import GNNAllocator
from backend.config import FLEET_COMPOSITION, DRONE_TYPES, TASK_TYPES, BASE_STATIONS, PRIORITY_WEIGHTS

random.seed(42); np.random.seed(42)

drones, tasks = [], []
drone_id, base_idx = 0, 0
for dtype, count in FLEET_COMPOSITION.items():
    cfg = DRONE_TYPES[dtype]
    for _ in range(count):
        base = BASE_STATIONS[base_idx % len(BASE_STATIONS)]; base_idx += 1
        d = DroneStateData(
            id=f'UAV-{drone_id+1:02d}', drone_type=dtype,
            position=np.array(base['pos'], dtype=float),
            velocity=np.zeros(3), acceleration=np.zeros(3), yaw=0.0,
            battery_remaining=cfg['battery_capacity'], payload_current=0.0,
            state=DroneState.IDLE,
            max_speed=cfg.get('max_speed',15), max_payload=cfg.get('max_payload',3),
            battery_capacity=cfg.get('battery_capacity',500),
            cruise_speed=cfg.get('cruise_speed',12),
            energy_per_meter=cfg.get('energy_per_meter',0.08),
            energy_per_kg_meter=cfg.get('energy_per_kg_meter',0.005),
        )
        drones.append(d); drone_id += 1

for i in range(100):
    rand, cumulative, ttype = random.random(), 0, 'regular'
    for tt, cfg in TASK_TYPES.items():
        cumulative += cfg['proportion']
        if rand <= cumulative: ttype = tt; break
    cfg = TASK_TYPES[ttype]
    p = np.random.uniform(-400, 400, 3); d = np.random.uniform(-400, 400, 3)
    p[2] = random.uniform(0, 50); d[2] = random.uniform(0, 50)
    t = Task(id=f'T-{i+1:03d}', task_type=ttype, priority=cfg['priority'],
        pickup_pos=p, delivery_pos=d, time_window=(0,cfg['time_window'][1]),
        payload_weight=random.uniform(*cfg['payload_range']),
        reward=cfg['reward'], deadline_penalty=cfg['deadline_penalty'],
        required_comms=cfg.get('required_comms',1), created_at=0.0)
    tasks.append(t)

print(f'Scene: {len(drones)} drones, {len(tasks)} tasks')

allocators = {
    'CBBA': CBBAAllocator(max_iterations=30, max_bundle_size=8),
    'Hungarian': HungarianAllocator(), 'Greedy': GreedyAllocator(),
    'Auction': AuctionAllocator(), 'Genetic': GeneticAllocator(population_size=20,generations=15),
    'Market': MarketAllocator(),
    'PSO': PSOAllocator(n_particles=40,n_iterations=60),
    'GWO': GWOAllocator(n_wolves=40,n_iterations=60),
    'ACO': ACOAllocator(n_ants=30,n_iterations=60),
    'WOA': WOAAllocator(n_whales=40,n_iterations=60),
    'SA': SAAllocator(T_init=1000,T_min=0.01,alpha=0.95,steps_per_T=40),
    'DE': DEAllocator(pop_size=40,n_iterations=60),
    'GNN': GNNAllocator(n_layers=5,embedding_dim=32),
}

pw = PRIORITY_WEIGHTS
all_results = []

print(f'{"Algorithm":<14} {"Assign":>7} {"Weighted":>9} {"Runtime":>9} {"LoadSTD":>8} {"Dist(km)":>9}')
print('-'*65)

for algo_name, allocator in allocators.items():
    for t in tasks: t.status = TaskStatus.PENDING; t.assigned_to = None
    for d in drones: d.assigned_tasks = []; d.current_task_id = None

    t0 = time.perf_counter()
    try:
        assignments = allocator.allocate(drones, tasks, None, current_time=0.0)
    except Exception as e:
        print(f'  {algo_name:<14} ERROR: {e}'); continue
    rt_ms = (time.perf_counter() - t0) * 1000.0

    all_assigned = set()
    for tids in assignments.values(): all_assigned.update(tids)

    wsum, wrate = 0.0, 0.0
    for t in tasks:
        w = pw.get(t.priority, 1.0); wsum += w
        if t.id in all_assigned: wrate += w

    on_time = 0
    for t in tasks:
        if t.id in all_assigned:
            for d in drones:
                if t.id in assignments.get(d.id,[]):
                    dist = np.linalg.norm(d.position-t.pickup_pos) + np.linalg.norm(t.pickup_pos-t.delivery_pos)
                    if dist/max(d.max_speed,0.1) <= t.time_window[1]: on_time += 1
                    break

    total_dist = 0.0
    for d in drones:
        curr = d.position
        for tid in assignments.get(d.id,[]):
            for t in tasks:
                if t.id == tid:
                    total_dist += np.linalg.norm(curr-t.pickup_pos) + np.linalg.norm(t.pickup_pos-t.delivery_pos)
                    curr = t.delivery_pos; break

    n_assign = len(all_assigned)
    bs = [len(v) for v in assignments.values()]
    r = {
        'algorithm': algo_name,
        'assignment_rate': n_assign/100,
        'weighted_priority_rate': wrate/wsum if wsum else 0,
        'on_time_rate': on_time/n_assign if n_assign else 0,
        'total_distance_m': total_dist, 'runtime_ms': rt_ms,
        'load_balance_std': float(np.std(bs)) if bs else 0,
        'total_assigned': n_assign,
    }
    all_results.append(r)
    print(f'  {algo_name:<14} {r["assignment_rate"]:>6.1%} {r["weighted_priority_rate"]:>8.1%} {r["runtime_ms"]:>8.0f}ms {r["load_balance_std"]:>7.1f} {r["total_distance_m"]/1000:>8.1f}')

def convert(obj):
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,dict): return {str(k):convert(v) for k,v in obj.items()}
    if isinstance(obj,list): return [convert(i) for i in obj]
    return obj

with open('data/all_algorithms.json','w',encoding='utf-8') as f:
    json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
print(f'\nExported {len(all_results)} results')
