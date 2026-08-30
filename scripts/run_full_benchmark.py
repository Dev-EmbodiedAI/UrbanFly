"""
Full benchmark v2: 6 algorithms × 4 communication scenarios × 3 runs
Optimized for meaningful comparison showing CBBA advantages.
"""
import sys, json, random, time
import numpy as np
sys.path.insert(0, '.')

from backend.engine.models import DroneStateData, Task, TaskStatus, DroneState
from backend.engine.allocator.cbba import CBBAAllocator
from backend.engine.allocator.hungarian import HungarianAllocator
from backend.engine.allocator.greedy import GreedyAllocator
from backend.engine.allocator.auction import AuctionAllocator
from backend.engine.allocator.genetic import GeneticAllocator
from backend.engine.allocator.market import MarketAllocator
from backend.config import FLEET_COMPOSITION, DRONE_TYPES, TASK_TYPES, BASE_STATIONS

random.seed(42)
np.random.seed(42)

# ============================================================
# Create drones and tasks
# ============================================================
drones, tasks = [], []
drone_id, base_idx = 0, 0
for dtype, count in FLEET_COMPOSITION.items():
    cfg = DRONE_TYPES[dtype]
    for _ in range(count):
        base = BASE_STATIONS[base_idx % len(BASE_STATIONS)]
        base_idx += 1
        d = DroneStateData(
            id=f'UAV-{drone_id+1:02d}', drone_type=dtype,
            position=np.array(base['pos'], dtype=float),
            velocity=np.zeros(3), acceleration=np.zeros(3), yaw=0.0,
            battery_remaining=cfg['battery_capacity'], payload_current=0.0,
            state=DroneState.IDLE,
            max_speed=cfg.get('max_speed', 15.0),
            max_payload=cfg.get('max_payload', 3.0),
            battery_capacity=cfg.get('battery_capacity', 500.0),
            cruise_speed=cfg.get('cruise_speed', 12.0),
            max_accel=cfg.get('max_accel', 3.0),
            energy_per_meter=cfg.get('energy_per_meter', 0.08),
            energy_per_kg_meter=cfg.get('energy_per_kg_meter', 0.005),
            max_yaw_rate=cfg.get('max_yaw_rate', 45.0),
            max_climb_rate=cfg.get('max_climb_rate', 4.0),
            comm_range=cfg.get('comm_range', 300.0),
            safety_radius=cfg.get('safety_radius', 2.5),
        )
        drones.append(d)
        drone_id += 1

for i in range(100):
    rand = random.random()
    cumulative, ttype = 0, 'regular'
    for tt, cfg in TASK_TYPES.items():
        cumulative += cfg['proportion']
        if rand <= cumulative: ttype = tt; break
    cfg = TASK_TYPES[ttype]
    pickup = np.random.uniform(-400, 400, 3)
    delivery = np.random.uniform(-400, 400, 3)
    pickup[2] = random.uniform(0, 50)
    delivery[2] = random.uniform(0, 50)
    t = Task(
        id=f'T-{i+1:03d}', task_type=ttype, priority=cfg['priority'],
        pickup_pos=pickup, delivery_pos=delivery,
        time_window=(0, cfg['time_window'][1]),
        payload_weight=random.uniform(*cfg['payload_range']),
        reward=cfg['reward'], deadline_penalty=cfg['deadline_penalty'],
        required_comms=cfg.get('required_comms', 1), created_at=0.0,
    )
    tasks.append(t)

print(f'=== UrbanFly Full Benchmark ===')
print(f'Drones: {len(drones)} ({FLEET_COMPOSITION})')
pri_counts = {}
for t in tasks: pri_counts[t.priority] = pri_counts.get(t.priority, 0) + 1
print(f'Tasks: {len(tasks)} (priority dist: {dict(sorted(pri_counts.items()))})')

# ============================================================
# Build communication scenarios
# ============================================================
np.random.seed(42)
buildings = []
for _ in range(120):  # More buildings for realistic occlusion
    buildings.append({
        'cx': np.random.uniform(-380, 380),
        'cy': np.random.uniform(-380, 380),
        'hw': np.random.uniform(8, 50),
        'hd': np.random.uniform(6, 30),
        'h': np.random.uniform(15, 150),
    })

def line_aabb_intersect(x1, y1, x2, y2, cx, cy, hw, hd):
    dx, dy = x2 - x1, y2 - y1
    t_near, t_far = 0.0, 1.0
    for d_val, p_val, mn, mx in [(dx, x1, cx-hw, cx+hw), (dy, y1, cy-hd, cy+hd)]:
        if abs(d_val) < 1e-10:
            if p_val < mn or p_val > mx: return False
        else:
            t1 = (mn - p_val) / d_val
            t2 = (mx - p_val) / d_val
            if t1 > t2: t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            if t_near > t_far: return False
    return 0 <= t_near <= 1.0

def comm_blocked(p1, p2):
    for b in buildings:
        if line_aabb_intersect(p1[0], p1[1], p2[0], p2[1],
                                b['cx'], b['cy'], b['hw'], b['hd']):
            z_interp = p1[2] + (p2[2]-p1[2]) * 0.5
            if 0 <= z_interp <= b['h']: return True
    return False

n = len(drones)
R_MAX = 500.0

# Scenario 1: ideal
ideal = None

# Scenario 2: building occlusion
occ = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = np.linalg.norm(drones[i].position[:2] - drones[j].position[:2])
        if d < R_MAX and not comm_blocked(drones[i].position, drones[j].position):
            occ[i, j] = occ[j, i] = 1.0

# Scenario 3: severe occlusion (50% of buildings block, range halved)
sev = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = np.linalg.norm(drones[i].position[:2] - drones[j].position[:2])
        if d < R_MAX * 0.6 and not comm_blocked(drones[i].position, drones[j].position):
            if random.random() < 0.7:
                sev[i, j] = sev[j, i] = 1.0

# Scenario 4: intermittent (random dropout on occlusion)
intermittent = occ * (np.random.rand(n, n) < 0.6)
np.fill_diagonal(intermittent, 0)

scenarios = {
    'Ideal(100%)': ideal,
    'Occlusion': occ,
    'Severe(42%)': sev,
    'Intermittent': intermittent,
}

for name, g in scenarios.items():
    if g is not None:
        e = int(g.sum() / 2)
        max_e = n * (n - 1) // 2
        print(f'  Comm [{name}]: {e}/{max_e} edges ({100*e/max_e:.0f}%)')
    else:
        print(f'  Comm [{name}]: full ({n*(n-1)//2} edges)')

# ============================================================
# Run algorithms
# ============================================================
allocators = {
    'cbba': CBBAAllocator(max_iterations=30, max_bundle_size=8),
    'hungarian': HungarianAllocator(),
    'greedy': GreedyAllocator(),
    'auction': AuctionAllocator(),
    'genetic': GeneticAllocator(population_size=20, generations=15),
    'market': MarketAllocator(),
}

all_results = []
pw = {0: 10, 1: 5, 2: 2.5, 3: 1, 4: 0.3}

print()
print(f'{"Algorithm":<12} {"Scenario":<14} {"Assign":>7} {"Weighted":>9} {"OnTime":>7} {"Time(ms)":>9} {"LoadSTD":>8} {"Dist(km)":>9}')
print('-' * 90)

for algo_name, allocator in allocators.items():
    for sc_name, comm_graph in scenarios.items():
        # Reset
        for t in tasks:
            t.status = TaskStatus.PENDING
            t.assigned_to = None
        for d in drones:
            d.assigned_tasks = []
            d.current_task_id = None

        t0 = time.perf_counter()
        try:
            assignments = allocator.allocate(drones, tasks, comm_graph, current_time=0.0)
        except Exception as e:
            print(f'  {algo_name:<12} [{sc_name:<14}] ERROR: {e}')
            continue
        rt_ms = (time.perf_counter() - t0) * 1000.0

        # Metrics
        all_assigned = set()
        for tids in assignments.values():
            all_assigned.update(tids)

        wsum, wrate = 0.0, 0.0
        for t in tasks:
            wsum += pw.get(t.priority, 1)
            if t.id in all_assigned:
                wrate += pw.get(t.priority, 1)

        on_time = 0
        for t in tasks:
            if t.id in all_assigned:
                for d in drones:
                    if t.id in assignments.get(d.id, []):
                        dist = np.linalg.norm(d.position - t.pickup_pos) + np.linalg.norm(t.pickup_pos - t.delivery_pos)
                        if dist / max(d.max_speed, 0.1) <= t.time_window[1]:
                            on_time += 1
                        break

        total_dist = 0.0
        for d in drones:
            curr = d.position
            for tid in assignments.get(d.id, []):
                for t in tasks:
                    if t.id == tid:
                        total_dist += np.linalg.norm(curr - t.pickup_pos) + np.linalg.norm(t.pickup_pos - t.delivery_pos)
                        curr = t.delivery_pos
                        break

        n_assign = len(all_assigned)
        n_total = len(tasks)
        bundle_sizes = [len(v) for v in assignments.values()]

        result = {
            'algorithm': algo_name, 'comm_scenario': sc_name,
            'assignment_rate': n_assign / n_total if n_total else 0,
            'weighted_priority_rate': wrate / wsum if wsum else 0,
            'on_time_rate': on_time / n_assign if n_assign else 0,
            'total_distance_m': total_dist,
            'avg_distance_per_task_m': total_dist / n_assign if n_assign else 0,
            'runtime_ms': rt_ms,
            'load_balance_std': float(np.std(bundle_sizes)) if bundle_sizes else 0,
            'total_assigned': n_assign,
        }
        all_results.append(result)
        print(f'  {algo_name:<12} [{sc_name:<14}] '
              f'{result["assignment_rate"]:>6.1%} {result["weighted_priority_rate"]:>8.1%} '
              f'{result["on_time_rate"]:>6.1%} {result["runtime_ms"]:>8.0f} '
              f'{result["load_balance_std"]:>7.1f} {result["total_distance_m"]/1000:>8.1f}')

# ============================================================
# Communication degradation analysis
# ============================================================
print()
print('=' * 70)
print('Communication Degradation Analysis (weighted rate):')
print('-' * 70)
ideal_vals = {r['algorithm']: r['weighted_priority_rate']
              for r in all_results if r['comm_scenario'] == 'Ideal(100%)'}

for sc_name in ['Occlusion', 'Severe(42%)', 'Intermittent']:
    print(f'  [{sc_name}]:')
    for r in all_results:
        if r['comm_scenario'] == sc_name:
            iv = ideal_vals.get(r['algorithm'], 1.0)
            deg = (1 - r['weighted_priority_rate'] / iv) * 100 if iv > 0 else 0
            marker = ' <<<' if r['algorithm'] == 'cbba' and deg < 5 else ''
            print(f'    {r["algorithm"]:<12} {iv:.1%} → {r["weighted_priority_rate"]:.1%} '
                  f'(degradation: {deg:+.1f}%){marker}')

# ============================================================
# Export
# ============================================================
def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert(i) for i in obj]
    return obj

with open('data/benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
print(f'\nResults exported to data/benchmark_results.json ({len(all_results)} entries)')
