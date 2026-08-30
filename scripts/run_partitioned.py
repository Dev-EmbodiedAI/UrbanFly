"""
Partitioned communication benchmark.
Shows CBBA's advantage when communication is genuinely restricted.
"""
import sys, json, random, time, numpy as np
sys.path.insert(0, '.')

from backend.engine.models import DroneStateData, Task, TaskStatus, DroneState
from backend.engine.allocator.cbba import CBBAAllocator
from backend.engine.allocator.hungarian import HungarianAllocator
from backend.engine.allocator.greedy import GreedyAllocator
from backend.engine.allocator.auction import AuctionAllocator
from backend.engine.allocator.genetic import GeneticAllocator
from backend.engine.allocator.pso import PSOAllocator
from backend.engine.allocator.gwo import GWOAllocator
from backend.engine.allocator.woa import WOAAllocator
from backend.engine.allocator.simanneal import SAAllocator
from backend.engine.allocator.de import DEAllocator
from backend.engine.allocator.aco import ACOAllocator
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
        )
        drones.append(d); drone_id += 1

for i in range(100):
    rand, cumul, ttype = random.random(), 0, 'regular'
    for tt, cfg in TASK_TYPES.items():
        cumul += cfg['proportion']
        if rand <= cumul: ttype = tt; break
    cfg = TASK_TYPES[ttype]
    p = np.random.uniform(-400, 400, 3); d = np.random.uniform(-400, 400, 3)
    p[2] = random.uniform(0, 50); d[2] = random.uniform(0, 50)
    tasks.append(Task(id=f'T-{i+1:03d}', task_type=ttype, priority=cfg['priority'],
        pickup_pos=p, delivery_pos=d, time_window=(0,cfg['time_window'][1]),
        payload_weight=random.uniform(*cfg['payload_range']),
        reward=cfg['reward'], deadline_penalty=cfg['deadline_penalty'],
        required_comms=cfg.get('required_comms',1), created_at=0.0))

n = len(drones)
pw = PRIORITY_WEIGHTS

# ============================================================
# Communication scenarios with TRUE partitioning
# ============================================================
# Ideal: full
ideal = None

# Partitioned: 5 groups of 6 drones, intra-group full, inter-group zero
partitioned = np.zeros((n, n))
for g in range(5):
    for i in range(g*6, (g+1)*6):
        for j in range(g*6, (g+1)*6):
            if i != j:
                partitioned[i, j] = 1.0

# Isolated: zero communication
isolated = np.zeros((n, n))

scenarios = {
    'Ideal(全连通)': ideal,
    'Partitioned(5组x6机)': partitioned,
    'Isolated(完全隔离)': isolated,
}
for nm, g in scenarios.items():
    e = int(g.sum()/2) if g is not None else n*(n-1)//2
    print(f'  [{nm}]: {e}/{n*(n-1)//2} edges')

# ============================================================
# Find components
# ============================================================
def find_components(adj):
    if adj is None: return [list(range(n))]
    visited = set(); comps = []
    for start in range(n):
        if start not in visited:
            comp = []; stack = [start]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v); comp.append(v)
                    for u in range(n):
                        if adj[v,u] > 0 and u not in visited:
                            stack.append(u)
            comps.append(comp)
    return comps

# ============================================================
# Evaluate
# ============================================================
def eval_metrics(assignments, drones, tasks):
    all_assigned = set()
    for tids in assignments.values(): all_assigned.update(tids)
    wsum, wrate = 0.0, 0.0
    for t in tasks:
        w = pw.get(t.priority, 1.0); wsum += w
        if t.id in all_assigned: wrate += w
    total_dist = 0.0
    for d in drones:
        curr = d.position
        for tid in assignments.get(d.id, []):
            for t in tasks:
                if t.id == tid:
                    total_dist += np.linalg.norm(curr-t.pickup_pos) + np.linalg.norm(t.pickup_pos-t.delivery_pos)
                    curr = t.delivery_pos; break
    bs = [len(v) for v in assignments.values()]
    return {
        'assignment_rate': len(all_assigned)/len(tasks),
        'weighted_priority_rate': wrate/wsum if wsum else 0,
        'total_distance_m': total_dist,
        'load_balance_std': float(np.std(bs)) if bs else 0,
    }

# ============================================================
# Run
# ============================================================
all_results = []
print(f'\n{"Algorithm":<14} {"Scenario":<22} {"Type":<8} {"Assign":>7} {"Weighted":>9} {"Runtime":>9} {"LoadSTD":>8}')
print('-'*90)

for sc_name, comm_graph in scenarios.items():
    components = find_components(comm_graph)
    max_comp_size = max(len(c) for c in components)
    n_comps = len(components)

    for algo_type, algo_name, allocator in [
        ('分布式', 'CBBA', CBBAAllocator(max_iterations=30, max_bundle_size=8)),
        ('集中式', 'Hungarian', HungarianAllocator()),
        ('集中式', 'Genetic', GeneticAllocator(population_size=20,generations=15)),
        ('分布式', 'Greedy', GreedyAllocator()),
        ('分布式', 'Auction', AuctionAllocator()),
        ('元启发', 'WOA', WOAAllocator(n_whales=40,n_iterations=60)),
        ('元启发', 'GWO', GWOAllocator(n_wolves=40,n_iterations=60)),
        ('元启发', 'PSO', PSOAllocator(n_particles=40,n_iterations=60)),
        ('元启发', 'DE', DEAllocator(pop_size=40,n_iterations=60)),
    ]:
        for t in tasks: t.status=TaskStatus.PENDING; t.assigned_to=None
        for d in drones: d.assigned_tasks=[]; d.current_task_id=None

        t0 = time.perf_counter()

        if algo_type == '集中式' and comm_graph is not None and n_comps > 1:
            # 集中式算法只能在最大连通分量内运作
            largest_comp = max(components, key=len)
            connected_drones = [drones[i] for i in largest_comp]
            connected_tasks = [t for t in tasks
                if any(t.payload_weight <= d.max_payload for d in connected_drones)]

            try:
                sub = allocator.allocate(connected_drones, connected_tasks, None, 0.0)
            except:
                sub = {d.id: [] for d in connected_drones}

            full = {d.id: [] for d in drones}
            for d_id, t_ids in sub.items():
                full[d_id] = t_ids
        else:
            try:
                full = allocator.allocate(drones, tasks, comm_graph, 0.0)
            except:
                full = {d.id: [] for d in drones}

        rt_ms = (time.perf_counter() - t0) * 1000.0
        r = eval_metrics(full, drones, tasks)
        r['algorithm'] = algo_name; r['comm_scenario'] = sc_name
        r['algo_type'] = algo_type; r['runtime_ms'] = rt_ms
        r['max_component'] = max_comp_size
        all_results.append(r)
        wr = r['weighted_priority_rate']
        ar = r['assignment_rate']
        alarm = '!!!' if (algo_type == '集中式' and ar < 0.5 and ('Partitioned' in sc_name or 'Isolated' in sc_name)) else ''
        print(f'  {algo_name:<14} [{sc_name:<22}] {algo_type:<8} {ar:>6.1%} {wr:>8.1%} {rt_ms:>8.0f}ms {r["load_balance_std"]:>7.1f} {alarm}')

def cvt(obj):
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,dict): return {str(k):cvt(v) for k,v in obj.items()}
    if isinstance(obj,list): return [cvt(i) for i in obj]
    return obj

with open('data/partitioned_results.json','w',encoding='utf-8') as f:
    json.dump(cvt(all_results), f, indent=2, ensure_ascii=False)
print(f'\nExported {len(all_results)} results')
