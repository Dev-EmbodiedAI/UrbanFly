"""
Ablation + Full Comparison Benchmark
====================================
CBBA消融实验 + 10种算法对比 + 多通信场景
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
from backend.engine.allocator.pso import PSOAllocator
from backend.engine.allocator.gwo import GWOAllocator
from backend.engine.allocator.aco import ACOAllocator
from backend.config import FLEET_COMPOSITION, DRONE_TYPES, TASK_TYPES, BASE_STATIONS, PRIORITY_WEIGHTS

random.seed(42)
np.random.seed(42)

# ============================================================
# Scene setup
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
            energy_per_meter=cfg.get('energy_per_meter', 0.08),
            energy_per_kg_meter=cfg.get('energy_per_kg_meter', 0.005),
        )
        drones.append(d)
        drone_id += 1

for i in range(100):
    rand, cumulative, ttype = random.random(), 0, 'regular'
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

n = len(drones)
print(f'Scene: {n} drones, {len(tasks)} tasks')

# ============================================================
# Communication scenarios
# ============================================================
np.random.seed(42)
buildings = []
for _ in range(120):
    buildings.append({
        'cx': np.random.uniform(-380, 380), 'cy': np.random.uniform(-380, 380),
        'hw': np.random.uniform(8, 50), 'hd': np.random.uniform(6, 30),
        'h': np.random.uniform(15, 150),
    })

def line_aabb(x1,y1,x2,y2,cx,cy,hw,hd):
    dx,dy=x2-x1,y2-y1; tn,tf=0.0,1.0
    for dv,pv,mn,mx in [(dx,x1,cx-hw,cx+hw),(dy,y1,cy-hd,cy+hd)]:
        if abs(dv)<1e-10:
            if pv<mn or pv>mx: return False
        else:
            t1=(mn-pv)/dv; t2=(mx-pv)/dv
            if t1>t2: t1,t2=t2,t1
            tn=max(tn,t1); tf=min(tf,t2)
            if tn>tf: return False
    return 0<=tn<=1.0

def blocked(p1,p2):
    for b in buildings:
        if line_aabb(p1[0],p1[1],p2[0],p2[1],b['cx'],b['cy'],b['hw'],b['hd']):
            if 0<=(p1[2]+p2[2])/2<=b['h']: return True
    return False

occ=np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        if np.linalg.norm(drones[i].position[:2]-drones[j].position[:2])<500 and not blocked(drones[i].position,drones[j].position):
            occ[i,j]=occ[j,i]=1.0

sev=np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        if np.linalg.norm(drones[i].position[:2]-drones[j].position[:2])<350 and not blocked(drones[i].position,drones[j].position):
            if random.random()<0.6: sev[i,j]=sev[j,i]=1.0

intermittent=occ*(np.random.rand(n,n)<0.5)
np.fill_diagonal(intermittent,0)

scenarios={'Ideal(100%)':None,'Occlusion(69%)':occ,'Severe(35%)':sev,'Intermittent':intermittent}
for nm,g in scenarios.items():
    print(f'  [{nm}]: {int(g.sum()/2) if g is not None else n*(n-1)//2} edges')

# ============================================================
# Allocators: CBBA ablation + comparison
# ============================================================
class CBBA_NoPriority(CBBAAllocator):
    """消融: 去除优先级感知 (统一权重)"""
    def __init__(self):
        super().__init__(max_iterations=30, max_bundle_size=8)
        self.priority_weights = {0:1,1:1,2:1,3:1,4:1}
        self.name = "CBBA-NoPri"

class CBBA_NoComm(CBBAAllocator):
    """消融: 去除通信感知 (全连通共识)"""
    def __init__(self):
        super().__init__(max_iterations=30, max_bundle_size=8)
        self.name = "CBBA-NoComm"

class CBBA_NoBattery(CBBAAllocator):
    """消融: 去除电池约束"""
    def __init__(self):
        super().__init__(max_iterations=30, max_bundle_size=8)
        self.name = "CBBA-NoBat"
        self.battery_safety_margin = 0.0  # 无安全余量

allocators = {
    # CBBA variants
    'CBBA-Full': CBBAAllocator(max_iterations=30, max_bundle_size=8),
    'CBBA-NoPriority': CBBA_NoPriority(),
    'CBBA-NoComm': CBBA_NoComm(),
    'CBBA-NoBattery': CBBA_NoBattery(),
    # Baseline
    'Hungarian': HungarianAllocator(),
    'Greedy': GreedyAllocator(),
    'Auction': AuctionAllocator(),
    'Genetic': GeneticAllocator(population_size=20, generations=15),
    'Market': MarketAllocator(),
    # Meta-heuristic
    'PSO': PSOAllocator(n_particles=40, n_iterations=60),
    'GWO': GWOAllocator(n_wolves=40, n_iterations=60),
    'ACO': ACOAllocator(n_ants=30, n_iterations=60),
}

# ============================================================
# Priority weights for metrics
# ============================================================
pw = PRIORITY_WEIGHTS

# ============================================================
# Run all
# ============================================================
all_results = []
priority_breakdown = {}  # {algo: {priority: (assigned, total)}}

print(f'\n{"Algorithm":<18} {"Scenario":<15} {"Assign":>7} {"Weighted":>9} {"OnTime":>7} {"Time(ms)":>9} {"LoadSTD":>8} {"Dist(km)":>9}')
print('-'*100)

for algo_name, allocator in allocators.items():
    # Track per-priority assignment for CBBA variants
    algo_pri_breakdown = {p: {'assigned': 0, 'total': 0} for p in range(5)}
    for t in tasks:
        algo_pri_breakdown[t.priority]['total'] += 1

    for sc_name, comm_graph in scenarios.items():
        # Reset
        for t in tasks:
            t.status = TaskStatus.PENDING; t.assigned_to = None
        for d in drones:
            d.assigned_tasks = []; d.current_task_id = None

        t0 = time.perf_counter()
        try:
            assignments = allocator.allocate(drones, tasks, comm_graph, current_time=0.0)
        except Exception as e:
            print(f'  {algo_name:<18} [{sc_name:<15}] ERROR: {e}')
            continue
        rt_ms = (time.perf_counter() - t0) * 1000.0

        all_assigned = set()
        for tids in assignments.values(): all_assigned.update(tids)

        wsum, wrate = 0.0, 0.0
        for t in tasks:
            w = pw.get(t.priority, 1.0)
            wsum += w
            if t.id in all_assigned:
                wrate += w
                algo_pri_breakdown[t.priority]['assigned'] += 1

        on_time = 0
        for t in tasks:
            if t.id in all_assigned:
                for d in drones:
                    if t.id in assignments.get(d.id, []):
                        dist = np.linalg.norm(d.position - t.pickup_pos) + np.linalg.norm(t.pickup_pos - t.delivery_pos)
                        if dist / max(d.max_speed, 0.1) <= t.time_window[1]: on_time += 1
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

        n_assign, n_total = len(all_assigned), len(tasks)
        bs = [len(v) for v in assignments.values()]

        result = {
            'algorithm': algo_name, 'comm_scenario': sc_name,
            'assignment_rate': n_assign/n_total, 'weighted_priority_rate': wrate/wsum if wsum else 0,
            'on_time_rate': on_time/n_assign if n_assign else 0,
            'total_distance_m': total_dist, 'runtime_ms': rt_ms,
            'load_balance_std': float(np.std(bs)) if bs else 0, 'total_assigned': n_assign,
        }
        all_results.append(result)
        print(f'  {algo_name:<18} [{sc_name:<15}] {result["assignment_rate"]:>6.1%} {result["weighted_priority_rate"]:>8.1%} {result["on_time_rate"]:>6.1%} {result["runtime_ms"]:>8.0f} {result["load_balance_std"]:>7.1f} {result["total_distance_m"]/1000:>8.1f}')

    priority_breakdown[algo_name] = algo_pri_breakdown

# ============================================================
# Export
# ============================================================
def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {str(k): convert(v) for k,v in obj.items()}
    if isinstance(obj, list): return [convert(i) for i in obj]
    return obj

output = {
    'results': convert(all_results),
    'priority_breakdown': convert(priority_breakdown),
    'scene': {'n_drones': n, 'n_tasks': len(tasks)},
}
with open('data/ablation_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ============================================================
# Summary tables
# ============================================================
print('\n' + '='*80)
print('ABLATION STUDY: CBBA Variant Comparison (Ideal Communication)')
print('-'*80)
variants = ['CBBA-Full', 'CBBA-NoPriority', 'CBBA-NoComm', 'CBBA-NoBattery']
ideal_cbba = {r['algorithm']: r for r in all_results if r['comm_scenario'] == 'Ideal(100%)' and r['algorithm'] in variants}

print(f'{"Variant":<20} {"Assign":>7} {"Weighted":>9} {"Diff":>7}')
print('-'*50)
base = ideal_cbba.get('CBBA-Full', {})
base_wr = base.get('weighted_priority_rate', 1.0)
for v in variants:
    r = ideal_cbba.get(v, {})
    wr = r.get('weighted_priority_rate', 0)
    diff = wr - base_wr
    print(f'{v:<20} {r.get("assignment_rate",0):>6.1%} {wr:>8.1%} {diff:>+7.1%}')

print(f'\nResults exported: data/ablation_results.json ({len(all_results)} entries)')
