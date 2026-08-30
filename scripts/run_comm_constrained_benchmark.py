"""
Communication-Constrained Benchmark
===================================
真实通信约束实验: 所有算法必须遵守通信拓扑。
集中式算法只能看到其连通分量内的无人机和任务。
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

# ============================================================
# Scene setup
# ============================================================
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

n_drones = len(drones)
n_tasks = len(tasks)
print(f'Scene: {n_drones} drones, {n_tasks} tasks')

# ============================================================
# Comm scenarios
# ============================================================
np.random.seed(42)
buildings = []
for _ in range(120):
    buildings.append({'cx':np.random.uniform(-380,380),'cy':np.random.uniform(-380,380),
        'hw':np.random.uniform(8,50),'hd':np.random.uniform(6,30),'h':np.random.uniform(15,150)})

def lai(x1,y1,x2,y2,cx,cy,hw,hd):
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
        if lai(p1[0],p1[1],p2[0],p2[1],b['cx'],b['cy'],b['hw'],b['hd']):
            if 0<=(p1[2]+p2[2])/2<=b['h']: return True
    return False

# Build communication graphs
n = n_drones
R_MAX = 500.0

graphs = {}
# Ideal: full
graphs['Ideal(100%)'] = None

# Occlusion: building-blocked
occ = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        if np.linalg.norm(drones[i].position[:2]-drones[j].position[:2])<R_MAX and not blocked(drones[i].position,drones[j].position):
            occ[i,j]=occ[j,i]=1.0
graphs['Occlusion(69%)'] = occ

# Severe occlusion
sev = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        if np.linalg.norm(drones[i].position[:2]-drones[j].position[:2])<R_MAX*0.6 and not blocked(drones[i].position,drones[j].position):
            if random.random()<0.6: sev[i,j]=sev[j,i]=1.0
graphs['Severe(35%)'] = sev

# Intermittent
inter = occ*(np.random.rand(n,n)<0.5)
np.fill_diagonal(inter,0)
graphs['Intermittent'] = inter

# ============================================================
# Find connected components for centralized algorithms
# ============================================================
def find_components(adj):
    """DFS to find connected components"""
    if adj is None:
        return [list(range(n))]
    visited = set()
    components = []
    for start in range(n):
        if start not in visited:
            comp = []
            stack = [start]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v)
                    comp.append(v)
                    for u in range(n):
                        if adj[v,u] > 0 and u not in visited:
                            stack.append(u)
            components.append(comp)
    return components

# ============================================================
# Allocators
# ============================================================
pw = PRIORITY_WEIGHTS

def eval_assignment(assignments, drones, tasks):
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
                    dist = np.linalg.norm(d.position-t.pickup_pos)+np.linalg.norm(t.pickup_pos-t.delivery_pos)
                    if dist/max(d.max_speed,0.1)<=t.time_window[1]: on_time+=1
                    break

    total_dist=0.0
    for d in drones:
        curr=d.position
        for tid in assignments.get(d.id,[]):
            for t in tasks:
                if t.id==tid:
                    total_dist+=np.linalg.norm(curr-t.pickup_pos)+np.linalg.norm(t.pickup_pos-t.delivery_pos)
                    curr=t.delivery_pos; break

    n_assign=len(all_assigned)
    bs=[len(v) for v in assignments.values()]
    return {
        'assignment_rate': n_assign/len(tasks),
        'weighted_priority_rate': wrate/wsum if wsum else 0,
        'on_time_rate': on_time/n_assign if n_assign else 0,
        'total_distance_m': total_dist,
        'load_balance_std': float(np.std(bs)) if bs else 0,
        'total_assigned': n_assign,
    }

# ============================================================
# Run all
# ============================================================
all_results = []
print(f'\n{"Algorithm":<14} {"Scenario":<16} {"Assign":>7} {"Weighted":>9} {"Runtime":>9} {"LoadSTD":>8}')
print('-'*75)

for sc_name, comm_graph in graphs.items():
    components = find_components(comm_graph)
    n_comp = len(components)
    comp_sizes = [len(c) for c in components]
    max_comp = max(comp_sizes)
    edges = int(comm_graph.sum()/2) if comm_graph is not None else n*(n-1)//2
    max_edges = n*(n-1)//2
    pct = 100*edges/max_edges

    for algo_type, algo_name, allocator in [
        ('分布式', 'CBBA', CBBAAllocator(max_iterations=30, max_bundle_size=8)),
        ('集中式', 'Hungarian', HungarianAllocator()),
        ('集中式', 'Genetic', GeneticAllocator(population_size=20,generations=15)),
        ('集中式', 'SA', SAAllocator(T_init=1000,T_min=0.01,alpha=0.95,steps_per_T=40)),
        ('分布式', 'Auction', AuctionAllocator()),
        ('分布式', 'Greedy', GreedyAllocator()),
        ('元启发', 'WOA', WOAAllocator(n_whales=40,n_iterations=60)),
        ('元启发', 'GWO', GWOAllocator(n_wolves=40,n_iterations=60)),
        ('元启发', 'PSO', PSOAllocator(n_particles=40,n_iterations=60)),
        ('元启发', 'DE', DEAllocator(pop_size=40,n_iterations=60)),
        ('元启发', 'ACO', ACOAllocator(n_ants=30,n_iterations=60)),
    ]:
        # Reset
        for t in tasks: t.status=TaskStatus.PENDING; t.assigned_to=None
        for d in drones: d.assigned_tasks=[]; d.current_task_id=None

        t0 = time.perf_counter()

        if algo_type == '集中式' and comm_graph is not None:
            # ====== CRITICAL: 集中式算法只能在最大连通分量内分配 ======
            # 模拟真实场景: 只有连通分量内的无人机可以汇报状态
            largest_comp = max(components, key=len)
            connected_drones = [drones[i] for i in largest_comp]
            # 只有可被connected_drones执行的任务
            connected_task_ids = set()
            for d in connected_drones:
                for t in tasks:
                    if t.payload_weight <= d.max_payload:
                        connected_task_ids.add(t.id)
            connected_tasks = [t for t in tasks if t.id in connected_task_ids]

            if len(connected_drones) > 0 and len(connected_tasks) > 0:
                try:
                    # 在连通子图上运行集中式算法
                    assignments = allocator.allocate(connected_drones, connected_tasks, None, current_time=0.0)
                    # 未在子图中的无人机得到空分配
                    full_assignments = {d.id: [] for d in drones}
                    for d_id, t_ids in assignments.items():
                        if d_id in [dd.id for dd in connected_drones]:
                            full_assignments[d_id] = t_ids
                except Exception as e:
                    full_assignments = {d.id: [] for d in drones}
                    # print(f'  {algo_name} partial: {e}')
            else:
                full_assignments = {d.id: [] for d in drones}
        else:
            # 分布式算法: 正常调用, 自动遵守comm_graph
            try:
                assignments = allocator.allocate(drones, tasks, comm_graph, current_time=0.0)
                full_assignments = assignments
            except Exception as e:
                full_assignments = {d.id: [] for d in drones}
                # print(f'  {algo_name}: ERROR {e}')

        rt_ms = (time.perf_counter() - t0) * 1000.0
        r = eval_assignment(full_assignments, drones, tasks)
        r['algorithm'] = algo_name
        r['comm_scenario'] = sc_name
        r['algo_type'] = algo_type
        r['runtime_ms'] = rt_ms
        r['n_components'] = n_comp
        r['max_component_size'] = max_comp
        all_results.append(r)
        print(f'  {algo_name:<14} [{sc_name:<16}] {r["assignment_rate"]:>6.1%} {r["weighted_priority_rate"]:>8.1%} {rt_ms:>8.0f}ms {r["load_balance_std"]:>7.1f}')

# ============================================================
# Export
# ============================================================
def convert(obj):
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,dict): return {str(k):convert(v) for k,v in obj.items()}
    if isinstance(obj,list): return [convert(i) for i in obj]
    return obj

with open('data/comm_constrained_results.json','w',encoding='utf-8') as f:
    json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
print(f'\nExported {len(all_results)} results to data/comm_constrained_results.json')
