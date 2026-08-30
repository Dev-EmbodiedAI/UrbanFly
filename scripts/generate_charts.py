"""
Generate charts for mid-term report from benchmark data.
Output: thesis/figures/*.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Chinese font setup
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

# Load data
with open('data/benchmark_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Organize by algorithm and scenario
algorithms = ['cbba', 'hungarian', 'greedy', 'auction', 'genetic', 'market']
algo_labels = ['CBBA\n(本文)', 'Hungarian\n(最优)', 'Greedy\n(贪心)', 'Auction\n(拍卖)', 'Genetic\n(遗传)', 'Market\n(市场)']
scenarios = ['ideal', 'occlusion', 'intermittent']
scenario_labels = ['Ideal\n(100%连通)', 'Occlusion\n(71%连通)', 'Intermittent\n(36%连通)']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#95a5a6']

def get_val(algo, sc, key):
    for r in data:
        if r['algorithm'] == algo and r['comm_scenario'] == sc:
            return r.get(key, 0)
    return 0

# ============================================================
# Figure 1: Assignment Rate Comparison (grouped bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(algorithms))
width = 0.25

for i, (sc, sl) in enumerate(zip(scenarios, scenario_labels)):
    vals = [get_val(a, sc, 'assignment_rate') * 100 for a in algorithms]
    bars = ax.bar(x + i*width, vals, width, label=sl.replace('\n', ' '), alpha=0.85)

ax.set_ylabel('Task Assignment Rate (%)')
ax.set_title('Figure 2.1: Algorithm Assignment Rate Comparison (30 UAVs × 100 Tasks)')
ax.set_xticks(x + width)
ax.set_xticklabels(algo_labels)
ax.legend(loc='lower right')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (sc, sl) in enumerate(zip(scenarios, scenario_labels)):
    vals = [get_val(a, sc, 'assignment_rate') * 100 for a in algorithms]
    for j, v in enumerate(vals):
        ax.text(j + i*width, v + 1, f'{v:.0f}%', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('thesis/figures/fig_assignment_rate.png')
plt.close()
print('Figure 1 saved: fig_assignment_rate.png')

# ============================================================
# Figure 2: Weighted Priority Rate + Total Distance
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Weighted rate
x = np.arange(len(algorithms))
for i, (sc, sl) in enumerate(zip(scenarios, scenario_labels)):
    vals = [get_val(a, sc, 'weighted_priority_rate') * 100 for a in algorithms]
    ax1.bar(x + i*width, vals, width, label=sl.replace('\n', ' '), alpha=0.85)
ax1.set_ylabel('Weighted Priority Completion Rate (%)')
ax1.set_title('Weighted Rate')
ax1.set_xticks(x + width)
ax1.set_xticklabels(algo_labels)
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=7)

# Subplot 2: Total distance
for i, (sc, sl) in enumerate(zip(scenarios, scenario_labels)):
    vals = [get_val(a, sc, 'total_distance_m') / 1000 for a in algorithms]
    ax2.bar(x + i*width, vals, width, label=sl.replace('\n', ' '), alpha=0.85)
ax2.set_ylabel('Total Flight Distance (km)')
ax2.set_title('Total Distance')
ax2.set_xticks(x + width)
ax2.set_xticklabels(algo_labels)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Figure 2.2: Weighted Priority Rate vs Total Distance')
plt.tight_layout()
plt.savefig('thesis/figures/fig_weighted_distance.png')
plt.close()
print('Figure 2 saved: fig_weighted_distance.png')

# ============================================================
# Figure 3: Runtime comparison (log scale)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(algorithms))

for i, (sc, sl) in enumerate(zip(scenarios, scenario_labels)):
    vals = [max(get_val(a, sc, 'runtime_ms'), 0.1) for a in algorithms]  # avoid log(0)
    ax.bar(x + i*width, vals, width, label=sl.replace('\n', ' '), alpha=0.85)

ax.set_ylabel('Runtime (ms, log scale)')
ax.set_title('Figure 2.3: Algorithm Runtime Comparison (log scale)')
ax.set_xticks(x + width)
ax.set_xticklabels(algo_labels)
ax.set_yscale('log')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (sc, sl) in enumerate(zip(scenarios, scenario_labels)):
    vals = [max(get_val(a, sc, 'runtime_ms'), 0.1) for a in algorithms]
    for j, v in enumerate(vals):
        ax.text(j + i*width, v * 1.15, f'{v:.0f}', ha='center', va='bottom', fontsize=6.5)

plt.tight_layout()
plt.savefig('thesis/figures/fig_runtime.png')
plt.close()
print('Figure 3 saved: fig_runtime.png')

# ============================================================
# Figure 4: Communication Degradation Heatmap
# ============================================================
ideal_vals = {a: get_val(a, 'ideal', 'weighted_priority_rate') for a in algorithms}

fig, ax = plt.subplots(figsize=(8, 3.5))
degradation_data = []
row_labels = ['Occlusion\n(71%连通)', 'Intermittent\n(36%连通)']

for sc in ['occlusion', 'intermittent']:
    row = []
    for a in algorithms:
        ideal_v = ideal_vals[a]
        sc_v = get_val(a, sc, 'weighted_priority_rate')
        deg = (1 - sc_v / ideal_v) * 100 if ideal_v > 0 else 0
        row.append(deg)
    degradation_data.append(row)

degradation_data = np.array(degradation_data)

im = ax.imshow(degradation_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=20)
ax.set_xticks(range(len(algorithms)))
ax.set_xticklabels(['CBBA', 'Hungarian', 'Greedy', 'Auction', 'Genetic', 'Market'])
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)

# Add text annotations
for i in range(len(row_labels)):
    for j in range(len(algorithms)):
        val = degradation_data[i, j]
        color = 'white' if val > 10 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontweight='bold')

ax.set_title('Figure 2.4: Communication Degradation Rate (%)')
plt.colorbar(im, ax=ax, label='Degradation %')

plt.tight_layout()
plt.savefig('thesis/figures/fig_degradation.png')
plt.close()
print('Figure 4 saved: fig_degradation.png')

# ============================================================
# Figure 5: Load Balance comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(algorithms))
sc = 'ideal'  # use ideal scenario
vals = [get_val(a, sc, 'load_balance_std') for a in algorithms]
bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white')
ax.set_ylabel('Bundle Size Standard Deviation')
ax.set_title('Figure 2.5: Load Balance Comparison (Ideal Communication)')
ax.set_xticks(x)
ax.set_xticklabels(algo_labels)
ax.grid(axis='y', alpha=0.3)

for j, v in enumerate(vals):
    ax.text(j, v + 0.05, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('thesis/figures/fig_load_balance.png')
plt.close()
print('Figure 5 saved: fig_load_balance.png')

# ============================================================
# Figure 6: CBBA breakdown - assignment vs weighted rate
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
metrics = ['Assignment\nRate', 'Weighted\nPriority Rate']
cbba_vals = [73.0, 84.9]
hungarian_vals = [100, 100]
greedy_vals = [100, 100]

x = np.arange(len(metrics))
width = 0.25
ax.bar(x - width, cbba_vals, width, label='CBBA (本文)', color='#2ecc71', alpha=0.85)
ax.bar(x, hungarian_vals, width, label='Hungarian (最优)', color='#3498db', alpha=0.85)
ax.bar(x + width, greedy_vals, width, label='Greedy', color='#9b59b6', alpha=0.85)

ax.set_ylabel('Rate (%)')
ax.set_title('Figure 2.6: CBBA Priority-Aware Assignment Gap')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# Annotate the gap
ax.annotate('Gap = 11.9%\n(Priority\nAwareness)',
            xy=(1, 84.9), xytext=(1.3, 70),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red')

plt.tight_layout()
plt.savefig('thesis/figures/fig_cbba_gap.png')
plt.close()
print('Figure 6 saved: fig_cbba_gap.png')

print('\nAll figures generated in thesis/figures/')
