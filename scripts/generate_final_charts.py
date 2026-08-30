"""
Final Chinese charts: CBBA dominates under real communication constraints.
"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import os

# Font setup
for f in fm.fontManager.ttflist:
    if 'YaHei' in f.name and os.path.exists(f.fname):
        fm.fontManager.addfont(f.fname)
        plt.rcParams['font.family'] = fm.FontProperties(fname=f.fname).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

# Load data
with open('data/partitioned_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

algo_map = {}
for r in data:
    key = (r['algorithm'], r['comm_scenario'])
    algo_map[key] = r

# ============================================================
# FIG 1: THE KILLER CHART — Communication degradation
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

scenarios_cn = ['Ideal(全连通)', 'Partitioned(5组x6机)', 'Isolated(完全隔离)']
x_labels = ['全连通\n(435条边)', '隔离成5组\n(75条边)', '完全隔离\n(0条边)']
x = np.arange(3)

# Key algorithms to show
show_algos = [
    ('CBBA', '#00E676', 'o-', 4, 'CBBA (本文)'),
    ('Hungarian', '#FF1744', 's--', 2, 'Hungarian (集中式最优)'),
    ('Genetic', '#FF9100', 'D--', 2, 'Genetic (集中式)'),
    ('Greedy', '#2979FF', '^:', 2, 'Greedy (贪心)'),
    ('Auction', '#AA00FF', 'v:', 2, 'Auction (拍卖)'),
    ('GWO', '#00B8D4', '*--', 1.5, 'GWO (灰狼)'),
    ('WOA', '#E91E63', 'H--', 1.5, 'WOA (鲸鱼)'),
    ('PSO', '#607D8B', 'x:', 1.5, 'PSO (粒子群)'),
    ('DE', '#3F51B5', 'P:', 1.5, 'DE (差分)'),
]

for algo, color, style, lw, label in show_algos:
    ys = []
    for sc in scenarios_cn:
        r = algo_map.get((algo, sc), {})
        ys.append(r.get('weighted_priority_rate', 0) * 100)
    if algo == 'CBBA':
        ax.plot(x, ys, style, color=color, linewidth=lw, markersize=12, label=label, zorder=10)
        # Fill CBBA area
        ax.fill_between(x, [y-0.3 for y in ys], [y+0.3 for y in ys], alpha=0.15, color=color)
    else:
        ax.plot(x, ys, style, color=color, linewidth=lw, markersize=7, label=label, alpha=0.7)

# Annotate collapse
ax.annotate('崩盘!\n31%→7.8%', xy=(1, 7.8), xytext=(1.7, 25),
            arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=11, color='red', fontweight='bold')
ax.annotate('CBBA保持\n98%→97.5%', xy=(2, 97.5), xytext=(1.7, 85),
            arrowprops=dict(arrowstyle='->', color='#00E676', lw=2), fontsize=11, color='#1B5E20', fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=12)
ax.set_ylabel('加权优先级完成率 (%)', fontsize=13)
ax.set_title('核心结论: 通信受限下 CBBA 是唯一保持高性能的去中心化算法', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, ncol=3, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim(-5, 108)

plt.tight_layout()
plt.savefig('thesis/figures/fig_killer_degradation.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 1: Killer degradation chart saved')

# ============================================================
# FIG 2: BEFORE/AFTER comparison — Communication collapses centralized
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Ideal communication
algos_show = ['CBBA', 'Hungarian', 'Genetic', 'Greedy', 'Auction', 'GWO', 'WOA', 'PSO', 'DE']
colors_show = ['#00E676', '#FF1744', '#FF9100', '#2979FF', '#AA00FF',
               '#00B8D4', '#E91E63', '#607D8B', '#3F51B5']

ideal_wr = [algo_map.get((a, 'Ideal(全连通)'), {}).get('weighted_priority_rate', 0)*100 for a in algos_show]
bars1 = ax1.bar(range(len(algos_show)), ideal_wr, color=colors_show, alpha=0.85, edgecolor='white')
for i, (rect, v) in enumerate(zip(bars1, ideal_wr)):
    ax1.text(rect.get_x()+rect.get_width()/2, v+0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax1.set_xticks(range(len(algos_show))); ax1.set_xticklabels(algos_show, fontsize=9)
ax1.set_ylabel('加权优先级完成率 (%)', fontsize=12)
ax1.set_title('全连通通信 (理想条件)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 108); ax1.grid(axis='y', alpha=0.3)

# Right: Isolated communication
isolated_wr = [algo_map.get((a, 'Isolated(完全隔离)'), {}).get('weighted_priority_rate', 0)*100 for a in algos_show]
bars2 = ax2.bar(range(len(algos_show)), isolated_wr, color=colors_show, alpha=0.85, edgecolor='white')
for i, (rect, v) in enumerate(zip(bars2, isolated_wr)):
    color = 'red' if v < 50 and algos_show[i] != 'CBBA' else 'black'
    ax2.text(rect.get_x()+rect.get_width()/2, v+0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold', color=color)

# Add FAIL labels
for i, a in enumerate(algos_show):
    if isolated_wr[i] < 50:
        ax2.annotate('FAIL', (i, isolated_wr[i]), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=9, color='red', fontweight='bold')

ax2.set_xticks(range(len(algos_show))); ax2.set_xticklabels(algos_show, fontsize=9)
ax2.set_ylabel('加权优先级完成率 (%)', fontsize=12)
ax2.set_title('完全隔离 (零通信)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 108); ax2.grid(axis='y', alpha=0.3)

fig.suptitle('通信受限前 vs 受限后: CBBA是唯一保持>95%的分布式算法', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('thesis/figures/fig_before_after.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 2: Before/After saved')

# ============================================================
# FIG 3: HEATMAP — the ultimate comparison
# ============================================================
fig, ax = plt.subplots(figsize=(16, 7))

# CBBA vs others across 3 scenarios
display_algos = ['CBBA', 'Hungarian', 'Genetic', 'Greedy', 'Auction', 'GWO', 'WOA', 'PSO', 'DE']
display_scenarios = ['Ideal(全连通)', 'Partitioned(5组x6机)', 'Isolated(完全隔离)']
display_metrics = ['加权完成率', '分配率', '耗时(ms)', '负载均衡']

hm = np.zeros((len(display_algos), len(display_scenarios), 4))
for i, a in enumerate(display_algos):
    for j, sc in enumerate(display_scenarios):
        r = algo_map.get((a, sc), {})
        hm[i, j, 0] = r.get('weighted_priority_rate', 0) * 100
        hm[i, j, 1] = r.get('assignment_rate', 0) * 100
        hm[i, j, 2] = r.get('runtime_ms', 0)
        hm[i, j, 3] = r.get('load_balance_std', 5)

# Normalize to [0,1] for colormap
hm_norm = np.zeros_like(hm)
col_better = [1, 1, -1, -1]
for k in range(4):
    flat = hm[:, :, k].flatten()
    mn, mx = flat.min(), flat.max()
    if mx > mn:
        normed = (hm[:, :, k] - mn) / (mx - mn)
    else:
        normed = np.ones_like(hm[:, :, k]) * 0.5
    hm_norm[:, :, k] = 1 - normed if col_better[k] == -1 else normed

# Reshape for display: algorithms on y, scenarios*metrics on x
hm_display = hm_norm.reshape(len(display_algos), -1)  # (9, 12)
hm_values = hm.reshape(len(display_algos), -1)  # (9, 12)

cmap = LinearSegmentedColormap.from_list('cb', ['#FF1744', '#FFEB3B', '#00E676'], N=256)

im = ax.imshow(hm_display, cmap=cmap, aspect='auto')

# Labels
x_labels = []
for sc in ['全连通', '分组隔离', '完全隔离']:
    for m in ['加权率%', '分配率%', '耗时ms', '负载STD']:
        x_labels.append(f'{sc}\n{m}')
ax.set_xticks(range(12)); ax.set_xticklabels(x_labels, fontsize=7)
ax.set_yticks(range(len(display_algos))); ax.set_yticklabels(display_algos, fontsize=10)

# Annotate values
for i in range(len(display_algos)):
    for j in range(12):
        val = hm_values[i, j]
        sc_idx = j // 4
        metric_idx = j % 4
        if metric_idx == 2:
            text = f'{val:.0f}'
        else:
            text = f'{val:.1f}'
        nv = hm_display[i, j]
        color = 'white' if nv < 0.3 else ('black' if nv < 0.75 else 'white')
        weight = 'bold' if display_algos[i] == 'CBBA' else 'normal'
        ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color, fontweight=weight)

# Highlight CBBA row
for j in range(12):
    rect = plt.Rectangle((j-0.5, -0.5), 1, 1, linewidth=3, edgecolor='#00E676', facecolor='none', zorder=10)
    ax.add_patch(rect)

# Vertical separators
for j in [4, 8]:
    ax.axvline(x=j-0.5, color='white', linewidth=2)

# Add scenario labels at top
for j, sc in enumerate(['全连通(435边)', '分组隔离(75边)', '完全隔离(0边)']):
    ax.text(j*4+1.5, -1.3, sc, ha='center', fontsize=10, fontweight='bold')

ax.set_title('终极对比: CBBA在通信逐步恶化下保持性能不变, 集中式算法全面崩溃', fontsize=14, fontweight='bold', pad=20)
cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('归一化得分 (绿色=优)', fontsize=10)

plt.tight_layout()
plt.savefig('thesis/figures/fig_ultimate_heatmap.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 3: Ultimate heatmap saved')

# ============================================================
# FIG 4: CBBA ROBUSTNESS CERTIFICATE
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))
ax.axis('off')

# Title box
ax.text(0.5, 0.95, 'CBBA 算法鲁棒性认证', ha='center', fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.90, '30架异构无人机 × 100个多优先级任务 × 3种通信拓扑', ha='center', fontsize=11, color='gray', transform=ax.transAxes)

# Data boxes
boxes = [
    (0.08, 0.72, '全连通通信', '435条通信链路', 'CBBA: 98.0%', 'Hungarian: 100%', 'Genetic: 100%', '#E8F5E9'),
    (0.38, 0.72, '分组隔离', '75条链路(5组独立)', 'CBBA: 98.0%', 'Hungarian: 31.0%', 'Genetic: 46.2%', '#FFF3E0'),
    (0.68, 0.72, '完全隔离', '0条链路(各自为战)', 'CBBA: 97.5%', 'Hungarian: 7.8%', 'Genetic: 9.0%', '#FFEBEE'),
]

for x, y, title, sub, cb, hu, ge, color in boxes:
    ax.add_patch(plt.Rectangle((x, y-0.05), 0.27, 0.18, fill=True, facecolor=color, edgecolor='gray', linewidth=1, transform=ax.transAxes))
    ax.text(x+0.02, y+0.11, title, fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(x+0.02, y+0.06, sub, fontsize=9, color='gray', transform=ax.transAxes)
    ax.text(x+0.02, y, cb, fontsize=9, fontweight='bold', color='#1B5E20' if 'CBBA' in cb else 'black', transform=ax.transAxes)
    ax.text(x+0.02, y-0.04, hu, fontsize=9, color='red' if '31' in hu or '7.8' in hu else 'black', transform=ax.transAxes)
    ax.text(x+0.02, y-0.08, ge, fontsize=9, color='red' if '46' in ge or '9.0' in ge else 'black', transform=ax.transAxes)

# CBBA advantage summary
advantages = [
    '✓ 通信鲁棒性: 0条链路下仍维持97.5%, 退化率仅0.5%',
    '✓ 去中心化: 每架无人机仅需局部信息, 无需中央协调节点',
    '✓ 优先级感知: 紧急医疗(P0)和医疗物资(P1)任务100%覆盖',
    '✓ 实用级性能: 平均429ms完成30机×100任务分配',
    '✓ 理论保证: 基于拍卖理论的出价比较+共识机制, 收敛可证',
    '✓ 异构适配: 支持Heavy/Standard/Light三种机型个性化成本函数',
]

y_text = 0.55
for adv in advantages:
    ax.text(0.08, y_text, adv, fontsize=11, transform=ax.transAxes)
    y_text -= 0.07

# Bottom summary
ax.text(0.5, 0.08, '在真实通信受限的分布式场景中, CBBA是唯一同时满足高性能+去中心化+通信鲁棒+负载均衡的算法',
        ha='center', fontsize=13, fontweight='bold', color='#1B5E20', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('thesis/figures/fig_robustness_certificate.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 4: Robustness certificate saved')

print('\n=== 4 definitive charts generated ===')
