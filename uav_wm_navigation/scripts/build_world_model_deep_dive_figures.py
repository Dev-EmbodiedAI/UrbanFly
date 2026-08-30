from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "world_model_paired_20260724"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#18324A"
BLUE = "#2E74B5"
TEAL = "#2B7A78"
ORANGE = "#D9822B"
RED = "#A33A32"
LIGHT = "#EEF3F7"
GRAY = "#5E6B76"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def canvas(title: str, subtitle: str = ""):
    fig, axis = plt.subplots(figsize=(13.2, 7.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.04, 0.94, title, fontsize=20, fontweight="bold", color=NAVY)
    if subtitle:
        axis.text(0.04, 0.895, subtitle, fontsize=10.5, color=GRAY)
    return fig, axis


def box(axis, x, y, w, h, title, lines=(), color=BLUE, fill="white"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.8, edgecolor=color, facecolor=fill,
    )
    axis.add_patch(patch)
    axis.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=12, fontweight="bold", color=color)
    for index, line in enumerate(lines):
        axis.text(x + w / 2, y + h - 0.09 - index * 0.035, line, ha="center", va="top", fontsize=9.1, color=NAVY)


def arrow(axis, start, end, color=GRAY, label=None, bend=0.0):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
        color=color, connectionstyle=f"arc3,rad={bend}",
    )
    axis.add_patch(patch)
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.018, label,
                  ha="center", fontsize=8.7, color=color)


# Dreamer RSSM deep dive.
fig, ax = canvas(
    "DreamerV3-style RSSM：先用真实历史定位当前 belief，再沿 15 条 YOPO 候选想象未来",
    "本项目保留世界模型，不训练 actor/critic；动作就是候选轨迹的 [p,v,a] 时间序列。",
)
box(ax, .04, .61, .15, .18, "历史观测", ("depth [B,4,1,96,160]", "state [B,4,13]", "goal [B,3]"), TEAL, "#F1F9F8")
box(ax, .24, .61, .15, .18, "观测编码", ("depth CNN", "state / goal MLP", "e_t"), BLUE, "#F3F7FB")
box(ax, .44, .61, .19, .18, "RSSM 后验", ("h_t = GRU(h,z,a,s)", "q(z_t | h_t,e_t,g)", "8×8 categorical"), ORANGE, "#FFF7EE")
box(ax, .70, .61, .22, .18, "当前 belief", ("确定状态 h_t", "随机状态 z_t", "部分可观测记忆"), TEAL, "#F1F9F8")
for a, b in [((.19, .70), (.24, .70)), ((.39, .70), (.44, .70)), ((.63, .70), (.70, .70))]:
    arrow(ax, a, b)

box(ax, .04, .25, .20, .20, "15 条真实 YOPO 候选", ("candidates [B,15,10,9]", "21点primitive重采样为10点", "不生成新轨迹"), BLUE, "#F3F7FB")
box(ax, .33, .25, .23, .20, "共享 prior 想象滚动", ("p(z_{t+k}|h_{t+k})", "候选条件 GRU 更新", "2 s × 15 branches"), ORANGE, "#FFF7EE")
box(ax, .65, .25, .27, .20, "逐候选风险头", ("P(collision), P(failure)", "minimum clearance, progress", "MC Dropout uncertainty"), RED, "#FCEFED")
arrow(ax, (.24, .35), (.33, .35), label="action sequence")
arrow(ax, (.56, .35), (.65, .35), label="latent rollout")
arrow(ax, (.81, .61), (.45, .45), TEAL, "复制 belief 到15分支", .12)
ax.text(.50, .12, "训练：posterior/prior KL/free-nats + factual future depth/reward/continuation + 15候选几何风险监督",
        ha="center", fontsize=11, color=NAVY,
        bbox=dict(boxstyle="round,pad=.45", facecolor=LIGHT, edgecolor=BLUE))
ax.text(.50, .055, "收益预期：随机潜状态能表示遮挡后的多种可能未来；代价是训练更难、rollout 漂移和推理更重。",
        ha="center", fontsize=10, color=RED)
fig.savefig(OUT / "dreamer_rssm_candidate_imagination.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# JEPA deep dive.
fig, ax = canvas(
    "Action-Conditioned JEPA：预测“执行该候选后未来应该处于什么表征”，而不是生成未来像素",
    "在线分支接收遮挡后的当前历史；EMA 目标分支只提供实际执行候选的真实未来 latent。",
)
box(ax, .03, .61, .18, .19, "当前历史（在线）", ("masked depth history", "state history + goal", "mask ratio = 0.5"), TEAL, "#F1F9F8")
box(ax, .27, .61, .18, .19, "在线编码器", ("CNN context encoder", "96-d context token", "可反向传播"), BLUE, "#F3F7FB")
box(ax, .51, .61, .20, .19, "动作条件 Predictor", ("context + 10 action tokens", "2-layer / 4-head Transformer", "causal attention"), ORANGE, "#FFF7EE")
box(ax, .77, .61, .19, .19, "预测未来 latent", ("15 candidates × horizon", "风险头读取整段", "无像素解码器"), RED, "#FCEFED")
for a, b in [((.21, .705), (.27, .705)), ((.45, .705), (.51, .705)), ((.71, .705), (.77, .705))]:
    arrow(ax, a, b)

box(ax, .03, .25, .18, .18, "真实未来（仅 factual）", ("executed candidate", "future depth × 5", "valid mask"), TEAL, "#F1F9F8")
box(ax, .27, .25, .18, .18, "EMA 目标编码器", ("stop-gradient", "decay = 0.996", "稳定 target latent"), BLUE, "#F3F7FB")
box(ax, .53, .25, .18, .18, "表征对齐损失", ("cosine distance", "variance regularization", "covariance regularization"), ORANGE, "#FFF7EE")
box(ax, .77, .25, .19, .18, "15候选风险监督", ("collision / failure", "clearance / progress", "source/confidence/mask"), RED, "#FCEFED")
arrow(ax, (.21, .34), (.27, .34))
arrow(ax, (.45, .34), (.53, .34), label="target z")
arrow(ax, (.77, .70), (.63, .43), ORANGE, "selected latent", -.12)
arrow(ax, (.71, .34), (.77, .34))
ax.text(.50, .115, "为什么可能更适合当前 8 GB GPU：不重建高维像素，把容量集中到候选之间的相对后果与时序一致性。",
        ha="center", fontsize=11, color=NAVY,
        bbox=dict(boxstyle="round,pad=.45", facecolor=LIGHT, edgecolor=TEAL))
ax.text(.50, .055, "主要风险：latent 可能忽略细杆/边缘，所以仍需显式碰撞与间距标签；当前实现不是十亿参数 V-JEPA 2 复现。",
        ha="center", fontsize=10, color=RED)
fig.savefig(OUT / "jepa_action_conditioned_latent_prediction.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# Ablation logic.
fig, ax = canvas(
    "为什么这样做消融：只改变“15候选如何评分”，不改变候选生成、控制或安全层",
    "这样闭环差异才能归因于世界表征，而不是规划器、控制器、路线或急停规则同时变化。",
)
box(ax, .04, .59, .15, .20, "固定输入", ("同一深度/状态", "同一路线/seed", "同一YOPO权重"), TEAL, "#F1F9F8")
box(ax, .25, .59, .15, .20, "固定候选", ("官方15 primitives", "同一p/v/a", "同一YOPO cost"), BLUE, "#F3F7FB")
box(ax, .46, .59, .18, .20, "唯一可变模块", ("A: raw cost", "B: RSSM risk", "C: JEPA risk"), ORANGE, "#FFF7EE")
box(ax, .70, .59, .12, .20, "固定执行", ("同一公式", "SafetyFilter", "50 Hz控制"), BLUE, "#F3F7FB")
box(ax, .87, .59, .09, .20, "结果", ("碰撞", "成功", "时延"), RED, "#FCEFED")
for a, b in [((.19, .69), (.25, .69)), ((.40, .69), (.46, .69)), ((.64, .69), (.70, .69)), ((.82, .69), (.87, .69))]:
    arrow(ax, a, b)

rows = [
    ("A 纯YOPO", "无记忆", "最低延迟；建立能力下界", "动态遮挡下只看当前深度"),
    ("B +Dreamer", "随机潜动力学", "部分可观测、多模态未来、不确定性", "训练/滚动更难，实时成本更高"),
    ("C +JEPA", "预测未来latent", "不重建像素，轻量、泛化潜力", "可能漏掉小障碍与精细几何"),
]
for row, (name, representation, benefit, risk) in enumerate(rows):
    y = .43 - row * .13
    ax.add_patch(FancyBboxPatch((.06, y), .88, .10, boxstyle="round,pad=.006",
                                facecolor="white" if row % 2 == 0 else "#F7F9FC", edgecolor="#C8D3DC"))
    ax.text(.09, y + .05, name, va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(.26, y + .05, representation, va="center", fontsize=9.8, color=BLUE)
    ax.text(.45, y + .05, benefit, va="center", fontsize=9.5, color=TEAL)
    ax.text(.75, y + .05, risk, va="center", fontsize=9.3, color=RED)
ax.text(.09, .55, "实验组", fontsize=9, color=GRAY)
ax.text(.26, .55, "新增表征", fontsize=9, color=GRAY)
ax.text(.45, .55, "预期收益", fontsize=9, color=GRAY)
ax.text(.75, .55, "必须验证的代价", fontsize=9, color=GRAY)
ax.text(.50, .055, "当前单 seed 配对闭环只证明链路打通；碰撞率收益必须靠多 seed、困难动态场景和配对统计。",
        ha="center", fontsize=10.5, color=RED,
        bbox=dict(boxstyle="round,pad=.38", facecolor="#FCEFED", edgecolor=RED))
fig.savefig(OUT / "ablation_causal_design.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# Bug and repair timeline.
fig, ax = canvas(
    "世界模型闭环故障链与修复：把偶发“卡住”变成可定位、可终止、可复验的状态机",
)
events = [
    (.09, "旧 Dreamer 回合", "前 7.54 s 发布计划\n随后停止产出", RED),
    (.31, "证据分离", "单次前向 P95 19.40 ms\n不是模型计算超时", ORANGE),
    (.53, "新看门狗", "heartbeat / stage\n0.5 s no-plan 硬失败", BLUE),
    (.75, "JEPA 竞态复现", "reset 与 history append 并发\nnp.stack([])", RED),
    (.91, "原子修复+复验", "history lock + snapshot\nDreamer/JEPA 均到达", TEAL),
]
ax.plot([.09, .91], [.56, .56], color="#AAB8C2", lw=3)
for x, title, detail, color in events:
    ax.scatter([x], [.56], s=150, color=color, zorder=3)
    ax.text(x, .66, title, ha="center", fontsize=11, fontweight="bold", color=color)
    ax.text(x, .47, detail, ha="center", va="top", fontsize=9.2, color=NAVY)
ax.text(.50, .20, "新配对闭环（seed 6102）：YOPO 38.40 s；Dreamer 40.03 s；JEPA 40.60 s；三者均无碰撞到达。",
        ha="center", fontsize=11.2, color=NAVY,
        bbox=dict(boxstyle="round,pad=.5", facecolor="#E8F4F2", edgecolor=TEAL))
ax.text(.50, .10, "仍未解决：世界模型规划率仅 6.77/7.22 Hz，低于 8 Hz 目标；因此“链路成功”不等于“实时指标全部达标”。",
        ha="center", fontsize=10.2, color=RED)
fig.savefig(OUT / "world_model_bug_fix_timeline.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(OUT.resolve())
