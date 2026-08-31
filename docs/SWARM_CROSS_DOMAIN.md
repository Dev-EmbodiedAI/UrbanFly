# UrbanFly × Swarm 跨环境多无人机实验支线

> 定位声明：UrbanFly 的系统主线保持为 **Agent 能力 + Action-Conditioned
> World Model + Helsinki 数字孪生闭环导航**。本文只定义 Swarm benchmark、
> 统一 policy contract 与跨域泛化实验，不改变 UrbanFly 的核心研究路线。

## 目标

本实验不复制或拼接两个仓库。Swarm 保持程序化环境与标准
`cf_swarm_autopilot` contract 的权威实现；UrbanFly 保持 Helsinki 实景数字
孪生、RGB-D、六自由度执行与三角网格碰撞的权威实现。两端通过同一个公开
policy contract 比较：

```text
同一 policy
  ├─ Swarm：City / Open / Mountain / Village / Forest，2–8 UAV
  └─ UrbanFly：Helsinki realistic digital twin，2–8 UAV
          ↓
zero-shot → Helsinki adaptation → World Model-assisted closed loop
```

研究问题是 Procedural → Realistic Digital Twin 的跨域泛化，不是把 Swarm 的
PyBullet 世界搬进 Helsinki，也不是用 Helsinki planner 替换 Swarm Benchmark。

## 与 UrbanFly 主线的关系

UrbanFly 的闭环层级保持不变：

1. **Agent 层**：理解自然语言任务和多机语义约束，调用 Qwen API 等外部能力，
   执行任务分配、共享线索推理、异常解释与动态重规划；
2. **World Model 层**：以 observation、state、local goal 和候选 action sequence
   为条件，预测未来隐状态、进展、碰撞/失效风险，并参与 receding-horizon 候选
   轨迹重排；
3. **执行与数字孪生层**：Helsinki RGB-D、6DoF、控制器、reset 与三角网格碰撞
   提供真实闭环和可复核数据；
4. **Swarm benchmark 层**：只提供额外的程序化训练/测试域和统一 policy 接口，
   衡量同一策略从程序化环境迁移到真实城市数字孪生后的泛化差距。

classical teacher、BC 和 DAgger 是统一 policy 的数据启动、预训练和消融基线，
不是 UrbanFly 的最终系统架构，也不会取代 Agent 或 World Model。

## 已建立的统一边界

`backend/integrations/swarm_policy.py` 独立实现并验证：

- `depth`: `[N,128,128,1]`，范围 `[0,1]`；
- `state`: `[N,190]`，包含位置、RPY、线/角速度、25 步动作历史、归一化高度、
  shared clue 和 7 个最近邻机槽；
- `action`: `[N,5]`，三维方向、速度比例和绝对 yaw；
- 动态 `N=2..8`；
- UrbanFly `[east,up,north]` 与 policy ENU `[east,north,up]` 的显式双向转换。

Helsinki 的规划器、控制器、采样器和三角几何保持冻结。统一 contract 只做数据
与坐标适配，不改变两端的仿真真值。

## 已完成的 Swarm 原生审计

上游源码固定到 `swarm-subnet/swarm@112a0592dab131f644cd6afdf7c6a9acd9de0a37`。

- 上游 `test_swarm_autopilot_family.py`: 14/14 PASS；
- 五类环境 × 2–8 UAV：35/35 原生 PyBullet contract 矩阵 PASS；
- depth/state/action shape、shared clue 一致性、邻机槽、评分向量与算术平均均 PASS；
- 强制两机同位姿接触后，双方均产生 `OBSTACLE_COLLISION`，每机 0.01、总分
  0.01，碰撞到评分链 PASS；
- 机器可读报告：`outputs/swarm_integration_v1/native_contract_matrix.json`。

限制：本地 Windows 只能安装 `swarm-bullet3==2.0.0.1`，而上游锁定
`2.0.0.3`；Docker/Cap'n Proto validator 隔离层尚未正式跑通。因此以上是原生
仿真/contract PASS，不写成官方 Docker Benchmark PASS。

## 已完成的五环境数字孪生闭环

`backend/digital_twin/` 已把 Swarm 原生环境收敛到与 Helsinki 相同的因果生命周期：

```text
目标/任务分配 → policy 候选动作 → World Model 预测与重排
              → 原生环境执行 → 新 depth/state 反馈到下一步
```

固定 seed `20260831`、每类 2 架无人机的真实结果如下；五个报告均由同一份
policy 源码生成，并通过独立合并 QA：

| 环境 | 成功 | 碰撞 | 控制步 | 原生分数 |
|---|---:|---:|---:|---:|
| City | 2/2 | 0 | 1,727 | 0.8514 |
| Open | 2/2 | 0 | 1,838 | 0.7530 |
| Mountain | 2/2 | 0 | 2,935 | 0.5727 |
| Village | 2/2 | 0 | 2,988 | 0.6028 |
| Forest | 2/2 | 0 | 2,743 | 0.6185 |

合计 **10/10 UAV 成功、0 collision、12,231 次 decision/execution/fresh-feedback**。
最终报告只保留五个环境 JSON 与
`outputs/cross_environment_digital_twin_v1/cross_environment_qa.json`；开发失败报告已清理。

必须明确三项限制：这些 episode 向策略暴露精确目标，因此
`benchmark_eligible=false`，不能写成官方隐藏目标 Benchmark；Swarm 当前使用的是
可解释的一步解析预测器，不是 Helsinki 已训练的 192-D latent checkpoint；单 seed、
2 UAV 只证明闭环可用性，不证明统计泛化。下一阶段才是共享 learned representation、
多 seed 与 2–8 UAV 的严格跨域比较。

## Baseline 决策：不把强化学习设为前提

上游 4 机共享参数 PPO 已真实训练 51,200 timesteps，模型梯度和打包链路通过；
但已完成的 City/Open/Mountain rollout 均出现失败或碰撞，表明 50k 稀疏团队
奖励 starter 不是可用导航策略。按项目精简原则，该失败权重和 submission 已
删除，不进入 `models/`、Git 或 Release。

Swarm 实验支线的非 RL Baseline 改为：

1. 可解释 teacher：共享线索搜索、任务/平台分配、速度障碍或 ORCA、多机间隔、
   深度安全层与降落状态机；
2. 行为克隆：用 teacher 在程序化五类环境生成公开 observation → action 示范；
3. DAgger：让学生闭环运行，teacher 只纠正学生访问到的失败状态；
4. 可选 residual RL：仅在 BC/DAgger 稳定后用于小幅优化时间/安全分，不作为
   能飞起来的前提。

`backend/integrations/swarm_imitation.py` 已实现真正可训练的共享多机网络：每机
共享 RGB-D/state 编码，团队 self-attention 支持动态 2–8 机，输出动作与碰撞
概率辅助头；forward、动作范围、mask、loss backward 和 checkpoint 回读共 4 项
测试均 PASS。它目前是 `TRAINABLE / NOT TRAINED`，不能声称已有成功率。

## 后续严格实验矩阵

### A. Procedural 训练与测试

- 训练：五类环境、2–8 机、不同 seed；
- 测试：固定 held-out seed；
- 指标：全队成功率、每机成功率、碰撞率、分数、完成时间、最小间距、推理延迟；
- 对照：确定性 classical、BC、DAgger、可选 residual RL。

### B. Helsinki zero-shot

- 同一 checkpoint，不改权重；
- 同一 `[N,128,128,1] + [N,190] → [N,5]` contract；
- 只允许环境 adapter 处理坐标、深度量程与执行接口；
- 报告 realistic-domain 性能下降和失败类型。

### C. Helsinki adaptation 与主线闭环

- Helsinki BC fine-tune；
- Helsinki DAgger；
- 接入 UrbanFly Agent 的任务分配、共享语义线索推理与动态重规划；
- 接入 action-conditioned World Model 的未来隐状态/风险预测与候选轨迹重排；
- 使用相同 held-out Helsinki 任务比较，禁止更换测试集。

## Swarm 实验支线的下一 Gate

`PLANNED` — 在已通过的五环境解析闭环基础上实现 Swarm privileged classical
teacher 与精简示范 writer，并扩展到多 seed、2–8 UAV。只有 teacher 的成功、碰撞
与 landing 链通过后才开始 BC/DAgger；随后让同一 learned checkpoint 经过统一
contract 进入 Helsinki zero-shot 和 adaptation 对照。不会先生成空洞模型权重，
该 Gate 也不阻塞或替换 UrbanFly Agent + World Model 主线。
