# Lightweight JEPA World-Action Model + MPPI（单无人机）

## 仓库审计结论

### Simulator / control

- `UrbanFlyWorldModelEnv` 是单无人机 Gym 风格闭环入口，提供 `reset()` / `step()`、成功驻留判断、碰撞终止、jerk 与进度 reward。
- 默认环境频率为 50 Hz physics / 10 Hz sensor / 5 Hz policy；WAM-MPC 配置将高层策略设为 10 Hz，`dt=0.1 s`。
- 高层动作是归一化 body-FLU `[forward_velocity, left_velocity, up_velocity, yaw_rate]`，经 `ActionLimits` 还原到物理量，再复用已有安全层与 simulator velocity command。WAM-MPC 不控制 motor RPM。
- 导航世界坐标为 NWU；body 为 FLU；UrbanFly 浏览器坐标 `[x, up, z]` 在 WebSocket adapter 中显式转换为 NWU `[x, z, up]`。
- `VehicleState` 包含 position、xyzw orientation、linear/angular velocity 和 linear acceleration，单位分别为 m、单位四元数、m/s、rad/s、m/s²。
- 深度通过 `SensorFrame` 提供米制 depth、valid mask、可选内参和 camera pose；RGB 可选。Mock 与 WebSocket backend 均实现同一 `SimulatorAdapter`。

### 已有 World Model

| Model | Encoder | Latent | Dynamics | Reward | Value | Policy | Multi-step rollout |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JEPA | depth-history CNN+GRU、state GRU、goal fusion | deterministic candidate latent | trajectory token + causal Transformer；原实现不是逐步 `F(z,a)` | 无 | 无 | 无 | 支持整条 YOPO primitive 条件预测，不支持通用递归 action rollout |
| DreamerV3 style | depth/RGB encoder + state/goal | GRU deterministic + grouped categorical stochastic | RSSM prior/posterior | reward + continuation | 无 critic | 无 actor | 支持候选 action-conditioned imagination；现有训练含 depth reconstruction，因此第一阶段 MPPI 未接它 |
| TD-MPC2 | compact RGB-D/state feature encoder（另有 visual v3） | deterministic task latent | residual latent transition | 有 | twin Q | tanh policy prior | 有；原生 planner 是 policy-guided CEM，不是本次统一 MPPI |

现有 Dreamer 实现没有 actor/critic，准确称谓是 DreamerV3-style world model。TD-MPC2 已有 native CEM baseline，后续统一 adapter 时应保留为额外 baseline，不能重复冒充 MPPI。

### Dataset

复用 HDF5 v2：包含 metric depth/valid mask、position/orientation/linear+angular velocity/acceleration、候选轨迹、selected index、candidate 与 factual collision labels、future positions 以及 episode-local sequence action/reward/continuation。新增 `WAMMPCTransitionDataset` 只派生单机 factual transition，不写 neighbor UAV 或通信字段。

## Phase 1 数据流

```text
Depth history + state history + body-frame goal
  -> existing JEPA ContextEncoder
  -> z_t
  -> JEPAWorldModelAdapter F(z_t, state_t, body-velocity action)
  -> z_(t+1), predicted world-NWU position/velocity, collision probability
  -> vectorized MPPI [samples, horizon, action_dim]
  -> goal + collision + smoothness + control cost
  -> execute first action only
  -> new synchronized observation
  -> re-encode and re-plan
```

`JEPAWorldModelAdapter` 使用速度运动学 backbone 加可学习 residual，而不是像素 decoder。Physics probe 预测 position/velocity residual；Safety probe 预测 collision logit，并与明确的近场 clearance prior 组合。该 prior 只用于安全初始化，正式研究指标必须来自训练 checkpoint。

## 统一接口

`WorldModelBase` 定义：

```python
encode(observation, state=None) -> latent
predict_step(latent, state, action, dt=...) -> {
    "latent": ...,
    "state": ...,
    "state_prediction": {"position": ..., "velocity": ...},
    "collision_probability": ...,
}
rollout(latent, state, action_sequence, dt=...) -> batched trajectory
predict_cost_features(latent, state) -> optional current features
```

MPPI 不依赖 JEPA 内部类。候选维完全 batch 化；只有短 horizon 的递归时间循环。Warm start 使用上一周期 `[a1, ..., aH, aH]`。

## 配置与命令

训练（需要真实数据 split；4090 正式训练时使用）：

```bash
python scripts/train_wam_world_model.py \
  --config configs/wam_mpc_jepa.yaml \
  --splits /path/to/splits.json \
  --encoder-checkpoint /path/to/existing_jepa.pt \
  --output outputs/models/jepa_wam_mpc.pt \
  --device cuda
```

5 episode debug：

```bash
python scripts/eval_wam_mpc.py \
  --config configs/wam_mpc_jepa.yaml \
  --sim-config configs/simulator_mock.yaml \
  --checkpoint outputs/models/jepa_wam_mpc.pt \
  --episodes 5 --debug --device cuda \
  --output-dir outputs/jepa_wam_mpc_debug
```

正式 evaluation：

```bash
python scripts/eval_wam_mpc.py \
  --config configs/wam_mpc_jepa.yaml \
  --sim-config configs/simulator_urbanfly_websocket.yaml \
  --checkpoint outputs/models/jepa_wam_mpc.pt \
  --episodes 100 --device cuda \
  --output-dir outputs/jepa_wam_mpc_formal
```

Sample/horizon ablation 可通过 `--num-samples`、`--horizon`、`--iterations` 覆盖配置。无 checkpoint 时只有显式加入 `--allow-untrained` 才能做接口 smoke，结果状态会写成 `smoke_only_untrained`。

## 已实现指标与调试

- success/collision rate、final goal distance、path length、flight time、average speed、jerk、control smoothness；
- encoder、batched rollout、MPPI optimization、total planning latency 的 mean/median/P95/max；
- GPU peak allocated memory；
- one-step latent/state prediction error；
- 可选保存 candidate actions、predicted trajectories、costs、best trajectory、collision probabilities 与分项 cost；
- episode top-down 图区分 candidates、best planned、executed、goal、UAV 和 obstacles；
- invalid depth、NaN、rollout/MPPI 数值失败统一回退 hover，reset/backend 失败不会中断后续 episode。

## 尚未完成（不能视为已实现）

- 4090 上正式 JEPA dynamics/probe 训练、checkpoint 选择与 3–5 episode 训练后验收；
- Action Proposal Head 与 policy-guided MPPI 64/128 vs vanilla MPPI-256；
- Dreamer adapter、uncertainty-aware cost；
- TD-MPC2 adapter 与 native CEM 公平对照；
- Oracle/ground-truth MPC；
- dynamic obstacle 正式实验、AUROC/AUPRC 与完整 sample/horizon/model-size ablation。
