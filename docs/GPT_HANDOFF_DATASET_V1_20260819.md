# UrbanFly Dataset v1 / Real Helsinki Collector 接管总结

更新日期：2026-08-19（Asia/Shanghai）  
项目路径：`D:\AI\UrbanFly`  
用途：将本文件完整交给新的 GPT/Codex session。新 session 不应依赖旧聊天记忆。

## 给下一位 GPT 的接管指令

1. 完整读取 `D:\AI\UrbanFly\AGENTS.md`。
2. 完整读取 `D:\AI\UrbanFly\docs\PROJECT_STATE.md`。
3. 核实本报告引用的 outputs、HDF5、视频和测试文件真实存在。
4. 以代码、实际 outputs 和 HDF5 readback 为最终事实来源。
5. 不要重复已经完成的昂贵实验，除非发现文件与报告矛盾。
6. 不要把两个历史失败 episode 当作合格训练数据。
7. 当前唯一正式下一里程碑是：连续 5 条 Real Helsinki smoke；在其通过前禁止开始 50 条或更大规模采集。

# A. Handoff Verification

## A.1 必读文件

| 项目 | 状态 | 路径 |
|---|---|---|
| `AGENTS.md` | `PASS`，本 session 已完整读取 | `D:\AI\UrbanFly\AGENTS.md` |
| `PROJECT_STATE.md` | `PASS`，本 session 已完整读取并更新 | `D:\AI\UrbanFly\docs\PROJECT_STATE.md` |
| 本总结 | `PASS` | `D:\AI\UrbanFly\docs\GPT_HANDOFF_DATASET_V1_20260819.md` |

## A.2 关键 outputs

| Output | 状态 | 说明 |
|---|---|---|
| `outputs/helsinki_dataset_v1/spatial_split_v1.json` | `PASS` | 可复现空间划分清单 |
| `outputs/helsinki_dataset_v1/spatial_split_v1.png` | `PASS` | Train/Validation/Test 地图 |
| `outputs/helsinki_dataset_v1/real_one_yaw_integrated_20260819/HelsinkiCentral1km_real_smoke_000_building_blocked.h5` | `PASS` | 当前唯一 yaw 合格的真实 Helsinki episode |
| `outputs/helsinki_dataset_v1/real_one_yaw_integrated_20260819/qa_overview.png` | `PASS` | 地图、轨迹、状态、动作和 RGB-D QA |
| `outputs/helsinki_dataset_v1/real_one_yaw_integrated_20260819/dataset_rgbd_telemetry.mp4` | `PASS` | 由同步 HDF5 前视 RGB-D/遥测重建的视频 |
| `outputs/helsinki_dataset_v1/real_one_yaw_integrated_20260819/collection_summary.json` | `PASS` | 单条真实采集摘要 |

## A.3 冻结模块状态

- `PASS` — Global XYZ A*、Helsinki 低空 3D Expert、triangle geometry、Urban-Core sampler、Local Goal Selector 的核心算法未修改。
- `PASS` — 没有重新设计 Global Planner。
- `P0 FIX` — `backend/engine/simulator.py` 的 external-policy yaw bridge 曾被列为冻结控制链的一部分，但真实视频和数据证明存在明确 P0 坐标/控制语义缺陷，且用户明确要求修复，因此只修改了外部策略 yaw 语义与目标保持逻辑。
- `P0 FIX` — `backend/engine/helsinki_frames.py` 增加 canonical FLU yaw-rate 与 backend yaw-rate 的显式转换。
- `LIMITATION` — 工作区根目录没有 `.git`，无法证明 clean/dirty 状态或提供可靠 Git diff。不要声称仓库干净。

# B. Coordinate Convention

## B.1 最终坐标定义

| 层级 | 坐标定义 |
|---|---|
| Helsinki asset/native | EPSG:3879 米制源资产，语义顺序为 `[easting, northing, elevation]` |
| Renderer/backend | `[east, up, south]`，即 `+x=east`、`+y=up`、`+z=south` |
| Canonical world | ENU `[east, north, up]`，右手系 |
| Canonical body/action | FLU `[forward, left, up]`，右手系 |
| Camera | RDF `[right, down, forward]`，光轴为 `+z` |
| Dataset quaternion | XYZW |

显式世界坐标转换：

```text
backend [x, y, z] -> ENU [x, -z, y]
ENU [east, north, up] -> backend [east, up, -north]
```

因此 manifest 中的 `negative_z: north` 是正确的：renderer/backend 的 `-z` 才是 canonical north，并不代表 Dataset 的 ENU `+north` 是负数。

## B.2 Yaw 语义

- Dataset action 顺序：`[forward_mps, left_mps, up_mps, yaw_rate_rps]`。
- canonical 正 yaw-rate：绕 body-up 向左转，即 FLU 右手系 CCW。
- backend 正 yaw 方向与 canonical ENU 正 yaw 相反。
- 映射关系：正 FLU yaw-rate 必须转成负 backend yaw degrees/s。
- 导出角速度 frame：ENU。
- 修复函数：`backend/engine/helsinki_frames.py::body_flu_yaw_rate_to_backend_degrees`。

## B.3 Dataset coordinate metadata

HDF5 metadata 已记录：

- `world_frame=ENU`
- `backend_world_frame=HELSINKI_RENDERER_Y_UP`
- `body_frame=FLU`
- `camera_frame=RDF`
- `quaternion_order=xyzw`
- `linear_velocity_frame=ENU`
- `angular_velocity_frame=ENU`
- `linear_acceleration_frame=ENU`
- `action_frame=FLU`
- `action_order=[forward_mps,left_mps,up_mps,yaw_rate_rps]`
- `yaw_rate_positive=left_ccw_about_body_up`

# C. Dataset v1 Schema

Canonical transition：

```text
(observation_t, state_t, goal_t, local_goal_t, action_t)
    ->
(observation_t+1, state_t+1, reward_t, safety/termination targets)
```

## C.1 Episode-level

- episode/scene/task：`episode_id`、`scene_id`、`scene_seed`、`task_type`、`collection_mode`
- geometry：`start_world`、`global_goal_world`、`global_route_world`
- split：`spatial_split`、`urban_region_type`
- outcome：`num_steps`、`success`、`collision`、`timeout`、`failure_reason`
- metrics：`path_length`、`flight_time`
- expert evidence：`expert_planning_result`
- coordinate metadata 和 transition timeline

## C.2 Per-step HDF5 groups

- `observations/`：`rgb_front`、`depth_front`、`depth_valid`
- `camera/`：`intrinsics`、`extrinsics_world_enu_from_camera_rdf`
- `state/`：`position_world`、`orientation_xyzw`、`linear_velocity`、`angular_velocity`、`linear_acceleration`
- `next_state/`：与 `state/` 对应的五项真实下一状态
- `goal/`：`global_goal_world`、`local_goal_world`、`local_goal_body`
- `route/`：`progress`、`remaining_distance`
- `actions/`：`commanded_body_flu`、`executed_body_flu`
- `labels/`：`reward`、`collision`、`minimum_clearance`、`success`、`terminated`、`truncated`、controller/safety flags、`safety_intervened`、`stale_action`
- `timestamps/`：`sim`、`sensor`、`action`、`next_sim`、`wall`、`dt`

## C.3 Action semantics

- `action_commanded_t`：Expert/策略请求的 physical FLU 动作。
- `action_executed_t`：backend safety/controller 实际接受并用于 `state_t -> state_t+1` 的 physical FLU 动作。
- 未来 dynamics/WAM 训练默认使用 `action_executed_t`。
- 当前 HDF5 同时保留 commanded 与 executed，不允许混用。

## C.4 Timestamp semantics

真实 timeline：

```text
synchronized RGB-D + state_t
-> privileged route + LocalGoalSelector
-> action_commanded_t
-> backend safety/controller
-> action_executed_t
-> 6-DOF integration
-> synchronized RGB-D + state_t+1
```

- `dt = next_sim - sim`，使用实际 sim time。
- 不假设 nominal 10 Hz 等于实际采集频率。
- 当前 yaw 合格 episode 的 mean/P95 `dt` 为 `0.470/0.600 s`。
- `action_timestamp` 必须满足 `state_t <= action_t < state_t+1`。

## C.5 RGB / Depth

- 前视 RGB：`uint8`，shape `T x 90 x 160 x 3`，RGB color order。
- metric depth：`float32`，shape `T x 90 x 160`，单位 meter。
- valid mask：`uint8`，shape `T x 90 x 160`。
- intrinsics 和 camera-to-world ENU/RDF extrinsics 均逐帧保存。
- 禁止把 metric depth 当普通 8-bit JPEG。

# D. Spatial Split

规则：沿 backend-z/canonical-north 建立带 20 m guard band 的空间隔离区；episode 的完整 global route 必须全部位于同一 buffered split；禁止 random frame split 和 trajectory 跨 split。

| Split | Canonical north interior (m) | Area (m²) | Urban area (m²) | Mean urban density | Dense core (m²) | Endpoint cells |
|---|---:|---:|---:|---:|---:|---:|
| Train | `[-280, 80]` | 360,000 | 308,000 | 0.7517 | 16,725 | 12,320 |
| Validation | `[-480, -320]` | 160,000 | 67,950 | 0.2386 | 9,825 | 2,718 |
| Test | `[120, 280]` | 160,000 | 91,050 | 0.3226 | 21,300 | 3,642 |

canonical north `[300,500]` 对应的开放水域/北侧带被排除，避免把 test 人为分到水域或城市边缘。

地图：`D:\AI\UrbanFly\outputs\helsinki_dataset_v1\spatial_split_v1.png`

# E. Collector Integration

| 组件 | 实际实现 | 状态 |
|---|---|---|
| Simulator | UrbanFly browser WebGL RGB-D + backend 6-DOF | `PASS` |
| Scene | HelsinkiCentral1km 真实 mesh | `PASS` |
| Planner/Expert | frozen Helsinki low-altitude 3-D privileged expert / bounded XYZ A* | `PASS` |
| Sampler | frozen geometry-derived Helsinki urban sampler | `PASS` |
| Local Goal | frozen `LocalGoalSelector`，20 m lookahead | `PASS` |
| Collision source | Helsinki L18 triangle mesh oracle | `PASS` |
| Writer | HDF5 `urbanfly-helsinki-dataset-v1` | `PASS` |
| MockCandidatePlanner | 未使用 | `PASS: NOT USED` |
| Straight-line fallback | fail-closed，未作为正式 smoke 结果 | `PASS: NOT USED` |

Collector entry point：

```text
uav_wm_navigation/scripts/collect_helsinki_dataset_v1.py
```

Writer/validator：

```text
uav_wm_navigation/src/uav_wm_navigation/data/helsinki_dataset_v1.py
```

# F. 5-Episode Real Smoke

## F.1 已完成的 yaw 合格真实 episode

| Episode | Task | Steps | Success | Collision | Min clearance | Mean dt | Action alignment | Local Goal QA | HDF5 readback |
|---|---|---:|---|---|---:|---:|---|---|---|
| `HelsinkiCentral1km_real_smoke_000_building_blocked` | building_blocked | 105 | true | false | 5.248 m | 0.470 s | PASS | PASS | PASS |

补充指标：

- start ENU：`[337.5, -77.5, 9.0723]`
- goal ENU：`[327.5, 52.5, 7.2653]`
- path length：`143.669 m`
- flight time：`49.4 s`
- mean/max speed：`2.897/3.569 m/s`
- RGB/depth/action/state count：`105/105/105/105`
- yaw start/end/net：`63.435/110.121/+46.686 deg ENU`
- commanded yaw integral：`+117.094 deg`
- yaw command/observed direction：`PASS`
- actual body-forward mean：`+2.716 m/s`
- backward fraction all/second half：`0.0%/0.0%`

## F.2 尚未完成

- `NOT TESTED` — 另外四种 deterministic task 尚未在同一 collector process 中连续采完。
- `NOT TESTED` — 五条之间的自动 reset/flush/reconnect 可靠性。
- 因此“连续 5 Episode Real Smoke”整体仍是 `NOT TESTED`，不能标记 `PASS`。

# G. Dataset QA

以下 QA 结论只适用于当前唯一 yaw 合格 episode：

| 检查项 | 结果 |
|---|---|
| RGB dtype/shape/frame count | `PASS` |
| Metric depth dtype/unit/valid mask | `PASS` |
| State/next_state count and finite values | `PASS` |
| Goal finite and constant global goal | `PASS` |
| Local Goal progression/corridor/body transform | `PASS` |
| Action commanded | `PASS` |
| Action executed and state transition semantics | `PASS` |
| Timestamp monotonicity | `PASS` |
| Sensor/state synchronization | `PASS` |
| Action timestamp within transition | `PASS` |
| Quaternion norm | `PASS` |
| Collision labels | `PASS` |
| Triangle minimum clearance | `PASS` |
| Termination/success/failure consistency | `PASS` |
| Spatial split leakage | `PASS` |
| Yaw-rate/orientation direction and response | `PASS` |
| HDF5 independent readback | `PASS` |

Focused regression：

```text
10 passed in 0.91s
```

覆盖文件：

```text
tests/test_helsinki_coordinate_frames.py
tests/test_external_world_model_policy.py
uav_wm_navigation/tests/test_helsinki_dataset_v1.py
```

视频 QA：

- MP4 共 105 帧，960x420，按真实 49.4 s 仿真时间重放。
- OpenCV 成功解码首帧、中帧、末帧。
- MP4 是同步 HDF5 的前视 RGB + metric depth + telemetry replay，不是浏览器 UI 屏幕录制。

## G.1 当前有效文件与哈希

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| `HelsinkiCentral1km_real_smoke_000_building_blocked.h5` | 2,783,482 | `E5EE8E2037C267FF8035A191867B445850D1ED1517F456CFDFB1527E8AF85952` |
| `qa_overview.png` | 1,440,349 | `D127C5EAA9E8F47C8F26AB73D3E1963448396CE95B3D9B9D693A0A1E4126EC9B` |
| `dataset_rgbd_telemetry.mp4` | 2,214,464 | `E368A84E680B862C044DF4CDE346E04F6384A547223D614900B4163ECC5474DE` |

## G.2 必须隔离的历史失败数据

以下文件真实存在，但当前 validator 会以 `yaw_rate_orientation_consistency` 拒绝；不得用于训练、验收或合格视频证明：

1. `outputs/helsinki_dataset_v1/real_one_20260819_retry2/`
   - `FAIL` — FLU yaw-rate 到 backend yaw 的符号反向。
   - 视频后半段看似倒退/侧退，实际统计也证明机头和速度语义不一致。
2. `outputs/helsinki_dataset_v1/real_one_yaw_fixed_20260819/`
   - `FAIL` — 符号已修正，但每个策略 action 都重置 desired-yaw，只产生微小航向 nudging，整体响应不足。

最终修复：

- 正 FLU yaw-rate 显式映射到负 backend yaw-rate。
- `desired_yaw_backend_degrees` 跨物理步和策略 action 持续保持并按实际 `dt` 积分。
- telemetry 同时区分 canonical yaw-rate、backend yaw-rate 与 desired backend yaw。
- validator 增加 aggregate yaw command/orientation consistency。

真实浏览器/backend yaw probe：12 个 action、实际 sim elapsed `5.5 s`、command integral `+247.5 deg`、observed ENU yaw `+157.5 deg`、response ratio `0.636`、positive executed yaw fraction `100%`，结果 `PASS`。

# H. Remaining Blockers

## P0

- 当前单条合格链路没有已知未修复 P0。
- 如果下一条出现 yaw 方向、时间对齐、RGB-D 同步或 action/state off-by-one，必须 fail-closed，不能通过放宽 validator 掩盖。

## P1

1. 必须在同一个 collector process 中完成 5 个连续真实 Helsinki episodes。
2. 必须覆盖 deterministic 五类：building-blocked、street-canyon、rooftop-to-ground、ground-to-rooftop、rooftop-to-rooftop。
3. 必须逐条通过 HDF5 readback、action alignment、Local Goal QA、yaw consistency 和可视化抽检。
4. 必须验证 episode 间 automatic reset/flush/reconnect。
5. 实际 transition 频率显著低于 nominal 10 Hz；虽然真实 `dt` 已正确记录，但应在五条报告中继续统计 mean/P95 `dt`，不要伪称固定 0.1 s。

## P2

1. 五条通过后才评估是否运行 50-episode Dataset QA。
2. 50 条需要审计 task/spatial/action/velocity/clearance/dt/local-goal 分布。
3. 工作区缺失 `.git` metadata，代码 provenance 仍是限制。
4. 500–1000 episode pilot、WAM、V-JEPA、MPPI、Dreamer、TD-MPC2、LeRobot 迁移均不属于当前下一步。

# I. Final Verdict

## NOT READY FOR 50-EPISODE DATASET QA

理由：当前已有一条真正通过 yaw、时间同步、HDF5 readback 和视频 QA 的 Real Helsinki episode，证明单条链路可用；但原始验收要求的“同一 collector process 连续 5 条、自动 reset、五种任务逐条 PASS”尚未完成。

只有以下条件全部满足后，才能把结论改为 `READY FOR 50-EPISODE DATASET QA`：

1. 五条真实 Helsinki episode 连续完成；
2. 五条均 success、无碰撞或有明确合规结果；
3. 五条 action/state alignment 均 PASS；
4. 五条 Local Goal QA 均 PASS；
5. 五条 yaw-rate/orientation consistency 均 PASS；
6. 五条 HDF5 readback 均 PASS；
7. 自动 reset/flush/reconnect 通过；
8. 至少一条新 episode 的 QA 图和前视 RGB-D/telemetry 视频通过人工抽检。

## 下一步建议命令

先确认 backend 和真实 Chrome Helsinki WebGL 前端已运行、网格加载完成，然后执行：

```powershell
$env:PYTHONPATH='uav_wm_navigation/src'
python uav_wm_navigation/scripts/collect_helsinki_dataset_v1.py `
  --episodes 5 `
  --output-dir outputs/helsinki_dataset_v1/real_smoke5_yaw_qualified_20260819
```

该命令可能耗时较长。不要并发启动第二个 collector，不要让浏览器页面失焦/断开，不要在五条完成前把部分临时 HDF5 声称为整体 PASS。

