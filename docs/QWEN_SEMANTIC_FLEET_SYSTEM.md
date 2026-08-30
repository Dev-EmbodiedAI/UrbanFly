# UrbanFly Qwen 语义多机闭环设计与验收基线

更新时间：2026-08-30

## 1. 结论先行

UrbanFly 当前已经具备一个可运行的分层闭环：Helsinki 实景网格与可计算碰撞场、50 Hz 6DOF、多机通信与避碰、任务执行、动态事件、确定性任务重分配、前端可视化，以及一个不会直接控制飞行器的 Qwen-VL 接口。正式的 300 秒、4 机闭环已经覆盖临时障碍、动态禁飞区、阵风和载货途中单机故障恢复。

此外，本机已完成固定版本 Qwen3-VL-2B 在完整 100 个真实 Helsinki RGB-D 无事件片段上的负例验收：每条 4 帧，共 400 帧，JSON/schema 100/100，通过且误报 0。该结果证明模型在这套全量负例上没有臆造动态事件；**动态事件正例识别、实时 WebSocket 链路和由 Qwen 触发的闭环重规划仍未验收**。正式 300 秒多机 PASS 仍使用确定性仿真事件，不能写成 Qwen 正例感知已经 PASS。

工程成熟度估计（不是论文指标）：

| 范围 | 当前完成度 | 依据 |
|---|---:|---|
| 数字孪生场景与仿真底座 | 约 80% | 1 km Helsinki 实景、碰撞/ESDF、RGB-D、6DOF、动态体、通信、UI、运行健康均已存在 |
| 用户构想的多机语义任务闭环 | 约 65% | 语义协议、安全门、动态约束、3–5 机调度、故障接管、UI 和真实 Qwen 负例推理已接通；正例感知未验收 |
| CityFly 式完整研究基准 | 约 42% | 缺少约 2.1k 语言导航轨迹、seen/unseen/3DGS 分割、VLM 微调与大规模正负例基准 |

## 2. 与 CityFly 和近期工作的关系

CityFly 的核心贡献是 UE5/Cesium/AirSim/3DGS 城市场景、约 2.1k 条语言导航轨迹，以及 Qwen3-VL-8B + LoRA 的多帧 signmark-action 导航。论文最强短/长航线成功率仍约为 30.1% / 17.9%，现实未见 3DGS 场景约 25.9%；因此它更像一个高价值研究基准，而不是已经成熟的飞控系统。[CityFly DOI](https://doi.org/10.1016/j.patcog.2026.114651)

近期工作给出的共同方向是“分层”而不是让大模型接管高频飞控：

- [VLN on the Fly](https://vln-on-the-fly.github.io/) 把量化语言模型、RGB-D 目标投影、局部规划和安全状态机拆开，是本项目最直接的架构参考。
- [NaVILA](https://arxiv.org/abs/2412.04453) 使用高层视觉语言动作和低层运动策略分层。
- [OpenFly](https://arxiv.org/abs/2502.18041) 提供 10 万级轨迹和多种渲染域，说明泛化结论必须依靠规模与域分割，而不是少量演示。
- [AeroVerse](https://arxiv.org/abs/2408.15511) 将场景理解、空间推理、导航、任务规划和运动决策放入统一基准。
- [AutoFly](https://arxiv.org/abs/2602.09657) 探索 RGB+语言直接到速度的 VLA，但其[官方代码库](https://github.com/xiaolousun/AutoFly-VLA)仍分阶段发布，不能把论文方向等同于现成工程能力。
- [Qwen3-VL 官方实现](https://github.com/QwenLM/Qwen3-VL)提供 2B/4B/8B 等尺寸和 OpenAI 兼容服务路径；本机 8 GB RTX 5060 的工程起点设为 2B，而不是直接假定 8B 可实时运行。

UrbanFly 不复制 CityFly 的离散语言导航路线。论文方向更适合定义为：**实景城市数字孪生中，快反应 RGB-D 局部策略与慢语义 VLM 协同的少机群动态任务重规划**。

## 3. 已实现架构

```text
浏览器同步 RGB-D + 飞行状态
              │
              ├── 10 Hz 现有局部策略/控制器 ──> 安全层 ──> 50 Hz 6DOF
              │                    （唯一高频动作路径）
              │
              └── 低频 Qwen3-VL 观察器
                         │ 仅 JSON 语义事件
                         ▼
                  确定性事件安全门
                         │
             临时障碍 / 禁飞 / 天气 / 故障
                         │
                         ▼
             确定性小机群协调与路径约束
                         │
                  静态网格路径再校验
```

关键约束：

1. Qwen 不输出电机、姿态、角速度、速度或航点指令。
2. 每个事件必须有类型、观测时间、来源无人机、ENU 位置、半径、置信度、严重度、TTL 和证据。
3. 低置信度、过期、未来时间、未知来源、非法半径/TTL、缺证据事件全部 fail closed。
4. 外部 Qwen 推理完成后，以**当前仿真时钟**重新检查观测年龄，避免慢推理把旧事件注入当前任务。
5. 动态路径仍必须通过现有 Helsinki 静态碰撞场修复/回读。
6. 单机故障是任务期内永久失效；载货故障位置成为显式恢复取件点，不允许隐式货物转移。

主要实现：

- `backend/agents/semantic_fleet.py`：事件协议、Qwen 客户端、安全门、调度器、审计。
- `backend/agents/simulator_bridge.py`：动态体/风/故障效应、任务接管、路径校验。
- `scripts/run_qwen_semantic_observer.py`：无落盘 RGB-D montage、直接本地 Transformers 或 OpenAI-compatible Qwen 调用、WebSocket 提案；不发送飞行动作。
- `frontend/src/semantic_fleet.js`：仅展示层的事件体积，不进入 RGB-D 传感器层，避免标签泄漏。
- `scripts/benchmark_semantic_fleet.py` 与 `scripts/run_qwen_fleet_closed_loop_qa.py`：正式组件和 6DOF 验收。

## 4. 已完成的正式验证

### 4.1 组件随机 QA

- 5,000 个确定性随机 case，机群规模 3–5。
- 30,038 个任务，覆盖载荷、电量、通信、障碍、禁飞、故障、阻塞和重规划。
- 约束违规 0；确定性失败 0。
- 4,334 个有效事件全部接受；4,334 个低置信度非法事件全部拒绝。
- 调度延迟 P95 5.515 ms，验收门 20 ms。
- 报告：`outputs/qwen_fleet_system_v1/semantic_coordinator_qa.json`。

### 4.2 Helsinki 4 机 6DOF 闭环

- 300.0 仿真秒，15,000 个 50 Hz 物理步。
- 4/4 任务完成，4/4 语义事件接受。
- 载货途中故障恢复 1 次；故障机不再重新进入任务池。
- 碰撞 0；静态环境最小净空 42.302 m。
- 机间最小距离 5.635 m，正式门为不低于 3 m。
- 路径校验失败 0；仿真墙钟约 13.8 s，约 21.7× realtime。
- 报告：`outputs/qwen_fleet_system_v1/helsinki_6dof_qa.json`。

### 4.3 软件回归与界面

- 语义/6DOF/服务端聚焦测试 22 项通过。
- 前端 19 项测试通过，生产构建通过。
- 主应用代码从单一 856 kB 包拆为约 122 kB 应用包、686 kB Three 引擎缓存包和 49 kB 空间索引包；大 Three 包仍是已知构建警告，但不再随业务代码一起失效缓存。
- 全量根测试为 69 PASS / 1 FAIL；唯一失败是缺失的旧 `data/citygs_collision/Residence/collision_geometry.json` 资产，与本次 Helsinki/Qwen 变更无关。不能把全量测试写成 PASS。

### 4.4 本地 Qwen 真实负例 QA

- 模型：官方 `Qwen/Qwen3-VL-2B-Instruct`，固定 revision `89644892e4d85e24eaac8bacfd4f463576704203`。
- 主权重 SHA256：`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`；Hub 文件清单校验 PASS。
- 设备：RTX 5060 Laptop 8 GB，bfloat16；模型加载 2.585 s，峰值 GPU 显存 4,617,452,032 bytes。
- 数据：正式 100 条 Helsinki 数据集的全部 100 个 episode，每条抽取 4 帧真实 RGB-D，共 400 帧，仅在内存生成 montage；五个任务类别各 20 条。
- 结果：JSON 100/100、schema 100/100、原始误报 0、通过安全门误报 0；延迟 mean/P50/P95/max 为 478.36/467.00/512.65/1512.37 ms。
- 报告：`outputs/qwen_fleet_system_v1/qwen_negative_perception_qa.json`。
- `LIMITATION`：这是完整主数据上的负例验收，但没有动态事件正例，仍不能产生事件识别 precision/recall/F1 结论；正例识别仍 `NOT TESTED`。

## 5. 尚未完成，禁止写成结果

| 项目 | 状态 | 成为论文结果前需要什么 |
|---|---|---|
| Qwen3-VL-2B 实际本地推理 | PASS（负例小样本） | 已固定权重并测显存/延迟；仍需扩大到正式标注集 |
| RGB-D 动态事件识别 | NOT TESTED | 标注场景集，按事件算 precision/recall/F1、位置/半径误差 |
| 实时 WebSocket Qwen 观察器 | NOT TESTED | 启动真实前后端与传感器表面，测慢推理、断连、超时和旧帧拒绝 |
| 4 机均使用主 RGB-D 学习策略 | NOT INTEGRATED | 当前 4 机正式 QA 使用后端 6DOF 航点跟踪；主学习策略只做过 5 条单机在线验收 |
| 语言任务/路标导航 | NOT IMPLEMENTED | 指令数据、signmark 或目标落点协议、seen/unseen 切分 |
| 第二城市/3DGS/现实迁移 | NOT TESTED | 至少一个未见城市或 3DGS 域，冻结模型评估 |
| PX4/HIL/实机 | NOT TESTED | 时钟、消息和安全语义适配；本项目不必把它作为毕设最低门槛 |

当前主环境已安装 `transformers 5.16.1`、`accelerate 1.14.0` 和 `qwen-vl-utils 0.0.14`。只保留一个经哈希和 Hub 清单验证的主模型目录 `models/qwen3_vl_2b_instruct/`；下载缓存已删除，不保留逐帧图片或失败 checkpoint。无需依赖 vLLM 即可直接本地推理。

## 6. 下一阶段实验门

只保留一个主 Qwen 模型目录、一个主实验报告目录，不保存逐帧 montage 或失败 checkpoint。

1. **Qwen 离线感知 QA**：至少 250 个独立 RGB-D 片段，五类事件每类正例 30+，另有无事件负例；事件 macro-F1 ≥ 0.80，负例误报率 ≤ 5%，位置中位误差 ≤ 10 m。
2. **实时性与失效 QA**：P95 推理时间、GPU 峰值、超时率；故意注入超时、坏 JSON、旧帧、低置信度和服务断开，全部不得产生飞行动作或绕过安全门。
3. **A/B/C 闭环**：A 为现有局部策略，B 为局部策略+确定性事件，C 为局部策略+Qwen 事件；同一组至少 25 个场景种子，最后再跑 100 episode。
4. **论文指标**：任务完成率、碰撞率、净空、机间距离、重分配延迟、SPL/额外里程、能耗、通信字节、事件 F1、推理 P50/P95/max、安全门拒绝统计。

在第 1–3 项完成以前，准确结论是：**语义多机执行闭环 PASS；真实 Qwen 负例推理 PASS（100 clips / 400 frames）；动态事件正例感知与 Qwen 触发闭环 NOT TESTED。**
