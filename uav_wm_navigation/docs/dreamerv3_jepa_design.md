# DreamerV3 / JEPA 双路线设计说明

## 1. 研究边界

两条路线都保留真实 YOPO 作为候选轨迹生成器，并复用同一套 `RiskPrediction`、`RiskReranker` 与 `SafetyFilter`。因此实验比较的是“候选条件世界模型”，不是替换 YOPO 的端到端控制策略。

- **DreamerV3 风格路线**：实现离散随机 RSSM、KL balancing、free nats 和候选动作条件 latent imagination。碰撞/间距/推进/失败头对应 Dreamer 中对未来结果的预测。未加入 actor-critic，因此不称为完整 DreamerV3 智能体。
- **Action-conditioned JEPA 路线**：对历史深度做块状遮挡，以在线上下文编码器产生条件表示，以 EMA 目标编码器产生未来深度 latent，只对实际执行候选施加未来 latent 预测损失；不重建像素。

## 2. 为什么这样落地

YOPO 每轮产生 15 条候选，而日志中只有实际执行轨迹拥有真实未来观测。风险标签可以覆盖全部候选，JEPA 自监督目标则必须只监督执行候选，避免把同一个未来画面错误地分配给未执行候选。DreamerV3 风格 RSSM 可对全部候选从当前后验状态向前想象。

## 3. 训练与评测

```powershell
conda activate uav-wm-nav
python scripts/train_world_model.py --config configs/world_model_dreamerv3.yaml --splits outputs/debug_dataset_v2/splits.json --output outputs/models/dreamerv3_debug.pt
python scripts/train_world_model.py --config configs/world_model_jepa.yaml --splits outputs/debug_dataset_v2/splits.json --output outputs/models/jepa_debug.pt
python scripts/evaluate_open_loop.py --checkpoint outputs/models/dreamerv3_debug.pt --splits outputs/debug_dataset_v2/splits.json --split validation --output outputs/models/dreamerv3_debug_validation.json
python scripts/evaluate_open_loop.py --checkpoint outputs/models/jepa_debug.pt --splits outputs/debug_dataset_v2/splits.json --split validation --output outputs/models/jepa_debug_validation.json
python scripts/evaluate_closed_loop.py --evaluation-config configs/evaluation_dreamer_jepa.yaml --sim-config configs/simulator_mock_open.yaml --planner-config configs/planner_yopo.yaml --dreamerv3-checkpoint outputs/models/dreamerv3_debug.pt --jepa-checkpoint outputs/models/jepa_debug.pt --output-dir outputs/dreamer_jepa_matrix
```

## 4. 正式实验要求

Debug 数据仅用于过拟合和接口验证。正式结论必须使用按 episode/scenario 分组的数据集，并报告：AUROC、AUPRC、Brier、ECE、碰撞率、成功率、最小间距、路径效率、推理时延，以及 RSSM KL 或 JEPA latent prediction error。两条路线应使用同一 YOPO 权重、起终点、交通、天气与随机种子。

## 5. 论文依据

- Hafner et al., *Mastering Diverse Domains through World Models*, arXiv:2301.04104 / Nature 2025.
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243.
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985.
