# UrbanFly VLN / Risk World Model

该包负责把 UrbanFly 飞行记录转换为可训练的 VLN 与世界模型数据，并提供状态模型、视觉模型、风险标定和评测工具。

- `schema.py`：版本化 episode 协议。
- `episode_builder.py`：把 UrbanFly 运行目录转换成 episode。
- `risk_world_model.py`：语言条件风险基线。
- `world_model_data.py`：按完整飞行划分训练/验证集并构造未来风险标签。
- `world_model_metrics.py`：AUROC、Brier、ECE 与安全阈值指标。
- `latent_world_model.py`：语言条件动力学集成模型。
- `visual_world_model.py`：RGB-D recurrent world model。
- `direct_visual_world_model.py`：直接预测下一状态与即时回报的 RGB-D 模型。

## 构建 episode

```powershell
python -m urbanfly_vln.episode_builder `
  --run-dir data\my_urbanfly_run `
  --output data\my_urbanfly_run\episode.json
```

## 训练与评估状态模型

```powershell
python scripts\train_latent_world_model.py `
  --run-dir data\urbanfly_vln_demo\run_train `
  --validation-run-dir data\urbanfly_vln_demo\run_validation `
  --output data\urbanfly_vln_demo\world_model\latent_world_model.pt

python scripts\evaluate_latent_world_model.py `
  --checkpoint data\urbanfly_vln_demo\world_model\latent_world_model.pt `
  --run-dir data\urbanfly_vln_demo\run_validation `
  --output data\urbanfly_vln_demo\world_model\latent_world_model.eval.json
```

## 训练视觉模型

```powershell
python scripts\train_visual_world_model.py `
  --data-root data\urbanfly_vln_demo `
  --preset large `
  --batch-size 4 `
  --sequence-length 16 `
  --output data\visual_world_model\visual_rssm.pt
```

数据审计：

```powershell
python scripts\audit_world_model_dataset.py --run-dir data\urbanfly_vln_demo\run_train
```

训练和验证必须按完整飞行划分，不能随机拆分同一轨迹中的相邻帧。
