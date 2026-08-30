# UrbanFly 单机世界模型导航

本子项目为 UrbanFly 平台提供 YOPO 候选轨迹、世界模型重排、安全过滤、实时控制、数据采集与评测能力。

支持的仿真后端：

- `urbanfly_websocket`：连接 UrbanFly 后端与浏览器 RGB-D 传感器。
- `mock`：用于单元测试和无界面冒烟验证。

外部仿真平台的适配器、配置、运行脚本、实验报告和历史材料已迁出本项目。

## 环境

```powershell
conda activate uav-wm-nav
python -m pip install -e ".[test]"
```

## UrbanFly 实时导航

先从项目根目录启动 UrbanFly 后端和前端：

```powershell
Set-Location D:\AI\UrbanFly
.\scripts\run_server.bat
.\scripts\run_frontend.bat
```

再启动导航：

```powershell
Set-Location D:\AI\UrbanFly\uav_wm_navigation
python scripts\run_yopo_baseline.py `
  --sim-config configs\simulator_urbanfly_websocket.yaml `
  --planner-config configs\planner_yopo.yaml
```

## 测试

```powershell
python -m pytest -q
```

正式数据协议、模型配置与评测配置位于 `configs/`；UrbanFly 在线采集脚本位于 `scripts/collect_urbanfly_world_model_v2_live.py` 和 `scripts/collect_urbanfly_world_model_v3_live.py`。
