# UrbanFly v1.0.0

UrbanFly v1.0.0 是首个可安装的 Helsinki 城市无人机数字孪生研究版。

## 本次包含

- 约 1 km × 1 km 的 Helsinki 实景摄影测量运行场景。
- L21 视觉分块、L18 总览/碰撞网格和 0.5 m 高度表面。
- Windows 自包含桌面程序与冻结的 Python 后端。
- Linux 原生后端与一条命令启动的浏览器界面。
- 50 Hz 六自由度仿真、碰撞、风场、通信与多无人机视图。
- RGB-D、全局与第三人称传感器，以及二进制 WebSocket 传输。
- 体积精简的 UrbanFly 观察策略和隐空间世界模型 checkpoint。
- 带确定性安全门的可选 Qwen API 语义观察器。

## 下载选择

- 普通 Windows 用户使用 Setup `.exe`。
- 免安装或开发场景使用 portable ZIP。
- x86-64 Linux 桌面使用 Linux tar.gz。
- 从源码运行时才需要单独下载城市资产 ZIP。
- 训练或独立 QA 时才需要下载数据集 ZIP。

## 已验证验收

- Windows 打包后端健康检查与前端回读：PASS。
- 前端源码测试：20/20 PASS。
- Python 源码测试：72 PASS；另有 1 个已知失败，仅因缺少未随本 Release 分发的旧 CityGS 资产。
- 主数据集：100 episodes、39,767 transitions、100% success、0 collision、0 stale action。
- 1,013.679 m 多途经点世界模型飞行：到达成功、0 collision、0 stale，最终目标距离 2.995 m，最大横向误差 13.970 m。

## Qwen 边界

发布物不含 Qwen checkpoint 或原始权重。设置 `DASHSCOPE_API_KEY` 或 `URBANFLY_QWEN_API_KEY` 后启用 API 推理。Qwen 只负责语义提议，确定性验证始终是强制环节。

## 许可证

UrbanFly 代码采用 MIT License。处理后的 Helsinki 3D 数据采用 CC BY 4.0，并署名 City of Helsinki, City Survey Services。详情见 `THIRD_PARTY_NOTICES.md`。
