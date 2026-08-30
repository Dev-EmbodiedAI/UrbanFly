# 第三方许可证与署名

## Helsinki 三维网格

UrbanFly 发布包包含 Helsinki 市中心 `HelsinkiCentral1km` 场景的处理后子集。

- 来源：City of Helsinki, City Survey Services / Helsinki 3D
- 来源页面：<https://www.hel.fi/en/decision-making/information-on-helsinki/maps-and-geospatial-data/helsinki-3d>
- 许可证：Creative Commons Attribution 4.0 International（CC BY 4.0）
- 必须保留的署名：**Data and maps (c) City of Helsinki, City Survey Services**
- UrbanFly 所做处理：坐标归一化、glTF 分块、LOD 生成、碰撞网格派生、高度图/ESDF 诊断与运行时 manifest。

Helsinki 数据仍受 CC BY 4.0 约束，不因 UrbanFly 采用 MIT License 而重新授权。

## Qwen

UrbanFly **不分发** Qwen checkpoint 或原始模型权重。正式集成调用用户自行配置的 OpenAI 兼容 Qwen API。用户应自行负责百炼/DashScope 账户、模型条款、区域 endpoint、费用与 API Key。Key 仅从环境变量读取，绝不会写入报告或源码仓库。

## JavaScript、Python 与桌面运行库

发布包包含 Three.js、three-mesh-bvh、GaussianSplats3D、aiohttp、NumPy、SciPy、trimesh、Pillow、Microsoft WebView2 绑定及 .NET Runtime 等第三方运行库。各组件附带的许可证与声明具有最终效力。

Windows 安装程序使用 Kira（Zhenghan Yang）维护的 Inno Setup 简体中文翻译，来源为 <https://github.com/kira-96/Inno-Setup-Chinese-Simplified-Translation>，采用 MIT License。
