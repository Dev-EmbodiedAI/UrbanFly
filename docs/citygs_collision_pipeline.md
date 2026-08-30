# CityGS 实景城市的可计算碰撞链路

## 当前边界

Residence 的 3DGS 负责照片级外观；3DGS 椭球本身不直接作为物理碰撞体。运行时使用与它处于同一米制、Y-up 局部坐标系的封闭网格和距离场。当前无人机仍采用 3DOF 平移 + 偏航的运动学模型，尚未接入电机、桨叶、姿态环和气动模型。

## 已生成的资产

目录：`data/citygs_collision/Residence/`

| 文件 | 用途 | 当前规模 |
|---|---|---|
| `city_collision.glb` | 封闭静态碰撞网格 | 891,024 三角面，watertight |
| `global_esdf.npz` | 全局有符号距离场 | 500 × 170 × 500，1 m，float16 |
| `local_collision_sparse.npz` | CityGS 屋檐、立面、植被细节 | 1,541,271 个 0.25 m 表面体素 |
| `collision_geometry.json` | 网格、ESDF 和切片元数据 | 可由前端直接读取 |
| `alignment_report.json` | 全局/局部层一致性抽样报告 | 20 万点确定性抽样 |

全域 0.25 m 稠密距离场需要约 27.2 亿个样本，单通道 float16 也约 5.44 GB，加载和随机查询都不合算。因此运行时使用 16³ 体素、边长 4 m 的惰性 LRU 分块，只实体化无人机或规划器真正访问的区域；256 块上限约 2.1 MB。最终碰撞净空取：

```text
min(全局 1 m 有符号 ESDF, 局部 0.25 m CityGS 表面距离)
```

## 运行时安全链

1. 规划器利用全局高度/占据层产生路径。
2. 路径分配给无人机前，所有折线段以安全半径做扫掠检测。
3. 穿透段被提升到沿途建筑包络之上的安全高度，再做一次全路径复核。
4. 每个运动学积分步检查从上一位置到新位置的完整线段，采样间隔不大于 0.125 m，防止“终点没撞、但中间穿墙”。
5. 地面、建筑内部和 500 m 地图边界外的 ESDF 都是负距离，默认判为不可飞。

核心实现：

- `backend/engine/collision.py`
- `backend/engine/simulator.py`
- `backend/server/server.py`

## 重建与验证命令

```powershell
python scripts/build_citygs_collision_geometry.py
python scripts/audit_citygs_collision_alignment.py
python -m pytest tests/test_citygs_collision_field.py -q
python scripts/smoke_citygs_twin.py
```

前端左侧“几何验证”区域可以打开：

- 橙色封闭碰撞网格；
- 30 m / 60 m ESDF 切片。

这两个调试层使用独立的末端叠加渲染通道，避免被 3DGS 合成结果覆盖。

## 后续接入真实飞行动力学时

不要改变城市资产和距离查询接口。将运动学积分器替换为飞控/动力学适配器即可：

1. 轨迹规划输出位置、速度、加速度和偏航参考；
2. UrbanFly 六自由度飞控仿真产生真实机体位姿；
3. 每个物理子步用机体包围球/胶囊对同一 `HierarchicalStaticCollisionMap` 做扫掠查询；
4. 接触需要刚体反作用时，把 `city_collision.glb` 导入 Unreal/Unity/Bullet/PhysX，ESDF 继续供规划器使用；
5. 保持 `world = Y-up local metric` 的唯一坐标契约，适配器只在边界处转换 NED/ENU。

对毕业设计最稳妥的升级顺序是：先接 PX4 SITL 的位置控制，再加风扰和传感器延迟，最后才做刚体接触。这样能够分别验证场景、规划、飞控三个层次，而不是一次把所有误差混在一起。
