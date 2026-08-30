"""
风场模型
=======
简化的城市风场模型，用于增加仿真真实性。

包含：
- 全局平均风（constant horizontal component）
- 局部湍流（随机高频扰动）
- 阵风（周期性低频变化）
- 建筑周围的简化涡流效应
"""

import numpy as np
from typing import List, Optional


class WindModel:
    """
    城市风场模型。

    生成随时间、位置变化的三维风向量。
    参考简化的大气边界层模型，叠加城市峡谷效应。
    """

    def __init__(self,
                 global_wind: np.ndarray = None,
                 turbulence_intensity: float = 0.5,
                 gust_amplitude: np.ndarray = None,
                 gust_frequency: float = 0.1,
                 random_seed: int = None):
        """
        Args:
            global_wind: 全局平均风向量 (m/s), 默认 (2.0, 0.0, 1.5)
            turbulence_intensity: 湍流强度标准差 (m/s)
            gust_amplitude: 阵风振幅向量 (m/s), 默认 (1.0, 0.3, 0.5)
            gust_frequency: 阵风频率 (Hz)
            random_seed: 随机种子
        """
        self.global_wind = global_wind if global_wind is not None else np.array([2.0, 0.0, 1.5])
        self.turbulence_intensity = turbulence_intensity
        self.gust_amplitude = gust_amplitude if gust_amplitude is not None else np.array([1.0, 0.3, 0.5])
        self.gust_frequency = gust_frequency
        self.rng = np.random.RandomState(random_seed or 42)

        # 建筑引起的局部风场 (可选)
        self.buildings: List = []
        self.building_wind_enabled = False

    def set_buildings(self, buildings: List):
        """设置建筑列表以启用建筑周围涡流效应"""
        self.buildings = buildings
        self.building_wind_enabled = len(buildings) > 0

    def get_wind(self, position: np.ndarray, time: float) -> np.ndarray:
        """
        获取指定位置和时刻的风向量。

        Args:
            position: (x, y, z) 位置 (m)
            time: 仿真时间 (秒)

        Returns:
            np.ndarray: 三维风向量 (m/s)
        """
        # 1. 全局平均风
        wind = self.global_wind.copy()

        # 2. 阵风 (周期性)
        gust_phase = np.sin(time * self.gust_frequency * 2 * np.pi)
        wind += self.gust_amplitude * gust_phase

        # 3. 湍流 (随机高频)
        turbulence = self.rng.normal(0, self.turbulence_intensity, 3)
        wind += turbulence

        # 4. 建筑周围涡流效应 (简化)
        if self.building_wind_enabled and self.buildings:
            building_wind = self._compute_building_effect(position, time)
            wind += building_wind

        return wind

    def _compute_building_effect(self, position: np.ndarray, time: float) -> np.ndarray:
        """
        计算建筑对局部风场的影响（简化模型）。

        靠近建筑时：
        - 迎风面：风速降低，有上升气流
        - 背风面：产生涡流（周期性方向变化）
        - 街道峡谷：风速沿街道方向加速
        """
        effect = np.zeros(3)

        for building in self.buildings:
            center = building.center
            size = building.bounds_max - building.bounds_min

            # 到建筑的水平距离
            dx = position[0] - center[0]
            dz = position[2] - center[2]
            h_dist = np.sqrt(dx * dx + dz * dz)

            # 影响范围：建筑宽度的2倍
            influence_radius = max(size[0], size[2]) * 2.0

            if h_dist < influence_radius and h_dist > 0.01:
                # 影响强度随距离衰减
                strength = 1.0 - min(h_dist / influence_radius, 1.0)
                strength *= strength  # 二次衰减

                # 背风面涡流
                lee_dir = np.array([dx, 0, dz]) / h_dist  # 建筑→无人机方向
                vortex = np.array([-lee_dir[2], 0, lee_dir[0]])  # 水平垂直方向

                # 周期性涡流
                vortex_strength = np.sin(time * 2.0 + h_dist * 0.5) * strength * 2.0
                effect += vortex * vortex_strength

                # 上升气流 (建筑侧面)
                if position[1] < building.roof_level:
                    updraft = np.array([0, 1, 0]) * strength * 1.5
                    effect += updraft

        return effect

    def get_wind_field(self, x_range: tuple, z_range: tuple, y: float, time: float,
                       resolution: int = 50) -> dict:
        """
        获取某个高度层上的二维风场 (用于可视化)。

        Returns:
            dict with 'x', 'z', 'u', 'w' arrays (u=wind_x, w=wind_z)
        """
        xs = np.linspace(x_range[0], x_range[1], resolution)
        zs = np.linspace(z_range[0], z_range[1], resolution)
        u = np.zeros((resolution, resolution))
        w = np.zeros((resolution, resolution))

        for i, x in enumerate(xs):
            for j, z in enumerate(zs):
                wind = self.get_wind(np.array([x, y, z]), time)
                u[i, j] = wind[0]
                w[i, j] = wind[2]

        return {"x": xs, "z": zs, "u": u, "w": w}

    def reset(self, random_seed: int = None):
        """重置风场状态"""
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
