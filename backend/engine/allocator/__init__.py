"""UrbanFly 任务分配算法模块

主算法: CBBA改进版 (Consensus-Based Bundle Algorithm)
Baseline对比: 匈牙利算法, 贪心, 拍卖, 遗传算法, 市场机制
"""
from .base import BaseAllocator
from .cbba import CBBAAllocator

__all__ = ["BaseAllocator", "CBBAAllocator"]
