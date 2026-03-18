"""
伪Cluster数据结构 (PseudoCluster)

与MissileCluster 100%接口对齐！
但内部导弹分布极其分散，导致专家先验"撕裂"
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class PseudoCluster:
    """
    伪Cluster - 与真实MissileCluster 100%接口对齐！

    字段完全一致，但内部导弹分布极其分散

    关键区别：
    - 真实Cluster: 导弹在空间和时间上聚集
    - 伪Cluster: 导弹可能分布在四个不同方向！

    例如：5架UAV，20枚导弹，轮询分配
    UAV1: [M1, M6, M11, M16]  ← 可能分布在整个战场！
    """
    cluster_id: int
    missile_ids: List[int]
    missile_configs: List[Dict]
    centroid: np.ndarray           # 计算得到（可能在"无人区"）
    earliest_arrival: float        # 时间窗起点
    latest_arrival: float          # 时间窗终点（可能很宽！）
    threat_priority: float         # 威胁优先级

    @property
    def n_missiles(self) -> int:
        """返回导弹数量"""
        return len(self.missile_ids)

    def __repr__(self):
        return (f"PseudoCluster(id={self.cluster_id}, "
                f"missiles={self.missile_ids}, "
                f"centroid={self.centroid}, "
                f"time_window=[{self.earliest_arrival:.1f}, {self.latest_arrival:.1f}])")
