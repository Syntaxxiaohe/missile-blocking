"""
M1 均匀分配器 (Naive Allocator)

策略：轮询分配（Round-Robin），完全不考虑空间距离！

例如：5架UAV，20枚导弹
UAV1: [M1, M6, M11, M16]  ← 可能在四个不同方向！
UAV2: [M2, M7, M12, M17]
UAV3: [M3, M8, M13, M18]
UAV4: [M4, M9, M14, M19]
UAV5: [M5, M10, M15, M20]

关键：创建PseudoCluster，与真实MissileCluster接口100%对齐！
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config.params import (
    UAV_PARAMS, MISSILE_SPEED, CAP_RADIUS, CAP_HEIGHT
)
from .pseudo_cluster import PseudoCluster


class NaiveAllocator:
    """
    M1 均匀分配器 - 轮询分配导弹给UAV

    核心特点：
    1. 策略简单：missile i → UAV (i % n_uavs)
    2. 创建PseudoCluster，接口与真实Cluster对齐
    3. 先验"撕裂"：质心可能在"无人区"，时间窗极宽
    """

    def __init__(self, uav_group: List[int], missile_configs: List[Dict],
                 wind_vector: np.ndarray = None):
        """
        Args:
            uav_group: UAV编号列表 [1, 2, 3, 4, 5]
            missile_configs: 导弹配置列表
            wind_vector: 风场向量（保持接口一致，不使用）
        """
        self.uav_group = uav_group
        self.missile_configs = missile_configs
        self.n_uavs = len(uav_group)
        self.n_missiles = len(missile_configs)
        self.wind_vector = wind_vector

        # 预计算UAV位置
        self.uav_positions = {uav_id: UAV_PARAMS["init_pos"][uav_id] for uav_id in uav_group}

        # 伪Cluster列表
        self.pseudo_clusters: List[PseudoCluster] = []

        # 分配结果 {uav_id: cluster_id}
        self.assignment = None

        # 执行均匀分配
        self._naive_assignment()

    def _naive_assignment(self):
        """
        轮询分配：missile i → UAV (i % n_uavs)

        然后为每个UAV创建PseudoCluster
        """
        # Step 1: 轮询分配导弹给UAV
        uav_missiles = {uav_id: [] for uav_id in self.uav_group}

        for i, missile_config in enumerate(self.missile_configs):
            uav_id = self.uav_group[i % self.n_uavs]
            uav_missiles[uav_id].append(missile_config)

        # Step 2: 为每个UAV创建PseudoCluster
        self.assignment = {}

        for cluster_id, uav_id in enumerate(self.uav_group):
            missiles = uav_missiles[uav_id]
            if not missiles:
                continue

            # 计算伪Cluster属性
            missile_ids = [m["id"] for m in missiles]

            # 质心（可能是"无人区"！）
            positions = np.array([m["position"] for m in missiles])
            centroid = np.mean(positions, axis=0)

            # 时间窗（可能很宽！）
            arrival_times = []
            for m in missiles:
                pos = np.array(m["position"])
                dist = np.linalg.norm(pos[:2])  # 水平距离
                arrival_time = dist / MISSILE_SPEED
                arrival_times.append(arrival_time)

            earliest_arrival = min(arrival_times)
            latest_arrival = max(arrival_times)

            # 威胁优先级（越早到达越高）
            threat_priority = 1.0 / (earliest_arrival + 1.0)

            # 创建PseudoCluster
            pseudo_cluster = PseudoCluster(
                cluster_id=cluster_id,
                missile_ids=missile_ids,
                missile_configs=missiles,
                centroid=centroid,
                earliest_arrival=earliest_arrival,
                latest_arrival=latest_arrival,
                threat_priority=threat_priority
            )

            self.pseudo_clusters.append(pseudo_cluster)
            self.assignment[uav_id] = cluster_id

    def get_assignment(self) -> Dict[int, int]:
        """返回分配结果 {uav_id: cluster_id}"""
        return self.assignment

    def get_clusters(self) -> List[PseudoCluster]:
        """返回伪Cluster列表"""
        return self.pseudo_clusters

    def get_expert_genes(self, uav_id: int) -> Dict:
        """
        生成专家先验基因（与SpatioTemporalAllocator接口对齐！）

        关键：先验存在但"撕裂"！
        - 方向指向质心（但质心可能在"无人区"）
        - 时间窗极宽（可能15秒以上）
        """
        if uav_id not in self.assignment:
            return None

        cluster_id = self.assignment[uav_id]
        cluster = self.pseudo_clusters[cluster_id]

        # UAV初始位置
        uav_pos = self.uav_positions[uav_id]

        # 飞向质心的方向（但质心可能在"无人区"！）
        direction = cluster.centroid - uav_pos
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-6:
            direction = direction / dir_norm

        # 时间窗（极宽！）
        t_min = max(5.0, cluster.earliest_arrival - 20.0)
        t_max = cluster.latest_arrival + 5.0

        return {
            "direction": direction,
            "time_window": (t_min, t_max),
            "cluster_centroid": cluster.centroid,
            "n_missiles": cluster.n_missiles
        }

    def print_assignment(self):
        """打印分配结果"""
        print("\n" + "=" * 60)
        print("M1 均匀分配结果 (Naive Allocation)")
        print("=" * 60)

        for uav_id, cluster_id in self.assignment.items():
            cluster = self.pseudo_clusters[cluster_id]
            print(f"\nUAV{uav_id} → PseudoCluster {cluster_id}")
            print(f"  导弹: {cluster.missile_ids}")
            print(f"  质心: ({cluster.centroid[0]:.0f}, {cluster.centroid[1]:.0f}, {cluster.centroid[2]:.0f})")
            print(f"  时间窗: [{cluster.earliest_arrival:.1f}s, {cluster.latest_arrival:.1f}s] "
                  f"(宽度: {cluster.latest_arrival - cluster.earliest_arrival:.1f}s)")

        print("\n" + "=" * 60)
        print("⚠️ 注意：质心可能在'无人区'，时间窗可能极宽！")
        print("   → 专家先验'撕裂'，信息利用效率低")
        print("=" * 60)
