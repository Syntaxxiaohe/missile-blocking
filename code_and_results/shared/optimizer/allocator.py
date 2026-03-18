"""
时空目标分配器 (Spatio-Temporal Allocator) - V2 支持饱和攻击

核心功能：
  1. 时空聚类：将来袭导弹按空间距离和时间差打包成Cluster
  2. 构建UAV-Cluster代价矩阵
  3. 使用匈牙利算法进行UAV-Cluster最优匹配
  4. 生成专家先验基因（精确航向 + 收窄时间窗）

V2 新增：
  - cluster_missiles(): 时空聚类算法
  - 支持非对称分配（1架UAV可对抗多枚导弹）
  - Cluster内最多3枚导弹（受限于UAV载弹量）
"""

import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# ===== 相对导入（shared模块内部）=====
from ..config.params import (
    UAV_PARAMS, SMOKE_PAYLOAD_PER_UAV, MISSILE_SPEED,
    CAP_RADIUS, CAP_HEIGHT
)


@dataclass
class MissileCluster:
    """导弹簇数据结构"""
    cluster_id: int
    missile_ids: List[int]
    missile_configs: List[Dict]
    centroid: np.ndarray  # 空间质心
    earliest_arrival: float  # 最早到达时间
    latest_arrival: float  # 最晚到达时间
    threat_priority: float  # 威胁优先级（越早到达越高）

    @property
    def n_missiles(self) -> int:
        return len(self.missile_ids)


class SpatioTemporalAllocator:
    """
    时空目标分配器 V2 - 支持饱和攻击

    核心流程：
    1. 时空聚类：将N枚导弹打包成M个Cluster (M <= 5)
    2. 匈牙利匹配：将5架UAV分配给M个Cluster
    3. 专家先验：为每架UAV生成针对其Cluster的优化参数
    """

    # 聚类参数（类级别默认值，可通过构造函数覆盖以支持灵敏度分析）
    SPATIAL_THRESHOLD = 2000.0  # 空间距离阈值 (m)
    TEMPORAL_THRESHOLD = 5.0    # 时间差阈值 (s)
    MAX_MISSILES_PER_CLUSTER = 3  # 每个Cluster最多导弹数（受限于UAV载弹量）

    def __init__(self, uav_group: List[int], missile_configs: List[Dict],
                 wind_vector: np.ndarray = None,
                 spatial_threshold: float = None,
                 temporal_threshold: float = None):
        """
        Args:
            uav_group:          UAV编号列表 [1, 2, 3, 4, 5]
            missile_configs:    导弹配置列表
            wind_vector:        风场向量
            spatial_threshold:  空间聚类距离阈值 (m)，None 时使用类默认值 2000.0
            temporal_threshold: 时间聚类阈值 (s)，None 时使用类默认值 5.0
        """
        self.uav_group = uav_group
        self.missile_configs = missile_configs
        self.n_uavs = len(uav_group)
        self.n_missiles = len(missile_configs)
        self.wind_vector = wind_vector if wind_vector is not None else np.array([5.0, -3.0, 0.0])

        # 支持实例级别覆盖，向后完全兼容
        self.Dc = spatial_threshold  if spatial_threshold  is not None else self.SPATIAL_THRESHOLD
        self.Tc = temporal_threshold if temporal_threshold is not None else self.TEMPORAL_THRESHOLD

        # 预计算UAV位置
        self.uav_positions = {uav_id: UAV_PARAMS["init_pos"][uav_id] for uav_id in uav_group}

        # 聚类结果
        self.clusters: List[MissileCluster] = []
        self.n_clusters = 0

        # 匹配结果（V2: UAV -> Cluster）
        self.assignment = None  # {uav_id: cluster_id}
        self.uav_cluster_map = None  # {uav_id: MissileCluster}
        self.cost_matrix = None

        # 时间可行性过滤
        self.time_infeasible_missiles = []  # 物理上无法拦截的导弹
        self.uav_speed = 100.0  # UAV平均速度 (m/s)
        self.time_margin = 10.0  # 最小时间余量 (s)

    def _check_time_feasibility(self, missile_config: Dict) -> Tuple[bool, float]:
        """
        检查导弹的时间可行性（基于拦截点，而非初始位置）

        核心逻辑：
        - UAV不需要飞到导弹初始位置
        - 只需要飞到导弹路径上的拦截点（更近）
        - 拦截点 = 导弹在 t_intercept 时刻的位置
        - t_intercept 应该 < arrival_time

        Args:
            missile_config: 导弹配置

        Returns:
            (is_feasible, best_margin): 是否可行，最佳时间余量
        """
        m_pos = np.array(missile_config["position"])
        m_vel = np.array(missile_config["velocity"])
        arrival_time = missile_config["arrival_time"]

        best_margin = -float('inf')
        best_uav = None

        for uav_id in self.uav_group:
            uav_pos = np.array(self.uav_positions[uav_id])

            # 遍历可能的拦截时间点（从5s到arrival_time-5s）
            # 找到UAV能到达的最近拦截点
            for t_intercept in np.linspace(5.0, max(arrival_time - 5.0, 6.0), 20):
                # 导弹在 t_intercept 时刻的位置
                intercept_point = m_pos + m_vel * t_intercept

                # UAV到拦截点的距离
                dist = np.linalg.norm(intercept_point[:2] - uav_pos[:2])

                # UAV飞行时间
                uav_fly_time = dist / self.uav_speed

                # 时间余量 = 拦截时刻 - UAV飞行时间
                margin = t_intercept - uav_fly_time

                if margin > best_margin:
                    best_margin = margin
                    best_uav = uav_id

            # 额外检查：如果UAV提前到达目标区域等待（防御阵型）
            # 可以在导弹到达前就部署好烟幕
            # 这种情况下，UAV只需要在导弹到达前到达目标附近
            dist_to_origin = np.linalg.norm(uav_pos[:2])
            uav_fly_to_origin = dist_to_origin / self.uav_speed
            margin_at_origin = arrival_time - uav_fly_to_origin - 5.0  # 留5s投放烟幕

            if margin_at_origin > best_margin:
                best_margin = margin_at_origin

        is_feasible = best_margin >= self.time_margin
        return is_feasible, best_margin

    def cluster_missiles(self) -> List[MissileCluster]:
        """
        时空聚类：将N枚导弹打包成M个Cluster

        聚类标准：
        - 空间距离 < SPATIAL_THRESHOLD (2000m)
        - 到达时间差 < TEMPORAL_THRESHOLD (5s)
        - 每个Cluster最多 MAX_MISSILES_PER_CLUSTER (3) 枚导弹
        - **时间可行性**: UAV飞行时间 < 导弹到达时间 - 10s

        如果Cluster数量超过UAV数量，选择威胁度最高的N个。

        Returns:
            clusters: 导弹簇列表
        """
        if len(self.missile_configs) == 0:
            return []

        # ========== 新增：时间可行性预过滤 ==========
        feasible_missiles = []
        infeasible_missiles = []

        for m in self.missile_configs:
            is_feasible, margin = self._check_time_feasibility(m)
            if is_feasible:
                feasible_missiles.append(m)
            else:
                infeasible_missiles.append((m, margin))

        if infeasible_missiles:
            print(f"\n[时间可行性过滤] 检测到 {len(infeasible_missiles)} 枚物理上无法拦截的导弹:")
            for m, margin in infeasible_missiles:
                print(f"  M{m['id']}: 到达时间={m['arrival_time']:.1f}s, 最佳余量={margin:.1f}s < {self.time_margin}s")
            print(f"  这些导弹将被提前放弃，不参与聚类！")

        self.time_infeasible_missiles = [m for m, _ in infeasible_missiles]

        # 只对可行的导弹进行聚类
        if len(feasible_missiles) == 0:
            print("\n[警告] 所有导弹都不可行！")
            self.clusters = []
            self.n_clusters = 0
            return []

        # 按到达时间排序（威胁度优先）
        sorted_missiles = sorted(feasible_missiles, key=lambda m: m["arrival_time"])

        # 初始化：每枚导弹先属于自己的Cluster
        clusters = []
        used = set()

        for i, m1 in enumerate(sorted_missiles):
            if m1["id"] in used:
                continue

            # 创建新Cluster
            cluster_missiles = [m1]
            used.add(m1["id"])

            pos1 = np.array(m1["position"])
            t1 = m1["arrival_time"]

            # 寻找可以合并的导弹
            for j, m2 in enumerate(sorted_missiles):
                if m2["id"] in used:
                    continue
                if len(cluster_missiles) >= self.MAX_MISSILES_PER_CLUSTER:
                    break

                pos2 = np.array(m2["position"])
                t2 = m2["arrival_time"]

                # 与Cluster内任意导弹满足条件即可
                can_merge = False
                for cm in cluster_missiles:
                    cm_pos = np.array(cm["position"])
                    cm_t = cm["arrival_time"]
                    if (np.linalg.norm(cm_pos[:2] - pos2[:2]) < self.Dc and
                        abs(cm_t - t2) < self.Tc):
                        can_merge = True
                        break

                if can_merge:
                    cluster_missiles.append(m2)
                    used.add(m2["id"])

            # 计算Cluster属性
            positions = np.array([m["position"] for m in cluster_missiles])
            centroid = np.mean(positions, axis=0)
            arrival_times = [m["arrival_time"] for m in cluster_missiles]
            earliest = min(arrival_times)
            latest = max(arrival_times)
            threat_priority = earliest  # 越早威胁越大

            cluster = MissileCluster(
                cluster_id=len(clusters) + 1,
                missile_ids=[m["id"] for m in cluster_missiles],
                missile_configs=cluster_missiles,
                centroid=centroid,
                earliest_arrival=earliest,
                latest_arrival=latest,
                threat_priority=threat_priority
            )
            clusters.append(cluster)

        # 如果Cluster数量超过UAV数量，选择威胁度最高的
        if len(clusters) > self.n_uavs:
            print(f"\n[聚类警告] Cluster数量({len(clusters)}) > UAV数量({self.n_uavs})")
            print(f"  将选择威胁度最高的{self.n_uavs}个Cluster,其余放弃！")
            clusters.sort(key=lambda c: c.threat_priority)
            clusters = clusters[:self.n_uavs]

        self.clusters = clusters
        self.n_clusters = len(clusters)

        return clusters

    def print_clusters(self):
        """打印聚类结果"""
        print("\n" + "=" * 70)
        print("[时空聚类] 导弹簇划分结果")
        print("=" * 70)
        print(f"  聚类参数: 空间阈值={self.Dc}m, 时间阈值={self.Tc}s")
        print(f"  总导弹数: {self.n_missiles} -> 可行导弹: {self.n_missiles - len(self.time_infeasible_missiles)} -> Cluster数: {self.n_clusters}")

        # 打印时间不可行的导弹
        if self.time_infeasible_missiles:
            print(f"\n  [时间不可行导弹] ({len(self.time_infeasible_missiles)} 枚, 已提前放弃):")
            for m in self.time_infeasible_missiles:
                print(f"    M{m['id']}: 到达时间={m['arrival_time']:.1f}s, 扇区={m['sector']}")
        print("-" * 70)

        for cluster in self.clusters:
            print(f"\n  Cluster {cluster.cluster_id}:")
            print(f"    导弹数: {cluster.n_missiles}")
            print(f"    导弹ID: {cluster.missile_ids}")
            print(f"    质心: ({cluster.centroid[0]:.0f}, {cluster.centroid[1]:.0f}, {cluster.centroid[2]:.0f})")
            print(f"    到达时间窗: [{cluster.earliest_arrival:.1f}s, {cluster.latest_arrival:.1f}s]")
            print(f"    威胁优先级: {cluster.threat_priority:.1f}s (越小越危险)")
            for m in cluster.missile_configs:
                print(f"      - M{m['id']}: sector={m['sector']}, "
                      f"pos=({m['position'][0]:.0f},{m['position'][1]:.0f},{m['position'][2]:.0f}), "
                      f"arrival={m['arrival_time']:.1f}s")

        # 统计被放弃的导弹
        covered_ids = set()
        for c in self.clusters:
            covered_ids.update(c.missile_ids)
        abandoned = [m["id"] for m in self.missile_configs if m["id"] not in covered_ids]
        if abandoned:
            print(f"\n  [放弃的导弹]: {abandoned}")

        print("=" * 70)

    def compute_cost_matrix(self) -> np.ndarray:
        """
        构建UAV-Cluster代价矩阵 (V2版本)

        代价函数考虑：
        1. UAV到Cluster质心的欧氏距离
        2. 方位角差异（惩罚侧向/背向）
        3. 高度差异
        4. 风场影响（迎风扇区优先）

        Returns:
            cost_matrix: shape (n_uavs, n_clusters)
        """
        # 确保已完成聚类
        if not self.clusters:
            self.cluster_missiles()

        cost_matrix = np.zeros((self.n_uavs, self.n_clusters))

        for i, uav_id in enumerate(self.uav_group):
            uav_pos = self.uav_positions[uav_id]

            for j, cluster in enumerate(self.clusters):
                # 使用Cluster质心作为目标点
                cluster_centroid = cluster.centroid

                # 基础代价：UAV到Cluster质心的欧氏距离
                distance = np.linalg.norm(uav_pos - cluster_centroid)

                # 方位角惩罚
                uav_azimuth = np.arctan2(uav_pos[1], uav_pos[0])
                cluster_azimuth = np.arctan2(cluster_centroid[1], cluster_centroid[0])
                azimuth_diff = abs(uav_azimuth - cluster_azimuth)
                azimuth_penalty = 1000 * (1 - np.cos(azimuth_diff))

                # 高度差异惩罚
                height_diff = abs(uav_pos[2] - cluster_centroid[2])
                height_penalty = 0.5 * height_diff

                # 风场影响
                wind_dir = np.arctan2(self.wind_vector[1], self.wind_vector[0])
                wind_alignment = np.cos(cluster_azimuth - wind_dir)
                wind_bonus = -500 * wind_alignment

                # 总代价
                cost_matrix[i, j] = distance + azimuth_penalty + height_penalty + wind_bonus

        self.cost_matrix = cost_matrix
        return cost_matrix

    def hungarian_assignment(self) -> Dict[int, int]:
        """
        使用匈牙利算法进行UAV-Cluster最优匹配 (V2版本)

        Returns:
            assignment: {uav_id: cluster_id}
        """
        if self.cost_matrix is None:
            self.compute_cost_matrix()

        # 确保已完成聚类
        if not self.clusters:
            self.cluster_missiles()

        # 处理非方阵情况
        n_rows, n_cols = self.cost_matrix.shape

        if n_rows <= n_cols:
            row_ind, col_ind = linear_sum_assignment(self.cost_matrix)
        else:
            row_ind, col_ind = linear_sum_assignment(self.cost_matrix.T)
            row_ind, col_ind = col_ind, row_ind

        # 构建分配字典
        assignment = {}
        uav_cluster_map = {}

        for uav_idx, cluster_idx in zip(row_ind, col_ind):
            uav_id = self.uav_group[uav_idx]
            cluster = self.clusters[cluster_idx]
            assignment[uav_id] = cluster.cluster_id
            uav_cluster_map[uav_id] = cluster

        self.assignment = assignment
        self.uav_cluster_map = uav_cluster_map
        return assignment

    def generate_expert_direction_for_cluster(self, uav_id: int,
                                               cluster: MissileCluster) -> Tuple[float, float, float]:
        """
        生成针对Cluster的专家先验航向

        核心逻辑：
        1. 计算Cluster的拦截区域（质心附近的导弹路径）
        2. 找到 UAV 能在导弹到达前到达的路径点
        3. 让 UAV 飞向该点

        Args:
            uav_id: UAV编号
            cluster: 导弹簇

        Returns:
            (dir_x, dir_y, dir_z): 归一化方向向量
        """
        uav_pos = self.uav_positions[uav_id]
        uav_speed = 100.0  # 平均速度

        # 收集所有导弹的可行拦截点
        candidate_points = []

        for missile_config in cluster.missile_configs:
            missile_pos = np.array(missile_config["position"])
            missile_vel = np.array(missile_config["velocity"])
            arrival_time = missile_config["arrival_time"]

            # 遍历导弹路径，找到 UAV 能在导弹经过前到达的点
            for t in np.linspace(0, arrival_time, 30):
                missile_at_t = missile_pos + missile_vel * t
                dist = np.linalg.norm(missile_at_t[:2] - uav_pos[:2])
                uav_fly_time = dist / uav_speed

                # 如果 UAV 能在导弹经过前到达（留 8s 余量）
                if uav_fly_time + 8 < t:
                    candidate_points.append(missile_at_t)

        if not candidate_points:
            # 如果找不到，飞向Cluster质心
            best_point = cluster.centroid
        else:
            # 选择距离UAV最近的可行点（最早可到达的点）
            distances = [np.linalg.norm(p[:2] - uav_pos[:2]) for p in candidate_points]
            best_point = candidate_points[np.argmin(distances)]

        # 计算方向
        direction = best_point - uav_pos
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return (1.0, 0.0, 0.0)

        unit_dir = direction / norm
        return (float(unit_dir[0]), float(unit_dir[1]), float(unit_dir[2]))

    def generate_expert_time_window_for_cluster(self, uav_id: int, cluster: MissileCluster,
                                                 window_before: float = 15.0,
                                                 window_after: float = 2.0) -> Tuple[float, float]:
        """
        生成针对Cluster的专属时间窗

        时间窗需要覆盖Cluster内所有导弹的到达时间范围

        Args:
            uav_id: UAV编号
            cluster: 导弹簇
            window_before: 最早到达时间前的提前量（秒）
            window_after: 最晚到达时间后的余量（秒）

        Returns:
            (t_drop_min, t_drop_max): 专属时间窗
        """
        # 关键：时间窗需要覆盖整个Cluster的到达时间范围
        # 最早到达时间决定投放下限
        # 最晚到达时间决定投放上限

        min_d_det = 6.0  # 最小起爆延迟

        # 基于最早到达时间计算投放下限
        t_drop_min = max(5.0, cluster.earliest_arrival - window_before - min_d_det)

        # 基于最晚到达时间计算投放上限
        t_drop_max = cluster.latest_arrival - window_after - min_d_det

        # 确保时间窗有效
        if t_drop_max <= t_drop_min:
            t_drop_max = t_drop_min + 2.0

        return (t_drop_min, t_drop_max)

    def generate_expert_genes(self, window_before: float = 15.0,
                               window_after: float = 2.0) -> Dict[int, Dict]:
        """
        生成所有UAV的专家先验基因 (V2版本 - 支持Cluster)

        Returns:
            expert_genes: {uav_id: {
                "direction": (dx,dy,dz),
                "time_window": (t_min, t_max),
                "target_cluster": cluster_id,
                "target_missiles": [missile_ids],
                "cluster": MissileCluster
            }}
        """
        if self.assignment is None:
            self.hungarian_assignment()

        expert_genes = {}

        for uav_id, cluster_id in self.assignment.items():
            cluster = self.uav_cluster_map[uav_id]
            direction = self.generate_expert_direction_for_cluster(uav_id, cluster)
            time_window = self.generate_expert_time_window_for_cluster(
                uav_id, cluster, window_before, window_after
            )

            expert_genes[uav_id] = {
                "direction": direction,
                "time_window": time_window,
                "target_cluster": cluster_id,
                "target_missiles": cluster.missile_ids,
                "cluster": cluster
            }

        return expert_genes

    def print_assignment(self):
        """打印分配结果 (V2版本 - 支持Cluster)"""
        if self.assignment is None:
            self.hungarian_assignment()

        print("\n" + "=" * 70)
        print("[时空目标分配器 V2] UAV-Cluster 匈牙利匹配结果")
        print("=" * 70)

        for uav_id, cluster_id in sorted(self.assignment.items()):
            uav_pos = self.uav_positions[uav_id]
            cluster = self.uav_cluster_map[uav_id]

            print(f"  UAV{uav_id} @ ({uav_pos[0]:.0f}, {uav_pos[1]:.0f}, {uav_pos[2]:.0f})")
            print(f"    -> Cluster {cluster_id} ({cluster.n_missiles} missiles)")
            print(f"       导弹ID: {cluster.missile_ids}")
            print(f"       质心: ({cluster.centroid[0]:.0f}, {cluster.centroid[1]:.0f}, {cluster.centroid[2]:.0f})")
            print(f"       到达时间窗: [{cluster.earliest_arrival:.1f}s, {cluster.latest_arrival:.1f}s]")

        print("=" * 70)


def create_allocator(uav_group: List[int], missile_configs: List[Dict]) -> SpatioTemporalAllocator:
    """
    工厂函数：创建时空目标分配器

    Args:
        uav_group: UAV编号列表
        missile_configs: 导弹配置列表

    Returns:
        SpatioTemporalAllocator 实例
    """
    return SpatioTemporalAllocator(uav_group, missile_configs)


def inject_expert_genes_to_population(population: List[List], uav_group: List[int],
                                       expert_genes: Dict[int, Dict],
                                       injection_ratio: float = 0.4) -> List[List]:
    """
    将专家先验基因注入种群

    Args:
        population: 原始种群
        uav_group: UAV编号列表
        expert_genes: 专家先验基因
        injection_ratio: 注入比例 (0.3-0.5)

    Returns:
        注入后的种群
    """
    import random

    gene_len = 4 + 2 * SMOKE_PAYLOAD_PER_UAV
    n_inject = int(len(population) * injection_ratio)

    for i in range(n_inject):
        individual = population[i]

        for uav_idx, uav_id in enumerate(uav_group):
            if uav_id not in expert_genes:
                continue

            expert = expert_genes[uav_id]
            start = uav_idx * gene_len

            # 注入专家方向
            dx, dy, dz = expert["direction"]
            individual[start] = dx
            individual[start + 1] = dy
            individual[start + 2] = dz

            # 注入专属时间窗
            t_min, t_max = expert["time_window"]

            # 第一发烟幕：在专属窗口内随机
            individual[start + 4] = random.uniform(t_min, t_max)

            # 后续烟幕：递增1-3秒
            for k in range(1, SMOKE_PAYLOAD_PER_UAV):
                prev_t = individual[start + 4 + k - 1]
                individual[start + 4 + k] = min(prev_t + random.uniform(1.0, 3.0), t_max)

        population[i] = individual

    return population
