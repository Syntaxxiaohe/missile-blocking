"""
OT 最优传输目标分配器 (Optimal Transport Allocator)

核心思想：
  彻底废弃基于硬阈值的 K-means / 贪婪聚类。
  将 UAV 蜂群视为「防御资源场」，将导弹群视为「威胁需求场」，
  建模为离散最优传输（Discrete Optimal Transport）规划问题。

相比 SpatioTemporalAllocator 的优势：
  - 不依赖任何硬编码的距离/时间阈值，自动全局最优分配
  - 通过「虚拟节点扩充法」支持 5 UAV vs 20 导弹的多对一映射
  - 在全向蜂群（Mode-4）等极端场景下不再丢失导弹覆盖

接口兼容性：
  OTAllocator 与 SpatioTemporalAllocator 完全兼容：
  - 相同的 __init__ 签名（uav_group, missile_configs）
  - 提供相同的 get_clusters()、get_assignment()、get_expert_genes()、
    print_assignment() 方法
  - 返回相同的 MissileCluster 对象格式
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

# 复用同目录下已定义的 MissileCluster 数据结构，保证与全系统兼容
from .allocator import MissileCluster
from ..config.params import UAV_PARAMS, SMOKE_PAYLOAD_PER_UAV


# ========================== OT 代价权重常量 ==========================

# 空间代价权重：单位 (m/s) → 最终代价量纲为秒
# 1/v_uav ≈ 1/100 = 0.01，直接使用归一化后的距离代价
W_S_DEFAULT = 1.0

# 时间紧迫权重：|TTI - TTI_ref| 的惩罚系数
# 设为 10.0 使时间项与空间项数量级相近（TTI 偏差约 5s, 距离 / 100m/s ≈ 50-150s）
W_T_DEFAULT = 10.0

# 基准参考时间（秒）：导弹池设计的同时弹着时刻
TTI_REFERENCE = 45.0


class OTAllocator:
    """
    基于最优传输理论的 UAV-导弹全局自适应分配器

    核心流程：
    1. 虚拟席位扩充：将 N_uav 架 UAV 克隆为 N_missile 个虚拟席位
    2. 构建时空联合代价矩阵 C (N_missile × N_missile)
    3. 调用匈牙利算法求解全局最优传输方案
    4. 将同一真实 UAV 的所有虚拟席位合并成 MissileCluster
    5. 生成每架 UAV 的专家先验（飞行方向 + 投放时间窗）

    代价函数：
        C[i, j] = W_S * distance(uav_i, missile_j) + W_T * |TTI_j - TTI_ref|

    其中 i 为扩充后的虚拟 UAV 席位索引，j 为导弹索引。
    """

    def __init__(self, uav_group: List[int], missile_configs: List[Dict],
                 w_s: float = W_S_DEFAULT, w_t: float = W_T_DEFAULT,
                 tti_ref: float = TTI_REFERENCE):
        """
        Args:
            uav_group:       UAV 编号列表，如 [1, 2, 3, 4, 5]
            missile_configs: 导弹配置列表，每项含 id/position/velocity/arrival_time 等字段
            w_s:             空间代价权重（默认 1.0）
            w_t:             时间紧迫代价权重（默认 10.0）
            tti_ref:         基准到达时间参考值，单位秒（默认 45.0s）
        """
        self.uav_group = uav_group
        self.missile_configs = missile_configs
        self.n_uavs = len(uav_group)
        self.n_missiles = len(missile_configs)
        self.w_s = w_s
        self.w_t = w_t
        self.tti_ref = tti_ref

        # 从全局参数读取 UAV 初始位置
        self.uav_positions: Dict[int, np.ndarray] = {
            uav_id: UAV_PARAMS["init_pos"][uav_id]
            for uav_id in uav_group
        }

        # 每架 UAV 最多负责多少枚导弹（向上取整，保证全覆盖）
        self.capacity = int(np.ceil(self.n_missiles / self.n_uavs))

        # 扩充后的虚拟席位总数（= n_uavs × capacity，必须 >= n_missiles）
        self.n_virtual = self.n_uavs * self.capacity

        # 运行状态（懒惰初始化，调用 allocate() 后填充）
        self._clusters: List[MissileCluster] = []
        self._assignment: Dict[int, int] = {}        # {uav_id: cluster_id}
        self._uav_cluster_map: Dict[int, MissileCluster] = {}  # {uav_id: cluster}
        self._cost_matrix: np.ndarray = None         # 扩充后的代价矩阵

    # ================================================================
    # 核心公共方法（与 SpatioTemporalAllocator 接口对齐）
    # ================================================================

    def allocate(self) -> List[MissileCluster]:
        """
        执行完整的 OT 分配流程，返回 5 个 MissileCluster 的列表。

        调用顺序：
            build_cost_matrix() -> solve_ot() -> build_clusters()

        Returns:
            clusters: 每架 UAV 对应一个 MissileCluster，共 n_uavs 个。
                     若某架 UAV 未被分配到任何导弹（极少数极端情况），
                     其 cluster 的 missile_configs 为空列表。
        """
        self._build_cost_matrix()
        uav_missile_mapping = self._solve_ot()
        self._build_clusters(uav_missile_mapping)
        return self._clusters

    def get_clusters(self) -> List[MissileCluster]:
        """
        返回当前聚类结果。若尚未运行 allocate()，则自动触发。
        """
        if not self._clusters:
            self.allocate()
        return self._clusters

    def get_assignment(self) -> Dict[int, int]:
        """
        返回 {uav_id: cluster_id} 映射字典。
        与 SpatioTemporalAllocator.assignment 格式一致。
        """
        if not self._assignment:
            self.allocate()
        return self._assignment

    def get_uav_cluster_map(self) -> Dict[int, MissileCluster]:
        """
        返回 {uav_id: MissileCluster} 映射字典。
        """
        if not self._uav_cluster_map:
            self.allocate()
        return self._uav_cluster_map

    def get_expert_genes(self, uav_id: int) -> Dict:
        """
        生成单架 UAV 的专家先验基因，格式与 SpatioTemporalAllocator 完全一致。

        Args:
            uav_id: UAV 编号（1-based）

        Returns:
            expert_gene: {
                "direction":      (dir_x, dir_y, dir_z),  # 归一化飞行方向
                "time_window":    (t_min, t_max),          # 烟幕投放时间窗
                "n_missiles":     int,                     # 负责导弹数
                "target_missiles": List[int],              # 导弹 ID 列表
                "cluster":        MissileCluster           # 簇对象
            }
        """
        if not self._uav_cluster_map:
            self.allocate()

        cluster = self._uav_cluster_map.get(uav_id)
        if cluster is None:
            # 极端情况：该 UAV 没有被分配到任何导弹，返回默认先验
            return self._default_expert_gene(uav_id)

        direction = self._calc_expert_direction(uav_id, cluster)
        time_window = self._calc_expert_time_window(cluster)

        return {
            "direction":       direction,
            "time_window":     time_window,
            "n_missiles":      cluster.n_missiles,
            "target_missiles": cluster.missile_ids,
            "cluster":         cluster,
        }

    def print_assignment(self):
        """
        打印 OT 分配结果，格式与 SpatioTemporalAllocator.print_assignment() 对齐。
        """
        if not self._uav_cluster_map:
            self.allocate()

        print("\n" + "=" * 70)
        print("[OT 最优传输分配器] UAV-导弹全局最优分配结果")
        print(f"  代价权重: W_S={self.w_s}, W_T={self.w_t},  TTI_ref={self.tti_ref}s")
        print(f"  虚拟席位: {self.n_uavs} UAV × {self.capacity} 容量 = {self.n_virtual} 席位")
        print("=" * 70)

        for uav_id in sorted(self._uav_cluster_map.keys()):
            uav_pos = self.uav_positions[uav_id]
            cluster = self._uav_cluster_map[uav_id]
            print(f"  UAV{uav_id} @ ({uav_pos[0]:.0f}, {uav_pos[1]:.0f}, {uav_pos[2]:.0f})")
            print(f"    -> Cluster {cluster.cluster_id} ({cluster.n_missiles} 枚导弹)")
            print(f"       导弹ID: {cluster.missile_ids}")
            if cluster.n_missiles > 0:
                cx, cy, cz = cluster.centroid
                print(f"       质心: ({cx:.0f}, {cy:.0f}, {cz:.0f})")
                print(f"       到达时间窗: [{cluster.earliest_arrival:.1f}s, "
                      f"{cluster.latest_arrival:.1f}s]")
            else:
                print(f"       （无导弹分配，空闲 UAV）")
        print("=" * 70)

    # ================================================================
    # 内部核心方法
    # ================================================================

    def _build_cost_matrix(self) -> np.ndarray:
        """
        构建扩充后的代价矩阵 C，形状为 (n_virtual, n_missiles)。

        虚拟席位扩充逻辑：
            真实 UAV 0 对应虚拟席位 [0, capacity-1]
            真实 UAV 1 对应虚拟席位 [capacity, 2*capacity-1]
            以此类推…

        代价公式：
            C[virtual_i, j] = W_S * dist(uav_i, missile_j)
                             + W_T * |TTI_j - TTI_ref|

        当 n_virtual > n_missiles 时（向上取整造成多余席位），
        多余的列用大惩罚值（LARGE_COST）填充，匈牙利算法不会选它们。
        """
        # 大惩罚值：确保多余虚拟席位（若方阵对角线需要）被忽略
        LARGE_COST = 1e9

        # 构造方形矩阵（n_virtual × n_virtual），多余列用大惩罚填充
        C = np.full((self.n_virtual, self.n_virtual), LARGE_COST, dtype=np.float64)

        for real_idx, uav_id in enumerate(self.uav_group):
            uav_pos = self.uav_positions[uav_id]  # shape (3,)

            for j, missile in enumerate(self.missile_configs):
                missile_pos = np.array(missile["position"])  # shape (3,)
                tti = missile["arrival_time"]

                # 空间代价：UAV 初始位置到导弹初始位置的 XY 平面欧氏距离（单位：m）
                spatial_cost = np.linalg.norm(uav_pos[:2] - missile_pos[:2])

                # 时间代价：导弹 TTI 偏离基准同时弹着时刻的程度（单位：s）
                temporal_cost = abs(tti - self.tti_ref)

                # 联合代价
                cost = self.w_s * spatial_cost + self.w_t * temporal_cost

                # 将同一真实 UAV 的所有虚拟席位赋予相同代价
                for k in range(self.capacity):
                    virtual_i = real_idx * self.capacity + k
                    C[virtual_i, j] = cost

        self._cost_matrix = C
        return C

    def _solve_ot(self) -> Dict[int, List[int]]:
        """
        调用匈牙利算法求解最优传输，返回真实 UAV 到导弹列表的映射。

        Returns:
            uav_missile_mapping: {real_uav_idx(0-based): [missile_idx_0, missile_idx_1, ...]}
        """
        if self._cost_matrix is None:
            self._build_cost_matrix()

        # scipy 匈牙利算法：O(n³)，对 20×20 矩阵瞬间完成
        row_ind, col_ind = linear_sum_assignment(self._cost_matrix)

        # 将虚拟席位映射回真实 UAV，并过滤掉指向虚拟填充列的无效匹配
        uav_missile_mapping: Dict[int, List[int]] = {
            i: [] for i in range(self.n_uavs)
        }

        for virtual_i, j in zip(row_ind, col_ind):
            # j >= n_missiles 说明匹配到了填充的虚拟导弹列，忽略
            if j >= self.n_missiles:
                continue
            real_uav_idx = virtual_i // self.capacity
            uav_missile_mapping[real_uav_idx].append(j)

        return uav_missile_mapping

    def _build_clusters(self, uav_missile_mapping: Dict[int, List[int]]):
        """
        将 OT 分配结果打包成 MissileCluster 对象列表。

        Args:
            uav_missile_mapping: {real_uav_idx(0-based): [missile_j 索引列表]}
        """
        clusters: List[MissileCluster] = []
        assignment: Dict[int, int] = {}
        uav_cluster_map: Dict[int, MissileCluster] = {}

        for real_uav_idx, uav_id in enumerate(self.uav_group):
            missile_indices = uav_missile_mapping.get(real_uav_idx, [])

            # 取出该 UAV 负责的所有导弹配置
            assigned_missiles = [self.missile_configs[j] for j in missile_indices]

            cluster_id = real_uav_idx + 1  # cluster_id 从 1 开始

            if len(assigned_missiles) == 0:
                # 极端情况：该 UAV 没有分配到导弹（例如导弹数 < UAV 数）
                # 仍然构造一个空 cluster，保证接口格式统一
                cluster = MissileCluster(
                    cluster_id=cluster_id,
                    missile_ids=[],
                    missile_configs=[],
                    centroid=self.uav_positions[uav_id].copy(),  # 质心设为 UAV 自身位置
                    earliest_arrival=self.tti_ref,
                    latest_arrival=self.tti_ref,
                    threat_priority=self.tti_ref,
                )
            else:
                # 计算质心（XYZ 均值）
                positions = np.array([m["position"] for m in assigned_missiles])
                centroid = np.mean(positions, axis=0)

                # 到达时间统计
                arrival_times = [m["arrival_time"] for m in assigned_missiles]
                earliest = float(min(arrival_times))
                latest = float(max(arrival_times))
                # 威胁优先级 = 最早到达时间（越小越危险）
                threat_priority = earliest

                cluster = MissileCluster(
                    cluster_id=cluster_id,
                    missile_ids=[m["id"] for m in assigned_missiles],
                    missile_configs=assigned_missiles,
                    centroid=centroid,
                    earliest_arrival=earliest,
                    latest_arrival=latest,
                    threat_priority=threat_priority,
                )

            clusters.append(cluster)
            assignment[uav_id] = cluster_id
            uav_cluster_map[uav_id] = cluster

        self._clusters = clusters
        self._assignment = assignment
        self._uav_cluster_map = uav_cluster_map

    # ================================================================
    # 专家先验生成（复用 SpatioTemporalAllocator 的成熟逻辑）
    # ================================================================

    def _calc_expert_direction(self, uav_id: int,
                                cluster: MissileCluster) -> Tuple[float, float, float]:
        """
        计算 UAV 的专家先验飞行方向。

        策略：沿导弹轨迹扫描，找到 UAV 能在导弹到达前赶到（留 8s 余量）
              的最近可行拦截点；若找不到则飞向簇质心。

        Args:
            uav_id:  UAV 编号
            cluster: 该 UAV 负责的导弹簇

        Returns:
            (dir_x, dir_y, dir_z): 归一化方向向量
        """
        uav_pos = self.uav_positions[uav_id]
        uav_speed = 100.0  # 平均飞行速度估算值 (m/s)

        if cluster.n_missiles == 0:
            # 空簇：默认向目标区域（原点）飞
            direction = -uav_pos
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return (1.0, 0.0, 0.0)
            unit = direction / norm
            return (float(unit[0]), float(unit[1]), float(unit[2]))

        # 沿每枚导弹的轨迹扫描可行拦截点
        candidate_points = []

        for missile_cfg in cluster.missile_configs:
            m_pos = np.array(missile_cfg["position"])
            m_vel = np.array(missile_cfg["velocity"])
            arrival_time = missile_cfg["arrival_time"]

            for t in np.linspace(0.0, arrival_time, 30):
                point_at_t = m_pos + m_vel * t
                dist = np.linalg.norm(point_at_t[:2] - uav_pos[:2])
                fly_time = dist / uav_speed
                # 需满足：UAV 飞行时间 + 8s 余量 < 导弹到达 t 时刻
                if fly_time + 8.0 < t:
                    candidate_points.append(point_at_t)

        if not candidate_points:
            # 无可行拦截点，飞向簇质心
            best_point = cluster.centroid
        else:
            # 选距离 UAV 最近的可行点（最早能到达的点）
            distances = [np.linalg.norm(p[:2] - uav_pos[:2]) for p in candidate_points]
            best_point = candidate_points[int(np.argmin(distances))]

        direction = best_point - uav_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return (1.0, 0.0, 0.0)

        unit = direction / norm
        return (float(unit[0]), float(unit[1]), float(unit[2]))

    def _calc_expert_time_window(self, cluster: MissileCluster,
                                  window_before: float = 15.0,
                                  window_after: float = 2.0) -> Tuple[float, float]:
        """
        计算烟幕投放时间窗，覆盖簇内所有导弹的到达时间范围。

        Args:
            cluster:       导弹簇
            window_before: 最早到达时间前的提前量（秒），默认 15s
            window_after:  最晚到达时间后的余量（秒），默认 2s

        Returns:
            (t_drop_min, t_drop_max): 合法的投放时间窗
        """
        min_d_det = 6.0  # 最小起爆延迟时间（秒）

        if cluster.n_missiles == 0:
            # 空簇：使用宽松的默认窗口
            return (20.0, 50.0)

        # 投放下限：基于最早到达时间留出提前量
        t_drop_min = max(5.0, cluster.earliest_arrival - window_before - min_d_det)

        # 投放上限：基于最晚到达时间留出余量
        t_drop_max = cluster.latest_arrival - window_after - min_d_det

        # 保证时间窗有效（至少有 2s 宽度）
        if t_drop_max <= t_drop_min:
            t_drop_max = t_drop_min + 2.0

        return (float(t_drop_min), float(t_drop_max))

    def _default_expert_gene(self, uav_id: int) -> Dict:
        """
        当 UAV 未分配到任何导弹时返回的默认先验基因。
        方向：指向防区中心（原点方向）；时间窗：宽松默认值。
        """
        uav_pos = self.uav_positions[uav_id]
        direction = -uav_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction_tuple = (1.0, 0.0, 0.0)
        else:
            d = direction / norm
            direction_tuple = (float(d[0]), float(d[1]), float(d[2]))

        return {
            "direction":       direction_tuple,
            "time_window":     (20.0, 50.0),
            "n_missiles":      0,
            "target_missiles": [],
            "cluster":         None,
        }
