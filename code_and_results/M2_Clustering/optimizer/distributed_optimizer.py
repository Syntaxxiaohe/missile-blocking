"""
M2_Clustering 两阶段分布式解耦优化器

核心创新：
  1. 时空聚类：将N枚导弹打包成M个Cluster (M <= 5)
  2. Phase 1: 分布式独立优化 - M个子任务并行，每对(UAV, Cluster)独立优化
  3. Phase 2: 基因拼接与微调 - 拼接成50维超级个体，处理跨区域串扰

M2 特点：
  - 使用 SpatioTemporalAllocator（时空聚类 + 匈牙利匹配）
  - 使用 MissileCluster（真实聚类）
  - 专家先验精准：质心指向真实目标群，时间窗收窄
"""

import os
import sys
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

# ===== ROOT_DIR 注入（指向终极实验根目录）=====
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.config.params import (
    UAV_PARAMS, SMOKE_PAYLOAD_PER_UAV, SUCCESS_SCORE_WEIGHT,
    compute_global_tdrop_range
)
from shared.simulation.single import multi_missile_objective
from shared.optimizer.allocator import SpatioTemporalAllocator, MissileCluster

# ========================== M2 算法常量 ==========================
GENE_DIM_PER_UAV = 4 + 2 * SMOKE_PAYLOAD_PER_UAV  # 4 + 2*3 = 10维


# ========================== 顶层函数（进程池兼容）==========================

def _evaluate_multi_missiles(args):
    """
    评估单个UAV拦截多枚导弹（进程池兼容）- M2版本

    Args:
        args: (params, uav_num, missile_configs, safe_distance)
            - params: 10维参数向量 [dir_x, dir_y, dir_z, speed, t_drop1, d_det1, t_drop2, d_det2, t_drop3, d_det3]
            - uav_num: UAV编号
            - missile_configs: 导弹配置列表（一个Cluster内的所有导弹）
            - safe_distance: 安全距离阈值

    Returns:
        (fitness, success_count, miss_distances)
    """
    params, uav_num, missile_configs, safe_distance = args

    # 构建单UAV配置
    uav_dir = np.array(params[0:3])
    dir_norm = np.linalg.norm(uav_dir)
    if dir_norm > 1e-6:
        uav_dir = uav_dir / dir_norm

    uav_config = {
        "uav_num": uav_num,
        "speed": params[3],
        "uav_dir": uav_dir,
        "t_drops": list(params[4:4+SMOKE_PAYLOAD_PER_UAV]),
        "d_dets": list(params[4+SMOKE_PAYLOAD_PER_UAV:4+2*SMOKE_PAYLOAD_PER_UAV])
    }

    # 导入物理模块（使用 shared 模块）
    from shared.physics.smoke import SmokeStateCache
    from shared.physics.blocking import is_smoke_blocking_vectorized
    from shared.physics.missile_png import PNGMissileFast
    from shared.config.params import SMOKE_PARAMS, MAX_SIM_TIME, ACCEL_PARAMS, PNG_PARAMS

    # 创建烟幕缓存
    smoke_cache = SmokeStateCache(
        [uav_config],
        t_max=MAX_SIM_TIME + 30,
        dt=ACCEL_PARAMS["smoke_cache_dt"]
    )

    # 烟幕检查函数
    def smoke_checker(missile_num, t, pos):
        centers, sigmas, is_eff, is_det, Q_vals = smoke_cache.get_states_at_t(t)
        valid_mask = is_eff & is_det & (sigmas <= 100.0) & (sigmas > 1e-6)
        if not np.any(valid_mask):
            return False
        return is_smoke_blocking_vectorized(
            pos, centers[valid_mask], sigmas[valid_mask],
            is_eff[valid_mask], is_det[valid_mask], Q_vals[valid_mask]
        )

    # 评估每枚导弹
    success_count = 0
    miss_distances = []

    for missile_config in missile_configs:
        init_pos = np.array(missile_config["position"])
        init_vel = np.array(missile_config["velocity"])

        missile = PNGMissileFast(
            missile_num=missile_config["id"],
            init_pos=init_pos,
            init_vel=init_vel,
            record_history=False
        )

        result = missile.simulate(smoke_checker=smoke_checker, dt=0.02, max_time=MAX_SIM_TIME + 30)
        miss_distance = result["miss_distance"]
        miss_distances.append(miss_distance)

        if miss_distance >= safe_distance:
            success_count += 1

    # 计算适应度：成功拦截数 × 10000 + 未成功导弹的平均脱靶量
    n_missiles = len(missile_configs)
    fitness = success_count * SUCCESS_SCORE_WEIGHT

    # 添加未成功导弹的平均脱靶量作为奖励（避免全是0的情况）
    failed_miss_distances = [md for i, md in enumerate(miss_distances)
                             if md < safe_distance]
    if failed_miss_distances:
        avg_miss = np.mean(failed_miss_distances)
        fitness += avg_miss

    return (fitness, success_count, miss_distances)


# ========================== 子任务GA优化器 ==========================

class SubTaskGA:
    """
    单UAV-多导弹子任务GA优化器（10维空间）- M2版本

    优化目标：最大化对Cluster内所有导弹的拦截成功率
    满分 = Cluster内导弹数量 × 10000
    """

    def __init__(self, uav_num: int, cluster: MissileCluster,
                 expert_gene: Dict, pop_size: int = 20, max_generations: int = 20,
                 n_threads: int = None):
        """
        Args:
            uav_num: UAV编号
            cluster: 导弹簇（包含多枚导弹）
            expert_gene: 专家先验基因 {"direction": (dx,dy,dz), "time_window": (t_min, t_max)}
            pop_size: 种群大小
            max_generations: 最大代数
            n_threads: 并行进程数
        """
        self.uav_num = uav_num
        self.cluster = cluster
        self.missile_configs = cluster.missile_configs
        self.n_missiles = cluster.n_missiles
        self.expert_gene = expert_gene
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.n_threads = n_threads if n_threads else os.cpu_count()
        self.safe_distance = 15.0

        # 满分 = 导弹数 × 10000
        self.perfect_score = self.n_missiles * SUCCESS_SCORE_WEIGHT

        # 生成边界（基于专家时间窗）
        self.bounds = self._generate_bounds()

    def _generate_bounds(self) -> List[Tuple[float, float]]:
        """生成10维边界"""
        t_min, t_max = self.expert_gene["time_window"]

        bounds = [
            (-1.0, 1.0),     # dir_x
            (-1.0, 1.0),     # dir_y
            (-1.0, 1.0),     # dir_z
            (70.0, 140.0),   # speed
        ]

        for _ in range(SMOKE_PAYLOAD_PER_UAV):
            bounds.append((t_min, t_max))  # t_drop
            bounds.append((6.0, 10.0))      # d_det

        return bounds

    def _init_population(self) -> List[List]:
        """初始化种群（注入专家先验）"""
        population = []
        gene_len = GENE_DIM_PER_UAV
        n_expert = int(self.pop_size * 0.5)  # 50%专家个体

        for i in range(self.pop_size):
            individual = [random.uniform(b[0], b[1]) for b in self.bounds]

            if i < n_expert:
                # 专家先验注入
                dx, dy, dz = self.expert_gene["direction"]
                individual[0] = dx
                individual[1] = dy
                individual[2] = dz

                # 时间窗内均匀分布（针对多导弹，需要更分散的投放时间）
                t_min, t_max = self.expert_gene["time_window"]
                time_span = t_max - t_min

                for k in range(SMOKE_PAYLOAD_PER_UAV):
                    # 均匀分布在整个时间窗内
                    t_base = t_min + time_span * k / max(SMOKE_PAYLOAD_PER_UAV - 1, 1)
                    individual[4 + k] = min(t_base, t_max)

            # 约束校验
            individual = self._enforce_constraints(individual)
            population.append(individual)

        return population

    def _enforce_constraints(self, individual: List) -> List:
        """约束校验"""
        # 方向归一化
        dir_x, dir_y, dir_z = individual[0:3]
        dir_norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        if dir_norm > 1e-6:
            individual[0] = dir_x / dir_norm
            individual[1] = dir_y / dir_norm
            individual[2] = dir_z / dir_norm

        # 速度约束
        individual[3] = np.clip(individual[3], 70.0, 140.0)

        # 时间窗约束
        t_min, t_max = self.expert_gene["time_window"]
        for k in range(SMOKE_PAYLOAD_PER_UAV):
            individual[4 + k] = np.clip(individual[4 + k], t_min, t_max)
            individual[4 + SMOKE_PAYLOAD_PER_UAV + k] = np.clip(
                individual[4 + SMOKE_PAYLOAD_PER_UAV + k], 6.0, 10.0)

        # 投放时间递增约束
        for k in range(1, SMOKE_PAYLOAD_PER_UAV):
            if individual[4 + k] < individual[4 + k - 1] + 0.5:
                individual[4 + k] = individual[4 + k - 1] + random.uniform(0.5, 2.0)
            individual[4 + k] = np.clip(individual[4 + k], t_min, t_max)

        return individual

    def _evaluate_population(self, population: List) -> List[Tuple[float, int, List]]:
        """评估种群（并行 or 串行）。
        当 n_threads==1 时退化为串行 for 循环，避免在子进程内嵌套
        ProcessPoolExecutor 导致 Windows DuplicateHandle [WinError 5]。
        """
        args_list = [
            (ind, self.uav_num, self.missile_configs, self.safe_distance)
            for ind in population
        ]

        if self.n_threads == 1:
            # 已在外层子进程中 —— 严格串行，不允许再开进程池
            results = [_evaluate_multi_missiles(a) for a in args_list]
        else:
            with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
                results = list(executor.map(_evaluate_multi_missiles, args_list))

        return results

    def _selection(self, fitness_scores: List[float]) -> List:
        """轮盘赌选择"""
        total = sum(max(s, 0) for s in fitness_scores)
        if total <= 0:
            return random.choice(self.population)
        probs = [max(s, 0) / total for s in fitness_scores]
        return random.choices(self.population, probs)[0]

    def _crossover(self, p1: List, p2: List) -> List:
        """单点交叉"""
        if random.random() < 0.9:
            cp = random.randint(1, len(p1) - 1)
            return p1[:cp] + p2[cp:]
        return p1.copy()

    def _mutation(self, individual: List, rate: float = 0.1) -> List:
        """变异"""
        for i in range(len(individual)):
            if random.random() < rate:
                lb, ub = self.bounds[i]
                rng = ub - lb
                individual[i] = np.clip(individual[i] + random.uniform(-0.1 * rng, 0.1 * rng), lb, ub)
        return self._enforce_constraints(individual)

    def optimize(self) -> Tuple[List, float, int]:
        """执行优化"""
        missile_ids = self.cluster.missile_ids
        print(f"\n[子任务 M2] UAV{self.uav_num} vs Cluster({self.n_missiles}导弹: {missile_ids}) 开始优化...")
        print(f"  时间窗: [{self.expert_gene['time_window'][0]:.1f}s, {self.expert_gene['time_window'][1]:.1f}s]")
        print(f"  满分: {self.perfect_score} ({self.n_missiles} × 10000)")

        # 初始化种群
        self.population = self._init_population()
        self.best_fitness = -np.inf
        self.best_params = None
        self.best_success_count = 0

        # 评估初始种群
        results = self._evaluate_population(self.population)
        fitness_scores = [r[0] for r in results]

        # 更新最优
        best_idx = np.argmax(fitness_scores)
        self.best_fitness = fitness_scores[best_idx]
        self.best_params = self.population[best_idx].copy()
        self.best_success_count = results[best_idx][1]

        print(f"  第0代: 最优={self.best_fitness:.2f}, 成功拦截={self.best_success_count}/{self.n_missiles}")

        # 完美熔断检查
        if self.best_fitness >= self.perfect_score:
            print(f"  [熔断] 第0代即达成完美拦截！{self.n_missiles}/{self.n_missiles}")
            return self.best_params, self.best_fitness, self.best_success_count

        # 早停参数
        _PATIENCE  = 30       # 增大以给GA更多时间找到有效遮蔽时间窗
        _MIN_DELTA = 1e-4
        _stagnation = 0
        _prev_best  = self.best_fitness

        # 迭代优化
        for gen in range(self.max_generations):
            new_pop = []

            # 精英保留
            elite_size = int(self.pop_size * 0.2)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_pop.append(self.population[idx].copy())

            # 交叉变异
            while len(new_pop) < self.pop_size:
                p1 = self._selection(fitness_scores)
                p2 = self._selection(fitness_scores)
                child = self._crossover(p1, p2)
                child = self._mutation(child, rate=0.1 * (1 - gen / self.max_generations))
                new_pop.append(child)

            self.population = new_pop

            # 评估
            results = self._evaluate_population(self.population)
            fitness_scores = [r[0] for r in results]

            # 更新最优
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_params = self.population[best_idx].copy()
                self.best_success_count = results[best_idx][1]

            # 早停：拦截数为0且连续_PATIENCE代无进展 → 终止
            if self.best_fitness - _prev_best < _MIN_DELTA and self.best_success_count == 0:
                _stagnation += 1
            else:
                _stagnation = 0
                _prev_best = self.best_fitness
            if _stagnation >= _PATIENCE:
                print(f"  [早停] 第{gen+1}代: 连续{_PATIENCE}代无进展且拦截=0，终止优化")
                break

            if (gen + 1) % 15 == 0:
                print(f"  第{gen+1}代: 最优={self.best_fitness:.2f}, 成功拦截={self.best_success_count}/{self.n_missiles}")

            # 完美熔断
            if self.best_fitness >= self.perfect_score:
                print(f"  [熔断] 第{gen+1}代达成完美拦截！{self.n_missiles}/{self.n_missiles}")
                break

        return self.best_params, self.best_fitness, self.best_success_count


# ========================== 两阶段分布式优化器 M2 ==========================

class DistributedOptimizer:
    """
    M2 两阶段分布式优化器 - 时空聚类版

    Phase 1: M个子任务独立优化（10维空间，M = Cluster数量 <= 5）
    Phase 2: 基因拼接与全局微调（50维空间）
    """

    def __init__(self, uav_group: List[int], missile_configs: List[Dict],
                 pop_size: int = 20, max_generations: int = 20, n_threads: int = None,
                 spatial_threshold: float = None, temporal_threshold: float = None):
        self.uav_group = uav_group
        self.missile_configs = missile_configs
        self.n_missiles = len(missile_configs)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.n_threads = n_threads if n_threads else os.cpu_count()

        # 创建时空聚类分配器（M2核心特点；可注入 Dc/Tc 用于灵敏度扫描）
        self.allocator = SpatioTemporalAllocator(
            uav_group, missile_configs,
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
        )

    def run_phase1(self) -> Dict[int, Tuple[List, float, int]]:
        """Phase 1: 分布式独立优化"""
        print("\n" + "=" * 70)
        print("  PHASE 1 M2: 分布式独立优化 (子任务 × 10维空间)")
        print("  策略: 时空聚类 -> 匈牙利匹配 -> 分布式优化")
        print("=" * 70)

        # 执行时空聚类
        self.allocator.cluster_missiles()
        self.allocator.print_clusters()

        # 执行匈牙利匹配
        self.allocator.compute_cost_matrix()
        self.allocator.hungarian_assignment()
        self.allocator.print_assignment()

        # 生成专家先验基因
        expert_genes = self.allocator.generate_expert_genes(window_before=15.0, window_after=2.0)

        print("\n[专家先验] 时间窗参数:")
        for uav_id, gene in sorted(expert_genes.items()):
            cluster = gene["cluster"]
            print(f"  UAV{uav_id} -> Cluster {gene['target_cluster']} (导弹: {gene['target_missiles']})")
            print(f"    时间窗: [{gene['time_window'][0]:.1f}s, {gene['time_window'][1]:.1f}s]")
            print(f"    覆盖到达时间: [{cluster.earliest_arrival:.1f}s, {cluster.latest_arrival:.1f}s]")

        # 串行执行子任务（避免进程池嵌套）
        subtask_results = {}
        t0 = time.time()

        for uav_id in sorted(self.allocator.assignment.keys()):
            cluster = self.allocator.uav_cluster_map[uav_id]
            expert_gene = expert_genes[uav_id]

            subtask = SubTaskGA(
                uav_num=uav_id,
                cluster=cluster,
                expert_gene=expert_gene,
                pop_size=self.pop_size,
                max_generations=self.max_generations,
                n_threads=self.n_threads
            )

            best_params, best_fitness, success_count = subtask.optimize()
            subtask_results[uav_id] = (best_params, best_fitness, success_count)

        phase1_time = time.time() - t0
        print(f"\n[Phase 1 完成] 耗时: {phase1_time:.1f}s")

        # 打印汇总
        print("\n[Phase 1 汇总]")
        total_success = 0
        total_target = 0
        for uav_id, (params, fitness, success_count) in sorted(subtask_results.items()):
            cluster = self.allocator.uav_cluster_map[uav_id]
            target_count = cluster.n_missiles
            status = "[OK] 完美" if success_count >= target_count else f"[..] {success_count}/{target_count}"
            print(f"  UAV{uav_id}: fitness={fitness:.2f}, 拦截={success_count}/{target_count} {status}")
            total_success += success_count
            total_target += target_count

        print(f"\n  总拦截: {total_success}/{total_target}")

        return subtask_results

    def splice_genes(self, subtask_results: Dict[int, Tuple[List, float, int]],
                     n_uavs: int = 5) -> List:
        """基因拼接：将M段10维基因拼接成50维超级个体"""
        print("\n" + "=" * 70)
        print("  基因拼接 M2: 子任务10维 → 50维超级个体")
        print("=" * 70)

        super_individual = []

        for uav_id in range(1, n_uavs + 1):
            if uav_id in subtask_results:
                params, _, _ = subtask_results[uav_id]
                super_individual.extend(params)
                print(f"  UAV{uav_id}: {params[:4]}... (len={len(params)})")
            else:
                # 默认参数
                default_params = [1.0, 0.0, 0.0, 100.0]
                t_min, t_max = 20.0, 50.0
                for k in range(SMOKE_PAYLOAD_PER_UAV):
                    default_params.append(t_min + (t_max - t_min) * k / SMOKE_PAYLOAD_PER_UAV)
                    default_params.append(8.0)
                super_individual.extend(default_params)
                print(f"  UAV{uav_id}: [默认参数] (未分配任务)")

        print(f"\n  超级个体维度: {len(super_individual)}")

        return super_individual

    def run_phase2(self, super_individual: List) -> Tuple[List, float, int]:
        """Phase 2: 零代验证 + 全局微调"""
        print("\n" + "=" * 70)
        print("  PHASE 2 M2: 零代验证 + 全局微调")
        print("=" * 70)

        # 零代验证
        print("\n[零代验证] 评估超级个体...")

        fitness, details = multi_missile_objective(
            super_individual, self.uav_group, self.missile_configs
        )

        success_count = details["success_count"]
        perfect_score = self.n_missiles * SUCCESS_SCORE_WEIGHT

        print(f"  超级个体适应度: {fitness:.2f}")
        print(f"  成功拦截: {success_count}/{self.n_missiles}")

        if fitness >= perfect_score:
            print("\n" + "=" * 70)
            print(f"  [零代验证成功] {success_count}/{self.n_missiles} 完美拦截！无需微调！")
            print("=" * 70)
            return super_individual, fitness, success_count

        # 全局微调
        print(f"\n[全局微调] 检测到串扰，启动微调 (pop={self.pop_size}, gen=10)...")

        best_params = super_individual.copy()
        best_fitness = fitness
        best_success = success_count

        for gen in range(10):
            population = [best_params.copy()]

            for _ in range(self.pop_size - 1):
                individual = best_params.copy()
                for i in range(len(individual)):
                    if random.random() < 0.2:
                        delta = random.uniform(-0.05, 0.05) * (1 if i < 4 else 2)
                        individual[i] += delta
                population.append(individual)

            for individual in population:
                fit, det = multi_missile_objective(
                    individual, self.uav_group, self.missile_configs
                )
                if fit > best_fitness:
                    best_fitness = fit
                    best_params = individual.copy()
                    best_success = det["success_count"]

            if (gen + 1) % 5 == 0:
                print(f"  微调第{gen+1}代: 最优={best_fitness:.2f}, 拦截={best_success}/{self.n_missiles}")

            if best_fitness >= perfect_score:
                print(f"\n  [微调成功] 第{gen+1}代达成完美拦截！")
                break

        return best_params, best_fitness, best_success

    def optimize(self) -> Tuple[List, float, Dict]:
        """执行完整优化流程"""
        t0 = time.time()

        # Phase 1
        subtask_results = self.run_phase1()

        # 基因拼接
        super_individual = self.splice_genes(subtask_results, n_uavs=len(self.uav_group))

        # Phase 2
        final_params, final_fitness, success_count = self.run_phase2(super_individual)

        total_time = time.time() - t0

        info = {
            "total_time": total_time,
            "success_count": success_count,
            "n_missiles": self.n_missiles,
            "final_fitness": final_fitness,
            "perfect_score": self.n_missiles * SUCCESS_SCORE_WEIGHT,
            "is_perfect": final_fitness >= self.n_missiles * SUCCESS_SCORE_WEIGHT,
            "n_clusters": self.allocator.n_clusters
        }

        return final_params, final_fitness, info


# ========================== 便捷函数 ==========================

def run_distributed_optimization(uav_group: List[int], missile_configs: List[Dict],
                                  pop_size: int = 20, max_generations: int = 20,
                                  n_threads: int = None,
                                  spatial_threshold: float = None,
                                  temporal_threshold: float = None) -> Tuple[List, float, Dict]:
    """运行 M2 分布式优化（spatial_threshold/temporal_threshold 用于灵敏度分析注入）"""
    optimizer = DistributedOptimizer(
        uav_group=uav_group,
        missile_configs=missile_configs,
        pop_size=pop_size,
        max_generations=max_generations,
        n_threads=n_threads,
        spatial_threshold=spatial_threshold,
        temporal_threshold=temporal_threshold,
    )

    return optimizer.optimize()
