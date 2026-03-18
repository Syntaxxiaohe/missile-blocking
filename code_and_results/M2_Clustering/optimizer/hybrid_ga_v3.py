"""
知识驱动混合遗传算法 (KI-HGA) — Baseline3 多导弹雨版

核心升级（扫雷修复）：
  1. 【战术熔断阈值动态化】：完美分数 = 导弹数 * 10000
  2. 【废除静态绑定】：使用全局时间窗替代 UAV_MISSILE_MAP
  3. 支持动态导弹配置
  4. 【性能修复】：强制使用 ProcessPoolExecutor（计算密集型RK4必须用进程池）
  5. 【降维打击】：匈牙利算法目标分配 + 专家先验注入
"""

import os
import pickle
import json
import random
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from config.params import (
    UAV_PARAMS, SMOKE_PAYLOAD_PER_UAV, ACCEL_PARAMS,
    SUCCESS_SCORE_WEIGHT, compute_global_tdrop_range
)
from simulation.single import compute_perfect_score
from optimizer.allocator import SpatioTemporalAllocator


# 【关键修复】顶层函数用于进程池并行评估（lambda无法pickle）
def _evaluate_individual(args):
    """评估单个个体（进程池兼容）"""
    objective_func, individual, uav_group, missile_configs = args
    return objective_func(individual, uav_group, missile_configs)


class HybridGeneticAlgorithmV3:
    """
    Baseline3 遗传算法优化器（导弹雨场景）
    """

    def __init__(self, objective_func, bounds, uav_group, missile_configs,
                 pop_size=50, max_generations=500,
                 crossover_rate=0.9, mutation_rate=0.2, n_threads=None,
                 checkpoint_dir="checkpoints"):
        """
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            uav_group: 无人机编号列表
            missile_configs: 导弹配置列表（动态）
            pop_size: 种群大小
            max_generations: 最大代数
            n_threads: 并行进程数（默认使用CPU核心数）
            checkpoint_dir: 检查点目录
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.uav_group = uav_group
        self.missile_configs = missile_configs
        self.n_missiles = len(missile_configs)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        # 【性能修复】默认使用全部CPU核心
        self.n_threads = n_threads if n_threads else os.cpu_count()
        self.checkpoint_dir = checkpoint_dir

        # 【关键修复1】动态计算完美分数（熔断阈值）
        self.perfect_score = compute_perfect_score(self.n_missiles)
        print(f"[Baseline3] 导弹数量: {self.n_missiles}, 完美分数: {self.perfect_score}")

        os.makedirs(checkpoint_dir, exist_ok=True)

        group_str = "_".join(map(str, uav_group))
        self.checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_v3_{group_str}.pkl")
        self.result_file = os.path.join(checkpoint_dir, f"result_v3_{group_str}.json")

        # 尝试加载checkpoint
        loaded = self._load_checkpoint()
        if loaded:
            print(f"[Checkpoint] 已加载断点: 第 {self.current_gen} 代, 最优: {self.best_fitness:.2f}")
        else:
            self.population = None
            self.best_fitness = -np.inf
            self.best_params = None
            self.current_gen = 0

    def _save_checkpoint(self, generation):
        checkpoint = {
            "generation": generation,
            "population": self.population,
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "uav_group": self.uav_group,
            "n_missiles": self.n_missiles,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return False
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            if checkpoint["n_missiles"] != self.n_missiles:
                return False
            self.population = checkpoint["population"]
            self.best_params = checkpoint["best_params"]
            self.best_fitness = checkpoint["best_fitness"]
            self.current_gen = checkpoint["generation"]
            return True
        except Exception as e:
            print(f"[Checkpoint] 加载失败: {e}")
            return False

    def _save_result(self, generation, finished=False, reason=""):
        result = {
            "uav_group": self.uav_group,
            "n_missiles": self.n_missiles,
            "generation": generation,
            "best_fitness": float(self.best_fitness) if self.best_fitness else None,
            "best_params": [float(x) for x in self.best_params] if self.best_params else None,
            "perfect_score": self.perfect_score,
            "finished": finished,
            "termination_reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def clear_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

    def _enforce_constraints(self, individual, global_tdrop_range):
        """强制约束（使用全局时间窗）"""
        gene_len = 4 + 2 * SMOKE_PAYLOAD_PER_UAV
        enforced = []
        t_drop_min, t_drop_max = global_tdrop_range

        for uav_idx in range(len(self.uav_group)):
            start = uav_idx * gene_len
            uav_ind = individual[start:start + gene_len]
            dir_x, dir_y, dir_z, speed = uav_ind[0:4]
            t_drops = list(uav_ind[4:4+SMOKE_PAYLOAD_PER_UAV])
            d_dets = list(uav_ind[4+SMOKE_PAYLOAD_PER_UAV:4+2*SMOKE_PAYLOAD_PER_UAV])

            # 速度约束
            speed = np.clip(speed, UAV_PARAMS["speed_range"][0], UAV_PARAMS["speed_range"][1])

            # 方向归一化
            dir_norm = max(np.sqrt(dir_x**2 + dir_y**2 + dir_z**2), 1e-6)
            dir_x, dir_y, dir_z = dir_x / dir_norm, dir_y / dir_norm, dir_z / dir_norm

            # 【关键修复2】使用全局时间窗（废除静态绑定）
            for idx in range(SMOKE_PAYLOAD_PER_UAV):
                if idx == 0:
                    t_drops[idx] = np.clip(t_drops[idx], t_drop_min, t_drop_max)
                else:
                    if t_drops[idx] - t_drops[idx-1] < 1.0:
                        t_drops[idx] = min(t_drops[idx-1] + random.uniform(1.0, 3.0), t_drop_max)
                    t_drops[idx] = np.clip(t_drops[idx], t_drop_min, t_drop_max)

            for idx in range(SMOKE_PAYLOAD_PER_UAV):
                d_dets[idx] = np.clip(d_dets[idx], 6.0, 10.0)

            uav_part = [dir_x, dir_y, dir_z, speed] + t_drops + d_dets
            enforced.extend(uav_part)
        return enforced

    def _init_population(self, global_tdrop_range):
        """
        初始化种群（注入专家先验）

        核心升级：使用匈牙利算法进行UAV-导弹匹配，
        将40%的个体用专家先验初始化，实现降维打击！
        """
        gene_len = 4 + 2 * SMOKE_PAYLOAD_PER_UAV
        population = []

        # ============ 降维打击：匈牙利算法目标分配 ============
        print("\n[降维打击] 启动时空目标分配器...")
        allocator = SpatioTemporalAllocator(self.uav_group, self.missile_configs)

        # 执行匈牙利匹配
        allocator.compute_cost_matrix()
        allocator.hungarian_assignment()
        allocator.print_assignment()

        # 生成专家先验基因
        expert_genes = allocator.generate_expert_genes(window_before=12.0, window_after=2.0)

        print("\n[专家先验] 注入参数:")
        for uav_id, gene in sorted(expert_genes.items()):
            print(f"  UAV{uav_id} -> M{gene['target_missile']}: "
                  f"时间窗=[{gene['time_window'][0]:.1f}s, {gene['time_window'][1]:.1f}s]")

        # 计算导弹质心位置（用于随机个体的方向引导）
        avg_missile_pos = np.mean([m["position"] for m in self.missile_configs], axis=0)

        # ============ 混合初始化 ============
        # 前40%：专家先验个体（神枪手）
        # 后60%：随机个体（保持多样性）
        n_expert = int(self.pop_size * 0.4)

        for i in range(self.pop_size):
            individual = [random.uniform(b[0], b[1]) for b in self.bounds]

            if i < n_expert:
                # ========== 专家先验注入 ==========
                for uav_idx, uav_id in enumerate(self.uav_group):
                    if uav_id not in expert_genes:
                        continue

                    expert = expert_genes[uav_id]
                    start = uav_idx * gene_len

                    # 注入精确航向
                    dx, dy, dz = expert["direction"]
                    individual[start] = dx
                    individual[start + 1] = dy
                    individual[start + 2] = dz

                    # 注入专属时间窗
                    t_min, t_max = expert["time_window"]

                    # 三发烟幕在专属窗口内均匀分布
                    for k in range(SMOKE_PAYLOAD_PER_UAV):
                        t_base = t_min + (t_max - t_min) * k / SMOKE_PAYLOAD_PER_UAV
                        t_random = random.uniform(t_base, t_base + 2.0)
                        individual[start + 4 + k] = np.clip(t_random, t_min, t_max)
            else:
                # ========== 随机初始化 + 质心方向引导 ==========
                for uav_idx in range(len(self.uav_group)):
                    uav_num = self.uav_group[uav_idx]
                    uav_pos = UAV_PARAMS["init_pos"][uav_num]

                    dir_vec = np.array(avg_missile_pos) - uav_pos
                    dir_vec[2] = 0
                    dir_norm = max(np.linalg.norm(dir_vec), 1e-6)
                    unit_dir = dir_vec / dir_norm
                    target_angle = np.arctan2(unit_dir[1], unit_dir[0])
                    random_angle = random.uniform(-np.pi / 12, np.pi / 12)

                    individual[uav_idx * gene_len] = np.cos(target_angle + random_angle)
                    individual[uav_idx * gene_len + 1] = np.sin(target_angle + random_angle)

            individual = self._enforce_constraints(individual, global_tdrop_range)
            population.append(individual)

        print(f"\n[种群初始化] 完成: {n_expert}个专家个体 + {self.pop_size - n_expert}个随机个体")
        return population

    def _local_search(self, individual, local_step=0.5):
        """局部搜索"""
        gene_len = 4 + 2 * SMOKE_PAYLOAD_PER_UAV
        best_local = individual.copy()
        best_fitness = self.objective_func(best_local, self.uav_group, self.missile_configs)[0]

        for uav_idx in range(len(self.uav_group)):
            for param_idx in range(4, 4 + SMOKE_PAYLOAD_PER_UAV * 2):
                param_pos = uav_idx * gene_len + param_idx
                for delta in [-local_step, local_step]:
                    temp = best_local.copy()
                    temp[param_pos] += delta
                    fitness = self.objective_func(temp, self.uav_group, self.missile_configs)[0]
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_local = temp.copy()

        return best_local

    def _selection(self, fitness_scores):
        """轮盘赌选择"""
        total = sum(max(s, 0) for s in fitness_scores)
        if total <= 0:
            return random.choice(self.population)
        probs = [max(s, 0) / total for s in fitness_scores]
        return random.choices(self.population, probs)[0]

    def _crossover(self, p1, p2):
        """单点交叉"""
        gene_len = 4 + 2 * SMOKE_PAYLOAD_PER_UAV
        if random.random() < self.crossover_rate:
            cross_points = [gene_len * i for i in range(1, len(self.uav_group))]
            if cross_points:
                cp = random.choice(cross_points)
                child = p1[:cp] + p2[cp:]
                return child
        return p1.copy()

    def _mutation(self, individual, generation, global_tdrop_range):
        """自适应变异"""
        rate = self.mutation_rate * (1 - generation / self.max_generations)
        for i in range(len(individual)):
            if random.random() < rate:
                lb, ub = self.bounds[i]
                rng = ub - lb
                individual[i] = np.clip(individual[i] + random.uniform(-0.1 * rng, 0.1 * rng), lb, ub)
        return self._enforce_constraints(individual, global_tdrop_range)

    def optimize(self, global_tdrop_range=None):
        """
        优化主流程

        【关键修复】熔断阈值动态化：
        - 完美防御分数 = N_missiles * 10000
        - 只有达到完美分数，或最大代数，才允许下班
        """
        if global_tdrop_range is None:
            global_tdrop_range = (20.0, 50.0)

        # 初始化种群
        if self.population is None or len(self.population) == 0:
            self.population = self._init_population(global_tdrop_range)

        start_gen = getattr(self, 'current_gen', 0)
        if start_gen > 0:
            print(f"[断点续传] 从第 {start_gen} 代继续...")

        # 【性能修复】强制使用进程池（计算密集型RK4必须用进程池绕过GIL）
        print(f"[并行配置] 使用 ProcessPoolExecutor, workers={self.n_threads}")

        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            for gen in tqdm(range(start_gen, self.max_generations), desc="Baseline3优化"):
                # 并行适应度计算（使用顶层函数避免pickle问题）
                args_list = [(self.objective_func, ind, self.uav_group, self.missile_configs)
                             for ind in self.population]
                results = list(executor.map(_evaluate_individual, args_list))
                fitness_scores = [res[0] for res in results]

                # 更新最优
                best_idx = np.argmax(fitness_scores)
                if fitness_scores[best_idx] > self.best_fitness:
                    self.best_fitness = fitness_scores[best_idx]
                    self.best_params = self.population[best_idx].copy()

                # 【关键修复1】动态熔断阈值
                # 只有达到完美分数（所有导弹都被拦截）才提前终止
                if self.best_fitness >= self.perfect_score:
                    print(f"\n[完美熔断] 全部 {self.n_missiles} 枚导弹成功拦截！")
                    print(f"  适应度: {self.best_fitness:.2f} (完美分数: {self.perfect_score})")
                    self._save_result(gen + 1, finished=True, reason="PERFECT_DEFENSE")
                    self.clear_checkpoint()
                    return self.best_params, self.best_fitness

                # 极速熔断：FOV脱锁（备用，单导弹可能触发）
                if self.best_fitness >= 99000.0:
                    print(f"\n[极速熔断] FOV脱锁！Fitness: {self.best_fitness:.2f}")
                    self._save_result(gen + 1, finished=True, reason="FOV_BREAK_LOCK")
                    self.clear_checkpoint()
                    return self.best_params, self.best_fitness

                # 周期性局部搜索
                if gen % 100 == 0 and gen > 0:
                    local_best = self._local_search(self.population[best_idx])
                    local_fitness = self.objective_func(local_best, self.uav_group, self.missile_configs)[0]
                    if local_fitness > self.best_fitness:
                        self.best_fitness = local_fitness
                        self.best_params = local_best.copy()
                        self.population[0] = local_best

                # 生成下一代
                new_pop = []
                elite_size = int(self.pop_size * 0.2)
                elite_indices = np.argsort(fitness_scores)[-elite_size:]
                for idx in elite_indices:
                    new_pop.append(self.population[idx].copy())

                while len(new_pop) < self.pop_size:
                    p1 = self._selection(fitness_scores)
                    p2 = self._selection(fitness_scores)
                    child = self._crossover(p1, p2)
                    child = self._mutation(child, gen, global_tdrop_range)
                    new_pop.append(child)

                self.population = new_pop

                # 保存checkpoint
                if (gen + 1) % 10 == 0:
                    self._save_checkpoint(gen + 1)
                self._save_result(gen + 1, finished=False)

                if (gen + 1) % 20 == 0:
                    success_count = int(self.best_fitness // SUCCESS_SCORE_WEIGHT)
                    print(f"\n迭代{gen+1}/{self.max_generations} | "
                          f"最优: {self.best_fitness:.2f} | "
                          f"成功拦截: {success_count}/{self.n_missiles}", flush=True)

        self._save_result(self.max_generations, finished=True, reason="MAX_GENERATIONS")
        self.clear_checkpoint()
        return self.best_params, self.best_fitness


# 兼容性别名
HybridGeneticAlgorithm = HybridGeneticAlgorithmV3
