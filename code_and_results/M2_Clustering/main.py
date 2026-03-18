"""
M2_Clustering 主入口 — 时空聚类优化

使用方法:
  python main.py --mode generate              # 生成随机导弹雨
  python main.py --mode optimize              # 运行优化（传统GA）
  python main.py --mode test                  # 测试模式（5枚导弹，10代）
  python main.py --mode quick_verify          # 快速验证（5代，验证专家先验）
  python main.py --mode distributed           # 分布式解耦优化（Wave 1）

  === 对照实验矩阵 ===
  python main.py --mode wave1                 # Wave 1 战役 (5v5, pop=30, gen=150) [Baseline]
  python main.py --mode wave2                 # Wave 2 战役 (5v10) [Proposed - 分布式聚类]
  python main.py --mode wave2_baseline        # Wave 2 战役 (5v10) [Baseline - 50维盲搜]
  python main.py --mode wave3                 # Wave 3 战役 (5v20) [Baseline - 50维盲搜]
  python main.py --mode wave3_distributed     # Wave 3 战役 (5v20) [Proposed - 分布式聚类]

  # ===== 新增：多轮并行评估（4 种攻击模式 × 10 次采样）=====
  python main.py --mode multi_eval --pattern mode_1_single_sector
  python main.py --mode multi_eval --pattern mode_2_orthogonal_pincer
  python main.py --mode multi_eval --pattern mode_3_asymmetric_saturation
  python main.py --mode multi_eval --pattern mode_4_full_360_swarm
  python main.py --mode multi_eval --pattern mode_1_single_sector --n_evals 10 --n_workers 4
"""

import os
import sys

# 在 import numpy 之前锁定底层 BLAS/MKL 线程数为 1，
# 防止多进程场景下 MKL Fortran runtime 线程冲突（forrtl error 200）
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
import json
import numpy as np
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== 第一步：ROOT_DIR注入（指向终极实验根目录）=====
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ===== 第二步：导入shared模块 =====
from shared.config.params import (
    UAV_PARAMS, get_uav_bounds, SMOKE_PAYLOAD_PER_UAV,
    compute_global_tdrop_range, SUCCESS_SCORE_WEIGHT
)
from shared.data.threat_generator import ThreatGenerator
from shared.simulation.single import (
    simulate_missile_rain, multi_missile_objective, compute_perfect_score
)
from shared.optimizer.hybrid_ga_v3 import HybridGeneticAlgorithmV3

# ===== 第三步：导入本地模块 =====
from optimizer.distributed_optimizer import DistributedOptimizer, run_distributed_optimization


def run_generate(seed=42, output_dir="data"):
    """生成随机导弹雨"""
    print("=" * 60)
    print("导弹威胁生成器")
    print("=" * 60)

    generator = ThreatGenerator(seed=seed)
    generator.generate_from_config({"A": 2, "B": 1, "C": 2})
    generator.print_summary()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"threat_{seed}.json")
    generator.save_to_json(output_path)

    return generator


def run_optimize(missile_configs, uav_group=None, pop_size=30, max_generations=100, n_threads=None):
    """
    运行导弹雨防御优化

    Args:
        missile_configs: 导弹配置列表
        uav_group: UAV编号列表（默认使用全部5架）
        pop_size: 种群大小
        max_generations: 最大代数
        n_threads: 并行进程数（默认自动检测）
    """
    print("=" * 70)
    print("  Baseline3 导弹雨防御优化")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if uav_group is None:
        uav_group = [1, 2, 3, 4, 5]

    n_missiles = len(missile_configs)
    print(f"\n[配置] 导弹数量: {n_missiles}")
    print(f"[配置] UAV数量: {len(uav_group)}")
    print(f"[配置] 种群大小: {pop_size}")
    print(f"[配置] 最大代数: {max_generations}")
    print(f"[配置] 并行进程: {n_threads if n_threads else '自动检测'}")

    # 计算全局时间窗（废除静态绑定）
    missile_positions = [m["position"] for m in missile_configs]
    global_tdrop_range = compute_global_tdrop_range(missile_positions, UAV_PARAMS["init_pos"])
    print(f"\n[时间窗] 全局投放窗: [{global_tdrop_range[0]:.1f}, {global_tdrop_range[1]:.1f}]s")

    # 计算完美分数（熔断阈值）
    perfect_score = compute_perfect_score(n_missiles)
    print(f"[目标] 完美分数: {perfect_score} (需拦截全部 {n_missiles} 枚)")

    # 生成边界
    bounds = get_uav_bounds(global_tdrop_range)
    full_bounds = bounds * len(uav_group)

    # 创建优化器（强制使用进程池）
    optimizer = HybridGeneticAlgorithmV3(
        objective_func=multi_missile_objective,
        bounds=full_bounds,
        uav_group=uav_group,
        missile_configs=missile_configs,
        pop_size=pop_size,
        max_generations=max_generations,
        n_threads=n_threads
    )

    # 运行优化
    t0 = time.time()
    best_params, best_fitness = optimizer.optimize(global_tdrop_range)
    elapsed = time.time() - t0

    # 分析结果
    success_count = int(best_fitness // SUCCESS_SCORE_WEIGHT)
    failed_score = best_fitness - success_count * SUCCESS_SCORE_WEIGHT

    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)
    print(f"最优适应度: {best_fitness:.2f}")
    print(f"成功拦截: {success_count}/{n_missiles} 枚")
    print(f"失败导弹得分: {failed_score:.2f}")
    print(f"耗时: {elapsed:.1f}s")

    if best_fitness >= perfect_score:
        print("\n[完美防御] 全部导弹成功拦截！")
    else:
        print(f"\n[部分防御] 还差 {n_missiles - success_count} 枚未拦截")

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return best_params, best_fitness


def load_benchmark(wave_name):
    """加载Benchmark JSON文件（使用绝对路径）"""
    data_dir = os.path.join(ROOT_DIR, 'data')
    filepath = os.path.join(data_dir, f"{wave_name}.json")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[Benchmark] 加载: {filepath}")
    print(f"[Benchmark] 导弹数: {data['metadata']['total_count']}")
    print(f"[Benchmark] 时间窗: {data['metadata']['time_window']}")

    return data["missiles"]


def run_test():
    """快速测试模式"""
    print("=" * 70)
    print("  Baseline3 测试模式 (5枚导弹, 10代)")
    print("=" * 70)

    # 使用Wave 1 Benchmark进行测试
    missile_configs = load_benchmark("Wave_1_5_Missiles")

    # 快速优化
    best_params, best_fitness = run_optimize(
        missile_configs,
        uav_group=[1, 2, 3, 4, 5],
        pop_size=10,
        max_generations=10,
        n_threads=4
    )

    return best_params, best_fitness


def run_quick_verify():
    """
    快速验证模式：验证专家先验注入效果

    参数：pop=30, gen=5（仅运行5代）
    目标：前5代内突破10000分大关（拦截1枚导弹）
    """
    print("\n" + "=" * 70)
    print("  [降维打击验证] 快速验证模式")
    print("  参数: pop_size=30, max_generations=5")
    print("  目标: 验证专家先验注入效果，前5代突破10000分")
    print("=" * 70)

    missile_configs = load_benchmark("Wave_1_5_Missiles")

    best_params, best_fitness = run_optimize(
        missile_configs,
        uav_group=[1, 2, 3, 4, 5],
        pop_size=30,
        max_generations=5
    )

    # 判断结果
    if best_fitness >= 10000:
        print("\n[验证成功] 专家先验注入有效！")
        print(f"  适应度: {best_fitness:.2f} >= 10000")
    else:
        print(f"\n[验证结果] 适应度: {best_fitness:.2f} < 10000")
        print("  建议：检查allocator参数或增加优化代数")

    return best_params, best_fitness


def run_distributed():
    """
    两阶段分布式解耦优化模式

    Phase 1: 5个子任务独立优化（10维空间）
    Phase 2: 基因拼接与全局微调（50维空间）

    目标：达成 5/5 完美拦截，耗时 < 10分钟
    """
    print("\n" + "=" * 70)
    print("  两阶段分布式解耦优化 (Proposed Strategy)")
    print("  Phase 1: 分布式独立优化 (5个子任务 × 10维)")
    print("  Phase 2: 基因拼接 + 零代验证 + 全局微调")
    print("=" * 70)

    missile_configs = load_benchmark("Wave_1_5_Missiles")

    t0 = time.time()

    # 运行分布式优化
    best_params, best_fitness, info = run_distributed_optimization(
        uav_group=[1, 2, 3, 4, 5],
        missile_configs=missile_configs,
        pop_size=20,
        max_generations=20
    )

    total_time = time.time() - t0

    # 打印最终结果
    print("\n" + "=" * 70)
    print("  最终战报")
    print("=" * 70)
    print(f"  最终适应度: {best_fitness:.2f}")
    print(f"  成功拦截: {info['success_count']}/5")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")

    if info['is_perfect']:
        print("\n" + "=" * 70)
        print("  [完美防御] 5/5 全部拦截成功！")
        print("=" * 70)
    else:
        print(f"\n  [部分防御] 还差 {5 - info['success_count']} 枚未拦截")

    return best_params, best_fitness, info


def run_wave1():
    """Wave 1 战役: 5v5 (pop=30, gen=150)"""
    print("\n" + "=" * 70)
    print("  WAVE 1 战役: 5 UAVs vs 5 Missiles")
    print("  参数: pop_size=30, max_generations=150")
    print("=" * 70)

    missile_configs = load_benchmark("Wave_1_5_Missiles")

    best_params, best_fitness = run_optimize(
        missile_configs,
        uav_group=[1, 2, 3, 4, 5],
        pop_size=30,
        max_generations=150
    )

    return best_params, best_fitness


def run_wave2():
    """
    Wave 2 战役: 5v10 饱和攻击 (时空聚类 + 分布式优化) - Proposed Strategy

    策略：
    1. 时空聚类：将10枚导弹打包成最多5个Cluster
    2. 匈牙利匹配：将5架UAV分配给Cluster
    3. 分布式优化：每个子任务独立优化10维空间
    4. 基因拼接：组合成50维超级个体
    5. 全局微调：处理跨Cluster串扰
    """
    print("\n" + "=" * 70)
    print("  WAVE 2 战役: 5 UAVs vs 10 Missiles (时空聚类 + 分布式优化)")
    print("  策略: 时空聚类 -> 匈牙利匹配 -> 分布式优化 -> 基因拼接")
    print("  参数: pop_size=20, max_generations=20 (子任务)")
    print("=" * 70)

    missile_configs = load_benchmark("Wave_2_10_Missiles")

    t0 = time.time()

    # 运行分布式优化（使用时空聚类）
    best_params, best_fitness, info = run_distributed_optimization(
        uav_group=[1, 2, 3, 4, 5],
        missile_configs=missile_configs,
        pop_size=20,
        max_generations=20
    )

    total_time = time.time() - t0

    # 打印最终结果
    print("\n" + "=" * 70)
    print("  WAVE 2 最终战报 (Proposed Strategy)")
    print("=" * 70)
    print(f"  最终适应度: {best_fitness:.2f}")
    print(f"  成功拦截: {info['success_count']}/{info['n_missiles']}")
    print(f"  Cluster数量: {info['n_clusters']}")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")

    if info['is_perfect']:
        print("\n" + "=" * 70)
        print(f"  [完美防御] {info['success_count']}/{info['n_missiles']} 全部拦截成功！")
        print("=" * 70)
    else:
        print(f"\n  [部分防御] 还差 {info['n_missiles'] - info['success_count']} 枚未拦截")

    return best_params, best_fitness, info


def run_wave2_baseline():
    """
    Wave 2 战役: 5v10 - Baseline对照组 (纯50维GA盲搜)

    使用原汁原味的 HybridGeneticAlgorithmV3，不加任何专家先验和聚类
    用于证明 Baseline 在 10 枚导弹饱和攻击下的无能为力
    """
    print("\n" + "=" * 70)
    print("  WAVE 2 战役 (BASELINE对照组): 5 UAVs vs 10 Missiles")
    print("  算法: 纯50维GA盲搜 (HybridGeneticAlgorithmV3)")
    print("  参数: pop_size=50, max_generations=200")
    print("=" * 70)

    missile_configs = load_benchmark("Wave_2_10_Missiles")

    best_params, best_fitness = run_optimize(
        missile_configs,
        uav_group=[1, 2, 3, 4, 5],
        pop_size=50,
        max_generations=200
    )

    return best_params, best_fitness


def run_wave3():
    """Wave 3 战役: 5v20 - Baseline对照组 (纯50维GA盲搜)"""
    print("\n" + "=" * 70)
    print("  WAVE 3 战役 (BASELINE对照组): 5 UAVs vs 20 Missiles")
    print("  算法: 纯50维GA盲搜 (HybridGeneticAlgorithmV3)")
    print("  参数: pop_size=100, max_generations=300")
    print("=" * 70)

    missile_configs = load_benchmark("Wave_3_20_Missiles")

    best_params, best_fitness = run_optimize(
        missile_configs,
        uav_group=[1, 2, 3, 4, 5],
        pop_size=100,
        max_generations=300
    )

    return best_params, best_fitness


def run_wave3_distributed(pop_size=30, max_gen=50, n_threads=None):
    """
    Wave 3 战役: 5v20 - Proposed Strategy (时空聚类 + 分布式优化)

    极限大考：5架飞机 vs 20枚导弹
    物理极限：15颗烟雾弹 (5架 × 3颗)
    """
    print("\n" + "=" * 70)
    print("  WAVE 3 战役 (Proposed Strategy): 5 UAVs vs 20 Missiles")
    print("  策略: 时空聚类 -> 匈牙利匹配 -> 分布式优化 -> 基因拼接")
    print("  物理极限: 15颗烟雾弹 (5架 × 3颗)")
    print(f"  参数: pop_size={pop_size}, max_generations={max_gen}")
    print("=" * 70)

    missile_configs = load_benchmark("Wave_3_20_Missiles")

    t0 = time.time()

    # 运行分布式优化（使用时空聚类）
    best_params, best_fitness, info = run_distributed_optimization(
        uav_group=[1, 2, 3, 4, 5],
        missile_configs=missile_configs,
        pop_size=pop_size,
        max_generations=max_gen,
        n_threads=n_threads
    )

    total_time = time.time() - t0

    # 打印最终结果
    print("\n" + "=" * 70)
    print("  WAVE 3 最终战报 (Proposed Strategy)")
    print("=" * 70)
    print(f"  最终适应度: {best_fitness:.2f}")
    print(f"  成功拦截: {info['success_count']}/{info['n_missiles']}")
    print(f"  Cluster数量: {info['n_clusters']}")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")

    if info['is_perfect']:
        print("\n" + "=" * 70)
        print(f"  [完美防御] {info['success_count']}/{info['n_missiles']} 全部拦截成功！")
        print("=" * 70)
    else:
        print(f"\n  [部分防御] 还差 {info['n_missiles'] - info['success_count']} 枚未拦截")

    return best_params, best_fitness, info


# ============================================================
# 顶层评估函数（必须定义在模块级别，ProcessPoolExecutor 可序列化）
# ============================================================

def _evaluate_single_run_m2(args):
    """
    M2 单次评估顶层函数（subprocess 安全）。

    Args:
        args: (eval_idx, missiles, pop_size, max_gen)

    Returns:
        结果字典：eval_idx, intercepted, smokes_used, best_fitness, n_missiles, n_clusters
    """
    eval_idx, missiles, pop_size, max_gen = args

    # 子进程中锁定 BLAS/MKL 线程 + 重新确保路径（Windows spawn 模式安全措施）
    import os as _os, sys as _sys
    _os.environ["MKL_NUM_THREADS"] = "1"
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["NUMEXPR_NUM_THREADS"] = "1"

    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _cur  = _os.path.dirname(_os.path.abspath(__file__))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    if _cur not in _sys.path:
        _sys.path.insert(0, _cur)

    from optimizer.distributed_optimizer import run_distributed_optimization

    best_params, best_fitness, info = run_distributed_optimization(
        uav_group=[1, 2, 3, 4, 5],
        missile_configs=missiles,
        pop_size=pop_size,
        max_generations=max_gen,
        n_threads=1,  # 子进程内禁用嵌套多进程
    )

    return {
        "eval_idx":    eval_idx,
        "best_fitness": float(best_fitness),
        "intercepted": int(info.get("success_count", 0)),
        "smokes_used": 15,  # M2 无弹药惩罚，固定耗弹 5×3=15
        "n_missiles":  int(info.get("n_missiles", len(missiles))),
        "n_clusters":  int(info.get("n_clusters", 0)),
    }


def run_multi_eval_m2(
    pattern_name: str,
    n_evals: int = 10,
    pop_size: int = 30,
    max_gen: int = 50,
    n_workers: int = None,
) -> dict:
    """
    M2 多轮并行评估（ProcessPoolExecutor）。

    Args:
        pattern_name: 攻击模式名称
        n_evals:      评估次数（默认 10）
        pop_size:     GA 种群大小（默认 30）
        max_gen:      GA 最大代数（默认 50）
        n_workers:    并行 worker 数（默认取 CPU 核心数）

    Returns:
        汇总结果字典
    """
    from shared.data.missile_pool_generator import load_missile_pool, sample_from_pool

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), n_evals)

    pool_path = os.path.join(ROOT_DIR, "data", "missile_pools",
                             f"{pattern_name}_pool.json")
    missile_pool = load_missile_pool(pool_path)

    print(f"\n{'='*70}")
    print(f"  M2_Clustering | Multi-Eval: {pattern_name}")
    print(f"  Evaluations: {n_evals}  Workers: {n_workers}"
          f"  Pop={pop_size}  Gen={max_gen}")
    print(f"{'='*70}")

    eval_tasks = []
    for eval_idx in range(n_evals):
        missiles = sample_from_pool(missile_pool, n_samples=20, seed=eval_idx)
        eval_tasks.append((eval_idx, missiles, pop_size, max_gen))

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_evaluate_single_run_m2, task): task[0]
                   for task in eval_tasks}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(f"  [Eval {r['eval_idx']+1:2d}/{n_evals}]"
                      f"  Intercepted: {r['intercepted']:2d}/20"
                      f"  Clusters: {r['n_clusters']:2d}"
                      f"  Smokes: {r['smokes_used']:2d}"
                      f"  Fitness: {r['best_fitness']:.1f}")
            except Exception as e:
                print(f"  [Eval {idx+1:2d}] FAILED: {e}")
                results.append({
                    "eval_idx": idx, "best_fitness": 0.0,
                    "intercepted": 0, "smokes_used": 15,
                    "n_missiles": 20, "n_clusters": 0,
                    "error": str(e),
                })

    results.sort(key=lambda x: x["eval_idx"])

    ok = [r for r in results if "error" not in r]
    intercepted_list = [r["intercepted"] for r in ok]
    smokes_list      = [r["smokes_used"]  for r in ok]

    summary = {
        "pattern":          pattern_name,
        "method":           "M2_Clustering",
        "n_evals":          n_evals,
        "n_workers":        n_workers,
        "pop_size":         pop_size,
        "max_gen":          max_gen,
        "intercepted_mean": float(np.mean(intercepted_list))  if ok else 0.0,
        "intercepted_std":  float(np.std(intercepted_list))   if ok else 0.0,
        "intercepted_max":  int(max(intercepted_list))        if ok else 0,
        "intercepted_min":  int(min(intercepted_list))        if ok else 0,
        "smokes_mean":      float(np.mean(smokes_list))       if ok else 15.0,
        "smokes_std":       float(np.std(smokes_list))        if ok else 0.0,
        "efficiency_mean":  (float(np.mean(intercepted_list)) /
                             float(np.mean(smokes_list)))     if ok else 0.0,
        "results":          results,
    }

    print(f"\n{'='*70}")
    print(f"  SUMMARY: M2_Clustering | {pattern_name}")
    print(f"  Intercepted : {summary['intercepted_mean']:.2f} ± "
          f"{summary['intercepted_std']:.2f}  "
          f"[{summary['intercepted_min']}, {summary['intercepted_max']}]")
    print(f"  Smokes Used : {summary['smokes_mean']:.2f} ± "
          f"{summary['smokes_std']:.2f}")
    print(f"  Efficiency  : {summary['efficiency_mean']:.3f}")
    print(f"{'='*70}\n")

    out_dir = os.path.join(ROOT_DIR, "M2_Clustering", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"M2_{pattern_name}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="M2_Clustering 导弹雨防御大阵")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["generate", "optimize", "test", "quick_verify", "distributed",
                                 "wave1", "wave2", "wave2_baseline", "wave3", "wave3_distributed",
                                 "multi_eval"],
                        help="运行模式")
    parser.add_argument("--seed",      type=int,  default=42,   help="随机种子")
    parser.add_argument("--pop_size",  type=int,  default=30,   help="种群大小")
    parser.add_argument("--max_gen",   type=int,  default=100,  help="最大代数")
    parser.add_argument("--n_threads", type=int,  default=None, help="并行进程数（wave 用）")
    parser.add_argument("--pattern",   type=str,
                        default="mode_1_single_sector",
                        choices=["mode_1_single_sector", "mode_2_orthogonal_pincer",
                                 "mode_3_asymmetric_saturation", "mode_4_full_360_swarm"],
                        help="攻击模式（multi_eval 用）")
    parser.add_argument("--n_evals",   type=int,  default=10,   help="评估次数（multi_eval 用）")
    parser.add_argument("--n_workers", type=int,  default=None, help="并行 worker 数（multi_eval 用）")

    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(seed=args.seed)
    elif args.mode == "optimize":
        generator = run_generate(seed=args.seed)
        run_optimize(
            generator.missiles,
            pop_size=args.pop_size,
            max_generations=args.max_gen,
            n_threads=args.n_threads
        )
    elif args.mode == "test":
        run_test()
    elif args.mode == "quick_verify":
        run_quick_verify()
    elif args.mode == "distributed":
        run_distributed()
    elif args.mode == "wave1":
        run_wave1()
    elif args.mode == "wave2":
        run_wave2()
    elif args.mode == "wave2_baseline":
        run_wave2_baseline()
    elif args.mode == "wave3":
        run_wave3()
    elif args.mode == "wave3_distributed":
        run_wave3_distributed(pop_size=args.pop_size, max_gen=args.max_gen,
                              n_threads=args.n_threads)
    elif args.mode == "multi_eval":
        run_multi_eval_m2(
            pattern_name=args.pattern,
            n_evals=args.n_evals,
            pop_size=args.pop_size,
            max_gen=args.max_gen,
            n_workers=args.n_workers,
        )


if __name__ == "__main__":
    main()
