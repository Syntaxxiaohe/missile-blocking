"""
M1_NaiveGA 主入口 — 均匀分配基线

基因维度: 10维/UAV × 5 = 50维
策略: 轮询分配 + 专家先验（先验撕裂）
预期效果: 防线崩溃

使用方法:
  python main.py --mode test          # 测试模式（Pop=10, Gen=10）
  python main.py --mode wave3         # Wave 3 战役（Pop=30, Gen=50）

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

# ===== 添加当前目录到路径（用于本地模块导入）=====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

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
from optimizer.distributed_optimizer import TwoStageOptimizer


def load_missile_data(wave_name="Wave_3_20_Missiles"):
    """
    加载导弹数据（使用绝对路径）

    Args:
        wave_name: Wave名称，如 "Wave_3_20_Missiles"

    Returns:
        missile_configs: 导弹配置列表
    """
    data_path = os.path.join(ROOT_DIR, 'data', f'{wave_name}.json')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[数据加载] {data_path}")
    print(f"[数据加载] 导弹数: {data['metadata']['total_count']}")
    print(f"[数据加载] 时间窗: {data['metadata']['time_window']}")

    return data["missiles"]


def run_optimize(missile_configs, uav_group=None, pop_size=30, max_generations=100, n_threads=None):
    """
    运行导弹雨防御优化（传统GA盲搜）
    """
    print("=" * 70)
    print("  M1_NaiveGA 导弹雨防御优化")
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

    # 计算全局时间窗
    missile_positions = [m["position"] for m in missile_configs]
    global_tdrop_range = compute_global_tdrop_range(missile_positions, UAV_PARAMS["init_pos"])
    print(f"\n[时间窗] 全局投放窗: [{global_tdrop_range[0]:.1f}, {global_tdrop_range[1]:.1f}]s")

    # 计算完美分数
    perfect_score = compute_perfect_score(n_missiles)
    print(f"[目标] 完美分数: {perfect_score} (需拦截全部 {n_missiles} 枚)")

    # 生成边界
    bounds = get_uav_bounds(global_tdrop_range)
    full_bounds = bounds * len(uav_group)

    # 创建优化器
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


def run_distributed(missile_configs, pop_size=30, max_generations=50, n_threads=None):
    """
    运行两阶段分布式优化（M1版本：使用NaiveAllocator）
    """
    print("\n" + "=" * 70)
    print("  M1_NaiveGA 两阶段分布式优化")
    print("  Phase 1: 均匀分配 -> 伪Cluster -> 分布式优化")
    print("  Phase 2: 基因拼接 -> 全局评估")
    print(f"  参数: pop_size={pop_size}, max_generations={max_generations}")
    print("=" * 70)

    t0 = time.time()

    # 运行分布式优化
    optimizer = TwoStageOptimizer(
        uav_group=[1, 2, 3, 4, 5],
        missile_configs=missile_configs,
        pop_size=pop_size,
        max_generations=max_generations,
        n_threads=n_threads
    )

    best_params, best_fitness, info = optimizer.optimize()

    total_time = time.time() - t0

    # 打印最终结果
    print("\n" + "=" * 70)
    print("  M1_NaiveGA 最终战报")
    print("=" * 70)
    print(f"  最终适应度: {best_fitness:.2f}")
    print(f"  成功拦截: {info['success_count']}/{info['n_missiles']}")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")

    if info['is_perfect']:
        print("\n" + "=" * 70)
        print(f"  [完美防御] {info['success_count']}/{info['n_missiles']} 全部拦截成功！")
        print("=" * 70)
    else:
        print(f"\n  [部分防御] 还差 {info['n_missiles'] - info['success_count']} 枚未拦截")

    return best_params, best_fitness, info


def run_test():
    """快速测试模式"""
    print("=" * 70)
    print("  M1_NaiveGA 测试模式 (Pop=10, Gen=10)")
    print("=" * 70)

    missile_configs = load_missile_data("Wave_3_20_Missiles")

    best_params, best_fitness = run_optimize(
        missile_configs,
        uav_group=[1, 2, 3, 4, 5],
        pop_size=10,
        max_generations=10,
        n_threads=4
    )

    return best_params, best_fitness


def run_wave3(n_threads=None):
    """
    Wave 3 战役: 5v20 - M1基线

    参数: Pop=30, Gen=50
    预期: 先验撕裂，防线崩溃
    """
    print("\n" + "=" * 70)
    print("  WAVE 3 战役 (M1_NaiveGA): 5 UAVs vs 20 Missiles")
    print("  策略: 均匀分配 -> 伪Cluster -> 先验撕裂")
    print("  参数: pop_size=30, max_generations=50")
    print("=" * 70)

    missile_configs = load_missile_data("Wave_3_20_Missiles")

    # 使用分布式优化（带NaiveAllocator）
    best_params, best_fitness, info = run_distributed(
        missile_configs,
        pop_size=30,
        max_generations=50,
        n_threads=n_threads
    )

    return best_params, best_fitness, info


# ============================================================
# 顶层评估函数（必须定义在模块级别，ProcessPoolExecutor 可序列化）
# ============================================================

def _evaluate_single_run_m1(args):
    """
    M1 单次评估顶层函数（subprocess 安全）。

    Args:
        args: (eval_idx, missiles, pop_size, max_gen)

    Returns:
        结果字典：eval_idx, intercepted, smokes_used, best_fitness, n_missiles
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

    from optimizer.distributed_optimizer import TwoStageOptimizer

    opt = TwoStageOptimizer(
        uav_group=[1, 2, 3, 4, 5],
        missile_configs=missiles,
        pop_size=pop_size,
        max_generations=max_gen,
        n_threads=1,  # 子进程内禁用嵌套多进程
    )
    best_params, best_fitness, info = opt.optimize()

    return {
        "eval_idx":    eval_idx,
        "best_fitness": float(best_fitness),
        "intercepted": int(info.get("success_count", 0)),
        "smokes_used": 15,  # M1 无弹药惩罚，固定耗弹 5×3=15
        "n_missiles":  int(info.get("n_missiles", len(missiles))),
    }


def run_multi_eval_m1(
    pattern_name: str,
    n_evals: int = 10,
    pop_size: int = 30,
    max_gen: int = 50,
    n_workers: int = None,
) -> dict:
    """
    M1 多轮并行评估（ProcessPoolExecutor）。

    从预生成的导弹池中随机采样 n_evals 次，每次优化一轮，
    并行执行，最后统计均值/标准差/效费比。

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

    # 加载导弹池
    pool_path = os.path.join(ROOT_DIR, "data", "missile_pools",
                             f"{pattern_name}_pool.json")
    missile_pool = load_missile_pool(pool_path)

    print(f"\n{'='*70}")
    print(f"  M1_NaiveGA | Multi-Eval: {pattern_name}")
    print(f"  Evaluations: {n_evals}  Workers: {n_workers}"
          f"  Pop={pop_size}  Gen={max_gen}")
    print(f"{'='*70}")

    # 预先生成各次采样的导弹组合（seed = eval_idx，保证可复现）
    eval_tasks = []
    for eval_idx in range(n_evals):
        missiles = sample_from_pool(missile_pool, n_samples=20, seed=eval_idx)
        eval_tasks.append((eval_idx, missiles, pop_size, max_gen))

    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_evaluate_single_run_m1, task): task[0]
                   for task in eval_tasks}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(f"  [Eval {r['eval_idx']+1:2d}/{n_evals}]"
                      f"  Intercepted: {r['intercepted']:2d}/20"
                      f"  Smokes: {r['smokes_used']:2d}"
                      f"  Fitness: {r['best_fitness']:.1f}")
            except Exception as e:
                print(f"  [Eval {idx+1:2d}] FAILED: {e}")
                results.append({
                    "eval_idx": idx, "best_fitness": 0.0,
                    "intercepted": 0, "smokes_used": 15, "n_missiles": 20,
                    "error": str(e),
                })

    results.sort(key=lambda x: x["eval_idx"])

    ok = [r for r in results if "error" not in r]
    intercepted_list = [r["intercepted"] for r in ok]
    smokes_list      = [r["smokes_used"]  for r in ok]

    summary = {
        "pattern":           pattern_name,
        "method":            "M1_NaiveGA",
        "n_evals":           n_evals,
        "n_workers":         n_workers,
        "pop_size":          pop_size,
        "max_gen":           max_gen,
        "intercepted_mean":  float(np.mean(intercepted_list))  if ok else 0.0,
        "intercepted_std":   float(np.std(intercepted_list))   if ok else 0.0,
        "intercepted_max":   int(max(intercepted_list))        if ok else 0,
        "intercepted_min":   int(min(intercepted_list))        if ok else 0,
        "smokes_mean":       float(np.mean(smokes_list))       if ok else 15.0,
        "smokes_std":        float(np.std(smokes_list))        if ok else 0.0,
        "efficiency_mean":   (float(np.mean(intercepted_list)) /
                              float(np.mean(smokes_list)))     if ok else 0.0,
        "results":           results,
    }

    print(f"\n{'='*70}")
    print(f"  SUMMARY: M1_NaiveGA | {pattern_name}")
    print(f"  Intercepted : {summary['intercepted_mean']:.2f} ± "
          f"{summary['intercepted_std']:.2f}  "
          f"[{summary['intercepted_min']}, {summary['intercepted_max']}]")
    print(f"  Smokes Used : {summary['smokes_mean']:.2f} ± "
          f"{summary['smokes_std']:.2f}")
    print(f"  Efficiency  : {summary['efficiency_mean']:.3f}")
    print(f"{'='*70}\n")

    # 保存结果
    out_dir = os.path.join(ROOT_DIR, "M1_NaiveGA", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"M1_{pattern_name}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="M1_NaiveGA 导弹雨防御大阵")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "wave3", "multi_eval"],
                        help="运行模式")
    parser.add_argument("--pop_size",  type=int,  default=30,   help="种群大小")
    parser.add_argument("--max_gen",   type=int,  default=50,   help="最大代数")
    parser.add_argument("--n_threads", type=int,  default=None, help="并行进程数（wave3 用）")
    parser.add_argument("--pattern",   type=str,
                        default="mode_1_single_sector",
                        choices=["mode_1_single_sector", "mode_2_orthogonal_pincer",
                                 "mode_3_asymmetric_saturation", "mode_4_full_360_swarm"],
                        help="攻击模式（multi_eval 用）")
    parser.add_argument("--n_evals",   type=int,  default=10,   help="评估次数（multi_eval 用）")
    parser.add_argument("--n_workers", type=int,  default=None, help="并行 worker 数（multi_eval 用）")

    args = parser.parse_args()

    if args.mode == "test":
        run_test()
    elif args.mode == "wave3":
        run_wave3(n_threads=args.n_threads)
    elif args.mode == "multi_eval":
        run_multi_eval_m1(
            pattern_name=args.pattern,
            n_evals=args.n_evals,
            pop_size=args.pop_size,
            max_gen=args.max_gen,
            n_workers=args.n_workers,
        )


if __name__ == "__main__":
    main()
