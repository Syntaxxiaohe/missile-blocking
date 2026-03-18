"""
run_exp1_physical.py  —  实验1：物理参数稳健性分析
=====================================================
OFAT 单因素扫描：
  · k_opt        (光学强度因子，同步缩放 alpha 和 Q)
  · diffusion_rate (烟团扩散速度)
目标场景: S2 (正交钳形), S4 (全向蜂群)
对比方法: M1, M2, M3（相同 seed 配对比较）
重复次数: 10 seeds / 参数点

用法:
    cd d:\\Work\\guosai\\终极实验\\sensitivity_analysis
    python run_exp1_physical.py [--n_jobs 20] [--pop_size 20] [--max_gen 30]
"""

import os
import sys
import argparse
import logging

# ===== 路径注册 =====
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SELF_DIR)
for _p in [_SELF_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sim_adapter import (
    load_missiles, run_trials, get_logger,
    RESULTS_DIR, NOMINAL_DIFFUSION_RATE,
)

# ========================== 实验配置 ==========================

SCENARIOS = [
    "mode_2_orthogonal_pincer",   # S2: 正交钳形
    "mode_4_full_360_swarm",      # S4: 全向蜂群（最具区分度）
]
METHODS   = ["M1", "M2", "M3"]
SEEDS     = list(range(10))       # seed 0-9，共 10 次重复
N_SAMPLES = 20                    # 每轮导弹数

# OFAT 参数网格：乘法扰动倍率
MULTIPLIERS = [0.70, 0.85, 1.00, 1.15, 1.30]

DB_PATH = os.path.join(RESULTS_DIR, "exp1_physical.db")


def build_tasks(pop_size: int, max_gen: int) -> list:
    """构建所有实验任务列表（主进程预采样导弹，保证配对比较）"""
    logger = get_logger("exp1")
    tasks  = []

    for scenario in SCENARIOS:
        # 预采样所有 seed 的导弹（配对比较：同 seed 用相同导弹）
        missiles_by_seed = {}
        for seed in SEEDS:
            try:
                missiles_by_seed[seed] = load_missiles(scenario, seed, N_SAMPLES)
            except FileNotFoundError as e:
                logger.error(f"导弹池文件缺失: {e}，跳过场景 {scenario}")
                missiles_by_seed = {}
                break

        if not missiles_by_seed:
            continue

        # --- 参数扫描 1: k_opt ---
        for mult in MULTIPLIERS:
            for method in METHODS:
                for seed in SEEDS:
                    tasks.append({
                        "method"         : method,
                        "scenario"       : scenario,
                        "seed"           : seed,
                        "missiles"       : missiles_by_seed[seed],
                        "pop_size"       : pop_size,
                        "max_gen"        : max_gen,
                        "param_overrides": {
                            "k_opt"          : mult,
                            "param_type"     : "k_opt",
                            "multiplier"     : mult,
                        },
                    })

        # --- 参数扫描 2: diffusion_rate ---
        for mult in MULTIPLIERS:
            dr = NOMINAL_DIFFUSION_RATE * mult
            for method in METHODS:
                for seed in SEEDS:
                    tasks.append({
                        "method"         : method,
                        "scenario"       : scenario,
                        "seed"           : seed,
                        "missiles"       : missiles_by_seed[seed],
                        "pop_size"       : pop_size,
                        "max_gen"        : max_gen,
                        "param_overrides": {
                            "diffusion_rate" : dr,
                            "param_type"     : "diffusion_rate",
                            "multiplier"     : mult,
                        },
                    })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Exp1: 物理参数稳健性分析")
    parser.add_argument("--n_jobs",   type=int, default=-1,
                        help="joblib 并行数（-1=全核，建议云端设为 20）")
    parser.add_argument("--pop_size", type=int, default=20,
                        help="GA 种群大小（灵敏度分析轻量设置，默认 20）")
    parser.add_argument("--max_gen",  type=int, default=30,
                        help="GA 最大代数（默认 30）")
    args = parser.parse_args()

    logger = get_logger("exp1")
    logger.info("=" * 60)
    logger.info("Exp1: 物理参数稳健性 OFAT 扫描 启动")
    logger.info(f"  n_jobs={args.n_jobs}  pop={args.pop_size}  gen={args.max_gen}")
    logger.info(f"  场景: {SCENARIOS}")
    logger.info(f"  方法: {METHODS}  seeds: {SEEDS}  倍率: {MULTIPLIERS}")
    logger.info("=" * 60)

    tasks = build_tasks(args.pop_size, args.max_gen)
    total = len(tasks)
    logger.info(f"总任务数: {total}  (预估耗时: {total * 40 / max(abs(args.n_jobs), 1) / 60:.0f} min)")

    run_trials(tasks, DB_PATH, n_jobs=args.n_jobs, logger=logger)

    logger.info("Exp1 全部完成。结果保存至: " + DB_PATH)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
