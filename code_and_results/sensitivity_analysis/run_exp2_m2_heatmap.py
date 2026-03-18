"""
run_exp2_m2_heatmap.py  —  实验2：M2 结构性缺陷暴露 (Dc × Tc 热力图)
======================================================================
在全向蜂群 (S4) 场景下，对 M2 进行 Dc × Tc 5×5 网格搜索。
预期：无论参数如何调整，M2 在 S4 均无法突破性能瓶颈，
      从而佐证 OT 全局分配（M3）的结构性必要性。

用法:
    cd d:\\Work\\guosai\\终极实验\\sensitivity_analysis
    python run_exp2_m2_heatmap.py [--n_jobs 20]
"""

import os
import sys
import argparse

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SELF_DIR)
for _p in [_SELF_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sim_adapter import load_missiles, run_trials, get_logger, RESULTS_DIR

# ========================== 实验配置 ==========================

SCENARIO  = "mode_4_full_360_swarm"   # 最具区分度的全向蜂群场景
METHOD    = "M2"
SEEDS     = list(range(10))            # 10 seeds
N_SAMPLES = 20

# 5×5 工程绝对值网格
DC_VALUES = [500.0, 1000.0, 2000.0, 3500.0, 5000.0]   # 空间聚类阈值 (m)
TC_VALUES = [2.0,   5.0,    10.0,   20.0,   40.0]      # 时间聚类阈值 (s)

POP_SIZE  = 20
MAX_GEN   = 30

DB_PATH = os.path.join(RESULTS_DIR, "exp2_m2_heatmap.db")


def build_tasks() -> list:
    logger = get_logger("exp2")
    tasks  = []

    # 预采样导弹（所有参数组合共享同一 seed 序列 → 配对比较）
    missiles_by_seed = {}
    for seed in SEEDS:
        try:
            missiles_by_seed[seed] = load_missiles(SCENARIO, seed, N_SAMPLES)
        except FileNotFoundError as e:
            logger.error(f"导弹池文件缺失: {e}")
            return []

    for Dc in DC_VALUES:
        for Tc in TC_VALUES:
            for seed in SEEDS:
                tasks.append({
                    "method"         : METHOD,
                    "scenario"       : SCENARIO,
                    "seed"           : seed,
                    "missiles"       : missiles_by_seed[seed],
                    "pop_size"       : POP_SIZE,
                    "max_gen"        : MAX_GEN,
                    "param_overrides": {
                        "Dc": Dc,
                        "Tc": Tc,
                    },
                })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Exp2: M2 Dc×Tc 热力图")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger = get_logger("exp2")
    logger.info("=" * 60)
    logger.info("Exp2: M2 结构性缺陷 Dc×Tc 热力图 启动")
    logger.info(f"  场景: {SCENARIO}  方法: {METHOD}  seeds: {len(SEEDS)}")
    logger.info(f"  Dc 网格: {DC_VALUES}")
    logger.info(f"  Tc 网格: {TC_VALUES}")
    logger.info("=" * 60)

    tasks = build_tasks()
    total = len(tasks)
    logger.info(f"总任务数: {total}  (预估耗时: {total * 40 / max(abs(args.n_jobs), 1) / 60:.0f} min)")

    run_trials(tasks, DB_PATH, n_jobs=args.n_jobs, logger=logger)

    logger.info("Exp2 全部完成。结果保存至: " + DB_PATH)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
