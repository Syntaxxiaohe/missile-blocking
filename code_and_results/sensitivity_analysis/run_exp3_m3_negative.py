"""
run_exp3_m3_negative.py  —  实验3：M3 数学负对照实验 (w_t 列偏置不变性)
========================================================================
理论依据（论文注记）：
  w_t 是 OT 代价矩阵中的列偏置项（仅依赖导弹列 m，不依赖虚拟席位 a）。
  对任意置换 σ ∈ S_{Nv}，列偏置总和恒为常量，不影响最优指派拓扑。
  因此改变 w_t 不会改变 OT 分配方案，进而不影响拦截率。

预期结果：
  · 拦截率随 w_t 变化的曲线近似水平线
  · N_drop 恒为 0（OT 全覆盖命题 2 保证）
  · 实验结论直接支撑论文数学推导

用法:
    cd d:\\Work\\guosai\\终极实验\\sensitivity_analysis
    python run_exp3_m3_negative.py [--n_jobs 20]
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

SCENARIO  = "mode_4_full_360_swarm"
METHOD    = "M3"
SEEDS     = list(range(15))    # 15 seeds，增强统计稳定性
N_SAMPLES = 20

# w_t 扫描值（从 0 到 500，跨越 4 个数量级）
WT_VALUES = [0.0, 10.0, 50.0, 100.0, 500.0]

POP_SIZE  = 20
MAX_GEN   = 30

DB_PATH = os.path.join(RESULTS_DIR, "exp3_m3_negative.db")


def build_tasks() -> list:
    logger = get_logger("exp3")
    tasks  = []

    missiles_by_seed = {}
    for seed in SEEDS:
        try:
            missiles_by_seed[seed] = load_missiles(SCENARIO, seed, N_SAMPLES)
        except FileNotFoundError as e:
            logger.error(f"导弹池文件缺失: {e}")
            return []

    for wt in WT_VALUES:
        for seed in SEEDS:
            tasks.append({
                "method"         : METHOD,
                "scenario"       : SCENARIO,
                "seed"           : seed,
                "missiles"       : missiles_by_seed[seed],
                "pop_size"       : POP_SIZE,
                "max_gen"        : MAX_GEN,
                "param_overrides": {"w_t": wt},
            })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Exp3: M3 w_t 负对照实验")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger = get_logger("exp3")
    logger.info("=" * 60)
    logger.info("Exp3: M3 w_t 列偏置不变性负对照实验 启动")
    logger.info(f"  场景: {SCENARIO}  方法: {METHOD}  seeds: {len(SEEDS)}")
    logger.info(f"  w_t 扫描值: {WT_VALUES}")
    logger.info("=" * 60)

    tasks = build_tasks()
    total = len(tasks)
    logger.info(f"总任务数: {total}  (预估耗时: {total * 40 / max(abs(args.n_jobs), 1) / 60:.1f} min)")

    run_trials(tasks, DB_PATH, n_jobs=args.n_jobs, logger=logger)

    logger.info("Exp3 全部完成。结果保存至: " + DB_PATH)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
