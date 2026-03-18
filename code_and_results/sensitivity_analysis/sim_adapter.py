"""
sim_adapter.py  —  灵敏度分析核心适配层
=========================================
将真实仿真优化器封装为统一接口，支持：
  - 物理参数注入   (k_opt, diffusion_rate)
  - 聚类参数注入   (Dc, Tc)       — M2 专用
  - OT权重注入     (w_t)          — M3 专用

设计原则：
  1. 每个 trial 运行在独立子进程（joblib loky/spawn），参数修改完全隔离。
  2. 导弹列表在主进程预采样，以保证 Paired Comparison（同 seed 同场景）。
  3. 返回统一字典，便于直接写入 SQLite。
"""

import os
import sys
import sqlite3
import hashlib
import json
import logging

# ========================== 路径常量 ==========================
ADAPTER_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(ADAPTER_DIR)
M1_DIR       = os.path.join(ROOT_DIR, "M1_NaiveGA")
M2_DIR       = os.path.join(ROOT_DIR, "M2_Clustering")
M3_DIR       = os.path.join(ROOT_DIR, "M3_AmmoPenalty")
RESULTS_DIR  = os.path.join(ADAPTER_DIR, "results")
FIGURES_DIR  = os.path.join(ADAPTER_DIR, "figures")

# 标称物理参数（用于乘法扰动的基准值）
NOMINAL_ALPHA          = 0.3
NOMINAL_Q              = 300000.0
NOMINAL_DIFFUSION_RATE = 8.0

# ========================== 日志设置 ==========================
def get_logger(name: str = "sensitivity") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(
            os.path.join(ADAPTER_DIR, "experiment_run.log"),
            encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger

# ========================== SQLite 工具 ==========================

def task_key(method: str, scenario: str, seed: int,
             param_overrides: dict) -> str:
    """生成任务的确定性唯一哈希键（用于断点续传去重）"""
    payload = json.dumps(
        {"method": method, "scenario": scenario,
         "seed": seed, "params": param_overrides},
        sort_keys=True
    )
    return hashlib.md5(payload.encode()).hexdigest()


def init_db(db_path: str) -> sqlite3.Connection:
    """初始化 SQLite 数据库，建表（幂等）"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            task_key        TEXT PRIMARY KEY,
            method          TEXT,
            scenario        TEXT,
            seed            INTEGER,
            param_json      TEXT,
            interception_rate REAL,
            intercepted     INTEGER,
            n_missiles      INTEGER,
            n_drop          INTEGER,
            best_fitness    REAL,
            ts              DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def is_done(conn: sqlite3.Connection, key: str) -> bool:
    """检查该任务是否已完成（断点续传核心）"""
    cur = conn.execute(
        "SELECT 1 FROM results WHERE task_key=?", (key,)
    )
    return cur.fetchone() is not None


def save_result(conn: sqlite3.Connection, key: str, row: dict):
    """将单次结果追加写入 SQLite"""
    conn.execute("""
        INSERT OR REPLACE INTO results
            (task_key, method, scenario, seed, param_json,
             interception_rate, intercepted, n_missiles, n_drop, best_fitness)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        key,
        row["method"],
        row["scenario"],
        row["seed"],
        json.dumps(row.get("param_overrides", {}), sort_keys=True),
        row["interception_rate"],
        row["intercepted"],
        row["n_missiles"],
        row.get("n_drop", 0),
        row.get("best_fitness", 0.0),
    ))
    conn.commit()

# ========================== 导弹池工具 ==========================

def load_missiles(scenario: str, seed: int, n_samples: int = 20) -> list:
    """
    从预生成导弹池中以固定 seed 采样导弹列表。
    在主进程完成后序列化传递给子进程，保证配对比较。
    """
    # 加入 ROOT_DIR 路径
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    from shared.data.missile_pool_generator import load_missile_pool, sample_from_pool

    pool_path = os.path.join(ROOT_DIR, "data", "missile_pools",
                             f"{scenario}_pool.json")
    pool = load_missile_pool(pool_path)
    return sample_from_pool(pool, n_samples=n_samples, seed=seed)

# ========================== 子进程 Worker 函数 ==========================
# 注意：这些函数必须在模块顶层定义，以支持 spawn/loky 序列化。

def _worker(task: dict) -> dict:
    """
    通用子进程 Worker。
    每次调用对应一个 (method, scenario, seed, param_overrides) 四元组。
    在子进程内完成参数注入 + 优化 + 返回结果。
    """
    import os as _os, sys as _sys

    # 锁定 BLAS/MKL 线程（防止嵌套并行占满 CPU）
    for k in ["MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        _os.environ[k] = "1"

    method          = task["method"]
    scenario        = task["scenario"]
    seed            = task["seed"]
    missiles        = task["missiles"]   # 已在主进程预采样
    pop_size        = task.get("pop_size", 20)
    max_gen         = task.get("max_gen", 30)
    param_overrides = task.get("param_overrides", {})

    # --- 路径确保 ---
    for p in [ROOT_DIR]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    # ---- 物理参数注入（k_opt / diffusion_rate） ----
    if "k_opt" in param_overrides or "diffusion_rate" in param_overrides:
        import shared.config.params as _params
        p = _params.GAUSSIAN_SMOKE_PARAMS
        if "k_opt" in param_overrides:
            kopt = float(param_overrides["k_opt"])
            p["alpha"]  = NOMINAL_ALPHA          * kopt
            p["Q"]      = NOMINAL_Q              * kopt
        if "diffusion_rate" in param_overrides:
            p["diffusion_rate"] = float(param_overrides["diffusion_rate"])

        # ★ 关键：blocking.py 和 smoke.py 在模块导入时会把 GAUSSIAN_SMOKE_PARAMS
        #   中的值缓存为模块级常量（_ALPHA、_Q、_DIFFUSION_RATE 等）。
        #   只修改 dict 不够——必须清除这些模块缓存，迫使下次 import 时重新读取新值。
        #   注意：保留 shared.config.params 缓存（含修改后的 dict），仅清除物理模块。
        for _mk in list(_sys.modules.keys()):
            if (_mk.startswith("shared.physics.")
                    or _mk.startswith("shared.simulation.")
                    or _mk == "shared.physics"
                    or _mk == "shared.simulation"):
                del _sys.modules[_mk]

    # ---- 清除 optimizer 包缓存 + 强制 sys.path 切换 ----
    # Bug 防护说明（两层防线）：
    #   防线 1: 清除 sys.modules 中 optimizer.* 缓存
    #     → 防止 loky worker 复用时上一个 method 的模块对象被下一个 method 命中。
    #   防线 2: 移除全部三个方法目录，再将当前方法目录 insert(0)
    #     → 防止 sys.path 顺序错位：例如先跑 M1 插入 M1_DIR，再跑 M2 插入 M2_DIR，
    #       第三次跑 M1 时 M1_DIR 已在 path 中但位置靠后（不会重新 insert），
    #       清了 modules 后 import 仍会优先从 M2_DIR 加载 → ImportError。
    for _mk in list(_sys.modules.keys()):
        if _mk == "optimizer" or _mk.startswith("optimizer."):
            del _sys.modules[_mk]

    for _md in [M1_DIR, M2_DIR, M3_DIR]:
        while _md in _sys.path:
            _sys.path.remove(_md)

    # ---- 根据方法调用对应优化器 ----
    n_drop = 0

    if method == "M1":
        _sys.path.insert(0, M1_DIR)
        from optimizer.distributed_optimizer import TwoStageOptimizer
        opt = TwoStageOptimizer(
            uav_group=[1, 2, 3, 4, 5],
            missile_configs=missiles,
            pop_size=pop_size,
            max_generations=max_gen,
            n_threads=1,
        )
        _, best_fitness, info = opt.optimize()
        intercepted = int(info.get("success_count", 0))
        n_missiles  = int(info.get("n_missiles", len(missiles)))

    elif method == "M2":
        _sys.path.insert(0, M2_DIR)
        from optimizer.distributed_optimizer import run_distributed_optimization
        Dc = float(param_overrides["Dc"]) if "Dc" in param_overrides else None
        Tc = float(param_overrides["Tc"]) if "Tc" in param_overrides else None
        _, best_fitness, info = run_distributed_optimization(
            uav_group=[1, 2, 3, 4, 5],
            missile_configs=missiles,
            pop_size=pop_size,
            max_generations=max_gen,
            n_threads=1,
            spatial_threshold=Dc,
            temporal_threshold=Tc,
        )
        intercepted = int(info.get("success_count", 0))
        n_missiles  = int(info.get("n_missiles", len(missiles)))
        n_drop = max(0, len(missiles) - int(info.get("n_assigned", len(missiles))))

    elif method == "M3":
        _sys.path.insert(0, M3_DIR)
        from optimizer.distributed_optimizer import run_distributed_optimization_M2
        w_t = float(param_overrides.get("w_t", 10.0))
        _, best_fitness, info = run_distributed_optimization_M2(
            uav_group=[1, 2, 3, 4, 5],
            missile_configs=missiles,
            pop_size=pop_size,
            max_generations=max_gen,
            n_threads=1,
            w_t=w_t,
        )
        intercepted = int(info.get("success_count", 0))
        n_missiles  = int(info.get("n_missiles", len(missiles)))
        n_drop      = 0

    else:
        raise ValueError(f"未知 method: {method}")

    return {
        "method"           : method,
        "scenario"         : scenario,
        "seed"             : seed,
        "param_overrides"  : param_overrides,
        "intercepted"      : intercepted,
        "n_missiles"       : n_missiles,
        "interception_rate": intercepted / max(n_missiles, 1),
        "n_drop"           : n_drop,
        "best_fitness"     : float(best_fitness),
    }


# ========================== 高层调度接口 ==========================

def run_trials(tasks: list, db_path: str, n_jobs: int = -1,
               logger: logging.Logger = None) -> None:
    """
    并行运行一批 trials，自动跳过已完成任务（断点续传）。

    Args:
        tasks:    任务字典列表，每项需含 method/scenario/seed/missiles/param_overrides
        db_path:  SQLite 文件路径
        n_jobs:   joblib 并行数，-1 = 使用全部核心
        logger:   logging.Logger 实例
    """
    if logger is None:
        logger = get_logger()

    conn = init_db(db_path)

    # 过滤已完成任务（断点续传）
    pending = []
    for t in tasks:
        key = task_key(t["method"], t["scenario"], t["seed"],
                       t.get("param_overrides", {}))
        if is_done(conn, key):
            logger.info(f"[SKIP] {t['method']} {t['scenario']} seed={t['seed']} "
                        f"params={t.get('param_overrides',{})}")
        else:
            t["_key"] = key
            pending.append(t)

    logger.info(f"共 {len(tasks)} 个任务，跳过 {len(tasks)-len(pending)} 个，"
                f"待运行 {len(pending)} 个")

    if not pending:
        conn.close()
        return

    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError("请安装 joblib: pip install joblib")

    # 顺序回调写库（joblib 返回结果列表后逐一写入，保证即时落盘）
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=5)(
        delayed(_worker)(t) for t in pending
    )

    for t, res in zip(pending, results):
        if res is not None:
            try:
                save_result(conn, t["_key"], res)
                logger.info(
                    f"[DONE] {res['method']} {res['scenario']} "
                    f"seed={res['seed']} "
                    f"rate={res['interception_rate']:.2%} "
                    f"params={res.get('param_overrides',{})}"
                )
            except Exception as e:
                logger.error(f"[SAVE ERROR] {e} | task={t}")
        else:
            logger.error(f"[WORKER ERROR] task={t}")

    conn.close()
    logger.info("本批次全部完成，结果已写入: " + db_path)
