"""
单组仿真模块 — 终极实验版（跨版本兼容）

核心升级：
  1. 支持动态导弹列表（从威胁生成器加载）
  2. 多目标适应度函数（权重陷阱已修复）
  3. 全局时间窗计算（废除静态绑定）
  4. 【新增】跨版本基因兼容：自动识别50维(M1/M2) vs 65维(M3)

基因维度：
  - M1/M2: 10维/UAV × 5 = 50维 [dir_x, dir_y, dir_z, speed, t1, d1, t2, d2, t3, d3]
  - M3:    13维/UAV × 5 = 65维 [dir_x, dir_y, dir_z, speed, t1, d1, s1, t2, d2, s2, t3, d3, s3]
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from shared.config.params import (
    SMOKE_PARAMS, MAX_SIM_TIME, SMOKE_PAYLOAD_PER_UAV, ACCEL_PARAMS,
    PNG_PARAMS, MISSILE_SPEED, SAFE_DISTANCE, SUCCESS_SCORE_WEIGHT
)
from shared.physics.uav import normalize_direction
from shared.physics.smoke import SmokeStateCache
from shared.physics.blocking import is_smoke_blocking_vectorized
from shared.physics.missile_png import PNGMissileFast


# ==================== 基因维度常量 ====================
GENE_DIM_M1M2 = 4 + 2 * SMOKE_PAYLOAD_PER_UAV  # 10维/UAV (无开关)
GENE_DIM_M3 = 4 + 3 * SMOKE_PAYLOAD_PER_UAV    # 13维/UAV (有开关)
SWITCH_THRESHOLD = 0.5                          # 开关阈值：> 0.5 则发射
AMMO_PENALTY_WEIGHT = 1000                      # M3弹药惩罚权重


def detect_gene_mode(params, n_uavs: int = 5) -> str:
    """
    动态检测基因模式

    Args:
        params: 基因向量
        n_uavs: UAV数量

    Returns:
        "M1M2" (50维, 无开关) 或 "M3" (65维, 有开关)
    """
    gene_len = len(params)
    expected_m1m2 = n_uavs * GENE_DIM_M1M2  # 50
    expected_m3 = n_uavs * GENE_DIM_M3      # 65

    if gene_len == expected_m1m2:
        return "M1M2"
    elif gene_len == expected_m3:
        return "M3"
    else:
        raise ValueError(f"无法识别的基因长度: {gene_len} (期望 {expected_m1m2} 或 {expected_m3})")


def parse_uav_config_m1m2(p, uav_num: int) -> dict:
    """
    解析M1/M2基因（10维，无开关）

    基因结构：[dir_x, dir_y, dir_z, speed, t1, d1, t2, d2, t3, d3]
    """
    uav_dir = normalize_direction(p[0], p[1], p[2])
    return {
        "uav_num": uav_num,
        "speed": p[3],
        "uav_dir": uav_dir,
        "t_drops": list(p[4:4+SMOKE_PAYLOAD_PER_UAV]),
        "d_dets": list(p[4+SMOKE_PAYLOAD_PER_UAV:4+2*SMOKE_PAYLOAD_PER_UAV]),
        "switches": [1.0] * SMOKE_PAYLOAD_PER_UAV,  # 全部发射
        "n_smokes_fired": SMOKE_PAYLOAD_PER_UAV     # M1/M2: 无条件发射3颗
    }


def parse_uav_config_m3(p, uav_num: int) -> dict:
    """
    解析M3基因（13维，有开关）

    基因结构：[dir_x, dir_y, dir_z, speed, t1, d1, s1, t2, d2, s2, t3, d3, s3]
    """
    uav_dir = normalize_direction(p[0], p[1], p[2])

    t_drops = []
    d_dets = []
    switches = []
    n_smokes_fired = 0

    for k in range(SMOKE_PAYLOAD_PER_UAV):
        base_idx = 4 + k * 3
        t_drops.append(p[base_idx])
        d_dets.append(p[base_idx + 1])
        switch = p[base_idx + 2]
        switches.append(switch)

        # 开关判断：> 0.5 则发射
        if switch > SWITCH_THRESHOLD:
            n_smokes_fired += 1

    return {
        "uav_num": uav_num,
        "speed": p[3],
        "uav_dir": uav_dir,
        "t_drops": t_drops,
        "d_dets": d_dets,
        "switches": switches,
        "n_smokes_fired": n_smokes_fired  # M3: 根据开关决定
    }


def simulate_missile_rain(params, uav_group, missile_configs, dt=None, n_workers=None, record_history=False):
    """
    导弹雨仿真（跨版本兼容）

    自动检测基因模式：
    - 50维 → M1/M2模式（10维/UAV，无开关，全部发射）
    - 65维 → M3模式（13维/UAV，有开关，按需发射）

    Args:
        params: 优化参数（50维或65维）
        uav_group: UAV编号列表
        missile_configs: 导弹配置列表
        dt: 仿真步长
        n_workers: 并行线程数
        record_history: 是否记录轨迹

    Returns:
        (min_miss_distance, results_list, uav_configs)
    """
    if n_workers is None:
        n_workers = min(len(missile_configs), ACCEL_PARAMS["n_workers"])

    if dt is None:
        dt = 0.02

    n_uavs = len(uav_group)

    # ===== 关键：动态检测基因模式 =====
    gene_mode = detect_gene_mode(params, n_uavs)
    gene_len = GENE_DIM_M1M2 if gene_mode == "M1M2" else GENE_DIM_M3

    # 解析UAV配置
    uav_configs = []
    for uav_idx, uav_num in enumerate(uav_group):
        p = params[uav_idx * gene_len:(uav_idx + 1) * gene_len]

        if gene_mode == "M1M2":
            uav_config = parse_uav_config_m1m2(p, uav_num)
        else:  # M3
            uav_config = parse_uav_config_m3(p, uav_num)

        uav_configs.append(uav_config)

    # 创建烟幕状态缓存
    smoke_cache = SmokeStateCache(
        uav_configs,
        t_max=MAX_SIM_TIME + 30,
        dt=ACCEL_PARAMS["smoke_cache_dt"]
    )

    # 导弹仿真函数
    def simulate_single_missile(m_config):
        def smoke_checker(missile_num, t, pos):
            centers, sigmas, is_eff, is_det, Q_vals = smoke_cache.get_states_at_t(t)
            valid_mask = is_eff & is_det & (sigmas <= 100.0) & (sigmas > 1e-6)
            if not np.any(valid_mask):
                return False
            return is_smoke_blocking_vectorized(
                pos, centers[valid_mask], sigmas[valid_mask],
                is_eff[valid_mask], is_det[valid_mask], Q_vals[valid_mask]
            )

        # 支持动态初始位置和速度
        init_pos = np.array(m_config["position"])
        init_vel = np.array(m_config["velocity"]) if "velocity" in m_config else None

        missile = PNGMissileFast(
            missile_num=m_config["id"],
            init_pos=init_pos,
            init_vel=init_vel,
            record_history=record_history
        )
        return missile.simulate(smoke_checker=smoke_checker, dt=dt, max_time=MAX_SIM_TIME + 30)

    # 并行仿真
    if len(missile_configs) > 1 and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(simulate_single_missile, missile_configs))
    else:
        results = [simulate_single_missile(m) for m in missile_configs]

    min_miss = min(r["miss_distance"] for r in results)
    return min_miss, results, uav_configs


# ==================== 适应度函数 ====================

def single_missile_objective(params, uav_group, missile_configs):
    """
    单目标适应度函数（最小脱靶量）

    Returns:
        (min_miss_distance, None)
    """
    min_miss, _, _ = simulate_missile_rain(params, uav_group, missile_configs)
    return (min_miss, None)


def multi_missile_objective(params, uav_group, missile_configs):
    """
    多目标适应度函数（跨版本兼容）

    M1/M2模式（50维）：
    - 适应度 = N_success * 10000 + 失败导弹限幅脱靶量
    - 无弹药惩罚（全部发射15颗）

    M3模式（65维）：
    - 适应度 = N_success * 10000 - N_ammo * 1000 + 失败导弹限幅脱靶量
    - 有弹药惩罚（根据开关计算实际发射数）

    核心逻辑：
    - 第一优先级：最大化"成功防御的导弹数量"
    - 第二优先级：在拦截数量相同的情况下，优化失败导弹的脱靶量
    - M3额外：在相同拦截效果下，最小化弹药消耗

    Returns:
        (fitness, details_dict)
    """
    # 检测基因模式
    gene_mode = detect_gene_mode(params, len(uav_group))

    # 运行仿真
    _, results, uav_configs = simulate_missile_rain(params, uav_group, missile_configs)

    # 统计成功防御的导弹
    success_count = 0
    failed_miss_distances = []

    for r in results:
        miss_dist = r["miss_distance"]
        if miss_dist >= SAFE_DISTANCE:
            success_count += 1
        else:
            failed_miss_distances.append(miss_dist)

    # 计算失败导弹得分（硬限幅防止极端值倒挂）
    if failed_miss_distances:
        clipped_failed = [min(d, SAFE_DISTANCE) for d in failed_miss_distances]
        failed_score = sum(clipped_failed)
    else:
        failed_score = 0.0

    # 基础适应度 = 成功数 * 10000 + 失败导弹的限幅后总分
    base_fitness = success_count * SUCCESS_SCORE_WEIGHT + failed_score

    # ===== M3特殊处理：弹药惩罚 =====
    ammo_penalty = 0.0
    total_smokes_fired = 0

    if gene_mode == "M3":
        # 计算实际发射的烟雾弹数量
        for uav_config in uav_configs:
            total_smokes_fired += uav_config["n_smokes_fired"]

        # 弹药惩罚 = 发射数量 × 1000
        ammo_penalty = total_smokes_fired * AMMO_PENALTY_WEIGHT

    fitness = base_fitness - ammo_penalty

    details = {
        "gene_mode": gene_mode,
        "total_missiles": len(missile_configs),
        "success_count": success_count,
        "failed_count": len(failed_miss_distances),
        "failed_score": failed_score,
        "raw_failed_distances": failed_miss_distances,
        "all_miss_distances": [r["miss_distance"] for r in results],
        "ammo_penalty": ammo_penalty,
        "total_smokes_fired": total_smokes_fired if gene_mode == "M3" else SMOKE_PAYLOAD_PER_UAV * len(uav_group)
    }

    return (fitness, details)


def compute_perfect_score(n_missiles: int, gene_mode: str = "M1M2") -> float:
    """
    计算完美防御得分（用于熔断阈值）

    完美防御 = 所有导弹都被成功拦截

    M1/M2: Perfect Score = N_missiles * 10000
    M3:    Perfect Score = N_missiles * 10000 - N_ammo * 1000 (需要考虑弹药惩罚)
    """
    base_score = n_missiles * SUCCESS_SCORE_WEIGHT

    if gene_mode == "M3":
        # M3模式下，完美防御还需要扣除弹药惩罚
        # 假设最少需要 n_missiles 颗烟雾弹（理想情况）
        min_ammo_penalty = n_missiles * AMMO_PENALTY_WEIGHT
        return base_score - min_ammo_penalty
    else:
        return base_score
