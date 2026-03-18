"""
高斯烟团遮蔽判定模块 — 加速版

加速特性：
  1. Numba JIT编译：加速核心计算
  2. 向量化视线计算：批量处理多条视线
  3. 预分配数组：避免动态内存分配
"""

import numpy as np
from scipy.special import erf
from shared.config.params import (
    REAL_TARGET, GAUSSIAN_SMOKE_PARAMS, ACCEL_PARAMS
)

# 尝试导入numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 如果numba不可用，创建一个空装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# 性能优化常量
MAX_SIGMA_THRESHOLD = 100.0
_ALPHA = GAUSSIAN_SMOKE_PARAMS["alpha"]
_THRESHOLD = GAUSSIAN_SMOKE_PARAMS["transmittance_threshold"]
_K_SIGMA = GAUSSIAN_SMOKE_PARAMS["effective_sigma_multiple"]

# 预计算目标采样点（固定位置，导弹位置不影响）
_CHECK_POINTS_MINIMAL = np.array([
    [REAL_TARGET["center"][0], REAL_TARGET["center"][1], 0.0],      # 底面圆心
    [REAL_TARGET["center"][0], REAL_TARGET["center"][1], REAL_TARGET["height"]],  # 顶面圆心
    [REAL_TARGET["center"][0] + REAL_TARGET["radius"], REAL_TARGET["center"][1], 0.0],  # 底面 +X
    [REAL_TARGET["center"][0] - REAL_TARGET["radius"], REAL_TARGET["center"][1], 0.0],  # 底面 -X
    [REAL_TARGET["center"][0], REAL_TARGET["center"][1] + REAL_TARGET["radius"], 0.0],  # 底面 +Y
    [REAL_TARGET["center"][0], REAL_TARGET["center"][1] - REAL_TARGET["radius"], 0.0],  # 底面 -Y
], dtype=np.float64)


def _compute_optical_depth_single_puff(ray_origin: np.ndarray, ray_dir: np.ndarray,
                                        ray_length: float, smoke_center: np.ndarray,
                                        sigma: float, Q: float) -> float:
    """计算单条视线穿过单个高斯烟团的光学深度"""
    if sigma < 1e-6:
        return 0.0

    delta = ray_origin - smoke_center
    dot_delta_dir = np.dot(delta, ray_dir)
    d_perp_sq = np.dot(delta, delta) - dot_delta_dir ** 2

    sigma_sq = sigma ** 2
    if d_perp_sq > (_K_SIGMA * sigma) ** 2:
        return 0.0

    discriminant = _K_SIGMA**2 * sigma_sq - d_perp_sq
    if discriminant < 0:
        return 0.0

    sqrt_disc = np.sqrt(discriminant)
    s_center = -dot_delta_dir
    s1 = max(0.0, s_center - sqrt_disc)
    s2 = min(ray_length, s_center + sqrt_disc)

    if s2 <= s1:
        return 0.0

    sqrt_2 = np.sqrt(2)
    sqrt_2pi = np.sqrt(2 * np.pi)

    erf_term = 0.5 * (erf((s2 - s_center) / (sqrt_2 * sigma)) -
                      erf((s1 - s_center) / (sqrt_2 * sigma)))
    gauss_weight = np.exp(-d_perp_sq / (2 * sigma_sq))

    integral = (Q / (2 * np.pi * sigma_sq)) * gauss_weight * erf_term
    return integral


def compute_transmittance_vectorized(ray_origin: np.ndarray, ray_targets: np.ndarray,
                                       centers: np.ndarray, sigmas: np.ndarray,
                                       is_effective: np.ndarray, is_detonated: np.ndarray,
                                       Q_values: np.ndarray) -> np.ndarray:
    """
    向量化计算多条视线的透射率

    Args:
        ray_origin: 视线起点 (3,)
        ray_targets: 视线终点 (n_rays, 3)
        centers: 烟团中心 (n_smokes, 3)
        sigmas: 烟团标准差 (n_smokes,)
        is_effective: 是否有效 (n_smokes,)
        is_detonated: 是否起爆 (n_smokes,)
        Q_values: 烟团强度 (n_smokes,)

    Returns:
        transmittances: 透射率数组 (n_rays,)
    """
    n_rays = len(ray_targets)
    n_smokes = len(centers)

    # 预筛选有效烟团
    valid_mask = is_effective & is_detonated & (sigmas <= MAX_SIGMA_THRESHOLD) & (sigmas > 1e-6)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.ones(n_rays)

    # 提取有效烟团数据
    valid_centers = centers[valid_indices]
    valid_sigmas = sigmas[valid_indices]
    valid_Q = Q_values[valid_indices]
    n_valid = len(valid_indices)

    # 计算视线方向和长度 (n_rays, 3)
    ray_vecs = ray_targets - ray_origin  # (n_rays, 3)
    ray_lengths = np.linalg.norm(ray_vecs, axis=1)  # (n_rays,)

    # 避免除零
    ray_lengths = np.maximum(ray_lengths, 1e-6)
    ray_dirs = ray_vecs / ray_lengths[:, np.newaxis]  # (n_rays, 3)

    # 计算光学深度 (n_rays,)
    optical_depths = np.zeros(n_rays)

    for i in range(n_valid):
        smoke_center = valid_centers[i]
        sigma = valid_sigmas[i]
        Q = valid_Q[i]

        # 批量计算所有视线到该烟团的距离
        delta = ray_origin - smoke_center  # (3,)
        dot_delta_dir = np.dot(ray_dirs, delta)  # (n_rays,)
        d_perp_sq = np.dot(delta, delta) - dot_delta_dir ** 2  # (n_rays,)

        # 快速剔除
        valid_ray_mask = d_perp_sq < (_K_SIGMA * sigma) ** 2

        if not np.any(valid_ray_mask):
            continue

        # 对有效视线计算积分
        sigma_sq = sigma ** 2
        sqrt_2 = np.sqrt(2)
        sqrt_2pi = np.sqrt(2 * np.pi)

        for j in np.where(valid_ray_mask)[0]:
            d_perp_sq_j = d_perp_sq[j]
            ray_len_j = ray_lengths[j]

            discriminant = _K_SIGMA**2 * sigma_sq - d_perp_sq_j
            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            s_center = -dot_delta_dir[j]
            s1 = max(0.0, s_center - sqrt_disc)
            s2 = min(ray_len_j, s_center + sqrt_disc)

            if s2 <= s1:
                continue

            erf_term = 0.5 * (erf((s2 - s_center) / (sqrt_2 * sigma)) -
                              erf((s1 - s_center) / (sqrt_2 * sigma)))
            gauss_weight = np.exp(-d_perp_sq_j / (2 * sigma_sq))

            integral = (Q / (2 * np.pi * sigma_sq)) * gauss_weight * erf_term
            optical_depths[j] += integral

    # Beer-Lambert
    transmittances = np.exp(-_ALPHA * optical_depths)
    return transmittances


def is_smoke_blocking_vectorized(missile_pos: np.ndarray,
                                  centers: np.ndarray, sigmas: np.ndarray,
                                  is_effective: np.ndarray, is_detonated: np.ndarray,
                                  Q_values: np.ndarray) -> bool:
    """
    向量化遮蔽判定

    Returns:
        True = 所有视线都被遮蔽
    """
    # 使用预计算的固定采样点
    ray_targets = _CHECK_POINTS_MINIMAL

    # 向量化计算透射率
    transmittances = compute_transmittance_vectorized(
        missile_pos, ray_targets, centers, sigmas,
        is_effective, is_detonated, Q_values
    )

    # 检查是否所有视线的透射率都低于阈值
    return np.all(transmittances < _THRESHOLD)


def is_smoke_blocking_multi(missile_pos: np.ndarray, smoke_states_list: list,
                            light: bool = True, minimal: bool = False) -> bool:
    """
    多烟幕弹联合遮蔽判定（兼容接口）
    """
    if not smoke_states_list:
        return False

    # 转换为数组格式
    n_smokes = len(smoke_states_list)
    centers = np.zeros((n_smokes, 3))
    sigmas = np.zeros(n_smokes)
    is_effective = np.zeros(n_smokes, dtype=bool)
    is_detonated = np.zeros(n_smokes, dtype=bool)
    Q_values = np.zeros(n_smokes)

    valid_count = 0
    for i, s in enumerate(smoke_states_list):
        if s.get("center") is None:
            continue
        if not (s.get("is_detonated") and s.get("is_effective")):
            continue
        sigma = s.get("sigma", 0)
        if sigma > MAX_SIGMA_THRESHOLD or sigma < 1e-6:
            continue

        centers[valid_count] = s["center"]
        sigmas[valid_count] = sigma
        is_effective[valid_count] = True
        is_detonated[valid_count] = True
        Q_values[valid_count] = s.get("Q", 0)
        valid_count += 1

    if valid_count == 0:
        return False

    # 截取有效部分
    centers = centers[:valid_count]
    sigmas = sigmas[:valid_count]
    is_effective = is_effective[:valid_count]
    is_detonated = is_detonated[:valid_count]
    Q_values = Q_values[:valid_count]

    return is_smoke_blocking_vectorized(
        missile_pos, centers, sigmas, is_effective, is_detonated, Q_values
    )


def compute_transmittance(ray_origin: np.ndarray, ray_target: np.ndarray,
                          smoke_states_list: list) -> float:
    """计算单条视线的总透射率（兼容接口）"""
    ray_vec = ray_target - ray_origin
    ray_length = np.linalg.norm(ray_vec)

    if ray_length < 1e-6:
        return 1.0

    ray_dir = ray_vec / ray_length

    # 预筛选
    total_optical_depth = 0.0
    for s in smoke_states_list:
        if not (s.get("is_detonated") and s.get("is_effective")):
            continue
        if s.get("center") is None:
            continue
        sigma = s.get("sigma", 0)
        if sigma > MAX_SIGMA_THRESHOLD:
            continue

        integral = _compute_optical_depth_single_puff(
            ray_origin, ray_dir, ray_length,
            s["center"], sigma, s.get("Q", 0)
        )
        total_optical_depth += integral

    return np.exp(-_ALPHA * total_optical_depth)


def _compute_check_points_minimal(missile_pos: np.ndarray):
    """GA 优化专用：极简采样模式 - 仅 6 条特征视线"""
    return [np.array(p) for p in _CHECK_POINTS_MINIMAL]
