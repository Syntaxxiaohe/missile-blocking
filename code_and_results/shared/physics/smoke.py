"""
高斯烟团平流扩散模块 — 加速版

加速特性：
  1. 预计算查表：提前计算所有时刻的烟团状态
  2. 向量化计算：批量计算多个烟团状态
  3. 结构化数组：使用numpy结构化数组代替字典
"""

import numpy as np
from shared.config.params import (
    GAUSSIAN_SMOKE_PARAMS, WIND_PARAMS, get_wind_vector, compute_sigma,
    MAX_SIM_TIME, ACCEL_PARAMS
)
from shared.physics.uav import get_uav_drop_position, normalize_direction


# 预计算常量
_WIND_VECTOR = get_wind_vector()
_SINK_VEL = np.array([0.0, 0.0, -GAUSSIAN_SMOKE_PARAMS["sink_speed"]])
_DRIFT_VEL = _WIND_VECTOR + _SINK_VEL
_FALL_ACCEL = GAUSSIAN_SMOKE_PARAMS["fall_accel"]
_SIGMA_0 = GAUSSIAN_SMOKE_PARAMS["sigma_0"]
_DIFFUSION_RATE = GAUSSIAN_SMOKE_PARAMS["diffusion_rate"]
_MAX_EFFECTIVE_TIME = GAUSSIAN_SMOKE_PARAMS["max_effective_time"]
_Q = GAUSSIAN_SMOKE_PARAMS["Q"]


class SmokeStateCache:
    """
    烟幕状态预计算缓存类

    在仿真开始前，预计算所有可能的烟幕状态，存储为结构化数组
    """

    def __init__(self, uav_configs: list, t_max: float = 100.0, dt: float = 0.1):
        """
        Args:
            uav_configs: UAV配置列表，每个元素包含:
                - uav_num: UAV编号
                - speed: 飞行速度
                - uav_dir: 飞行方向
                - t_drops: 投放时刻列表
                - d_dets: 起爆延迟列表
            t_max: 最大预计算时间
            dt: 预计算时间步长
        """
        self.uav_configs = uav_configs
        self.t_max = t_max
        self.dt = dt

        # 时间点数组
        self.t_array = np.arange(0, t_max + dt, dt)
        self.n_times = len(self.t_array)

        # 预计算所有烟幕状态
        self._precompute()

    def _precompute(self):
        """预计算所有烟幕状态"""
        # 统计烟幕弹总数
        self.n_smokes = sum(len(cfg["t_drops"]) for cfg in self.uav_configs)

        # 创建状态数组 (n_smokes, n_times)
        # 每个状态: center(3), sigma(1), is_effective(bool), is_detonated(bool)
        self.centers = np.zeros((self.n_smokes, self.n_times, 3), dtype=np.float32)
        self.sigmas = np.zeros((self.n_smokes, self.n_times), dtype=np.float32)
        self.is_effective = np.zeros((self.n_smokes, self.n_times), dtype=bool)
        self.is_detonated = np.zeros((self.n_smokes, self.n_times), dtype=bool)
        self.Q_values = np.full(self.n_smokes, _Q, dtype=np.float32)

        # 烟幕弹索引
        smoke_idx = 0

        for cfg in self.uav_configs:
            uav_num = cfg["uav_num"]
            speed = cfg["speed"]
            uav_dir = cfg["uav_dir"]

            for smoke_i, (t_drop, d_det) in enumerate(zip(cfg["t_drops"], cfg["d_dets"])):
                detonate_time = t_drop + d_det

                # 预计算投放点位置
                drop_pos = get_uav_drop_position(uav_num, t_drop, speed, uav_dir)

                # 预计算起爆点位置
                fall_time = d_det
                horizontal_disp = speed * fall_time * uav_dir
                det_x = drop_pos[0] + horizontal_disp[0]
                det_y = drop_pos[1] + horizontal_disp[1]
                z_fall = 0.5 * _FALL_ACCEL * (fall_time ** 2)
                det_z = max(drop_pos[2] - z_fall, 0)
                det_pos = np.array([det_x, det_y, det_z])

                # 批量计算所有时刻的状态
                self._compute_smoke_states_vectorized(
                    smoke_idx, t_drop, detonate_time, det_pos
                )

                smoke_idx += 1

    def _compute_smoke_states_vectorized(self, smoke_idx: int,
                                          t_drop: float, detonate_time: float,
                                          det_pos: np.ndarray):
        """向量化计算单个烟幕弹在所有时刻的状态"""

        # 阶段1: 未投放
        mask_not_dropped = self.t_array < t_drop
        # 保持默认值 (center=0, sigma=0, is_effective=False)

        # 阶段2: 下落中
        mask_falling = (self.t_array >= t_drop) & (self.t_array < detonate_time)
        if np.any(mask_falling):
            t_fall = self.t_array[mask_falling] - t_drop
            # 使用UAV速度计算水平位移 (简化：假设下落期间水平速度不变)
            # 这里需要从配置获取，暂时简化处理
            self.centers[smoke_idx, mask_falling, 2] = np.maximum(
                det_pos[2] + 0.5 * _FALL_ACCEL * ((detonate_time - self.t_array[mask_falling]) ** 2), 0
            )

        # 阶段3: 起爆后
        mask_detonated = self.t_array >= detonate_time
        if np.any(mask_detonated):
            t_elapsed = self.t_array[mask_detonated] - detonate_time

            # 风场漂移 + 下沉
            drift = np.outer(t_elapsed, _DRIFT_VEL)  # (n_det, 3)
            centers = det_pos + drift
            centers[:, 2] = np.maximum(centers[:, 2], 0)

            self.centers[smoke_idx, mask_detonated] = centers

            # 标准差
            self.sigmas[smoke_idx, mask_detonated] = _SIGMA_0 + _DIFFUSION_RATE * t_elapsed

            # 状态标志
            self.is_detonated[smoke_idx, mask_detonated] = True
            self.is_effective[smoke_idx, mask_detonated] = t_elapsed <= _MAX_EFFECTIVE_TIME

    def get_states_at_t(self, t: float) -> tuple:
        """
        获取时刻t的所有烟幕状态

        Returns:
            (centers, sigmas, is_effective, is_detonated, Q_values)
        """
        # 找到最近的时间索引
        idx = int(round(t / self.dt))
        idx = min(idx, self.n_times - 1)

        return (
            self.centers[:, idx, :],      # (n_smokes, 3)
            self.sigmas[:, idx],          # (n_smokes,)
            self.is_effective[:, idx],    # (n_smokes,)
            self.is_detonated[:, idx],    # (n_smokes,)
            self.Q_values                 # (n_smokes,)
        )


def get_smoke_state(uav_num: int, smoke_index: int,
                    drop_delay: float, detonate_delay: float,
                    uav_speed: float, uav_dir: np.ndarray,
                    t: float) -> dict:
    """
    计算高斯烟团在时刻 t 的状态（兼容接口）
    """
    detonate_time = drop_delay + detonate_delay

    # 阶段1: 未投放
    if t < drop_delay - 1e-6:
        return {
            "id": f"FY{uav_num}-{smoke_index}",
            "is_detonated": False,
            "center": None,
            "sigma": 0.0,
            "is_effective": False,
            "t_elapsed": 0.0,
            "Q": _Q,
        }

    # 计算投放点
    drop_pos = get_uav_drop_position(uav_num, drop_delay, uav_speed, uav_dir)

    # 阶段2: 下落中
    if t < detonate_time - 1e-6:
        fall_time = t - drop_delay
        horizontal_disp = uav_speed * fall_time * uav_dir
        current_x = drop_pos[0] + horizontal_disp[0]
        current_y = drop_pos[1] + horizontal_disp[1]
        z_fall = 0.5 * _FALL_ACCEL * (fall_time ** 2)
        current_z = max(drop_pos[2] - z_fall, 0)

        return {
            "id": f"FY{uav_num}-{smoke_index}",
            "is_detonated": False,
            "center": np.array([current_x, current_y, current_z]),
            "sigma": 0.0,
            "is_effective": False,
            "t_elapsed": 0.0,
            "Q": _Q,
        }

    # 阶段3: 起爆后
    t_elapsed = t - detonate_time
    is_effective = t_elapsed <= _MAX_EFFECTIVE_TIME

    # 起爆点位置
    total_fall_time = detonate_delay
    horizontal_disp = uav_speed * total_fall_time * uav_dir
    det_x = drop_pos[0] + horizontal_disp[0]
    det_y = drop_pos[1] + horizontal_disp[1]
    z_fall = 0.5 * _FALL_ACCEL * (total_fall_time ** 2)
    det_z = max(drop_pos[2] - z_fall, 0)
    det_pos = np.array([det_x, det_y, det_z])

    # 风场漂移
    drift = _DRIFT_VEL * t_elapsed
    current_center = det_pos + drift
    current_center[2] = max(current_center[2], 0)

    # 标准差
    sigma = compute_sigma(t_elapsed)

    return {
        "id": f"FY{uav_num}-{smoke_index}",
        "is_detonated": True,
        "center": current_center,
        "sigma": sigma,
        "is_effective": is_effective,
        "t_elapsed": t_elapsed,
        "Q": _Q,
    }


def compute_concentration_at_point(point: np.ndarray, smoke_center: np.ndarray,
                                    sigma: float, Q: float) -> float:
    """计算空间某点的高斯烟团浓度"""
    if sigma < 1e-6:
        return 0.0

    dist_sq = np.sum((point - smoke_center) ** 2)
    norm_factor = Q / ((2 * np.pi) ** 1.5 * sigma ** 3)
    return norm_factor * np.exp(-dist_sq / (2 * sigma ** 2))
