"""
无人机运动学模块 — 加速版

加速特性：
  1. 向量化计算多个时刻的位置
  2. 预计算UAV方向向量
"""

import numpy as np
from shared.config.params import UAV_PARAMS


def get_uav_position_at_t(uav_num: int, t: float,
                           uav_speed: float, uav_dir: np.ndarray) -> tuple:
    """
    计算无人机在时刻 t 的空间坐标（单点计算）
    """
    if uav_num not in UAV_PARAMS["init_pos"]:
        raise ValueError(f"无人机编号仅支持 1~5，当前输入：{uav_num}")
    if not (UAV_PARAMS["speed_range"][0] <= uav_speed <= UAV_PARAMS["speed_range"][1]):
        raise ValueError(f"无人机速度需在 {UAV_PARAMS['speed_range']} m/s 内，当前：{uav_speed}")
    if abs(np.linalg.norm(uav_dir) - 1) > 1e-6:
        raise ValueError(f"飞行方向需为单位向量，当前模长：{np.linalg.norm(uav_dir)}")

    init_pos = UAV_PARAMS["init_pos"][uav_num]
    displacement = uav_speed * t * uav_dir
    return tuple(init_pos + displacement)


def get_uav_positions_vectorized(uav_num: int, t_array: np.ndarray,
                                  uav_speed: float, uav_dir: np.ndarray) -> np.ndarray:
    """
    向量化计算无人机在多个时刻的空间坐标

    Args:
        uav_num: 无人机编号 (1~5)
        t_array: 时刻数组 (N,)
        uav_speed: 飞行速度 (m/s)
        uav_dir: 飞行方向单位向量 (3D)

    Returns:
        positions: 位置数组 (N, 3)
    """
    init_pos = UAV_PARAMS["init_pos"][uav_num]

    # 向量化计算位移
    displacements = np.outer(t_array, uav_speed * uav_dir)  # (N, 3)
    positions = init_pos + displacements

    return positions


def get_uav_drop_position(uav_num: int, t_drop: float,
                           uav_speed: float, uav_dir: np.ndarray) -> np.ndarray:
    """
    计算投放时刻UAV位置（返回numpy数组，避免tuple转换）
    """
    init_pos = UAV_PARAMS["init_pos"][uav_num]
    return init_pos + uav_speed * t_drop * uav_dir


def calculate_uav_direction(uav_num: int) -> np.ndarray:
    """
    计算无人机朝向目标原点的三维飞行方向
    """
    uav_init_pos = UAV_PARAMS["init_pos"][uav_num]
    target_pos = np.array([0.0, 0.0, 0.0])

    direction_vector = target_pos - uav_init_pos
    direction_norm = np.linalg.norm(direction_vector)

    if direction_norm < 1e-6:
        raise ValueError("无人机与目标在空间上重合，无法计算方向！")

    return direction_vector / direction_norm


def normalize_direction(dir_x: float, dir_y: float, dir_z: float) -> np.ndarray:
    """
    快速归一化方向向量
    """
    dir_norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
    if dir_norm < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return np.array([dir_x / dir_norm, dir_y / dir_norm, dir_z / dir_norm])
