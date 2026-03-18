"""
导弹轨迹查询模块 — 匀速直线模型

用于快速估算导弹位置（优化器初始化时使用）
"""

import numpy as np
from shared.config.params import MISSILE_PARAMS, REAL_TARGET


def get_missile_position_at_t(missile_num: int, t: float) -> tuple:
    """
    计算导弹在时刻 t 的空间坐标（匀速直线近似）

    Args:
        missile_num: 导弹编号 (1/2/3)
        t: 时刻 (s)

    Returns:
        (x, y, z) 坐标元组
    """
    init_pos = MISSILE_PARAMS["init_pos"][missile_num]
    target_pos = REAL_TARGET["center"]

    # 飞行方向
    direction = target_pos - init_pos
    direction_norm = np.linalg.norm(direction)
    unit_direction = direction / direction_norm

    # 飞行速度
    speed = MISSILE_PARAMS["speed"]

    # 位移
    displacement = speed * t * unit_direction
    current_pos = init_pos + displacement

    return tuple(current_pos)


def get_missile_arrival_time(missile_num: int) -> float:
    """
    计算导弹到达目标的预计时间

    Args:
        missile_num: 导弹编号

    Returns:
        预计到达时间 (s)
    """
    init_pos = MISSILE_PARAMS["init_pos"][missile_num]
    target_pos = REAL_TARGET["center"]

    distance = np.linalg.norm(target_pos - init_pos)
    speed = MISSILE_PARAMS["speed"]

    return distance / speed
