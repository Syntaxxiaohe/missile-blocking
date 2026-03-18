"""
Baseline3 全局参数配置模块

核心升级：
  1. 四大战术扇区（移除天顶E）
  2. 环形CAP巡逻阵型（5架UAV）
  3. 切线方向初始化
  4. 全局时间窗计算
"""

import numpy as np

# ========================== 场景几何参数 ==========================

REAL_TARGET = {
    "center": np.array([0.0, 0.0, 0.0]),
    "radius": 7.0,
    "height": 10.0
}

ORIGIN = REAL_TARGET["center"]

# ========================== 风场参数 ==========================

WIND_PARAMS = {
    "vector": np.array([5.0, -3.0, 0.0]),  # 向东南吹
    "enabled": True,
}

# ========================== 导弹参数 ==========================

MISSILE_SPEED = 300.0  # m/s

# 兼容Baseline2的静态导弹配置（回退用）
MISSILE_PARAMS = {
    "speed": 300.0,
    "init_pos": {
        1: np.array([20000.0, 0.0, 2000.0]),
        2: np.array([19000.0, 600.0, 2100.0]),
        3: np.array([18000.0, -600.0, 1900.0])
    }
}

# ========================== 四大战术扇区 ==========================
# 考虑风场 V_wind = [5.0, -3.0, 0.0] m/s (向东南吹)

MISSILE_SECTORS = {
    "A": {
        "name": "西北迎风/高空",
        "description": "迎风方向，高空突防，烟幕会被风吹向目标",
        "angle_range": (120, 150),         # 方位角范围（度）
        "distance_range": (15000, 20000),  # 距离范围 (m)
        "height_range": (2000, 4000),      # 高度范围 (m)
    },
    "B": {
        "name": "东南背风/超低空",
        "description": "【极度危险】背风方向，超低空突防，烟幕会被风吹离目标",
        "angle_range": (-60, -30),         # 等价于 300°-330°
        "distance_range": (10000, 15000),  # 距离范围 (m)
        "height_range": (50, 300),         # 超低空 (m)
    },
    "C": {
        "name": "东北侧风/中空",
        "description": "侧风方向，中空突防",
        "angle_range": (30, 60),           # 方位角范围（度）
        "distance_range": (12000, 18000),  # 距离范围 (m)
        "height_range": (500, 1500),       # 高度范围 (m)
    },
    "D": {
        "name": "西南侧风/中空",
        "description": "侧风方向，中空突防",
        "angle_range": (210, 240),         # 方位角范围（度）
        "distance_range": (12000, 18000),  # 距离范围 (m)
        "height_range": (500, 1500),       # 高度范围 (m)
    },
}

# ========================== UAV 环形CAP巡逻阵型 ==========================

N_UAVS = 5
CAP_RADIUS = 3000.0  # 巡逻圆环半径 (m) - 收缩防线，建立近程铁桶阵
CAP_HEIGHT = 800.0   # 巡逻高度 (m) - 更贴近掠海导弹高度


def compute_uav_cap_positions():
    """
    计算环形CAP巡逻阵型的UAV初始位置

    5架UAV均匀分布在半径8000m的圆环上（每72度一架）
    """
    positions = {}
    for i in range(1, N_UAVS + 1):
        theta = 2 * np.pi * (i - 1) / N_UAVS  # 0°, 72°, 144°, 216°, 288°
        x = CAP_RADIUS * np.cos(theta)
        y = CAP_RADIUS * np.sin(theta)
        z = CAP_HEIGHT
        positions[i] = np.array([x, y, z])
    return positions


UAV_PARAMS = {
    "init_pos": compute_uav_cap_positions(),
    "speed_range": [70.0, 140.0],
    "n_uavs": N_UAVS,
    "cap_radius": CAP_RADIUS,
    "cap_height": CAP_HEIGHT,
}

# ========================== 高斯烟团参数 ==========================

GAUSSIAN_SMOKE_PARAMS = {
    "Q": 300000.0,
    "sigma_0": 25.0,
    "diffusion_rate": 8.0,
    "alpha": 0.3,
    "transmittance_threshold": 0.10,
    "sink_speed": 3.0,
    "fall_accel": 9.8,
    "effective_sigma_multiple": 3.0,
    "max_effective_time": 60.0,
    "drop_interval": 1.0
}

SMOKE_PARAMS = GAUSSIAN_SMOKE_PARAMS

# ========================== PNG 导弹制导参数 ==========================

PNG_PARAMS = {
    "N": 4.0,
    "n_max": 25.0,
    "mass": 200.0,
    "C_d": 0.3,
    "S_ref": 0.02,
    "rho": 1.225,
    "g_vec": np.array([0.0, 0.0, -9.8]),
    "g_scalar": 9.8,
    "R_base": 7.0,
    "R_kill": 5.0,
    "FOV_max_deg": 30.0,
    "FOV_max_rad": np.radians(30.0),
    "dt": 0.01,
}

# ========================== 仿真控制参数 ==========================

MAX_SIM_TIME = 100.0  # 增加以适应多导弹场景
BASE_TIME_STEP = 0.1
LARGE_TIME_STEP = 0.5

# ========================== 加速参数 ==========================

ACCEL_PARAMS = {
    "smoke_cache_dt": 0.1,
    "smoke_cache_size": 1000,
    "use_jit": True,
    "vectorize_rays": True,
    "record_trajectory": False,
    "early_stop_blind": True,
    "n_workers": 4,
}

# ========================== 优化参数 ==========================

SMOKE_PAYLOAD_PER_UAV = 3
SAFE_DISTANCE = 15.0
SUCCESS_SCORE_WEIGHT = 10000


def get_uav_bounds(global_t_drop_range=None):
    """生成优化变量边界"""
    if global_t_drop_range is None:
        global_t_drop_range = (20.0, 50.0)

    bounds = [
        (-1.0, 1.0),     # dir_x
        (-1.0, 1.0),     # dir_y
        (-1.0, 1.0),     # dir_z
        (70.0, 140.0),   # speed (m/s)
    ]

    for _ in range(SMOKE_PAYLOAD_PER_UAV):
        bounds.append(global_t_drop_range)  # t_drop
        bounds.append((6.0, 10.0))          # d_det

    return bounds


def compute_global_tdrop_range(missile_positions, uav_positions=None):
    """
    计算全局投放时间窗（简化版）

    窗口: [t_earliest - 20.0, t_latest + 5.0]
    UAV本就在巡逻，不需要精算飞行时间
    """
    if not missile_positions:
        return (20.0, 60.0)

    # 计算每枚导弹的到达时间
    arrival_times = []
    for m_pos in missile_positions:
        D_m = np.sqrt(m_pos[0]**2 + m_pos[1]**2)
        t_arrival = D_m / MISSILE_SPEED
        arrival_times.append(t_arrival)

    t_earliest = min(arrival_times)
    t_latest = max(arrival_times)

    # 简化时间窗
    t_drop_min = max(5.0, t_earliest - 20.0)
    t_drop_max = t_latest + 5.0

    return (t_drop_min, t_drop_max)


# ========================== 辅助函数 ==========================

def get_wind_vector():
    if WIND_PARAMS["enabled"]:
        return WIND_PARAMS["vector"].copy()
    return np.zeros(3)


def compute_sigma(t_elapsed: float) -> float:
    return GAUSSIAN_SMOKE_PARAMS["sigma_0"] + \
           GAUSSIAN_SMOKE_PARAMS["diffusion_rate"] * t_elapsed


def get_uav_smoke_payloads():
    return {i: SMOKE_PAYLOAD_PER_UAV for i in range(1, N_UAVS + 1)}


# ========================== 打印配置摘要 ==========================

def print_config_summary():
    """打印当前配置摘要"""
    print("=" * 60)
    print("Baseline3 配置摘要")
    print("=" * 60)
    print(f"UAV数量: {N_UAVS}")
    print(f"CAP巡逻半径: {CAP_RADIUS}m")
    print(f"CAP巡逻高度: {CAP_HEIGHT}m")
    print(f"风场: {WIND_PARAMS['vector']} m/s")
    print("\nUAV阵位:")
    for i, pos in UAV_PARAMS["init_pos"].items():
        theta = np.degrees(2 * np.pi * (i - 1) / N_UAVS)
        print(f"  UAV{i}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) @ {theta:.0f}°")
    print("\n四大战术扇区:")
    for key, sector in MISSILE_SECTORS.items():
        print(f"  {key}: {sector['name']}")
        print(f"      角度: {sector['angle_range']}°, 距离: {sector['distance_range']}m")
        print(f"      高度: {sector['height_range']}m")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
