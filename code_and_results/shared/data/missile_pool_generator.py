"""
导弹池生成器

为每种攻击模式生成包含 100 枚候选导弹的池（missile pool）。
每轮实验从池中随机采样 20 枚进行防御评估。

设计原则：
  1. 导弹位置严格在对应模式的空间约束范围内
  2. 所有导弹的到达时间（ToA）使用正态分布确保"同时弹着"特征：
       arrival_time ~ N(base_arrival_time, (sync_window/6)²)
     3σ范围恰好落在 [base - sync_window/2, base + sync_window/2] 内
  3. 固定随机种子确保实验可复现
  4. 输出格式与 shared/simulation/single.py 完全兼容（需含 id, position, velocity, arrival_time）
"""

import json
import os
from typing import Dict, List

import numpy as np

from .attack_patterns import ATTACK_PATTERNS, TEMPORAL_SYNC

# ============================================================
# 物理常数（与 shared/config/params.py 保持一致）
# ============================================================
MISSILE_SPEED = 300.0   # m/s，不得修改


# ============================================================
# 核心生成函数
# ============================================================

def _generate_single_missile(
    missile_id: int,
    sector_config: Dict,
    temporal_config: Dict,
    rng: np.random.Generator,
) -> Dict:
    """
    根据扇区配置和时间约束生成单枚导弹。

    Args:
        missile_id:     导弹编号（从1开始）
        sector_config:  扇区配置（angle_range, distance_range, height_range, sector_id）
        temporal_config: 时间约束（base_arrival_time, sync_window）
        rng:            已初始化的随机数生成器

    Returns:
        与 threat_generator.py 格式兼容的导弹配置字典
    """
    # 方位角（度 → 弧度）
    angle_deg = rng.uniform(
        sector_config["angle_range"][0],
        sector_config["angle_range"][1],
    )
    angle_rad = np.radians(angle_deg)

    # 随机距离和高度
    distance = rng.uniform(
        sector_config["distance_range"][0],
        sector_config["distance_range"][1],
    )
    height = rng.uniform(
        sector_config["height_range"][0],
        sector_config["height_range"][1],
    )

    # 三维位置（坐标系：X东，Y北，Z高）
    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    z = float(height)

    # 速度方向指向原点（目标中心）
    position  = np.array([x, y, z])
    direction = -position                       # 目标在原点
    dist_norm = np.linalg.norm(direction)
    velocity  = (direction / dist_norm) * MISSILE_SPEED

    # ——————————————————————————————————————————————
    # ToA 正态分布：确保所有导弹时间同步（饱和攻击核心特征）
    # std = sync_window / 6，使 99.7% 样本落在 [base ± sync_window/2] 内
    # ——————————————————————————————————————————————
    base_t = temporal_config["base_arrival_time"]
    sigma_t = temporal_config["sync_window"] / 6.0
    arrival_time = float(rng.normal(loc=base_t, scale=sigma_t))

    return {
        "id":           missile_id,
        "sector":       str(sector_config.get("sector_id", "pool")),
        "position":     [float(x), float(y), float(z)],
        "velocity":     [float(v) for v in velocity],
        "distance":     float(distance),
        "height":       float(height),
        "azimuth_deg":  float(angle_deg),
        "arrival_time": arrival_time,
    }


def _generate_uniform_azimuth_pool(
    pattern_config: Dict,
    pool_size: int,
    temporal_config: Dict,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Mode-4 专用：全向均匀分布导弹池。

    方位角在 [-180°, 180°] 上均匀铺开（相邻间隔 360°/pool_size），
    并加入小扰动避免对称；奇偶导弹高低交替模拟立体攻势。
    ToA 同样使用正态分布。
    """
    sector = pattern_config["sectors"][0]
    height_pattern = pattern_config.get("height_pattern", {})
    missile_pool: List[Dict] = []

    for i in range(pool_size):
        # 均匀方位角 + 小扰动（±5°）
        base_angle = -180.0 + 360.0 * i / pool_size
        angle_deg  = base_angle + rng.uniform(-5.0, 5.0)
        angle_rad  = np.radians(angle_deg)

        distance = rng.uniform(
            sector["distance_range"][0],
            sector["distance_range"][1],
        )

        # 高低交替
        if height_pattern:
            if i % 2 == 0:
                height = rng.uniform(*height_pattern["even_missiles"])
            else:
                height = rng.uniform(*height_pattern["odd_missiles"])
        else:
            height = rng.uniform(*sector["height_range"])

        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        z = float(height)

        position  = np.array([x, y, z])
        direction = -position
        dist_norm = np.linalg.norm(direction)
        velocity  = (direction / dist_norm) * MISSILE_SPEED

        # ToA 正态分布（与其他模式保持一致）
        base_t   = temporal_config["base_arrival_time"]
        sigma_t  = temporal_config["sync_window"] / 6.0
        arrival_time = float(rng.normal(loc=base_t, scale=sigma_t))

        missile_pool.append({
            "id":           i + 1,
            "sector":       f"azimuth_{i:03d}",
            "position":     [float(x), float(y), float(z)],
            "velocity":     [float(v) for v in velocity],
            "distance":     float(distance),
            "height":       float(height),
            "azimuth_deg":  float(angle_deg),
            "arrival_time": arrival_time,
        })

    return missile_pool


def generate_missile_pool(
    pattern_name: str,
    pool_size: int = 100,
    seed: int = None,
) -> List[Dict]:
    """
    为指定攻击模式生成导弹池。

    Args:
        pattern_name: 攻击模式名称（见 attack_patterns.ATTACK_PATTERNS）
        pool_size:    导弹池大小，默认 100 枚
        seed:         随机种子，固定后实验可复现

    Returns:
        包含 pool_size 枚导弹配置的列表
    """
    if pattern_name not in ATTACK_PATTERNS:
        raise KeyError(f"未知攻击模式: {pattern_name}")

    pattern_config = ATTACK_PATTERNS[pattern_name]
    sectors  = pattern_config["sectors"]
    temporal = pattern_config["temporal"]
    rng      = np.random.default_rng(seed)

    # Mode-4 全向蜂群：特殊均匀方位角分布
    if pattern_name == "mode_4_full_360_swarm":
        return _generate_uniform_azimuth_pool(pattern_config, pool_size, temporal, rng)

    # 普通模式：按扇区权重分配导弹数量
    missile_pool: List[Dict] = []
    missile_id = 1

    for sector in sectors:
        sector_count = round(pool_size * sector["weight"])
        for _ in range(sector_count):
            missile = _generate_single_missile(missile_id, sector, temporal, rng)
            missile_pool.append(missile)
            missile_id += 1

    # 补齐到 pool_size（权重四舍五入可能有误差）
    while len(missile_pool) < pool_size:
        sector_idx = int(rng.integers(len(sectors)))
        sector = sectors[sector_idx]
        missile = _generate_single_missile(missile_id, sector, temporal, rng)
        missile_pool.append(missile)
        missile_id += 1

    # 截断到 pool_size（避免超出）
    missile_pool = missile_pool[:pool_size]

    # 重新编号（确保 id 连续）
    for idx, m in enumerate(missile_pool):
        m["id"] = idx + 1

    return missile_pool


# ============================================================
# 采样工具
# ============================================================

def sample_from_pool(
    missile_pool: List[Dict],
    n_samples: int = 20,
    seed: int = None,
) -> List[Dict]:
    """
    从导弹池中无重复随机采样，并重新分配连续 id。

    Args:
        missile_pool: 导弹池（generate_missile_pool 的返回值）
        n_samples:    采样数量，默认 20
        seed:         随机种子，固定后每次采样结果可复现

    Returns:
        采样得到的 n_samples 枚导弹列表
    """
    if n_samples > len(missile_pool):
        raise ValueError(
            f"采样数 ({n_samples}) 超过池容量 ({len(missile_pool)})"
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(missile_pool), size=n_samples, replace=False)
    sampled = [dict(missile_pool[i]) for i in indices]  # 浅拷贝避免污染原池

    # 重新分配连续 id（1, 2, ..., n_samples）
    for new_id, m in enumerate(sampled, start=1):
        m["id"] = new_id
    return sampled


# ============================================================
# I/O 工具
# ============================================================

def save_missile_pool(missile_pool: List[Dict], filepath: str) -> None:
    """将导弹池序列化到 JSON 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {"pool_size": len(missile_pool), "missiles": missile_pool},
            f, indent=2, ensure_ascii=False,
        )
    print(f"[Pool] 已保存 {len(missile_pool)} 枚导弹 → {filepath}")


def load_missile_pool(filepath: str) -> List[Dict]:
    """从 JSON 文件加载导弹池"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"导弹池文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["missiles"]


# ============================================================
# 快速自测
# ============================================================

if __name__ == "__main__":
    from .attack_patterns import list_all_patterns

    for pname in list_all_patterns():
        pool = generate_missile_pool(pname, pool_size=100, seed=42)
        arrival_times = [m["arrival_time"] for m in pool]
        print(
            f"{pname:40s}  n={len(pool)}"
            f"  ToA: mean={np.mean(arrival_times):.2f}s"
            f"  std={np.std(arrival_times):.3f}s"
            f"  [{min(arrival_times):.2f}, {max(arrival_times):.2f}]"
        )
