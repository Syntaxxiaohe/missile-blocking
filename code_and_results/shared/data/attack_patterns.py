"""
四种经典导弹攻击模式参数定义

基于以下学术文献:
- IEEE T-AES 2021: Three-dimensional impact angle constrained distributed
  cooperative guidance law for anti-ship missiles
- Aerospace 2022: Design and Optimization of Multimissile Formation Based on
  the Adaptive SA-PSO Algorithm
- IEEE Xplore 2023: Weapon-Target Assignment Strategy in Joint Combat
  Decision-Making based on Multi-head Deep Reinforcement Learning
- MDPI Applied Sciences 2021: Studies on Multi-Constraints Cooperative
  Guidance Method Based on Distributed MPC for Multi-Missiles
- IEEE T-AES 2024: Distributed Cooperative Strategy of UAV Swarm Without
  Speed Measurement Under Saturation Attack Mission
"""

from typing import Dict, List


# ============================================================
# 全局时间同步参数（饱和攻击核心特征）
# 所有导弹在 [base - sync_window/2, base + sync_window/2] 窗口内到达
# 使用正态分布: mean=base_arrival_time, std=sync_window/6
# ============================================================
TEMPORAL_SYNC = {
    "base_arrival_time": 45.0,   # 基准到达时间 (s)
    "sync_window":        5.0,   # 同步窗口宽度 (s)，3σ覆盖99.7%
}

# ============================================================
# 四种攻击模式定义
# ============================================================
ATTACK_PATTERNS: Dict[str, Dict] = {

    # ----------------------------------------------------------
    # Mode-1: 单向窄扇区饱和打击
    # 战术：地形/海岸线限制，全部导弹从同一方向集中突防
    # 学术依据：IEEE T-AES 2021，三维撞击角约束制导
    # ----------------------------------------------------------
    "mode_1_single_sector": {
        "name":    "单向窄扇区饱和打击",
        "name_en": "Single-Sector Saturation Attack",
        "description": "30°~60° 窄扇区内高密度分布，考验防御资源复用能力",
        "sectors": [
            {
                "sector_id":      "single",
                "angle_range":    [30, 60],
                "distance_range": [12000, 18000],
                "height_range":   [800, 2000],
                "missile_count":  20,
                "weight":         1.0,
            }
        ],
        "temporal": TEMPORAL_SYNC,
        "references": [
            "IEEE T-AES 2021: Three-dimensional impact angle constrained guidance",
            "IEEE Access 2023: Weapon Target Assessment With Dynamic Hit Probabilities",
        ],
        "validation_target": "验证极高密度下的资源复用率与防过载能力",
    },

    # ----------------------------------------------------------
    # Mode-2: 正交钳形打击
    # 战术：多平台协同，从北向和东向同时发起90°正交夹击
    # 学术依据：Aerospace 2022，多导弹编队优化
    # ----------------------------------------------------------
    "mode_2_orthogonal_pincer": {
        "name":    "正交钳形打击",
        "name_en": "Orthogonal Pincer Attack",
        "description": "北向10枚 + 东向10枚，90°正交夹击，考验无人机兵力分割调度",
        "sectors": [
            {
                "sector_id":      "north",
                "angle_range":    [80, 100],
                "distance_range": [12000, 18000],
                "height_range":   [500, 1500],
                "missile_count":  10,
                "weight":         0.5,
            },
            {
                "sector_id":      "east",
                "angle_range":    [-10, 10],
                "distance_range": [12000, 18000],
                "height_range":   [1500, 3000],
                "missile_count":  10,
                "weight":         0.5,
            },
        ],
        "temporal": TEMPORAL_SYNC,
        "references": [
            "Aerospace 2022: Design and Optimization of Multimissile Formation",
            "MODSIM 2013: A Many-on-Many Simulation Framework",
        ],
        "validation_target": "验证跨越超大空间隔离带的无人机兵力分割调度逻辑",
    },

    # ----------------------------------------------------------
    # Mode-3: 非对称多向饱和攻击
    # 战术：主攻14枚+两翼牵制各3枚，极度兵力不对称
    # 学术依据：IEEE Xplore 2023，多头深度强化学习WTA
    # ----------------------------------------------------------
    "mode_3_asymmetric_saturation": {
        "name":    "非对称多向饱和攻击",
        "name_en": "Asymmetric Multi-Directional Saturation Attack",
        "description": "北向14枚 + 东南3枚 + 西南3枚，极度兵力不对称",
        "sectors": [
            {
                "sector_id":      "main_assault",
                "angle_range":    [70, 110],
                "distance_range": [12000, 18000],
                "height_range":   [800, 2000],
                "missile_count":  14,
                "weight":         0.70,
            },
            {
                "sector_id":      "flank_a",
                "angle_range":    [100, 130],
                "distance_range": [12000, 18000],
                "height_range":   [800, 2000],
                "missile_count":  3,
                "weight":         0.15,
            },
            {
                "sector_id":      "flank_b",
                "angle_range":    [-130, -100],
                "distance_range": [12000, 18000],
                "height_range":   [800, 2000],
                "missile_count":  3,
                "weight":         0.15,
            },
        ],
        "temporal": TEMPORAL_SYNC,
        "references": [
            "IEEE Xplore 2023: Weapon-Target Assignment via Multi-head Deep RL",
            "ResearchGate: WTA Strategy in Joint Combat Decision-Making",
        ],
        "validation_target": "验证极度兵力劣势下基于威胁权重的动态优先级与资源舍弃策略",
    },

    # ----------------------------------------------------------
    # Mode-4: 全向蜂群突防
    # 战术：360°均匀包围，奇偶导弹高低交替，同时向心突击
    # 学术依据：MDPI Applied Sciences 2021，分布式MPC协同制导
    # ----------------------------------------------------------
    "mode_4_full_360_swarm": {
        "name":    "全向蜂群突防",
        "name_en": "Omnidirectional Swarm Attack",
        "description": "360°均匀包围，同时向心攻击，终极空间离散测试",
        "sectors": [
            {
                "sector_id":      "full_circle",
                "angle_range":    [-180, 180],
                "distance_range": [12000, 18000],
                "height_range":   [500, 3000],
                "missile_count":  20,
                "weight":         1.0,
                "distribution":   "uniform_azimuth",
            }
        ],
        "height_pattern": {
            "odd_missiles":  [500, 1000],    # 奇数编号：超低空
            "even_missiles": [2000, 3000],   # 偶数编号：高空俯冲
        },
        "temporal": TEMPORAL_SYNC,
        "references": [
            "MDPI Applied Sciences 2021: DMPC for Multi-Missiles",
            "IEEE T-AES 2024: UAV Swarm Without Speed Measurement",
        ],
        "validation_target": "验证无明确空间簇环境下的高维时空视线几何阻断聚类能力",
    },
}


def get_pattern_config(pattern_name: str) -> Dict:
    """获取指定攻击模式的配置，不存在则抛出 KeyError"""
    if pattern_name not in ATTACK_PATTERNS:
        raise KeyError(
            f"未知攻击模式: '{pattern_name}'。"
            f"可用模式: {list(ATTACK_PATTERNS.keys())}"
        )
    return ATTACK_PATTERNS[pattern_name]


def list_all_patterns() -> List[str]:
    """列出所有攻击模式名称"""
    return list(ATTACK_PATTERNS.keys())


if __name__ == "__main__":
    print("已定义的攻击模式：")
    for name, cfg in ATTACK_PATTERNS.items():
        total_missiles = sum(s["missile_count"] for s in cfg["sectors"])
        print(f"  {name:40s}  导弹总数={total_missiles:3d}  {cfg['name_en']}")
