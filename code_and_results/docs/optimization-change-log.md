# 方案改进优化记录表

> **创建时间**: 2026-03-08
> **创建人**: Claude Code
> **状态**: 待审核
> **关联文件**: 导弹攻击模式仿真场景研究.pdf (Gemini深度搜索报告)

---

## 一、改进背景与问题分析

### 1.1 当前方案存在的问题

| 问题编号 | 问题描述 | 风险等级 | 来源 |
|---------|---------|---------|------|
| P-001 | 四方位攻击策略（A/B/C/D扇区）过于极端，现实中难以出现 | 高 | 清华哥反馈 |
| P-002 | 20枚导弹初始位置固定，审稿人可能质疑模型过拟合 | 高 | 用户判断 |
| P-003 | 单一场景验证不足以证明算法泛化能力 | 中 | 学术规范 |
| P-004 | 缺乏对真实导弹协同制导约束的体现 | 中 | Gemini报告 |

### 1.2 审稿人可能的质疑

```
质疑1: "为什么导弹分布是这种四扇区模式？有什么军事或物理依据？"

质疑2: "作者只在一种固定的导弹分布下验证了算法，如何证明不是过拟合？"

质疑3: "导弹的到达时间差异很大，不符合饱和攻击的'同时弹着'特征"

质疑4: "场景设定脱离实战实际"
```

### 1.3 改进目标

1. **学术严谨性**：基于顶刊文献构建符合实战逻辑的攻击场景
2. **泛化能力验证**：多场景、多次随机评估
3. **避免过拟合质疑**：导弹池随机采样
4. **可复现性**：固定随机种子，确保实验可复现

---

## 二、四种经典攻击模式定义

### 2.1 模式概述

基于IEEE T-AES、Defense Technology等顶刊文献，以及Gemini深度搜索报告，定义以下四种经典导弹饱和攻击模式：

| 模式编号 | 模式名称 | 英文名称 | 战术特征 | 学术依据 |
|---------|---------|---------|---------|---------|
| Mode-1 | 单向窄扇区饱和打击 | Single-Sector Saturation Attack | 30°~60°窄扇区内高密度分布 | IEEE T-AES, 三维撞击角约束制导 |
| Mode-2 | 正交钳形打击 | Orthogonal Pincer Attack | 北向10枚 + 东向10枚，90°正交 | 多导弹协同制导律研究 |
| Mode-3 | 非对称多向饱和攻击 | Asymmetric Multi-Directional Saturation | 北向14枚 + 东西各3枚 | WTA武器目标分配文献 |
| Mode-4 | 全向蜂群突防 | Omnidirectional Swarm Attack | 360°均匀包围，同时向心攻击 | 分布式MPC协同制导 |

### 2.2 模式一：单向窄扇区饱和打击 (Mode-1)

#### 2.2.1 战术背景

单向扇区突防是防空反导仿真中最基础但也最考验防御资源调度极限的一种分布模式。攻击方由于地理地形限制（如海岸线山脉遮挡只能从海平面一侧突防）或防御方外围远程雷达盲区的限制，将所有导弹集中在单一的主威胁轴向上，形成极其密集的"导弹墙"。

#### 2.2.2 学术依据

- **IEEE T-AES 2021**: 《Three-dimensional impact angle constrained distributed cooperative guidance law for anti-ship missiles》
- **IEEE Access 2023**: 《Weapon Target Assessment With Dynamic Hit Probabilities and Heterogeneous Targets》

#### 2.2.3 参数定义

```python
MODE_1_SINGLE_SECTOR = {
    "name": "单向窄扇区饱和打击",
    "name_en": "Single-Sector Saturation Attack",
    "description": "30°~60°窄扇区内高密度分布，考验防御资源复用能力",

    # 空间分布参数
    "sectors": [
        {
            "sector_id": 1,
            "angle_range": [30, 60],           # 方位角范围 (度)
            "distance_range": [12000, 18000],   # 距离范围 (m)
            "height_range": [800, 2000],        # 高度范围 (m)
            "missile_count": 20,                # 导弹数量
            "weight": 1.0                       # 权重
        }
    ],

    # 时间同步参数（饱和攻击核心特征）
    "temporal": {
        "base_arrival_time": 45.0,              # 基准到达时间 (s)
        "sync_window": 5.0,                     # 同步窗口 (s)，所有导弹在此窗口内到达
        "description": "所有导弹在[42.5s, 47.5s]窗口内同时命中"
    },

    # 学术引用
    "references": [
        "IEEE T-AES 2021: Three-dimensional impact angle constrained guidance",
        "IEEE Access 2023: Weapon Target Assessment With Dynamic Hit Probabilities"
    ],

    # 算法验证目标
    "validation_target": "验证极高密度下的资源复用率与防过载能力"
}
```

#### 2.2.4 验证价值

- **空间密度极高**：20枚导弹集中在30°~60°扇区内
- **资源复用测试**：验证算法能否识别出"一颗烟幕弹同时遮蔽多枚导弹"的时空节点
- **预期表现**：M2/M3应能用少量烟雾弹（如6-8颗）化解20枚导弹威胁

---

### 2.3 模式二：正交钳形打击 (Mode-2)

#### 2.3.1 战术背景

当攻击方具备充裕的发射平台（如多艘水面舰艇、潜艇及空中挂载平台从不同方位同时发起攻击）时，为了最大限度地撕裂防空系统的火力配置，往往采用多轴向钳形打击。这是学术界应用最为广泛的中高阶仿真场景。

#### 2.3.2 学术依据

- **Aerospace 2022**: 《Design and Optimization of Multimissile Formation Based on the Adaptive SA-PSO Algorithm》
- **MODSIM 2013**: 基于多方向攻击在线目标分配模型

#### 2.3.3 参数定义

```python
MODE_2_ORTHOGONAL_PINCER = {
    "name": "正交钳形打击",
    "name_en": "Orthogonal Pincer Attack",
    "description": "北向10枚 + 东向10枚，90°正交夹击，考验无人机兵力分割调度",

    # 空间分布参数
    "sectors": [
        {
            "sector_id": "north",
            "angle_range": [80, 100],            # 北向扇区 (度)
            "distance_range": [12000, 18000],    # 距离范围 (m)
            "height_range": [500, 1500],         # 低空掠海 (m)
            "missile_count": 10,                 # 导弹数量
            "weight": 0.5                        # 权重 50%
        },
        {
            "sector_id": "east",
            "angle_range": [-10, 10],            # 东向扇区 (度)
            "distance_range": [12000, 18000],    # 距离范围 (m)
            "height_range": [1500, 3000],        # 高空俯冲 (m)
            "missile_count": 10,                 # 导弹数量
            "weight": 0.5                        # 权重 50%
        }
    ],

    # 时间同步参数
    "temporal": {
        "base_arrival_time": 45.0,              # 基准到达时间 (s)
        "sync_window": 5.0,                     # 同步窗口 (s)
        "description": "两个子集群必须同时命中目标"
    },

    # 学术引用
    "references": [
        "Aerospace 2022: Design and Optimization of Multimissile Formation",
        "MODSIM 2013: A Many-on-Many Simulation Framework"
    ],

    # 算法验证目标
    "validation_target": "验证跨越超大空间隔离带的无人机兵力分割调度逻辑"
}
```

#### 2.3.4 验证价值

- **空间隔离测试**：两个子集群相距90°，物理上完全隔离
- **降维拆分测试**：验证算法能否将20×5的WTA问题拆解为两个独立的10×?子问题
- **兵力调度测试**：验证算法能否计算出最优的无人机分配（如3架去北向，2架去东向）

---

### 2.4 模式三：非对称多向饱和攻击 (Mode-3)

#### 2.4.1 战术背景

在真实的战争对抗中，由于攻击方各平台的弹药基数不同，多向饱和攻击往往呈现出严重的兵力不对称。即在一个主攻方向上倾泻绝大多数火力，而在侧翼方向上部署少量导弹进行牵制。这种模式要求防空决策系统具备极强的高动态权重评估和优先级排序能力。

#### 2.4.2 学术依据

- **IEEE Xplore 2023**: 《Weapon-Target Assignment Strategy in Joint Combat Decision-Making based on Multi-head Deep Reinforcement Learning》
- **ResearchGate**: 非对称目标分配研究

#### 2.4.3 参数定义

```python
MODE_3_ASYMMETRIC_SATURATION = {
    "name": "非对称多向饱和攻击",
    "name_en": "Asymmetric Multi-Directional Saturation Attack",
    "description": "北向14枚 + 东南3枚 + 西南3枚，极度兵力不对称",

    # 空间分布参数
    "sectors": [
        {
            "sector_id": "main_assault",
            "angle_range": [70, 110],             # 北向主攻扇区 (度)
            "distance_range": [12000, 18000],     # 距离范围 (m)
            "height_range": [800, 2000],          # 中空突防 (m)
            "missile_count": 14,                  # 导弹数量 (70%)
            "weight": 0.70                        # 权重 70%
        },
        {
            "sector_id": "flank_a",
            "angle_range": [100, 130],            # 东南牵制扇区 (度)
            "distance_range": [12000, 18000],     # 距离范围 (m)
            "height_range": [800, 2000],          # 中空突防 (m)
            "missile_count": 3,                   # 导弹数量 (15%)
            "weight": 0.15                        # 权重 15%
        },
        {
            "sector_id": "flank_b",
            "angle_range": [-130, -100],          # 西南牵制扇区 (度)
            "distance_range": [12000, 18000],     # 距离范围 (m)
            "height_range": [800, 2000],          # 中空突防 (m)
            "missile_count": 3,                   # 导弹数量 (15%)
            "weight": 0.15                        # 权重 15%
        }
    ],

    # 时间同步参数
    "temporal": {
        "base_arrival_time": 45.0,              # 基准到达时间 (s)
        "sync_window": 5.0,                     # 同步窗口 (s)
        "description": "三个子集群同时命中目标"
    },

    # 学术引用
    "references": [
        "IEEE Xplore 2023: Weapon-Target Assignment via Multi-head Deep RL",
        "ResearchGate: WTA Strategy in Joint Combat Decision-Making"
    ],

    # 算法验证目标
    "validation_target": "验证极度兵力劣势下基于威胁权重的动态优先级与资源舍弃策略"
}
```

#### 2.4.4 验证价值

- **威胁权重评估**：验证算法能否识别出主攻方向的高威胁
- **资源取舍策略**：验证算法是否敢于"牺牲"部分防线以保全重点
- **运筹学取舍**：验证算法在资源绝对匮乏时的决策能力

---

### 2.5 模式四：全向蜂群突防 (Mode-4)

#### 2.5.1 战术背景

全向蜂群突防代表了现代协同攻击的最前沿技术，也是对当前防空系统火力通道理论上限的终极压迫。在此模式下，攻击方通过前期复杂的航路规划或利用空中母舰释放，使得导弹群在目标外围形成一个完整的360°包围圈，并从四面八方同时向中心目标发起向心突击。

#### 2.5.2 学术依据

- **MDPI Applied Sciences 2021**: 《Studies on Multi-Constraints Cooperative Guidance Method Based on Distributed MPC for Multi-Missiles》
- **IEEE T-AES 2024**: 《Distributed Cooperative Strategy of UAV Swarm Without Speed Measurement Under Saturation Attack Mission》

#### 2.5.3 参数定义

```python
MODE_4_FULL_360_SWARM = {
    "name": "全向蜂群突防",
    "name_en": "Omnidirectional Swarm Attack",
    "description": "360°均匀包围，同时向心攻击，终极空间离散测试",

    # 空间分布参数
    "sectors": [
        {
            "sector_id": "full_circle",
            "angle_range": [-180, 180],           # 全向覆盖 (度)
            "distance_range": [12000, 18000],     # 距离范围 (m)
            "height_range": [500, 3000],          # 高低搭配 (m)
            "missile_count": 20,                  # 导弹数量
            "weight": 1.0,                        # 权重
            "distribution": "uniform_azimuth"     # 方位角均匀分布
        }
    ],

    # 高低角波浪配置（模拟智能蜂群行为）
    "height_pattern": {
        "odd_missiles": [500, 1000],              # 奇数编号：超低空
        "even_missiles": [2000, 3000],            # 偶数编号：高空俯冲
        "description": "相邻导弹高低交替，形成立体攻势"
    },

    # 时间同步参数
    "temporal": {
        "base_arrival_time": 45.0,              # 基准到达时间 (s)
        "sync_window": 5.0,                     # 同步窗口 (s)
        "description": "20枚导弹从20个方向同时命中"
    },

    # 学术引用
    "references": [
        "MDPI Applied Sciences 2021: DMPC for Multi-Missiles",
        "IEEE T-AES 2024: UAV Swarm Without Speed Measurement"
    ],

    # 算法验证目标
    "validation_target": "验证无明确空间簇环境下的高维时空视线几何阻断聚类能力"
}
```

#### 2.5.4 验证价值

- **空间聚类失效测试**：20枚导弹均匀分布在360°圆周上，任意相邻导弹夹角仅18°
- **高维时空降维测试**：算法必须在时间+径向深度维度上寻找解域
- **理论极限测试**：这是对算法鲁棒性的终极考验

---

## 三、实验设计改进方案

### 3.1 导弹池设计方案

#### 3.1.1 设计原则

```
原则1: 每种攻击模式生成一个包含100枚候选导弹的池
原则2: 导弹位置必须在对应模式的空间约束范围内
原则3: 所有导弹必须满足时间同步约束（同时弹着窗口）
原则4: 使用固定随机种子确保可复现性
```

#### 3.1.2 导弹池参数

| 攻击模式 | 导弹池大小 | 空间约束 | 时间约束 | 随机种子 |
|---------|-----------|---------|---------|---------|
| Mode-1 单向窄扇区 | 100枚 | 方位角[30°,60°]，距离[12,18]km | 到达时间[42.5s,47.5s] | 20260308_01 |
| Mode-2 正交钳形 | 100枚 | 北向+东向各50枚 | 到达时间[42.5s,47.5s] | 20260308_02 |
| Mode-3 非对称多向 | 100枚 | 北向70枚+东西各15枚 | 到达时间[42.5s,47.5s] | 20260308_03 |
| Mode-4 全向蜂群 | 100枚 | 360°均匀分布 | 到达时间[42.5s,47.5s] | 20260308_04 |

#### 3.1.3 采样策略

```python
def sample_missiles_from_pool(missile_pool, n_samples=20, seed=None):
    """
    从导弹池中随机采样n_samples枚导弹

    Args:
        missile_pool: 包含100枚导弹的列表
        n_samples: 采样数量，默认20枚
        seed: 随机种子，用于可复现性

    Returns:
        采样得到的导弹配置列表
    """
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.choice(len(missile_pool), size=n_samples, replace=False)
    return [missile_pool[i] for i in indices]
```

### 3.2 多次评估设计

#### 3.2.1 评估次数设定

| 参数 | 值 | 说明 |
|------|-----|------|
| 评估次数 (N_EVAL) | 10 | 每种组合评估10次 |
| 随机种子范围 | [0, 9] | 每次评估使用不同种子 |
| 统计指标 | 均值、标准差、最大值、最小值 | 综合评估 |

#### 3.2.2 统计汇总表格模板

| 攻击模式 | 方法 | 评估次数 | 拦截数(均值±std) | 耗弹数(均值±std) | 效费比(均值±std) | 最优/最差 |
|---------|------|---------|-----------------|-----------------|-----------------|----------|
| Mode-1 | M1 | 10 | ? ± ? | 15.0 ± 0.0 | ? ± ? | ?/? |
| Mode-1 | M2 | 10 | ? ± ? | ? ± ? | ? ± ? | ?/? |
| Mode-1 | M3 | 10 | ? ± ? | ? ± ? | ? ± ? | ?/? |
| ... | ... | ... | ... | ... | ... | ... |

### 3.3 完整实验矩阵

```
实验维度:
├── 攻击模式: 4种 (Mode-1, Mode-2, Mode-3, Mode-4)
├── 方法: 3种 (M1_NaiveGA, M2_Clustering, M3_AmmoPenalty)
├── 评估次数: 10次 (随机采样不同导弹组合)
└── 总实验数: 4 × 3 × 10 = 120次
```

---

## 四、代码改动详细方案

### 4.1 文件改动清单

| 序号 | 操作类型 | 文件路径 | 改动内容 | 新增行数 |
|------|---------|---------|---------|---------|
| 1 | 新建 | `shared/data/attack_patterns.py` | 四种攻击模式参数定义 | ~80行 |
| 2 | 新建 | `shared/data/missile_pool_generator.py` | 导弹池生成器 | ~100行 |
| 3 | 新建 | `generate_all_scenarios.py` | 一键生成所有场景数据集 | ~60行 |
| 4 | 修改 | `shared/data/threat_generator.py` | 增加时间同步约束 | ~30行 |
| 5 | 修改 | `M1_NaiveGA/main.py` | 增加采样+多次评估逻辑 | ~40行 |
| 6 | 修改 | `M2_Clustering/main.py` | 增加采样+多次评估逻辑 | ~40行 |
| 7 | 修改 | `M3_AmmoPenalty/main.py` | 增加采样+多次评估逻辑 | ~40行 |
| 8 | 新建 | `summarize_results.py` | 结果汇总与统计 | ~80行 |
| **总计** | - | - | - | **~470行** |

### 4.2 核心模块改动评估

| 模块 | 是否需要改动 | 改动量 | 说明 |
|------|-------------|--------|------|
| **物理引擎** (`physics/*`) | ❌ 否 | 0 | 完全复用现有代码 |
| **评估函数** (`simulation/single.py`) | ❌ 否 | 0 | 完全复用现有代码 |
| **优化器** (`optimizer/*`) | ❌ 否 | 0 | 完全复用现有代码 |
| **分配器** (`allocator.py`) | ❌ 否 | 0 | 完全复用现有代码 |
| **威胁生成器** | ✅ 是 | ~30行 | 增加时间同步约束 |
| **主入口** (`main.py`) | ✅ 是 | ~40行/个 | 增加采样逻辑 |

### 4.3 新建文件详细设计

#### 4.3.1 `shared/data/attack_patterns.py`

```python
"""
四种经典导弹攻击模式参数定义

基于以下学术文献:
- IEEE T-AES 2021: Three-dimensional impact angle constrained guidance
- Aerospace 2022: Design and Optimization of Multimissile Formation
- IEEE Xplore 2023: Weapon-Target Assignment via Multi-head Deep RL
- MDPI Applied Sciences 2021: DMPC for Multi-Missiles
"""

import numpy as np
from typing import Dict, List, Tuple

# 时间同步参数（饱和攻击核心特征）
TEMPORAL_SYNC = {
    "base_arrival_time": 45.0,      # 基准到达时间 (s)
    "sync_window": 5.0,             # 同步窗口 (s)
}

# 四种攻击模式定义
ATTACK_PATTERNS = {
    "mode_1_single_sector": {
        "name": "单向窄扇区饱和打击",
        "name_en": "Single-Sector Saturation Attack",
        "sectors": [
            {"angle_range": [30, 60], "distance_range": [12000, 18000],
             "height_range": [800, 2000], "missile_count": 20, "weight": 1.0}
        ],
        "temporal": TEMPORAL_SYNC,
        "validation_target": "验证极高密度下的资源复用率与防过载能力"
    },

    "mode_2_orthogonal_pincer": {
        "name": "正交钳形打击",
        "name_en": "Orthogonal Pincer Attack",
        "sectors": [
            {"angle_range": [80, 100], "distance_range": [12000, 18000],
             "height_range": [500, 1500], "missile_count": 10, "weight": 0.5},
            {"angle_range": [-10, 10], "distance_range": [12000, 18000],
             "height_range": [1500, 3000], "missile_count": 10, "weight": 0.5}
        ],
        "temporal": TEMPORAL_SYNC,
        "validation_target": "验证跨越超大空间隔离带的无人机兵力分割调度逻辑"
    },

    "mode_3_asymmetric_saturation": {
        "name": "非对称多向饱和攻击",
        "name_en": "Asymmetric Multi-Directional Saturation Attack",
        "sectors": [
            {"angle_range": [70, 110], "distance_range": [12000, 18000],
             "height_range": [800, 2000], "missile_count": 14, "weight": 0.70},
            {"angle_range": [100, 130], "distance_range": [12000, 18000],
             "height_range": [800, 2000], "missile_count": 3, "weight": 0.15},
            {"angle_range": [-130, -100], "distance_range": [12000, 18000],
             "height_range": [800, 2000], "missile_count": 3, "weight": 0.15}
        ],
        "temporal": TEMPORAL_SYNC,
        "validation_target": "验证极度兵力劣势下基于威胁权重的动态优先级与资源舍弃策略"
    },

    "mode_4_full_360_swarm": {
        "name": "全向蜂群突防",
        "name_en": "Omnidirectional Swarm Attack",
        "sectors": [
            {"angle_range": [-180, 180], "distance_range": [12000, 18000],
             "height_range": [500, 3000], "missile_count": 20, "weight": 1.0,
             "distribution": "uniform_azimuth"}
        ],
        "temporal": TEMPORAL_SYNC,
        "height_pattern": {
            "odd_missiles": [500, 1000],
            "even_missiles": [2000, 3000]
        },
        "validation_target": "验证无明确空间簇环境下的高维时空视线几何阻断聚类能力"
    }
}


def get_pattern_config(pattern_name: str) -> Dict:
    """获取指定攻击模式的配置"""
    return ATTACK_PATTERNS.get(pattern_name)


def list_all_patterns() -> List[str]:
    """列出所有攻击模式名称"""
    return list(ATTACK_PATTERNS.keys())
```

#### 4.3.2 `shared/data/missile_pool_generator.py`

```python
"""
导弹池生成器

为每种攻击模式生成包含100枚候选导弹的池，
每轮实验从池中随机采样20枚。
"""

import numpy as np
import json
import os
from typing import List, Dict
from .attack_patterns import ATTACK_PATTERNS, TEMPORAL_SYNC

# 物理常数
MISSILE_SPEED = 300.0  # m/s


def generate_single_missile(
    sector_config: Dict,
    temporal_config: Dict,
    rng: np.random.Generator
) -> Dict:
    """
    根据扇区配置和时间约束生成单枚导弹

    Args:
        sector_config: 扇区配置（方位角、距离、高度范围）
        temporal_config: 时间约束配置
        rng: 随机数生成器

    Returns:
        导弹配置字典
    """
    # 随机方位角
    angle_deg = rng.uniform(
        sector_config["angle_range"][0],
        sector_config["angle_range"][1]
    )
    angle_rad = np.radians(angle_deg)

    # 随机距离和高度
    distance = rng.uniform(
        sector_config["distance_range"][0],
        sector_config["distance_range"][1]
    )
    height = rng.uniform(
        sector_config["height_range"][0],
        sector_config["height_range"][1]
    )

    # 计算位置
    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    z = height

    # 计算速度方向（指向原点）
    target = np.array([0.0, 0.0, 0.0])
    position = np.array([x, y, z])
    direction = target - position
    direction_norm = np.linalg.norm(direction)
    velocity = (direction / direction_norm) * MISSILE_SPEED

    # 计算到达时间（使用高斯分布确保钟形曲线集中到达）
    # 修正：使用正态分布而非均匀分布
    # scale = sync_window/6 确保 99.7% 的样本在 [base - sync_window/2, base + sync_window/2] 内
    base_arrival = temporal_config["base_arrival_time"]
    sync_window = temporal_config["sync_window"]
    arrival_time = rng.normal(loc=base_arrival, scale=sync_window / 6.0)

    return {
        "position": [float(x), float(y), float(z)],
        "velocity": [float(v) for v in velocity],
        "arrival_time": float(arrival_time),
        "sector_id": sector_config.get("sector_id", "default"),
        "angle_deg": float(angle_deg),
        "distance_m": float(distance)
    }


def generate_missile_pool(
    pattern_name: str,
    pool_size: int = 100,
    seed: int = None
) -> List[Dict]:
    """
    为指定攻击模式生成导弹池

    Args:
        pattern_name: 攻击模式名称
        pool_size: 导弹池大小，默认100枚
        seed: 随机种子

    Returns:
        导弹配置列表
    """
    pattern_config = ATTACK_PATTERNS[pattern_name]
    sectors = pattern_config["sectors"]
    temporal = pattern_config["temporal"]

    rng = np.random.default_rng(seed)
    missile_pool = []

    # 特殊处理：全向蜂群模式
    if pattern_name == "mode_4_full_360_swarm":
        return _generate_uniform_circle_pool(
            pattern_config, pool_size, temporal, rng
        )

    # 常规模式：按扇区权重分配导弹数量
    for sector in sectors:
        sector_count = int(pool_size * sector["weight"])
        sector_id = sector.get("sector_id", f"sector_{len(missile_pool)}")

        for _ in range(sector_count):
            missile = generate_single_missile(sector, temporal, rng)
            missile["sector_id"] = sector_id
            missile_pool.append(missile)

    # 补齐到pool_size
    while len(missile_pool) < pool_size:
        # 随机选择一个扇区补充
        sector = rng.choice(sectors)
        missile = generate_single_missile(sector, temporal, rng)
        missile_pool.append(missile)

    return missile_pool[:pool_size]


def _generate_uniform_circle_pool(
    pattern_config: Dict,
    pool_size: int,
    temporal: Dict,
    rng: np.random.Generator
) -> List[Dict]:
    """
    生成全向均匀分布的导弹池（用于Mode-4）
    """
    sector = pattern_config["sectors"][0]
    height_pattern = pattern_config.get("height_pattern", {})

    missile_pool = []

    for i in range(pool_size):
        # 方位角均匀分布
        angle_deg = -180 + (360 * i / pool_size) + rng.uniform(-5, 5)
        angle_rad = np.radians(angle_deg)

        # 随机距离
        distance = rng.uniform(
            sector["distance_range"][0],
            sector["distance_range"][1]
        )

        # 高低交替
        if height_pattern:
            if i % 2 == 0:
                height = rng.uniform(*height_pattern["even_missiles"])
            else:
                height = rng.uniform(*height_pattern["odd_missiles"])
        else:
            height = rng.uniform(*sector["height_range"])

        # 计算位置和速度
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        z = height

        target = np.array([0.0, 0.0, 0.0])
        position = np.array([x, y, z])
        direction = target - position
        direction_norm = np.linalg.norm(direction)
        velocity = (direction / direction_norm) * MISSILE_SPEED

        # 时间同步
        base_arrival = temporal["base_arrival_time"]
        sync_window = temporal["sync_window"]
        arrival_time = base_arrival + rng.uniform(-sync_window/2, sync_window/2)

        missile_pool.append({
            "position": [float(x), float(y), float(z)],
            "velocity": [float(v) for v in velocity],
            "arrival_time": float(arrival_time),
            "sector_id": f"azimuth_{i}",
            "angle_deg": float(angle_deg),
            "distance_m": float(distance)
        })

    return missile_pool


def save_missile_pool(missile_pool: List[Dict], filepath: str):
    """保存导弹池到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "pool_size": len(missile_pool),
            "missiles": missile_pool
        }, f, indent=2, ensure_ascii=False)


def load_missile_pool(filepath: str) -> List[Dict]:
    """从JSON文件加载导弹池"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["missiles"]


def sample_from_pool(
    missile_pool: List[Dict],
    n_samples: int = 20,
    seed: int = None
) -> List[Dict]:
    """
    从导弹池中随机采样

    Args:
        missile_pool: 导弹池
        n_samples: 采样数量
        seed: 随机种子

    Returns:
        采样得到的导弹列表
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(missile_pool), size=n_samples, replace=False)
    return [missile_pool[i] for i in indices]
```

### 4.4 main.py改动示例

```python
# 在各方法的main.py中增加以下逻辑

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ============================================================
# 重要：顶层函数（必须定义在模块级别，以便pickle序列化）
# ============================================================

def _evaluate_single_run(args):
    """
    单次评估的顶层函数（ProcessPoolExecutor要求）

    Args:
        args: (eval_idx, missiles, pop_size, max_gen, method_name)

    Returns:
        评估结果字典
    """
    eval_idx, missiles, pop_size, max_gen, method_name = args

    # 根据方法名导入对应的优化函数（延迟导入避免循环依赖）
    if method_name == "M1_NaiveGA":
        from M1_NaiveGA.optimizer.distributed_optimizer import run_optimization
    elif method_name == "M2_Clustering":
        from M2_Clustering.optimizer.distributed_optimizer import run_optimization
    elif method_name == "M3_AmmoPenalty":
        from M3_AmmoPenalty.optimizer.distributed_optimizer import run_optimization_M2 as run_optimization
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # 运行优化
    best_params, best_fitness, info = run_optimization(
        missile_configs=missiles,
        pop_size=pop_size,
        max_generations=max_gen
    )

    return {
        "eval_idx": eval_idx,
        "best_fitness": best_fitness,
        "intercepted": info.get("intercepted_count", 0),
        "smokes_used": info.get("smokes_used", 15),
        **info
    }


def run_multi_evaluation(
    pattern_name: str,
    method_name: str = "M2_Clustering",
    n_evals: int = 10,
    pop_size: int = 30,
    max_gen: int = 50,
    n_workers: int = None
) -> Dict:
    """
    多次并行评估入口（使用ProcessPoolExecutor）

    Args:
        pattern_name: 攻击模式名称
        method_name: 方法名称 (M1_NaiveGA, M2_Clustering, M3_AmmoPenalty)
        n_evals: 评估次数
        pop_size: 种群大小
        max_gen: 最大代数
        n_workers: 并行worker数，默认使用CPU核心数

    Returns:
        汇总结果字典
    """
    from shared.data.missile_pool_generator import (
        load_missile_pool, sample_from_pool
    )

    # 确定并行worker数
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), n_evals)

    # 加载导弹池
    pool_path = f"data/missile_pools/{pattern_name}_pool.json"
    missile_pool = load_missile_pool(pool_path)

    print(f"\n{'='*70}")
    print(f"  Multi-Evaluation: {pattern_name} | {method_name}")
    print(f"  Evaluations: {n_evals} | Workers: {n_workers}")
    print(f"{'='*70}")

    # 预先生成所有采样的导弹配置（避免在子进程中重复加载池）
    eval_tasks = []
    for eval_idx in range(n_evals):
        seed = eval_idx  # 固定种子确保可复现
        missiles = sample_from_pool(missile_pool, n_samples=20, seed=seed)
        eval_tasks.append((eval_idx, missiles, pop_size, max_gen, method_name))

    # 并行执行评估
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_evaluate_single_run, task): task[0]
                   for task in eval_tasks}

        for future in as_completed(futures):
            eval_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  [Eval {eval_idx + 1}/{n_evals}] "
                      f"Intercepted: {result['intercepted']}/20, "
                      f"Smokes: {result['smokes_used']}, "
                      f"Fitness: {result['best_fitness']:.2f}")
            except Exception as e:
                print(f"  [Eval {eval_idx + 1}] FAILED: {e}")
                results.append({
                    "eval_idx": eval_idx,
                    "error": str(e),
                    "intercepted": 0,
                    "smokes_used": 15,
                    "best_fitness": 0.0
                })

    # 按eval_idx排序
    results.sort(key=lambda x: x["eval_idx"])

    # 统计汇总
    intercepted_list = [r["intercepted"] for r in results if "error" not in r]
    smokes_list = [r["smokes_used"] for r in results if "error" not in r]

    summary = {
        "pattern": pattern_name,
        "method": method_name,
        "n_evals": n_evals,
        "n_workers": n_workers,
        "intercepted_mean": np.mean(intercepted_list) if intercepted_list else 0,
        "intercepted_std": np.std(intercepted_list) if intercepted_list else 0,
        "intercepted_max": np.max(intercepted_list) if intercepted_list else 0,
        "intercepted_min": np.min(intercepted_list) if intercepted_list else 0,
        "smokes_mean": np.mean(smokes_list) if smokes_list else 15,
        "smokes_std": np.std(smokes_list) if smokes_list else 0,
        "efficiency_mean": np.mean(intercepted_list) / np.mean(smokes_list) if intercepted_list and smokes_list else 0,
        "results": results
    }

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {pattern_name} | {method_name}")
    print(f"  Intercepted: {summary['intercepted_mean']:.2f} +/- {summary['intercepted_std']:.2f}")
    print(f"  Smokes Used: {summary['smokes_mean']:.2f} +/- {summary['smokes_std']:.2f}")
    print(f"  Efficiency:  {summary['efficiency_mean']:.3f}")
    print(f"{'='*70}\n")

    return summary
```

---

## 五、时间成本估算

### 5.1 开发时间估算

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| Phase 1 | 创建attack_patterns.py | 0.5小时 |
| Phase 2 | 创建missile_pool_generator.py | 1小时 |
| Phase 3 | 创建generate_all_scenarios.py | 0.5小时 |
| Phase 4 | 修改threat_generator.py | 0.5小时 |
| Phase 5 | 修改M1/M2/M3的main.py | 1.5小时 |
| Phase 6 | 创建summarize_results.py | 1小时 |
| Phase 7 | 测试验证 | 0.5小时 |
| **总计** | - | **5.5小时** |

### 5.2 实验运行时间估算

假设单次实验（Pop=30, Gen=50）耗时约20分钟：

| 并行度 | 120次实验总耗时 |
|--------|----------------|
| 串行（1核） | 40小时 |
| 4核并行 | 10小时 |
| 10核并行 | 4小时 |
| 20核并行 | **2小时** |

### 5.3 总时间估算

| 阶段 | 时间 |
|------|------|
| 代码开发 | 5.5小时 |
| 数据集生成 | 0.5小时 |
| 实验运行（20核） | 2小时 |
| 结果分析 | 1小时 |
| **总计** | **9小时** |

---

## 六、预期实验结果

### 6.1 拦截率预期

| 攻击模式 | M1_NaiveGA | M2_Clustering | M3_AmmoPenalty |
|---------|------------|---------------|----------------|
| Mode-1 单向窄扇区 | 30-50% | 80-95% | 80-95% |
| Mode-2 正交钳形 | 20-40% | 70-90% | 70-90% |
| Mode-3 非对称多向 | 25-45% | 75-90% | 75-90% |
| Mode-4 全向蜂群 | 15-30% | 60-80% | 60-80% |

### 6.2 弹药消耗预期

| 攻击模式 | M1_NaiveGA | M2_Clustering | M3_AmmoPenalty |
|---------|------------|---------------|----------------|
| Mode-1 | 15颗 | 12-15颗 | **6-10颗** |
| Mode-2 | 15颗 | 12-15颗 | **7-11颗** |
| Mode-3 | 15颗 | 12-15颗 | **7-11颗** |
| Mode-4 | 15颗 | 13-15颗 | **8-12颗** |

### 6.3 效费比预期

| 攻击模式 | M1 | M2 | M3 |
|---------|-----|-----|-----|
| Mode-1 | ~0.4 | ~1.3 | **~2.0** |
| Mode-2 | ~0.3 | ~1.2 | **~1.8** |
| Mode-3 | ~0.35 | ~1.25 | **~1.9** |
| Mode-4 | ~0.25 | ~1.0 | **~1.5** |

---

## 七、风险评估与应对

### 7.1 潜在风险

| 风险编号 | 风险描述 | 可能性 | 影响 | 应对措施 |
|---------|---------|--------|------|---------|
| R-001 | 全向蜂群模式下M2/M3拦截率过低 | 中 | 高 | 调整聚类参数或增加UAV数量 |
| R-002 | 时间同步约束导致部分导弹位置不合理 | 低 | 中 | 增加位置验证逻辑 |
| R-003 | 120次实验运行时间超预期 | 中 | 中 | 增加并行度或减少评估次数 |
| R-004 | 采样分布不均匀导致统计偏差 | 低 | 中 | 增加评估次数到15次 |

### 7.2 回退方案

如果新方案出现问题，可以回退到原有方案：
- 保留原有的Wave_3_20_Missiles.json
- 代码改动都在新文件中，不影响现有功能

---

## 八、审核检查清单

### 8.1 方案设计审核

- [ ] 四种攻击模式的定义是否合理？
- [ ] 参数范围（方位角、距离、高度）是否合理？
- [ ] 时间同步窗口（5秒）是否合理？
- [ ] 导弹池大小（100枚）是否足够？
- [ ] 评估次数（10次）是否足够？

### 8.2 代码改动审核

- [ ] 新建文件列表是否完整？
- [ ] 核心模块是否确实无需改动？
- [ ] 代码改动量估算是否准确？

### 8.3 时间成本审核

- [ ] 开发时间估算是否合理？
- [ ] 实验运行时间估算是否合理？
- [ ] 总时间是否可接受？

### 8.4 预期结果审核

- [ ] 拦截率预期是否合理？
- [ ] 弹药消耗预期是否合理？
- [ ] 效费比预期是否合理？

---

## 九、审核意见区

### 9.1 审核人意见

> （待填写）

### 9.2 修改记录

| 日期 | 修改内容 | 修改人 |
|------|---------|--------|
| 2026-03-08 | 初版创建 | Claude Code |
| 2026-03-08 | 修正4.3.2节：arrival_time改用高斯分布确保钟形曲线 | Claude Code |
| 2026-03-08 | 修正4.4节：main.py改用ProcessPoolExecutor并行执行 | Claude Code |

### 9.3 审核状态

- [ ] 待审核
- [ ] 审核通过
- [ ] 需要修改

---

## 十、参考文献

1. IEEE T-AES 2021: Three-dimensional impact angle constrained distributed cooperative guidance law for anti-ship missiles
2. Aerospace 2022: Design and Optimization of Multimissile Formation Based on the Adaptive SA-PSO Algorithm
3. IEEE Xplore 2023: Weapon-Target Assignment Strategy in Joint Combat Decision-Making based on Multi-head Deep Reinforcement Learning
4. MDPI Applied Sciences 2021: Studies on Multi-Constraints Cooperative Guidance Method Based on Distributed MPC for Multi-Missiles
5. IEEE T-AES 2024: Distributed Cooperative Strategy of UAV Swarm Without Speed Measurement Under Saturation Attack Mission
6. Wikipedia: Saturation Attack - https://en.wikipedia.org/wiki/Saturation_attack
7. CSIS: Russian Firepower Strike Tracker - https://www.csis.org/programs/futures-lab/projects/russian-firepower-strike-tracker-analyzing-missile-attacks-ukraine

---

*文档版本: v1.0*
*创建时间: 2026-03-08*
*最后更新: 2026-03-08*
