# 终极实验架构设计文档

> 最后更新: 2026-03-05
> 状态: 🔒 参数已锁定，准备执行

---

## 一、实验核心目标

**唯一目标**: 在 Wave 3 (20枚导弹) 场景下，证明"分配策略决定信息利用效率"

---

## 二、控制变量（绝对不变）

### 2.1 底层物理引擎

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|------|
| 烟幕扩散 | `physics/smoke.py` | 高斯烟团 + 风场漂移 + 预计算缓存 | 🔒 冻结 |
| 遮蔽判定 | `physics/blocking.py` | Beer-Lambert + erf解析解 + 向量化 | 🔒 冻结 |
| 导弹制导 | `physics/missile_png.py` | PNG三阶段 + FOV脱锁 + RK4积分 | 🔒 冻结 |
| UAV运动 | `physics/uav.py` | 匀速直线运动模型 | 🔒 冻结 |

### 2.2 场景参数（绝对不变）

```python
# 目标参数
REAL_TARGET = {
    "center": np.array([0.0, 0.0, 0.0]),
    "radius": 7.0,      # m
    "height": 10.0      # m
}

# 风场参数
WIND_PARAMS = {
    "vector": np.array([5.0, -3.0, 0.0]),  # m/s, 向东南吹
    "enabled": True
}

# 导弹参数
MISSILE_SPEED = 300.0   # m/s
PNG_PARAMS = {
    "N": 4.0,            # 导航比
    "n_max": 25.0,       # 最大过载 (g)
    "FOV_max_rad": np.radians(30.0),  # FOV角度
    "dt": 0.01           # 仿真步长
}

# 烟幕参数
GAUSSIAN_SMOKE_PARAMS = {
    "Q": 300000.0,         # 烟团强度
    "sigma_0": 25.0,     # 初始标准差 (m)
    "diffusion_rate": 8.0,  # 扩散速率 (m/s)
    "alpha": 0.3,        # 消光系数 (m^-1)
    "transmittance_threshold": 0.10,  # 透射率阈值
    "max_effective_time": 60.0  # 最大有效时间 (s)
}

# 优化参数
SMOKE_PAYLOAD_PER_UAV = 3   # 每架UAV载弹量
SAFE_DISTANCE = 15.0        # 安全距离阈值 (m)
SUCCESS_SCORE_WEIGHT = 10000  # 成功拦截得分
```

### 2.3 UAV初始位置（绝对固定）

```
CAP_RADIUS = 3000 m  # 巡逻圆环半径
CAP_HEIGHT = 800 m   # 巡逻高度

UAV1: (3000, 0, 800)      @ 0°
UAV2: (927, 2853, 800)    @ 72°
UAV3: (-2427, 1763, 800)  @ 144°
UAV4: (-2427, -1763, 800) @ 216°
UAV5: (927, -2853, 800)   @ 288°
```

### 2.4 算力参数（绝对公平）

```python
# 🔒 锁定！三个方法完全一致
POP_SIZE = 30        # 种群大小
MAX_GENERATIONS = 50 # 最大代数
N_THREADS = CPU核心数  # 并行数
```

---

## 三、实验变量（唯一差异）

### 3.1 核心差异矩阵

| 方法 | 分配策略 | 基因维度 | 适应度函数 | 预期效果 |
|------|---------|---------|-----------|---------|
| **M1** | 均匀分配 | 50维 | 单目标 | 先验撕裂 → 防线崩溃 |
| **M2** | 时空聚类 | 50维 | 单目标 | 先验协同 → 防线稳固 |
| **M3** | 时空聚类 | 65维 | 双目标+弹药惩罚 | 先验协同+效费比优化 |

### 3.2 专家先验（所有方法共享）

```python
# 所有方法都使用相同的专家先验接口
expert_genes[uav_id] = {
    "direction": (dx, dy, dz),      # 飞向拦截点的方向
    "time_window": (t_min, t_max),  # 根据导弹到达时间计算的窗口
}
```

**关键论点**:
- M1: 先验存在但**撕裂**（方向矛盾、时间窗过宽）
- M2: 先验存在且**协同**（方向一致、时间窗精确）
- M3: 先验协同 + 弹药节约

---

## 四、实验数据集

### 4.1 唯一测试场景

| Wave | 导弹数 | 扇区分布 | 文件路径 | 用途 |
|------|--------|---------|---------|------|
| **Wave 3** | **20** | A:5, B:7, C:4, D:4 | `data/Wave_3_20_Missiles.json` | **唯一目标** |

### 4.2 扇区定义

| 扇区 | 名称 | 方位角 | 距离 | 高度 | 特点 |
|------|------|------------|----------|----------|------|
| A | 西北迎风/高空 | 120-150 | 15-20 km | 2000-4000 m | 烟幕被风吹向目标 |
| B | 东南背风/超低空 | -60~-30 | 10-15 km | 50-300 m | **极度危险** |
| C | 东北侧风/中空 | 30-60 | 12-18 km | 500-1500 m | 侧向攻击 |
| D | 西南侧重/中空 | 210-240 | 12-18 km | 500-1500 m | 侧向攻击 |

---

## 五、M1-M3 详细设计

### 5.1 M1_NaiveGA（均匀分配基线）

#### 分配策略

```python
class NaiveAllocator:
    """
    M1 均匀分配器

    策略：轮询分配，不考虑空间距离
    例如：5架UAV，20枚导弹
    UAV1: [M1, M6, M11, M16]  ← 可能在四个不同方向！
    UAV2: [M2, M7, M12, M17]
    UAV3: [M3, M8, M13, M18]
    UAV4: [M4, M9, M14, M19]
    UAV5: [M5, M10, M15, M20]
    """

    def naive_assignment(self):
        """轮询分配导弹给UAV"""
        for i, missile in enumerate(self.missile_configs):
            uav_id = self.uav_group[i % self.n_uavs]
            # 添加到该UAV的"伪Cluster"
```

#### 伪Cluster数据结构（关键！）

```python
@dataclass
class PseudoCluster:
    """
    伪Cluster - 与真实MissileCluster 100%接口对齐！

    字段完全一致，但内部导弹分布极其分散
    """
    cluster_id: int
    missile_ids: List[int]
    missile_configs: List[Dict]
    centroid: np.ndarray           # 计算得到（可能在"无人区"）
    earliest_arrival: float        # 时间窗起点
    latest_arrival: float          # 时间窗终点（可能很宽！）
    threat_priority: float

    @property
    def n_missiles(self) -> int:
        return len(self.missile_ids)
```

#### 先验撕裂效果

```
假设 UAV1 被分配 [M1, M6, M11, M16]：
  M1  @ (15000, 5000, 2000)   到达 35s  ← 东北
  M6  @ (-14000, 6000, 1800)  到达 42s  ← 西北
  M11 @ (2000, 16000, 1500)   到达 38s  ← 正北
  M16 @ (10000, -10000, 500)  到达 50s  ← 东南

伪Cluster属性：
  质心 = (3500, 4250, 1450)  ← 指向"无人区"！
  时间窗 = [35s, 50s]  ← 宽度15秒！

专家先验：
  方向 = 指向质心（东北偏北）
  时间窗 = [14s, 42s]  ← 宽度28秒！

结果：先验撕裂，无法有效利用！
```

#### 基因编码

```python
# M1: 10维/UAV × 5 UAV = 50维
gene_M1 = [dir_x, dir_y, dir_z, speed,    # 4维
           t_drop1, d_det1,                # 烟幕弹1
           t_drop2, d_det2,                # 烟幕弹2
           t_drop3, d_det3]                # 烟幕弹3
```

---

### 5.2 M2_Clustering（时空聚类优化）

#### 分配策略

```python
class SpatioTemporalAllocator:
    """
    M2 时空聚类分配器

    聚类参数：
    - SPATIAL_THRESHOLD = 2000 m
    - TEMPORAL_THRESHOLD = 5 s
    - MAX_MISSILES_PER_CLUSTER = 3
    """

    def cluster_missiles(self):
        """
        时空聚类算法：
        1. 按到达时间排序（威胁优先）
        2. 贪婪合并：空间<2000m 且 时间<5s
        3. 每簇最多3枚
        """

    def hungarian_assignment(self):
        """
        匈牙利匹配：
        代价 = 距离 + 方位角惩罚 + 高度惩罚 + 风场补偿
        """
```

#### 先验协同效果

```
聚类结果：
  Cluster 1: [M1, M5, M9]
    - 质心: (14500, 4800, 1900)
    - 时间窗: [34s, 37s]  ← 宽度仅3秒！

匈牙利匹配：
  UAV1 @ 0°   → Cluster 1 (东北) ✅ 方位匹配！

UAV1的专家先验：
  方向：指向东北 ✅ 三枚导弹都在这个方向！
  时间窗：[13s, 29s] 宽度16s ✅ 足够精确！

结果：先验高度一致，优势最大化！
```

#### 基因编码

```python
# M2: 10维/UAV × 5 UAV = 50维（与M1相同）
gene_M2 = [dir_x, dir_y, dir_z, speed,
           t_drop1, d_det1,
           t_drop2, d_det2,
           t_drop3, d_det3]
```

---

### 5.3 M3_AmmoPenalty（帕累托弹药优化）

#### 分配策略

```python
# M3 继承 M2 的时空聚类分配器
class SpatioTemporalAllocator_M3(SpatioTemporalAllocator):
    """完全继承M2的分配策略"""
    pass
```

#### 基因编码（新增弹药开关）

```python
# M3: 13维/UAV × 5 UAV = 65维
gene_M3 = [dir_x, dir_y, dir_z, speed,     # 4维
           t_drop1, d_det1, switch1,        # 烟幕弹1 + 开关
           t_drop2, d_det2, switch2,        # 烟幕弹2 + 开关
           t_drop3, d_det3, switch3]        # 烟幕弹3 + 开关

# 开关判定
SWITCH_THRESHOLD = 0.5   # > 0.5 则发射
AMMO_PENALTY_WEIGHT = 1000  # 每颗弹的惩罚
```

#### 双目标适应度函数

```python
def multi_missile_objective_M3(params, uav_group, missile_configs):
    """
    M3 双目标适应度

    Fitness = (拦截数 × 10000) - (耗弹数 × 1000) + 失败脱靶补偿
    """
    # 解析开关，统计实际发射的烟雾弹
    used_smokes = 0
    for k in range(SMOKE_PAYLOAD_PER_UAV):
        switch = params[base_idx + 2]
        if switch > SWITCH_THRESHOLD:
            used_smokes += 1

    # 双目标适应度
    fitness = success_count * 10000
    fitness -= used_smokes * 1000  # 弹药惩罚！

    return fitness, {"used_smokes": used_smokes}
```

#### 学习行为预期

```
M3 在 Wave 3 (20枚导弹) 的预期学习结果：

UAV  | 分配导弹 | Switches预期        | 耗弹 | 拦截结果
-----|---------|--------------------|------|----------
UAV1 | 3枚     | [0.9, 0.9, 0.9]    | 3    | 3/3
UAV2 | 3枚     | [0.9, 0.4, 0.2]    | 1    | 2/3 (主动关闭)
UAV3 | 3枚     | [0.9, 0.9, 0.9]    | 3    | 3/3
UAV4 | 2枚     | [0.9, 0.9, 0.3]    | 2    | 2/2
UAV5 | 1枚     | [0.9, 0.1, 0.1]    | 1    | 1/1

总计 | 12枚    |                    | 10   | 19/20 (95%)
效费比: 1.9 (vs M2的 1.27)
```

---

## 六、文件夹结构

```
D:\Work\guosai\终极实验\
│
├── 终极架构.md                      # 本文档
│
├── data/                            # 共享数据
│   └── Wave_3_20_Missiles.json      # 唯一测试集
│
├── M1_NaiveGA/                      # M1: 均匀分配基线
│   ├── main.py                      # 主入口
│   ├── config/
│   │   └── params.py                # 参数配置
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── uav.py
│   │   ├── smoke.py
│   │   ├── blocking.py
│   │   └── missile_png.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── naive_allocator.py       # 均匀分配器
│   │   └── distributed_optimizer.py # 分布式优化器
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── single.py
│   └── results/                     # 实验结果
│       └── M1_Wave3_Results.json
│
├── M2_Clustering/                   # M2: 时空聚类优化
│   ├── main.py
│   ├── config/
│   │   └── params.py
│   ├── physics/
│   │   └── ... (同M1)
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── allocator.py             # 时空聚类分配器
│   │   └── distributed_optimizer.py # 分布式优化器
│   ├── simulation/
│   │   └── ... (同M1)
│   └── results/
│       └── M2_Wave3_Results.json
│
└── M3_AmmoPenalty/                  # M3: 帕累托弹药优化
    ├── main.py
    ├── config/
    │   └── params.py
    ├── physics/
    │   └── ... (同M1)
    ├── optimizer/
    │   ├── __init__.py
    │   ├── allocator.py             # 继承M2
    │   └── distributed_optimizer.py # M3版本（含弹药惩罚）
    ├── simulation/
    │   └── ... (同M1)
    └── results/
        └── M3_Wave3_Results.json
```

---

## 七、执行计划

### 7.1 Phase 1: 环境搭建

| 步骤 | 任务 | 状态 |
|------|------|------|
| 1.1 | 创建文件夹结构 | ⏳ 待执行 |
| 1.2 | 复制共享数据 | ⏳ 待执行 |
| 1.3 | 创建M2_Clustering（从baseline3迁移） | ⏳ 待执行 |
| 1.4 | 创建M3_AmmoPenalty（从M2文件夹迁移） | ⏳ 待执行 |
| 1.5 | 创建M1_NaiveGA（复制物理引擎+新开发分配器） | ⏳ 待执行 |

### 7.2 Phase 2: 代码开发

| 步骤 | 任务 | 状态 |
|------|------|------|
| 2.1 | 开发 NaiveAllocator | ⏳ 待执行 |
| 2.2 | 开发 PseudoCluster（100%接口对齐） | ⏳ 待执行 |
| 2.3 | 修改M3的distributed_optimizer | ⏳ 待执行 |
| 2.4 | 统一三个方法的main.py入口 | ⏳ 待执行 |

### 7.3 Phase 3: 实验执行

| 步骤 | 任务 | 参数 | 状态 |
|------|------|------|------|
| 3.1 | 运行M1 | Pop=30, Gen=50, Wave3 | ⏳ 待执行 |
| 3.2 | 运行M2 | Pop=30, Gen=50, Wave3 | ⏳ 待执行 |
| 3.3 | 运行M3 | Pop=30, Gen=50, Wave3 | ⏳ 待执行 |

### 7.4 Phase 4: 结果分析

| 步骤 | 任务 | 状态 |
|------|------|------|
| 4.1 | 汇总三个方法的结果 | ⏳ 待执行 |
| 4.2 | 对比分析 | ⏳ 待执行 |
| 4.3 | 生成论文图表 | ⏳ 待执行 |

---

## 八、预期结果

### 8.1 拦截率对比

| 方法 | Wave 3 (20弹) | 说明 |
|------|--------------|------|
| M1 | 25-40% | 先验撕裂，防线崩溃 |
| M2 | 90-95% | 先验协同，防线稳固 |
| M3 | 90-95% | 先验协同+弹药节约 |

### 8.2 弹药消耗对比

| 方法 | 弹药消耗 | 效费比 |
|------|---------|--------|
| M1 | 15颗 | ~0.4 |
| M2 | 15颗 | ~1.27 |
| M3 | **8-10颗** | **~2.0** |

### 8.3 核心论点

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   "Information is Power, but Distribution is Key"                   │
│                                                                     │
│   M1, M2, M3 都拥有相同的战场态势感知（专家先验），                  │
│   但分配策略的差异导致信息利用效率天壤之别：                         │
│                                                                     │
│   M1: 均匀分配 → 先验撕裂 → 信息冲突 → 防线崩溃                     │
│   M2: 时空聚类 → 先验协同 → 信息一致 → 防线稳固                     │
│   M3: 时空聚类 + 弹药惩罚 → 信息一致 + 效费比优化 → 最优防线        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 九、风险控制

### 9.1 M1伪Cluster接口风险

**风险**: PseudoCluster与真实MissileCluster接口不对齐，导致运行时报错

**对策**:
1. 严格复制MissileCluster的所有字段
2. 添加单元测试验证接口一致性
3. 先运行单个UAV测试

### 9.2 运行时间风险

**风险**: Pop=30, Gen=50 可能运行时间较长

**对策**:
1. 使用ProcessPoolExecutor并行
2. 添加完美熔断机制
3. 保存中间结果

---

## 十、变更日志

| 日期 | 变更内容 |
|------|---------|
| 2026-03-05 | 根据Gemini纠正意见创建文档 |
| 2026-03-05 | 锁定实验规模为Wave 3 (20弹) |
| 2026-03-05 | 锁定算力参数 Pop=30, Gen=50 |
| 2026-03-05 | 添加PseudoCluster接口对齐要求 |

---

**文档状态**: 🔒 参数已锁定，准备执行
