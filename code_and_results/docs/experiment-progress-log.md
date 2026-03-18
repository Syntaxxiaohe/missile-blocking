# 实验数据与进度记录

> **项目**: UAV 蜂群烟幕干扰防空优化系统（M1 / M2 / M3 三方法对比实验）
> **论文目标期刊**: *Mathematics* (MDPI, SCI Q1)
> **最后更新**: **2026-03-09（今日）**
> **维护说明**: 本文档是 AI 工程师的"工作记忆"，每日更新，次日阅读本文件即可无缝续接

---

## ★ 2026-03-09 今日工作全记录（重要！次日必读）

### 【今日任务背景】
用户要求进行一次"重大架构升级"，核心动机是：
- 原方案仅用一个固定场景（Wave_3_20_Missiles.json）做验证，审稿人会质疑"过拟合"
- 原来的四扇区（A/B/C/D）攻击分布过于极端，缺乏文献支撑
- 需要引入 **4 种基于顶刊文献的攻击模式 + 100枚导弹池随机采样机制**

---

### 【今日新建文件清单】

#### 1. `shared/data/attack_patterns.py`（全新，约194行）

**作用**: 定义 4 种经典导弹饱和攻击模式。

**4 种模式定义**（关键参数）：

| 模式键名 | 中文名 | 扇区数 | 总导弹数 | 空间约束 |
|---------|--------|--------|---------|---------|
| `mode_1_single_sector` | 单向窄扇区饱和打击 | 1 | 20 | 方位角[30°,60°]，距离[12,18]km，高度[800,2000]m |
| `mode_2_orthogonal_pincer` | 正交钳形打击 | 2 | 10+10=20 | 北向[80°,100°]+东向[-10°,10°] |
| `mode_3_asymmetric_saturation` | 非对称多向饱和攻击 | 3 | 14+3+3=20 | 北向70%+东南牵制15%+西南牵制15% |
| `mode_4_full_360_swarm` | 全向蜂群突防 | 1 | 20 | 360°均匀分布，奇偶导弹高低交替 |

**全局时间同步参数（TEMPORAL_SYNC）**：
```python
TEMPORAL_SYNC = {
    "base_arrival_time": 45.0,   # 基准到达时间 (s)
    "sync_window":        5.0,   # 同步窗口宽度 (s)
}
# 所有导弹 ToA ~ N(45.0, (5/6)²)，3σ覆盖[42.5s, 47.5s]
```

**学术依据（论文必引）**：
- IEEE T-AES 2021: 三维撞击角约束制导（Mode-1依据）
- Aerospace 2022: 多导弹编队优化（Mode-2依据）
- IEEE Xplore 2023: 多头深度强化学习WTA（Mode-3依据）
- MDPI Applied Sciences 2021 + IEEE T-AES 2024: 分布式MPC协同制导（Mode-4依据）

---

#### 2. `shared/data/missile_pool_generator.py`（全新，约230行）

**作用**: 为每种攻击模式生成 100 枚候选导弹的"池"，每次实验从池中随机采样 20 枚。

**核心设计决策（重要！）**：
- **ToA 必须用正态分布**：`arrival_time = rng.normal(loc=45.0, scale=5/6)`，确保饱和攻击"同时弹着"特征，99.7% 样本落在 [42.5s, 47.5s] 内。**绝对不能用均匀分布。**
- 输出格式与 `shared/simulation/single.py` 完全兼容，必须含字段：`id`, `sector`, `position`, `velocity`, `distance`, `height`, `azimuth_deg`, `arrival_time`
- Mode-4 全向蜂群：方位角均匀铺开（每隔 360°/100 = 3.6°），加 ±5° 随机扰动，**但 ToA 依然用正态分布**

**主要函数**：
```python
generate_missile_pool(pattern_name, pool_size=100, seed=int) -> List[Dict]
sample_from_pool(missile_pool, n_samples=20, seed=int) -> List[Dict]  # 无重复采样
save_missile_pool(pool, filepath)
load_missile_pool(filepath) -> List[Dict]
```

---

#### 3. `generate_all_scenarios.py`（全新，约70行，放在根目录）

**作用**: 一键生成 4 种模式的导弹池 JSON 文件。

**运行命令**（已执行，文件已生成）：
```bash
cd "d:\Work\guosai\终极实验"
python generate_all_scenarios.py
```

**输出文件**（已存在于 `data/missile_pools/`）：

| 文件名 | 随机种子 | 实际ToA均值 | 实际ToA std |
|--------|---------|------------|------------|
| `mode_1_single_sector_pool.json` | 20260301 | ~44.95s | ~0.873s |
| `mode_2_orthogonal_pincer_pool.json` | 20260302 | ~44.99s | ~0.827s |
| `mode_3_asymmetric_saturation_pool.json` | 20260303 | ~45.01s | ~0.772s |
| `mode_4_full_360_swarm_pool.json` | 20260304 | ~45.10s | ~0.788s |

---

### 【今日修改文件清单】

#### 4. `M1_NaiveGA/main.py`（重大修改）

**新增内容**（在文件末尾、`main()` 之前插入）：

**① 顶层评估函数** `_evaluate_single_run_m1(args)`
- args = `(eval_idx, missiles, pop_size, max_gen)`
- 内部实例化 `TwoStageOptimizer`，`n_threads=1`（防止嵌套多进程）
- 返回：`{eval_idx, best_fitness, intercepted, smokes_used(固定15), n_missiles}`
- **Windows spawn 模式要求**：函数内部重新 `sys.path.insert`，保证子进程可导入本地模块

**② 多轮并行评估函数** `run_multi_eval_m1(pattern_name, n_evals, pop_size, max_gen, n_workers)`
- 加载对应导弹池 JSON
- 预生成 n_evals 次采样（seed=eval_idx，保证可复现）
- 用 `ProcessPoolExecutor` 并行执行
- 统计：intercepted_mean/std/max/min，smokes_mean/std，efficiency_mean
- 结果自动保存至 `M1_NaiveGA/results/M1_{pattern}_{timestamp}.json`

**③ 新增 CLI 参数**：
```bash
python main.py --mode multi_eval \
               --pattern mode_1_single_sector \
               --n_evals 10 \
               --n_workers 10 \
               --pop_size 30 \
               --max_gen 50
```

**已有 `--mode test` 和 `--mode wave3` 完全保留，无破坏性改动。**

---

#### 5. `M2_Clustering/main.py`（重大修改，结构与M1对称）

- 新增 `_evaluate_single_run_m2(args)` — 调用 `run_distributed_optimization()`
- 新增 `run_multi_eval_m2(...)` — 与 M1 结构完全一致
- 返回值额外含 `n_clusters` 字段（M2特有：聚类数）
- 结果保存至 `M2_Clustering/results/M2_{pattern}_{timestamp}.json`

---

#### 6. `M3_AmmoPenalty/main.py`（重大修改，结构与M1对称）

- 新增 `_evaluate_single_run_m3(args)` — 调用 `run_distributed_optimization_M2()`
- 新增 `run_multi_eval_m3(...)` — **M3 专属**：smokes_used 来自 info["used_smokes"]（动态，非固定15）
- 返回值额外含 `ammo_efficiency` 字段（M3特有：效费比）
- 结果保存至 `M3_AmmoPenalty/results/M3_{pattern}_{timestamp}.json`

---

#### 7. 三个 GA 优化器早停改造（今日第二轮任务）

**修改文件**：
- `M1_NaiveGA/optimizer/distributed_optimizer.py`
- `M2_Clustering/optimizer/distributed_optimizer.py`
- `M3_AmmoPenalty/optimizer/distributed_optimizer.py`

**修改位置**：每个文件的 `SubTaskOptimizer.optimize()` 方法（子任务GA主循环）

**早停逻辑（植入内容，三个文件完全一致）**：
```python
# 在迭代循环开始前添加：
_PATIENCE  = 15       # 连续多少代无进展则放弃
_MIN_DELTA = 1e-4     # 最小有效进步量
_stagnation = 0
_prev_best  = self.best_fitness

# 在每代更新最优后添加：
if self.best_fitness - _prev_best < _MIN_DELTA and self.best_success_count == 0:
    _stagnation += 1
else:
    _stagnation = 0
    _prev_best = self.best_fitness
if _stagnation >= _PATIENCE:
    print(f"  [早停] 第{gen+1}代: 连续{_PATIENCE}代无进展且拦截=0，终止优化")
    break
```

**触发条件（必须同时满足）**：
1. 连续 15 代 `best_fitness` 提升 < 1e-4
2. `best_success_count == 0`（完全无拦截）

**注意**：M3 的 `best_fitness` 可能为负值（弹药惩罚），但只要拦截数=0，早停依然触发。

**打印频率同步降频**：

| 位置 | 修改前 | 修改后 |
|------|--------|--------|
| 子任务 GA 主循环 | 每 **5** 代打印 | 每 **15** 代打印 |
| Phase 2 全局微调循环 | 每 **2** 代打印 | 每 **5** 代打印 |

---

### 【今日未执行的任务（明日继续）】

**待执行实验**（按此顺序）：

```bash
# Step 1: M1 全部4种模式（预期快速完成，因早停机制大幅缩短时间）
cd "d:\Work\guosai\终极实验\M1_NaiveGA"
python main.py --mode multi_eval --pattern mode_1_single_sector --n_evals 10 --n_workers 10 --pop_size 30 --max_gen 50
python main.py --mode multi_eval --pattern mode_2_orthogonal_pincer --n_evals 10 --n_workers 10 --pop_size 30 --max_gen 50
python main.py --mode multi_eval --pattern mode_3_asymmetric_saturation --n_evals 10 --n_workers 10 --pop_size 30 --max_gen 50
python main.py --mode multi_eval --pattern mode_4_full_360_swarm --n_evals 10 --n_workers 10 --pop_size 30 --max_gen 50

# Step 2: M2（预期拦截率 70-95%）
cd "d:\Work\guosai\终极实验\M2_Clustering"
python main.py --mode multi_eval --pattern mode_1_single_sector --n_evals 10 --n_workers 10 --pop_size 30 --max_gen 50
# ... 其余3种模式同理

# Step 3: M3（预期效费比最优）
cd "d:\Work\guosai\终极实验\M3_AmmoPenalty"
python main.py --mode multi_eval --pattern mode_1_single_sector --n_evals 10 --n_workers 10 --pop_size 30 --max_gen 50
# ... 其余3种模式同理
```

**得到结果后需要做**：
- 将结果填入本文档第二章的数据表格
- 分析 M1 vs M2 vs M3 的显著性差异（Wilcoxon rank-sum 检验）
- 如果 M1 Mode-4 全向蜂群 M2 拦截率 < 60%，考虑调整聚类参数

---

### 【当前工程文件树（今日状态）】

```
d:\Work\guosai\终极实验\
├── 终极架构.md                         # 架构设计文档（物理参数全部锁定于此）
├── 物理引擎技术文档.md                  # 物理公式与代码对照文档
├── 方案改进优化记录表.md               # 今日升级方案的完整设计来源
├── 实验数据与进度记录.md               # 本文档
├── generate_all_scenarios.py           # ★ 今日新建：一键生成导弹池
│
├── data/
│   ├── Wave_3_20_Missiles.json         # 原有固定场景（保留不动）
│   └── missile_pools/                  # ★ 今日生成
│       ├── mode_1_single_sector_pool.json
│       ├── mode_2_orthogonal_pincer_pool.json
│       ├── mode_3_asymmetric_saturation_pool.json
│       └── mode_4_full_360_swarm_pool.json
│
├── shared/
│   ├── config/params.py                # 物理参数（冻结，勿改）
│   ├── data/
│   │   ├── threat_generator.py         # 原有生成器（保留）
│   │   ├── attack_patterns.py          # ★ 今日新建
│   │   └── missile_pool_generator.py   # ★ 今日新建
│   ├── physics/                        # 物理引擎（冻结，勿改）
│   ├── simulation/single.py            # 仿真函数（冻结，勿改）
│   └── optimizer/                      # 共享优化器（保留）
│
├── M1_NaiveGA/
│   ├── main.py                         # ★ 今日修改：+multi_eval模式
│   └── optimizer/
│       └── distributed_optimizer.py   # ★ 今日修改：+早停+降频打印
│
├── M2_Clustering/
│   ├── main.py                         # ★ 今日修改：+multi_eval模式
│   └── optimizer/
│       └── distributed_optimizer.py   # ★ 今日修改：+早停+降频打印
│
└── M3_AmmoPenalty/
    ├── main.py                         # ★ 今日修改：+multi_eval模式
    └── optimizer/
        └── distributed_optimizer.py   # ★ 今日修改：+早停+降频打印
```

---

### 【关键物理参数（绝对不可修改）】

| 参数 | 值 | 位置 |
|------|-----|------|
| PNG 导航比 N | 4.0 | `shared/config/params.py` |
| 导弹速度 | 300 m/s | `shared/config/params.py` |
| 高斯烟团初始标准差 σ₀ | 25 m | `shared/config/params.py` |
| 扩散速率 | 8 m/s | `shared/config/params.py` |
| 消光系数 α | 0.3 m⁻¹ | `shared/config/params.py` |
| 透射率阈值 | 0.10 | `shared/config/params.py` |
| 安全距离阈值 | 15 m | `shared/config/params.py` |
| 风场向量 | [5.0, -3.0, 0.0] m/s | `shared/config/params.py` |
| CAP 巡逻半径 | 3000 m | `shared/config/params.py` |
| CAP 巡逻高度 | 800 m | `shared/config/params.py` |
| POP_SIZE | 30 | main.py 默认值 |
| MAX_GENERATIONS | 50 | main.py 默认值 |

---

### 【已知问题 / 风险提示】

| 编号 | 问题描述 | 严重性 | 状态 |
|------|---------|--------|------|
| W-001 | M1 在 Mode-4 全向蜂群下，均匀分配导致先验极度撕裂，早停会频繁触发，拦截率可能接近0 | 低（这是预期的实验结论） | 已知 |
| W-002 | ProcessPoolExecutor 在 Windows spawn 模式下需要 `if __name__ == "__main__"` 保护，三个 main.py 均已正确配置 | 低 | 已处理 |
| W-003 | 子进程内设置 `n_threads=1` 禁用嵌套多进程，导致单次 eval 内 GA 为串行，速度比单独运行慢，但总体并行度由 n_workers 控制 | 低 | 已知 |
| W-004 | Mode-3 的 flank_a 扇区 [100°,130°] 与 main_assault 扇区 [70°,110°] 有重叠，属于有意设计（牵制弹藏于主攻后方），无需修改 | 低 | 已知 |

---

## 二、实验对比数据表（待填写）

### 2.1 拦截率对比（均值 ± 标准差 / 20枚）

| 攻击模式 | M1_NaiveGA | M2_Clustering | M3_AmmoPenalty |
|---------|-----------|--------------|----------------|
| Mode-1 单向窄扇区 | — | — | — |
| Mode-2 正交钳形 | — | — | — |
| Mode-3 非对称多向 | — | — | — |
| Mode-4 全向蜂群 | — | — | — |

### 2.2 弹药消耗对比（均值 ± 标准差 / 15颗上限）

| 攻击模式 | M1_NaiveGA | M2_Clustering | M3_AmmoPenalty |
|---------|-----------|--------------|----------------|
| Mode-1 单向窄扇区 | 15.0 ± 0.0 | 15.0 ± 0.0 | — |
| Mode-2 正交钳形 | 15.0 ± 0.0 | 15.0 ± 0.0 | — |
| Mode-3 非对称多向 | 15.0 ± 0.0 | 15.0 ± 0.0 | — |
| Mode-4 全向蜂群 | 15.0 ± 0.0 | 15.0 ± 0.0 | — |

### 2.3 效费比对比（拦截数 / 耗弹数，均值 ± 标准差）

| 攻击模式 | M1_NaiveGA | M2_Clustering | M3_AmmoPenalty |
|---------|-----------|--------------|----------------|
| Mode-1 单向窄扇区 | — | — | — |
| Mode-2 正交钳形 | — | — | — |
| Mode-3 非对称多向 | — | — | — |
| Mode-4 全向蜂群 | — | — | — |

---

## 三、实验运行日志

| 日期 | 事件 | 状态 |
|------|------|------|
| 2026-03-09 | 架构升级：新建 attack_patterns.py + missile_pool_generator.py + generate_all_scenarios.py | ✅ |
| 2026-03-09 | 生成4种模式导弹池（各100枚） | ✅ |
| 2026-03-09 | M1/M2/M3 main.py 新增 multi_eval 模式（ProcessPoolExecutor） | ✅ |
| 2026-03-09 | M1/M2/M3 GA 优化器植入早停(patience=15)，打印降频(每15代) | ✅ |
| 2026-03-10 | M1 四种模式 × 10次 评估 | ⏳ 待执行 |
| 2026-03-10 | M2 四种模式 × 10次 评估 | ⏳ 待执行 |
| 2026-03-10 | M3 四种模式 × 10次 评估 | ⏳ 待执行 |
| 2026-03-10 | 填写数据表 + 显著性检验 | ⏳ 待执行 |

---

*本文档由 Claude Code 维护 | 2026-03-09*
