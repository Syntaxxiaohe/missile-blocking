"""
一键生成所有攻击场景数据集

运行方法（从终极实验根目录）：
    python generate_all_scenarios.py

输出目录：
    data/missile_pools/
        mode_1_single_sector_pool.json      (100 枚)
        mode_2_orthogonal_pincer_pool.json  (100 枚)
        mode_3_asymmetric_saturation_pool.json (100 枚)
        mode_4_full_360_swarm_pool.json     (100 枚)

随机种子设计（确保可复现性）：
    Mode-1: 20260301
    Mode-2: 20260302
    Mode-3: 20260303
    Mode-4: 20260304
"""

import os
import sys
import numpy as np

# ===== ROOT_DIR 注入 =====
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.data.attack_patterns import ATTACK_PATTERNS, list_all_patterns
from shared.data.missile_pool_generator import (
    generate_missile_pool,
    save_missile_pool,
)

# ============================================================
# 配置
# ============================================================
POOL_SIZE   = 100    # 每种模式的导弹池大小
OUTPUT_DIR  = os.path.join(ROOT_DIR, "data", "missile_pools")

# 固定随机种子（格式：年月日 + 模式序号）
SEEDS = {
    "mode_1_single_sector":          20260301,
    "mode_2_orthogonal_pincer":      20260302,
    "mode_3_asymmetric_saturation":  20260303,
    "mode_4_full_360_swarm":         20260304,
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  生成四种攻击模式导弹池")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  每池导弹数: {POOL_SIZE}")
    print("=" * 70)

    all_patterns = list_all_patterns()

    for pattern_name in all_patterns:
        seed = SEEDS[pattern_name]
        config = ATTACK_PATTERNS[pattern_name]
        total_per_scenario = sum(s["missile_count"] for s in config["sectors"])

        print(f"\n[{pattern_name}]")
        print(f"  名称   : {config['name_en']}")
        print(f"  每波次 : {total_per_scenario} 枚导弹")
        print(f"  池大小 : {POOL_SIZE} 枚（每次从中随机采样 {total_per_scenario} 枚）")
        print(f"  随机种子: {seed}")

        # 生成导弹池
        pool = generate_missile_pool(pattern_name, pool_size=POOL_SIZE, seed=seed)

        # 统计 ToA 分布
        arrival_times = [m["arrival_time"] for m in pool]
        print(
            f"  ToA 统计: mean={np.mean(arrival_times):.2f}s"
            f"  std={np.std(arrival_times):.3f}s"
            f"  范围=[{min(arrival_times):.2f}, {max(arrival_times):.2f}]s"
        )

        # 验证：采样一次，检查导弹格式
        from shared.data.missile_pool_generator import sample_from_pool
        sample = sample_from_pool(pool, n_samples=total_per_scenario, seed=0)
        assert all("id" in m and "position" in m and "velocity" in m for m in sample), \
            "导弹格式校验失败：缺少必要字段"

        # 保存
        filepath = os.path.join(OUTPUT_DIR, f"{pattern_name}_pool.json")
        save_missile_pool(pool, filepath)

    print("\n" + "=" * 70)
    print("  全部导弹池生成完成！")
    print(f"  文件位置: {OUTPUT_DIR}")
    print("=" * 70)
    print()
    print("下一步：")
    print("  cd M1_NaiveGA && python main.py --mode multi_eval --pattern mode_1_single_sector")
    print("  cd M2_Clustering && python main.py --mode multi_eval --pattern mode_1_single_sector")
    print("  cd M3_AmmoPenalty && python main.py --mode multi_eval --pattern mode_1_single_sector")


if __name__ == "__main__":
    main()
