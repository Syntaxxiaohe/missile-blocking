"""
威胁生成器模块 — Baseline3

核心升级：
  1. 四大战术扇区（移除天顶E）
  2. 考虑风场影响的扇区设计
  3. 生成标准化Benchmark测试集

扇区定义（考虑风场 V_wind = [5.0, -3.0, 0.0] m/s）：
  - Sector A (西北迎风/高空): 方位角 120°-150°, 距离 15-20km, 高度 2000-4000m
  - Sector B (东南背风/超低空): 方位角 -60°~-30°, 距离 10-15km, 高度 50-300m
  - Sector C (东北侧风/中空): 方位角 30°-60°, 距离 12-18km, 高度 500-1500m
  - Sector D (西南侧风/中空): 方位角 210°-240°, 距离 12-18km, 高度 500-1500m
"""

import numpy as np
from typing import List, Dict, Tuple
import json
import os

# 从params导入扇区配置（使用相对导入）
from ..config.params import MISSILE_SECTORS, MISSILE_SPEED


class ThreatGenerator:
    """
    随机导弹威胁生成器（四扇区版）
    """

    def __init__(self, seed: int = None):
        """
        Args:
            seed: 随机种子（用于可重复实验）
        """
        self.rng = np.random.default_rng(seed)
        self.missiles = []
        self.seed = seed

    def generate_from_config(self, sector_counts: Dict[str, int]) -> List[Dict]:
        """
        根据扇区配置生成导弹雨

        Args:
            sector_counts: 各扇区导弹数量，如 {"A": 2, "B": 1, "C": 1, "D": 1}

        Returns:
            导弹列表
        """
        self.missiles = []
        missile_id = 1

        for sector_name in ["A", "B", "C", "D"]:  # 固定顺序保证可重复
            count = sector_counts.get(sector_name, 0)
            if count == 0:
                continue

            if sector_name not in MISSILE_SECTORS:
                print(f"警告: 未知扇区 '{sector_name}'，已跳过")
                continue

            sector = MISSILE_SECTORS[sector_name]

            for _ in range(count):
                missile = self._generate_single_missile(missile_id, sector_name, sector)
                self.missiles.append(missile)
                missile_id += 1

        return self.missiles

    def _generate_single_missile(self, missile_id: int, sector_name: str,
                                   sector: Dict) -> Dict:
        """
        在指定扇区内生成单枚导弹

        坐标系说明：
        - x轴指向东
        - y轴指向北
        - 方位角从x轴正向（东）逆时针计算
        """
        # 随机方位角（度）
        angle_deg = self.rng.uniform(sector["angle_range"][0], sector["angle_range"][1])

        # 转换为弧度（数学坐标系：x轴正向为0°，逆时针为正）
        angle_rad = np.radians(angle_deg)

        # 随机距离和高度
        distance = self.rng.uniform(sector["distance_range"][0], sector["distance_range"][1])
        height = self.rng.uniform(sector["height_range"][0], sector["height_range"][1])

        # 计算位置
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        z = height

        # 速度方向指向原点
        target = np.array([0.0, 0.0, 0.0])
        position = np.array([x, y, z])
        direction = target - position
        direction_norm = np.linalg.norm(direction)
        velocity = (direction / direction_norm) * MISSILE_SPEED

        # 计算到达时间（直线距离/速度）
        arrival_time = direction_norm / MISSILE_SPEED

        return {
            "id": missile_id,
            "sector": sector_name,
            "position": [float(x), float(y), float(z)],
            "velocity": [float(v) for v in velocity],
            "distance": float(distance),
            "height": float(height),
            "azimuth_deg": float(angle_deg),
            "arrival_time": float(arrival_time),
        }

    def get_positions_array(self) -> np.ndarray:
        """获取所有导弹位置数组 (N, 3)"""
        if not self.missiles:
            return np.array([])
        return np.array([m["position"] for m in self.missiles])

    def get_arrival_times(self) -> List[float]:
        """获取所有导弹到达时间列表"""
        return [m["arrival_time"] for m in self.missiles]

    def get_global_time_window(self) -> Tuple[float, float]:
        """获取全局拦截时间窗口"""
        if not self.missiles:
            return (0.0, 60.0)
        arrival_times = self.get_arrival_times()
        return (min(arrival_times), max(arrival_times))

    def print_summary(self):
        """打印导弹雨摘要"""
        print("=" * 70)
        print(f"导弹威胁摘要 (共 {len(self.missiles)} 枚)")
        print("=" * 70)
        print(f"{'ID':<4} {'扇区':<6} {'位置 (x, y, z)':<35} {'到达时间(s)':<12}")
        print("-" * 70)

        for m in self.missiles:
            pos = m["position"]
            print(f"{m['id']:<4} {m['sector']:<6} "
                  f"({pos[0]:>8.1f}, {pos[1]:>8.1f}, {pos[2]:>6.1f})    "
                  f"{m['arrival_time']:>8.2f}")

        print("-" * 70)
        t_min, t_max = self.get_global_time_window()
        print(f"到达时间窗: [{t_min:.2f}s, {t_max:.2f}s]")

        # 按扇区统计
        sector_counts = {}
        for m in self.missiles:
            s = m["sector"]
            sector_counts[s] = sector_counts.get(s, 0) + 1
        print(f"扇区分布: {sector_counts}")
        print("=" * 70)

    def save_to_json(self, filepath: str):
        """保存到JSON文件"""
        data = {
            "metadata": {
                "seed": self.seed,
                "total_count": len(self.missiles),
                "time_window": list(self.get_global_time_window()),
                "missile_speed": MISSILE_SPEED,
                "sectors_used": list(set(m["sector"] for m in self.missiles)),
            },
            "missiles": self.missiles,
            "sector_summary": {
                s: sum(1 for m in self.missiles if m["sector"] == s)
                for s in ["A", "B", "C", "D"]
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已保存: {filepath}")

    def load_from_json(self, filepath: str) -> List[Dict]:
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.missiles = data["missiles"]
        return self.missiles


def generate_benchmark_wave(seed: int, sector_counts: Dict[str, int],
                            wave_name: str, output_dir: str):
    """
    生成单个Benchmark测试集

    Args:
        seed: 随机种子
        sector_counts: 扇区配置
        wave_name: 波次名称
        output_dir: 输出目录
    """
    generator = ThreatGenerator(seed=seed)
    generator.generate_from_config(sector_counts)

    print(f"\n{'='*70}")
    print(f"  {wave_name}")
    print(f"{'='*70}")
    generator.print_summary()

    filepath = os.path.join(output_dir, f"{wave_name}.json")
    generator.save_to_json(filepath)

    return generator


def generate_all_benchmarks(output_dir: str = None):
    """
    生成三套标准化Benchmark测试集

    - Wave_1_5_Missiles: A:1, B:2, C:1, D:1 (5枚，轻量测试)
    - Wave_2_10_Missiles: A:2, B:4, C:2, D:2 (10枚，标准测试)
    - Wave_3_20_Missiles: A:5, B:7, C:4, D:4 (20枚，极限测试)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  生成标准化 Benchmark 测试集")
    print("=" * 70)

    # Wave 1: 5枚导弹（轻量测试）
    # B扇区（东南背风超低空）最危险，给2枚
    generate_benchmark_wave(
        seed=42,
        sector_counts={"A": 1, "B": 2, "C": 1, "D": 1},
        wave_name="Wave_1_5_Missiles",
        output_dir=output_dir
    )

    # Wave 2: 10枚导弹（标准测试）
    generate_benchmark_wave(
        seed=123,
        sector_counts={"A": 2, "B": 4, "C": 2, "D": 2},
        wave_name="Wave_2_10_Missiles",
        output_dir=output_dir
    )

    # Wave 3: 20枚导弹（极限测试）
    generate_benchmark_wave(
        seed=456,
        sector_counts={"A": 5, "B": 7, "C": 4, "D": 4},
        wave_name="Wave_3_20_Missiles",
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("  全部 Benchmark 生成完成！")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_benchmarks()
