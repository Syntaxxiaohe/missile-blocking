"""
UAV Swarm Smoke Defense - Experiment Result Visualization & Statistical Analysis
================================================================================
生成 M1 / M2 / M3 三种方法在 4 种饱和攻击模式下的对比图表，共 6 张图 + 统计检验。

使用方法（在 tools/ 目录下运行）:
    cd <project_root>/tools
    python visualize_results.py

输出目录: tools/figures/
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无头模式，保证 Windows 服务器环境也能运行
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# ============================================================
# 路径配置
# ============================================================
TOOLS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(TOOLS_DIR)
RESULT_DIRS = {
    "M1": os.path.join(ROOT_DIR, "all result", "M1 result"),
    "M2": os.path.join(ROOT_DIR, "all result", "M2 result"),
    "M3": os.path.join(ROOT_DIR, "all result", "M3 result"),
}
OUTPUT_DIR = os.path.join(TOOLS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 全局绘图样式
# ============================================================
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "axes.titlepad"    : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.28,
    "grid.linestyle"   : "--",
    "figure.dpi"       : 140,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "legend.framealpha": 0.92,
    "legend.edgecolor" : "#cccccc",
})

# ============================================================
# 颜色 / 标签常量
# ============================================================
COLORS = {
    "M1": "#9e9e9e",   # 灰色 - NaiveGA 基线
    "M2": "#42a5f5",   # 蓝色 - 聚类+匈牙利
    "M3": "#ef5350",   # 红色 - OT 最优传输（本文提出）
}
HATCHES = {"M1": "", "M2": "///", "M3": ""}

LABELS = {
    "M1": "M1  NaiveGA (Baseline)",
    "M2": "M2  Clustering + Hungarian",
    "M3": "M3  OT-Allocator (Ours)",
}
LABELS_SHORT = {"M1": "M1", "M2": "M2", "M3": "M3"}

MODES = [
    "mode_1_single_sector",
    "mode_2_orthogonal_pincer",
    "mode_3_asymmetric_saturation",
    "mode_4_full_360_swarm",
]
MODE_NAMES = {
    "mode_1_single_sector"        : "Mode 1\nSingle Sector",
    "mode_2_orthogonal_pincer"    : "Mode 2\nOrthogonal Pincer",
    "mode_3_asymmetric_saturation": "Mode 3\nAsymmetric Saturation",
    "mode_4_full_360_swarm"       : "Mode 4\nFull-360° Swarm",
}
MODE_NAMES_SHORT = {
    "mode_1_single_sector"        : "Mode 1",
    "mode_2_orthogonal_pincer"    : "Mode 2",
    "mode_3_asymmetric_saturation": "Mode 3",
    "mode_4_full_360_swarm"       : "Mode 4",
}
N_MISSILES = 20    # 每次实验中的导弹总数

# ============================================================
# 数据加载
# ============================================================
def load_all_results() -> dict:
    """
    从 JSON 文件中读取所有实验结果。
    返回嵌套字典: data[method][mode] = {mean, std, max, min, all, rate_mean, rate_std}
    """
    data: dict = {}
    for method, dir_path in RESULT_DIRS.items():
        data[method] = {}
        for mode in MODES:
            matched = [f for f in os.listdir(dir_path)
                       if mode in f and f.endswith(".json")]
            if not matched:
                print(f"[WARNING] 未找到 {method}/{mode} 的结果文件，跳过。")
                continue
            # 取最新的文件（文件名末尾含时间戳，字典序最大 = 最新）
            filepath = os.path.join(dir_path, sorted(matched)[-1])
            with open(filepath, "r", encoding="utf-8") as fh:
                res = json.load(fh)
            intercepted_list = [r["intercepted"] for r in res["results"]]
            data[method][mode] = {
                "mean"     : res["intercepted_mean"],
                "std"      : res["intercepted_std"],
                "max"      : res["intercepted_max"],
                "min"      : res["intercepted_min"],
                "all"      : intercepted_list,
                "rate_mean": res["intercepted_mean"] / N_MISSILES * 100.0,
                "rate_std" : res["intercepted_std"]  / N_MISSILES * 100.0,
            }
    return data


# ============================================================
# 图 1 - 分组柱状图（带误差棒 ± 1σ）
# ============================================================
def fig1_grouped_bar(data: dict):
    """3 方法 × 4 模式分组柱状图，每柱显示均值 ± 1σ 及具体数值。"""
    fig, ax = plt.subplots(figsize=(13, 6))

    methods  = ["M1", "M2", "M3"]
    x        = np.arange(len(MODES))
    width    = 0.24
    offsets  = [-width, 0, width]

    for method, offset in zip(methods, offsets):
        means = [data[method][m]["rate_mean"] for m in MODES]
        stds  = [data[method][m]["rate_std"]  for m in MODES]
        bars  = ax.bar(
            x + offset, means, width,
            label=LABELS[method],
            color=COLORS[method],
            hatch=HATCHES[method],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        # 误差棒
        ax.errorbar(
            x + offset, means, yerr=stds,
            fmt="none", color="#333333",
            capsize=4, capthick=1.2, linewidth=1.2, zorder=5,
        )
        # 柱顶数值标注
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.8,
                f"{mean:.0f}%",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color=COLORS[method],
            )

    # Mode 4 M2 失效标注箭头
    m2_mode4_y = data["M2"]["mode_4_full_360_swarm"]["rate_mean"]
    m1_mode4_y = data["M1"]["mode_4_full_360_swarm"]["rate_mean"]
    ax.annotate(
        f"M2({m2_mode4_y:.0f}%) < M1({m1_mode4_y:.0f}%)\nclustering failure",
        xy=(3 + offsets[1], m2_mode4_y + 3),
        xytext=(2.3, 55),
        fontsize=8.5, color="#1565c0",
        arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.3),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_NAMES_SHORT[m] for m in MODES], fontsize=11)
    ax.set_ylabel("Interception Rate (%)", fontsize=12)
    ax.set_xlabel("Attack Mode", fontsize=12)
    ax.set_ylim(0, 122)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_title(
        "Figure 1 — Interception Rate: M1 vs M2 vs M3 across Four Attack Modes\n"
        "(Mean ± 1σ, n=10 trials per condition)",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)

    _save(fig, "fig1_grouped_bar.png")


# ============================================================
# 图 2 - 箱线图（含散点 jitter）
# ============================================================
def fig2_boxplot(data: dict):
    """4 列子图，每列对应一种攻击模式，展示 3 种方法的拦截数分布。"""
    np.random.seed(42)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), sharey=True)
    fig.suptitle(
        "Figure 2 — Distribution of Interception Count per Trial (n=10 trials)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    methods   = ["M1", "M2", "M3"]
    positions = [1, 2, 3]

    for ax, mode in zip(axes, MODES):
        box_data = [data[m][mode]["all"] for m in methods]

        bp = ax.boxplot(
            box_data, positions=positions,
            patch_artist=True, widths=0.52,
            medianprops=dict(color="black", linewidth=2.2),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
            flierprops=dict(marker="D", markersize=4.5, alpha=0.5,
                            markeredgewidth=0.5),
        )
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.75)

        # jitter 散点叠加
        for pos, method, ydata in zip(positions, methods, box_data):
            jitter = np.random.uniform(-0.14, 0.14, len(ydata))
            ax.scatter(
                np.full(len(ydata), pos) + jitter, ydata,
                alpha=0.65, s=28, color=COLORS[method],
                edgecolors="white", linewidths=0.4, zorder=6,
            )

        ax.set_title(MODE_NAMES[mode], fontsize=10.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(["M1", "M2", "M3"], fontsize=10.5)
        ax.set_xlim(0.3, 3.7)
        ax.set_ylim(-1.5, 23)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))

    axes[0].set_ylabel("Intercepted Count (out of 20)", fontsize=11)

    patches = [mpatches.Patch(color=COLORS[m], alpha=0.8, label=LABELS[m])
               for m in methods]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), fontsize=10)

    _save(fig, "fig2_boxplot.png")


# ============================================================
# 图 3 - 雷达图（蜘蛛网图）
# ============================================================
def fig3_radar(data: dict):
    """雷达图，展示各方法对 4 种攻击模式的综合防御覆盖能力。"""
    categories = [MODE_NAMES_SHORT[m] for m in MODES]
    N      = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]                   # 闭合多边形

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

    for method in ["M1", "M2", "M3"]:
        values = [data[method][m]["rate_mean"] for m in MODES]
        values += values[:1]
        ax.plot(angles, values,
                color=COLORS[method], linewidth=2.5,
                linestyle="solid", label=LABELS[method], zorder=5)
        ax.fill(angles, values, color=COLORS[method], alpha=0.10)
        # 顶点数值标注
        for angle, val in zip(angles[:-1], values[:-1]):
            ax.annotate(
                f"{val:.0f}%",
                xy=(angle, val),
                xytext=(angle, val + 6),
                ha="center", va="center",
                fontsize=9, color=COLORS[method], fontweight="bold",
            )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11.5)
    ax.set_ylim(0, 115)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                       size=8.5, color="gray")
    ax.set_title(
        "Figure 3 — Overall Defense Coverage Radar\nacross Four Attack Modes",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18),
              fontsize=9.5)

    _save(fig, "fig3_radar.png")


# ============================================================
# 图 4 - 热力图（方法 × 攻击模式）
# ============================================================
def fig4_heatmap(data: dict):
    """热力图矩阵，行=方法，列=攻击模式，单元格=平均拦截率。"""
    methods = ["M1", "M2", "M3"]
    matrix  = np.array([[data[m][mode]["rate_mean"] for mode in MODES]
                         for m in methods])

    fig, ax = plt.subplots(figsize=(9.5, 3.8))

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    # 单元格数值标注
    for i in range(len(methods)):
        for j in range(len(MODES)):
            val = matrix[i, j]
            text_col = "white" if (val < 25 or val > 80) else "black"
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=12.5, fontweight="bold", color=text_col)

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([MODE_NAMES_SHORT[m] for m in MODES], fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABELS[m] for m in methods], fontsize=10.5)
    ax.set_title("Figure 4 — Interception Rate Heatmap: Method × Attack Mode",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Interception Rate (%)", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    # 高亮 Mode4 × M2「聚类失效格」
    rect = plt.Rectangle((2.5, 0.5), 1.0, 1.0,
                          fill=False, edgecolor="#1565c0",
                          linewidth=2.8, linestyle="--", zorder=10)
    ax.add_patch(rect)
    ax.text(3, 1.8, "clustering\nfailure",
            ha="center", va="bottom", color="#1565c0",
            fontsize=8.5, fontweight="bold")

    _save(fig, "fig4_heatmap.png")


# ============================================================
# 图 5 - Mode 4 逐次实验折线图
# ============================================================
def fig5_mode4_trials(data: dict):
    """Mode 4（全向蜂群）10 次独立实验的逐次拦截数，突出 OT 优势。"""
    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(1, 11)
    for method in ["M1", "M2", "M3"]:
        y        = data[method]["mode_4_full_360_swarm"]["all"]
        mean_val = np.mean(y)
        ax.plot(x, y, marker="o", markersize=7.5,
                color=COLORS[method], linewidth=2.2,
                label=f"{LABELS[method]}  (μ={mean_val:.1f})",
                alpha=0.88, zorder=5)
        # 均值虚线
        ax.axhline(mean_val, color=COLORS[method],
                   linestyle="--", linewidth=1.2, alpha=0.55)
        # 右侧均值标注
        ax.text(10.25, mean_val,
                f"μ={mean_val:.1f}",
                color=COLORS[method], va="center",
                fontsize=9.5, fontweight="bold")

    # 标注 M2 失效区间
    ax.axhspan(-1, data["M1"]["mode_4_full_360_swarm"]["mean"] + 2,
               alpha=0.04, color="blue")
    ax.annotate(
        "M2 underperforms M1\n(K-means collapses\nunder 360° attack)",
        xy=(4, data["M2"]["mode_4_full_360_swarm"]["all"][3]),
        xytext=(1.5, 13),
        fontsize=8.5, color="#1565c0",
        arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Trial {i}" for i in x], fontsize=9.5)
    ax.set_ylabel("Intercepted Count (out of 20)", fontsize=12)
    ax.set_title(
        "Figure 5 — Mode 4 (Full-360° Swarm): Per-Trial Interception Results\n"
        "M2 clustering degrades below M1 baseline; M3-OT achieves consistent uplift",
        fontweight="bold",
    )
    ax.set_ylim(-1.5, 23)
    ax.legend(fontsize=10, loc="upper left")

    _save(fig, "fig5_mode4_trials.png")


# ============================================================
# 图 6 - M3 各模式拦截数直方图
# ============================================================
def fig6_m3_histograms(data: dict):
    """M3 四种模式各 10 次实验的拦截数频率分布直方图。"""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    fig.suptitle(
        "Figure 6 — M3 (OT-Allocator): Interception Count Distribution per Mode",
        fontsize=13, fontweight="bold",
    )

    bins = np.arange(-0.5, 21.5, 1)

    for ax, mode in zip(axes, MODES):
        vals     = data["M3"][mode]["all"]
        mean_val = np.mean(vals)
        std_val  = np.std(vals)

        ax.hist(vals, bins=bins,
                color=COLORS["M3"], alpha=0.80,
                edgecolor="white", linewidth=0.7)
        ax.axvline(mean_val, color="black", linestyle="--",
                   linewidth=2.0, label=f"μ={mean_val:.1f}")
        ax.axvspan(mean_val - std_val, mean_val + std_val,
                   alpha=0.10, color="gray", label=f"σ={std_val:.1f}")

        ax.set_title(MODE_NAMES[mode], fontsize=10.5)
        ax.set_xlabel("Intercepted", fontsize=10)
        ax.set_xlim(-0.5, 21)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8.5, loc="upper left")

    axes[0].set_ylabel("Frequency (out of 10 trials)", fontsize=11)

    _save(fig, "fig6_m3_histograms.png")


# ============================================================
# 统计显著性检验（Wilcoxon 配对符号秩检验）
# ============================================================
def run_statistical_tests(data: dict):
    """
    由于每个 eval_idx 使用相同随机种子采样导弹（seed=eval_idx），
    M1/M2/M3 在同一 eval_idx 下面对完全相同的导弹场景 → 数据是**配对**的。
    因此使用 Wilcoxon 配对符号秩检验（比 Mann-Whitney U 更有统计功效）。
    """
    print("\n" + "=" * 72)
    print("  Wilcoxon Signed-Rank Test  (paired, two-sided)")
    print("  H0: no performance difference between two methods")
    print("  Note: paired because same seed => same missile scenario per trial")
    print("=" * 72)

    pairs = [("M1", "M2"), ("M2", "M3"), ("M1", "M3")]

    all_pvals = {}    # 供 bonferroni 校正
    for mode in MODES:
        label = MODE_NAMES_SHORT[mode]
        print(f"\n  [{label}]")
        for m_a, m_b in pairs:
            x = np.array(data[m_a][mode]["all"])
            y = np.array(data[m_b][mode]["all"])
            diff = y - x
            if np.all(diff == 0):
                print(f"    {m_a} vs {m_b}: all differences = 0, skip.")
                continue
            try:
                stat, p = stats.wilcoxon(x, y, alternative="two-sided",
                                         zero_method="wilcox")
            except ValueError as e:
                print(f"    {m_a} vs {m_b}: {e}")
                continue

            sig = ("***" if p < 0.001 else
                   "**"  if p < 0.01  else
                   "*"   if p < 0.05  else "ns")
            direction = "↑" if np.mean(y) > np.mean(x) else "↓"
            print(
                f"    {m_a}({np.mean(x):.1f}) vs {m_b}({np.mean(y):.1f})  "
                f"W={stat:.1f}  p={p:.4f}  {sig}  {direction}"
            )
            all_pvals[f"{mode}_{m_a}_{m_b}"] = p

    print("\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05")
    print("=" * 72)

    # Bonferroni 校正后的显著对数
    alpha_corrected = 0.05 / max(len(all_pvals), 1)
    n_sig = sum(1 for p in all_pvals.values() if p < alpha_corrected)
    print(f"\n  Bonferroni 校正后 α = {alpha_corrected:.4f}，"
          f"共 {n_sig}/{len(all_pvals)} 个比较达到显著。\n")


# ============================================================
# 汇总结果表（控制台输出）
# ============================================================
def print_summary_table(data: dict):
    print("\n" + "=" * 76)
    print("  SUMMARY TABLE — Mean Interception Rate (± Std)")
    print(f"  {'Mode':<28} {'M1':>12} {'M2':>12} {'M3':>12} "
          f"{'Δ(M3-M1)':>10} {'Ratio':>7}")
    print("-" * 76)

    for mode in MODES:
        r1 = data["M1"][mode]["rate_mean"]
        r2 = data["M2"][mode]["rate_mean"]
        r3 = data["M3"][mode]["rate_mean"]
        s1 = data["M1"][mode]["rate_std"]
        s2 = data["M2"][mode]["rate_std"]
        s3 = data["M3"][mode]["rate_std"]
        delta = r3 - r1
        ratio = (r3 / r1) if r1 > 0.1 else float("inf")
        label = MODE_NAMES_SHORT[mode]
        print(
            f"  {label:<28} "
            f"{r1:>6.1f}±{s1:<4.1f}% "
            f"{r2:>6.1f}±{s2:<4.1f}% "
            f"{r3:>6.1f}±{s3:<4.1f}% "
            f"{delta:>+9.1f}pp "
            f"{ratio:>6.2f}x"
        )

    print("-" * 76)
    for method in ["M1", "M2", "M3"]:
        avg = np.mean([data[method][m]["rate_mean"] for m in MODES])
        std = np.std ([data[method][m]["rate_mean"] for m in MODES])
        print(f"  {'Overall avg (' + method + ')':<28} {avg:>6.1f}±{std:<4.1f}%")

    print("=" * 76)


# ============================================================
# 辅助：保存图像
# ============================================================
def _save(fig: plt.Figure, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 72)
    print("  UAV Swarm Smoke Defense — Result Visualization & Analysis")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 72)

    data = load_all_results()

    # 确认数据完整性
    for method in ["M1", "M2", "M3"]:
        loaded = list(data[method].keys())
        print(f"  {method}: loaded {len(loaded)} modes → {[MODE_NAMES_SHORT[m] for m in loaded]}")

    print_summary_table(data)
    run_statistical_tests(data)

    print("\n  [Generating figures ...]")
    fig1_grouped_bar(data)
    fig2_boxplot(data)
    fig3_radar(data)
    fig4_heatmap(data)
    fig5_mode4_trials(data)
    fig6_m3_histograms(data)

    print(f"\n  Done! 6 figures saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
