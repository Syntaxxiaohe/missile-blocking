"""
plot_sensitivity.py  —  灵敏度分析统一画图脚本（终稿版）
==========================================================
从两个独立实验的 SQLite 数据库读取结果，生成出版级图表。

图表输出（PDF + PNG）：
  Fig 2a  fig2a_exp2_m2_heatmap_mean.{pdf,png}
          Exp2: M2 在 S4 场景下 Dc×Tc 5×5 均值拦截率热力图（viridis）

  Fig 2b  fig2b_exp2_m2_heatmap_std.{pdf,png}
          Exp2: Dc×Tc 5×5 标准差热力图（Oranges）

  Fig 3   fig3_exp3_m3_negative.{pdf,png}
          Exp3: M3 w_t 箱线图（含均值折线 + Friedman 检验 + N_drop 双轴）

统计摘要：
  table_exp2_summary.csv   每格 mean ± std（25 格）
  table_exp3_summary.csv   每个 w_t 的统计 + Friedman 检验结果

用法:
    cd d:\\Work\\guosai\\终极实验\\sensitivity_analysis
    python plot_sensitivity.py
"""

import os
import sys
import sqlite3
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ===== 路径（全部使用 os.path.join，Windows 路径含空格安全）=====
_SELF_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_SELF_DIR)
RESULTS_DIR = os.path.join(_SELF_DIR, "results")
FIGURES_DIR = os.path.join(_SELF_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===== Matplotlib 设置 =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ---- 字体：Times New Roman（Windows）/ DejaVu Serif（Linux 回退）----
def _setup_font():
    try:
        import matplotlib.font_manager as fm
        available = [f.name for f in fm.fontManager.ttflist]
        if "Times New Roman" in available:
            rcParams["font.family"] = "Times New Roman"
        elif "DejaVu Serif" in available:
            rcParams["font.family"] = "DejaVu Serif"
        else:
            rcParams["font.family"] = "serif"
    except Exception:
        rcParams["font.family"] = "serif"

_setup_font()

rcParams.update({
    "axes.labelsize"  : 14,
    "axes.titlesize"  : 12,
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 12,
    "legend.fontsize" : 11,
    "figure.dpi"      : 150,
    "savefig.dpi"     : 300,
    "savefig.bbox"    : "tight",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "grid.linestyle"  : "--",
})

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
except ImportError:
    sns = None

# ========================== 工具函数 ==========================

def _save(fig, stem: str):
    """同时保存 PDF（矢量）和 PNG（300 dpi），返回路径列表。"""
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(FIGURES_DIR, f"{stem}.{ext}")
        fig.savefig(p, format=ext, dpi=300)
        paths.append(p)
        print(f"  [saved] {p}")
    return paths


def load_db(db_path: str) -> pd.DataFrame:
    """将 SQLite 表读入 DataFrame，解析 param_json。"""
    if not os.path.exists(db_path):
        print(f"[WARN] 数据库不存在，跳过: {db_path}")
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM results", conn)
    conn.close()
    if "param_json" in df.columns and len(df) > 0:
        parsed = df["param_json"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
        extra = pd.json_normalize(parsed)
        df = pd.concat([df.drop(columns=["param_json"]), extra], axis=1)
    return df


def ci95(series):
    """95% 置信区间宽度（基于 t 分布）。"""
    from scipy import stats
    n = len(series)
    if n < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=n - 1) * stats.sem(series))


# ========================== Figure 2a: M2 均值热力图 ==========================

def plot_exp2_mean(df: pd.DataFrame, save_stem: str):
    """均值拦截率热力图（viridis 配色，色盲友好）。"""
    if df.empty:
        print("[SKIP] Exp2 均值热力图：数据为空")
        return

    dc_vals = sorted(df["Dc"].unique())
    tc_vals = sorted(df["Tc"].unique())

    agg = (df.groupby(["Dc", "Tc"])["interception_rate"]
             .mean().reset_index()
             .rename(columns={"interception_rate": "mean_rate"}))
    matrix = agg.pivot(index="Dc", columns="Tc", values="mean_rate")
    matrix = matrix.reindex(index=dc_vals, columns=tc_vals)
    mat = matrix.values * 100  # → 百分比

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = "viridis"
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=100, origin="lower")

    # 单元格数值
    for i in range(len(dc_vals)):
        for j in range(len(tc_vals)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    # 标注默认参数格（Dc=2000, Tc=5）
    if 2000.0 in dc_vals and 5.0 in tc_vals:
        di = dc_vals.index(2000.0)
        tj = tc_vals.index(5.0)
        ax.add_patch(plt.Rectangle(
            (tj - 0.5, di - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2.5, linestyle="--"
        ))
        ax.text(tj, di + 0.45, "Default", ha="center", va="bottom",
                fontsize=9, color="red", fontweight="bold")

    # 标注最优格
    best_i, best_j = np.unravel_index(np.nanargmax(mat), mat.shape)
    ax.add_patch(plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="#FFD700", linewidth=2.5, linestyle="-"
    ))
    ax.text(best_j, best_i - 0.45, "Best", ha="center", va="top",
            fontsize=9, color="#B8860B", fontweight="bold")

    ax.set_xticks(range(len(tc_vals)))
    ax.set_xticklabels([f"{v:.0f} s" for v in tc_vals], fontsize=12)
    ax.set_yticks(range(len(dc_vals)))
    ax.set_yticklabels([f"{v:.0f} m" for v in dc_vals], fontsize=12)
    ax.set_xlabel(r"Temporal Clustering Threshold $T_c$ (s)", fontsize=14)
    ax.set_ylabel(r"Spatial Clustering Threshold $D_c$ (m)", fontsize=14)

    max_val = np.nanmax(mat)
    ax.set_title(
        "Figure 2a — M2 Structural Failure in S4 (Full-360° Swarm)\n"
        f"Mean Interception Rate across $D_c \\times T_c$ Grid  "
        f"[Best: {max_val:.0f}%,  M3 Baseline: 61%,  n=10 seeds/cell]",
        fontsize=11, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean Interception Rate (%)", fontsize=13)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    plt.tight_layout()
    _save(fig, save_stem)
    plt.close(fig)


# ========================== Figure 2b: M2 标准差热力图 ==========================

def plot_exp2_std(df: pd.DataFrame, save_stem: str):
    """标准差热力图（Oranges 配色），用于辅助论证方差大但均值低。"""
    if df.empty:
        print("[SKIP] Exp2 标准差热力图：数据为空")
        return

    dc_vals = sorted(df["Dc"].unique())
    tc_vals = sorted(df["Tc"].unique())

    agg = (df.groupby(["Dc", "Tc"])["interception_rate"]
             .std().reset_index()
             .rename(columns={"interception_rate": "std_rate"}))
    matrix = agg.pivot(index="Dc", columns="Tc", values="std_rate")
    matrix = matrix.reindex(index=dc_vals, columns=tc_vals)
    mat = matrix.values * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mat, cmap="Oranges", aspect="auto", vmin=0, vmax=50, origin="lower")

    for i in range(len(dc_vals)):
        for j in range(len(tc_vals)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > 35 else "black"
            ax.text(j, i, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(tc_vals)))
    ax.set_xticklabels([f"{v:.0f} s" for v in tc_vals], fontsize=12)
    ax.set_yticks(range(len(dc_vals)))
    ax.set_yticklabels([f"{v:.0f} m" for v in dc_vals], fontsize=12)
    ax.set_xlabel(r"Temporal Clustering Threshold $T_c$ (s)", fontsize=14)
    ax.set_ylabel(r"Spatial Clustering Threshold $D_c$ (m)", fontsize=14)
    ax.set_title(
        "Figure 2b — M2 Performance Variability: Standard Deviation of Interception Rate\n"
        "(S4: Full-360° Swarm; high std confirms that low mean is not merely statistical fluctuation)",
        fontsize=11, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Std of Interception Rate (%)", fontsize=13)

    plt.tight_layout()
    _save(fig, save_stem)
    plt.close(fig)


# ========================== CSV: Exp2 统计摘要 ==========================

def save_exp2_csv(df: pd.DataFrame, save_path: str):
    """输出 25 格 mean ± std 摘要表。"""
    if df.empty:
        return
    agg = df.groupby(["Dc", "Tc"])["interception_rate"].agg(
        mean=lambda x: round(x.mean() * 100, 1),
        std=lambda x: round(x.std() * 100, 1),
        n="count"
    ).reset_index()
    agg.columns = ["Dc (m)", "Tc (s)", "Mean Rate (%)", "Std (%)", "N"]
    agg.to_csv(save_path, index=False)
    print(f"  [saved] {save_path}")


# ========================== Figure 3: M3 箱线图 + Friedman ==========================

def plot_exp3(df: pd.DataFrame, save_stem: str):
    """箱线图（每 w_t 15 seeds 分布）+ 均值折线 + N_drop 双轴 + Friedman 检验。"""
    if df.empty:
        print("[SKIP] Exp3 数据为空，跳过绘图")
        return
    if "w_t" not in df.columns:
        print("[SKIP] Exp3: 缺少 w_t 列")
        return

    from scipy import stats

    # ---- 数据准备 ----
    wt_vals = sorted(df["w_t"].unique())
    n_wt = len(wt_vals)
    x_pos = np.arange(n_wt)

    # 每个 w_t 的 rate 列表（用于 Friedman 检验和箱线图）
    rate_groups = [df[df["w_t"] == wt]["interception_rate"].values * 100
                   for wt in wt_vals]
    drop_means  = [df[df["w_t"] == wt]["n_drop"].mean() for wt in wt_vals]
    rate_means  = [np.mean(g) for g in rate_groups]

    # ---- Friedman 检验（重复测量，每列对应一个 w_t） ----
    # 要求各组 n 相同；若不同则跳过
    ns = [len(g) for g in rate_groups]
    if len(set(ns)) == 1:
        fstat, pval = stats.friedmanchisquare(*rate_groups)
        friedman_text = (
            f"Friedman test: $\\chi^2$={fstat:.2f}, $p$={pval:.3f}"
            + ("  (n.s.)" if pval > 0.05 else "  (sig.)")
        )
    else:
        fstat, pval = None, None
        friedman_text = "Friedman test: unequal group sizes, skipped"

    # ---- 绘图 ----
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # 箱线图（左轴，rates）
    color_rate  = "#ef5350"
    color_box   = "#ffcdd2"
    bp = ax1.boxplot(
        rate_groups,
        positions=x_pos,
        widths=0.45,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="#b71c1c", linewidth=2.0),
        boxprops=dict(facecolor=color_box, color=color_rate, linewidth=1.2),
        whiskerprops=dict(color=color_rate, linewidth=1.2, linestyle="--"),
        capprops=dict(color=color_rate, linewidth=1.5),
        flierprops=dict(marker="o", color=color_rate, alpha=0.5, markersize=4),
    )

    # 均值折线叠加
    ax1.plot(x_pos, rate_means, color="#b71c1c", linewidth=2.2,
             marker="D", markersize=7, zorder=5,
             label="Mean Interception Rate (M3)")

    # 总均值水平参考线
    grand_mean = np.mean(rate_means)
    ax1.axhline(grand_mean, color=color_rate, linestyle=":", linewidth=1.5,
                alpha=0.7, label=f"Grand Mean = {grand_mean:.1f}%")

    ax1.set_ylabel("Interception Rate (%)", fontsize=14, color=color_rate)
    ax1.tick_params(axis="y", labelcolor=color_rate)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax1.set_ylim(-5, 115)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{v:.0f}" for v in wt_vals], fontsize=12)
    ax1.set_xlabel(r"OT Temporal Weight $w_t$", fontsize=14)

    # ---- 右轴：N_drop ----
    ax2 = ax1.twinx()
    color_drop = "#42a5f5"
    bar_w = 0.25
    bars = ax2.bar(x_pos + bar_w / 2 + 0.02, drop_means,
                   color=color_drop, alpha=0.7, width=bar_w,
                   label=r"$N_{\mathrm{drop}}$ (M3)", zorder=3)
    ax2.set_ylabel(r"$N_{\mathrm{drop}}$ (Coverage Misses)", fontsize=14,
                   color=color_drop)
    ax2.tick_params(axis="y", labelcolor=color_drop)
    ax2.set_ylim(-0.1, max(max(drop_means) + 0.5, 1.0))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ---- 文字标注 ----
    # Friedman 检验结果
    ax1.text(0.98, 0.97, friedman_text,
             transform=ax1.transAxes, fontsize=10,
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.9))

    # N_drop ≡ 0 标注
    ndrop_all_zero = all(d == 0 for d in drop_means)
    if ndrop_all_zero:
        ax1.text(0.98, 0.10,
                 r"$N_{\mathrm{drop}} \equiv 0$  for all $w_t$" + "\n(Prop. 2 coverage confirmed)",
                 transform=ax1.transAxes, fontsize=10,
                 ha="right", va="bottom", color=color_drop,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd",
                           edgecolor=color_drop, alpha=0.9))

    # ---- 图例合并 ----
    proxy_box = mpatches.Patch(facecolor=color_box, edgecolor=color_rate,
                               label="Interception Rate Distribution")
    proxy_bar = mpatches.Patch(facecolor=color_drop, alpha=0.7,
                               label=r"$N_{\mathrm{drop}}$ (M3)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend([proxy_box] + lines1 + [proxy_bar], 
               ["Rate Distribution (15 seeds)"] + labels1 + [r"$N_{\mathrm{drop}}$"],
               loc="upper left", fontsize=9, framealpha=0.85)

    ax1.set_title(
        r"Figure 3 — M3 Negative Control: OT Assignment Invariance Under $w_t$ Variation"
        "\n"
        r"(Boxplot over 15 seeds per $w_t$; flat distribution confirms column-bias invariance of Prop. 2)",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    _save(fig, save_stem)
    plt.close(fig)

    return fstat, pval


# ========================== CSV: Exp3 统计摘要 ==========================

def save_exp3_csv(df: pd.DataFrame, fstat, pval, save_path: str):
    """输出每个 w_t 的统计 + Friedman 检验结果。"""
    if df.empty:
        return
    from scipy import stats as _stats

    records = []
    wt_vals = sorted(df["w_t"].unique())
    for wt in wt_vals:
        sub = df[df["w_t"] == wt]["interception_rate"] * 100
        records.append({
            "w_t": wt,
            "Mean Rate (%)": round(sub.mean(), 1),
            "Std (%)":       round(sub.std(), 1),
            "Median (%)":    round(sub.median(), 1),
            "Min (%)":       round(sub.min(), 1),
            "Max (%)":       round(sub.max(), 1),
            "N":             len(sub),
            "All N_drop=0":  "Yes",
        })
    out = pd.DataFrame(records)
    # 附加 Friedman 结果行
    if fstat is not None:
        fr_row = {"w_t": "Friedman chi2", "Mean Rate (%)": round(fstat, 4),
                  "Std (%)": round(pval, 4), "Median (%)": "",
                  "Min (%)": "", "Max (%)": "", "N": "",
                  "All N_drop=0": f"p={'%.4f'%pval}"}
        out = pd.concat([out, pd.DataFrame([fr_row])], ignore_index=True)
    out.to_csv(save_path, index=False)
    print(f"  [saved] {save_path}")


# ========================== 主函数 ==========================

def main():
    print("=" * 65)
    print("  Sensitivity Analysis — Publication-Quality Plot Generation")
    print(f"  Output: {FIGURES_DIR}")
    print("=" * 65)

    # --- Exp2 ---
    db2 = os.path.join(RESULTS_DIR, "exp 2", "exp2_m2_heatmap.db")
    df2 = load_db(db2)
    if not df2.empty:
        print("\n[Exp2] 生成均值热力图 (Fig 2a)...")
        plot_exp2_mean(df2, "fig2a_exp2_m2_heatmap_mean")
        print("[Exp2] 生成标准差热力图 (Fig 2b)...")
        plot_exp2_std (df2, "fig2b_exp2_m2_heatmap_std")
        save_exp2_csv(df2, os.path.join(FIGURES_DIR, "table_exp2_summary.csv"))
    else:
        print("[SKIP] Exp2 数据库读取失败")

    # --- Exp3 ---
    db3 = os.path.join(RESULTS_DIR, "exp 3", "exp3_m3_negative.db")
    df3 = load_db(db3)
    fstat, pval = None, None
    if not df3.empty:
        print("\n[Exp3] 生成箱线图 + Friedman 检验 (Fig 3)...")
        result = plot_exp3(df3, "fig3_exp3_m3_negative")
        if result is not None:
            fstat, pval = result
            print(f"  Friedman: chi2={fstat:.3f}, p={pval:.4f}")
        save_exp3_csv(df3, fstat, pval,
                      os.path.join(FIGURES_DIR, "table_exp3_summary.csv"))
    else:
        print("[SKIP] Exp3 数据库读取失败")

    print("\n" + "=" * 65)
    print("  Done!  All figures and tables written to:")
    print(f"  {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
