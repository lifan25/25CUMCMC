# -*- coding: utf-8 -*-
"""Q3 v5 绘图：折中曲线、嵌套区间、偏好敏感性与稳定性。"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "04_结果")
FIG = os.path.join(ROOT, "03_可视化")
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300})
C1, C2, C_PT, C_TH = "#0072B2", "#D55E00", "#999999", "#CC0000"

TGRID = np.arange(11.0, 25.01, 0.5)
P = np.load(os.path.join(RES, "q3_grid_P.npy"))
S = np.load(os.path.join(RES, "q3_grid_R.npy"))
cuts = json.load(open(os.path.join(RES, "q3_cuts.json")))["cuts"]
main = pd.read_csv(os.path.join(RES, "q3_groups_main.csv"))

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 图1 核心：折中情景组内得分曲线与当次达标概率
fig, axes = plt.subplots(1, len(cuts), figsize=(3.3 * len(cuts), 3.0), squeeze=False)
axes = axes.ravel()
fig.subplots_adjust(wspace=0.45)
for gi, (i, j) in enumerate(cuts):
    ax = axes[gi]
    cc = [C1, C2][gi]
    score = S[:, i:j].mean(axis=1)
    pbar = P[:, i:j].mean(axis=1)
    row = main.iloc[gi]
    ax.plot(TGRID, score, color=cc, lw=1.6, label="折中得分 S(t)")
    k = int(np.argmin(score))
    ax.plot([TGRID[k]], [score[k]], "o", color=cc, ms=6)
    ax.axvline(TGRID[k], color=cc, lw=0.7, ls=":", alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(TGRID, pbar, color="#555555", lw=1.1, ls="--", label="当次达标概率 p(t)")
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0.7, 1.0)
    if gi == 1:
        ax2.set_ylabel("当次达标概率 p(t)", fontsize=8)
    else:
        ax2.set_yticks([])
    ax.set(xlabel="检测孕周（周）", ylabel="归一化折中得分 S(t)",
           title=f"{row['BMI区间']}（n={row['人数']}）：t*={row['基础推荐时点t*']:.1f} 周")
axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
save(fig, "result_q3_risk_curves.png")

# 图2 核心：可靠性—延迟相对偏好敏感性
scen = pd.read_csv(os.path.join(RES, "q3_risk_scenarios.csv"))
order_scen = ["仅延迟(0:1)", "偏早(0.5:1)", "折中(1:1)", "偏稳(2:1)", "仅可靠(1:0)"]
xlabels = {"仅延迟(0:1)": "仅延迟", "偏早(0.5:1)": "0.5:1", "折中(1:1)": "1:1\n(主)",
           "偏稳(2:1)": "2:1", "仅可靠(1:0)": "仅可靠"}
fig, ax = plt.subplots(figsize=(6.3, 3.2))
for gi, cc in zip([1, 2], [C1, C2]):
    xs, ys = [], []
    for xi, s in enumerate(order_scen):
        r = scen[(scen["情景"] == s) & (scen["组"] == gi)]
        if len(r):
            xs.append(xi); ys.append(float(r["t*"].iloc[0]))
    ax.plot(xs, ys, "o-", color=cc, lw=1.5, ms=5, label=f"组 {gi}")
ax.set_xticks(range(len(order_scen)))
ax.set_xticklabels([xlabels[s] for s in order_scen], fontsize=7)
ax.set(xlabel="可靠性偏好 : 延迟偏好（wr : wd）", ylabel="最优检测时点 t*（周）",
       title="推荐时点对可靠性—延迟偏好的敏感性")
ax.legend(frameon=False, fontsize=7)
save(fig, "result_q3_weight_sensitivity.png")

# 图3 核心：基础分组之上的个体建议与复检标记
adj = pd.read_csv(os.path.join(RES, "q3_individual_adjust.csv"))
fig, ax = plt.subplots(figsize=(6.3, 3.2))
colors = np.where(adj["复检建议"].astype(str).str.lower().isin(["true", "1"]), C_TH,
                  np.where(adj["调整量Δ"] > 0, "#E69F00", C_PT))
ax.scatter(adj["BMI"], adj["实际建议时点"], s=14, c=colors, alpha=0.75, linewidths=0)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", mfc=C_PT, mec="none", label="基础方案"),
           Line2D([], [], marker="o", ls="", mfc="#E69F00", mec="none", label="组内后移"),
           Line2D([], [], marker="o", ls="", mfc=C_TH, mec="none", label="建议复检")]
for gi, (i, j) in enumerate(cuts):
    row = main.iloc[gi]
    lo = float(row["BMI区间"].split(",")[0].strip("["))
    hi_txt = row["BMI区间"].split(",")[1].strip(") ∞")
    hi = 47.5 if "∞" in row["BMI区间"] else float(hi_txt)
    ax.plot([lo, hi], [row["基础推荐时点t*"], row["基础推荐时点t*"]], color=[C1, C2][gi], lw=1.5)
    ax.text(hi - 0.3, row["基础推荐时点t*"] + 0.25, f"组{gi + 1}基础 {row['基础推荐时点t*']:.1f} 周",
            ha="right", fontsize=7, color=[C1, C2][gi])
n_shift = int((adj["调整量Δ"] > 0).sum())
n_repeat = int(adj["复检建议"].astype(str).str.lower().isin(["true", "1"]).sum())
ax.set(xlabel="基线 BMI（kg/m²）", ylabel="实际建议时点（周）",
       title=f"组内个体建议：{n_shift} 人后移，{n_repeat} 人建议复检")
ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper left")
save(fig, "result_q3_adjustments.png")

# 图4 辅助：bootstrap 稳定性
stab = pd.read_csv(os.path.join(RES, "q3_stability_boot.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1, 1.2]})
bvals = pd.to_numeric(stab.loc[stab["组"] == 1, "bounds"], errors="coerce").dropna()
axes[0].hist(bvals, bins=20, color=C1, alpha=0.8, edgecolor="white", linewidth=0.5)
axes[0].set(xlabel="分界点（kg/m²）", ylabel="频次", title="分界点稳定性（基准情景）")
rng = np.random.default_rng(1)
for gi, cc in zip([1, 2], [C1, C2]):
    v = stab.loc[stab["组"] == gi, "t*"]
    axes[1].scatter(np.full(len(v), gi) + rng.normal(0, 0.06, len(v)), v, s=6,
                    color=cc, alpha=0.35, linewidths=0)
    axes[1].boxplot([v], positions=[gi], widths=0.4, showfliers=False,
                    boxprops=dict(color=cc), medianprops=dict(color="#333333"),
                    whiskerprops=dict(color=cc), capprops=dict(color=cc))
axes[1].set(xlabel="组", ylabel="t*（周）", title="t* 的 bootstrap 分布",
            xticks=[1, 2], xticklabels=["低 BMI 组", "高 BMI 组"])
save(fig, "process_q3_stability.png")

# 图5 核心：BMI—年龄—身高嵌套区间及组内可靠性
nested = pd.read_csv(os.path.join(RES, "q3_nested_groups_main.csv"))
labels = [f"{r.子组}  {r.年龄区间}，{r.身高区间}" for r in nested.itertuples()]
y = np.arange(len(nested))
bar_colors = [C1 if g == 1 else C2 for g in nested["BMI主组"]]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.barh(y, nested["组内平均当次p(t*)"], color=bar_colors, alpha=0.82, height=0.65)
ax.axvline(0.8, color=C_TH, lw=0.9, ls="--", label="p=0.8 复检阈值")
for yi, r in nested.iterrows():
    ax.text(float(r["组内平均当次p(t*)"]) + 0.004, yi,
            f"{r['推荐时点t*']:.1f}周，n={int(r['人数'])}", va="center", fontsize=6.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0.70, 1.01)
ax.set(xlabel="组内平均当次达标概率 p(t*)",
       title="BMI主组内年龄—身高嵌套区间的检测可靠性")
from matplotlib.patches import Patch
nested_handles = [Patch(facecolor=C1, alpha=0.82, label="BMI < 33.5"),
                  Patch(facecolor=C2, alpha=0.82, label="BMI ≥ 33.5"),
                  Line2D([], [], color=C_TH, ls="--", lw=0.9, label="p=0.8 复检阈值")]
ax.legend(handles=nested_handles, frameon=False, fontsize=6.5, loc="lower right")
save(fig, "result_q3_nested_groups.png")
print("ALL DONE")
