# -*- coding: utf-8 -*-
"""Q2 绘图：从 04_结果 主表/曲线重画论文可用图。运行: python q2_plot.py"""
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
C1, C2, C_PT, C_TH, C_CMP = "#0072B2", "#D55E00", "#999999", "#CC0000", "#009E73"

TGRID = np.arange(11.0, 25.01, 0.5)
F = np.load(os.path.join(RES, "q2_grid_F.npy"))
Pg = np.load(os.path.join(RES, "q2_grid_pGEE.npy"))
order = pd.read_csv(os.path.join(RES, "q2_order.csv"))
cuts = json.load(open(os.path.join(RES, "q2_cuts.json")))["cuts"]
main = pd.read_csv(os.path.join(RES, "q2_groups_main.csv"))
B0v = order["B0"].values

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 图1 结果图：两组平均达标概率曲线与推荐时点（核心交付）
fig, ax = plt.subplots(figsize=(6.3, 3.4))
colors = [C1, C2]
for gi, (i, j) in enumerate(cuts):
    pbar = F[:, i:j].mean(axis=1)
    pgee = Pg[:, i:j].mean(axis=1)
    row = main[(main["组"] == gi + 1) & (main["方案"].str.contains("主方案"))].iloc[0]
    ax.plot(TGRID, pbar, color=colors[gi], lw=1.6,
            label=f"{row['BMI区间']}（n={row['人数']}）：F 口径，推荐 {row['推荐孕周']:.1f} 周")
    ax.plot(TGRID, pgee, color=colors[gi], lw=1, ls="--", alpha=0.7,
            label=f"　当次 p 对照（GEE）")
    ax.plot([row["推荐孕周"]], [row["p(推荐时点)"]], "o", color=colors[gi], ms=6)
    ax.axvline(row["推荐孕周"], color=colors[gi], lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color=C_TH, lw=1, ls=":")
ax.text(11.2, 0.952, "可靠度 0.95", color=C_TH, fontsize=7)
ax.axhline(0.04 * 0 + 0.5, lw=0)  # no-op 保持布局
ax.set(xlabel="检测孕周（周）", ylabel="组内平均达标概率", ylim=(0, 1.02))
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
save(fig, "result_q2_group_timing.png")

# 图2 结果图：个体 F 曲线族 + 分界点（支持分组合理性）
fig, ax = plt.subplots(figsize=(6.3, 3.2))
for k in range(0, len(B0v), 4):
    ax.plot(TGRID, F[:, k], color=C_PT, alpha=0.10, lw=0.6)
q = [0.1, 0.5, 0.9]
for qq, cc, lab in zip(q, [C1, "#333333", C2], ["BMI 10% 分位", "BMI 中位", "BMI 90% 分位"]):
    b0 = np.quantile(B0v, qq)
    k = int(np.argmin(np.abs(B0v - b0)))
    ax.plot(TGRID, F[:, k], color=cc, lw=1.5, label=f"{lab}（{B0v[k]:.1f}）")
cut_bmi = (B0v[cuts[0][1] - 1] + B0v[cuts[0][1]]) / 2
ax.set(xlabel="检测孕周（周）", ylabel="首次达标累计概率 F(t|BMI)",
       title=f"个体达标曲线族（灰）与代表 BMI 曲线；优化分界点 ≈ {cut_bmi:.1f} kg/m²")
ax.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "result_q2_survival_curves.png")

# 图3 原始数据图：校准双面板（当次 GEE + 生存 F，目标不同分开呈现）
cal = pd.read_csv(os.path.join(RES, "q2_calibration.csv"))
scal = pd.read_csv(os.path.join(RES, "q2_calibration_survival.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
mk = {"低": "o", "中": "s", "高": "^"}
for b3, g in cal.groupby("BMI组"):
    axes[0].scatter(g["观测当次达标率"], g["预测p(GEE)"], marker=mk[b3], s=30,
                    facecolors="none", edgecolors=C2, label=f"{b3}BMI组")
axes[0].plot([0.5, 1], [0.5, 1], color="#333333", lw=0.8)
axes[0].set(xlabel="观测当次达标率", ylabel="预测 p(GEE)", xlim=(0.5, 1.0), ylim=(0.5, 1.0),
            title="当次达标概率校准（GEE）")
axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
xr = [scal["预测均值"].min() - 0.02, 1.0]
axes[1].scatter(scal["预测均值"], scal["观测比例"], s=30, facecolors="none", edgecolors=C1)
axes[1].plot([0.8, 1], [0.8, 1], color="#333333", lw=0.8)
axes[1].set(xlabel="预测 F(t_last)（十分位箱均值）", ylabel="观测首次达标比例",
            xlim=(0.8, 1.0), ylim=(0.8, 1.05), title="首次跨越概率校准（AFT）")
save(fig, "raw_q2_calibration.png")
# 图4 过程图：bootstrap 稳定性（分界点分布 + t* 分布）
stab = pd.read_csv(os.path.join(RES, "q2_stability_boot.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1, 1.2]})
bvals = stab.loc[stab["组"] == 1, "bounds"].astype(float)
axes[0].hist(bvals, bins=20, color=C1, alpha=0.8, edgecolor="white", linewidth=0.5)
axes[0].axvline(cut_bmi, color=C_TH, lw=1.2)
axes[0].set(xlabel="分界点（kg/m²）", ylabel="频次", title=f"分界点稳定性（主解 {cut_bmi:.1f}）")
for gi, cc in zip([1, 2], colors):
    v = stab.loc[stab["组"] == gi, "t*"]
    axes[1].scatter(np.full(len(v), gi) + np.random.default_rng(1).normal(0, 0.06, len(v)),
                    v, s=6, color=cc, alpha=0.35, linewidths=0)
    axes[1].boxplot([v], positions=[gi], widths=0.4, showfliers=False,
                    boxprops=dict(color=cc), medianprops=dict(color="#333333"),
                    whiskerprops=dict(color=cc), capprops=dict(color=cc))
axes[1].set(xlabel="组", ylabel="推荐孕周 t*（周）", title="t* 的 bootstrap 分布",
            xticks=[1, 2], xticklabels=["低 BMI 组", "高 BMI 组"])
save(fig, "process_q2_stability.png")
print("ALL DONE")
