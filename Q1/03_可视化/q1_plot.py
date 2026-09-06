# -*- coding: utf-8 -*-
"""Q1 绘图：从 04_结果 主表重画论文可用图。运行: python q1_plot.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "04_结果")
FIG = os.path.join(ROOT, "03_可视化")
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300})
C_MAIN, C_CMP, C_PT, C_TH = "#0072B2", "#D55E00", "#999999", "#CC0000"

ev = pd.read_csv(os.path.join(RES, "q1_blood_events.csv"))
curve = pd.read_csv(os.path.join(RES, "q1_effect_curve.csv"))
dg = pd.read_csv(os.path.join(RES, "q1_diagnostics.csv"))
rep = pd.read_csv(os.path.join(RES, "q1_technical_replicates.csv"))
chosen = dg["model"].iloc[0]

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 图1 结果图：孕周-浓度关系与模型曲线（支持：浓度随孕周上升，M0/M1 几乎重合）
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ax.scatter(ev["t_weeks"], ev["Y"], s=6, c=C_PT, alpha=0.35, linewidths=0, label="采血级观测")
ax.fill_between(curve["t_weeks"], curve["M0_lo"], curve["M0_hi"], color=C_MAIN, alpha=0.18, linewidths=0)
ax.plot(curve["t_weeks"], curve["M0_pred"], color=C_MAIN, lw=1.6, label="M0（LMM，采用）边际均值 95%CI")
ax.plot(curve["t_weeks"], curve["M1_pred"], color=C_CMP, lw=1.2, ls="--", label="M1（样条）对照")
ax.axhline(0.04, color=C_TH, lw=1, ls=":")
ax.text(ev["t_weeks"].max() - 0.2, 0.0415, "4% 阈值", color=C_TH, ha="right", fontsize=7)
ax.set(xlabel="检测孕周（周）", ylabel="Y 染色体浓度（比例）", xlim=(10.5, 25.5))
ax.legend(frameon=False, loc="upper left", fontsize=7)
save(fig, "result_q1_week_curve.png")

# 图2 原始数据图：基线 BMI 与浓度（支持：基线 BMI 越高浓度越低）
fig, ax = plt.subplots(figsize=(6.3, 3.2))
first = ev.sort_values("t_weeks").groupby("pid").first().reset_index()
ax.scatter(first["B0"], first["Y"], s=10, c=C_PT, alpha=0.45, linewidths=0, label="首次事件观测")
bins = np.arange(20, 48, 3)
first["bin"] = pd.cut(first["B0"], bins)
bm = first.groupby("bin", observed=True)["Y"].agg(["mean", "sem", "count"])
xc = [i.mid for i in bm.index]
ax.errorbar(xc, bm["mean"], yerr=1.96 * bm["sem"], fmt="o-", color=C_MAIN, ms=4, lw=1.2,
            capsize=2, label="分箱均值 ±95%CI")
ns = ",".join(str(int(n)) for n in bm["count"])
ax.set_title(f"各箱人数（左→右）：{ns}", fontsize=7, loc="right", color="#555555")
ax.set(xlabel="基线 BMI（kg/m²）", ylabel="首次事件 Y 浓度（比例）")
ax.legend(frameon=False, loc="upper right", fontsize=7)
save(fig, "raw_q1_bmi_firsthit.png")

# 图3 过程图：残差诊断（支持：高斯假设可接受，轻度右偏）
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
sps.probplot(dg["resid"], dist="norm", plot=axes[0])
axes[0].get_lines()[0].set(ms=3, color=C_PT, alpha=0.5)
axes[0].get_lines()[1].set(color=C_MAIN, lw=1.2)
axes[0].set_title("残差正态 QQ", fontsize=8)
axes[0].set_xlabel("理论分位数"); axes[0].set_ylabel("样本分位数")
axes[1].scatter(dg["fitted"], dg["resid"], s=6, c=C_PT, alpha=0.35, linewidths=0)
axes[1].axhline(0, color=C_MAIN, lw=1)
lo = lowess(dg["resid"], dg["fitted"], frac=0.3, return_sorted=True)
axes[1].plot(lo[:, 0], lo[:, 1], color=C_CMP, lw=1.4)
axes[1].set(xlabel="拟合值（比例）", ylabel="残差（比例）", title="残差-拟合值")
save(fig, "process_q1_diagnostics.png")

# 图4 原始数据图：技术重复差分布（支持：单次测序误差约0.6个百分点，相对4%不可忽略）
diffs = np.array([float(d) for s in rep["pairwise_diffs"] for d in str(s).split(";")])
fig, ax = plt.subplots(figsize=(6.3, 3.0))
ax.hist(diffs, bins=18, color=C_MAIN, alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="#333333", lw=1)
sd = diffs.std(ddof=1)
ax.set(xlabel="同次采血重复检测浓度差（比例）", ylabel="频数",
       title=f"技术重复差（n={len(diffs)} 对）：SD={sd:.4f}，单次误差估计 ŝ=SD/√2={sd/np.sqrt(2):.4f}")
save(fig, "raw_q1_replicate_diffs.png")
print("ALL DONE")
